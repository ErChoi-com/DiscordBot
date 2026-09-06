"""Two launchers, one marker, and only one of them was writing it.

`run.bat` runs on a scheduled-task interval and decides each tick whether to
restart the bot: if the bot is running and `.last_start.txt` is under 24 hours
old it skips, otherwise it kills the process and starts its own. `run.py` is the
other launcher -- it is what `.reset` spawns, via `run.py --forcerun` -- and it
never wrote that file.

So a bot restarted by `.reset` left the timestamp ageing under a brand-new
process. Once it crossed 24 hours the next tick of the task killed the bot and
started a replacement: a restart nobody asked for, at an arbitrary time, with
nothing in the logs connecting it to the reset hours earlier that caused it.
Observed on 2026-09-06 -- a `run.py --forcerun` at 23:00 was killed by run.bat
at 00:09 and replaced.

The tests that matter here are the cross-file ones. This is not a bug in either
launcher read on its own; it is the two disagreeing, so the pins are on the
contract between them: same path, same format, written on every start.
"""
from __future__ import annotations

import datetime
import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _load_run():
    """Import run.py by path -- it is a top-level script, not a package module."""
    spec = importlib.util.spec_from_file_location("_run_launcher", REPO / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run(tmp_path, monkeypatch):
    module = _load_run()
    monkeypatch.setattr(module, "LAST_START_FILE", tmp_path / ".last_start.txt")
    return module


# -- the marker gets written --------------------------------------------------

def test_recording_a_start_writes_the_marker(run):
    run._record_start()
    assert run.LAST_START_FILE.exists()


def test_the_marker_holds_the_current_time(run):
    before = datetime.datetime.now().astimezone()
    run._record_start()
    written = datetime.datetime.fromisoformat(
        run.LAST_START_FILE.read_text(encoding="ascii").strip())
    after = datetime.datetime.now().astimezone()

    assert before <= written <= after


def test_a_later_start_replaces_the_earlier_stamp(run):
    """It answers "when did the bot last start", not "when did it first".
    Appending would leave run.bat parsing a stale first line forever."""
    run.LAST_START_FILE.write_text("2020-01-01T00:00:00.000000-05:00", encoding="ascii")
    run._record_start()
    text = run.LAST_START_FILE.read_text(encoding="ascii")

    assert "2020" not in text
    assert len(text.strip().splitlines()) == 1


# -- in a format the other launcher can actually read -------------------------

def test_the_stamp_is_offset_aware(run):
    """run.bat parses with [DateTimeOffset]::Parse and subtracts from local
    time. A naive stamp is read in whatever zone the parser assumes, which
    silently shifts the age by hours -- in the wrong direction it manufactures
    the very restart this fixes.
    """
    run._record_start()
    parsed = datetime.datetime.fromisoformat(
        run.LAST_START_FILE.read_text(encoding="ascii").strip())
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None


def test_the_stamp_matches_the_shape_run_bat_writes(run):
    """run.bat writes `Get-Date -Format o`: ISO 8601, fractional seconds, and a
    numeric offset. Both launchers write the same file, so both must write the
    same shape or whichever wrote last decides whether the other can read it.
    """
    run._record_start()
    text = run.LAST_START_FILE.read_text(encoding="ascii").strip()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{2}:\d{2}", text), text


def test_the_stamp_is_ascii_readable(run):
    """run.bat writes it `-Encoding ascii` and reads it back with Get-Content;
    anything wider risks a BOM or a mojibake first line."""
    run._record_start()
    raw = run.LAST_START_FILE.read_bytes()
    assert raw.decode("ascii")
    assert not raw.startswith(b"\xef\xbb\xbf")


def test_a_freshly_written_marker_reads_as_under_the_guard(run):
    """The end of the mechanism: run.bat skips its restart when the age is
    under 24 hours. A stamp this file just wrote must land well inside that."""
    run._record_start()
    parsed = datetime.datetime.fromisoformat(
        run.LAST_START_FILE.read_text(encoding="ascii").strip())
    age_hours = (datetime.datetime.now().astimezone() - parsed).total_seconds() / 3600

    assert 0 <= age_hours < 24


# -- the contract between the two launchers -----------------------------------

def _run_bat_text() -> str:
    return (REPO / "run.bat").read_text(encoding="utf-8", errors="replace")


def test_both_launchers_name_the_same_file():
    """The whole defect in one assertion. If these ever diverge, each launcher
    maintains a marker the other never reads and the unexplained restarts come
    straight back.
    """
    module = _load_run()
    declared = re.search(r'set "LAST_START_FILE=%~dp0([^"]+)"', _run_bat_text())

    assert declared, "run.bat no longer declares LAST_START_FILE"
    assert module.LAST_START_FILE.name == declared.group(1)


def test_the_marker_sits_beside_the_launchers():
    """`%~dp0` is the batch file's own directory, so run.py's copy has to be
    repo-root-relative too, not cwd-relative."""
    module = _load_run()
    assert module.LAST_START_FILE.parent == module.ROOT


def test_run_bat_still_restarts_on_a_stale_marker():
    """The behaviour that makes writing the marker matter. If run.bat stopped
    restarting on staleness this fix would be pointless, and the test should
    say so rather than pass quietly.
    """
    text = _run_bat_text()
    assert "24h+ since last start" in text
    assert "TotalHours -lt 24" in text


def test_run_bat_is_the_other_writer():
    """Pinned so a future change that stops run.bat writing does not leave
    run.py as the silent sole owner of a file two things depend on."""
    assert "Set-Content -Path $env:LAST_START_FILE" in _run_bat_text()


# -- and it happens on every start --------------------------------------------

def test_main_records_the_start_before_launching_the_bot():
    """Source-level pin: main() is not callable in a test without launching a
    Discord client. Recording after the launch would never run -- the launcher
    blocks for the life of the bot.
    """
    import inspect

    module = _load_run()
    src = inspect.getsource(module.main)
    assert "_record_start()" in src
    assert src.index("_record_start()") < src.index("_run_bot_forwarding_signals")


def test_the_forcerun_path_reaches_the_same_recording():
    """`.reset` uses --forcerun, and that is the path that was leaving the
    marker stale. It must not skip the stamp."""
    import inspect

    module = _load_run()
    src = inspect.getsource(module.main)
    forcerun = src.index("args.forcerun")
    record = src.index("_record_start()")
    assert record > forcerun, "the stamp is written before the forcerun branch"
    # ...and is not nested inside a branch that --forcerun can miss.
    line = next(l for l in src.splitlines() if "_record_start()" in l)
    assert len(line) - len(line.lstrip()) == 4, line


# -- failure to record must not stop the bot ----------------------------------

def test_an_unwritable_marker_does_not_stop_the_launch(run, tmp_path, capsys):
    """Cost of failing to write: one spurious restart, a day later. Cost of
    refusing to boot: no bot. Never trade the second for the first.
    """
    run.LAST_START_FILE = tmp_path / "missing" / "dir" / ".last_start.txt"
    run._record_start()  # must not raise
    assert "could not record start time" in capsys.readouterr().out


def test_the_failure_says_what_it_will_cost(run, tmp_path, capsys):
    """A warning that does not name the consequence gets ignored, and this one
    predicts a restart hours later that is otherwise unexplainable."""
    run.LAST_START_FILE = tmp_path / "missing" / "dir" / ".last_start.txt"
    run._record_start()
    assert "restart" in capsys.readouterr().out.lower()
