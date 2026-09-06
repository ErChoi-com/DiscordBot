"""The guard against a platform that yields nothing has to run on its own.

A scraper returning [] is indistinguishable from a platform whose companies
have nothing open, so this failure is silent by construction. It was found by
reading the archive by source: bamboohr and paylocity contributed nothing
across a full week against 13,755 and 9,325 confirmed-live companies -- one
switched off by a config flag, the other harvested and probed with no scraper
behind it at all.

scripts/check_platform_yield.py exists for that, and like validate_ats_slugs
before it, nothing ever ran it. The per-cycle silence counter in `.health` is
not a substitute: it lives in the process and resets on every restart, while
this reads a week of committed archive and survives one.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from watchers.manager import WatcherManager  # noqa: E402


class _Config:
    def __init__(self, base_dir):
        self.base_dir = base_dir


def _manager(tmp_path):
    mgr = WatcherManager.__new__(WatcherManager)
    mgr.config = _Config(tmp_path)
    return mgr


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


@pytest.fixture
def run(monkeypatch, tmp_path):
    seen = {"argv": None, "label": None, "rc": 0, "stdout": "", "boom": False}

    async def _to_thread(self, fn, *args, label=None, **kwargs):
        seen["label"] = label
        return fn(*args, **kwargs)

    def _subprocess_run(argv, **_kwargs):
        seen["argv"] = list(argv)
        if seen["boom"]:
            raise OSError("no interpreter")
        return type("R", (), {"returncode": seen["rc"],
                              "stdout": seen["stdout"], "stderr": ""})()

    monkeypatch.setattr(WatcherManager, "_tracked_to_thread", _to_thread)
    monkeypatch.setattr("subprocess.run", _subprocess_run)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "check_platform_yield.py").write_text("", encoding="utf-8")
    return seen


# -- it runs -----------------------------------------------------------------

def test_the_check_is_actually_invoked(run, tmp_path):
    asyncio.run(_manager(tmp_path)._check_platform_yield())
    assert run["argv"] is not None
    assert run["argv"][1].endswith("check_platform_yield.py")


def test_it_reads_a_week(run, tmp_path):
    """One day of archive cannot tell a quiet platform from a broken one --
    the finding that motivated the script was a full week of silence."""
    asyncio.run(_manager(tmp_path)._check_platform_yield())
    argv = run["argv"]
    assert "--days" in argv and argv[argv.index("--days") + 1] == "7"


def test_it_runs_under_its_own_cost_label(run, tmp_path):
    asyncio.run(_manager(tmp_path)._check_platform_yield())
    assert run["label"] == "ats_yield_check"


def test_it_does_not_read_the_archive_on_the_event_loop():
    compact = " ".join(
        inspect.getsource(WatcherManager._check_platform_yield).split()
    )
    assert "await self._tracked_to_thread( _check" in compact


# -- what it reports ---------------------------------------------------------

def test_a_silent_platform_is_reported(run, tmp_path, capsys):
    run["rc"] = 1
    run["stdout"] = ("platform     live  jobs\nbamboohr    13755     0\n\n"
                     "Silent: bamboohr, paylocity\nEach has companies confirmed live")
    asyncio.run(_manager(tmp_path)._check_platform_yield())
    assert "Silent: bamboohr, paylocity" in capsys.readouterr().out


def test_the_table_is_not_dumped_into_the_bot_log(run, tmp_path, capsys):
    """Eighteen rows of columns every day buries the one line that matters."""
    run["rc"] = 1
    run["stdout"] = ("platform     live  jobs\nbamboohr    13755     0\n"
                     "lever        7686  1204\n\nSilent: bamboohr")
    asyncio.run(_manager(tmp_path)._check_platform_yield())
    out = capsys.readouterr().out
    assert "Silent: bamboohr" in out
    assert "13755" not in out


def test_thin_fields_are_reported_too(run, tmp_path, capsys):
    """A platform delivering jobs with no location is dropped from every
    location-scoped search -- quieter than silence and just as costly."""
    run["stdout"] = "Thin location: icims (3%)\n  -> dropped from location searches"
    asyncio.run(_manager(tmp_path)._check_platform_yield())
    assert "Thin location: icims (3%)" in capsys.readouterr().out


def test_an_unvalidated_backlog_is_reported(run, tmp_path, capsys):
    run["stdout"] = "Unvalidated: 10,667 companies never resolved either way"
    asyncio.run(_manager(tmp_path)._check_platform_yield())
    assert "Unvalidated: 10,667" in capsys.readouterr().out


def test_a_clean_week_says_so_rather_than_saying_nothing(run, tmp_path, capsys):
    """Silence from the checker is indistinguishable from the checker not
    having run, which is the failure this whole change is about."""
    run["rc"] = 0
    run["stdout"] = "platform     live  jobs\nlever        7686  1204"
    asyncio.run(_manager(tmp_path)._check_platform_yield())
    assert "every platform with coverage contributed jobs" in capsys.readouterr().out


# -- it never costs the scrape -----------------------------------------------

def test_a_silent_platform_does_not_stop_the_cycle(run, tmp_path):
    """Exit 1 is this script saying "someone should look", not "abandon the
    scrape that is about to run"."""
    run["rc"] = 1
    run["stdout"] = "Silent: bamboohr"
    asyncio.run(_manager(tmp_path)._check_platform_yield())  # must not raise


def test_a_crash_does_not_stop_the_cycle(run, tmp_path, capsys):
    run["boom"] = True
    asyncio.run(_manager(tmp_path)._check_platform_yield())
    assert "could not check platform yield" in capsys.readouterr().out


def test_a_missing_script_is_not_an_error(run, tmp_path, capsys):
    (tmp_path / "scripts" / "check_platform_yield.py").unlink()
    asyncio.run(_manager(tmp_path)._check_platform_yield())
    assert run["argv"] is None
    assert capsys.readouterr().out == ""


# -- it recurs, and in the right order ---------------------------------------

def test_the_daily_branch_runs_it():
    compact = " ".join(
        inspect.getsource(WatcherManager._run_ats_scrape_loop).split()
    )
    assert "await self._check_platform_yield()" in compact


def test_it_runs_after_the_three_that_write():
    """It reads the confirmed-live store the validation pass just wrote and the
    fleet the sync just pulled; running it first reports yesterday."""
    compact = " ".join(
        inspect.getsource(WatcherManager._run_ats_scrape_loop).split()
    )
    for earlier in ("_refresh_company_lists()", "_validate_company_slugs()",
                    "_refresh_geo_index()"):
        assert compact.index(earlier) < compact.index("_check_platform_yield()"), earlier


def test_it_is_not_wired_into_every_cycle():
    lines = inspect.getsource(WatcherManager._run_ats_scrape_loop).splitlines()
    guard = next(i for i, l in enumerate(lines) if "if rolled_over:" in l)
    call = next(i for i, l in enumerate(lines) if "_check_platform_yield()" in l)

    assert call > guard
    body = [l for l in lines[guard + 1:call + 1] if l.strip()]
    assert all(_indent(l) > _indent(lines[guard]) for l in body)


def test_it_is_the_one_daily_hook_that_writes_nothing():
    """The other three change stores this one reads, which is why it runs last.
    A writer here would make that ordering meaningless.
    """
    compact = " ".join(
        inspect.getsource(WatcherManager._check_platform_yield).split()
    )
    assert "--dry-run" not in compact
    assert "save" not in compact and "write" not in compact
