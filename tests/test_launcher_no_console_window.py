"""The watchdog task fires every minute, and its launch must never draw a
window.

Two regressions are guarded here, both already observed in this repo:

1. run.bat used to background the bot with `start "Discord Bot" cmd /c ...`.
   `start` asks for a console, and on Windows 11 that console is hosted by
   Windows Terminal, which draws it regardless of any "minimized" flag. The
   fix routes the normal path through `run_hidden.vbs` (a GUI-subsystem
   binary, wscript.exe, which allocates no console at all) and demotes the
   `start` line to a documented fallback used only if wscript is missing.

2. While writing that fix, the nested quotes and `>>` redirection in the
   command line were shell-escaped incorrectly at least once, corrupting the
   backslashes in `src\\app.py` / `logs\\bot_console.log` and leaving stray
   control bytes in the file. A test that only checks "the file still
   contains wscript" would not catch that; this file parses the actual
   launch lines.

Everything below is derived from what is on disk right now, not from what
the fix was supposed to do -- if a future edit reintroduces the bare `start`
as the primary path, or reverts the show-state argument to non-zero, these
should fail.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_bat_text() -> str:
    return (REPO_ROOT / "run.bat").read_text(encoding="utf-8", errors="replace")


def _run_bat_bytes() -> bytes:
    return (REPO_ROOT / "run.bat").read_bytes()


def _vbs_text() -> str:
    return (REPO_ROOT / "run_hidden.vbs").read_text(encoding="utf-8", errors="replace")


def _ps1_text() -> str:
    return (REPO_ROOT / "scripts" / "fix_watchdog_window.ps1").read_text(
        encoding="utf-8", errors="replace"
    )


# -- run.bat: the primary path no longer asks for a console -------------------


def test_every_launch_hidden_call_sets_the_launch_cmd_first():
    """`call :launch_hidden` reads %BOT_LAUNCH_CMD%, so a call site that forgot
    to set it first would launch a stale or empty command. Every call must be
    preceded, within a few lines, by the variable being set.
    """
    text = _run_bat_text()
    lines = text.splitlines()
    call_indices = [i for i, line in enumerate(lines) if "call :launch_hidden" in line]

    assert call_indices, "run.bat no longer calls :launch_hidden anywhere"

    for idx in call_indices:
        window = lines[max(0, idx - 5): idx]
        assert any('set "BOT_LAUNCH_CMD=' in line for line in window), (
            f"call :launch_hidden at line {idx + 1} is not preceded by a "
            "BOT_LAUNCH_CMD assignment"
        )


def test_the_only_bare_start_is_the_documented_fallback_inside_launch_hidden():
    """The old default path was `start "Discord Bot" cmd /c ...` run
    unconditionally. The only occurrence of `start "Discord Bot"` left must
    live inside :launch_hidden, after the wscript attempt -- i.e. it is a
    fallback, not the default.
    """
    text = _run_bat_text()
    lines = text.splitlines()

    start_indices = [i for i, line in enumerate(lines) if 'start "Discord Bot"' in line]
    assert len(start_indices) == 1, (
        f"expected exactly one 'start \"Discord Bot\"' occurrence (the "
        f"documented fallback), found {len(start_indices)}"
    )

    subroutine_start = next(i for i, line in enumerate(lines) if line.strip() == ":launch_hidden")
    wscript_line = next(i for i, line in enumerate(lines) if i > subroutine_start and "wscript" in line.lower())

    (start_idx,) = start_indices
    assert subroutine_start < wscript_line < start_idx, (
        "the fallback 'start' line must appear inside :launch_hidden, after "
        "the wscript attempt"
    )


def test_launch_hidden_subroutine_invokes_run_hidden_vbs_via_wscript():
    text = _run_bat_text()
    lines = text.splitlines()
    subroutine_start = next(i for i, line in enumerate(lines) if line.strip() == ":launch_hidden")
    # a subroutine body runs to the next label or EOF
    remaining = lines[subroutine_start:]
    next_label = next(
        (i for i, line in enumerate(remaining[1:], start=1) if re.match(r"^:\w+", line.strip())),
        len(remaining),
    )
    body = "\n".join(remaining[:next_label])

    assert "wscript" in body.lower()
    assert "run_hidden.vbs" in body
    assert "--nowait-env BOT_LAUNCH_CMD" in body


def test_launch_hidden_returns_before_the_fallback_on_success():
    """A successful wscript launch must short-circuit with `exit /b 0` before
    execution can fall through to the visible-window fallback.
    """
    text = _run_bat_text()
    lines = text.splitlines()
    subroutine_start = next(i for i, line in enumerate(lines) if line.strip() == ":launch_hidden")
    wscript_idx = next(i for i, line in enumerate(lines) if i > subroutine_start and "wscript" in line.lower())
    fallback_idx = next(i for i, line in enumerate(lines) if i > subroutine_start and 'start "Discord Bot"' in line)

    between = lines[wscript_idx:fallback_idx]
    assert any(re.search(r"exit /b 0", line) for line in between), (
        "no 'exit /b 0' guards the fallback from running after a successful "
        "hidden launch"
    )
    # and that exit is conditioned on the wscript call, not unconditional
    guard_line = next(line for line in between if "exit /b 0" in line)
    assert "errorlevel" in guard_line.lower()


# -- run.bat: no corruption from shell-escaping --------------------------------


def test_run_bat_has_no_bel_or_backspace_bytes():
    raw = _run_bat_bytes()
    assert b"\x07" not in raw, "run.bat contains a BEL byte (0x07) -- shell-escaping corruption"
    assert b"\x08" not in raw, "run.bat contains a backspace byte (0x08) -- shell-escaping corruption"


def test_run_bat_launch_lines_still_have_literal_windows_paths():
    """This is exactly the corruption that happened while writing the fix:
    the nested quoting swallowed backslashes in the path literals."""
    text = _run_bat_text()
    assert r"src\app.py" in text
    assert r"logs\bot_console.log" in text
    # and specifically on the BOT_LAUNCH_CMD lines, not just anywhere in the file
    launch_cmd_lines = [l for l in text.splitlines() if 'set "BOT_LAUNCH_CMD=' in l]
    assert launch_cmd_lines, "no BOT_LAUNCH_CMD assignment lines found"
    for line in launch_cmd_lines:
        assert r"src\app.py" in line, line
        assert r"logs\bot_console.log" in line, line


# -- run_hidden.vbs: both documented modes -------------------------------------


def test_vbs_handles_nowait_env_argument():
    text = _vbs_text()
    assert '"--nowait-env"' in text or "\"--nowait-env\"".lower() in text.lower()
    assert "--nowait-env" in text


def test_vbs_reads_named_variable_from_process_environment():
    text = _vbs_text()
    assert 'shell.Environment("PROCESS")' in text


def test_vbs_default_mode_builds_command_from_run_bat():
    text = _vbs_text()
    assert 'here & "\\run.bat"' in text or 'here & "\\\\run.bat"' in text


def test_vbs_both_run_calls_pass_zero_show_state():
    """The bug this whole fix targets: a non-zero show state draws a window
    regardless of which subsystem started the process. Both Run() call sites
    (the --nowait-env branch and the default run.bat branch) must pass the
    literal 0.
    """
    text = _vbs_text()
    run_calls = re.findall(r"\.?Run\s*\(?\s*cmd\s*,\s*([^,\)]+)", text)
    assert len(run_calls) == 2, f"expected exactly two Run(cmd, ...) call sites, found {run_calls}"
    for show_state in run_calls:
        assert show_state.strip() == "0", f"Run() called with show state {show_state!r}, not 0"


# -- fix_watchdog_window.ps1: points the task at wscript, not cmd -------------


def _new_scheduled_task_action_block(text: str) -> str:
    """Isolate the New-ScheduledTaskAction call, not the doc-comment above it
    that narrates the *old*, cmd.exe-based action for context."""
    start = text.index("$action = New-ScheduledTaskAction")
    end = text.index("\n\n", start)
    return text[start:end]


def test_ps1_points_scheduled_task_action_at_wscript_not_cmd():
    block = _new_scheduled_task_action_block(_ps1_text())
    assert re.search(r"-Execute\s+'wscript\.exe'", block)
    assert not re.search(r"-Execute\s+'cmd\.exe'", block)


def test_ps1_passes_scheduler_argument():
    block = _new_scheduled_task_action_block(_ps1_text())
    argument_match = re.search(r"-Argument\s+\"(.*)\"", block)
    assert argument_match, f"no -Argument found in action block: {block!r}"
    assert re.search(r"\bscheduler\b", argument_match.group(1)), (
        "the scheduled task action must pass the 'scheduler' argument so "
        "run.bat takes its foreground branch"
    )


# -- end-to-end: actually run run_hidden.vbs -----------------------------------


def _cscript_available() -> bool:
    return sys.platform == "win32" and shutil.which("cscript") is not None


@pytest.mark.skipif(sys.platform != "win32", reason="run_hidden.vbs only runs on Windows")
@pytest.mark.skipif(not _cscript_available(), reason="cscript.exe not available")
def test_nowait_env_actually_runs_the_command_hidden(tmp_path):
    sentinel = tmp_path / "sentinel.txt"
    # keep the child command simple and self-contained: no nested quoting to
    # get wrong here, unlike the real BOT_LAUNCH_CMD.
    marker_cmd = f'cmd /c echo hello> "{sentinel}"'

    env = dict(os.environ)
    env["TEST_LAUNCH_CMD"] = marker_cmd

    proc = subprocess.run(
        [
            "cscript", "//Nologo", "//E:vbscript",
            str(REPO_ROOT / "run_hidden.vbs"),
            "--nowait-env", "TEST_LAUNCH_CMD",
        ],
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")

    deadline = time.time() + 10
    while time.time() < deadline and not sentinel.exists():
        time.sleep(0.2)

    assert sentinel.exists(), "run_hidden.vbs did not run the command in time"
    assert "hello" in sentinel.read_text(encoding="utf-8", errors="replace").lower()


@pytest.mark.skipif(sys.platform != "win32", reason="run_hidden.vbs only runs on Windows")
@pytest.mark.skipif(not _cscript_available(), reason="cscript.exe not available")
def test_nowait_env_with_empty_variable_exits_3_and_runs_nothing(tmp_path):
    sentinel = tmp_path / "should_not_exist.txt"

    env = dict(os.environ)
    env.pop("TEST_EMPTY_LAUNCH_CMD", None)  # ensure genuinely unset

    proc = subprocess.run(
        [
            "cscript", "//Nologo", "//E:vbscript",
            str(REPO_ROOT / "run_hidden.vbs"),
            "--nowait-env", "TEST_EMPTY_LAUNCH_CMD",
        ],
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        timeout=15,
    )

    assert proc.returncode == 3, (
        f"expected exit code 3 for an unset/empty variable, got "
        f"{proc.returncode}: {proc.stderr.decode(errors='replace')}"
    )
    # give any wrongly-launched process a moment, then confirm nothing landed
    time.sleep(0.5)
    assert not sentinel.exists()
