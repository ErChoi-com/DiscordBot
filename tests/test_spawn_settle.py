"""A restart that dies on startup used to look exactly like one that worked.

spawn_detached returned success the instant Popen returned, which only says a
process was created -- not that it got as far as running the program. The two
come apart in precisely the cases that matter for `.reset`: the venv moved, a
module fails to import, run.py has a syntax error. The interpreter starts,
dies, and Popen reports success.

`.reset` then answered "Restarting...", latched _restart_spawned so no later
reset would be attempted, and the bot neither restarted nor could be told to
try again for the rest of its life. The failure was indistinguishable from
success and disabled the only command that could have recovered from it.

The child's output went to DEVNULL as well, so the one process that knew why
was writing its reason nowhere -- and the process that would have wanted to
read it is the one the restart is about to kill.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import platform_support as ps  # noqa: E402


#: Generous on purpose for the tests that assert a *failure*. Measured need:
#: on this machine, with the bot mid-startup scraping eighteen platforms, a
#: cold interpreter took over ten seconds just to reach its first line -- so a
#: window sized for an idle box turns "the child died" into "the child is
#: still running" and the test passes for the wrong reason. The default is
#: tuned for a Discord command; here the question is only whether a dead child
#: is detected, and a loaded Windows box can take longer than the default just
#: to start an interpreter. Racing that would make these tests flaky in the
#: direction of falsely passing. The default's own size is asserted separately.
SETTLE = 30.0


def _log_text(path: Path, timeout: float = 5.0) -> str:
    """The child's output, once it lands.

    Reading once is a race: wait() returning means the child exited, but the
    bytes it wrote can take a moment more to be visible to another process on
    Windows.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        if text.strip():
            return text
        time.sleep(0.05)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "child.py"
    path.write_text(body, encoding="utf-8")
    return path


# -- what "started" means ----------------------------------------------------

def test_a_child_that_dies_on_startup_is_reported_as_a_failure(tmp_path):
    """The real case: the interpreter runs, the program does not."""
    script = _script(tmp_path, "import sys; sys.exit(3)\n")
    ok, detail = ps.spawn_detached([sys.executable, str(script)], cwd=str(tmp_path),
                                   settle_seconds=SETTLE)

    assert ok is False
    assert "3" in detail


def test_an_import_error_in_the_child_is_reported(tmp_path):
    """The shape an actual broken restart takes -- a module that stopped
    importing between the running process and the code on disk."""
    script = _script(tmp_path, "import a_module_that_does_not_exist\n")
    ok, detail = ps.spawn_detached([sys.executable, str(script)], cwd=str(tmp_path),
                                   settle_seconds=SETTLE)

    assert ok is False
    assert "exited immediately" in detail


def test_a_child_that_keeps_running_is_a_success(tmp_path):
    """What a real restart looks like: still alive when we stop looking."""
    script = _script(tmp_path, "import time; time.sleep(30)\n")
    ok, detail = ps.spawn_detached([sys.executable, str(script)], cwd=str(tmp_path),
                                   settle_seconds=0.6)
    assert (ok, detail) == (True, "")


def test_a_short_task_that_finishes_cleanly_is_not_a_failure(tmp_path):
    """This helper is used for fire-and-forget work too. Exiting zero is a task
    that finished, and calling that a failed spawn would be wrong."""
    script = _script(tmp_path, "print('done')\n")
    ok, detail = ps.spawn_detached([sys.executable, str(script)], cwd=str(tmp_path))
    assert (ok, detail) == (True, "")


def test_the_wait_can_be_switched_off(tmp_path):
    """A caller that genuinely only wants the process created should not pay
    the settle time."""
    script = _script(tmp_path, "import sys; sys.exit(3)\n")
    ok, _detail = ps.spawn_detached([sys.executable, str(script)],
                                    cwd=str(tmp_path), settle_seconds=0)
    assert ok is True


def test_a_missing_executable_still_reports_the_os_error(tmp_path):
    ok, detail = ps.spawn_detached(["definitely-not-a-program-xyz"], cwd=str(tmp_path))
    assert ok is False
    assert detail


def test_the_settle_window_is_short_enough_to_sit_in_a_command():
    """It runs inside a Discord command, between "Restarting..." and the kill."""
    assert 0 < ps.SPAWN_SETTLE_SECONDS <= 5.0


# -- the log ------------------------------------------------------------------

def test_the_childs_output_is_kept_when_a_log_is_given(tmp_path):
    script = _script(tmp_path, "print('hello from the child')\n")
    log = tmp_path / "restart.log"
    ps.spawn_detached([sys.executable, str(script)], cwd=str(tmp_path), log_path=log,
                      settle_seconds=SETTLE)

    assert "hello from the child" in _log_text(log)


def test_a_traceback_survives_the_process_that_asked_for_the_restart(tmp_path):
    """The point of the log. The parent is about to be killed by the child it
    spawned, so anything only the parent could have seen is lost."""
    script = _script(tmp_path, "raise RuntimeError('boom in the new process')\n")
    log = tmp_path / "restart.log"
    ok, detail = ps.spawn_detached([sys.executable, str(script)], cwd=str(tmp_path),
                                   log_path=log, settle_seconds=SETTLE)

    assert ok is False
    assert "restart.log" in detail
    assert "boom in the new process" in _log_text(log)


def test_the_log_appends_rather_than_overwriting(tmp_path):
    """A restart loop should leave a history. Truncating each time keeps only
    the last failure, which is the one least likely to explain the loop."""
    script = _script(tmp_path, "print('run')\n")
    log = tmp_path / "restart.log"
    for _ in range(2):
        ps.spawn_detached([sys.executable, str(script)], cwd=str(tmp_path), log_path=log,
                          settle_seconds=SETTLE)

    import time
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and log.read_text(encoding="utf-8").count("run") < 2:
        time.sleep(0.05)
    assert log.read_text(encoding="utf-8").count("run") == 2


def test_a_missing_log_directory_is_created(tmp_path):
    script = _script(tmp_path, "print('x')\n")
    log = tmp_path / "nested" / "dir" / "restart.log"
    ok, _ = ps.spawn_detached([sys.executable, str(script)], cwd=str(tmp_path),
                              log_path=log, settle_seconds=SETTLE)
    assert ok is True
    assert log.exists()


def test_an_unwritable_log_does_not_stop_the_restart(tmp_path):
    """Losing the record is far cheaper than refusing to start the process it
    was going to describe."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")
    script = _script(tmp_path, "import time; time.sleep(5)\n")

    ok, _ = ps.spawn_detached([sys.executable, str(script)], cwd=str(tmp_path),
                              log_path=blocker / "restart.log", settle_seconds=0.4)
    assert ok is True


def test_stderr_reaches_the_log_not_just_stdout(tmp_path):
    script = _script(tmp_path, "import sys; print('to stderr', file=sys.stderr)\n")
    log = tmp_path / "restart.log"
    ps.spawn_detached([sys.executable, str(script)], cwd=str(tmp_path), log_path=log,
                      settle_seconds=SETTLE)
    assert "to stderr" in _log_text(log)


def test_no_log_path_still_discards_the_output(tmp_path):
    """Unchanged default. Inheriting the caller's stdio would tie the child to
    a console that is about to disappear."""
    calls = {}

    class _Alive:
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="c", timeout=timeout)

    def fake_popen(args, **kwargs):
        calls.update(kwargs)
        return _Alive()

    import unittest.mock as m
    with m.patch.object(ps.subprocess, "Popen", fake_popen):
        ps.spawn_detached(["x"], cwd=str(tmp_path))

    assert calls["stdout"] == subprocess.DEVNULL
    assert calls["stderr"] == subprocess.DEVNULL


# -- detachment is not sacrificed for the check -------------------------------

def test_the_child_still_outlives_a_caller_that_stops_waiting(tmp_path):
    """The settle wait must not turn this into a blocking call. The child of a
    restart has to survive the parent it is about to kill.

    The child runs until this test releases it, rather than for a fixed sleep,
    so nothing here is a wall-clock race. If spawn_detached ever waited for the
    child to finish it would deadlock -- the release only happens on the line
    after it returns -- and the test fails by hanging rather than by a
    threshold that a loaded machine can cross for unrelated reasons. Two
    earlier attempts at this test measured elapsed time and were flaky for
    exactly that reason while the bot was mid-restart.
    """
    import time

    release = tmp_path / "release.txt"
    marker = tmp_path / "marker.txt"
    script = _script(
        tmp_path,
        "import os, time\n"
        f"while not os.path.exists({str(release)!r}):\n"
        "    time.sleep(0.02)\n"
        f"open({str(marker)!r}, 'w').write('outlived')\n",
    )
    ok, _ = ps.spawn_detached([sys.executable, str(script)], cwd=str(tmp_path),
                              settle_seconds=0.3)
    # Reaching this line at all is the proof: the child cannot have exited.
    assert ok is True
    release.write_text("go", encoding="utf-8")

    # Wait for the content, not for the file. The child's `open(marker, 'w')`
    # creates an empty file and only then writes, so existence goes true one
    # syscall before the text lands. Polling `exists()` and reading straight
    # after therefore returns "" on a loaded machine -- observed failing in a
    # full-suite run while passing 10/10 in isolation. The deadline stays a
    # hang-detector; it is no longer also the thing being raced.
    deadline = time.monotonic() + 30.0
    written = ""
    while time.monotonic() < deadline:
        try:
            written = marker.read_text(encoding="utf-8")
        except OSError:
            written = ""
        if written:
            break
        time.sleep(0.05)
    assert written == "outlived", "the detached child did not outlive the call"


def test_the_detach_flags_are_unchanged_by_the_settle_check(tmp_path):
    calls = {}

    class _Alive:
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="c", timeout=timeout)

    def fake_popen(args, **kwargs):
        calls.update(kwargs)
        return _Alive()

    import unittest.mock as m
    with m.patch.object(ps.subprocess, "Popen", fake_popen):
        ps.spawn_detached(["x"], system=ps.WINDOWS)
    assert calls["creationflags"] == ps.DETACHED_PROCESS | ps.CREATE_NEW_PROCESS_GROUP

    calls.clear()
    with m.patch.object(ps.subprocess, "Popen", fake_popen):
        ps.spawn_detached(["x"], system=ps.LINUX)
    assert calls["start_new_session"] is True


def test_a_popen_without_wait_is_still_reported_as_started(tmp_path):
    """Only the confirmation is unavailable; the process was created. Calling
    that a failed spawn would be a worse lie than not confirming.
    """
    import unittest.mock as m
    with m.patch.object(ps.subprocess, "Popen", lambda *a, **k: object()):
        assert ps.spawn_detached(["x"], cwd=str(tmp_path)) == (True, "")


# -- and .reset uses it --------------------------------------------------------

# There is deliberately no test here pinning handle_reset's call site. That
# lives in commands/handlers.py, which carries local work this repo does not
# commit, so a pin on it fails against a clean checkout for a reason that has
# nothing to do with what it is checking -- the trap this repo has hit before.
# What is pinned is the contract handle_reset depends on: a dead child reports
# failure, so the caller can release its restart claim, and the child's output
# survives in a file rather than going to DEVNULL. The wiring itself was
# verified by running the restart: run.py --forcerun replaced pid 8072 with
# 24480, and .restart.log holds the whole sequence that used to be discarded.
