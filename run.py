"""Cross-platform launcher: works on Linux and Windows.

    python run.py                # start the bot in the foreground
    python run.py --forcerun     # stop any existing instance first

Thin on purpose. Most of run.bat re-implemented things that already exist:
single-instance enforcement lives in src/app.py's acquire_runtime_lock()
(portable already -- O_CREAT|O_EXCL plus a liveness check on the recorded pid),
and the restart policy and log rotation belong to the supervisor (systemd on
Linux, Task Scheduler on Windows). Foreground is the right shape for both.

run.bat stays for existing Task Scheduler entries; the two are interchangeable.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from services import platform_support  # noqa: E402  (needs the path insert above)

APP = ROOT / "src" / "app.py"
OWNERSHIP_FILES = (ROOT / ".bot.lock", ROOT / ".bot.pid")

# Shared with run.bat, which sets LAST_START_FILE to the same path and restarts
# the bot when this timestamp is over 24 hours old. Both launchers must write
# it or the one that does not causes restarts the other cannot explain.
LAST_START_FILE = ROOT / ".last_start.txt"

# Bound on _forcerun's stop loop. Each pass sends SIGTERM and escalates to
# SIGKILL, so a process that survives several rounds is not going to die.
MAX_STOP_ATTEMPTS = 5


def _log(message: str) -> None:
    print(f"[run] {message}", flush=True)


def _interpreter() -> str:
    """PYTHON_EXE wins (run.bat honours it too, and the semantic extras are
    often installed into a specific interpreter). Otherwise reuse the current
    one, which is correct inside a venv."""
    return (os.environ.get("PYTHON_EXE") or "").strip() or sys.executable


def _read_pid(path: Path) -> int | None:
    """Both ownership files are written by src/app.py: .bot.lock holds
    {"pid": N} (with a bare-int legacy form) and .bot.pid a bare int."""
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        return int(json.loads(raw)["pid"]) if raw.startswith("{") else int(raw)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _is_our_bot(pid: int) -> bool:
    """Corroborate that `pid` really is this bot before acting on it.

    A pid alone is not enough: Linux recycles pids from a 32768-default space
    within hours, so a stale lock file can name a pid that now belongs to
    something unrelated -- and --forcerun would then kill *that*. If the command
    line cannot be read at all (permissions, or a Windows query failure), fall
    back to trusting the lock file rather than refusing to ever recover.
    """
    cmdline = platform_support.process_cmdline(pid)
    if cmdline is None:
        return True
    return "app.py" in cmdline


def _running_pid() -> int | None:
    for path in OWNERSHIP_FILES:
        pid = _read_pid(path)
        if pid is None or pid <= 0:
            continue
        if not platform_support.is_process_alive(pid):
            continue
        if not _is_our_bot(pid):
            _log(f"ignoring stale {path.name}: pid={pid} is not this bot (pid reuse)")
            continue
        return pid
    return None


def _forcerun() -> bool:
    """Stop any running instance, then clear its ownership files.

    Bounded, not `while pid is not None`: a POSIX zombie stays in the process
    table until its parent reaps it, so kill(pid, 0) keeps succeeding while
    SIGTERM/SIGKILL are no-ops on it. An unbounded loop would therefore spin
    forever here -- a hang that cannot happen on Windows, where there is no
    equivalent state.
    """
    for _ in range(MAX_STOP_ATTEMPTS):
        pid = _running_pid()
        if pid is None:
            break
        _log(f"stopping existing instance pid={pid}")
        ok, detail = platform_support.terminate_process(pid)
        if not ok:
            _log(f"failed to stop pid={pid}: {detail}")
            return False
    else:
        pid = _running_pid()
        if pid is not None:
            _log(
                f"pid={pid} still present after {MAX_STOP_ATTEMPTS} stop attempts "
                "(unreapable zombie, or a pid we cannot signal). Not starting."
            )
            return False

    for path in OWNERSHIP_FILES:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            _log(f"could not remove {path.name}: {exc}")
            return False
    return True


def _record_start() -> None:
    """Stamp `.last_start.txt`, the marker run.bat's restart guard reads.

    Two supervisors can start this bot and only one of them was maintaining the
    marker. run.bat writes it; run.py did not -- and `.reset` starts the bot
    through `run.py --forcerun`. So every reset left the timestamp ageing while
    the process underneath it was brand new.

    run.bat's scheduled task re-runs on a short interval and restarts the bot
    whenever that timestamp is over 24 hours old. A bot restarted only ever by
    `.reset` therefore crossed the threshold while running, and the next tick of
    the task killed it and started its own -- a restart nobody asked for,
    arriving at an arbitrary time, with nothing in the logs tying it to the
    reset hours earlier that actually caused it.

    Written here rather than in run.bat's guard because the marker means "when
    did the bot last start", and this is a place the bot starts. Any third
    launcher must call this too.

    Best-effort: a bot that cannot write a marker file should still boot. The
    cost of failing is a spurious restart later, not a failure now.
    """
    try:
        LAST_START_FILE.write_text(
            _dt.datetime.now().astimezone().isoformat(), encoding="ascii"
        )
    except OSError as exc:
        _log(f"could not record start time ({exc}); run.bat may restart this "
             "instance once its 24h guard expires")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Discord bot.")
    parser.add_argument("--forcerun", action="store_true",
                        help="stop any running instance before starting")
    parser.add_argument("--skip-ats-sync", action="store_true",
                        help="skip the ATS company list refresh")
    args = parser.parse_args()

    if not APP.exists():
        _log(f"ERROR: {APP} not found")
        return 1

    if args.forcerun:
        if not _forcerun():
            _log("forcerun failed: an existing instance survived. Not starting.")
            return 1
    elif (pid := _running_pid()) is not None:
        _log(f"already running (pid={pid}). Use --forcerun to replace it.")
        return 0

    python_exe = _interpreter()

    if not args.skip_ats_sync and (sync := ROOT / "sync_ats_companies.py").exists():
        _log("syncing ATS company lists...")
        rc = subprocess.run([python_exe, str(sync)], cwd=ROOT).returncode
        if rc != 0:
            # Non-fatal: the bot runs with whatever lists are already on disk.
            _log(f"ATS sync exited {rc}; continuing with existing lists")

    _record_start()
    _log(f"starting: {python_exe} {APP}")
    return _run_bot_forwarding_signals(python_exe)


def _run_bot_forwarding_signals(python_exe: str) -> int:
    """Launch the bot and relay shutdown signals to it.

    This wrapper is the unit's MainPID, so it must not exit while the bot is
    still draining -- systemd would see MainPID gone and SIGKILL the rest of the
    cgroup mid-shutdown. Forward the signal, then wait for the child to finish
    its own drain.
    """
    proc = subprocess.Popen([python_exe, str(APP)], cwd=ROOT)

    def _forward(signum, _frame):
        _log(f"received signal {signum}; forwarding to bot pid={proc.pid}")
        try:
            proc.send_signal(signum)
        except (ProcessLookupError, OSError):
            pass

    installed = []
    for name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            installed.append((sig, signal.signal(sig, _forward)))
        except (ValueError, OSError):
            pass

    try:
        while True:
            try:
                return proc.wait()
            except KeyboardInterrupt:
                # Our own handler already forwarded it; keep waiting for the
                # child rather than abandoning it mid-drain.
                continue
    finally:
        for sig, previous in installed:
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError):
                pass


if __name__ == "__main__":
    raise SystemExit(main())
