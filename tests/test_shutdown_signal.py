"""SIGTERM handling.

Everything the bot does on shutdown -- draining scrapes and watcher sends,
stopping Chrome cleanly, releasing .bot.lock/.bot.pid -- hangs off `finally:`
blocks and a patched `client.close`. Python's DEFAULT SIGTERM disposition
terminates the process outright, so none of that runs under a supervisor.

That made the whole drain dead code under systemd: every `systemctl stop` and
every RuntimeMaxSec restart stranded the lock files and killed Chrome with its
cookie DB unflushed. These tests pin the handler that fixes it.
"""
from __future__ import annotations

import dis
import os
import signal

import pytest

import app as app_module
import run as run_module


@pytest.fixture(autouse=True)
def _restore_sigterm():
    """Never leak a handler into the rest of the suite.

    Defensive on teardown: a test may have monkeypatched the signal module, and
    an exception here would be reported as an error against every such test.
    """
    previous = signal.getsignal(signal.SIGTERM) if hasattr(signal, "SIGTERM") else None
    yield
    if previous is not None:
        try:
            signal.signal(signal.SIGTERM, previous)
        except (AttributeError, ValueError, OSError, TypeError):
            pass


def test_handler_is_installed_for_sigterm():
    assert app_module.install_shutdown_signal_handler() is True
    assert signal.getsignal(signal.SIGTERM) not in (
        signal.SIG_DFL, signal.SIG_IGN, None
    ), "SIGTERM still has its default disposition, which kills without draining"


def test_installed_handler_raises_keyboardinterrupt(capsys):
    """The mechanism: KeyboardInterrupt is what discord.py's Client.run() already
    catches, and its handling awaits client.close() -- the hook _graceful_close
    is installed on. Anything else would not reach the drain."""
    app_module.install_shutdown_signal_handler()
    handler = signal.getsignal(signal.SIGTERM)

    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGTERM, None)

    assert "SIGTERM" in capsys.readouterr().out


def test_run_bot_installs_the_handler():
    """A handler that is never installed is worth nothing. Checked structurally
    against the compiled function rather than by reading the source text."""
    referenced = {
        instruction.argval for instruction in dis.get_instructions(app_module.run_bot)
    }
    assert "install_shutdown_signal_handler" in referenced, (
        "run_bot no longer installs the SIGTERM handler; the shutdown drain is "
        "dead code again under systemd"
    )


def test_handler_install_reports_failure_rather_than_raising():
    """Python refuses signal.signal() outside the main thread. Exercised with a
    REAL thread rather than a patched signal module, so it pins the actual
    interpreter behaviour: this must degrade to False, not crash startup."""
    import threading

    result: list[object] = []

    def _install_off_main_thread():
        try:
            result.append(app_module.install_shutdown_signal_handler())
        except Exception as exc:            # must not escape
            result.append(exc)

    thread = threading.Thread(target=_install_off_main_thread)
    thread.start()
    thread.join(timeout=5)

    assert result == [False], f"expected a clean False off the main thread, got {result}"


def test_handler_is_skipped_when_platform_lacks_sigterm(monkeypatch):
    monkeypatch.delattr(signal, "SIGTERM", raising=False)
    assert app_module.install_shutdown_signal_handler() is False


# ---------------------------------------------------------------------------
# The launcher must not abandon the bot mid-drain
# ---------------------------------------------------------------------------

class _FakeProc:
    """Stands in for the bot process: records signals, and only exits once it
    has been signalled -- so a wrapper that returns early is detectable."""

    def __init__(self) -> None:
        self.pid = 4242
        self.signals: list[int] = []
        self._waits = 0

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)

    def wait(self) -> int:
        self._waits += 1
        return 0


def test_launcher_forwards_sigterm_to_the_bot(monkeypatch):
    """run.py is the unit's MainPID. If it exits without passing the signal on,
    systemd sees MainPID gone and SIGKILLs the cgroup mid-shutdown."""
    fake = _FakeProc()
    monkeypatch.setattr(run_module.subprocess, "Popen", lambda *a, **k: fake)

    captured: dict[int, object] = {}
    monkeypatch.setattr(
        run_module.signal, "signal",
        lambda sig, handler: captured.setdefault(sig, handler) or signal.SIG_DFL,
    )

    assert run_module._run_bot_forwarding_signals("python") == 0

    assert signal.SIGTERM in captured, "no SIGTERM handler installed by the launcher"
    captured[signal.SIGTERM](signal.SIGTERM, None)
    assert fake.signals == [signal.SIGTERM], "signal was not forwarded to the bot"


def test_launcher_returns_the_bot_exit_code(monkeypatch):
    class _Failing(_FakeProc):
        def wait(self) -> int:
            return 3

    monkeypatch.setattr(run_module.subprocess, "Popen", lambda *a, **k: _Failing())
    assert run_module._run_bot_forwarding_signals("python") == 3


def test_launcher_keeps_waiting_through_keyboardinterrupt(monkeypatch):
    """Ctrl-C is forwarded by our handler; the wrapper must resume waiting
    rather than abandoning a bot that is still draining."""
    class _InterruptsOnce(_FakeProc):
        def wait(self) -> int:
            self._waits += 1
            if self._waits == 1:
                raise KeyboardInterrupt
            return 0

    proc = _InterruptsOnce()
    monkeypatch.setattr(run_module.subprocess, "Popen", lambda *a, **k: proc)

    assert run_module._run_bot_forwarding_signals("python") == 0
    assert proc._waits == 2, "wrapper gave up instead of waiting for the drain"


def test_launcher_tolerates_a_bot_that_already_exited(monkeypatch):
    class _Gone(_FakeProc):
        def send_signal(self, signum: int) -> None:
            raise ProcessLookupError

    proc = _Gone()
    monkeypatch.setattr(run_module.subprocess, "Popen", lambda *a, **k: proc)

    captured: dict[int, object] = {}
    monkeypatch.setattr(
        run_module.signal, "signal",
        lambda sig, handler: captured.setdefault(sig, handler) or signal.SIG_DFL,
    )

    assert run_module._run_bot_forwarding_signals("python") == 0
    captured[signal.SIGTERM](signal.SIGTERM, None)   # must not raise


# ---------------------------------------------------------------------------
# Runtime lock: pid reuse
# ---------------------------------------------------------------------------

def test_lock_is_reclaimed_when_the_owner_pid_was_recycled(tmp_path, monkeypatch):
    """A stranded .bot.lock naming a recycled pid must not wedge the bot. That
    path exits 0, so the supervisor restarts into the same refusal forever --
    silently, and `systemctl status` still reports healthy."""
    lock = tmp_path / ".bot.lock"
    lock.write_text('{"pid": 4242}', encoding="utf-8")

    monkeypatch.setattr(app_module, "is_process_running", lambda pid: True)
    monkeypatch.setattr(app_module.platform_support, "process_cmdline",
                        lambda pid, system=None: "/usr/sbin/nginx -g daemon off;")

    assert app_module.acquire_runtime_lock(lock) is True, (
        "refused the lock to a recycled pid that is not this bot"
    )
    assert app_module.read_runtime_lock_owner(lock) == os.getpid()


def test_lock_is_refused_while_the_real_bot_holds_it(tmp_path, monkeypatch):
    """The negative case -- otherwise the check above would pass on a lock that
    never refuses anything, defeating single-instance entirely."""
    lock = tmp_path / ".bot.lock"
    lock.write_text('{"pid": 4242}', encoding="utf-8")

    monkeypatch.setattr(app_module, "is_process_running", lambda pid: True)
    monkeypatch.setattr(app_module.platform_support, "process_cmdline",
                        lambda pid, system=None: "/usr/bin/python3 /opt/discordbot/src/app.py")

    assert app_module.acquire_runtime_lock(lock) is False
    assert app_module.read_runtime_lock_owner(lock) == 4242, "stole a live owner's lock"


def test_unreadable_cmdline_falls_back_to_trusting_liveness(tmp_path, monkeypatch):
    """Windows returns a blank CommandLine for many processes. An unreadable
    cmdline must not be treated as 'not the bot'."""
    lock = tmp_path / ".bot.lock"
    lock.write_text('{"pid": 4242}', encoding="utf-8")

    monkeypatch.setattr(app_module, "is_process_running", lambda pid: True)
    monkeypatch.setattr(app_module.platform_support, "process_cmdline",
                        lambda pid, system=None: None)

    assert app_module.acquire_runtime_lock(lock) is False


def test_dead_owner_lock_is_reclaimed(tmp_path, monkeypatch):
    lock = tmp_path / ".bot.lock"
    lock.write_text('{"pid": 4242}', encoding="utf-8")
    monkeypatch.setattr(app_module, "is_process_running", lambda pid: False)

    assert app_module.acquire_runtime_lock(lock) is True
