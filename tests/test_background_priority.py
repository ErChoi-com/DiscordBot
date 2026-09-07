"""Background work yields the CPU to interactive work, at the OS level.

The scheduler reserves *workers* for interactive commands, but a worker is not
a core. Measured on the live bot: 12.4 of 12 cores busy, 541 threads, the
Discord gateway logging "can't keep up ... websocket is 3.5s behind" while the
reserved workers sat ready -- ready, and then competing on equal terms with
every scrape thread for a core the moment a command arrived. Priority is the
layer the reservation cannot reach, so background tasks and scrape
subprocesses now run below normal.

These are real checks against the OS where the OS lets us read the value
back, and both-branch checks of the argument shaping everywhere.
"""
from __future__ import annotations

import asyncio
import platform
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import platform_support as ps  # noqa: E402
from services.priority_scheduler import BACKGROUND, INTERACTIVE, PriorityWorkScheduler  # noqa: E402

ON_WINDOWS = platform.system() == "Windows"
windows_only = pytest.mark.skipif(not ON_WINDOWS, reason="reads the priority back through kernel32")


# ── the thread primitive ─────────────────────────────────────────────────────

@windows_only
def test_a_thread_can_be_lowered_and_raised_back():
    seen: dict[str, object] = {}

    def _body():
        seen["before"] = ps.current_thread_priority()
        seen["lowered"] = ps.set_current_thread_background(True)
        seen["during"] = ps.current_thread_priority()
        seen["raised"] = ps.set_current_thread_background(False)
        seen["after"] = ps.current_thread_priority()

    t = threading.Thread(target=_body)
    t.start()
    t.join(5)
    assert seen == {
        "before": ps.THREAD_PRIORITY_NORMAL,
        "lowered": True,
        "during": ps.THREAD_PRIORITY_BELOW_NORMAL,
        "raised": True,
        "after": ps.THREAD_PRIORITY_NORMAL,
    }


@windows_only
def test_lowering_one_thread_leaves_the_others_alone():
    """The whole point: the event loop's thread must keep its priority while
    a worker gives up its own."""
    worker_done = threading.Event()
    release = threading.Event()

    def _worker():
        ps.set_current_thread_background(True)
        worker_done.set()
        release.wait(5)

    t = threading.Thread(target=_worker)
    t.start()
    worker_done.wait(5)
    try:
        assert ps.current_thread_priority() == ps.THREAD_PRIORITY_NORMAL
    finally:
        release.set()
        t.join(5)


def test_the_helpers_never_raise_on_a_platform_that_refuses():
    """Priority is an optimisation; failing to set it must never cost a task."""
    assert ps.set_current_thread_background(True, system="Plan9") in (True, False)
    assert ps.current_thread_priority(system="Plan9") is None or isinstance(
        ps.current_thread_priority(system="Plan9"), int
    )


# ── subprocess shaping, both branches ────────────────────────────────────────

def test_windows_subprocesses_get_the_below_normal_creation_flag():
    cmd, extra = ps.low_priority_popen_args(["python", "-c", "1"], system=ps.WINDOWS)
    assert cmd == ["python", "-c", "1"]
    assert extra["creationflags"] & ps.BELOW_NORMAL_PRIORITY_CLASS


def test_posix_subprocesses_are_prefixed_with_nice_not_preexec_fn(monkeypatch):
    """preexec_fn is documented unsafe with threads and this process has
    hundreds; the nice binary costs one exec and is safe."""
    monkeypatch.setattr(ps.shutil, "which", lambda name: "/usr/bin/nice" if name == "nice" else None)
    cmd, extra = ps.low_priority_popen_args(["python", "-c", "1"], system=ps.LINUX)
    assert cmd == ["/usr/bin/nice", "-n", str(ps.BACKGROUND_NICE), "python", "-c", "1"]
    assert "preexec_fn" not in extra
    assert "creationflags" not in extra


def test_posix_without_a_nice_binary_still_runs_the_command(monkeypatch):
    monkeypatch.setattr(ps.shutil, "which", lambda name: None)
    cmd, extra = ps.low_priority_popen_args(["python", "-c", "1"], system=ps.LINUX)
    assert cmd == ["python", "-c", "1"]
    assert extra == {}


def test_the_input_command_is_not_mutated():
    original = ["python", "-c", "1"]
    ps.low_priority_popen_args(original, system=ps.LINUX)
    ps.low_priority_popen_args(original, system=ps.WINDOWS)
    assert original == ["python", "-c", "1"]


@windows_only
def test_a_real_subprocess_starts_below_normal():
    """Read the priority class back from inside the child."""
    probe = (
        "import ctypes; from ctypes import wintypes; k=ctypes.windll.kernel32; "
        "k.GetCurrentProcess.restype=wintypes.HANDLE; "
        "k.GetPriorityClass.argtypes=[wintypes.HANDLE]; k.GetPriorityClass.restype=wintypes.DWORD; "
        "print(k.GetPriorityClass(k.GetCurrentProcess()))"
    )
    cmd, extra = ps.low_priority_popen_args([sys.executable, "-c", probe])
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False, **extra).stdout
    assert int(out.strip()) == ps.BELOW_NORMAL_PRIORITY_CLASS


# ── the scheduler applies it per task, not per worker ────────────────────────

def _run(scheduler: PriorityWorkScheduler, fn, tier: int):
    async def _go():
        return await scheduler.run(fn, tier=tier)

    return asyncio.run(_go())


@windows_only
def test_background_tasks_run_below_normal_and_interactive_tasks_do_not():
    scheduler = PriorityWorkScheduler(max_workers=2, reserved_interactive=1)
    try:
        assert _run(scheduler, ps.current_thread_priority, BACKGROUND) == ps.THREAD_PRIORITY_BELOW_NORMAL
        assert _run(scheduler, ps.current_thread_priority, INTERACTIVE) == ps.THREAD_PRIORITY_NORMAL
    finally:
        scheduler.shutdown(wait=False) if hasattr(scheduler, "shutdown") else None


@windows_only
def test_a_worker_is_back_at_normal_before_its_next_task():
    """The general worker that just ran background work must not carry the
    lowered priority into an interactive task that spills onto it."""
    scheduler = PriorityWorkScheduler(max_workers=1, reserved_interactive=0)
    try:
        assert _run(scheduler, ps.current_thread_priority, BACKGROUND) == ps.THREAD_PRIORITY_BELOW_NORMAL
        # Same single worker, next task: interactive, so it must read normal.
        assert _run(scheduler, ps.current_thread_priority, INTERACTIVE) == ps.THREAD_PRIORITY_NORMAL
        # And a task that raises still restores the worker.
        def _boom():
            raise RuntimeError("x")
        with pytest.raises(RuntimeError):
            _run(scheduler, _boom, BACKGROUND)
        assert _run(scheduler, ps.current_thread_priority, INTERACTIVE) == ps.THREAD_PRIORITY_NORMAL
    finally:
        scheduler.shutdown(wait=False) if hasattr(scheduler, "shutdown") else None


def test_the_scrape_subprocess_uses_the_low_priority_shaping(monkeypatch):
    """The call site, not just the helper: a scrape subprocess must go
    through low_priority_popen_args or the whole thing is decorative."""
    from services import job_service

    seen: dict = {}

    def _fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs

        class _P:
            returncode = 0
            stdout = "[]"
            stderr = ""

        return _P()

    monkeypatch.setattr(job_service.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        job_service.platform_support, "low_priority_popen_args",
        lambda cmd, system=None: (["nice-ish", *cmd], {"creationflags": 0x4000}),
    )
    job_service.run_site_scrape_subprocess(Path("python"), "indeed", "print([])")
    assert seen["cmd"][0] == "nice-ish"
    assert seen["kwargs"]["creationflags"] == 0x4000
    assert seen["kwargs"]["timeout"] == job_service.SUBPROCESS_SCRAPE_TIMEOUT
