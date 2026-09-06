"""Cross-platform behaviour of services.platform_support.

The point of the shim is that BOTH branches stay correct while only one of them
is ever exercised at runtime on a given machine. So these tests drive Windows
and Linux explicitly via `system=` and never depend on the host OS -- the whole
file must pass identically on Windows and on Linux.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from services import platform_support as ps


def _norm(paths) -> list[str]:
    """Compare candidate paths without caring about the host's separator.

    `pathlib.Path` is flavoured by the OS running the tests, so a POSIX
    candidate built on Windows stringifies with backslashes. That is harmless at
    runtime -- each branch only ever executes on its own OS -- but it would make
    these deliberately cross-platform assertions unrunnable on one of the two.
    """
    return [str(p).replace("\\", "/") for p in paths]


# ---------------------------------------------------------------------------
# Chrome executable discovery
# ---------------------------------------------------------------------------

def test_windows_candidates_cover_the_three_standard_install_locations():
    env = {
        "LOCALAPPDATA": r"C:\Users\bob\AppData\Local",
        "ProgramFiles": r"C:\Program Files",
        "ProgramFiles(x86)": r"C:\Program Files (x86)",
    }
    found = _norm(ps.chrome_executable_candidates(ps.WINDOWS, env))

    assert all(p.endswith("chrome.exe") for p in found)
    assert "C:/Users/bob/AppData/Local/Google/Chrome/Application/chrome.exe" in found
    assert "C:/Program Files/Google/Chrome/Application/chrome.exe" in found
    assert "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe" in found


def test_linux_candidates_prefer_branded_chrome_over_chromium():
    found = _norm(ps.chrome_executable_candidates(ps.LINUX, {}))

    google_idx = min(i for i, p in enumerate(found) if "google-chrome" in p or "/opt/google" in p)
    chromium_idx = min(i for i, p in enumerate(found) if "chromium" in p)
    assert google_idx < chromium_idx, "branded Chrome must be probed before Chromium"
    assert "/opt/google/chrome/chrome" in found
    assert not any(p.endswith(".exe") for p in found)


def test_env_override_is_probed_before_every_builtin_location():
    for system in (ps.WINDOWS, ps.LINUX):
        candidates = ps.chrome_executable_candidates(system, {"CHROME_EXECUTABLE": "/custom/chrome"})
        assert _norm(candidates)[0] == "/custom/chrome"


def test_blank_env_override_is_ignored_not_treated_as_a_path():
    candidates = ps.chrome_executable_candidates(ps.LINUX, {"CHROME_EXECUTABLE": "   "})
    assert _norm(candidates)[0] == "/opt/google/chrome/chrome"


def test_find_chrome_returns_first_existing_candidate():
    present = {"/usr/bin/chromium", "/usr/bin/google-chrome"}
    resolved = ps.find_chrome(
        system=ps.LINUX, env={},
        exists=lambda p: str(p).replace("\\", "/") in present,
        which=lambda _n: None,
    )
    # google-chrome is earlier in the candidate order than chromium.
    assert resolved.replace("\\", "/") == "/usr/bin/google-chrome"


def test_find_chrome_falls_back_to_path_lookup_when_nothing_is_installed():
    resolved = ps.find_chrome(
        system=ps.LINUX, env={},
        exists=lambda _p: False,
        which=lambda name: "/snap/bin/chromium" if name == "chromium" else None,
    )
    assert resolved == "/snap/bin/chromium"


def test_find_chrome_returns_none_when_no_browser_exists_anywhere():
    assert ps.find_chrome(
        system=ps.LINUX, env={}, exists=lambda _p: False, which=lambda _n: None,
    ) is None


# ---------------------------------------------------------------------------
# User-data directory discovery
# ---------------------------------------------------------------------------

def test_linux_user_data_uses_xdg_config_home_when_set():
    found = _norm(ps.chrome_user_data_candidates(
        ps.LINUX, {"XDG_CONFIG_HOME": "/xdg"}, home=Path("/home/bob"),
    ))
    assert "/xdg/google-chrome" in found
    assert not any(p.startswith("/home/bob/.config") for p in found)


def test_linux_user_data_defaults_to_dot_config():
    found = _norm(ps.chrome_user_data_candidates(ps.LINUX, {}, home=Path("/home/bob")))
    assert found == ["/home/bob/.config/google-chrome", "/home/bob/.config/chromium"]


def test_windows_user_data_is_localappdata_chrome_user_data():
    found = _norm(ps.chrome_user_data_candidates(
        ps.WINDOWS, {"LOCALAPPDATA": r"C:\Users\bob\AppData\Local"},
    ))
    assert "C:/Users/bob/AppData/Local/Google/Chrome/User Data" in found


def test_user_data_dirs_filters_to_what_actually_exists(tmp_path):
    real = tmp_path / ".config" / "google-chrome"
    real.mkdir(parents=True)

    dirs = ps.chrome_user_data_dirs(system=ps.LINUX, env={}, home=tmp_path)
    assert dirs == [real]


# ---------------------------------------------------------------------------
# User agent
# ---------------------------------------------------------------------------

def test_user_agent_platform_token_tracks_the_host_os():
    assert "Windows NT 10.0" in ps.chrome_user_agent(ps.WINDOWS)
    assert "X11; Linux x86_64" in ps.chrome_user_agent(ps.LINUX)
    # Never leaks the headless marker Reddit blocks on.
    for system in (ps.WINDOWS, ps.LINUX):
        assert "HeadlessChrome" not in ps.chrome_user_agent(system)


# ---------------------------------------------------------------------------
# Sandbox flags
# ---------------------------------------------------------------------------

def test_no_sandbox_is_opt_in_and_never_applied_on_windows():
    assert ps.chrome_sandbox_args(ps.LINUX, {}) == []
    assert ps.chrome_sandbox_args(ps.WINDOWS, {"CHROME_NO_SANDBOX": "1"}) == []

    args = ps.chrome_sandbox_args(ps.LINUX, {"CHROME_NO_SANDBOX": "1"})
    assert "--no-sandbox" in args and "--disable-dev-shm-usage" in args


# ---------------------------------------------------------------------------
# Process discovery via /proc
# ---------------------------------------------------------------------------

def _fake_proc(root: Path, pid: int, comm: str, cmdline: str) -> None:
    d = root / str(pid)
    d.mkdir(parents=True)
    (d / "comm").write_text(comm + "\n", encoding="utf-8")
    (d / "cmdline").write_bytes(cmdline.replace(" ", "\0").encode() + b"\0")


def test_proc_scan_matches_only_chrome_processes_holding_this_profile(tmp_path):
    _fake_proc(tmp_path, 100, "chrome", "/opt/google/chrome/chrome --user-data-dir=/bot/chrome_profile_runtime")
    _fake_proc(tmp_path, 101, "chrome", "/opt/google/chrome/chrome --user-data-dir=/home/bob/.config/google-chrome")
    _fake_proc(tmp_path, 102, "python3", "python3 src/app.py chrome_profile_runtime")
    (tmp_path / "not-a-pid").mkdir()

    found = ps._posix_chrome_processes("chrome_profile_runtime", proc_root=tmp_path)

    pids = {pid for pid, _ in found}
    assert pids == {100}, "must exclude the user's own Chrome and the non-Chrome process"


def test_proc_scan_tolerates_a_process_exiting_mid_scan(tmp_path):
    _fake_proc(tmp_path, 200, "chrome", "chrome --user-data-dir=/bot/runtime_profile")
    # A pid dir with no readable comm/cmdline: exactly what a race looks like.
    (tmp_path / "201").mkdir()

    found = ps._posix_chrome_processes("runtime_profile", proc_root=tmp_path)
    assert {pid for pid, _ in found} == {200}


def test_proc_scan_returns_empty_when_proc_is_absent(tmp_path):
    assert ps._posix_chrome_processes("anything", proc_root=tmp_path / "missing") == []


def test_iter_proc_cmdlines_decodes_nul_separated_args(tmp_path):
    _fake_proc(tmp_path, 300, "chrome", "chrome --flag-a --flag-b")
    entries = list(ps.iter_proc_cmdlines(tmp_path))
    assert entries[0][0] == 300
    assert entries[0][1] == "chrome"
    assert "--flag-a --flag-b" in entries[0][2]


def test_parse_pid_session_lines_skips_headers_and_blank_lines():
    parsed = ps.parse_pid_session_lines("\nProcessId,SessionId\n1234,1\n5678\nnoise\n")
    assert parsed == [(1234, 1), (5678, None)]


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------

def test_terminate_escalates_to_sigkill_when_sigterm_is_ignored(monkeypatch):
    sent: list[int] = []
    alive = {"value": True}

    def fake_kill(pid, sig):
        sent.append(sig)
        if sig == ps.SIGKILL:
            alive["value"] = False
        elif sig == 0 and not alive["value"]:
            raise ProcessLookupError

    monkeypatch.setattr(os, "kill", fake_kill)

    ok, detail = ps.terminate_process(
        999, system=ps.LINUX, grace_seconds=0.3, sleep=lambda _s: None,
    )

    assert ok is True
    assert ps.SIGTERM in sent, "must try a graceful stop first so Chrome flushes its cookie DB"
    assert ps.SIGKILL in sent, "must escalate when the process ignores SIGTERM"
    assert detail == "required SIGKILL"


def test_terminate_stops_at_sigterm_when_the_process_exits(monkeypatch):
    sent: list[int] = []

    def fake_kill(pid, sig):
        sent.append(sig)
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(os, "kill", fake_kill)

    ok, detail = ps.terminate_process(999, system=ps.LINUX, grace_seconds=1.0, sleep=lambda _s: None)

    assert (ok, detail) == (True, "")
    assert ps.SIGKILL not in sent


def test_terminate_treats_an_already_dead_process_as_success(monkeypatch):
    def fake_kill(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(os, "kill", fake_kill)
    assert ps.terminate_process(999, system=ps.LINUX) == (True, "")


def test_terminate_reports_permission_denied_rather_than_claiming_success(monkeypatch):
    def fake_kill(pid, sig):
        raise PermissionError("not owner")

    monkeypatch.setattr(os, "kill", fake_kill)
    ok, detail = ps.terminate_process(999, system=ps.LINUX)
    assert ok is False
    assert "permission denied" in detail


# ---------------------------------------------------------------------------
# Liveness (run.py's _forcerun depends on this to find the process to kill)
# ---------------------------------------------------------------------------

def test_is_process_alive_posix_reports_dead_on_lookup_error(monkeypatch):
    def fake_kill(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(os, "kill", fake_kill)
    assert ps.is_process_alive(999, system=ps.LINUX) is False


def test_is_process_alive_posix_reports_alive_when_signal_delivers(monkeypatch):
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    assert ps.is_process_alive(999, system=ps.LINUX) is True


def test_is_process_alive_posix_treats_permission_denied_as_alive(monkeypatch):
    """Exists but not ours to signal is still alive -- same rule terminate_process
    and _running_pid's caller both already rely on."""
    def fake_kill(pid, sig):
        raise PermissionError

    monkeypatch.setattr(os, "kill", fake_kill)
    assert ps.is_process_alive(999, system=ps.LINUX) is True


def test_is_process_alive_never_raises_on_this_host():
    """Windows has no signal 0, so os.kill(pid, 0) raises the identical
    generic OSError for a pid whether it is alive or does not exist --
    ProcessLookupError is POSIX-only, so a bare `except OSError` built on it
    silently reports every pid as dead there (the bug this function fixes).
    Its Windows branch is real ctypes (OpenProcess/GetExitCodeProcess), which
    -- like current_session_id's ctypes branch -- cannot be forced
    cross-platform from a non-Windows host, so this exercises the host's own
    real branch rather than passing an explicit system=.
    """
    assert ps.is_process_alive(os.getpid()) is True
    assert ps.is_process_alive(999_999_999) is False


def test_kill_processes_logs_and_counts_each_outcome(monkeypatch):
    monkeypatch.setattr(ps, "find_processes_using_profile", lambda d, system=None: [(1, 5), (2, 5)])
    monkeypatch.setattr(ps, "current_session_id", lambda system=None: 5)
    results = {1: (True, ""), 2: (False, "permission denied")}
    monkeypatch.setattr(ps, "terminate_process", lambda pid, system=None: results[pid])

    logged: list[str] = []
    killed = ps.kill_processes_using_profile("runtime_profile", system=ps.LINUX, log=logged.append)

    assert killed == 1
    assert any("killed PID 1" in m for m in logged)
    assert any("FAILED to kill PID 2" in m and "permission denied" in m for m in logged)


def test_kill_processes_flags_cross_session_owners(monkeypatch):
    monkeypatch.setattr(ps, "find_processes_using_profile", lambda d, system=None: [(7, 0)])
    monkeypatch.setattr(ps, "current_session_id", lambda system=None: 1)
    monkeypatch.setattr(ps, "terminate_process", lambda pid, system=None: (False, "denied"))

    logged: list[str] = []
    ps.kill_processes_using_profile("runtime_profile", system=ps.WINDOWS, log=logged.append)

    assert any("owned by session 0" in m and "ours: 1" in m for m in logged)


def test_kill_processes_reports_when_nothing_holds_the_profile(monkeypatch):
    monkeypatch.setattr(ps, "find_processes_using_profile", lambda d, system=None: [])
    logged: list[str] = []
    assert ps.kill_processes_using_profile("p", system=ps.LINUX, log=logged.append) == 0
    assert any("no Chrome processes found" in m for m in logged)


# ---------------------------------------------------------------------------
# Detached spawn (.reset)
# ---------------------------------------------------------------------------

class _StillRunning:
    """A Popen stand-in for a child that is alive and stays alive.

    wait() raising TimeoutExpired is what a real detached child does inside the
    settle window, and it is the whole point of the check: a bare object() with
    no wait() at all tests nothing about liveness and quietly exercised the
    "could not confirm" path instead.
    """

    def wait(self, timeout=None):
        raise ps.subprocess.TimeoutExpired(cmd="child", timeout=timeout)


def test_spawn_detached_uses_windows_detached_creationflags(monkeypatch):
    calls: list[tuple[list[str], dict]] = []

    def fake_popen(args, **kwargs):
        calls.append((list(args), kwargs))
        return _StillRunning()

    monkeypatch.setattr(ps.subprocess, "Popen", fake_popen)

    ok, detail = ps.spawn_detached(
        ["python", "run.py", "--forcerun"], cwd="/bot", system=ps.WINDOWS
    )

    assert (ok, detail) == (True, "")
    args, kwargs = calls[0]
    assert args == ["python", "run.py", "--forcerun"]
    assert kwargs["creationflags"] == ps.DETACHED_PROCESS | ps.CREATE_NEW_PROCESS_GROUP
    assert "start_new_session" not in kwargs
    assert kwargs["cwd"] == "/bot"


def test_spawn_detached_uses_posix_new_session(monkeypatch):
    calls: list[tuple[list[str], dict]] = []

    def fake_popen(args, **kwargs):
        calls.append((list(args), kwargs))
        return _StillRunning()

    monkeypatch.setattr(ps.subprocess, "Popen", fake_popen)

    ok, detail = ps.spawn_detached(["python3", "run.py", "--forcerun"], system=ps.LINUX)

    assert (ok, detail) == (True, "")
    _, kwargs = calls[0]
    assert kwargs["start_new_session"] is True
    assert "creationflags" not in kwargs


def test_spawn_detached_reports_popen_failure_without_raising(monkeypatch):
    def fake_popen(args, **kwargs):
        raise OSError("no such file")

    monkeypatch.setattr(ps.subprocess, "Popen", fake_popen)

    ok, detail = ps.spawn_detached(["python3", "run.py"], system=ps.LINUX)

    assert ok is False
    assert "no such file" in detail


def test_spawn_detached_actually_runs_independently_of_the_caller(tmp_path: Path):
    """Real (unmocked) proof that .reset's restart is a genuine new process --
    not a fork/exec-replace of the caller and not a clone that would keep
    running the caller's already-imported code.

    If spawn_detached ever changed to os.execv() the calling process instead of
    subprocess.Popen()-ing a new one, this test process would itself be
    replaced by the child and never reach the assertions below. Reaching them
    at all is already part of the proof; the marker file, written by a fresh
    interpreter reading the script argument off disk, is the rest.
    """
    marker = tmp_path / "marker.txt"
    script = tmp_path / "child.py"
    script.write_text(
        f"open({str(marker)!r}, 'w').write('ran-independently')\n",
        encoding="utf-8",
    )

    import sys
    import time

    ok, detail = ps.spawn_detached([sys.executable, str(script)], cwd=str(tmp_path))
    assert (ok, detail) == (True, "")

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)

    assert marker.exists(), "the detached child never ran its own (fresh, on-disk) code"
    assert marker.read_text(encoding="utf-8") == "ran-independently"


# ---------------------------------------------------------------------------
# Lock artifacts
# ---------------------------------------------------------------------------

def test_profile_lock_paths_cover_singleton_and_lockfile_artifacts():
    names = {p.name for p in ps.profile_lock_paths("/bot/profile")}
    assert {"SingletonLock", "SingletonCookie", "SingletonSocket", "lockfile"} <= names
    # The nested Default/lockfile must be included, not just the top-level one.
    assert any(p.parent.name == "Default" for p in ps.profile_lock_paths("/bot/profile"))


def test_current_session_id_never_raises_on_this_host():
    value = ps.current_session_id()
    assert value is None or isinstance(value, int)


@pytest.mark.skipif(not ps.is_windows(), reason="exercises the Windows ctypes branch on its own host")
def test_is_process_alive_treats_windows_access_denied_as_alive():
    """The POSIX branch already answers True for "exists, just not ours".
    OpenProcess returning a null handle for ERROR_ACCESS_DENIED must mean the
    same, not "dead" -- reporting a live instance dead is what lets _forcerun
    clear its lock and start a second bot beside it.

    pid 4 is the Windows System process: always running, never openable.
    """
    assert ps.is_process_alive(4) is True


def test_process_cmdline_falls_back_to_cim_when_the_fast_path_cannot_answer(monkeypatch):
    """The ntdll reader needs Windows 8.1+ for its information class, so the
    original CIM query is kept rather than replaced. If the fast path cannot
    answer, the slow one must still get its turn -- otherwise this "speedup"
    would silently remove the only branch that works on an older host.
    """
    calls: list[int] = []

    monkeypatch.setattr(ps, "_windows_cmdline_ctypes", lambda pid: None)
    monkeypatch.setattr(
        ps, "_windows_cmdline_powershell",
        lambda pid: calls.append(pid) or "python.exe src\app.py",
    )

    assert ps.process_cmdline(4321, system=ps.WINDOWS) == "python.exe src\app.py"
    assert calls == [4321], "the CIM fallback was not reached"


def test_process_cmdline_prefers_the_fast_path_and_skips_cim_when_it_answers(monkeypatch):
    """The whole point is to stop paying for the subprocess. If the fast path
    answered, CIM must not be invoked at all -- a fallback that always runs
    anyway would keep the latency this change exists to remove.
    """
    monkeypatch.setattr(ps, "_windows_cmdline_ctypes", lambda pid: "python.exe src\app.py")

    def _must_not_run(pid):  # pragma: no cover - asserts by being called
        raise AssertionError("CIM ran even though the fast path answered")

    monkeypatch.setattr(ps, "_windows_cmdline_powershell", _must_not_run)
    assert ps.process_cmdline(4321, system=ps.WINDOWS) == "python.exe src\app.py"


@pytest.mark.skipif(not ps.is_windows(), reason="exercises the Windows ctypes branch on its own host")
def test_process_cmdline_reads_this_processs_arguments_not_just_its_exe():
    """_is_our_bot looks for "app.py", which lives in the *arguments*, so an
    exe-path-only answer (what QueryFullProcessImageNameW would give) is
    useless to the only caller. This asserts against the real running
    interpreter: pytest is always invoked with arguments.
    """
    line = ps.process_cmdline(os.getpid())
    assert line is not None, "could not read this process's own command line"
    assert "python" in line.lower()
    # The executable alone is not enough -- there must be something after it.
    assert len(line.split()) > 1, f"got an exe path with no arguments: {line!r}"


@pytest.mark.skipif(not ps.is_windows(), reason="exercises the Windows ctypes branch on its own host")
def test_process_cmdline_answers_well_inside_the_timeout_it_used_to_blow():
    """The bug this guards: CIM took 10.7s-30s under load against a 10s timeout,
    so process_cmdline returned None for a live pid, and run.py's _is_our_bot
    fails open on None -- leaving --forcerun with no pid-reuse guard at all.

    The ceiling has to sit between the two implementations to mean anything.
    Measured on this host: ntdll 0.28ms, CIM 380ms unloaded and 10s+ under
    load. 50ms is ~180x the fast path (so load cannot make it flake) and ~7x
    below the slow one's *best* case (so it cannot pass while the subprocess is
    still there). An earlier 0.5s ceiling was verified useless by mutation --
    disabling the fast path left it passing.
    """
    import time

    ps.process_cmdline(os.getpid())  # discount one-time WinDLL load
    start = time.perf_counter()
    line = ps.process_cmdline(os.getpid())
    elapsed = time.perf_counter() - start

    assert line is not None
    assert elapsed < 0.05, f"process_cmdline took {elapsed:.3f}s -- the subprocess is still in the path"


@pytest.mark.skipif(not ps.is_windows(), reason="exercises the Windows ctypes branch on its own host")
def test_process_cmdline_returns_none_for_a_pid_that_does_not_exist():
    """None means "cannot be read". A dead pid must reach that, not raise and
    not invent a command line for a process that is not there.
    """
    assert ps.process_cmdline(999_999_999) is None


@pytest.mark.skipif(not ps.is_windows(), reason="exercises the Windows ctypes branch on its own host")
def test_both_windows_readers_agree_on_this_process():
    """The fast path is only a safe substitute if it reports the same thing the
    branch it front-runs would have. Compared on the executable, which both
    return; CIM omits some argument forms the ntdll reader keeps, so the
    arguments themselves are deliberately not compared.
    """
    fast = ps._windows_cmdline_ctypes(os.getpid())
    slow = ps._windows_cmdline_powershell(os.getpid())
    assert fast is not None, "ntdll reader failed on this host"
    if slow is None:
        pytest.skip("CIM did not answer on this host, nothing to compare against")
    assert fast.split()[0].lower() == slow.split()[0].lower()
