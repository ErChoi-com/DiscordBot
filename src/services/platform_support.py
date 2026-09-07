"""Single home for every OS-specific decision the app makes.

The rest of the codebase must not branch on `os.name` / `sys.platform`; it calls
these helpers instead. That keeps the Windows and Linux paths side by side (so
neither rots unnoticed) and lets tests exercise *both* branches on whichever OS
is running, by passing an explicit `system=` argument.

Env overrides, honoured on both platforms:
    CHROME_EXECUTABLE      path to the Chrome/Chromium binary
    CHROME_USER_DATA_DIR   the browser's user-data dir (the parent of Default/)
    CHROME_NO_SANDBOX      set to 1 in containers only
"""
from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterator, Mapping

WINDOWS = "Windows"
LINUX = "Linux"

# signal.SIGKILL is undefined on Windows, so referencing it directly would make
# the POSIX termination path un-importable there -- and so untestable from a
# Windows dev machine, which is the branch most likely to rot.
SIGKILL = getattr(signal, "SIGKILL", 9)
SIGTERM = getattr(signal, "SIGTERM", 15)

# subprocess.DETACHED_PROCESS / CREATE_NEW_PROCESS_GROUP are Windows-only
# attributes of the subprocess module -- same rot risk as SIGKILL above.
DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)

# Scheduling priority for work the user is not waiting on. The scheduler
# reserves *workers* for interactive commands, but a worker is not a core:
# with twelve cores pinned by scrape threads, a reserved worker that has its
# task still competes for the CPU with every one of them, and so does the
# event loop that carries the Discord heartbeat. Priority is the layer the
# reservation cannot reach.
BELOW_NORMAL_PRIORITY_CLASS = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
THREAD_PRIORITY_BELOW_NORMAL = -1
THREAD_PRIORITY_NORMAL = 0
# POSIX nice for background work: comfortably below 0 without reaching the
# 19 that lets a busy foreground starve it entirely.
BACKGROUND_NICE = 10


def current_system() -> str:
    return platform.system()


def is_windows(system: str | None = None) -> bool:
    return (system or current_system()) == WINDOWS


# ---------------------------------------------------------------------------
# Chrome discovery
# ---------------------------------------------------------------------------

def chrome_executable_candidates(
    system: str | None = None,
    env: Mapping[str, str] | None = None,
) -> list[Path]:
    """Ordered Chrome/Chromium locations to probe. Pure: no filesystem access,
    so tests can assert the Linux ordering while running on Windows."""
    system = system or current_system()
    env = os.environ if env is None else env

    override = (env.get("CHROME_EXECUTABLE") or "").strip()
    candidates: list[Path] = [Path(override)] if override else []

    if system == WINDOWS:
        for base in (env.get("LOCALAPPDATA", ""),
                     env.get("ProgramFiles", r"C:\Program Files"),
                     env.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
            if base:
                candidates.append(Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe")
    else:
        # Branded Chrome first: it ships the proprietary codecs and the build
        # Reddit's fingerprinting expects, so Chromium is a real fallback rather
        # than an equal choice.
        candidates += [
            Path("/opt/google/chrome/chrome"),
            Path("/usr/bin/google-chrome"),
            Path("/usr/bin/google-chrome-stable"),
            Path("/usr/bin/chromium"),
            Path("/usr/bin/chromium-browser"),
        ]
    return candidates


_CHROME_PATH_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")


def find_chrome(
    system: str | None = None,
    env: Mapping[str, str] | None = None,
    exists: Callable[[Path], bool] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> str | None:
    """Absolute path to an installed Chrome/Chromium, or None."""
    # Resolved at call time, not bound as a default: a default argument would
    # capture shutil.which at import and silently ignore any later patching of
    # it (which is exactly how the LaTeX probe tests intercept lookups).
    which = which or shutil.which
    exists = exists or (lambda p: p.is_file())
    for candidate in chrome_executable_candidates(system=system, env=env):
        if exists(candidate):
            return str(candidate)
    for name in _CHROME_PATH_NAMES:
        resolved = which(name)
        if resolved:
            return resolved
    return None


def chrome_user_data_candidates(
    system: str | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> list[Path]:
    """User-data dirs of the *user's own* browsers -- harvest sources for the
    Reddit session cookie, never launch targets."""
    system = system or current_system()
    env = os.environ if env is None else env
    home = home or Path(env.get("HOME") or env.get("USERPROFILE") or Path.home())

    override = (env.get("CHROME_USER_DATA_DIR") or "").strip()
    candidates: list[Path] = [Path(override)] if override else []

    if system == WINDOWS:
        local = env.get("LOCALAPPDATA", "")
        if local:
            candidates.append(Path(local) / "Google" / "Chrome" / "User Data")
    else:
        config = Path(env.get("XDG_CONFIG_HOME") or (home / ".config"))
        candidates += [config / "google-chrome", config / "chromium"]
    return candidates


def chrome_user_data_dirs(
    system: str | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    exists: Callable[[Path], bool] | None = None,
) -> list[Path]:
    exists = exists or (lambda p: p.is_dir())
    return [p for p in chrome_user_data_candidates(system=system, env=env, home=home) if exists(p)]


def chrome_user_agent(system: str | None = None) -> str:
    """UA to present. Chrome sends Sec-CH-UA-Platform from the real OS and sites
    cross-check it against the UA, so a Windows UA sent from Linux is a
    detectable inconsistency. Keep the platform honest; spoof nothing else."""
    token = "Windows NT 10.0; Win64; x64" if is_windows(system) else "X11; Linux x86_64"
    return (
        f"Mozilla/5.0 ({token}) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
    )


def chrome_sandbox_args(system: str | None = None, env: Mapping[str, str] | None = None) -> list[str]:
    """Chrome's setuid sandbox cannot initialise as an unprivileged container
    PID 1 -- the most common Linux launch failure. Opt-in, because dropping the
    sandbox on a normal host is a real security regression."""
    env = os.environ if env is None else env
    if is_windows(system):
        return []
    if (env.get("CHROME_NO_SANDBOX") or "").strip().lower() in {"1", "true", "yes"}:
        return ["--no-sandbox", "--disable-dev-shm-usage"]
    return []


# ---------------------------------------------------------------------------
# Process inspection / termination
# ---------------------------------------------------------------------------

def current_session_id(system: str | None = None) -> int | None:
    """Login-session id, for attributing profile-lock owners. Windows session 0
    is the non-interactive service session, whose Chrome locks are unkillable
    from a desktop session; POSIX sids carry no such privilege boundary but
    still distinguish this unit from a desktop login."""
    if is_windows(system):
        try:
            import ctypes

            sid = ctypes.c_ulong()
            pid = ctypes.windll.kernel32.GetCurrentProcessId()
            if ctypes.windll.kernel32.ProcessIdToSessionId(pid, ctypes.byref(sid)):
                return int(sid.value)
        except Exception:
            pass
        return None
    try:
        return os.getsid(0)  # type: ignore[attr-defined]  # POSIX-only
    except (AttributeError, OSError):
        return None


def parse_pid_session_lines(out: str) -> list[tuple[int, int | None]]:
    """Parse 'pid,sessionid' lines (sessionid optional)."""
    procs: list[tuple[int, int | None]] = []
    for line in out.splitlines():
        parts = line.strip().split(",")
        if not parts[0].strip().isdigit():
            continue
        session = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip().isdigit() else None
        procs.append((int(parts[0].strip()), session))
    return procs


def _windows_chrome_processes(profile_dir: str) -> list[tuple[int, int | None]]:
    # PowerShell/CIM -- reliable on Windows 11 where WMIC is deprecated.
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"name='chrome.exe'\" | "
             f"Where-Object {{ $_.CommandLine -like '*{profile_dir}*' }} | "
             "ForEach-Object { \"$($_.ProcessId),$($_.SessionId)\" }"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
        return parse_pid_session_lines(out)
    except Exception:
        pass
    # Fallback to WMIC for older Windows. CSV columns are alphabetical:
    # Node,ProcessId,SessionId
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where",
             f"name='chrome.exe' and commandline like '%{profile_dir}%'",
             "get", "processid,sessionid", "/format:csv"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
    except Exception:
        return []
    procs: list[tuple[int, int | None]] = []
    for line in out.splitlines():
        parts = line.strip().split(",")
        if "ProcessId" in line or len(parts) < 3 or not parts[-2].strip().isdigit():
            continue
        session = int(parts[-1].strip()) if parts[-1].strip().isdigit() else None
        procs.append((int(parts[-2].strip()), session))
    return procs


_POSIX_CHROME_NAMES = ("chrome", "chromium")


def iter_proc_cmdlines(proc_root: Path = Path("/proc")) -> Iterator[tuple[int, str, str]]:
    """Yield (pid, comm, cmdline) from /proc. Reading /proc directly rather than
    shelling out to pgrep/ps: no dependency on which procps variant is
    installed, and a process vanishing mid-scan is a caught OSError."""
    try:
        entries = sorted(proc_root.iterdir())
    except OSError:
        return
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            comm = (entry / "comm").read_text(encoding="utf-8", errors="replace").strip()
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        yield int(entry.name), comm, raw.replace(b"\0", b" ").decode("utf-8", "replace")


def _posix_chrome_processes(
    profile_dir: str,
    proc_root: Path = Path("/proc"),
) -> list[tuple[int, int | None]]:
    procs: list[tuple[int, int | None]] = []
    for pid, comm, cmdline in iter_proc_cmdlines(proc_root):
        if not any(name in comm for name in _POSIX_CHROME_NAMES):
            continue
        if profile_dir not in cmdline:
            continue
        try:
            session: int | None = os.getsid(pid)  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            session = None
        procs.append((pid, session))
    return procs


def total_physical_memory(system: str | None = None) -> int | None:
    """Physical RAM in bytes, or None if it cannot be determined.

    Lives here rather than in capacity.py so the ctypes/Windows call stays in
    the one module that is allowed to branch on the OS. On Linux the caller
    reads /proc/meminfo first, so this is the Windows path in practice.
    """
    if not is_windows(system):
        return None
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except Exception:
        pass
    return None


def _windows_cmdline_powershell(pid: int) -> str | None:
    """Read `pid`'s command line via CIM. The original implementation, kept.

    Correct wherever it answers, and it is the only branch that works before
    Windows 8.1, so it stays as the fallback. Its weakness is latency, not
    accuracy: CIM/WMI has been measured on this host taking 10.7s and upwards
    of 30s under load, past this timeout, and a timeout here is reported as
    "cannot be read" -- which _is_our_bot fails open on.
    """
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        ).strip()
        return out or None
    except Exception:
        return None


def _windows_cmdline_ctypes(pid: int) -> str | None:
    """Read `pid`'s command line straight from ntdll, or None if unavailable.

    Same reasoning as is_process_alive's ctypes branch, for the same reason:
    the subprocess this replaces is slow enough under load to blow its own
    timeout, and a query that times out is indistinguishable from a pid that
    does not exist. This is a direct call with no subprocess, so it answers in
    well under a millisecond and the timeout case stops arising in practice.

    ProcessCommandLineInformation (class 60) needs Windows 8.1 or newer; on
    anything older NtQueryInformationProcess reports an invalid info class and
    this returns None so the CIM fallback runs. It reads under
    PROCESS_QUERY_LIMITED_INFORMATION, the same right is_process_alive already
    asks for -- notably *not* PROCESS_VM_READ, so this does not need the PEB
    walk that reading another process's memory would.

    QueryFullProcessImageNameW is not a substitute: it yields the executable
    path only, and the caller (run.py's _is_our_bot) has to find "app.py",
    which lives in the arguments.
    """
    try:
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        ProcessCommandLineInformation = 60
        STATUS_SUCCESS = 0x00000000
        STATUS_INFO_LENGTH_MISMATCH = 0xC0000004

        class _UnicodeString(ctypes.Structure):
            _fields_ = [
                ("Length", wintypes.USHORT),
                ("MaximumLength", wintypes.USHORT),
                ("Buffer", ctypes.c_void_p),
            ]

        ntdll = ctypes.WinDLL("ntdll")
        kernel32 = ctypes.WinDLL("kernel32")
        ntdll.NtQueryInformationProcess.restype = ctypes.c_ulong
        ntdll.NtQueryInformationProcess.argtypes = [
            wintypes.HANDLE, ctypes.c_ulong, ctypes.c_void_p,
            ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]

        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return None
        try:
            # Ask for the size first: the answer is a UNICODE_STRING whose text
            # follows it in the same allocation, so the length is not knowable
            # up front. A zero-length probe is expected to come back
            # STATUS_INFO_LENGTH_MISMATCH with the size written out.
            needed = ctypes.c_ulong(0)
            status = ntdll.NtQueryInformationProcess(
                handle, ProcessCommandLineInformation, None, 0, ctypes.byref(needed)
            )
            if status != STATUS_INFO_LENGTH_MISMATCH or not needed.value:
                return None
            buf = (ctypes.c_byte * needed.value)()
            status = ntdll.NtQueryInformationProcess(
                handle, ProcessCommandLineInformation, ctypes.byref(buf),
                needed.value, ctypes.byref(needed),
            )
            if status != STATUS_SUCCESS:
                return None
            us = ctypes.cast(buf, ctypes.POINTER(_UnicodeString)).contents
            if not us.Buffer or not us.Length:
                # An empty command line is a real answer, but it tells the
                # caller nothing, so report it the same as unreadable and let
                # the fallback have a turn.
                return None
            text = ctypes.wstring_at(us.Buffer, us.Length // ctypes.sizeof(ctypes.c_wchar))
            return text.strip() or None
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return None


def process_cmdline(pid: int, system: str | None = None) -> str | None:
    """Full command line of `pid`, or None if it cannot be read.

    Used to corroborate a pid recorded in a lock file before acting on it.
    Linux recycles pids from a 32768-default space within hours on a busy host,
    so a stale lock can name a pid that now belongs to something else entirely
    -- killing it on that basis would take down an unrelated process.

    On Windows this tries ntdll directly and falls back to the original CIM
    query. Both branches are kept because they fail differently: the fast one
    is unavailable before Windows 8.1, the slow one times out under load.
    """
    if is_windows(system):
        fast = _windows_cmdline_ctypes(pid)
        if fast is not None:
            return fast
        # Deliberately still reached: the ctypes path needs Windows 8.1+ for the
        # information class it asks for, so the original query stays as the
        # fallback rather than being replaced by it.
        return _windows_cmdline_powershell(pid)
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return None
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip() or None


def is_process_alive(pid: int, system: str | None = None) -> bool:
    """Whether `pid` currently names a live process.

    os.kill(pid, 0) is the POSIX idiom for this, but 0 is not a valid signal
    value on Windows: CPython's os.kill raises the same generic OSError
    (WinError 87, "the parameter is incorrect") for a pid whether it is alive
    or does not exist at all, so a POSIX-style except-OSError check built on
    it reports every pid as dead there.

    The obvious Windows fix -- shell out to Get-CimInstance, as
    process_cmdline does -- trades that bug for a slower one: CIM/WMI can
    take 10s of seconds to answer under load, well past any timeout short
    enough for a liveness check to still be useful, and a caller looping
    while a process shuts down (see run.py's _forcerun) needs this to answer
    near-instantly. OpenProcess+GetExitCodeProcess is a direct kernel32 call
    with no subprocess involved -- the same ctypes approach current_session_id
    already uses in this module -- so it answers in microseconds.
    """
    if is_windows(system):
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            ERROR_ACCESS_DENIED = 5
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                # Access denied means the process exists and is not ours to
                # open -- the same case the POSIX branch answers True for.
                # Reporting it dead would let _forcerun clear a live instance's
                # lock and start a second bot beside it.
                return kernel32.GetLastError() == ERROR_ACCESS_DENIED
            try:
                exit_code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours to signal
    except OSError:
        return False
    return True


def find_processes_using_profile(
    profile_dir: str,
    system: str | None = None,
) -> list[tuple[int, int | None]]:
    """(pid, session_id) of every Chrome process holding this profile dir."""
    if is_windows(system):
        return _windows_chrome_processes(profile_dir)
    return _posix_chrome_processes(profile_dir)


def terminate_process(
    pid: int,
    system: str | None = None,
    grace_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bool, str]:
    """Kill `pid`. Returns (succeeded, detail); detail is empty on clean success.

    POSIX escalates SIGTERM -> SIGKILL so Chrome can flush its cookie DB first;
    a profile killed mid-write is how stale SingletonLock artifacts appear."""
    if is_windows(system):
        result = subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, text=True)
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or result.stdout or "").strip() or f"rc={result.returncode}"

    try:
        os.kill(pid, SIGTERM)
    except ProcessLookupError:
        return True, ""
    except PermissionError as exc:
        return False, f"permission denied ({exc})"
    except OSError as exc:
        return False, str(exc)

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        sleep(0.1)
        try:
            os.kill(pid, 0)
        except OSError:
            return True, ""

    try:
        os.kill(pid, SIGKILL)
    except ProcessLookupError:
        return True, ""
    except OSError as exc:
        return False, f"SIGKILL failed ({exc})"
    return True, "required SIGKILL"


#: How long spawn_detached waits to see whether the child is still alive.
#:
#: Long enough for an interpreter to fail on the things that actually fail at
#: startup -- a missing venv, an ImportError, a syntax error in a module
#: imported at the top of run.py -- and short enough to sit inside a Discord
#: command without the caller noticing. A child that is still running after
#: this has reached its own code; nothing here can promise more than that.
SPAWN_SETTLE_SECONDS = 1.5


def spawn_detached(
    args: list[str],
    cwd: Path | str | None = None,
    system: str | None = None,
    log_path: Path | str | None = None,
    settle_seconds: float = SPAWN_SETTLE_SECONDS,
) -> tuple[bool, str]:
    """Launch `args` as a fully independent process: no inherited stdio, and no
    process-group / job-object membership that would pull it down if the
    spawning process dies moments later.

    Built for the self-restart command (`.reset`): it spawns the *next*
    `run.py --forcerun` before that invocation terminates this very process,
    so the child must keep running after its parent is gone -- a plain
    subprocess.Popen() child is not guaranteed to survive that on Windows.

    Reports whether the child is still alive after `settle_seconds`, not merely
    whether Popen returned. Those are different claims and the gap between them
    is where the restart command was losing. A child that dies immediately --
    the venv moved, a module fails to import, run.py has a syntax error -- made
    Popen succeed, so `.reset` answered "Restarting..." and latched
    _restart_spawned, and the bot then neither restarted nor accepted another
    reset for the rest of its life. The failure looked exactly like success and
    disabled the one command that could have retried it.

    `log_path` is the other half. The child's output went to DEVNULL, so the
    one process that knew why the restart failed wrote its reason nowhere.
    Point this at a file and the traceback survives the parent that is about to
    be killed. It is opened in append mode -- a restart loop should leave a
    history, not overwrite the evidence each time round.

    settle_seconds=0 skips the wait for a caller that genuinely only wants the
    process created.
    """
    sink = subprocess.DEVNULL
    handle = None
    if log_path is not None:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            handle = open(log_path, "a", encoding="utf-8", errors="replace")
        except OSError:
            # An unwritable log must not stop the restart; losing the record is
            # far cheaper than refusing to start the process it describes.
            handle = None
        else:
            sink = handle

    kwargs: dict = {
        "cwd": str(cwd) if cwd is not None else None,
        "stdin": subprocess.DEVNULL,
        "stdout": sink,
        "stderr": subprocess.STDOUT if handle is not None else sink,
    }
    if is_windows(system):
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(args, **kwargs)
    except OSError as exc:
        return False, str(exc)
    finally:
        # This process's copy of the handle; the child holds its own.
        if handle is not None:
            handle.close()

    if settle_seconds <= 0:
        return True, ""

    try:
        code = proc.wait(timeout=settle_seconds)
    except subprocess.TimeoutExpired:
        return True, ""  # still running, which is what "started" means here
    except Exception:  # noqa: BLE001 -- a Popen without wait() must not fail a spawn
        # The process was created; that much is certain. Only the liveness
        # check is unavailable, and reporting a spawn as failed because the
        # confirmation could not run would be a worse lie than not confirming.
        return True, ""

    # Exiting zero is a short task that finished, not a failure -- callers use
    # this for fire-and-forget work too. A non-zero exit inside the settle
    # window is the case worth catching: the interpreter never reached the
    # program, which for `.reset` means the bot did not restart.
    if code == 0:
        return True, ""
    detail = f"exited immediately with code {code}"
    if log_path is not None:
        detail += f"; see {Path(log_path).name}"
    return False, detail


def kill_processes_using_profile(
    profile_dir: str,
    system: str | None = None,
    log: Callable[[str], None] = lambda _msg: None,
) -> int:
    """Terminate every Chrome holding `profile_dir`. Returns the count killed."""
    procs = find_processes_using_profile(profile_dir, system=system)
    if not procs:
        log(f"[browser] kill({profile_dir}): no Chrome processes found using this profile")
        return 0

    ours = current_session_id(system)
    killed = 0
    for pid, session in procs:
        if session is not None and ours is not None and session != ours:
            log(f"[browser] kill({profile_dir}): PID {pid} owned by session {session} "
                f"(ours: {ours}) -- different login session; kill may be denied")
        ok, detail = terminate_process(pid, system=system)
        shown = session if session is not None else "?"
        if ok:
            killed += 1
            log(f"[browser] kill({profile_dir}): killed PID {pid} (session {shown})"
                + (f" -- {detail}" if detail else ""))
        else:
            log(f"[browser] kill({profile_dir}): FAILED to kill PID {pid} -- {detail}")
    return killed


# ---------------------------------------------------------------------------
# LaTeX toolchain discovery
# ---------------------------------------------------------------------------

def latex_search_dirs(
    system: str | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> list[Path]:
    """Directories holding TeX binaries that are commonly NOT on PATH.

    This matters more on Linux than it looks: systemd gives a service a minimal
    default PATH, so a TeX Live install under /usr/local/texlive is invisible to
    shutil.which() even though an interactive shell finds it fine.
    """
    system = system or current_system()
    env = os.environ if env is None else env
    home = home or Path(env.get("HOME") or env.get("USERPROFILE") or Path.home())

    override = (env.get("LATEX_BIN_DIR") or "").strip()
    dirs: list[Path] = [Path(override)] if override else []

    if system == WINDOWS:
        local = Path(env.get("LOCALAPPDATA", ""))
        program_files = Path(env.get("ProgramFiles", ""))
        program_files_x86 = Path(env.get("ProgramFiles(x86)", ""))
        dirs += [
            local / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64",
            local / "MiKTeX" / "miktex" / "bin" / "x64",
            program_files / "MiKTeX" / "miktex" / "bin" / "x64",
            program_files_x86 / "MiKTeX" / "miktex" / "bin" / "x64",
        ]
    else:
        # TeX Live nests binaries under a year and an arch directory, so these
        # are resolved by glob rather than listed exhaustively.
        for root in (Path("/usr/local/texlive"), home / ".local" / "texlive"):
            try:
                for year in sorted(root.iterdir(), reverse=True):
                    dirs.extend(sorted((year / "bin").iterdir()))
            except OSError:
                continue
        dirs += [Path("/usr/local/bin"), home / ".local" / "bin", Path("/opt/texbin")]
    return dirs


def find_latex_executable(
    command_name: str,
    system: str | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    which: Callable[[str], str | None] | None = None,
) -> str | None:
    """Locate a TeX binary (pdflatex, xelatex, kpsewhich, ...). PATH first."""
    which = which or shutil.which
    resolved = which(command_name)
    if resolved:
        return resolved

    name = command_name
    if is_windows(system) and not name.lower().endswith(".exe"):
        name = f"{name}.exe"

    for directory in latex_search_dirs(system=system, env=env, home=home):
        candidate = directory / name
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:
            continue
    return None


def _kernel32():
    """kernel32 with the thread-priority prototypes declared.

    GetCurrentThread returns a pseudo-handle (-2). Left to ctypes' defaults it
    comes back as a C int and goes out again as one, which on 64-bit Windows
    is not the HANDLE the next call wants: SetThreadPriority then fails and
    GetThreadPriority answers THREAD_PRIORITY_ERROR_RETURN, both silently.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.GetCurrentThread.restype = wintypes.HANDLE
    kernel32.GetCurrentThread.argtypes = []
    kernel32.SetThreadPriority.restype = wintypes.BOOL
    kernel32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
    kernel32.GetThreadPriority.restype = ctypes.c_int
    kernel32.GetThreadPriority.argtypes = [wintypes.HANDLE]
    return kernel32


def set_current_thread_background(background: bool, system: str | None = None) -> bool:
    """Run the calling thread below normal priority, or back at normal.

    Returns whether the change took. Windows can move a thread both ways.
    Linux threads are schedulable tasks with their own nice value, but an
    unprivileged process may only ever *raise* nice, so the return trip fails
    there and a worker that has run background work stays niced -- which is
    the right outcome for a general worker, and why the scheduler never lowers
    its reserved interactive workers in the first place.
    """
    if is_windows(system):
        try:
            kernel32 = _kernel32()
            level = THREAD_PRIORITY_BELOW_NORMAL if background else THREAD_PRIORITY_NORMAL
            return bool(kernel32.SetThreadPriority(kernel32.GetCurrentThread(), level))
        except Exception:
            return False
    try:
        import threading

        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(),  # type: ignore[attr-defined]
                       BACKGROUND_NICE if background else 0)
        return True
    except Exception:
        return False


def current_thread_priority(system: str | None = None) -> int | None:
    """The calling thread's scheduling priority, in the OS's own units
    (Windows: 0 normal, -1 below normal; POSIX: the nice value). None when
    it cannot be read."""
    if is_windows(system):
        try:
            kernel32 = _kernel32()
            return int(kernel32.GetThreadPriority(kernel32.GetCurrentThread()))
        except Exception:
            return None
    try:
        import threading

        return int(os.getpriority(os.PRIO_PROCESS, threading.get_native_id()))  # type: ignore[attr-defined]
    except Exception:
        return None


def low_priority_popen_args(
    cmd: list[str], system: str | None = None
) -> tuple[list[str], dict[str, object]]:
    """`cmd` plus the Popen kwargs that start it below normal priority.

    Windows takes a creation flag. POSIX would take `preexec_fn`, but that is
    documented unsafe in a process with threads, and this one runs hundreds;
    prefixing `nice` costs one exec and is safe. Without a `nice` binary the
    command runs at normal priority rather than not at all.
    """
    if is_windows(system):
        return list(cmd), {"creationflags": BELOW_NORMAL_PRIORITY_CLASS}
    nice = shutil.which("nice")
    if not nice:
        return list(cmd), {}
    return [nice, "-n", str(BACKGROUND_NICE), *cmd], {}


def profile_lock_paths(profile_path: str | Path) -> list[Path]:
    """Chromium lock artifacts that can survive a crash. Same names everywhere:
    on POSIX the Singleton* entries are dangling symlinks (so exists() is False
    while is_symlink() is True -- callers must check both)."""
    root = Path(profile_path)
    return [
        root / "SingletonLock",
        root / "SingletonCookie",
        root / "SingletonSocket",
        root / "lockfile",
        root / "Default" / "lockfile",
    ]
