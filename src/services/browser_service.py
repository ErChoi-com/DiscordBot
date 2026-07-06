from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright as _sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROFILE = str(_REPO_ROOT / "chrome_profile")
# Chrome headless=new reports "HeadlessChrome" in UA; Reddit blocks it. Override via Playwright context.
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)

# Single-worker executor so ALL Playwright calls run in one dedicated background thread.
# This keeps asyncio state out of the main thread, avoiding conflicts with discord.py's
# asyncio.run() which requires no running event loop in the main thread.
_executor: concurrent.futures.ThreadPoolExecutor | None = None
_pw = None
_context = None

_restart_lock = threading.Lock()
_last_profile_path: str | None = None
_last_start_attempt: float = 0.0
_START_COOLDOWN = 120

_session_valid: bool = False
_last_session_check: float = 0.0
_SESSION_CHECK_INTERVAL = 1800
_RESYNC_INTERVAL = 600

_launched_profile: str | None = None
_primary_profile: str | None = None
_last_resync_attempt: float = 0.0


def _clear_profile_locks(profile_path: str) -> None:
    """Remove stale Chromium singleton lock artifacts for this profile."""
    lock_candidates = [
        Path(profile_path) / "SingletonLock",
        Path(profile_path) / "SingletonCookie",
        Path(profile_path) / "SingletonSocket",
        Path(profile_path) / "lockfile",
        Path(profile_path) / "Default" / "lockfile",
    ]
    for lock_path in lock_candidates:
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            # If a file is truly in use, the retry below will still fail with a clear error.
            pass


def _sync_profile_snapshot(source_profile: str, runtime_profile: str) -> bool:
    """Best-effort copy of login/session artifacts into a dedicated runtime profile.
    Returns True if the critical Cookies file was copied successfully."""
    src = Path(source_profile)
    dst = Path(runtime_profile)
    src_default = src / "Default"
    dst_default = dst / "Default"
    dst_default.mkdir(parents=True, exist_ok=True)

    cookies_copied = False
    files = ["Cookies", "Preferences", "Login Data"]
    dirs = ["Network", "Local Storage", "Session Storage", "IndexedDB"]

    for name in files:
        f = src_default / name
        if not f.exists():
            continue
        try:
            shutil.copy2(f, dst_default / name)
            if name == "Cookies":
                cookies_copied = True
        except Exception:
            pass

    for name in dirs:
        d = src_default / name
        if not d.exists():
            continue
        target = dst_default / name
        try:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(d, target)
            if name == "Network":
                cookies_copied = True
        except Exception:
            pass

    local_state = src / "Local State"
    if local_state.exists():
        try:
            shutil.copy2(local_state, dst / "Local State")
        except Exception:
            pass

    return cookies_copied


_CHROME_EPOCH_OFFSET = 11644473600


def _find_chrome_profile_with_reddit_session() -> Path | None:
    """Scan the user's Chrome profiles for one with a valid reddit_session cookie."""
    user_data = Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"
    if not user_data.exists():
        return None

    now = time.time()
    for entry in sorted(user_data.iterdir()):
        if entry.name != "Default" and not entry.name.startswith("Profile "):
            continue
        cookies_db = entry / "Network" / "Cookies"
        if not cookies_db.exists():
            cookies_db = entry / "Cookies"
        if not cookies_db.exists():
            continue
        try:
            conn = sqlite3.connect(f"file:{cookies_db}?mode=ro", uri=True)
            rows = conn.execute(
                "SELECT expires_utc FROM cookies "
                "WHERE host_key LIKE '%reddit.com%' AND name = 'reddit_session'"
            ).fetchall()
            conn.close()
            for (expires_utc,) in rows:
                expires_unix = (expires_utc / 1_000_000) - _CHROME_EPOCH_OFFSET if expires_utc > 0 else float("inf")
                if expires_unix > now:
                    return entry
        except Exception:
            continue
    return None


def _stored_session_expiry_ok(profile_path: str) -> bool:
    """True if the bot's own profile already holds a locally-unexpired
    reddit_session cookie. Must be called while no Chrome is using the profile,
    otherwise the cookie DB may be locked (a locked DB reads as "no session")."""
    base = Path(profile_path) / "Default"
    now = time.time()
    for cookies_db in (base / "Network" / "Cookies", base / "Cookies"):
        if not cookies_db.exists():
            continue
        try:
            conn = sqlite3.connect(f"file:{cookies_db}?mode=ro", uri=True)
            rows = conn.execute(
                "SELECT expires_utc FROM cookies "
                "WHERE host_key LIKE '%reddit.com%' AND name = 'reddit_session'"
            ).fetchall()
            conn.close()
        except Exception:
            continue
        for (expires_utc,) in rows:
            expires_unix = (expires_utc / 1_000_000) - _CHROME_EPOCH_OFFSET if expires_utc > 0 else float("inf")
            if expires_unix > now:
                return True
    return False


def _harvest_chrome_session(target_profile: str) -> bool:
    """Copy Reddit session from a user's real Chrome profile into the bot profile."""
    source = _find_chrome_profile_with_reddit_session()
    if source is None:
        return False

    print(f"[browser] Found valid Reddit session in Chrome {source.name}")
    dst = Path(target_profile) / "Default"
    dst.mkdir(parents=True, exist_ok=True)

    files = ["Cookies", "Preferences", "Login Data"]
    dirs = ["Network", "Local Storage", "Session Storage", "IndexedDB"]
    copied_cookies = False

    for name in files:
        f = source / name
        if not f.exists():
            continue
        try:
            shutil.copy2(f, dst / name)
            if name == "Cookies":
                copied_cookies = True
        except Exception:
            pass

    for name in dirs:
        d = source / name
        if not d.exists():
            continue
        target = dst / name
        try:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(d, target)
            if name == "Network":
                copied_cookies = True
        except Exception:
            pass

    user_data = source.parent
    local_state = user_data / "Local State"
    if local_state.exists():
        try:
            shutil.copy2(local_state, Path(target_profile) / "Local State")
        except Exception:
            pass

    if copied_cookies:
        print(f"[browser] Copied Reddit session from Chrome {source.name} -> bot profile")
    return copied_cookies


def _launch_context(path: str):
    return _pw.chromium.launch_persistent_context(
        user_data_dir=path,
        channel="chrome",
        headless=True,
        user_agent=_UA,
        service_workers="block",
        args=["--no-first-run", "--no-default-browser-check"],
    )


def _find_chrome() -> str | None:
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _kill_chrome_using_profile(profile_path: str) -> None:
    """Kill Chrome processes using this profile, then clear lock artifacts."""
    profile_dir = Path(profile_path).name
    pids: list[int] = []

    # PowerShell/CIM -- reliable on Windows 11 where WMIC is deprecated
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"name='chrome.exe'\" | "
             f"Where-Object {{ $_.CommandLine -like '*{profile_dir}*' }} | "
             "Select-Object -ExpandProperty ProcessId"],
            stderr=subprocess.DEVNULL, text=True, timeout=10
        )
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
    except Exception:
        # Fallback to WMIC for older Windows
        try:
            out = subprocess.check_output(
                ["wmic", "process", "where",
                 f"name='chrome.exe' and commandline like '%{profile_dir}%'",
                 "get", "processid", "/format:csv"],
                stderr=subprocess.DEVNULL, text=True, timeout=10
            )
            for line in out.splitlines():
                line = line.strip()
                if not line or "ProcessId" in line:
                    continue
                parts = line.split(",")
                try:
                    pids.append(int(parts[-1]))
                except ValueError:
                    pass
        except Exception:
            pass

    for pid in pids:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if pids:
        time.sleep(1.5)
    _clear_profile_locks(profile_path)


# ---------------------------------------------------------------------------
# All functions below run inside the single-worker executor thread
# ---------------------------------------------------------------------------

def _do_check_session() -> bool:
    """Verify the Reddit session in the Playwright context. Runs in executor.

    A locally-unexpired reddit_session cookie can still be rejected server-side
    (revoked/rotated), so after the cheap expiry check we probe an authenticated
    endpoint through the browser."""
    global _session_valid, _last_session_check
    _last_session_check = time.monotonic()
    if _context is None:
        _session_valid = False
        return False
    try:
        cookies = _context.cookies("https://www.reddit.com")
    except Exception:
        _session_valid = False
        return False

    now = time.time()
    session_cookie = next((c for c in cookies if c["name"] == "reddit_session"), None)
    if session_cookie is None:
        _session_valid = False
        return False
    expires = session_cookie.get("expires", -1)
    if expires > 0 and expires < now:
        days_ago = (now - expires) / 86400
        print(f"[browser] reddit_session expired {days_ago:.0f}d ago -- run setup_reddit_browser.py")
        _session_valid = False
        return False

    page = None
    try:
        page = _context.new_page()
        resp = page.goto(
            "https://www.reddit.com/api/me.json?raw_json=1",
            timeout=8_000, wait_until="domcontentloaded",
        )
        if resp is None or resp.status != 200:
            print(f"[browser] session probe got {resp.status if resp else 'no resp'} -- session invalid")
            _session_valid = False
            return False
        data = json.loads(resp.text())
        logged_in = isinstance(data, dict) and bool((data.get("data") or {}).get("name"))
        if not logged_in:
            print("[browser] session probe: Reddit does not recognize the session -- session invalid")
        _session_valid = logged_in
    except Exception as exc:
        # Transient failure (timeout, navigation error): keep the previous
        # verdict rather than triggering a restart loop on network blips.
        print(f"[browser] session probe inconclusive: {exc}")
    finally:
        if page:
            try:
                page.close()
            except Exception:
                pass
    return _session_valid


def _do_resync_session() -> bool:
    """Try to restore a valid Reddit session by copying fresh cookies from the
    user's Chrome profile and restarting the Playwright context so it loads them."""
    global _last_resync_attempt, _last_start_attempt
    _last_resync_attempt = time.monotonic()
    _last_start_attempt = time.monotonic()
    path = _primary_profile
    if not path:
        return False

    # Our own running Chrome holds an exclusive lock on the profile's Cookies
    # file, so harvesting must happen only after the context is stopped.
    _do_stop()
    time.sleep(2)
    _clear_profile_locks(path)

    if _stored_session_expiry_ok(path):
        print("[browser] Stored reddit_session still locally valid; relaunching to re-probe server-side")
    elif _harvest_chrome_session(path):
        print("[browser] Harvested fresh cookies from user Chrome")
    else:
        print("[browser] No harvestable Chrome session; relaunching with existing cookies")
    return _do_start(path, skip_harvest=True)


def _do_start(path: str, *, skip_harvest: bool = False) -> bool:
    global _pw, _context, _launched_profile, _primary_profile, _last_resync_attempt

    chrome_exe = _find_chrome()
    if chrome_exe is None:
        print("[browser] Chrome not found -- skipping Chrome layer")
        return False

    _kill_chrome_using_profile(path)

    if not skip_harvest:
        if _stored_session_expiry_ok(path):
            print("[browser] Stored reddit_session still valid -- skipping harvest")
        else:
            _harvest_chrome_session(path)

    _primary_profile = path
    try:
        _pw = _sync_playwright().start()
        launched_profile = path
        try:
            _context = _launch_context(path)
        except Exception as exc:
            # On Windows, stale singleton lock files can remain after crashes.
            if "ProcessSingleton" in str(exc):
                print("[browser] Profile lock detected, attempting lock cleanup and one retry")
                _kill_chrome_using_profile(path)
                time.sleep(0.5)
                _clear_profile_locks(path)
                try:
                    _context = _launch_context(path)
                except Exception as retry_exc:
                    if "ProcessSingleton" not in str(retry_exc):
                        raise
                    runtime_path = os.getenv("REDDIT_CHROME_RUNTIME_PROFILE") or f"{path}_runtime"
                    print(f"[browser] Primary profile still in use, falling back to runtime profile: {runtime_path}")
                    _kill_chrome_using_profile(runtime_path)
                    if not _stored_session_expiry_ok(runtime_path):
                        _harvest_chrome_session(runtime_path)
                    _context = _launch_context(runtime_path)
                    launched_profile = runtime_path
            else:
                raise

        _launched_profile = launched_profile
        _context.add_cookies([{
            "name": "over18",
            "value": "1",
            "domain": ".reddit.com",
            "path": "/",
        }])
        page = _context.new_page()
        try:
            page.goto("https://www.reddit.com", timeout=30_000, wait_until="networkidle")
        except Exception:
            pass
        finally:
            try:
                page.close()
            except Exception:
                pass

        if _do_check_session():
            print(f"[browser] Chrome ready -- Reddit session valid (profile: {launched_profile})")
        else:
            print(f"[browser] Chrome ready -- no Reddit session (profile: {launched_profile})")
            print(f"[browser]   Gallery/NSFW scraping degraded. Run: python setup_reddit_browser.py")
        _last_resync_attempt = time.monotonic()
        return True
    except Exception as exc:
        print(f"[browser] Persistent context launch failed: {exc}")
        if _pw:
            try:
                _pw.stop()
            except Exception:
                pass
        _pw = None
        _context = None
        return False


def _do_stop() -> None:
    global _context, _pw
    profile = _launched_profile
    if _context:
        try:
            _context.close()
        except Exception:
            pass
        _context = None
    if _pw:
        try:
            _pw.stop()
        except Exception:
            pass
        _pw = None
    if profile:
        _kill_chrome_using_profile(profile)


def _do_fetch_json(url: str, timeout_ms: int) -> dict | list | None:
    if _context is None:
        return None
    page = None
    try:
        page = _context.new_page()
        resp = page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
        if resp is None or resp.status != 200:
            print(f"[browser] {resp.status if resp else 'no resp'} -- {url}")
            return None
        return json.loads(resp.text())
    except Exception as exc:
        print(f"[browser] fetch_json failed: {exc}")
        return None
    finally:
        if page:
            try:
                page.close()
            except Exception:
                pass


def _do_fetch_html(url: str, timeout_ms: int) -> str | None:
    if _context is None:
        return None
    page = None
    try:
        page = _context.new_page()
        resp = page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
        if resp is None or resp.status >= 400:
            print(f"[browser] {resp.status if resp else 'no resp'} -- {url}")
            return None
        return page.content()
    except Exception as exc:
        print(f"[browser] fetch_html failed: {exc}")
        return None
    finally:
        if page:
            try:
                page.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Public API -- dispatches to the executor thread, safe to call from anywhere
# ---------------------------------------------------------------------------

def start(profile_path: str | None = None) -> bool:
    """Launch Chrome and prime Reddit cookies. Call once at bot startup."""
    global _executor, _last_profile_path, _last_start_attempt
    if not _PLAYWRIGHT_AVAILABLE:
        print("[browser] playwright not installed -- skipping Chrome layer")
        return False

    path = profile_path or os.getenv("REDDIT_CHROME_PROFILE") or _DEFAULT_PROFILE
    _last_profile_path = path
    _last_start_attempt = time.monotonic()
    Path(path).mkdir(parents=True, exist_ok=True)

    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="playwright")
    try:
        return _executor.submit(_do_start, path).result(timeout=90)
    except Exception as exc:
        print(f"[browser] start failed: {exc}")
        return False


def stop() -> None:
    """Close the browser. Call at bot shutdown."""
    global _executor
    if _executor is not None:
        try:
            _executor.submit(_do_stop).result(timeout=15)
        except Exception:
            pass
        _executor.shutdown(wait=False)
        _executor = None


def fetch_json(url: str, timeout_ms: int = 15_000) -> dict | list | None:
    """Navigate to a JSON URL and return parsed response. Returns None on any failure."""
    if _executor is None or _context is None:
        return None
    try:
        return _executor.submit(_do_fetch_json, url, timeout_ms).result(timeout=(timeout_ms / 1000) + 5)
    except Exception as exc:
        print(f"[browser] fetch_json dispatch failed: {exc}")
        return None


def fetch_html(url: str, timeout_ms: int = 20_000) -> str | None:
    """Navigate to a page in the persistent Chrome context and return its rendered HTML.

    Useful for sites (e.g. LinkedIn) that block plain `requests` calls via
    TLS/JA3 fingerprinting or bot detection but allow a real browser through.
    Returns None on any failure.
    """
    if _executor is None or _context is None:
        return None
    try:
        return _executor.submit(_do_fetch_html, url, timeout_ms).result(timeout=(timeout_ms / 1000) + 5)
    except Exception as exc:
        print(f"[browser] fetch_html dispatch failed: {exc}")
        return None


def is_ready() -> bool:
    return _executor is not None and _context is not None


def check_session() -> bool:
    """Check Reddit session validity. Thread-safe."""
    if _executor is None or _context is None:
        return False
    try:
        return _executor.submit(_do_check_session).result(timeout=10)
    except Exception:
        return False


def session_valid() -> bool:
    return _session_valid


def ensure_ready() -> bool:
    """If Chrome isn't running or has crashed, (re)start it. Thread-safe with cooldown."""
    if is_ready():
        if time.monotonic() - _last_session_check > _SESSION_CHECK_INTERVAL:
            try:
                _executor.submit(_do_check_session).result(timeout=10)
            except Exception:
                pass
        if not _session_valid and time.monotonic() - _last_resync_attempt > _RESYNC_INTERVAL:
            try:
                _executor.submit(_do_resync_session).result(timeout=120)
            except Exception:
                pass
        return is_ready()
    now = time.monotonic()
    if now - _last_start_attempt < _START_COOLDOWN:
        return False
    with _restart_lock:
        if is_ready():
            return True
        if time.monotonic() - _last_start_attempt < _START_COOLDOWN:
            return False
        print("[browser] Chrome not ready -- attempting (re)start")
        stop()
        return start(_last_profile_path)
