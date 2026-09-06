from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

from services import capacity, platform_support

try:
    from playwright.sync_api import sync_playwright as _sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROFILE = str(_REPO_ROOT / "chrome_profile")

# The Windows launcher starts the bot in a detached console window and the
# systemd unit sends stdout to the journal, so plain print() output is easy to
# lose -- these [browser] diagnostics (Chrome launch failures, session probe
# results, fetch errors) were invisible when diagnosing why .resumebuild
# scrapes were failing. Persist them alongside stdout so browser-layer failures
# are diagnosable after the fact.
_LOG_PATH = _REPO_ROOT / ".browser_service.log"


def _log(message: str) -> None:
    print(message)
    try:
        if _LOG_PATH.exists() and _LOG_PATH.stat().st_size > 5_000_000:
            _LOG_PATH.replace(_LOG_PATH.with_suffix(".log.1"))
        with _LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")
    except OSError:
        pass


# Chrome headless=new reports "HeadlessChrome" in UA; Reddit blocks it. Override via Playwright context.
# The claimed platform tracks the real host OS -- Chrome's Sec-CH-UA-Platform
# header comes from the OS and cannot be overridden here, so a mismatched UA
# would be a detectable inconsistency.
_UA = platform_support.chrome_user_agent()

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


def _read_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


_PLAYWRIGHT_FETCH_QUEUE_WAIT_SECONDS = _read_env_float(
    "PLAYWRIGHT_FETCH_QUEUE_WAIT_SECONDS", 3.0
)
_PLAYWRIGHT_DISPATCH_TIMEOUT_PAD_SECONDS = _read_env_float(
    "PLAYWRIGHT_DISPATCH_TIMEOUT_PAD_SECONDS", 5.0
)
class _PriorityDispatchGate:
    """Single-slot gate where a bulk-tier waiter always yields to a
    priority-tier waiter (interactive .resumebuild/.resumecoverbuild scrapes
    over reddit watcher polls), including one that arrives AFTER the bulk
    waiter already started waiting.

    A plain `threading.Semaphore` cannot do this: it wakes waiters strictly in
    the order they entered `.wait()` (a FIFO deque under the hood), so an
    already-queued bulk waiter beats a later, more important priority waiter
    to the slot every time — verified empirically (8/8 trials) before this
    class replaced the first cut of this mechanism, which only checked
    priority status "at the door" and so had exactly that hole.

    The fix: every waiter re-checks the priority condition on EVERY wakeup
    (`notify_all()` on release), not just at entry. A bulk waiter that wakes
    and sees a priority waiter still present steps aside instead of taking
    the free slot, however it was scheduled relative to the priority waiter.

    Deliberately no aging/starvation-promotion here, unlike
    services.priority_scheduler.PriorityWorkScheduler: a bulk caller that
    yields the gate doesn't block waiting for it -- it returns immediately
    (see _acquire_fetch_slot) and the reddit scrape pipeline falls through to
    RSS/curl_cffi for that attempt, then gets another shot at this gate on
    its next poll cycle. A queued PriorityWorkScheduler task has no such
    fallback; it must eventually run on that exact mechanism, which is why
    that one needs aging and this one doesn't. Track how often bulk callers
    actually yield via BrowserServiceHealth.dispatch_yield_count (/status)
    before adding aging here -- if reddit's Playwright layer is measurably
    starved in practice, that's the signal to revisit this.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._busy = False
        self._priority_waiting = 0
        self._waiting = 0

    def acquire_with_reason(self, timeout: float | None = None, priority: bool = False) -> tuple[bool, str]:
        """Returns (acquired, reason). reason is "acquired", "yielded" (a
        priority waiter is/was present and this is a bulk caller), or
        "timeout" — computed atomically so callers never have to re-check
        priority_waiting themselves and risk a second race on the way out."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            if priority:
                self._priority_waiting += 1
            try:
                while True:
                    if not self._busy and (priority or self._priority_waiting == 0):
                        self._busy = True
                        return True, "acquired"
                    if not priority and self._priority_waiting > 0:
                        return False, "yielded"
                    self._waiting += 1
                    try:
                        if deadline is None:
                            self._cv.wait()
                            continue
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            return False, "timeout"
                        self._cv.wait(timeout=remaining)
                    finally:
                        self._waiting -= 1
            finally:
                if priority:
                    self._priority_waiting -= 1

    def acquire(self, timeout: float | None = None, priority: bool = False) -> bool:
        acquired, _ = self.acquire_with_reason(timeout=timeout, priority=priority)
        return acquired

    def release(self) -> None:
        with self._cv:
            if not self._busy:
                raise RuntimeError("release() called without a matching acquire()")
            self._busy = False
            self._cv.notify_all()

    @property
    def priority_waiting(self) -> int:
        with self._cv:
            return self._priority_waiting

    @property
    def waiting(self) -> int:
        """Callers currently parked in the wait, priority and bulk alike.

        Symmetric with priority_waiting. Lets a caller (and the dispatch tests)
        observe that a waiter has actually parked, instead of sleeping long
        enough to assume it has.
        """
        with self._cv:
            return self._waiting


_fetch_dispatch_semaphore = _PriorityDispatchGate()
_ensure_ready_lock = threading.Lock()

_session_valid: bool = False
_last_session_check: float = 0.0
_SESSION_CHECK_INTERVAL = 1800
_RESYNC_INTERVAL = 600

_launched_profile: str | None = None
_primary_profile: str | None = None
_runtime_profile: str | None = None
_last_resync_attempt: float = 0.0

# Optional telemetry hook: set once at startup via set_health_hook(). Must be
# fast and non-raising; failures are swallowed so telemetry can never break
# the fetch path.
_health_hook = None


def set_health_hook(hook) -> None:
    """Register a callback(event: str, value) that receives browser telemetry
    events: dispatch_saturated, dispatch_timeout, session_probe, resync_attempt,
    active_profile."""
    global _health_hook
    _health_hook = hook


def _emit_health(event: str, value=None) -> None:
    hook = _health_hook
    if hook is None:
        return
    try:
        hook(event, value)
    except Exception:
        pass


def _acquire_fetch_slot(op_name: str, timeout_s: float, url: str, priority: bool = False) -> bool:
    acquired, reason = _fetch_dispatch_semaphore.acquire_with_reason(timeout=timeout_s, priority=priority)
    if acquired:
        return True
    if reason == "yielded":
        _log(f"[browser] {op_name} yielding dispatch slot to a priority fetch -- {url}")
        _emit_health("dispatch_yield", op_name)
    else:
        label = "PRIORITY " if priority else ""
        _log(f"[browser] {op_name} {label}dispatch saturated after {timeout_s:.1f}s -- {url}")
        _emit_health("dispatch_saturated", op_name)
    return False


# orphan-ok: no external Playwright user remains. The ZipRecruiter scraper was
# the one this was written for, and it was deliberately taken back out of the
# gate when curl_cffi replaced its headless launch (see the note above
# ZIPRECRUITER_HOME_URL in job_service) -- holding an app-wide browser slot for
# a plain HTTP fetch stalled real browser work for nothing.
#
# Kept rather than deleted for two reasons: this module is now the only thing
# in the tree that launches Playwright, so this is the documented entry point
# the next external launcher needs; and test_ziprecruiter_scraper's
# `test_no_browser_is_ever_launched` monkeypatches this name to a raising stub
# to prove the scraper never reaches for it, which needs the attribute to
# exist.
def acquire_browser_work_slot(op_name: str, timeout_s: float, detail: str = "", priority: bool = False) -> bool:
    """Gate entry for a Playwright user outside this module.

    All headless-browser work app-wide counts against this one dispatch slot,
    so a second Chromium process cannot spin up invisibly while a priority
    resume scrape is waiting -- bulk callers yield exactly like
    fetch_html/fetch_json bulk callers do. Pair every True return with
    release_browser_work_slot() in a finally.

    There is no caller today; see the note above.
    """
    return _acquire_fetch_slot(op_name, timeout_s, detail, priority=priority)


# orphan-ok: the release half of acquire_browser_work_slot, unused for the same
# reason and necessarily kept with it.
def release_browser_work_slot() -> None:
    _fetch_dispatch_semaphore.release()


def _clear_profile_locks(profile_path: str) -> None:
    """Remove stale Chromium singleton lock artifacts for this profile."""
    for lock_path in platform_support.profile_lock_paths(profile_path):
        try:
            # On POSIX the Singleton* entries are symlinks pointing at
            # host-pid-token targets that no longer resolve once Chrome dies, so
            # exists() is False for exactly the stale locks we need to remove.
            # is_symlink() is the check that catches them.
            if lock_path.exists() or lock_path.is_symlink():
                lock_path.unlink()
        except Exception:
            # If a file is truly in use, the retry below will still fail with a clear error.
            pass


_SESSION_FILES = ["Cookies", "Preferences", "Login Data"]
_SESSION_DIRS = ["Network", "Local Storage", "Session Storage", "IndexedDB"]


def _copy_session_artifacts(src_default: Path, dst_default: Path, label: str) -> None:
    """Copy session files/dirs, logging each artifact's outcome. Non-critical
    failures are logged but tolerated; the caller validates the cookie DB."""
    for name in _SESSION_FILES:
        f = src_default / name
        if not f.exists():
            continue
        try:
            shutil.copy2(f, dst_default / name)
            _log(f"[browser] {label}: {name} -> OK")
        except Exception as exc:
            _log(f"[browser] {label}: {name} -> FAILED ({exc})")
    for name in _SESSION_DIRS:
        d = src_default / name
        if not d.exists():
            continue
        target = dst_default / name
        try:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(d, target)
            _log(f"[browser] {label}: {name}/ -> OK")
        except Exception as exc:
            _log(f"[browser] {label}: {name}/ -> FAILED ({exc})")


def _cookie_db_present(profile_default: Path) -> bool:
    """True if a non-empty Chromium cookie DB exists under this profile's Default dir."""
    for db in (profile_default / "Network" / "Cookies", profile_default / "Cookies"):
        try:
            if db.exists() and db.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def _sync_profile_snapshot(source_profile: str, runtime_profile: str) -> bool:
    """Copy login/session artifacts into the dedicated runtime profile.
    Strict on the cookie DB: returns False when no non-empty cookie DB exists
    in the runtime profile after the copy, so callers never see a false success."""
    src = Path(source_profile)
    dst = Path(runtime_profile)
    dst_default = dst / "Default"
    dst_default.mkdir(parents=True, exist_ok=True)

    _copy_session_artifacts(src / "Default", dst_default, "sync")

    local_state = src / "Local State"
    if local_state.exists():
        try:
            shutil.copy2(local_state, dst / "Local State")
        except Exception as exc:
            _log(f"[browser] sync: Local State -> FAILED ({exc})")

    if not _cookie_db_present(dst_default):
        _log(f"[browser] sync FAILED: no cookie DB in runtime profile after copy from {source_profile}")
        return False
    return True


_CHROME_EPOCH_OFFSET = 11644473600


def _find_chrome_profile_with_reddit_session() -> Path | None:
    """Scan the user's Chrome profiles for one with a valid reddit_session cookie.

    Checks every browser user-data dir this platform knows about (Chrome and
    Chromium, plus Flatpak locations on Linux), so a session logged in under
    whichever browser the user actually runs is still harvestable.
    """
    now = time.time()
    for user_data in platform_support.chrome_user_data_dirs():
        try:
            entries = sorted(user_data.iterdir())
        except OSError:
            continue
        for entry in entries:
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
    """Copy Reddit session from a user's real Chrome profile into the bot profile.
    Strict on the cookie DB: returns False (with an explicit reason logged) when
    no non-empty cookie DB exists in the target after the copy."""
    source = _find_chrome_profile_with_reddit_session()
    if source is None:
        _log("[browser] harvest: no Chrome profile with a valid reddit_session found")
        return False

    _log(f"[browser] Found valid Reddit session in Chrome {source.name}")
    dst = Path(target_profile) / "Default"
    dst.mkdir(parents=True, exist_ok=True)

    _copy_session_artifacts(source, dst, "harvest")

    local_state = source.parent / "Local State"
    if local_state.exists():
        try:
            shutil.copy2(local_state, Path(target_profile) / "Local State")
        except Exception as exc:
            _log(f"[browser] harvest: Local State -> FAILED ({exc})")

    if not _cookie_db_present(dst):
        _log(f"[browser] harvest FAILED: no cookie DB present after copy from Chrome {source.name}")
        return False
    _log(f"[browser] Copied Reddit session from Chrome {source.name} -> bot profile")
    return True


def _launch_context(path: str):
    # executable_path rather than channel="chrome": the channel lookup only
    # knows about branded Chrome install locations, so it fails on hosts where
    # only Chromium is present. _do_start has already resolved a real binary
    # through the platform shim, so point Playwright straight at it.
    return _pw.chromium.launch_persistent_context(
        user_data_dir=path,
        executable_path=_find_chrome(),
        headless=True,
        user_agent=_UA,
        service_workers="block",
        args=["--no-first-run", "--no-default-browser-check", *platform_support.chrome_sandbox_args()],
    )


def _find_chrome() -> str | None:
    return platform_support.find_chrome()


def _kill_chrome_using_profile(profile_path: str) -> None:
    """Kill Chrome processes using this profile, then clear lock artifacts.
    Logs each PID with its owning session so cross-session lock owners (Windows
    Task Scheduler / Session 0, or a desktop login on Linux) are attributable
    instead of silently unkillable."""
    profile_dir = Path(profile_path).name
    killed = platform_support.kill_processes_using_profile(profile_dir, log=_log)
    if killed:
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
        _log(f"[browser] reddit_session expired {days_ago:.0f}d ago -- run setup_reddit_browser.py")
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
            _log(f"[browser] session probe got {resp.status if resp else 'no resp'} -- session invalid")
            _session_valid = False
            _emit_health("session_probe", False)
            return False
        data = json.loads(resp.text())
        logged_in = isinstance(data, dict) and bool((data.get("data") or {}).get("name"))
        if not logged_in:
            _log("[browser] session probe: Reddit does not recognize the session -- session invalid")
        _session_valid = logged_in
        _emit_health("session_probe", logged_in)
    except Exception as exc:
        # Transient failure (timeout, navigation error): keep the previous
        # verdict rather than triggering a restart loop on network blips.
        _log(f"[browser] session probe inconclusive: {exc}")
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
    primary = _primary_profile
    runtime = _runtime_profile
    if not primary or not runtime:
        _emit_health("resync_attempt", False)
        return False

    # Our own running Chrome holds an exclusive lock on the runtime profile's
    # Cookies file, so harvest/sync must happen only after the context is stopped.
    _do_stop()
    time.sleep(2)
    _clear_profile_locks(runtime)

    if _stored_session_expiry_ok(primary):
        _log("[browser] Stored reddit_session still locally valid; relaunching to re-probe server-side")
    elif _harvest_chrome_session(primary):
        _log("[browser] Harvested fresh cookies from user Chrome")
    else:
        _log("[browser] No harvestable Chrome session; relaunching with existing cookies")
    ok = _do_start(primary, runtime, skip_harvest=True)
    _emit_health("resync_attempt", ok)
    return ok


def _do_start(primary_path: str, runtime_path: str, *, skip_harvest: bool = False) -> bool:
    """Launch Chrome on the runtime profile. The primary profile is source-only:
    cookies are harvested into it (from the user's real Chrome) and synced into
    the runtime profile, but Chrome is never launched against the primary."""
    global _pw, _context, _launched_profile, _primary_profile, _runtime_profile, _last_resync_attempt

    chrome_exe = _find_chrome()
    if chrome_exe is None:
        _log("[browser] Chrome not found -- skipping Chrome layer")
        return False

    _kill_chrome_using_profile(runtime_path)

    if not skip_harvest:
        if _stored_session_expiry_ok(primary_path):
            _log("[browser] Stored reddit_session still valid -- skipping harvest")
        elif not _harvest_chrome_session(primary_path):
            _log("[browser] harvest failed -- continuing with existing profile cookies (if any)")

    if not _sync_profile_snapshot(primary_path, runtime_path):
        _log("[browser] runtime profile sync incomplete -- launching anyway; session probe will report state")

    _primary_profile = primary_path
    _runtime_profile = runtime_path
    try:
        _pw = _sync_playwright().start()
        try:
            _context = _launch_context(runtime_path)
        except Exception as exc:
            # On Windows, stale singleton lock files can remain after crashes.
            if "ProcessSingleton" in str(exc):
                _log("[browser] Runtime profile lock detected, attempting lock cleanup and one retry")
                _kill_chrome_using_profile(runtime_path)
                time.sleep(0.5)
                _clear_profile_locks(runtime_path)
                _context = _launch_context(runtime_path)
            else:
                raise

        _launched_profile = runtime_path
        _emit_health("active_profile", runtime_path)
        _log(f"[browser] Launched runtime profile: {runtime_path} (source: {primary_path})")
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
            _log(f"[browser] Chrome ready -- Reddit session valid (profile: {runtime_path})")
        else:
            _log(f"[browser] Chrome ready -- no Reddit session (profile: {runtime_path})")
            _log(f"[browser]   Gallery/NSFW scraping degraded. Run: python setup_reddit_browser.py")
        _last_resync_attempt = time.monotonic()
        return True
    except Exception as exc:
        _log(f"[browser] Persistent context launch failed: {exc}")
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
            _log(f"[browser] {resp.status if resp else 'no resp'} -- {url}")
            return None
        return json.loads(resp.text())
    except Exception as exc:
        _log(f"[browser] fetch_json failed: {exc}")
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
            _log(f"[browser] {resp.status if resp else 'no resp'} -- {url}")
            return None
        return page.content()
    except Exception as exc:
        _log(f"[browser] fetch_html failed: {exc}")
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
        _log("[browser] playwright not installed -- skipping Chrome layer")
        return False

    primary = profile_path or os.getenv("REDDIT_CHROME_PROFILE") or _DEFAULT_PROFILE
    runtime = os.getenv("REDDIT_CHROME_RUNTIME_PROFILE") or f"{primary}_runtime"
    _last_profile_path = primary
    _last_start_attempt = time.monotonic()
    Path(primary).mkdir(parents=True, exist_ok=True)
    Path(runtime).mkdir(parents=True, exist_ok=True)

    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="playwright")
    try:
        # Chromium cold-start is CPU- and disk-bound; on a small host 90s is
        # marginal and a timeout here disables the whole browser layer.
        return _executor.submit(_do_start, primary, runtime).result(timeout=capacity.timeout(90))
    except Exception as exc:
        _log(f"[browser] start failed: {exc}")
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


def fetch_json(url: str, timeout_ms: int = 15_000, priority: bool = False) -> dict | list | None:
    """Navigate to a JSON URL and return parsed response. Returns None on any failure.

    priority=True marks an interactive caller: it waits the full dispatch
    window for the slot and bulk callers yield to it (see _acquire_fetch_slot).
    """
    if _executor is None or _context is None:
        return None

    dispatch_timeout = (timeout_ms / 1000) + _PLAYWRIGHT_DISPATCH_TIMEOUT_PAD_SECONDS
    slot_wait = dispatch_timeout if priority else min(_PLAYWRIGHT_FETCH_QUEUE_WAIT_SECONDS, dispatch_timeout)
    if not _acquire_fetch_slot("fetch_json", slot_wait, url, priority=priority):
        return None

    try:
        return _executor.submit(_do_fetch_json, url, timeout_ms).result(timeout=dispatch_timeout)
    except concurrent.futures.TimeoutError:
        # TimeoutError string is often empty; include explicit timing for diagnostics.
        _log(f"[browser] fetch_json dispatch timed out after {dispatch_timeout:.1f}s -- {url}")
        _emit_health("dispatch_timeout", "fetch_json")
        return None
    except Exception as exc:
        _log(f"[browser] fetch_json dispatch failed ({type(exc).__name__}): {exc!r}")
        return None
    finally:
        _fetch_dispatch_semaphore.release()


def fetch_html(url: str, timeout_ms: int = 20_000, priority: bool = False) -> str | None:
    """Navigate to a page in the persistent Chrome context and return its rendered HTML.

    Useful for sites (e.g. LinkedIn) that block plain `requests` calls via
    TLS/JA3 fingerprinting or bot detection but allow a real browser through.
    Returns None on any failure.

    priority=True marks an interactive caller: it waits the full dispatch
    window for the slot and bulk callers yield to it (see _acquire_fetch_slot).
    """
    if _executor is None or _context is None:
        return None

    dispatch_timeout = (timeout_ms / 1000) + _PLAYWRIGHT_DISPATCH_TIMEOUT_PAD_SECONDS
    slot_wait = dispatch_timeout if priority else min(_PLAYWRIGHT_FETCH_QUEUE_WAIT_SECONDS, dispatch_timeout)
    if not _acquire_fetch_slot("fetch_html", slot_wait, url, priority=priority):
        return None

    try:
        return _executor.submit(_do_fetch_html, url, timeout_ms).result(timeout=dispatch_timeout)
    except concurrent.futures.TimeoutError:
        _log(f"[browser] fetch_html dispatch timed out after {dispatch_timeout:.1f}s -- {url}")
        _emit_health("dispatch_timeout", "fetch_html")
        return None
    except Exception as exc:
        _log(f"[browser] fetch_html dispatch failed: {exc}")
        return None
    finally:
        _fetch_dispatch_semaphore.release()


def is_ready() -> bool:
    return _executor is not None and _context is not None


def session_valid() -> bool:
    """Whether the last session probe found the Chrome profile still logged in.

    A plain read of the flag `_do_check_session` maintains, so it costs
    nothing and is safe to call while rendering an embed. There used to be a
    `check_session()` beside this that forced a probe through the executor and
    blocked up to ten seconds; nothing ever called it, and ensure_ready already
    reprobes on _SESSION_CHECK_INTERVAL, so a caller asking "is the session
    valid" has a fresh answer without paying for one.
    """
    return _session_valid


def ensure_ready() -> bool:
    """If Chrome isn't running or has crashed, (re)start it. Thread-safe with cooldown."""
    if not _ensure_ready_lock.acquire(blocking=False):
        # Another caller is already checking/recovering browser state.
        return is_ready()
    try:
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
            _log("[browser] Chrome not ready -- attempting (re)start")
            stop()
            return start(_last_profile_path)
    finally:
        _ensure_ready_lock.release()
