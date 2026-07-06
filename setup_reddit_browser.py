"""
One-time setup: log into Reddit using the bot's Chrome profile.
Optionally copies an existing Chrome profile's session first (if you're already logged in there).
Run once — the bot uses the saved session automatically on every start.

Usage:
    python setup_reddit_browser.py
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("playwright not installed. Run:  pip install playwright && playwright install chrome")
    sys.exit(1)

_SESSION_FILES = ["Cookies", "Preferences", "Login Data"]
_SESSION_DIRS  = ["Network", "Local Storage", "Session Storage", "IndexedDB"]

BOT_PROFILE = Path(__file__).parent / "chrome_profile"


def _chrome_user_data() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"


def _find_profiles() -> list[tuple[str, str, Path]]:
    user_data = _chrome_user_data()
    if not user_data.exists():
        return []
    profiles = []
    for entry in sorted(user_data.iterdir()):
        if entry.name != "Default" and not entry.name.startswith("Profile "):
            continue
        if not (entry / "Preferences").exists():
            continue
        try:
            prefs = json.loads((entry / "Preferences").read_text(encoding="utf-8", errors="ignore"))
            name = (prefs.get("profile") or {}).get("name") or entry.name
            accounts = prefs.get("account_info") or []
            email = accounts[0].get("email") if accounts else None
            label = f"{name} — {email}  ({entry.name})" if email else f"{name}  ({entry.name})"
        except Exception:
            label = entry.name
        profiles.append((label, entry.name, entry))
    return profiles


def _copy_profile(src_profile: Path) -> None:
    """Copy session files from an existing Chrome profile into the bot profile."""
    dst = BOT_PROFILE / "Default"
    dst.mkdir(parents=True, exist_ok=True)
    for name in _SESSION_FILES:
        f = src_profile / name
        if f.exists():
            try:
                shutil.copy2(f, dst / name)
                print(f"  copied {name}")
            except Exception as exc:
                print(f"  skipped {name}: {exc}")
    for name in _SESSION_DIRS:
        d = src_profile / name
        if d.exists():
            dd = dst / name
            try:
                if dd.exists():
                    shutil.rmtree(dd)
                shutil.copytree(d, dd)
                print(f"  copied {name}/")
            except Exception as exc:
                print(f"  skipped {name}/: {exc}")
    ls = _chrome_user_data() / "Local State"
    if ls.exists():
        try:
            shutil.copy2(ls, BOT_PROFILE / "Local State")
            print("  copied Local State")
        except Exception as exc:
            print(f"  skipped Local State: {exc}")


def _has_reddit_session() -> bool:
    """Check if the bot profile already has a Reddit session cookie."""
    cookies_db = BOT_PROFILE / "Default" / "Network" / "Cookies"
    if not cookies_db.exists():
        cookies_db = BOT_PROFILE / "Default" / "Cookies"
    if not cookies_db.exists():
        return False
    try:
        import sqlite3
        conn = sqlite3.connect(str(cookies_db))
        rows = conn.execute(
            "SELECT name FROM cookies WHERE host_key LIKE '%reddit%' AND name IN ('reddit_session','token_v2','session_tracker')"
        ).fetchall()
        conn.close()
        return len(rows) > 0
    except Exception:
        return False


def _launch_chrome_profile() -> subprocess.Popen | None:
    """Launch Chrome with the bot user-data-dir. Returns process on success."""
    chrome_candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    chrome_exe = next((p for p in chrome_candidates if p.exists()), None)
    if chrome_exe is None:
        print("[setup] Chrome not found. Install Google Chrome and try again.")
        return None

    args = [
        str(chrome_exe),
        f"--user-data-dir={BOT_PROFILE}",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    try:
        return subprocess.Popen(args)
    except Exception as exc:
        print(f"[setup] Failed to launch Chrome: {exc}")
        return None


def _open_for_login() -> bool:
    """Open the bot profile at Reddit login. Returns True if session found after."""
    print("\nOpening Chrome — log into Reddit, then come back here and press Enter.")

    try:
        with sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context(
                user_data_dir=str(BOT_PROFILE),
                channel="chrome",
                headless=False,
                args=["--no-first-run", "--no-default-browser-check"],
            )
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto("https://www.reddit.com/login", timeout=20_000, wait_until="domcontentloaded")
            input("\nPress Enter once you're logged into Reddit... ")
            cookies = ctx.cookies("https://www.reddit.com")
            session = next(
                (c for c in cookies if c["name"] in ("reddit_session", "token_v2", "session_tracker", "loid")),
                None,
            )
            # Persistent context writes updated cookies/state to the profile on close.
            return session is not None
    except Exception as exc:
        print(f"\nFailed to open Chrome profile via Playwright: {exc}")
        print("Tip: Close all existing Chrome windows and run this setup again.")
        # Fallback path: open Chrome directly so user can still log in manually.
        proc = _launch_chrome_profile()
        if proc is None:
            return False
        try:
            time.sleep(1.0)
            input("\nPress Enter once you're logged into Reddit in Chrome... ")
            return _has_reddit_session()
        finally:
            try:
                proc.terminate()
            except Exception:
                pass


def main() -> None:
    profiles = _find_profiles()

    print()
    if profiles:
        print("Existing Chrome profiles:\n")
        for i, (label, _, _) in enumerate(profiles, 1):
            print(f"  [{i}] {label}")
        print(f"  [{len(profiles) + 1}] Skip — just open the bot profile and log in manually")
        print()

        while True:
            raw = input(f"Pick a profile to copy session from (or {len(profiles) + 1} to skip): ").strip()
            try:
                choice = int(raw)
            except ValueError:
                continue
            if 1 <= choice <= len(profiles):
                label, dir_name, src = profiles[choice - 1]
                print(f"\nCopying session from {label}")
                print("Close Chrome completely first, then press Enter.")
                input()
                _copy_profile(src)
                break
            if choice == len(profiles) + 1:
                break

    # Always open for login — either the copy had a session (skips straight through)
    # or user needs to log in manually
    if _has_reddit_session():
        print("\nReddit session already detected in bot profile — skipping login step.")
        print("Done. Bot is ready.")
    else:
        logged_in = _open_for_login()
        if logged_in:
            print("\nReddit session confirmed. Bot is ready.")
        else:
            print("\nNo session detected. Make sure you're fully logged in before pressing Enter.")
            sys.exit(1)


if __name__ == "__main__":
    main()
