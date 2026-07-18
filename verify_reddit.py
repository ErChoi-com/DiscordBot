"""
Reddit bypass verification script.

Run from the project root:
    python verify_reddit.py

What it checks:
  1. Which bypass layers are active (SOCKS5 proxy / curl_cffi)
  2. Whether Reddit actually responds with posts
  3. Proxy connectivity independently
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# ── Bootstrap: load .env into os.environ (same logic as load_config) ─────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _raw in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _raw.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.reddit_service import (
    _CURL_CFFI_AVAILABLE,
    parse_proxy_pool,
    pick_proxy,
    scrape_subreddit_media,
)

SUBREDDIT = "pics"
DIVIDER = "-" * 60


def section(title: str) -> None:
    print(f"\n{DIVIDER}\n{title}\n{DIVIDER}")


def check_layer_status() -> tuple[bool, bool]:
    """Returns (proxy_ready, cffi_ready)."""
    proxy_pool = parse_proxy_pool(os.getenv("REDDIT_PROXIES", ""))
    proxy_ready = bool(proxy_pool)
    return proxy_ready, _CURL_CFFI_AVAILABLE


def test_proxy_connectivity(proxy_url: str) -> tuple[bool, int | None, str]:
    """Hit a neutral endpoint through the proxy to confirm it routes."""
    try:
        import requests
        resp = requests.get(
            "https://httpbin.org/ip",
            proxies={"http": proxy_url, "https": proxy_url},
            timeout=15,
        )
        if resp.status_code == 200:
            ip = resp.json().get("origin", "unknown")
            return True, resp.status_code, f"exit IP = {ip}"
        return False, resp.status_code, resp.text[:120]
    except Exception as exc:
        return False, None, str(exc)


def test_reddit(proxy_url: str | None = None) -> tuple[bool, str]:
    try:
        results = scrape_subreddit_media(
            SUBREDDIT,
            sort="new",
            limit=3,
            proxy_url=proxy_url,
        )
        if results:
            titles = [r["title"][:60] for r in results]
            return True, f"{len(results)} posts\n  " + "\n  ".join(titles)
        return False, "0 posts returned (possible auth/filter issue)"
    except Exception as exc:
        return False, str(exc)


def main() -> None:
    section("Layer status")
    proxy_ready, cffi_ready = check_layer_status()

    proxy_pool = parse_proxy_pool(os.getenv("REDDIT_PROXIES", ""))
    proxy_status = f"READY — {len(proxy_pool)} server(s): {proxy_pool}" if proxy_ready else "not configured (REDDIT_PROXIES is empty)"
    cffi_status = "READY" if cffi_ready else "not installed"

    print(f"  Proxy pool  : {proxy_status}")
    print(f"  curl_cffi   : {cffi_status}")

    active = []
    if proxy_ready and cffi_ready:
        active.append("curl_cffi + proxy")
    elif proxy_ready:
        active.append("requests + proxy")
    if not active:
        active.append("NONE - will be blocked")
    print(f"\n  Active path : {' -> '.join(active)}")

    # ── Proxy connectivity test ───────────────────────────────────────────────
    if proxy_ready:
        section("Proxy connectivity test")
        proxy_url = pick_proxy(proxy_pool, 0)
        print(f"  Testing proxy: {proxy_url}")
        ok, code, detail = test_proxy_connectivity(proxy_url)
        print(f"  Result: {'OK' if ok else 'FAIL'}  {detail}")

    # ── Reddit scrape test ────────────────────────────────────────────────────
    section("Reddit scrape test  (r/pics, 3 posts)")
    proxy_url = pick_proxy(proxy_pool, 0) if proxy_ready else None
    ok, detail = test_reddit(proxy_url=proxy_url)
    status = "SUCCESS" if ok else "FAIL"
    print(f"  {status}: {detail}".encode("ascii", "replace").decode())

    # ── Summary ───────────────────────────────────────────────────────────────
    section("Summary")
    if ok:
        print("  Reddit scraper is working.")
        if proxy_ready:
            print("  Using SOCKS5 proxy.")
        print("\n  No further action needed — start the bot normally.")
    else:
        print("  Reddit scraper is BLOCKED.")
        print()
        if not proxy_ready:
            print("  To fix - SOCKS5 proxy:")
            print("    1. Get SOCKS5 proxy credentials from your provider")
            print("    2. In .env set:")
            print("       REDDIT_PROXIES=socks5://USER:PASS@HOST:PORT")
            print("    3. Re-run: python verify_reddit.py")
        print()
        print("  Also check the Playwright/Chrome session layer (browser_service):")
        print("    Run: python setup_reddit_browser.py")


if __name__ == "__main__":
    main()
