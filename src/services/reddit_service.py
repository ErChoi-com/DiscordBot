from __future__ import annotations

import asyncio
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from services.priority_scheduler import BACKGROUND, PriorityWorkScheduler
from services.rss_service import extract_link_from_html, fetch_and_parse_atom
from services import scheduler_labels


def parse_proxy_pool(raw: str | None) -> list[str]:
    """Split a newline/comma-separated proxy string into a clean list."""
    if not raw:
        return []
    return [p.strip() for p in raw.replace(",", "\n").splitlines() if p.strip()]


def pick_proxy(proxy_pool: list[str], cursor: int) -> str | None:
    if not proxy_pool:
        return None
    return proxy_pool[cursor % len(proxy_pool)]

try:
    from curl_cffi import requests as cffi_requests
    _CURL_CFFI_AVAILABLE = True
except ImportError:
    import requests as cffi_requests  # type: ignore[no-redef]
    _CURL_CFFI_AVAILABLE = False

import requests


REDDIT_FINGERPRINTS: tuple[str, ...] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
)

# curl_cffi browser impersonation targets (rotated alongside UA)
_CFFI_IMPERSONATIONS: tuple[str, ...] = (
    "chrome120",
    "chrome131",
    "firefox133",
    "safari18_0",
)

SEEN_URL_LIMIT = 2000


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# --- Playwright call gate (upstream of browser_service's own dispatch gate) ---
# Multiple subreddits are polled concurrently via asyncio.gather; each poll runs
# scrape_subreddit_media in its own to_thread worker. Without this gate every
# worker stampedes browser_service.ensure_ready()/fetch_json() at once. This
# semaphore fails callers fast so excess pollers fall through to RSS instead of
# queueing on the single Playwright dispatch slot.
_PLAYWRIGHT_GATE_LIMIT = _env_int("REDDIT_PLAYWRIGHT_CONCURRENCY", 2)
_PLAYWRIGHT_GATE_WAIT_SECONDS = float(_env_int("REDDIT_PLAYWRIGHT_GATE_WAIT_SECONDS", 10))
_playwright_call_gate = threading.BoundedSemaphore(value=_PLAYWRIGHT_GATE_LIMIT)

# --- Playwright circuit breaker ---
# After repeated consecutive failures, pause Playwright attempts for a cooldown
# window so every subreddit poll doesn't pay the full ensure_ready/fetch timeout
# cost while the browser layer is down.
_PLAYWRIGHT_BREAKER_THRESHOLD = _env_int("REDDIT_PLAYWRIGHT_BREAKER_THRESHOLD", 3)
_PLAYWRIGHT_BREAKER_COOLDOWN_SECONDS = float(_env_int("REDDIT_PLAYWRIGHT_BREAKER_COOLDOWN_SECONDS", 120))
_playwright_breaker_lock = threading.Lock()
_playwright_consecutive_failures = 0
_playwright_breaker_open_until = 0.0

# --- Shared 429 cooldown across all subreddits ---
# One subreddit's rate limit throttles the shared curl_cffi path for everyone,
# instead of letting each subreddit independently hammer Reddit into more 429s.
_ratelimit_lock = threading.Lock()
_ratelimit_until = 0.0


def _playwright_breaker_is_open() -> bool:
    return time.monotonic() < _playwright_breaker_open_until


def _record_playwright_result(success: bool) -> None:
    global _playwright_consecutive_failures, _playwright_breaker_open_until
    with _playwright_breaker_lock:
        if success:
            _playwright_consecutive_failures = 0
            return
        _playwright_consecutive_failures += 1
        if _playwright_consecutive_failures >= _PLAYWRIGHT_BREAKER_THRESHOLD:
            _playwright_breaker_open_until = time.monotonic() + _PLAYWRIGHT_BREAKER_COOLDOWN_SECONDS
            _playwright_consecutive_failures = 0
            print(
                f"[reddit] Playwright circuit breaker OPEN for {_PLAYWRIGHT_BREAKER_COOLDOWN_SECONDS:.0f}s "
                f"after {_PLAYWRIGHT_BREAKER_THRESHOLD} consecutive failures"
            )


def _shared_ratelimit_remaining() -> int:
    remaining = _ratelimit_until - time.monotonic()
    return int(remaining) if remaining > 0 else 0


def _note_shared_ratelimit(retry_after_seconds: int | None) -> None:
    global _ratelimit_until
    cooldown = max(60, int(retry_after_seconds or 0))
    with _ratelimit_lock:
        _ratelimit_until = max(_ratelimit_until, time.monotonic() + cooldown)


class RedditRateLimitError(Exception):
    def __init__(self, retry_after_seconds: int | None = None):
        self.retry_after_seconds = retry_after_seconds
        super().__init__("Reddit rate limited request (HTTP 429)")


def reddit_headers_for_fingerprint(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.reddit.com/",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }


def pick_reddit_user_agent(fingerprint_seed: int = 0) -> str:
    if not REDDIT_FINGERPRINTS:
        return "Mozilla/5.0 (compatible; RebuiltDiscordBot/1.0)"
    return REDDIT_FINGERPRINTS[fingerprint_seed % len(REDDIT_FINGERPRINTS)]


def _pick_cffi_impersonation(seed: int) -> str:
    return _CFFI_IMPERSONATIONS[seed % len(_CFFI_IMPERSONATIONS)]




def time_filter_cutoff_unix(time_filter: str) -> int | None:
    now = int(time.time())
    windows = {
        "hour": 60 * 60,
        "day": 24 * 60 * 60,
        "week": 7 * 24 * 60 * 60,
        "month": 30 * 24 * 60 * 60,
        "year": 365 * 24 * 60 * 60,
    }
    seconds = windows.get(time_filter)
    if seconds is None:
        return None
    return now - seconds


def _candidate_posts(post: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [post]
    crossposts = post.get("crosspost_parent_list") if isinstance(post.get("crosspost_parent_list"), list) else []
    for entry in crossposts:
        if isinstance(entry, dict):
            candidates.append(entry)
    return candidates


def _normalize_reddit_url(raw_url: str) -> str:
    return str(raw_url or "").replace("&amp;", "&").strip()


def _extract_gallery_metadata_urls(post: dict[str, Any]) -> list[str]:
    gallery_data = post.get("gallery_data") if isinstance(post.get("gallery_data"), dict) else {}
    media_metadata = post.get("media_metadata") if isinstance(post.get("media_metadata"), dict) else {}
    items = gallery_data.get("items") if isinstance(gallery_data.get("items"), list) else []
    if not items:
        return []

    out: list[str] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        media_id = str(entry.get("media_id") or "")
        if not media_id:
            continue
        metadata = media_metadata.get(media_id) if isinstance(media_metadata, dict) else None
        if not isinstance(metadata, dict):
            continue

        raw_url = ""
        source = metadata.get("s") if isinstance(metadata.get("s"), dict) else {}
        if source:
            raw_url = str(source.get("u") or source.get("gif") or source.get("mp4") or "")
        if not raw_url:
            previews = metadata.get("p") if isinstance(metadata.get("p"), list) else []
            if previews and isinstance(previews[-1], dict):
                raw_url = str(previews[-1].get("u") or "")

        url = _normalize_reddit_url(raw_url)
        if url.startswith("http") and url not in out:
            out.append(url)
    return out


def _extract_preview_image_urls(post: dict[str, Any]) -> list[str]:
    preview = post.get("preview") if isinstance(post.get("preview"), dict) else {}
    images = preview.get("images") if isinstance(preview.get("images"), list) else []
    out: list[str] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        source = image.get("source") if isinstance(image.get("source"), dict) else {}
        url = _normalize_reddit_url(str(source.get("url") or ""))
        if url.startswith("http") and url not in out:
            out.append(url)
    return out


def extract_gallery_image_urls(post: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for candidate in _candidate_posts(post):
        for url in _extract_gallery_metadata_urls(candidate):
            if url not in out:
                out.append(url)
        for url in _extract_preview_image_urls(candidate):
            if url not in out:
                out.append(url)
    return out


# ---------------------------------------------------------------------------
# curl_cffi (TLS impersonation) HTTP layer
# ---------------------------------------------------------------------------

def _make_cffi_session(proxy_url: str | None, impersonation: str) -> Any:
    """Return a curl_cffi Session with browser TLS fingerprint."""
    if not _CURL_CFFI_AVAILABLE:
        # Fall back to plain requests session
        sess = requests.Session()
        if proxy_url:
            sess.proxies.update({"http": proxy_url, "https": proxy_url})
        return sess

    sess = cffi_requests.Session(impersonate=impersonation)
    if proxy_url:
        sess.proxies = {"http": proxy_url, "https": proxy_url}
    return sess


def _cffi_get(session: Any, url: str, params: dict, headers: dict, timeout: int) -> Any:
    """Unified GET that works for both curl_cffi and plain requests sessions."""
    if _CURL_CFFI_AVAILABLE and isinstance(session, cffi_requests.Session):
        return session.get(url, params=params, headers=headers, timeout=timeout)
    return session.get(url, params=params, headers=headers, timeout=timeout)


# ---------------------------------------------------------------------------
# RSS scraper (Layer 2 -- no auth, no proxy, works when JSON is blocked)
# Fetching and Atom parsing are delegated to rss_service; only Reddit-specific
# URL construction, ID extraction, and post filtering live here.
# ---------------------------------------------------------------------------

# Links to skip when extracting post URLs from Reddit RSS content HTML.
_RSS_REDDIT_SKIP = ("/comments/", "/user/", "/u/", "reddit.com/r/")
_RSS_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)


def _rss_post_id(atom_id: str) -> str:
    return atom_id.split("_", 1)[-1] if "_" in atom_id else atom_id


def _scrape_via_rss(
    sub: str,
    listing: str,
    tf: str,
    max_items: int,
    cutoff_unix: int | None,
    seen_lookup: set[str] | None,
    include_nsfw: bool,
    include_spoiler: bool,
    wanted_flairs: set[str],
    mode: str,
    seen_url_set: set[str] | None = None,
) -> list[dict[str, Any]] | None:
    """
    Fetch posts via Reddit's Atom RSS feed.  Returns None on any error so the
    caller can fall through to the next layer.

    Limitations vs JSON API:
    - Max 25 posts per request (no pagination)
    - No flair, NSFW, or spoiler metadata (all posts treated as SFW)
    - Time filter applied client-side by published timestamp
    """
    sort_rss = {"new": "new", "hot": "hot", "top": "top", "rising": "rising"}.get(listing, "new")
    url = f"https://www.reddit.com/r/{sub}/{sort_rss}.rss"
    entries = fetch_and_parse_atom(url, user_agent=_RSS_UA)
    if entries is None:
        return None

    out: list[dict[str, Any]] = []
    for entry in entries:
        if len(out) >= max_items:
            break

        post_id = _rss_post_id(entry["id"])
        if seen_lookup is not None and post_id and post_id in seen_lookup:
            continue

        if cutoff_unix is not None and entry["published"]:
            try:
                ts = int(time.mktime(time.strptime(entry["published"][:19], "%Y-%m-%dT%H:%M:%S")))
                if ts < cutoff_unix:
                    continue
            except Exception:
                pass

        # RSS carries no NSFW/spoiler/flair metadata — those filters are skipped here.
        # Downstream Discord channel rules handle NSFW gating.

        title = entry["title"]
        permalink = entry["permalink"]
        post_link = extract_link_from_html(entry["content_html"], skip=_RSS_REDDIT_SKIP) or permalink

        # Determine media type and link based on mode
        is_video  = "v.redd.it" in post_link
        is_image  = any(post_link.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"))
        is_gallery = "/gallery/" in post_link

        if mode == "video":
            if not is_video:
                continue
            media_type, link = "video", post_link
        elif mode == "link":
            media_type, link = "link", permalink
        elif mode == "all":
            if is_video:
                media_type, link = "video", post_link
            elif is_image or is_gallery:
                media_type, link = "image", post_link
            else:
                media_type, link = "link", post_link
        else:  # image mode
            if not (is_image or is_gallery):
                continue
            media_type, link = "image", post_link

        if not link:
            continue

        candidate = {
            "id":          post_id,
            "title":       title,
            "link":        link,
            "permalink":   permalink,
            "source_url":  f"https://www.reddit.com/r/{sub}/",
            "subreddit":   sub,
            "flair":       "",
            "nsfw":        False,
            "spoiler":     False,
            "type":        media_type,
            "gallery_links": [],
        }
        if seen_url_set is not None:
            if any(u in seen_url_set for u in _item_dedup_urls(candidate)):
                continue
        out.append(candidate)

    return out


# ---------------------------------------------------------------------------
# Gallery supplementation — RSS can't extract gallery image URLs, so after
# an RSS scrape we try to backfill from a JSON-capable layer (Playwright or
# curl_cffi).
# ---------------------------------------------------------------------------

def _supplement_rss_galleries(
    rss_items: list[dict[str, Any]],
    sub: str,
    listing: str,
    tf: str,
    max_items: int,
    cutoff_unix: int | None,
    mode: str,
    user_agent: str,
    fingerprint_seed: int,
    proxy_url: str | None,
) -> list[dict[str, Any]]:
    """Backfill gallery image URLs for RSS posts that link to /gallery/ but have no gallery_links."""
    needs = {
        str(item.get("id") or ""): item
        for item in rss_items
        if "/gallery/" in str(item.get("link") or "") and not item.get("gallery_links")
    }
    needs.pop("", None)
    if not needs:
        return rss_items

    print(f"[reddit][diag] r/{sub}: {len(needs)} RSS gallery posts need image URL supplementation")

    # Playwright gallery-only fetch — may succeed if Chrome came back since the earlier attempt
    json_items = _scrape_via_playwright(
        sub=sub, listing=listing, tf=tf, max_items=max_items * 3,
        cutoff_unix=cutoff_unix, seen_lookup=None,
        include_nsfw=True, include_spoiler=True,
        wanted_flairs=set(), mode=mode,
        gallery_only=True, seen_url_set=None,
    )

    if json_items is None:
        try:
            all_items = _scrape_once(
                proxy_url=proxy_url, sub=sub, listing=listing, tf=tf,
                max_items=max_items * 3, cutoff_unix=cutoff_unix,
                seen_lookup=None, include_nsfw=True, include_spoiler=True,
                wanted_flairs=set(), mode=mode,
                user_agent=user_agent, fingerprint_seed=fingerprint_seed,
                seen_url_set=None,
            )
            json_items = [p for p in all_items if p.get("gallery_links")]
        except RedditRateLimitError:
            raise
        except Exception as exc:
            print(f"[reddit][diag] r/{sub}: gallery supplement via curl_cffi failed: {exc}")
            json_items = []

    supplemented = 0
    for json_item in (json_items or []):
        pid = str(json_item.get("id") or "")
        rss_item = needs.get(pid)
        if rss_item and json_item.get("gallery_links"):
            rss_item["gallery_links"] = json_item["gallery_links"]
            rss_item["link"] = json_item.get("link") or rss_item["link"]
            supplemented += 1

    print(f"[reddit][diag] r/{sub}: supplemented {supplemented}/{len(needs)} gallery posts")
    return rss_items


# ---------------------------------------------------------------------------
# Main scrape function
# ---------------------------------------------------------------------------

def scrape_subreddit_media(
    subreddit: str,
    sort: str = "new",
    time_filter: str = "day",
    flair_tags: list[str] | None = None,
    seen_lookup: set[str] | None = None,
    include_nsfw: bool = False,
    include_spoiler: bool = False,
    media_mode: str = "image",
    limit: int = 25,
    user_agent: str = "Mozilla/5.0 (compatible; RebuiltDiscordBot/1.0)",
    fingerprint_seed: int = 0,
    proxy_url: str | None = None,
    seen_url_set: set[str] | None = None,
) -> list[dict[str, Any]]:
    sort_map = {"new": "new", "top": "top", "trending": "rising", "rising": "rising", "hot": "hot"}
    allowed_time = {"hour", "day", "week", "month", "year", "all"}
    listing = sort_map.get(sort.lower(), "new")
    tf = time_filter.lower() if time_filter.lower() in allowed_time else "day"
    sub = subreddit.strip()
    if sub.lower().startswith("r/"):
        sub = sub[2:]
    cutoff_unix = None if listing == "top" else time_filter_cutoff_unix(tf)

    wanted_flairs = {part.strip().lower() for part in (flair_tags or []) if part.strip()}
    mode = media_mode.lower()
    max_items = max(1, min(100, limit))

    # --- Layer 1: Playwright JSON via Chrome (best data — galleries, NSFW, flairs) ---
    playwright_result = _scrape_via_playwright(
        sub=sub,
        listing=listing,
        tf=tf,
        max_items=max_items,
        cutoff_unix=cutoff_unix,
        seen_lookup=seen_lookup,
        include_nsfw=include_nsfw,
        include_spoiler=include_spoiler,
        wanted_flairs=wanted_flairs,
        mode=mode,
        seen_url_set=seen_url_set,
    )
    if playwright_result is not None:
        print(f"[reddit][diag] r/{sub}: Playwright returned {len(playwright_result)} items")
        return playwright_result
    print(f"[reddit][diag] r/{sub}: Playwright unavailable, trying RSS")

    # --- Layer 2: RSS feed (no auth, no proxy, works when JSON is blocked) ---
    rss_result = _scrape_via_rss(
        sub=sub,
        listing=listing,
        tf=tf,
        max_items=max_items,
        cutoff_unix=cutoff_unix,
        seen_lookup=seen_lookup,
        include_nsfw=include_nsfw,
        include_spoiler=include_spoiler,
        wanted_flairs=wanted_flairs,
        mode=mode,
        seen_url_set=seen_url_set,
    )
    if rss_result is not None and len(rss_result) > 0:
        print(f"[reddit][diag] r/{sub}: RSS returned {len(rss_result)} items")
        rss_result = _supplement_rss_galleries(
            rss_result, sub=sub, listing=listing, tf=tf,
            max_items=max_items, cutoff_unix=cutoff_unix, mode=mode,
            user_agent=user_agent, fingerprint_seed=fingerprint_seed,
            proxy_url=proxy_url,
        )
        return rss_result

    print(f"[reddit][diag] r/{sub}: RSS failed, trying curl_cffi")

    # --- Layer 3: curl_cffi TLS impersonation (with proxy if provided) ---
    return _scrape_once(
        sub=sub,
        listing=listing,
        tf=tf,
        max_items=max_items,
        cutoff_unix=cutoff_unix,
        seen_lookup=seen_lookup,
        include_nsfw=include_nsfw,
        include_spoiler=include_spoiler,
        wanted_flairs=wanted_flairs,
        mode=mode,
        user_agent=user_agent,
        fingerprint_seed=fingerprint_seed,
        proxy_url=proxy_url,
        seen_url_set=seen_url_set,
    )


_PROXY_PROBE_TIMEOUT = 8    # seconds per proxy attempt during rotation
_PROXY_BATCH_SIZE    = 10   # concurrent probes per round


def _scrape_with_proxy_rotation(
    proxy_candidates: list[str | None],
    sub: str,
    listing: str,
    tf: str,
    max_items: int,
    cutoff_unix: int | None,
    seen_lookup: set[str] | None,
    include_nsfw: bool,
    include_spoiler: bool,
    wanted_flairs: set[str],
    mode: str,
    user_agent: str,
    fingerprint_seed: int,
    seen_url_set: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Try proxies in parallel batches of _PROXY_BATCH_SIZE.
    First success wins; rotate to next batch on all-403/timeout.
    """
    import concurrent.futures

    kwargs = dict(
        sub=sub, listing=listing, tf=tf, max_items=max_items,
        cutoff_unix=cutoff_unix, seen_lookup=seen_lookup,
        include_nsfw=include_nsfw, include_spoiler=include_spoiler,
        wanted_flairs=wanted_flairs, mode=mode, user_agent=user_agent,
        seen_url_set=seen_url_set,
    )
    total = len(proxy_candidates)
    last_exc: Exception | None = None

    for batch_start in range(0, total, _PROXY_BATCH_SIZE):
        batch = proxy_candidates[batch_start:batch_start + _PROXY_BATCH_SIZE]
        batch_end = batch_start + len(batch)
        print(f"[reddit] Probing proxies {batch_start + 1}-{batch_end}/{total}...")

        winner: list[dict[str, Any]] | None = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
            future_to_proxy = {
                pool.submit(
                    _scrape_once,
                    proxy_url=proxy,
                    fingerprint_seed=fingerprint_seed + batch_start + i,
                    timeout_override=_PROXY_PROBE_TIMEOUT,
                    **kwargs,
                ): proxy
                for i, proxy in enumerate(batch)
            }
            for fut in concurrent.futures.as_completed(future_to_proxy):
                proxy = future_to_proxy[fut]
                label = proxy.split("@")[-1] if proxy else "no-proxy"
                try:
                    result = fut.result()
                    print(f"[reddit] Success via {label}")
                    winner = result
                    # Cancel remaining futures in this batch
                    for f in future_to_proxy:
                        f.cancel()
                    break
                except RedditRateLimitError:
                    for f in future_to_proxy:
                        f.cancel()
                    raise
                except Exception as exc:
                    last_exc = exc

        if winner is not None:
            return winner

    raise last_exc or Exception("All proxies exhausted (all 403 or timed out)")


def _parse_reddit_json_page(
    children: list[dict[str, Any]],
    sub: str,
    max_items: int,
    cutoff_unix: int | None,
    seen_lookup: set[str] | None,
    include_nsfw: bool,
    include_spoiler: bool,
    wanted_flairs: set[str],
    mode: str,
    seen_url_set: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse one page of Reddit JSON children into post dicts. Shared by all JSON-based layers."""
    out: list[dict[str, Any]] = []
    for child in children:
        if len(out) >= max_items:
            break
        post = child.get("data", {})
        candidate_posts = _candidate_posts(post)
        primary = candidate_posts[0]
        post_id = str(post.get("id") or "")
        if seen_lookup is not None and post_id and post_id in seen_lookup:
            continue
        if cutoff_unix is not None:
            created_utc = post.get("created_utc")
            try:
                if int(float(created_utc)) < cutoff_unix:
                    continue
            except (TypeError, ValueError):
                continue
        if post.get("stickied"):
            continue
        if post.get("over_18") and not include_nsfw:
            continue
        if post.get("spoiler") and not include_spoiler:
            continue
        flair = str(post.get("link_flair_text") or "")
        if wanted_flairs and flair.lower() not in wanted_flairs:
            continue
        media = primary.get("media") if isinstance(primary.get("media"), dict) else {}
        secure_media = primary.get("secure_media") if isinstance(primary.get("secure_media"), dict) else {}
        preview = primary.get("preview") if isinstance(primary.get("preview"), dict) else {}
        image_url = ""
        gallery_urls = extract_gallery_image_urls(post)
        video_url = str(
            media.get("reddit_video", {}).get("fallback_url")
            or secure_media.get("reddit_video", {}).get("fallback_url")
            or preview.get("reddit_video_preview", {}).get("fallback_url")
            or ""
        ).replace("&amp;", "&")
        raw_url = str(primary.get("url_overridden_by_dest") or primary.get("url") or "")
        if (
            primary.get("post_hint") == "image"
            or "i.redd.it/" in raw_url
            or raw_url.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))
        ):
            image_url = raw_url
        elif preview.get("images"):
            image_url = str(preview["images"][0].get("source", {}).get("url") or "").replace("&amp;", "&")
        elif gallery_urls:
            image_url = gallery_urls[0]
        permalink = f"https://www.reddit.com{post.get('permalink', '')}"
        media_type = "image"
        link = image_url
        if mode == "video":
            media_type = "video"
            link = video_url
        elif mode == "link":
            media_type = "link"
            link = permalink
        elif mode == "all":
            if video_url:
                media_type = "video"
                link = video_url
            elif image_url:
                media_type = "image"
                link = image_url
            elif primary.get("post_hint") != "image":
                # Only fall back to permalink for actual link posts, not failed image extractions.
                media_type = "link"
                link = permalink
        if not link:
            continue
        candidate = {
            "id": post_id,
            "title": str(post.get("title") or "(untitled)"),
            "link": link,
            "permalink": permalink,
            "source_url": f"https://www.reddit.com/r/{sub}/",
            "subreddit": sub,
            "flair": flair,
            "nsfw": bool(post.get("over_18")),
            "spoiler": bool(post.get("spoiler")),
            "type": media_type,
            "gallery_links": gallery_urls,
        }
        if seen_url_set is not None:
            if any(u in seen_url_set for u in _item_dedup_urls(candidate)):
                continue
        out.append(candidate)
    return out


def _scrape_via_playwright(
    sub: str,
    listing: str,
    tf: str,
    max_items: int,
    cutoff_unix: int | None,
    seen_lookup: set[str] | None,
    include_nsfw: bool,
    include_spoiler: bool,
    wanted_flairs: set[str],
    mode: str,
    gallery_only: bool = False,
    seen_url_set: set[str] | None = None,
) -> list[dict[str, Any]] | None:
    """
    Fetch via Chrome puppeteered by Playwright using the user's persistent profile.
    Returns None if the browser isn't ready or the request fails, so the caller
    can fall through to the next layer.

    When gallery_only=True, only gallery posts are processed — used to supplement
    the RSS layer which handles single images but cannot extract gallery image URLs.
    """
    try:
        from services import browser_service
    except ImportError:
        print(f"[reddit][diag] r/{sub}: Playwright layer import failed")
        return None

    if _playwright_breaker_is_open():
        print(f"[reddit][diag] r/{sub}: Playwright breaker open, skipping to next layer")
        return None
    if not _playwright_call_gate.acquire(timeout=_PLAYWRIGHT_GATE_WAIT_SECONDS):
        # Gate saturation is congestion, not brokenness -- don't feed the breaker.
        print(f"[reddit][diag] r/{sub}: Playwright call gate saturated, falling through")
        return None

    try:
        if not browser_service.ensure_ready():
            print(f"[reddit][diag] r/{sub}: Playwright ensure_ready returned False")
            _record_playwright_result(False)
            return None

        url = (
            f"https://www.reddit.com/r/{sub}/{listing}.json"
            f"?limit={max_items * 3}&t={tf}&raw_json=1&include_over_18=1&over18=yes"
        )
        data = browser_service.fetch_json(url)
        if data is None:
            print(f"[reddit][diag] r/{sub}: Playwright fetch_json returned None")
            _record_playwright_result(False)
            return None
        if not isinstance(data, dict):
            print(f"[reddit][diag] r/{sub}: Playwright fetch_json returned {type(data).__name__}, expected dict")
            _record_playwright_result(False)
            return None
        _record_playwright_result(True)
    finally:
        _playwright_call_gate.release()

    children = (data.get("data") or {}).get("children") or []
    if not children:
        return []

    if gallery_only:
        children = [c for c in children if (c.get("data") or {}).get("is_gallery")]
        if not children:
            return []

    return _parse_reddit_json_page(
        children=children,
        sub=sub,
        max_items=max_items,
        cutoff_unix=cutoff_unix,
        seen_lookup=seen_lookup,
        include_nsfw=include_nsfw,
        include_spoiler=include_spoiler,
        wanted_flairs=wanted_flairs,
        mode=mode,
        seen_url_set=seen_url_set,
    )


def _scrape_once(
    proxy_url: str | None,
    sub: str,
    listing: str,
    tf: str,
    max_items: int,
    cutoff_unix: int | None,
    seen_lookup: set[str] | None,
    include_nsfw: bool,
    include_spoiler: bool,
    wanted_flairs: set[str],
    mode: str,
    user_agent: str,
    fingerprint_seed: int,
    timeout_override: int | None = None,
    seen_url_set: set[str] | None = None,
) -> list[dict[str, Any]]:
    remaining = _shared_ratelimit_remaining()
    if remaining > 0:
        # Another subreddit already got 429'd; pace the shared curl_cffi path
        # instead of piling on and extending the rate limit.
        raise RedditRateLimitError(retry_after_seconds=remaining)

    timeout = timeout_override if timeout_override is not None else 25
    impersonation = _pick_cffi_impersonation(fingerprint_seed)
    out: list[dict[str, Any]] = []
    session = _make_cffi_session(proxy_url, impersonation)

    after: str | None = None
    request_count = 0
    while len(out) < max_items:
        active_user_agent = user_agent or pick_reddit_user_agent(fingerprint_seed + request_count)
        response = _cffi_get(
            session,
            f"https://www.reddit.com/r/{sub}/{listing}.json",
            params={"limit": max_items, "t": tf, "after": after, "raw_json": "1"},
            headers=reddit_headers_for_fingerprint(active_user_agent),
            timeout=timeout,
        )
        request_count += 1
        if response.status_code == 429:
            retry_after_raw = response.headers.get("Retry-After")
            try:
                retry_after_seconds = int(float(retry_after_raw)) if retry_after_raw else None
            except (TypeError, ValueError):
                retry_after_seconds = None
            _note_shared_ratelimit(retry_after_seconds)
            raise RedditRateLimitError(retry_after_seconds=retry_after_seconds)
        response.raise_for_status()
        payload = response.json()
        children = payload.get("data", {}).get("children", [])
        if not children:
            break

        page_items = _parse_reddit_json_page(
            children=children,
            sub=sub,
            max_items=max_items - len(out),
            cutoff_unix=cutoff_unix,
            seen_lookup=seen_lookup,
            include_nsfw=include_nsfw,
            include_spoiler=include_spoiler,
            wanted_flairs=wanted_flairs,
            mode=mode,
            seen_url_set=seen_url_set,
        )
        out.extend(page_items)

        after = payload.get("data", {}).get("after")
        if not after:
            break

    return out


def _normalize_seen_url(raw_url: str) -> str:
    """Strip query params from reddit media hosts for stable dedup comparison."""
    url = str(raw_url or "").strip()
    if not url:
        return ""
    if "redd.it/" in url or "redditmedia.com/" in url:
        return url.split("?")[0].split("#")[0]
    return url


def _item_dedup_urls(item: dict[str, Any]) -> list[str]:
    """All normalized media URLs from a post — link + every gallery image."""
    urls: list[str] = []
    link = _normalize_seen_url(str(item.get("link") or ""))
    if link:
        urls.append(link)
    for gallery_url in item.get("gallery_links", []):
        normalized = _normalize_seen_url(str(gallery_url or ""))
        if normalized and normalized not in urls:
            urls.append(normalized)
    return urls


def load_seen_ids(state_file: Path) -> tuple[list[str], list[str]]:
    if not state_file.exists():
        return [], []
    try:
        raw = json.loads(state_file.read_text(encoding="utf-8"))
        seen_ids: list[str] = []
        for value in raw.get("seen_ids", []):
            item = str(value)
            if item not in seen_ids:
                seen_ids.append(item)
        seen_urls: list[str] = []
        for value in raw.get("seen_urls", []):
            item = str(value)
            if item not in seen_urls:
                seen_urls.append(item)
        return seen_ids[-SEEN_URL_LIMIT:], seen_urls[-SEEN_URL_LIMIT:]
    except Exception:
        return [], []


def save_seen_ids(state_file: Path, seen_ids: list[str], seen_urls: list[str] | None = None, max_keep: int = SEEN_URL_LIMIT) -> None:
    trimmed = seen_ids[-max(1, max_keep):]
    payload: dict[str, Any] = {"seen_ids": trimmed}
    if seen_urls is not None:
        payload["seen_urls"] = seen_urls[-max(1, max_keep):]
    state_file.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


async def poll_subreddit_media(
    subreddit: str,
    post_callback: Callable[[dict[str, Any]], Any | Awaitable[Any]],
    state_file: Path,
    sort: str = "new",
    time_filter: str = "day",
    flair_tags: list[str] | None = None,
    include_nsfw: bool = False,
    include_spoiler: bool = False,
    media_mode: str = "image",
    limit: int = 25,
    interval_seconds: int = 300,
    scheduler: PriorityWorkScheduler | None = None,
) -> None:
    seen_ids, seen_urls = load_seen_ids(state_file)
    seen_lookup = set(seen_ids)
    seen_url_set = set(seen_urls)
    fingerprint_cursor = random.randint(0, max(0, len(REDDIT_FINGERPRINTS) - 1))
    proxy_pool = parse_proxy_pool(os.getenv("REDDIT_PROXIES"))
    proxy_cursor = random.randint(0, max(0, len(proxy_pool) - 1)) if proxy_pool else 0
    failure_streak = 0

    while True:
        sleep_seconds = max(60, interval_seconds)
        try:
            _scrape_kwargs = dict(
                subreddit=subreddit,
                sort=sort,
                time_filter=time_filter,
                flair_tags=flair_tags,
                seen_lookup=seen_lookup,
                include_nsfw=include_nsfw,
                include_spoiler=include_spoiler,
                media_mode=media_mode,
                limit=limit,
                user_agent=pick_reddit_user_agent(fingerprint_cursor),
                fingerprint_seed=fingerprint_cursor,
                proxy_url=pick_proxy(proxy_pool, proxy_cursor),
                seen_url_set=seen_url_set,
            )
            if scheduler is not None:
                items = await scheduler.run(
                    scrape_subreddit_media,
                    tier=BACKGROUND,
                    label=scheduler_labels.reddit_scrape_label(subreddit),
                    **_scrape_kwargs,
                )
            else:
                items = await asyncio.to_thread(scrape_subreddit_media, **_scrape_kwargs)
            print(f"[reddit][diag] r/{subreddit}: scraped {len(items)} items, seen_ids={len(seen_ids)}, seen_urls={len(seen_urls)}")
            batch_seen_ids = set(seen_lookup)
            batch_seen_urls = set(seen_url_set)
            fresh: list[dict[str, Any]] = []
            for item in reversed(items):
                post_id = str(item.get("id") or "").strip()
                if not post_id or post_id in batch_seen_ids:
                    continue
                item_urls = _item_dedup_urls(item)
                hit = [u for u in item_urls if u in batch_seen_urls]
                if hit:
                    print(f"[reddit][diag] r/{subreddit}: SKIP id={post_id} url-dup={hit[0][:80]}")
                    batch_seen_ids.add(post_id)
                    continue
                batch_seen_ids.add(post_id)
                batch_seen_urls.update(item_urls)
                fresh.append(item)
            print(f"[reddit][diag] r/{subreddit}: {len(fresh)} fresh posts after filter")
            for item in fresh:
                post_id = item["id"]
                item_urls = _item_dedup_urls(item)
                glinks = item.get("gallery_links", [])
                print(f"[reddit][diag] r/{subreddit}: POST id={post_id} link={str(item.get('link',''))[:80]} gallery={len(glinks)} urls_tracked={len(item_urls)}")
                result = post_callback(item)
                if asyncio.iscoroutine(result):
                    await result
                seen_ids.append(post_id)
                seen_lookup.add(post_id)
                for url in item_urls:
                    if url not in seen_url_set:
                        seen_urls.append(url)
                        seen_url_set.add(url)
                if len(seen_ids) > SEEN_URL_LIMIT:
                    evicted_id = seen_ids.pop(0)
                    seen_lookup.discard(evicted_id)
                while len(seen_urls) > SEEN_URL_LIMIT:
                    evicted_url = seen_urls.pop(0)
                    seen_url_set.discard(evicted_url)
                try:
                    save_seen_ids(state_file, seen_ids, seen_urls)
                except Exception as save_exc:
                    print(f"[reddit] Failed to persist seen state for r/{subreddit}: {save_exc}")
            failure_streak = 0
            fingerprint_cursor += 1
            proxy_cursor += 1
        except asyncio.CancelledError:
            raise
        except RedditRateLimitError as exc:
            failure_streak = min(6, failure_streak + 1)
            fingerprint_cursor += 1
            proxy_cursor += 1
            base_backoff = max(60, interval_seconds) * min(4, 2 ** min(2, failure_streak))
            sleep_seconds = max(base_backoff, int(exc.retry_after_seconds or 0))
            print(f"Reddit polling failed for r/{subreddit}: 429 rate limited. Retrying in {sleep_seconds}s")
        except Exception as exc:
            failure_streak = min(6, failure_streak + 1)
            fingerprint_cursor += 1
            proxy_cursor += 1
            sleep_seconds = max(60, interval_seconds) * min(4, 2 ** min(2, failure_streak))
            print(f"Reddit polling failed for r/{subreddit}: {exc}")

        await asyncio.sleep(max(60, int(sleep_seconds)))
