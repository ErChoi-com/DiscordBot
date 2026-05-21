from __future__ import annotations

import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import requests


REDDIT_FINGERPRINTS: tuple[str, ...] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
)
SEEN_URL_LIMIT = 1000


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


def parse_proxy_pool(raw: str | None) -> list[str]:
    if not raw:
        return []
    normalized = raw.replace("\n", ",").replace(";", ",")
    proxies: list[str] = []
    for part in normalized.split(","):
        candidate = part.strip()
        if candidate and candidate not in proxies:
            proxies.append(candidate)
    return proxies


def pick_proxy(proxy_pool: list[str], cursor: int) -> str | None:
    if not proxy_pool:
        return None
    return proxy_pool[cursor % len(proxy_pool)]


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
) -> list[dict[str, Any]]:
    sort_map = {"new": "new", "top": "top", "trending": "rising", "rising": "rising", "hot": "hot"}
    allowed_time = {"hour", "day", "week", "month", "year", "all"}
    listing = sort_map.get(sort.lower(), "new")
    tf = time_filter.lower() if time_filter.lower() in allowed_time else "day"
    sub = subreddit.strip().lstrip("r/")
    cutoff_unix = None if listing == "top" else time_filter_cutoff_unix(tf)

    wanted_flairs = {part.strip().lower() for part in (flair_tags or []) if part.strip()}
    mode = media_mode.lower()
    max_items = max(1, min(100, limit))
    out: list[dict[str, Any]] = []
    session = requests.Session()
    if proxy_url:
        session.proxies.update({"http": proxy_url, "https": proxy_url})

    after: str | None = None
    request_count = 0
    while len(out) < max_items:
        active_user_agent = user_agent or pick_reddit_user_agent(fingerprint_seed + request_count)
        response = session.get(
            f"https://www.reddit.com/r/{sub}/{listing}.json",
            params={"limit": max_items, "t": tf, "after": after},
            headers=reddit_headers_for_fingerprint(active_user_agent),
            timeout=25,
        )
        request_count += 1
        if response.status_code == 429:
            retry_after_raw = response.headers.get("Retry-After")
            try:
                retry_after_seconds = int(float(retry_after_raw)) if retry_after_raw else None
            except (TypeError, ValueError):
                retry_after_seconds = None
            raise RedditRateLimitError(retry_after_seconds=retry_after_seconds)
        response.raise_for_status()
        payload = response.json()
        children = payload.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
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
            if primary.get("post_hint") == "image":
                image_url = str(primary.get("url_overridden_by_dest") or primary.get("url") or "")
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
                else:
                    media_type = "link"
                    link = permalink

            if not link:
                continue

            out.append(
                {
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
            )
            if len(out) >= max_items:
                break

        after = payload.get("data", {}).get("after")
        if not after:
            break

    return out


def load_seen_ids(state_file: Path) -> list[str]:
    if not state_file.exists():
        return []
    try:
        raw = json.loads(state_file.read_text(encoding="utf-8"))
        seen_ids: list[str] = []
        for value in raw.get("seen_ids", []):
            item = str(value)
            if item not in seen_ids:
                seen_ids.append(item)
        return seen_ids[-SEEN_URL_LIMIT:]
    except Exception:
        return []


def save_seen_ids(state_file: Path, seen_ids: list[str], max_keep: int = SEEN_URL_LIMIT) -> None:
    trimmed = seen_ids[-max(1, max_keep) :]
    state_file.write_text(json.dumps({"seen_ids": trimmed}, ensure_ascii=True), encoding="utf-8")


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
) -> None:
    seen_ids = load_seen_ids(state_file)
    seen_lookup = set(seen_ids)
    fingerprint_cursor = random.randint(0, max(0, len(REDDIT_FINGERPRINTS) - 1))
    proxy_pool = parse_proxy_pool(os.getenv("REDDIT_PROXIES"))
    proxy_cursor = random.randint(0, max(0, len(proxy_pool) - 1)) if proxy_pool else 0
    failure_streak = 0

    while True:
        sleep_seconds = max(60, interval_seconds)
        try:
            items = await asyncio.to_thread(
                scrape_subreddit_media,
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
            )
            fresh = [item for item in reversed(items) if item.get("id") and item["id"] not in seen_lookup]
            for item in fresh:
                result = post_callback(item)
                if asyncio.iscoroutine(result):
                    await result
                seen_ids.append(item["id"])
                seen_lookup.add(item["id"])
                if len(seen_ids) > SEEN_URL_LIMIT:
                    dropped = seen_ids.pop(0)
                    seen_lookup.discard(dropped)
            save_seen_ids(state_file, seen_ids)
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
