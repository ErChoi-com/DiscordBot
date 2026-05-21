from __future__ import annotations

from typing import Any
from urllib import robotparser
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


def parse_scrape_command(content: str) -> tuple[str, str | None]:
    payload = content[len("$scrape") :].strip()
    if not payload:
        raise ValueError("Missing URL")
    if "|" in payload:
        url, selector = payload.split("|", 1)
        return url.strip(), selector.strip() or None
    return payload, None


def scrape_url(
    url: str,
    selector: str | None = None,
    user_agent: str = "Mozilla/5.0 (compatible; RebuiltScraper/1.0)",
    timeout_seconds: int = 20,
    max_retries: int = 3,
    max_items: int = 20,
    respect_robots: bool = False,
) -> list[dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    # Default behavior is to ignore robots.txt unless explicitly requested.
    if respect_robots:
        robots_url = urljoin(url, "/robots.txt")
        parser = robotparser.RobotFileParser()
        parser.set_url(robots_url)
        try:
            parser.read()
            if not parser.can_fetch(user_agent, url):
                raise PermissionError(f"Blocked by robots.txt: {robots_url}")
        except PermissionError:
            raise
        except Exception:
            # If robots cannot be fetched/parsed, continue scraping.
            pass

    html: str | None = None
    for attempt in range(1, max(1, max_retries) + 1):
        try:
            response = session.get(url, timeout=max(1, timeout_seconds))
            response.raise_for_status()
            html = response.text
            break
        except Exception:
            if attempt >= max(1, max_retries):
                raise

    if html is None:
        return []

    soup = BeautifulSoup(html, "html.parser")
    nodes = soup.select(selector or "a[href]")

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes:
        text = node.get_text(" ", strip=True)
        href = node.get("href") if node.name == "a" else None
        if not href:
            nested = node.select_one("a[href]")
            if nested:
                href = nested.get("href")

        full_link = urljoin(url, href) if href else url
        title = text or "(no title)"
        key = f"{title}|{full_link}"
        if key in seen:
            continue
        seen.add(key)

        items.append({"title": title[:300], "link": full_link, "source_url": url})
        if len(items) >= max(1, max_items):
            break

    if items:
        return items

    page_title = (soup.title.string or "Untitled page").strip() if soup.title else "Untitled page"
    return [{"title": page_title, "link": url, "source_url": url}]


def clean_with_openrouter(
    source_url: str,
    items: list[dict[str, Any]],
    api_key: str,
    model: str = "openai/gpt-4o-mini",
) -> dict[str, str]:
    compact = [{"title": i.get("title", ""), "link": i.get("link", "")} for i in items[:15]]
    prompt = (
        "You are a text formatting assistant. Reformat scraped data into clean plain text, "
        "remove duplicates/noise, and keep it concise. Return plain text only.\n\n"
        f"Source: {source_url}\n"
        f"Items: {compact}"
    )
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"].strip()[:1800]
    return {"text": text, "source_url": source_url, "model": model}


def format_items(url: str, items: list[dict[str, Any]]) -> str:
    lines = [f"Results from {url}:"]
    for idx, item in enumerate(items[:10], start=1):
        title = str(item.get("title", "(untitled)"))
        link = str(item.get("link", url))
        site_label = str(item.get("site_label") or "").strip()
        prefix = f"[{site_label}] " if site_label else ""
        lines.append(f"{idx}. {prefix}{title} -> {link}")
    if len(items) > 10:
        lines.append(f"... and {len(items) - 10} more")
    return "\n".join(lines)


def chunk_text_for_discord(text: str, max_len: int = 1900) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_len:
        split_at = remaining.rfind("\n", 0, max_len)
        if split_at < max_len // 2:
            split_at = max_len
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks
