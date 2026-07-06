from __future__ import annotations

import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

_ATOM_NS = "http://www.w3.org/2005/Atom"
_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)


class _LinkExtractor(HTMLParser):
    def __init__(self, skip: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.link: str = ""
        self._skip = skip

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a" or self.link:
            return
        href = dict(attrs).get("href", "") or ""
        if href and href.startswith("http") and not any(s in href for s in self._skip):
            self.link = href


def extract_link_from_html(content_html: str, skip: tuple[str, ...] = ()) -> str:
    """Return the first HTTP link in HTML, skipping any URLs that contain a skip pattern."""
    p = _LinkExtractor(skip=skip)
    p.feed(content_html)
    return p.link


def fetch_and_parse_atom(
    url: str,
    user_agent: str = _DEFAULT_UA,
    timeout: int = 15,
) -> list[dict] | None:
    """
    Fetch an Atom feed and return a list of entry dicts, or None on failure.

    Each dict contains:
        id           - atom:id text (raw)
        title        - atom:title text
        permalink    - atom:link href
        content_html - atom:content text (may be empty)
        published    - atom:published text (ISO-8601)
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except Exception as exc:
        print(f"[rss] fetch failed ({url}): {exc}")
        return None

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        print(f"[rss] XML parse error ({url}): {exc}")
        return None

    out: list[dict] = []
    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        atom_id = (entry.findtext(f"{{{_ATOM_NS}}}id") or "").strip()
        title_el = entry.find(f"{{{_ATOM_NS}}}title")
        title = (title_el.text or "(untitled)") if title_el is not None else "(untitled)"
        link_el = entry.find(f"{{{_ATOM_NS}}}link")
        permalink = (link_el.get("href") or "") if link_el is not None else ""
        content_el = entry.find(f"{{{_ATOM_NS}}}content")
        content_html = (content_el.text or "") if content_el is not None else ""
        published = (entry.findtext(f"{{{_ATOM_NS}}}published") or "").strip()
        out.append({
            "id": atom_id,
            "title": title,
            "permalink": permalink,
            "content_html": content_html,
            "published": published,
        })
    return out
