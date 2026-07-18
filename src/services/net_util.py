"""Shared networking helpers for the scraper services.

Before this module, the exponential-backoff delay formula was hand-rolled at
~8 call sites across ats_service and jba/scraper, and the proxy-pool helpers
existed as two diverging copies (reddit_service and job_service). The retry
LOOPS themselves stay at the call sites on purpose — each scraper's
status-code handling (dead-slug marking, UA rotation, SSL retries) is
genuinely different — only the shared arithmetic lives here.
"""

from __future__ import annotations

import random


def retry_backoff_delay(attempt: int) -> float:
    """Delay in seconds before retry number `attempt` (0-based): exponential
    with jitter, ~1.5s / ~3s / ~5s for attempts 0/1/2."""
    return (2 ** attempt) + random.uniform(0.5, 1.5)


def parse_proxy_pool(raw: str | None) -> list[str]:
    """Split a comma/semicolon/newline-separated proxy string into a clean,
    order-preserving, deduplicated list."""
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
    """Rotate through the pool by cursor; None when the pool is empty."""
    if not proxy_pool:
        return None
    return proxy_pool[cursor % len(proxy_pool)]
