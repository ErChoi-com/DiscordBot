"""Playwright circuit breaker + shared 429 pacing (item 6)."""
from __future__ import annotations

import time

import pytest

from services import browser_service, reddit_service


def _call_scrape(sub: str = "testsub"):
    return reddit_service._scrape_via_playwright(
        sub=sub,
        listing="new",
        tf="day",
        max_items=5,
        cutoff_unix=None,
        seen_lookup=None,
        include_nsfw=False,
        include_spoiler=False,
        wanted_flairs=set(),
        mode="image",
    )


def test_breaker_opens_after_threshold_and_skips_playwright(monkeypatch, capsys):
    ensure_calls = {"count": 0}

    def failing_ensure_ready() -> bool:
        ensure_calls["count"] += 1
        return False

    monkeypatch.setattr(browser_service, "ensure_ready", failing_ensure_ready)
    reddit_service._playwright_consecutive_failures = 0
    reddit_service._playwright_breaker_open_until = 0.0

    for _ in range(reddit_service._PLAYWRIGHT_BREAKER_THRESHOLD):
        assert _call_scrape() is None
    assert reddit_service._playwright_breaker_is_open()
    assert "circuit breaker OPEN" in capsys.readouterr().out

    # While open: Playwright must not even be attempted.
    calls_before = ensure_calls["count"]
    assert _call_scrape() is None
    assert ensure_calls["count"] == calls_before
    assert "breaker open" in capsys.readouterr().out


def test_breaker_closes_after_cooldown(monkeypatch):
    monkeypatch.setattr(browser_service, "ensure_ready", lambda: True)
    monkeypatch.setattr(
        browser_service, "fetch_json", lambda url, timeout_ms=15_000: {"data": {"children": []}}
    )
    reddit_service._playwright_breaker_open_until = time.monotonic() - 1  # cooldown elapsed
    reddit_service._playwright_consecutive_failures = 0

    assert not reddit_service._playwright_breaker_is_open()
    assert _call_scrape() == []


def test_success_resets_failure_streak(monkeypatch):
    reddit_service._playwright_consecutive_failures = reddit_service._PLAYWRIGHT_BREAKER_THRESHOLD - 1
    reddit_service._playwright_breaker_open_until = 0.0

    reddit_service._record_playwright_result(True)
    assert reddit_service._playwright_consecutive_failures == 0
    assert not reddit_service._playwright_breaker_is_open()


def test_shared_ratelimit_paces_other_subreddits(monkeypatch):
    reddit_service._ratelimit_until = 0.0
    assert reddit_service._shared_ratelimit_remaining() == 0

    reddit_service._note_shared_ratelimit(90)
    remaining = reddit_service._shared_ratelimit_remaining()
    assert 0 < remaining <= 90

    # Any subreddit entering the curl_cffi layer during cooldown gets paced.
    with pytest.raises(reddit_service.RedditRateLimitError) as exc_info:
        reddit_service._scrape_once(
            sub="othersub",
            listing="new",
            tf="day",
            max_items=5,
            cutoff_unix=None,
            seen_lookup=None,
            include_nsfw=False,
            include_spoiler=False,
            wanted_flairs=set(),
            mode="image",
            user_agent="test",
            fingerprint_seed=0,
            proxy_url=None,
        )
    assert exc_info.value.retry_after_seconds <= 90


def test_shared_ratelimit_floors_at_60s():
    reddit_service._ratelimit_until = 0.0
    reddit_service._note_shared_ratelimit(None)
    assert reddit_service._shared_ratelimit_remaining() >= 55  # 60s floor minus test time
