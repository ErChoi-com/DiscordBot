"""A timed-out ATS platform must not render as a quiet one.

str(asyncio.TimeoutError()) is '' -- and from Python 3.11 asyncio.TimeoutError
IS the builtin TimeoutError -- so `error=str(exc)` handed health an empty
string, and health branched on `if error:`. The empty string took the SUCCESS
branch: last_was_error stayed False, total_errors never moved, job_count was
recorded as 0 and consecutive_silent kept climbing.

.health then showed every timed-out platform as "0 scraped / 0 new" with the
quiet icon and no error count -- identical to a platform that genuinely had
nothing new. A fleet-wide timeout was indistinguishable from a slow news day,
which is the one case where the distinction matters most.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.health import WatcherHealthTracker
from watchers.manager import exception_text


def test_a_timeouts_str_really_is_empty():
    """The premise. If this ever changes the rest of this file is moot."""
    assert str(asyncio.TimeoutError()) == ""
    assert not str(TimeoutError())
    assert asyncio.TimeoutError is TimeoutError


def test_exception_text_is_never_empty():
    assert exception_text(asyncio.TimeoutError()) == "TimeoutError"
    assert exception_text(TimeoutError()) == "TimeoutError"
    assert exception_text(asyncio.CancelledError()) == "CancelledError"
    assert exception_text(ValueError("boom")) == "boom", "a real message survives"


def _tracker():
    return WatcherHealthTracker()


def test_a_timeout_is_recorded_as_an_error_not_a_quiet_platform():
    """The regression, driven through the real recorder."""
    t = _tracker()
    t.record_ats_platform_result("lever", 0, error=exception_text(asyncio.TimeoutError()))
    ph = t._ats.per_platform["lever"]
    assert ph.last_was_error is True
    assert ph.total_errors == 1
    assert ph.last_error


def test_even_a_blank_message_reaching_health_counts_as_an_error():
    """Defence in depth: the call sites go through exception_text now, but
    health must not be the thing that decides an error did not happen."""
    t = _tracker()
    t.record_ats_platform_result("lever", 0, error="")
    ph = t._ats.per_platform["lever"]
    assert ph.last_was_error is True
    assert ph.total_errors == 1
    assert ph.last_error, "an empty message must still leave something readable"


def test_a_genuinely_quiet_platform_is_still_quiet():
    """The other half. Marking every silent platform broken would be worse."""
    t = _tracker()
    t.record_ats_platform_result("lever", 0, new_count=0)
    ph = t._ats.per_platform["lever"]
    assert ph.last_was_error is False
    assert ph.total_errors == 0


def test_the_two_render_differently():
    """What .health actually shows -- the thing that was indistinguishable."""
    quiet = _tracker()
    quiet.record_ats_platform_result("lever", 0)
    timed_out = _tracker()
    timed_out.record_ats_platform_result("lever", 0, error=exception_text(TimeoutError()))

    a = quiet._ats.per_platform["lever"]
    b = timed_out._ats.per_platform["lever"]
    icon = lambda ph: "❌" if ph.last_was_error else ("✅" if ph.last_job_count else "🔇")
    assert icon(a) == "🔇"
    assert icon(b) == "❌"
    assert icon(a) != icon(b), "a timeout must not look like a quiet day"


def test_a_timeout_does_not_inflate_the_silent_streak():
    """consecutive_silent drives the 'silent N×' note. Counting timeouts into
    it makes a broken platform look like a boring one, with evidence."""
    t = _tracker()
    for _ in range(5):
        t.record_ats_platform_result("lever", 0, error=exception_text(TimeoutError()))
    assert t._ats.per_platform["lever"].consecutive_silent == 0
    assert t._ats.per_platform["lever"].total_errors == 5


def test_coverage_counts_survive_the_error_path():
    """Reaching 9,000 of 10,000 companies then timing out is a different
    problem from reaching 12, and the counts are what tell them apart."""
    t = _tracker()
    t.record_ats_platform_result(
        "lever", 0, error=exception_text(TimeoutError()), submitted=10_000, completed=9_000
    )
    ph = t._ats.per_platform["lever"]
    assert (ph.last_submitted, ph.last_completed) == (10_000, 9_000)
