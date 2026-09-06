"""A signed-out browser looks exactly like a quiet subreddit.

The Reddit watcher scrapes through the shared Chrome profile. When that profile
loses its Reddit login the watcher does not stop and does not error -- Reddit
still answers, with less -- so the run completes, reports nothing new, and
`.health` says "Running". Every symptom points at a quiet week.

browser_service already tracked the answer in `session_valid()`, and nothing
called it: the flag was maintained on a timer and read by no one. This is the
same shape as the ATS platform lines, where "returned nothing" and "was never
asked" had to be told apart before workable's 45 silent runs were visible.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.health import (  # noqa: E402
    WatcherHealthTracker,
    build_channel_health_embed,
)


def _reddit_field(session_valid, *, alive=True):
    main, _ats = build_channel_health_embed(
        WatcherHealthTracker(), channel_id=1, channel_name="jobs",
        job_task_alive=False, reddit_task_alive=alive,
        active_job_watchers=0, active_reddit_watchers=1,
        reddit_session_valid=session_valid,
    )
    return next(f for f in main.fields if f.name == "Reddit Watcher").value


# ── the distinction the field exists to make ────────────────────────────────

def test_a_signed_out_session_is_called_out():
    value = _reddit_field(False)
    assert "Signed out" in value


def test_a_signed_out_watcher_still_reads_as_running():
    """It is running. Saying otherwise would send someone looking at the task
    when the problem is the login."""
    value = _reddit_field(False)
    assert "Running" in value and "Signed out" in value


def test_a_logged_in_session_says_so_rather_than_saying_nothing():
    """Silence would be indistinguishable from a build that never checked."""
    assert "Logged in" in _reddit_field(True)


def test_the_two_states_do_not_render_the_same():
    assert _reddit_field(True) != _reddit_field(False)


def test_a_signed_out_session_says_what_it_costs():
    """"Signed out" alone reads like a login prompt is coming. What actually
    happens is that scraping continues against the logged-out view."""
    assert "public view" in _reddit_field(False)


# ── the state that is not a session problem ─────────────────────────────────

def test_a_browser_that_is_not_up_reports_no_session_state_at_all():
    """session_valid() is False before the first probe, so a down browser would
    otherwise be reported as a signed-out account -- a wrong diagnosis, and one
    that sends you to re-authenticate something that is not broken.
    """
    value = _reddit_field(None)
    assert "Signed out" not in value
    assert "Logged in" not in value
    assert value == "🟢 Running"


def test_an_inactive_watcher_still_reports_the_session():
    """The session is a property of the browser, not of this channel's task;
    a channel with no watcher is where you would go to ask why."""
    assert "Signed out" in _reddit_field(False, alive=False)
    assert "Not active" in _reddit_field(False, alive=False)


def test_the_parameter_defaults_to_unknown():
    """Every other caller of this builder -- and there are tests that call it
    positionally -- must keep working and must not gain a false diagnosis."""
    main, _ats = build_channel_health_embed(
        WatcherHealthTracker(), channel_id=1, channel_name="jobs",
        job_task_alive=False, reddit_task_alive=True,
        active_job_watchers=0, active_reddit_watchers=1,
    )
    value = next(f for f in main.fields if f.name == "Reddit Watcher").value
    assert value == "🟢 Running"


# ── it is actually asked ────────────────────────────────────────────────────

def test_the_health_command_reads_the_real_session_state():
    """The flag existed and was maintained for months with no reader. A pin on
    the call site is the only thing that keeps this one from going the same way.
    """
    from commands import handlers

    src = inspect.getsource(handlers)
    compact = " ".join(src.split())
    assert "reddit_session_valid=reddit_session" in compact
    assert "browser_service.session_valid()" in compact


def test_the_command_asks_whether_the_browser_is_up_first():
    """Without the guard, every bot with the browser disabled reports a signed
    out Reddit account permanently."""
    from commands import handlers

    compact = " ".join(inspect.getsource(handlers).split())
    assert ("browser_service.session_valid() if browser_service.is_ready() "
            "else None") in compact
