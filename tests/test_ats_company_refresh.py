"""A running bot has to be able to see a company discovered after it started.

Three things had to line up and the last two were missing. The harvest
workflow republishes weekly; `run.py` runs the sync exactly once, at boot; and
`load_company_lists` caches for the life of the process. So a bot up for a
fortnight scraped the fleet it booted with and could not see anything found
since, no matter how many times the harvest was republished.

`ats_service.reload_company_lists` was written for this and had never been
called from anywhere -- `grep -rn reload_company_lists src/` found only its own
definition.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from watchers.manager import WatcherManager  # noqa: E402


class _Config:
    def __init__(self, base_dir):
        self.base_dir = base_dir


def _manager(tmp_path):
    mgr = WatcherManager.__new__(WatcherManager)
    mgr.config = _Config(tmp_path)
    return mgr


@pytest.fixture
def calls(monkeypatch, tmp_path):
    """Record what the refresh does without running git or a subprocess."""
    from services import ats_service

    seen = {"sync": 0, "reload": 0, "rc": 0}

    async def _to_thread(self, fn, *args, label=None, **kwargs):
        seen["label"] = label
        return fn(*args, **kwargs)

    monkeypatch.setattr(WatcherManager, "_tracked_to_thread", _to_thread)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: type("R", (), {"returncode": seen["rc"], "stdout": "", "stderr": ""})(),
    )

    real_reload = ats_service.reload_company_lists

    def _counted_reload():
        seen["reload"] += 1
        real_reload()

    monkeypatch.setattr(ats_service, "reload_company_lists", _counted_reload)
    monkeypatch.setattr(ats_service, "load_company_lists", lambda: {"lever": ["a"]})
    (tmp_path / "sync_ats_companies.py").write_text("", encoding="utf-8")
    return seen


def test_the_refresh_drops_the_cached_fleet(calls, tmp_path):
    """The whole point. Re-running the sync without this writes new files that
    the process then never reads, because the cache is already populated.
    """
    asyncio.run(_manager(tmp_path)._refresh_company_lists())
    assert calls["reload"] == 1


def test_the_sync_runs_under_its_own_cost_label(calls, tmp_path):
    """Sharing ATS_SCRAPE's bucket would blend a git fetch into the scrape's
    learned duration and skew what the scheduler arbitrates on.
    """
    asyncio.run(_manager(tmp_path)._refresh_company_lists())
    assert calls["label"] == "ats_company_sync"


def test_a_failing_sync_still_reloads_what_is_on_disk(calls, tmp_path, capsys):
    """The sync is additive over the files already there, so a refresh that
    cannot reach GitHub must leave the fleet exactly as it was rather than
    skipping the reload and pretending nothing happened.
    """
    calls["rc"] = 1
    asyncio.run(_manager(tmp_path)._refresh_company_lists())
    assert calls["reload"] == 1
    assert "exited 1" in capsys.readouterr().out


def test_a_missing_sync_script_is_not_an_error(tmp_path):
    """A deployment that ships without the sync script -- the container image
    does -- must not raise inside the scrape loop.
    """
    asyncio.run(_manager(tmp_path)._refresh_company_lists())


def test_a_raising_refresh_does_not_take_the_scrape_down(monkeypatch, tmp_path):
    """It runs immediately before a scrape cycle. An exception here would cost
    the cycle, which is strictly worse than running on a slightly stale fleet.
    """
    async def _boom(self, fn, *args, **kwargs):
        raise RuntimeError("network gone")

    monkeypatch.setattr(WatcherManager, "_tracked_to_thread", _boom)
    (tmp_path / "sync_ats_companies.py").write_text("", encoding="utf-8")
    asyncio.run(_manager(tmp_path)._refresh_company_lists())   # must not raise


# ── the loop actually calling it ─────────────────────────────────────────────

def test_a_rolled_over_day_triggers_the_refresh():
    assert WatcherManager._is_new_utc_day("2026-09-06", "2026-09-05") is True


def test_the_first_pass_does_not_re_sync_what_boot_just_synced():
    """`run.py` syncs immediately before starting the bot, so the loop's first
    pass has nothing to fetch. Refreshing there repeats that fetch seconds
    later for nothing.
    """
    assert WatcherManager._is_new_utc_day("2026-09-05", None) is False


def test_the_same_day_does_not_refresh():
    """The loop spins several times a day. Refreshing on every pass would turn
    a weekly harvest into a git fetch every few minutes.
    """
    assert WatcherManager._is_new_utc_day("2026-09-05", "2026-09-05") is False


def test_the_loop_calls_the_refresh_under_the_rollover_guard():
    """A source-level pin, and deliberately so: the call site sits inside a
    `while True` whose next statement is a real scrape, so executing it in a
    test would mean scraping. What is checked is the wiring itself -- that the
    refresh is reached from the rollover decision, not merely mentioned --
    because "the piece exists but nothing calls it" is the exact shape of the
    bug this feature fixes, twice over.

    The decision itself is tested directly above; this pins that the loop
    consults it.
    """
    import inspect
    from watchers import manager

    compact = " ".join(
        inspect.getsource(manager.WatcherManager._run_ats_scrape_loop).split()
    )
    assert "rolled_over = self._is_new_utc_day(" in compact,         "the loop no longer asks _is_new_utc_day"
    assert "if rolled_over: await self._refresh_company_lists()" in compact,         "the refresh is no longer guarded by the rollover decision"
