"""The country index has to rebuild itself, or the ordering it feeds decays.

`_ordered_slugs` puts boards known to post in a channel's country at the front
of the fleet, which is what decides who survives a cycle cut off at the budget
(measured: CA-yielding boards inside the first 250 per platform, 43 -> 1,031).
It asks `geo_priority.slugs_for`, which reads a cache that only
`geo_priority.refresh` writes -- and nothing in `src/` or `scripts/` called
refresh.

That failed in two directions, both silent. On a machine where someone had once
run it by hand, the ordering used a snapshot older than the archive it claimed
to describe. On a fresh checkout there is no cache at all, `slugs_for` returns
an empty set, `partition` puts nothing at the front, and the prioritisation does
exactly nothing while still appearing wired.

The index is derived from the archive, and every scrape cycle adds to the
archive. So the fix cannot be to build it once.
"""
from __future__ import annotations

import asyncio
import inspect
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
def calls(monkeypatch):
    """Record the rebuild without reading a single archive zip."""
    from services.jba import geo_priority

    seen = {"refresh": 0, "label": None, "raises": False}

    async def _to_thread(self, fn, *args, label=None, **kwargs):
        seen["label"] = label
        return fn(*args, **kwargs)

    def _refresh(*_a, **_k):
        seen["refresh"] += 1
        if seen["raises"]:
            raise RuntimeError("archive unreadable")
        return {"CA": 1031, "US": 1843, "GB": 12}

    monkeypatch.setattr(WatcherManager, "_tracked_to_thread", _to_thread)
    monkeypatch.setattr(geo_priority, "refresh", _refresh)
    return seen


def test_the_index_is_rebuilt(calls, tmp_path):
    asyncio.run(_manager(tmp_path)._refresh_geo_index())
    assert calls["refresh"] == 1


def test_the_rebuild_runs_under_its_own_cost_label(calls, tmp_path):
    """It reads every archive zip, which is nothing like the git fetch beside
    it or the scrape after it; pooling the three skews all three medians."""
    asyncio.run(_manager(tmp_path)._refresh_geo_index())
    assert calls["label"] == "geo_index_refresh"


def test_the_rebuild_does_not_run_on_the_event_loop():
    """Scanning every archive zip inline stalls every watcher and every command
    for as long as it takes. Read from the source rather than from a stub,
    because a stubbed _tracked_to_thread runs inline too and so cannot tell a
    threaded call from a direct one.
    """
    compact = " ".join(
        inspect.getsource(WatcherManager._refresh_geo_index).split()
    )
    assert "await self._tracked_to_thread( geo_priority.refresh" in compact
    assert "geo_priority.refresh()" not in compact


def test_a_failed_rebuild_does_not_stop_the_scrape(calls, tmp_path, capsys):
    """Ordering is an optimisation. A corrupt zip must cost the ordering, not
    the cycle that was about to run."""
    calls["raises"] = True
    asyncio.run(_manager(tmp_path)._refresh_geo_index())  # must not raise
    assert "could not refresh the geo index" in capsys.readouterr().out


def test_what_was_rebuilt_is_reported(calls, tmp_path, capsys):
    """A rebuild that silently produced an empty index looks identical to one
    that worked, and an empty index is precisely the broken state."""
    asyncio.run(_manager(tmp_path)._refresh_geo_index())
    out = capsys.readouterr().out
    assert "3 countries" in out
    assert "US 1,843" in out


# ── it actually recurs ───────────────────────────────────────────────────────

def test_the_daily_branch_rebuilds_the_index():
    """Source-level pin, because the branch sits inside a `while True` that no
    test can drive. A capability nothing calls is the failure this whole change
    is about -- pinning the call site is the only way it stays called.
    """
    compact = " ".join(
        inspect.getsource(WatcherManager._run_ats_scrape_loop).split()
    )
    assert "await self._refresh_geo_index()" in compact


def test_the_index_is_rebuilt_after_the_company_sync_not_before():
    """A company discovered upstream this morning should be orderable this
    morning; rebuilding first indexes the fleet as it was yesterday."""
    compact = " ".join(
        inspect.getsource(WatcherManager._run_ats_scrape_loop).split()
    )
    assert (compact.index("_refresh_company_lists()")
            < compact.index("_refresh_geo_index()"))


def test_the_rebuild_is_not_wired_into_every_cycle():
    """It is guarded by the same day-rollover check as the company sync. Left
    unguarded it would rescan every archive four times a day to refresh an
    index the ordering can happily run a few hours behind.
    """
    lines = inspect.getsource(WatcherManager._run_ats_scrape_loop).splitlines()
    indent = lambda l: len(l) - len(l.lstrip())
    guard = next(i for i, l in enumerate(lines) if "if rolled_over:" in l)
    call = next(i for i, l in enumerate(lines) if "_refresh_geo_index()" in l)

    assert call > guard, "it runs before the day-rollover guard"
    # Every line between the guard and the call must stay inside the guard's
    # block. Comparing the call's indent to the guard's alone is not enough:
    # a branch that has been emptied out still leaves an `if rolled_over:`
    # line above a call that no longer sits inside it.
    body = [l for l in lines[guard + 1:call + 1] if l.strip()]
    assert all(indent(l) > indent(lines[guard]) for l in body), (
        "it is not inside the day-rollover branch")


def test_refresh_is_the_only_thing_that_writes_the_cache():
    """If some other path wrote it, this wiring would be redundant. Recorded
    as a test because that is the assumption the fix rests on.
    """
    from services.jba import geo_priority

    source = Path(inspect.getfile(geo_priority)).read_text(encoding="utf-8")
    writers = [
        name for name in ("save_cache",)
        if source.count(f"{name}(") > 1
    ]
    assert writers == ["save_cache"]
    callers = [
        l.strip() for l in source.splitlines()
        if "save_cache(" in l and not l.strip().startswith("def ")
    ]
    assert len(callers) == 1, callers
