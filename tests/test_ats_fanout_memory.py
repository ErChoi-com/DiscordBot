"""What the last fan-out reached survives a restart, next to the rotation cursor.

The counts size the next ask (ats_fanout_cap). This bot is restarted daily by
its supervisor and more often by hand, and a restart that forgot them re-asked
icims the 600 boards the floor promised -- right after a cycle had measured
that 103 fit. The rotation cursor already survived restarts for exactly this
reason; the counts now travel with it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_service  # noqa: E402
from watchers.manager import ATS_PLATFORM_FANOUT_BUDGET_S, WatcherManager  # noqa: E402


class _Store:
    def __init__(self, channels):
        self.channel_job_settings = channels


class _Config:
    def __init__(self, base_dir):
        self.base_dir = base_dir


def _manager(tmp_path, channels=None):
    """A fresh process, as far as the fan-out record is concerned."""
    mgr = WatcherManager.__new__(WatcherManager)
    mgr.store = _Store(channels or {})
    mgr.config = _Config(tmp_path)
    return mgr


@pytest.fixture(autouse=True)
def _no_process_memory(monkeypatch):
    monkeypatch.setattr(ats_service, "LAST_FANOUT", {})


def test_nothing_recorded_reads_as_zeroes(tmp_path):
    assert _manager(tmp_path)._ats_last_fanout("icims") == {"submitted": 0, "completed": 0}


def test_a_recorded_fanout_is_read_back_by_a_new_process(tmp_path):
    _manager(tmp_path)._remember_fanout("icims", {"submitted": 600, "completed": 103})
    assert _manager(tmp_path)._ats_last_fanout("icims") == {"submitted": 600, "completed": 103}


def test_this_process_own_record_wins_over_the_saved_one(tmp_path, monkeypatch):
    _manager(tmp_path)._remember_fanout("icims", {"submitted": 600, "completed": 103})
    monkeypatch.setattr(ats_service, "LAST_FANOUT", {"icims": {"submitted": 180, "completed": 180}})
    assert _manager(tmp_path)._ats_last_fanout("icims") == {"submitted": 180, "completed": 180}


def test_a_fanout_that_never_happened_does_not_overwrite_a_measurement(tmp_path):
    mgr = _manager(tmp_path)
    mgr._remember_fanout("icims", {"submitted": 600, "completed": 103})
    mgr._remember_fanout("icims", {"submitted": 0, "completed": 0})   # queued past its bound
    assert _manager(tmp_path)._ats_last_fanout("icims") == {"submitted": 600, "completed": 103}


def test_the_record_lives_beside_the_cursor_not_in_place_of_it(tmp_path):
    mgr = _manager(tmp_path)
    path = mgr._ats_rotation_state_path()
    path.write_text(json.dumps({"platforms": {"icims": {"last_digest": "abc", "slices": 3}}}), encoding="utf-8")
    mgr._remember_fanout("icims", {"submitted": 600, "completed": 103})
    entry = json.loads(path.read_text(encoding="utf-8"))["platforms"]["icims"]
    assert entry["last_digest"] == "abc" and entry["slices"] == 3
    assert entry["fanout"]["submitted"] == 600 and entry["fanout"]["completed"] == 103


def test_a_corrupt_state_file_reads_as_zeroes_not_an_error(tmp_path):
    mgr = _manager(tmp_path)
    mgr._ats_rotation_state_path().write_text("{not json", encoding="utf-8")
    assert mgr._ats_last_fanout("icims") == {"submitted": 0, "completed": 0}
    mgr._remember_fanout("icims", {"submitted": 5, "completed": 5})       # must not raise
    assert _manager(tmp_path)._ats_last_fanout("icims") == {"submitted": 5, "completed": 5}


def test_the_next_process_sizes_its_ask_from_the_saved_measurement(tmp_path, monkeypatch):
    """The point, end to end: icims after a restart asks what fit, not the floor."""
    from services.jba import geo_priority

    fleet = [f"c{i}" for i in range(2_000)]
    head = {f"c{i}" for i in range(150)}
    monkeypatch.setattr(ats_service, "load_company_lists", lambda: {"icims": list(fleet)})
    monkeypatch.setattr(geo_priority, "slugs_for", lambda country, *a, **k: frozenset(head))
    monkeypatch.setattr(ats_service, "fanout_workers", lambda platform, fleet_size: 30)
    canada = {1: {"enabled": True, "location": "Canada"}}

    first = _manager(tmp_path, canada)._ordered_slugs("icims")
    assert len(first) == 150 + ats_service.fanout_capacity(ATS_PLATFORM_FANOUT_BUDGET_S, 30)   # the floor: 600

    _manager(tmp_path, canada)._remember_fanout("icims", {"submitted": len(first), "completed": 103})
    after_restart = _manager(tmp_path, canada)._ordered_slugs("icims")
    assert len(after_restart) == 150 + 30, "head plus one wave, not the floor again"


def test_the_scrape_records_what_it_reached(tmp_path, monkeypatch):
    """_scrape_one_platform is where the counts are known; it must hand them on."""
    from types import SimpleNamespace

    mgr = _manager(tmp_path)
    mgr.health = SimpleNamespace(record_ats_platform_result=lambda *a, **k: None)
    monkeypatch.setattr(mgr, "_ordered_slugs", lambda platform: ["a", "b"])
    monkeypatch.setattr("watchers.manager._scrape_ats_platform", lambda **kw: [])
    monkeypatch.setattr(ats_service, "LAST_FANOUT", {"icims": {"submitted": 2, "completed": 1}})

    mgr._scrape_one_platform("icims")
    monkeypatch.setattr(ats_service, "LAST_FANOUT", {})
    assert _manager(tmp_path)._ats_last_fanout("icims") == {"submitted": 2, "completed": 1}
