"""The company lists are parsed from disk once, however many threads ask.

The ATS cycle hands every platform to its own thread at once. On a cold cache
each of them parsed all 135k slugs independently: five identical "+5333
companies from local harvest" lines interleaved in the startup log, five
copies of the fleet in memory until the last writer won.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_service as a  # noqa: E402


def _fleet_on_disk(monkeypatch, tmp_path: Path, slugs: list[str]) -> None:
    (tmp_path / "greenhouse.json").write_text(json.dumps(slugs), encoding="utf-8")
    monkeypatch.setattr(a, "_JBA_DIR", tmp_path)
    monkeypatch.setattr(a, "_PLATFORM_FILES", {"greenhouse": "greenhouse.json"})
    monkeypatch.setattr(a, "_harvest_sources", lambda: [])
    a.reload_company_lists()


def test_eight_cold_callers_parse_the_files_once(monkeypatch, tmp_path):
    _fleet_on_disk(monkeypatch, tmp_path, [f"c{i}" for i in range(500)])
    parses = {"n": 0}
    real_loads = a.json.loads
    gate = threading.Barrier(8)

    def _counting_loads(text, *args, **kwargs):
        parses["n"] += 1
        return real_loads(text, *args, **kwargs)

    monkeypatch.setattr(a.json, "loads", _counting_loads)
    results: list = []

    def _ask():
        gate.wait(5)                      # all eight arrive together
        results.append(a.load_company_lists())

    threads = [threading.Thread(target=_ask) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert parses["n"] == 1, f"the fleet was parsed {parses['n']} times"
    assert len(results) == 8
    assert all(r is results[0] for r in results), "every caller must get the one cached object"
    assert results[0]["greenhouse"] == [f"c{i}" for i in range(500)]


def test_reload_really_drops_the_cache(monkeypatch, tmp_path):
    _fleet_on_disk(monkeypatch, tmp_path, ["a"])
    first = a.load_company_lists()
    (tmp_path / "greenhouse.json").write_text(json.dumps(["a", "b"]), encoding="utf-8")
    assert a.load_company_lists() is first, "no reload, no re-read"
    a.reload_company_lists()
    assert a.load_company_lists()["greenhouse"] == ["a", "b"]


def test_a_missing_upstream_still_warns_loudly(monkeypatch, tmp_path, capsys):
    """The warning keyed on upstream, not the merged result, must survive the
    refactor -- it is what tells a fresh deployment its lists are absent."""
    monkeypatch.setattr(a, "_JBA_DIR", tmp_path / "nowhere")
    monkeypatch.setattr(a, "_PLATFORM_FILES", {"greenhouse": "greenhouse.json"})
    monkeypatch.setattr(a, "_harvest_sources", lambda: [])
    a.reload_company_lists()
    assert a.load_company_lists() == {}
    assert "WARNING: no company lists found" in capsys.readouterr().out
