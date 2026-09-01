"""Tests for the upstream comparison.

The measurement it replaces was wrong by 9,340 companies, in a way that looked
entirely reasonable: a set difference over raw strings. These cover the two
spellings that caused it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compare_upstream as cu  # noqa: E402


def test_icims_prefix_variants_are_one_company():
    """Upstream carries "careers-2u" and we store "2u". A raw set difference
    reports 3,255 iCIMS companies missing that we already have."""
    assert cu.canonical("icims", ["careers-2u"]) == cu.canonical("icims", ["2u"])


def test_unresolvable_upstream_entries_count_for_nobody():
    """wd1|wd1|careers cannot resolve. Counting upstream's 6,068 of those
    credits them with companies that do not exist."""
    assert cu.canonical("workday", ["wd1|wd1|careers"]) == set()
    assert cu.canonical("workday", ["acme|wd1|external"]) == {"acme|wd1|external"}


def test_paylocity_objects_are_unwrapped(tmp_path, monkeypatch):
    """Upstream ships {guid, name, jobs} objects for this platform; stringifying
    them would make every entry a unique non-company."""
    path = tmp_path / "paylocity_companies_clean.json"
    guid = "2eba1a0a-d60f-4fd8-95ab-90b070f1d9f2"
    path.write_text(json.dumps([{"guid": guid, "name": "X", "jobs": 5}]))
    assert cu._read(path) == [guid]


def test_comparison_is_symmetric_on_canonical_forms(tmp_path, monkeypatch):
    monkeypatch.setattr(cu, "COMPANY_DIR", tmp_path)
    monkeypatch.setattr(cu, "HARVEST_DIR", tmp_path)
    (tmp_path / "icims_companies.json").write_text(json.dumps(["careers-acme", "solo"]))
    (tmp_path / "icims.json").write_text(json.dumps(["acme", "ours"]))
    row = cu.compare("icims")
    # careers-acme and acme are the same company, so neither side "adds" it.
    assert row["we_add"] == 1 and row["they_add"] == 1
    assert row["union"] == 3


def test_missing_list_is_the_canonical_difference(tmp_path, monkeypatch):
    monkeypatch.setattr(cu, "COMPANY_DIR", tmp_path)
    monkeypatch.setattr(cu, "HARVEST_DIR", tmp_path)
    (tmp_path / "icims_companies.json").write_text(json.dumps(["careers-acme", "gone"]))
    (tmp_path / "icims.json").write_text(json.dumps(["acme"]))
    assert cu.compare("icims")["_missing"] == ["gone"]
