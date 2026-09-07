"""Single-region boards (naukri/bdjobs/bayt) must not be asked for a channel
outside the country they serve when the channel expanded "all", while a site
named explicitly is always honoured and an undecidable location keeps every
site (no evidence is not evidence of absence)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import services.job_service as job_service
from services.job_service import scrape_job_postings, sites_serving

ALL_SITES = ["glassdoor", "linkedin", "naukri", "bdjobs", "bayt", "indeed"]


@pytest.fixture(autouse=True)
def _fresh_cache():
    job_service.clear_job_scrape_cache()
    yield
    job_service.clear_job_scrape_cache()


# ---------------------------------------------------------------------------
# sites_serving (pure)
# ---------------------------------------------------------------------------


def test_unmapped_sites_are_always_kept():
    assert sites_serving(["indeed", "linkedin", "glassdoor"], "CA") == [
        "indeed", "linkedin", "glassdoor",
    ]


def test_mapped_site_dropped_outside_its_country():
    assert sites_serving(["indeed", "naukri"], "CA") == ["indeed"]


def test_mapped_site_kept_inside_its_country():
    assert sites_serving(["indeed", "naukri"], "IN") == ["indeed", "naukri"]


def test_bayt_region_set():
    assert sites_serving(["bayt"], "AE") == ["bayt"]
    assert sites_serving(["bayt"], "SA") == ["bayt"]
    assert sites_serving(["bayt"], "CA") == []


def test_none_country_keeps_everything():
    assert sites_serving(ALL_SITES, None) == ALL_SITES


def test_empty_string_country_keeps_everything():
    assert sites_serving(ALL_SITES, "") == ALL_SITES


def test_order_is_preserved():
    mixed = ["bayt", "indeed", "naukri", "linkedin", "bdjobs", "glassdoor"]
    assert sites_serving(mixed, "CA") == ["indeed", "linkedin", "glassdoor"]


def test_input_list_not_mutated():
    original = ["indeed", "naukri"]
    snapshot = list(original)
    sites_serving(original, "CA")
    assert original == snapshot


# ---------------------------------------------------------------------------
# scrape_job_postings integration
# ---------------------------------------------------------------------------


def _stub_all_sites(monkeypatch, sites=ALL_SITES):
    monkeypatch.setattr(job_service, "all_supported_job_sites", lambda exe=None: list(sites))
    monkeypatch.setattr(
        job_service,
        "jobspy_runtime_metadata",
        lambda exe=None: {"sites": tuple(sites), "params": tuple()},
    )


def _record_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        job_service,
        "_scrape_filtered_rows_uncached",
        lambda sites, *a, **k: calls.append(list(sites)) or [],
    )
    return calls


def test_all_for_canadian_channel_drops_single_region_boards(monkeypatch):
    _stub_all_sites(monkeypatch)
    calls = _record_calls(monkeypatch)

    scrape_job_postings(["all"], "intern", "Toronto, ON, Canada")

    assert calls, "the stubbed scraper was never invoked"
    sent = calls[0]
    assert "naukri" not in sent
    assert "bdjobs" not in sent
    assert "bayt" not in sent
    assert "glassdoor" in sent
    assert "linkedin" in sent
    assert "indeed" in sent


def test_all_for_indian_channel_keeps_naukri_drops_others(monkeypatch):
    _stub_all_sites(monkeypatch)
    calls = _record_calls(monkeypatch)

    scrape_job_postings(["all"], "intern", "Bengaluru, Karnataka, India")

    assert calls, "the stubbed scraper was never invoked"
    sent = calls[0]
    assert "naukri" in sent
    assert "bdjobs" not in sent
    assert "bayt" not in sent


def test_explicit_site_request_bypasses_the_country_filter(monkeypatch):
    # normalize_requested_sites() intersects the request against
    # jobspy_runtime_metadata()["sites"]; stub that so no subprocess spawns.
    monkeypatch.setattr(
        job_service,
        "jobspy_runtime_metadata",
        lambda exe=None: {"sites": ("naukri", "linkedin", "indeed"), "params": tuple()},
    )
    calls = _record_calls(monkeypatch)

    scrape_job_postings(["naukri"], "python", "Toronto, ON, Canada")

    assert calls, "the stubbed scraper was never invoked"
    assert calls[0] == ["naukri"]


def test_undecidable_location_keeps_every_site(monkeypatch):
    _stub_all_sites(monkeypatch)
    calls = _record_calls(monkeypatch)

    scrape_job_postings(["all"], "intern", "Anywhere Really")

    assert calls, "the stubbed scraper was never invoked"
    sent = calls[0]
    for site in ALL_SITES:
        assert site in sent
