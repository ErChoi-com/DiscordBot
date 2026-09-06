"""Dead-slug TTL expiry.

ATS companies that 404/410 are cached as "dead" so later scrapes skip them.
DEAD_SLUG_TTL_DAYS exists so a company that comes back is eventually re-probed;
if the TTL never fires, the effective company list shrinks permanently.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from services import ats_service


def _is_recorded(platform: str, slug: str) -> bool:
    """Whether a dead mark exists at all.

    Distinct from ats_service._is_dead, which answers the narrower "skip this
    slug without a request" and reports False once a mark is older than
    DEAD_SLUG_RECHECK_DAYS. These TTL tests are about retention of the record.
    """
    ats_service._is_dead(platform, slug)   # force the lazy file load
    return slug in ats_service._dead_slug_dates.get(platform, {})


@pytest.fixture(autouse=True)
def _isolated_dead_slugs(tmp_path, monkeypatch):
    monkeypatch.setattr(ats_service, "_DEAD_SLUG_DIR", tmp_path)
    ats_service.clear_dead_slugs()
    yield
    ats_service.clear_dead_slugs()


def _write_dead_file(tmp_path, platform: str, slugs: dict[str, str]) -> None:
    (tmp_path / f"{platform}.json").write_text(json.dumps(slugs), encoding="utf-8")


def _days_ago(days: int) -> str:
    return (date.today() - timedelta(days=days)).isoformat()


def test_expired_slug_is_repurged_even_when_no_new_deaths_occurred(tmp_path):
    """The real-world case: a platform whose dead-slug file is old but which
    recorded no NEW deaths this run.

    flush_dead_slugs used to iterate only the *dirty* set, so a platform with no
    fresh deaths was never purged -- its expired entries survived forever and
    those companies were never re-probed.
    """
    stale = _days_ago(ats_service.DEAD_SLUG_TTL_DAYS + 10)
    _write_dead_file(tmp_path, ats_service.GREENHOUSE, {"revived-co": stale})

    # Load the file (as a scrape would) without marking anything newly dead.
    assert _is_recorded(ats_service.GREENHOUSE, "revived-co") is True
    assert ats_service._dead_slugs_dirty == set(), "no new deaths were recorded"

    ats_service.flush_dead_slugs()

    assert _is_recorded(ats_service.GREENHOUSE, "revived-co") is False, (
        "expired slug survived the flush, so this company is never re-probed"
    )


def test_expired_slug_removal_is_persisted_to_disk(tmp_path):
    """Purging in memory but not writing the file would resurrect the slug on
    the next process start."""
    stale = _days_ago(ats_service.DEAD_SLUG_TTL_DAYS + 10)
    _write_dead_file(tmp_path, ats_service.LEVER, {"revived-co": stale})

    ats_service._is_dead(ats_service.LEVER, "revived-co")
    ats_service.flush_dead_slugs()

    on_disk = json.loads((tmp_path / f"{ats_service.LEVER}.json").read_text(encoding="utf-8"))
    assert "revived-co" not in on_disk, "purge was not written back to disk"


def test_unexpired_slug_is_kept(tmp_path):
    """The TTL must not evict entries that are still within their window."""
    fresh = _days_ago(max(1, ats_service.DEAD_SLUG_TTL_DAYS - 10))
    _write_dead_file(tmp_path, ats_service.GREENHOUSE, {"still-dead": fresh})

    ats_service._is_dead(ats_service.GREENHOUSE, "still-dead")
    ats_service.flush_dead_slugs()

    assert _is_recorded(ats_service.GREENHOUSE, "still-dead") is True


def test_newly_marked_slug_is_saved(tmp_path):
    """The original behaviour must survive the fix: a fresh death is persisted."""
    ats_service._mark_dead(ats_service.ASHBY, "gone-co")
    ats_service.flush_dead_slugs()

    on_disk = json.loads((tmp_path / f"{ats_service.ASHBY}.json").read_text(encoding="utf-8"))
    assert "gone-co" in on_disk


def test_flush_survives_a_platform_with_no_lookup_set(tmp_path):
    """_purge_expired_dead_slugs indexes the fast-lookup dict. If a platform is
    present in the date map but absent from that dict, a bare index would raise
    and abort the whole flush -- losing every save queued behind it."""
    stale = _days_ago(ats_service.DEAD_SLUG_TTL_DAYS + 10)
    ats_service._dead_slug_dates[ats_service.ICIMS] = {"orphan-co": stale}
    ats_service._dead_slugs.pop(ats_service.ICIMS, None)   # lookup set missing

    ats_service._mark_dead(ats_service.WORKDAY, "gone-co")  # queued behind it

    ats_service.flush_dead_slugs()

    on_disk = json.loads((tmp_path / f"{ats_service.WORKDAY}.json").read_text(encoding="utf-8"))
    assert "gone-co" in on_disk, "an unrelated platform aborted the flush"


def test_flush_clears_the_dirty_set(tmp_path):
    ats_service._mark_dead(ats_service.GREENHOUSE, "gone-co")
    assert ats_service._dead_slugs_dirty

    ats_service.flush_dead_slugs()
    assert ats_service._dead_slugs_dirty == set()


# ---------------------------------------------------------------------------
# Company-list bootstrap
# ---------------------------------------------------------------------------

def test_missing_company_lists_warn_loudly(tmp_path, monkeypatch, capsys):
    """data/ats_companies is gitignored, so a fresh deployment has none. Silent
    failure here is indistinguishable from 'no jobs matched'."""
    monkeypatch.setattr(ats_service, "_JBA_DIR", tmp_path / "absent")
    # The harvest dir is unioned in by load_company_lists, so it has to be
    # absent too -- otherwise this reads the real repo's harvest and the
    # missing-upstream case never happens.
    monkeypatch.setattr(ats_service, "_HARVEST_DIR", tmp_path / "absent-harvest")
    monkeypatch.setattr(ats_service, "_HARVEST_CI_DIR", tmp_path / "absent-ci")
    ats_service.reload_company_lists()

    assert ats_service.load_company_lists() == {}
    out = capsys.readouterr().out
    assert "WARNING" in out and "sync_ats_companies.py" in out
    ats_service.reload_company_lists()


def test_present_company_lists_do_not_warn(tmp_path, monkeypatch, capsys):
    """The negative case -- otherwise the warning could fire unconditionally."""
    (tmp_path / f"{ats_service.GREENHOUSE}_companies.json").write_text(
        json.dumps(["acme", "globex"]), encoding="utf-8"
    )
    monkeypatch.setattr(ats_service, "_JBA_DIR", tmp_path)
    monkeypatch.setattr(ats_service, "_HARVEST_DIR", tmp_path / "absent-harvest")
    monkeypatch.setattr(ats_service, "_HARVEST_CI_DIR", tmp_path / "absent-ci")
    ats_service.reload_company_lists()

    loaded = ats_service.load_company_lists()
    assert loaded[ats_service.GREENHOUSE] == ["acme", "globex"]
    assert "WARNING" not in capsys.readouterr().out
    ats_service.reload_company_lists()


def test_missing_upstream_still_warns_when_harvest_is_present(tmp_path, monkeypatch, capsys):
    """The harvest must not mask a failed upstream sync.

    load_company_lists unions the local harvest on top of upstream, so once a
    harvest exists the merged cache is non-empty even when data/ats_companies is
    missing entirely. Keying the warning on the merged result would silence the
    one signal that a sync broke.
    """
    harvest = tmp_path / "harvest"
    harvest.mkdir()
    (harvest / f"{ats_service.GREENHOUSE}.json").write_text(
        json.dumps(["harvested-co"]), encoding="utf-8"
    )
    monkeypatch.setattr(ats_service, "_JBA_DIR", tmp_path / "absent")
    monkeypatch.setattr(ats_service, "_HARVEST_DIR", harvest)
    monkeypatch.setattr(ats_service, "_HARVEST_CI_DIR", tmp_path / "absent-ci")
    ats_service.reload_company_lists()

    loaded = ats_service.load_company_lists()
    assert loaded[ats_service.GREENHOUSE] == ["harvested-co"]
    out = capsys.readouterr().out
    assert "WARNING" in out and "sync_ats_companies.py" in out
    ats_service.reload_company_lists()


def test_harvest_is_unioned_without_duplicating_upstream(tmp_path, monkeypatch):
    harvest = tmp_path / "harvest"
    harvest.mkdir()
    (tmp_path / f"{ats_service.GREENHOUSE}_companies.json").write_text(
        json.dumps(["acme", "globex"]), encoding="utf-8"
    )
    (harvest / f"{ats_service.GREENHOUSE}.json").write_text(
        json.dumps(["globex", "newco"]), encoding="utf-8"
    )
    monkeypatch.setattr(ats_service, "_JBA_DIR", tmp_path)
    monkeypatch.setattr(ats_service, "_HARVEST_DIR", harvest)
    monkeypatch.setattr(ats_service, "_HARVEST_CI_DIR", tmp_path / "absent-ci")
    ats_service.reload_company_lists()

    loaded = ats_service.load_company_lists()
    assert loaded[ats_service.GREENHOUSE] == ["acme", "globex", "newco"]
    ats_service.reload_company_lists()


def test_corrupt_harvest_file_cannot_break_upstream(tmp_path, monkeypatch):
    """A bad harvest degrades to 'upstream only', never to an empty list."""
    harvest = tmp_path / "harvest"
    harvest.mkdir()
    (tmp_path / f"{ats_service.GREENHOUSE}_companies.json").write_text(
        json.dumps(["acme"]), encoding="utf-8"
    )
    (harvest / f"{ats_service.GREENHOUSE}.json").write_text("{ not json",
                                                            encoding="utf-8")
    monkeypatch.setattr(ats_service, "_JBA_DIR", tmp_path)
    monkeypatch.setattr(ats_service, "_HARVEST_DIR", harvest)
    monkeypatch.setattr(ats_service, "_HARVEST_CI_DIR", tmp_path / "absent-ci")
    ats_service.reload_company_lists()

    assert ats_service.load_company_lists()[ats_service.GREENHOUSE] == ["acme"]
    ats_service.reload_company_lists()


# ── the published harvest, which is a third source alongside the local one ──

def _three_sources(tmp_path, monkeypatch, upstream, local, published):
    """Wire all three company-list sources at once.

    Every source has to be redirected or the real 130k-slug fleet leaks in --
    which is exactly what happened to three tests here when the published
    harvest was added and only two were isolated.
    """
    lo, pu = tmp_path / "local", tmp_path / "published"
    lo.mkdir(); pu.mkdir()
    if upstream is not None:
        (tmp_path / f"{ats_service.GREENHOUSE}_companies.json").write_text(
            json.dumps(upstream), encoding="utf-8")
    if local is not None:
        (lo / f"{ats_service.GREENHOUSE}.json").write_text(json.dumps(local), encoding="utf-8")
    if published is not None:
        (pu / f"{ats_service.GREENHOUSE}.json").write_text(json.dumps(published), encoding="utf-8")
    monkeypatch.setattr(ats_service, "_JBA_DIR", tmp_path)
    monkeypatch.setattr(ats_service, "_HARVEST_DIR", lo)
    monkeypatch.setattr(ats_service, "_HARVEST_CI_DIR", pu)
    ats_service.reload_company_lists()
    return ats_service.load_company_lists().get(ats_service.GREENHOUSE, [])


def test_the_published_harvest_is_read_at_all(tmp_path, monkeypatch):
    """The CI workflow publishes company lists that nothing read for weeks.
    This is the assertion that the third source is actually loaded.
    """
    got = _three_sources(tmp_path, monkeypatch, ["acme"], None, ["ci-co"])
    assert got == ["acme", "ci-co"]
    ats_service.reload_company_lists()


def test_all_three_sources_are_unioned(tmp_path, monkeypatch):
    got = _three_sources(tmp_path, monkeypatch, ["acme"], ["local-co"], ["ci-co"])
    assert sorted(got) == ["acme", "ci-co", "local-co"]
    ats_service.reload_company_lists()


def test_the_published_harvest_cannot_remove_a_local_company(tmp_path, monkeypatch):
    """The reason the two directories are separate. Measured 2026-09-05, the
    local files held 14,278 slugs the published ones did not, so writing the
    published lists over them would have deleted companies the bot scrapes.
    """
    got = _three_sources(tmp_path, monkeypatch, None, ["only-local"], ["ci-co"])
    assert "only-local" in got
    ats_service.reload_company_lists()


def test_a_slug_in_both_harvests_is_not_duplicated(tmp_path, monkeypatch):
    got = _three_sources(tmp_path, monkeypatch, None, ["shared"], ["shared"])
    assert got == ["shared"]
    ats_service.reload_company_lists()


def test_a_corrupt_published_harvest_keeps_the_local_one(tmp_path, monkeypatch):
    lo, pu = tmp_path / "local", tmp_path / "published"
    lo.mkdir(); pu.mkdir()
    (lo / f"{ats_service.GREENHOUSE}.json").write_text(json.dumps(["local-co"]), encoding="utf-8")
    (pu / f"{ats_service.GREENHOUSE}.json").write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(ats_service, "_JBA_DIR", tmp_path / "absent")
    monkeypatch.setattr(ats_service, "_HARVEST_DIR", lo)
    monkeypatch.setattr(ats_service, "_HARVEST_CI_DIR", pu)
    ats_service.reload_company_lists()

    assert ats_service.load_company_lists()[ats_service.GREENHOUSE] == ["local-co"]
    ats_service.reload_company_lists()


def test_an_absent_published_harvest_changes_nothing(tmp_path, monkeypatch):
    """A checkout that has never synced must behave exactly as before."""
    got = _three_sources(tmp_path, monkeypatch, ["acme"], ["local-co"], None)
    assert got == ["acme", "local-co"]
    ats_service.reload_company_lists()
