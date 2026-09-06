"""Three fields, three consequences, and for a while one threshold.

check_platform_yield tracks description, location and date_posted, and its own
FIELD_CONSEQUENCE table says what each blank field costs -- one degrades a job,
one deletes it from a search, and one exempts it from a filter so a stale
posting is shown as fresh. All three were then rated against a single 20% floor
named --min-description-pct, which was the description figure and never a
judgement about the other two.

It missed a real finding. Ashby carries date_posted on 22% of its postings, so
78% of them return True from _posting_age_ok and skip the "newer than N hours"
filter entirely -- and 22% cleared the shared floor by two points. Nothing lost,
nothing to notice, which is exactly why a field whose failure adds rather than
removes needs its own number.

The floors here are measured, not chosen: every platform except icims carries
location on 100% of its jobs, description sits between 19% and 100% across the
fleet because most ATS list endpoints return a title and a link and nothing
else, and date_posted is 100% everywhere but ashby.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_platform_yield as y  # noqa: E402


# -- the floors differ, on purpose --------------------------------------------

def test_every_tracked_field_has_a_floor():
    """A tracked field with no floor is one nobody decided about."""
    assert set(y.FIELD_FLOORS) == set(y.TRACKED_FIELDS)


def test_the_floors_are_not_all_the_same():
    """The whole point. One number for three consequences is what let ashby
    through."""
    assert len(set(y.FIELD_FLOORS.values())) > 1


def test_location_is_held_to_a_higher_standard_than_description():
    """A blank description degrades a job's ranking; a blank location removes
    it from every location-scoped search. The second is worse."""
    assert y.FIELD_FLOORS["location"] > y.FIELD_FLOORS["description"]


def test_date_posted_is_held_to_a_higher_standard_than_description():
    """A blank date is not a missing job, it is a stale job shown as fresh --
    invisible to anyone looking for what is absent."""
    assert y.FIELD_FLOORS["date_posted"] > y.FIELD_FLOORS["description"]


def test_the_description_floor_does_not_condemn_the_whole_fleet():
    """Measured: greenhouse 25%, lever 26%, ashby 22%. That band is the ATS
    list endpoints, not this repo, and a floor above it would report most
    platforms as broken for behaving the way the APIs behave.
    """
    assert y.FIELD_FLOORS["description"] <= 22.0


# -- what the floors actually catch -------------------------------------------

def _thin(field, pct, override=None):
    return pct < y.floor_for(field, override)


def test_ashbys_date_coverage_is_now_reported():
    """The finding the shared floor missed by two points."""
    assert _thin("date_posted", 22.0)


def test_ashbys_date_coverage_passed_the_old_shared_floor():
    """Recorded so the regression is legible: this is what was being missed,
    not a number picked to make a test pass."""
    assert 22.0 >= y.MIN_DESCRIPTION_PCT


def test_a_platform_losing_a_tenth_of_its_locations_is_named():
    """Every platform but icims is at 100%, so a real slip is visible long
    before it becomes a fifth of the jobs."""
    assert _thin("location", 89.0)
    assert not _thin("location", 91.0)


def test_a_normal_platform_trips_nothing():
    for field in y.TRACKED_FIELDS:
        assert not _thin(field, 100.0), field


def test_icims_trips_both_of_its_fields():
    """19% on both, and they are different failures: invisible to semantic
    matching, and dropped from location search outright."""
    assert _thin("description", 19.0)
    assert _thin("location", 19.0)


# -- the override --------------------------------------------------------------

def test_one_number_overrides_every_field():
    for field in y.TRACKED_FIELDS:
        assert y.floor_for(field, 50.0) == 50.0


def test_an_override_of_zero_is_honoured_rather_than_ignored():
    """`or`-style defaulting would read 0.0 as "not given" and quietly restore
    the floors, which is the opposite of what asking for zero means."""
    assert y.floor_for("location", 0.0) == 0.0
    assert not _thin("location", 1.0, override=0.0)


def test_no_override_leaves_each_field_on_its_own_floor():
    assert y.floor_for("location") == y.FIELD_FLOORS["location"]
    assert y.floor_for("date_posted") == y.FIELD_FLOORS["date_posted"]


def test_an_unknown_field_is_reported_not_exempted():
    """A newly tracked field with no floor yet should be held to the
    description floor until someone measures it. Falling back to zero would
    make adding a field the way to stop it ever being reported.
    """
    assert y.floor_for("brand_new_field") == y.MIN_DESCRIPTION_PCT
    assert _thin("brand_new_field", 1.0)


# -- the report says which floor it used --------------------------------------

@pytest.fixture
def archive(tmp_path, monkeypatch):
    """One platform, plenty of confirmed-live companies, thin on date only."""
    records = [{"_source_site": "ashby", "description": "d", "location": "Toronto",
                "date_posted": "2026-09-01" if i < 2 else ""} for i in range(10)]
    checked = tmp_path / "checked"
    checked.mkdir()
    (checked / "ashby.json").write_text(
        '{' + ",".join(f'"c{i}":"2026-09-01"' for i in range(500)) + '}',
        encoding="utf-8")
    monkeypatch.setattr(y, "CHECKED_DIR", checked)
    monkeypatch.setattr(y, "_candidate_loader", lambda: (lambda _p: []))

    import services.ats_service as ats
    monkeypatch.setattr(ats, "ATS_PLATFORMS", ["ashby"], raising=False)
    monkeypatch.setattr("services.jba.merge_data.load_daily_log",
                        lambda _d: records)
    return records


def test_the_report_names_the_floor_it_judged_against(archive, capsys):
    """"Thin date_posted: ashby (22%)" is unreadable without the number it is
    thin against, especially now that the three numbers differ.
    """
    y.main(["--days", "1", "--min-live", "100"])
    out = capsys.readouterr().out
    assert "Thin date_posted (under 80%)" in out
    assert "ashby (20%)" in out


def test_the_override_is_reflected_in_what_the_report_claims(archive, capsys):
    y.main(["--days", "1", "--min-live", "100", "--min-field-pct", "10"])
    out = capsys.readouterr().out
    assert "Thin date_posted" not in out


def test_a_thin_field_is_still_not_fatal(archive, capsys):
    """Advisory. A silent platform is broken; a thin field is still shipping
    jobs, and collapsing the two makes the exit status useless for gating.
    """
    assert y.main(["--days", "1", "--min-live", "100"]) == 0
    assert "Thin date_posted" in capsys.readouterr().out
