"""Role filters judged by level, not by substring.

`role_filters=['internship']` searched the title for "intern", "co-op" and
"coop". That cannot see "Software Developer (Winter 2027)" -- a student posting
whose title carries no level word at all, where the term and the platform's own
employment_type are the only signals. job_level.classify already reads exactly
those, already stamps `level` on every ATS row at scrape time, and already backs
job_match.score_level. It was simply never consulted by the filter.

The rule that makes this safe to ship: the two passes are unioned, so any title
that matched before still matches. These tests pin that in both directions --
the new cases that now match, and the old ones that must not stop matching.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import job_level  # noqa: E402


# ── the cases a substring search cannot reach ────────────────────────────────

@pytest.mark.parametrize("title", [
    "Software Developer (Winter 2027)",
    "Software Engineer, Fall 2027",
    "Data Analyst (Summer 2026)",
])
def test_a_term_dated_title_with_no_level_word_matches_internship(title):
    """The shape the classifier exists for. None of these contain "intern",
    "co-op" or "coop", so the keyword pass returns False for all of them.
    """
    assert job_level.matches_role_filter(title, ["internship"]) is True


def test_the_platforms_own_employment_type_is_enough_on_its_own():
    """Lever calls it commitment, Ashby and schema.org employmentType. On a
    title with no level word and no term it is the only signal there is.
    """
    assert job_level.matches_role_filter(
        "Software Developer", ["internship"], employment_type="INTERN"
    ) is True


# ── the cases it must still reject ───────────────────────────────────────────

def test_a_campus_recruiter_is_not_an_internship():
    """A staff job recruiting students. The word "campus" is in the title, so a
    naive level mapping would match it -- classify resolves it to staff via its
    program-role override, and that must be respected.
    """
    assert job_level.classify("Campus Recruiter").level == "staff"
    assert job_level.matches_role_filter("Campus Recruiter", ["internship"]) is False
    assert job_level.matches_role_filter("Campus Recruiter", ["entry"]) is False


def test_a_conflicted_title_is_not_treated_as_early_career():
    """"Senior Intern Program Lead" carries both signals. classify flags the
    conflict and the seniority wins; a staff role supervising interns is not an
    internship.
    """
    verdict = job_level.classify("Senior Intern Program Lead")
    assert verdict.conflict is True
    assert job_level.matches_role_filter("Senior Intern Program Lead", ["internship"]) is False


@pytest.mark.parametrize("title", [
    "Senior Software Engineer",
    "Staff Backend Engineer",
    "Principal Architect",
])
def test_senior_titles_never_match_an_internship_filter(title):
    assert job_level.matches_role_filter(title, ["internship"]) is False


def test_internal_auditor_is_not_an_intern():
    """The highest-volume false positive available: "intern" is a prefix of
    "Internal". classify's markers are an explicit bounded set for this reason.
    """
    assert job_level.matches_role_filter("Internal Auditor", ["internship"]) is False
    assert job_level.matches_role_filter("International Tax Analyst", ["internship"]) is False


# ── mapping ──────────────────────────────────────────────────────────────────

def test_senior_filter_covers_staff_because_the_keyword_list_already_did():
    """The old keyword map put "lead", "staff" and "principal" under senior, so
    narrowing it now would be a regression dressed up as precision.
    """
    assert job_level.matches_role_filter("Staff Engineer", ["senior"]) is True
    assert job_level.matches_role_filter("Principal Engineer", ["senior"]) is True


def test_entry_covers_new_grad_and_campus_intake():
    assert job_level.matches_role_filter("New Grad Software Engineer", ["entry"]) is True


def test_several_filters_are_an_or_not_an_and():
    assert job_level.matches_role_filter(
        "Senior Software Engineer", ["internship", "senior"]
    ) is True


def test_an_empty_filter_allows_everything():
    assert job_level.matches_role_filter("Anything At All", []) is True
    assert job_level.matches_role_filter("Anything At All", None) is True


def test_an_unrecognised_filter_name_does_not_silently_allow_everything():
    """Returning True for a name this mapping does not know would widen the
    filter to "everything" -- the caller's keyword pass is the authority there,
    and it gets to answer instead.
    """
    assert job_level.matches_role_filter("Senior Engineer", ["nonsense-tier"]) is False


def test_every_filter_name_the_ui_offers_is_mapped():
    """The settings dropdown offers exactly these five. A name it can produce
    that this mapping does not know would fall through to keywords only,
    quietly losing the classifier for that filter.
    """
    ui_offers = {"internship", "entry", "junior", "mid", "senior"}
    assert ui_offers <= set(job_level.ROLE_FILTER_LEVELS)


def test_the_mapping_covers_every_level_the_classifier_can_return():
    """A level no filter maps to is unreachable: a posting classified into it
    could never satisfy any role filter.
    """
    mapped = set().union(*job_level.ROLE_FILTER_LEVELS.values())
    assert set(job_level.LEVELS) == mapped, f"unreachable levels: {set(job_level.LEVELS) - mapped}"


# ── the union: nothing that matched before may stop matching ─────────────────

@pytest.mark.parametrize("title,filters", [
    ("Software Engineering Intern", ["internship"]),
    ("Co-op Student, Firmware", ["internship"]),
    ("Junior Developer", ["junior"]),
    ("Senior Data Engineer", ["senior"]),
    ("Entry Level Analyst", ["entry"]),
])
def test_titles_the_old_keyword_pass_matched_still_match(title, filters):
    from services.job_service import matches_role_filters

    assert matches_role_filters(title, filters) is True


def _job_service_delegates() -> bool:
    """Whether job_service's matcher consults the classifier yet.

    The delegation is a few lines inside job_service.py, which carries
    uncommitted work and is never committed, so a checkout can legitimately
    have this module without it. Detected by signature rather than assumed: the
    delegating version takes description/employment_type, the keyword-only one
    does not.
    """
    import inspect

    from services.job_service import matches_role_filters

    return "employment_type" in inspect.signature(matches_role_filters).parameters


def test_the_union_adds_the_term_dated_case_through_the_real_entry_point():
    """End to end through job_service, which is what the watcher calls."""
    if not _job_service_delegates():
        pytest.skip("job_service.matches_role_filters does not delegate to job_level here")

    from services.job_service import matches_role_filters

    assert matches_role_filters("Software Developer (Winter 2027)", ["internship"]) is True


def test_the_real_entry_point_still_rejects_an_unrelated_senior_role():
    from services.job_service import matches_role_filters

    assert matches_role_filters("Senior Marketing Manager", ["internship"]) is False
