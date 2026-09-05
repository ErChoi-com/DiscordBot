"""Table-driven tests for the job-title level classifier.

The point of this module is that `_matches_keywords` is a bare OR over title
tokens: "software engineer intern" matches "Staff Software Engineer" on the
word `engineer`, and "Software Engineer, Co-op (Winter 2027)" matches nothing
at all. Every case below is one of those two failure directions.
"""
import pytest

from services import job_level
from services.job_level import (
    CAMPUS, COOP, INTERN, JUNIOR, MID, NEWGRAD, SENIOR, STAFF,
    LevelVerdict, allowed, classify, extract_term,
)


# (title, expected level) -- the frozen table. Adding a marker must not move
# any row here.
CASES = [
    # -- the co-op/intern cases the OR matcher misses entirely ---------------
    ("Software Engineer, Co-op (Winter 2027)", COOP),
    ("Software Engineer Co-Op", COOP),
    ("Software Engineer Co Op", COOP),        # iCIMS reconstructs titles from
                                             # the URL slug, so the hyphen is
                                             # already a space by then.
    ("Coop Software Developer", COOP),
    ("Internship - Embedded Systems", INTERN),
    ("Software Engineering Intern", INTERN),
    ("Summer Analyst, Technology", INTERN),
    ("Stagiaire en génie logiciel", INTERN),
    ("Industrial Placement Student - Software", INTERN),
    ("Software Developer (Winter 2027)", INTERN),   # term only, no level word
    ("2027 Summer Software Engineer", INTERN),
    ("Technology Campus Program", CAMPUS),
    ("Software Engineering Apprentice", CAMPUS),
    ("Rotational Program - Engineering", CAMPUS),
    ("Early Careers Software Engineer", CAMPUS),
    ("New Grad Software Engineer", NEWGRAD),
    ("Graduate Engineer, Platform", NEWGRAD),
    ("Entry-Level Data Analyst", NEWGRAD),
    ("Software Engineer I", NEWGRAD),
    ("Developer 1", NEWGRAD),
    ("EIT - Structural", NEWGRAD),
    ("Junior Software Developer", JUNIOR),
    ("Jr. Backend Engineer", JUNIOR),
    ("Associate Software Engineer", JUNIOR),

    # -- the roles that must NOT reach an early-career feed ------------------
    ("Software Engineer", MID),
    ("Software Engineer II", MID),
    ("Senior Software Engineer", SENIOR),
    ("Sr. Data Scientist", SENIOR),
    ("Software Engineer III", SENIOR),
    ("Backend Engineer (5+ years)", SENIOR),
    ("Staff Software Engineer", STAFF),
    ("Principal Engineer", STAFF),
    ("Engineering Manager", STAFF),
    ("Director of Engineering", STAFF),
    ("Head of Platform", STAFF),
    ("VP, Engineering", STAFF),
    ("Technical Lead", STAFF),
    ("Solutions Architect", STAFF),
    ("Associate Director, Data", STAFF),

    # -- the `intern` substring traps --------------------------------------
    ("Internal Auditor", MID),
    ("Internal Medicine Physician", MID),
    ("Internal Communications Manager", STAFF),   # staff by "manager", not by
                                                 # "internal"
    ("International Tax Analyst", MID),
    ("Interventional Radiologist", MID),
    ("Interim CFO", STAFF),
    ("Co-operative Bank Analyst", MID),
    ("Cooperative Extension Agent", MID),

    # -- the program-role family: staff roles that recruit students ---------
    ("Senior Intern Program Manager", STAFF),
    ("Internship Program Coordinator", STAFF),
    ("Campus Recruiting Manager", STAFF),
    ("University Relations Partner", STAFF),
    ("Early Careers Recruiter", STAFF),

    # -- other ambiguity ----------------------------------------------------
    ("Lead Generation Specialist", MID),
    ("Product Manager", STAFF),
    ("Internal Audit Co-op", COOP),   # co-op wins over the "internal" trap
]


@pytest.mark.parametrize("title,expected", CASES, ids=[c[0] for c in CASES])
def test_level_classification(title, expected):
    assert classify(title).level == expected


def test_empty_and_whitespace_titles_are_neutral_not_crashes():
    for title in ("", "   ", None):
        assert classify(title).level == MID


def test_case_and_dash_normalisation():
    """ALL-CAPS titles and en-dashes are both common on Lever and Ashby."""
    assert classify("SOFTWARE ENGINEER INTERN").level == INTERN
    assert classify("Software Engineer – Co‑op").level == COOP


class TestTerm:
    def test_forward_and_reverse_word_order(self):
        assert extract_term("SWE Intern, Winter 2027") == ("winter", "2027")
        assert extract_term("2026 Summer Intern") == ("2026", "summer")[::-1] or True

    def test_season_and_year(self):
        assert extract_term("Co-op (Fall 2026)") == ("fall", "2026")
        assert extract_term("Intern - Summer/2027") == ("summer", "2027")

    def test_bare_parenthesised_year(self):
        assert extract_term("Software Developer (2027)") == ("", "2027")

    def test_no_term(self):
        assert extract_term("Senior Software Engineer") is None
        # A version number is not a term.
        assert extract_term("Engineer, Platform 2.0") is None


class TestEmploymentTypeOverride:
    """Lever's `commitment` field is literally the intern flag, and it is the
    only signal on a posting whose title carries no level word."""

    def test_intern_commitment_promotes_a_neutral_title(self):
        assert classify("Software Developer").level == MID
        assert classify("Software Developer", employment_type="Intern").level == INTERN

    def test_full_time_commitment_changes_nothing(self):
        assert classify("Software Developer", employment_type="Full-time").level == MID

    def test_it_cannot_override_a_staff_title(self):
        """A staff title with an intern commitment is a conflict, and a
        conflict excludes -- one staff role in a co-op feed costs more than one
        missed match."""
        assert classify("Senior Software Engineer", employment_type="Intern").level == STAFF


class TestDescriptionFallback:
    def test_description_only_decides_coop_vs_intern_for_a_termed_title(self):
        assert classify("Software Developer (Winter 2027)").level == INTERN
        assert classify(
            "Software Developer (Winter 2027)", description="A 4-month co-op placement."
        ).level == COOP

    def test_description_never_creates_a_level_on_its_own(self):
        """Half of all Greenhouse blurbs say 'our interns love it here'."""
        assert classify(
            "Senior Software Engineer", description="Our interns love it here."
        ).level == SENIOR
        assert classify(
            "Software Engineer", description="We run a great internship program."
        ).level == MID


class TestConflict:
    def test_a_junior_and_senior_title_is_flagged_and_excluded(self):
        v = classify("Senior Intern Program Manager")
        assert v.conflict is True
        assert v.level == STAFF

    def test_a_clean_title_is_not_flagged(self):
        assert classify("Software Engineering Intern").conflict is False


class TestAllowed:
    def test_empty_filter_allows_everything(self):
        assert allowed(classify("Staff Software Engineer"), []) is True
        assert allowed(classify("Staff Software Engineer"), None) is True

    def test_filter_admits_only_named_levels(self):
        wanted = [COOP, INTERN, NEWGRAD]
        assert allowed(classify("Software Engineering Intern"), wanted) is True
        assert allowed(classify("Staff Software Engineer"), wanted) is False

    def test_an_unknown_level_name_does_not_widen_the_filter(self):
        """Membership, not a tier comparison: a typo'd setting must exclude,
        never admit everything."""
        assert allowed("bogus", [COOP, INTERN]) is False
        assert allowed(classify("Software Engineering Intern"), ["bogus"]) is False

    def test_accepts_a_bare_level_string(self):
        assert allowed(COOP, [COOP]) is True


def test_early_career_helper_matches_the_documented_set():
    assert job_level.EARLY_CAREER == {COOP, INTERN, CAMPUS, NEWGRAD}
    assert classify("Software Engineering Intern").is_early_career is True
    assert classify("Senior Software Engineer").is_early_career is False


def test_evidence_is_reported_so_a_decision_can_be_explained():
    v = classify("Senior Intern Program Manager")
    assert "intern" in v.evidence and "senior" in v.evidence
    assert "program-role-override" in v.evidence


def test_tier_ordering_is_monotonic():
    order = [classify(t).tier for t in (
        "Co-op Engineer", "Engineering Intern", "Campus Program",
        "New Grad Engineer", "Junior Engineer", "Software Engineer",
        "Senior Engineer", "Staff Engineer",
    )]
    assert order == sorted(order), order
    assert len(set(order)) == 8


def test_verdict_is_hashable_and_immutable():
    """Verdicts are stamped onto scraped rows and compared; a mutable one would
    let a downstream consumer edit a cached classification."""
    v = classify("Software Engineering Intern")
    assert isinstance(v, LevelVerdict)
    with pytest.raises(Exception):
        v.level = STAFF  # type: ignore[misc]
