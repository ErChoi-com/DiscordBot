"""services.resumes.posting_segments: recovering pieces from flattened postings.

Every case is a failure seen while tuning against real postings (see
scripts/segment_eval.py), in both directions: a break that must be found, and a
cut that must not be made.
"""
from __future__ import annotations

import pytest

from services.resumes.posting_segments import split_description


def _has_piece(pieces: list[str], text: str) -> bool:
    return text in pieces


@pytest.mark.parametrize("flat, expected_pieces", [
    # Glued bullets with no punctuation, split on a line-starting word.
    ("Experience with drawings for manufacturing and an understanding of GD&T Experience with 3D printing is an asset",
     ["Experience with drawings for manufacturing and an understanding of GD&T",
      "Experience with 3D printing is an asset"]),
    ("Work from the comfort of your home Great experience for teachers",
     ["Work from the comfort of your home", "Great experience for teachers"]),
    # An ALL CAPS section header between sentences.
    ("Apply below. ABOUT THE ROLE As a Product Lead you will own the roadmap.",
     ["Apply below.", "ABOUT THE ROLE", "As a Product Lead you will own the roadmap."]),
    # A short heading closed by a colon.
    ("Key Responsibilities: Coordinate candidate interviews.",
     ["Key Responsibilities", "Coordinate candidate interviews."]),
    # Real line breaks always win.
    ("Line one without punctuation\nLine two", ["Line one without punctuation", "Line two"]),
    ("Remote · Full-Time Contractor · Philippines-Based",
     ["Remote", "Full-Time Contractor", "Philippines-Based"]),
])
def test_breaks_are_found(flat, expected_pieces):
    assert split_description(flat) == expected_pieces


@pytest.mark.parametrize("flat, must_stay_whole", [
    # Common words are not headers in running text (was 87% of bad cuts).
    ("You have 5+ years of experience in software development.", "You have 5+ years of experience in software development."),
    ("Strong Excel skills (SQL is a plus).", "Strong Excel skills (SQL is a plus)."),
    # A header phrase in lowercase is running text, not a header.
    ("Read more about us At Acme we build tools.", "Read more about us At Acme we build tools."),
    # Title Case phrases are not line breaks.
    ("We are an Equal Employment Opportunity employer.", "We are an Equal Employment Opportunity employer."),
    ("We have a generous Paid Time Off policy.", "We have a generous Paid Time Off policy."),
    # A determiner or preposition continues its phrase.
    ("Currently pursuing a Bachelor's degree in Computer Science.", "Currently pursuing a Bachelor's degree in Computer Science."),
    ("You bring 3+ Years of Experience in Product Management.", "You bring 3+ Years of Experience in Product Management."),
    # A field label keeps its value.
    ("Location: Colindale", "Location: Colindale"),
    # A multi-word ALL CAPS header is one piece, not cut at each inner word.
    ("REQUIREMENTS AND PREFERRED EXPERIENCE", "REQUIREMENTS AND PREFERRED EXPERIENCE"),
    # "/" joins a job-title acronym to its role.
    ("Work under the direction of an A/R Lead and provide support.",
     "Work under the direction of an A/R Lead and provide support."),
    # A multi-word label is split before its first word, never inside it.
    ("Scale the portfolio. Deep Subject Matter Expertise: Develop and share expertise.",
     "Deep Subject Matter Expertise"),
])
def test_cuts_are_not_made(flat, must_stay_whole):
    assert _has_piece(split_description(flat), must_stay_whole)


@pytest.mark.parametrize("flat, first", [
    ("Experience managing projects across cross functional teams Amazon is an equal opportunity employer and does not discriminate.",
     "Experience managing projects across cross functional teams"),
    ("Work closely with Business Analysts throughout the testing lifecycle. *We may use artificial intelligence tools to assist with screening.",
     "Work closely with Business Analysts throughout the testing lifecycle."),
    ('Passion for innovation and technology "We use AI in our Hiring Process" Mindlance is an equal opportunity employer.',
     "Passion for innovation and technology"),
])
def test_legal_notice_glued_to_a_requirement_is_split_off(flat, first):
    assert split_description(flat)[0] == first


def test_equal_opportunity_in_running_text_is_not_split():
    text = "We believe in equal opportunity for everyone we hire."
    assert split_description(text) == [text]


def test_slash_list_before_a_starter_still_breaks():
    pieces = split_description("Outgoing/friendly/patient Detail focused and results-oriented")
    assert pieces == ["Outgoing/friendly/patient", "Detail focused and results-oriented"]


def test_empty_and_punctuation_only_input():
    assert split_description("") == []
    assert split_description(" - • : ") == []
