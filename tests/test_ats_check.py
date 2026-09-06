"""ATS readability audit tests.

These run against real PDFs committed under ``tests/fixtures/ats/``, each
produced by an actual LaTeX engine, because the defect being guarded against is
invisible in the LaTeX source and only exists in the compiled bytes.  Mocking
the PDF would test nothing.

Fixture provenance:
  seed_pdflatex_ok.pdf            resumes_cache/example compiled with pdflatex
  fontspec_xelatex_ok.pdf         a fontspec template compiled with xelatex
  font_pkg_stripped_ok.pdf        rickydisappoints after sourcesanspro is dropped
  cross_family_xelatex_type3.pdf  a pdfTeX-family template forced through xelatex
"""

from __future__ import annotations

from pathlib import Path

import pytest

from services.resumes.ats_check import (
    TYPE3_DEGRADED_RATIO,
    audit_pdf_ats,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ats"

SEED = FIXTURES / "seed_pdflatex_ok.pdf"
FONTSPEC = FIXTURES / "fontspec_xelatex_ok.pdf"
STRIPPED = FIXTURES / "font_pkg_stripped_ok.pdf"
CROSS_FAMILY = FIXTURES / "cross_family_xelatex_type3.pdf"


def _audit(path: Path):
    return audit_pdf_ats(path.read_bytes())


# ── the reference implementation must pass its own audit ─────────────────────


def test_seed_template_output_is_ok() -> None:
    """The seed is copied byte-identically into every new profile.

    If the audit does not score the seed's own output as ok, the audit is wrong
    -- every user's first build would be flagged.
    """
    audit = _audit(SEED)
    assert audit.status == "ok", audit.summary()
    assert audit.findings == []
    assert audit.found_email is True


def test_seed_carries_a_harmless_type3_font_below_threshold() -> None:
    """The seed emits Type 3 glyphs for its own reference URLs, not for content.

    This is the reason Type 3 is measured as a share of drawn characters rather
    than as a boolean: a presence check would reject the reference resume.
    """
    audit = _audit(SEED)
    assert any(font.is_type3 for font in audit.fonts)
    assert 0 < audit.type3_char_ratio <= TYPE3_DEGRADED_RATIO


def test_seed_has_no_phone_and_is_still_ok() -> None:
    """Most stored profiles ship without a phone number, the seed included."""
    audit = _audit(SEED)
    assert audit.found_phone is False
    assert any("phone" in warning for warning in audit.warnings)
    assert audit.status == "ok"


# ── the defect this module exists to catch ───────────────────────────────────


def test_cross_family_xelatex_output_is_unreadable() -> None:
    """A pdfTeX template forced through xelatex renders as Type 3 bitmaps.

    The PDF is visually perfect; its text extracts as glyph names.
    """
    audit = _audit(CROSS_FAMILY)
    assert audit.status == "unreadable"
    assert audit.type3_char_ratio > 0.9
    assert audit.no_tounicode_char_ratio > 0.9
    assert audit.found_email is False
    assert any("glyph names" in finding for finding in audit.findings)


# ── healthy output from both engine families ─────────────────────────────────


@pytest.mark.parametrize("fixture", [FONTSPEC, STRIPPED], ids=["fontspec", "stripped"])
def test_healthy_outputs_pass(fixture: Path) -> None:
    audit = _audit(fixture)
    assert audit.status == "ok", audit.summary()
    assert audit.found_email is True
    # Not zero: the stripped fixture carries a single stray Type 3 glyph
    # (~0.09% of drawn characters).  The contract is "below the threshold",
    # which is the whole reason this is a ratio and not a boolean.
    assert audit.type3_char_ratio <= TYPE3_DEGRADED_RATIO


def test_fontspec_phone_survives_unicode_hyphen_normalization() -> None:
    """xelatex renders an ASCII hyphen as U+2010, breaking naive phone regexes.

    The source is ASCII; the substitution happens in the font's ToUnicode map,
    so it cannot be fixed upstream.  The audit normalizes before matching so it
    reports on content rather than on typography.
    """
    audit = _audit(FONTSPEC)
    assert audit.found_phone is True


# ── pass-through behaviour ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "payload", [b"", None, b"%PDF-1.4 not really a pdf"], ids=["empty", "none", "stub"]
)
def test_unparseable_input_is_unknown_not_failure(payload) -> None:
    """The compile suite drives the pipeline with stub bytes.

    An audit that failed closed on unparseable input would reject those builds
    outright, so ``unknown`` must never be treated as a failure.
    """
    audit = audit_pdf_ats(payload)
    assert audit.status == "unknown"
    assert audit.ok is True


def test_expect_text_reports_missing_content() -> None:
    audit = audit_pdf_ats(SEED.read_bytes(), expect_text=["John Doe"])
    assert audit.status == "ok"

    missing = audit_pdf_ats(SEED.read_bytes(), expect_text=["Nonexistent Person"])
    assert missing.status == "unreadable"
    assert any("expected text missing" in finding for finding in missing.findings)
