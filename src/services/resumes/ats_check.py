"""Post-compile ATS readability audit for generated resume PDFs.

A LaTeX compile that exits 0 proves nothing about whether an applicant
tracking system can read the result.  The failure mode this module exists to
catch is a PDF that renders perfectly for a human but carries its glyphs as
Type 3 bitmaps with no ToUnicode map: text extraction returns glyph names
(``/x45/x72/x6E``) instead of characters, and the resume reaches the parser as
gibberish.

The audit deliberately uses pypdf rather than poppler.  poppler recovers
Type 3 text heuristically and would mask exactly the defect we care about;
pypdf's behaviour is closer to PDFBox, which is what a large share of real ATS
parse with.  Validating with the lenient reader would defeat the purpose.

Thresholds are calibrated against the seed template in ``resumes_cache/example``
(see ``EXAMPLE_PROFILE_KEY``).  That template is copied byte-identically into
every new profile, so it is the reference implementation: if the audit does not
score the seed's own output as ``ok``, the audit is wrong, not the template.
Notably the seed *does* emit a Type 3 font, and it is not decorative: it draws
the three reference URLs the template carries in its own guidance comments
(the r/EngineeringResumes checklist and friends), which is ~9% of the page's
drawn characters.  None of it is resume content.  That is precisely why Type 3
presence is measured as a share of rendered text rather than as a boolean -- a
boolean check would reject the reference implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

AtsStatus = Literal["ok", "degraded", "unreadable", "unknown"]

# Share of rendered characters drawn with Type 3 (bitmap) fonts.  The seed
# template scores 0.09 here (three non-content reference URLs), so the
# threshold must sit above that.  A large share means the body copy itself is
# unextractable; a small one means incidental glyphs outside the resume text.
TYPE3_DEGRADED_RATIO = 0.15
TYPE3_UNREADABLE_RATIO = 0.50

# A one-page resume that extracts to less than this is not being read at all.
# The seed extracts 2172 characters; xboxsignout's real builds extract ~5300.
MIN_EXTRACTED_CHARS = 600

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Accept the separators resumes actually use, including the Unicode dashes that
# fontspec/xelatex substitutes for an ASCII hyphen (see _normalize_for_match).
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
SECTION_HINTS = ("experience", "education", "skills", "project")

# Unicode dashes and spaces that break naive ATS regexes when substituted for
# their ASCII equivalents.  U+2010 is the one xelatex actually emits.
_DASH_TRANSLATION = {
    0x2010: "-",  # HYPHEN
    0x2011: "-",  # NON-BREAKING HYPHEN
    0x2012: "-",  # FIGURE DASH
    0x2013: "-",  # EN DASH
    0x2014: "-",  # EM DASH
    0x2212: "-",  # MINUS SIGN
    0x00A0: " ",  # NO-BREAK SPACE
    0x2007: " ",  # FIGURE SPACE
    0x202F: " ",  # NARROW NO-BREAK SPACE
}


@dataclass(slots=True)
class FontUsage:
    """One font resource as it is actually used to draw text."""

    name: str
    subtype: str
    has_tounicode: bool
    rendered_chars: int = 0

    @property
    def is_type3(self) -> bool:
        return self.subtype == "Type3"


@dataclass(slots=True)
class AtsAudit:
    status: AtsStatus
    findings: list[str] = field(default_factory=list)
    degradations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extracted_chars: int = 0
    page_count: int = 0
    type3_char_ratio: float = 0.0
    no_tounicode_char_ratio: float = 0.0
    found_email: bool = False
    found_phone: bool = False
    sections_found: list[str] = field(default_factory=list)
    fonts: list[FontUsage] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "unknown")

    def summary(self) -> str:
        head = f"ATS audit: {self.status}"
        if self.findings:
            head += " | " + "; ".join(self.findings)
        if self.degradations:
            head += " | degraded: " + "; ".join(self.degradations)
        if self.warnings:
            head += " | notes: " + "; ".join(self.warnings)
        return head


def _normalize_for_match(text: str) -> str:
    """Fold the Unicode punctuation ATS regexes trip over into ASCII.

    xelatex renders an ASCII hyphen as U+2010 HYPHEN, which makes a phone
    number like 647-713-8997 invisible to a ``\\d{3}-\\d{3}-\\d{4}`` matcher.
    Normalizing here means the audit reports on content, not on typography.
    """
    return text.translate(_DASH_TRANSLATION)


def _font_usage_for_page(page) -> dict[str, FontUsage]:
    """Inventory the font resources declared on one page."""
    usage: dict[str, FontUsage] = {}
    try:
        resources = page.get("/Resources")
        if resources is None:
            return usage
        fonts = resources.get_object().get("/Font")
        if fonts is None:
            return usage
        for name, ref in fonts.get_object().items():
            font = ref.get_object()
            subtype = str(font.get("/Subtype", "")).lstrip("/")
            usage[str(name)] = FontUsage(
                name=str(name),
                subtype=subtype,
                has_tounicode="/ToUnicode" in font,
            )
    except Exception:  # noqa: BLE001 - a malformed resource dict is not fatal
        return usage
    return usage


def _count_rendered_chars(page, usage: dict[str, FontUsage]) -> None:
    """Walk the content stream and attribute drawn characters to their font.

    Counting rendered characters (rather than merely listing fonts) is what
    lets a decorative Type 3 rule coexist with a clean bill of health.
    """
    try:
        from pypdf.generic import ContentStream
    except ImportError:  # pragma: no cover - pypdf always ships this
        return

    try:
        content = ContentStream(page.get_contents(), page.pdf)
    except Exception:  # noqa: BLE001 - unparseable stream => no attribution
        return

    current: FontUsage | None = None
    for operands, operator in content.operations:
        if operator == b"Tf" and operands:
            current = usage.get(str(operands[0]))
        elif operator in (b"Tj", b"'", b'"') and operands:
            if current is not None:
                payload = operands[-1]
                current.rendered_chars += len(payload) if payload else 0
        elif operator == b"TJ" and operands:
            if current is not None:
                for item in operands[0]:
                    if isinstance(item, (bytes, str)):
                        current.rendered_chars += len(item)


def audit_pdf_ats(
    pdf_bytes: bytes | None,
    *,
    expect_text: Iterable[str] = (),
) -> AtsAudit:
    """Judge whether an ATS can actually read ``pdf_bytes``.

    Returns ``status="unknown"`` for anything that cannot be parsed as a PDF.
    Callers must treat ``unknown`` as pass-through: the compile pipeline's own
    test suite drives it with stub bytes, and an audit that fails closed on
    unparseable input would reject those outright.
    """
    audit = AtsAudit(status="unknown")
    if not pdf_bytes:
        audit.findings.append("no PDF bytes produced")
        return audit

    try:
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = list(reader.pages)
    except Exception as exc:  # noqa: BLE001 - stubs and corrupt files land here
        audit.findings.append(f"unreadable as PDF ({type(exc).__name__})")
        return audit

    if not pages:
        audit.status = "unreadable"
        audit.findings.append("PDF contains no pages")
        return audit

    audit.page_count = len(pages)

    fonts: dict[str, FontUsage] = {}
    text_parts: list[str] = []
    for page in pages:
        usage = _font_usage_for_page(page)
        _count_rendered_chars(page, usage)
        for key, font in usage.items():
            merged = fonts.get(key)
            if merged is None or merged.subtype != font.subtype:
                fonts[f"{key}:{font.subtype}"] = font
            else:
                merged.rendered_chars += font.rendered_chars
        try:
            text_parts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - one bad page should not void the audit
            continue

    audit.fonts = sorted(fonts.values(), key=lambda f: -f.rendered_chars)

    total_drawn = sum(f.rendered_chars for f in audit.fonts)
    if total_drawn:
        type3_drawn = sum(f.rendered_chars for f in audit.fonts if f.is_type3)
        no_uni_drawn = sum(
            f.rendered_chars for f in audit.fonts if not f.has_tounicode
        )
        audit.type3_char_ratio = type3_drawn / total_drawn
        audit.no_tounicode_char_ratio = no_uni_drawn / total_drawn

    raw_text = "\n".join(text_parts)
    text = _normalize_for_match(raw_text)
    audit.extracted_chars = len(raw_text.strip())
    audit.found_email = bool(EMAIL_PATTERN.search(text))
    audit.found_phone = bool(PHONE_PATTERN.search(text))
    lowered = text.lower()
    audit.sections_found = [hint for hint in SECTION_HINTS if hint in lowered]

    # Glyph-name leakage is the signature of a Type 3 PDF read by a strict
    # parser: the extractor hands back "/x45/x72" instead of "Er".
    glyph_leak = len(re.findall(r"/x[0-9A-Fa-f]{2}", raw_text)) > 20

    findings: list[str] = []
    degradations: list[str] = []
    warnings: list[str] = []

    if glyph_leak:
        findings.append("text extracts as glyph names, not characters")
    if audit.type3_char_ratio >= TYPE3_UNREADABLE_RATIO:
        findings.append(
            f"{audit.type3_char_ratio:.0%} of text drawn with Type 3 bitmap fonts"
        )
    elif audit.type3_char_ratio > TYPE3_DEGRADED_RATIO:
        degradations.append(
            f"{audit.type3_char_ratio:.0%} of text drawn with Type 3 bitmap fonts"
        )
    if audit.no_tounicode_char_ratio >= TYPE3_UNREADABLE_RATIO:
        findings.append(
            f"{audit.no_tounicode_char_ratio:.0%} of text uses fonts with no ToUnicode map"
        )
    elif audit.no_tounicode_char_ratio > TYPE3_DEGRADED_RATIO:
        degradations.append(
            f"{audit.no_tounicode_char_ratio:.0%} of text uses fonts with no ToUnicode map"
        )

    if audit.extracted_chars < MIN_EXTRACTED_CHARS:
        findings.append(
            f"only {audit.extracted_chars} characters extract from {audit.page_count} page(s)"
        )
    if not audit.found_email:
        findings.append("no email address survives text extraction")

    # Phone is a warning, never a gate: the seed template ships without one and
    # most stored profiles have no phone number at all.
    if not audit.found_phone:
        warnings.append("no phone number found (seed template ships without one)")
    if len(audit.sections_found) < 2:
        warnings.append(
            f"few section headings extracted ({', '.join(audit.sections_found) or 'none'})"
        )

    for needle in expect_text:
        if needle and _normalize_for_match(needle).lower() not in lowered:
            findings.append(f"expected text missing from extraction: {needle!r}")

    audit.findings = findings
    audit.degradations = degradations
    audit.warnings = warnings
    # Only real degradation signals move the status.  Informational notes (no
    # phone number, sparse headings) must not downgrade a healthy PDF -- the
    # seed template ships with no phone, and downgrading it would make the
    # reference implementation fail its own audit.
    if findings:
        audit.status = "unreadable"
    elif degradations:
        audit.status = "degraded"
    else:
        audit.status = "ok"
    return audit
