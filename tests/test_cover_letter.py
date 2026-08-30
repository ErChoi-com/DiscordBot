"""Tests for the .resumecoverbuild pipeline (JD-driven cover letters built
from everything the tailored resume could not include)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.resumes.configkey import GeminiSettings
from services.resumes.cover import (
    _audit_cover_paragraphs,
    build_cover_inventory,
    build_cover_letter_prompt,
    build_legacy_inventory,
    deterministic_cover_paragraphs,
    extract_listing_points,
    generate_cover_letter,
    latex_escape,
    latex_to_plain,
    parse_candidate_identity,
    render_cover_letter_latex,
    validate_cover_paragraph,
)
from services.resumes.listing import JobContext, ScrapedJobPosting
from services.resumes.structured import StructuredSelection, load_structured_profile


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Keep selection cache, telemetry files, and provider cooldowns test-local,
    and make sure no test can fall through to a REAL provider key from .env."""
    import services.resumes.cover as cover_module
    import services.resumes.listing as listing_module

    monkeypatch.setattr(listing_module, "STRUCTURED_SELECTION_CACHE_DIR", tmp_path / "sel_cache")
    monkeypatch.setattr(listing_module, "STRUCTURED_TELEMETRY_PATH", tmp_path / "telemetry.jsonl")
    monkeypatch.setattr(listing_module, "SCRAPE_TELEMETRY_PATH", tmp_path / "scrape.jsonl")
    monkeypatch.setattr(cover_module, "COVER_TELEMETRY_PATH", tmp_path / "cover.jsonl")
    monkeypatch.setattr(listing_module, "_openrouter_api_key", lambda: None)
    monkeypatch.setattr(listing_module, "_groq_api_key", lambda: None)
    listing_module._PROVIDER_COOLDOWN_UNTIL.clear()


PROFILE_DIR = Path(__file__).resolve().parents[1] / "src" / "services" / "resumes" / "resumes_cache" / "xboxsignout._"

# resumes_cache/ is gitignored: it holds Ernest's real profile (template.tex,
# baseinfo.txt), which is personal data and deliberately not in the repo. These
# tests read it at import time, so on a clean checkout -- every CI runner -- the
# read raises during COLLECTION, and a collection error aborts the entire pytest
# run rather than just this file. Skip at module level instead.
if not (PROFILE_DIR / "template.tex").exists():
    pytest.skip(
        "resumes_cache profile data not present (gitignored); "
        "these tests require the local profile.",
        allow_module_level=True,
    )
TEMPLATE_PATH = PROFILE_DIR / "template.tex"
BASEINFO_PATH = PROFILE_DIR / "baseinfo.txt"

CATALOG = load_structured_profile(TEMPLATE_PATH, BASEINFO_PATH)
assert CATALOG is not None

# A ranking that fills the page with Experience entries, pushing the embedded
# project (eebot) and most other Projects off the resume — the letter's
# material. Mirrors what a real ML-listing selection looks like.
ML_FLAVOURED_RANKING = [
    next(e.entry_id for e in CATALOG.entries if fragment in e.entry_id)
    for fragment in ("goopter", "markham", "mcg3d", "obotz")
]

KEYLESS_SETTINGS = GeminiSettings(api_key=None, model="gemini-3.5-flash")
GEMINI_SETTINGS = GeminiSettings(api_key="fake-key", model="gemini-3.5-flash")

JOB = JobContext(
    title="Machine Learning Engineer Intern",
    posting_url="https://example.com/job/123",
    apply_url=None,
    source_message="[Example] Machine Learning Engineer Intern\nhttps://example.com/job/123",
)


def _scraped(
    description: str = "Build ML pipelines with Python and Kafka. 12-month internship.",
    highlights: list[str] | None = None,
    company: str = "Example Co",
) -> ScrapedJobPosting:
    return ScrapedJobPosting(
        title="Machine Learning Engineer Intern",
        company=company,
        location="Toronto, ON",
        description=description,
        highlights=highlights
        if highlights is not None
        else ["Experience with Python data pipelines and model training is required"],
        source_url=JOB.posting_url,
    )


class _FakeGenAIClient:
    """Mimics google.genai Client.models.generate_content for the cover prompt."""

    def __init__(self, payload: str):
        outer = self

        class _Models:
            def generate_content(self, **kwargs):
                class _Response:
                    text = outer.payload

                return _Response()

        self.payload = payload
        self.models = _Models()


def _gemini_factory(payload: str):
    return lambda api_key: _FakeGenAIClient(payload)


# ---------------------------------------------------------------------------
# Inventory: the letter's material is exactly what the resume left out
# ---------------------------------------------------------------------------


def test_inventory_contains_hidden_and_excluded_but_not_visible() -> None:
    obotz = next(e for e in CATALOG.entries if "obotz" in e.entry_id)
    selection = StructuredSelection(exclusions=[obotz.entry_id])
    inventory = build_cover_inventory(CATALOG, selection, "deterministic")

    assert inventory.omitted, "the bullet budget must leave at least one entry off the page"
    omitted_titles = {entry.title for entry in inventory.omitted}
    assert not omitted_titles & set(inventory.visible_titles)
    # The exclusion is either honoured (entry becomes cover material) or
    # rolled back for budget (entry stays on the resume) — never lost.
    assert any("Obotz" in title for title in omitted_titles | set(inventory.visible_titles))


def test_inventory_carries_baseinfo_facts_for_omitted_entries() -> None:
    # An Experience-first ranking leaves eebot (embedded project, tagged
    # baseinfo block) off the page — it must arrive as cover material.
    selection = StructuredSelection(ranking=ML_FLAVOURED_RANKING)
    inventory = build_cover_inventory(CATALOG, selection, "deterministic")
    eebot = next((entry for entry in inventory.omitted if "eebot" in entry.title.lower()), None)
    assert eebot is not None
    assert eebot.bullets and all(isinstance(b, str) for b in eebot.bullets)


def test_legacy_inventory_uses_full_baseinfo() -> None:
    baseinfo = "Header facts\n\n- Built a thing with Python.\n- Shipped another thing.\n"
    inventory = build_legacy_inventory(baseinfo)
    assert inventory.selection_source == "legacy"
    assert inventory.omitted[0].bullets == ["Built a thing with Python.", "Shipped another thing."]


# ---------------------------------------------------------------------------
# JD-driven listing points
# ---------------------------------------------------------------------------


def test_extract_listing_points_prefers_highlights_then_keywords() -> None:
    scraped = _scraped(
        highlights=[
            "Experience with Python data pipelines and model training is required",
            "Strong communication skills and experience presenting to stakeholders",
        ]
    )
    points = extract_listing_points(scraped, ["kubernetes", "python"])
    assert points[0].startswith("Experience with Python")
    assert "kubernetes" in points  # keyword not already covered by a highlight
    assert "python" not in points  # already covered inside a highlight line


def test_prompt_enumerates_listing_requirements_and_separates_material() -> None:
    selection = StructuredSelection(ranking=ML_FLAVOURED_RANKING, keywords=["python"])
    inventory = build_cover_inventory(CATALOG, selection, "deterministic")
    scraped = _scraped()
    identity = parse_candidate_identity(BASEINFO_PATH.read_text(encoding="utf-8"))
    points = extract_listing_points(scraped, inventory.keywords)

    prompt = build_cover_letter_prompt(JOB, scraped, inventory, identity, points)

    assert "<listing_requirements>" in prompt
    assert "1. " in prompt
    assert "<omitted>" in prompt and "<on_resume>" in prompt
    for entry in inventory.omitted:
        assert entry.title in prompt
    for title in inventory.visible_titles:
        assert title in prompt
    assert "never restate" in prompt or "never fabricate" in prompt


# ---------------------------------------------------------------------------
# Paragraph validation
# ---------------------------------------------------------------------------


def test_validate_paragraph_rejects_ungrounded_number() -> None:
    grounding = "Reduced downtime by 41% at the City of Markham."
    cleaned, reason = validate_cover_paragraph(
        "I improved reliability by 99% across all systems and saved 41% too.",
        grounding,
    )
    assert cleaned is None
    assert "99" in reason


def test_validate_paragraph_allows_numbers_from_listing_or_resume() -> None:
    grounding = "12-month internship. Reduced downtime by 41%."
    cleaned, reason = validate_cover_paragraph(
        "During the 12-month term I can extend the 41% downtime reduction work I led before.",
        grounding,
    )
    assert cleaned is not None, reason


def test_validate_paragraph_rejects_markup() -> None:
    grounding = "anything"
    assert validate_cover_paragraph("Uses <b>HTML</b> tags in prose here today.", grounding)[0] is None
    assert validate_cover_paragraph("Contains \\textbf{latex} commands in prose today.", grounding)[0] is None


# ---------------------------------------------------------------------------
# Grounding audit (replaces the old static forbidden-terms list)
# ---------------------------------------------------------------------------


def test_audit_cover_paragraphs_flags_ungrounded_claim(monkeypatch) -> None:
    # Provider dispatch is centralized in listing._iter_provider_responses,
    # so the fake must be installed on the listing module.
    import services.resumes.listing as listing_module

    monkeypatch.setattr(
        listing_module,
        "_generate_with_gemini",
        lambda *a, **k: json.dumps(
            {"verdicts": [{"id": 0, "fabricated": True, "phrase": "Kubernetes"}]}
        ),
    )
    flagged, reasons = _audit_cover_paragraphs(
        GEMINI_SETTINGS,
        [
            "I have deep production Kubernetes experience running clusters at scale.",
            "I reduced downtime by 41% during my internship at the City of Markham.",
        ],
        grounding="Reduced downtime by 41% at the City of Markham.",
        skill_anchors=(),
        client_factory=None,
    )
    assert flagged == {0}
    assert any("paragraph 0" in reason for reason in reasons)


def test_audit_cover_paragraphs_fails_open_without_provider() -> None:
    flagged, reasons = _audit_cover_paragraphs(
        KEYLESS_SETTINGS,
        ["Some paragraph with no verifiable providers configured to judge it."],
        grounding="anything",
        skill_anchors=(),
        client_factory=None,
    )
    assert flagged == set()
    assert reasons == []


def test_audit_cover_paragraphs_skips_empty_list() -> None:
    assert _audit_cover_paragraphs(GEMINI_SETTINGS, [], "g", (), None) == (set(), [])


# ---------------------------------------------------------------------------
# Deterministic (zero-LLM) letter
# ---------------------------------------------------------------------------


def test_deterministic_letter_pairs_requirements_with_omitted_facts() -> None:
    selection = StructuredSelection(
        ranking=ML_FLAVOURED_RANKING, keywords=["python", "kafka"]
    )
    inventory = build_cover_inventory(CATALOG, selection, "deterministic")
    scraped = _scraped(
        highlights=["Experience with embedded firmware and microcontrollers is an asset"]
    )
    points = extract_listing_points(scraped, inventory.keywords)
    paragraphs = deterministic_cover_paragraphs(JOB, scraped, inventory, points)

    # The opener names what the letter ADDS rather than announcing itself;
    # "I am writing to apply" is the most-cited cover-letter cliche.
    assert not paragraphs[0].startswith("I am writing to apply")
    assert "could not fit" in paragraphs[0]
    assert "Example Co" in paragraphs[0]
    # The embedded-firmware ask must be answered by the hidden embedded entry.
    assert any("Where the posting asks for" in p for p in paragraphs)
    assert any("firmware" in p.lower() for p in paragraphs[1:])


def test_generate_with_no_providers_renders_deterministic_letter() -> None:
    result = generate_cover_letter(
        KEYLESS_SETTINGS,
        JOB,
        TEMPLATE_PATH,
        BASEINFO_PATH,
        scraper=lambda url: _scraped(),
    )
    assert result.status == "ok"
    assert result.summary is not None and result.summary["deterministic"] is True
    assert result.latex_document is not None
    assert "\\documentclass" in result.latex_document
    assert result.latex_document.count("\\end{document}") == 1
    assert "Ernest Choi" in result.latex_document


# ---------------------------------------------------------------------------
# Provider path
# ---------------------------------------------------------------------------


def _grounded_payload() -> str:
    return json.dumps(
        {
            "body_paragraphs": [
                "I am writing to apply for the Machine Learning Engineer Intern role at Example Co, where the posting's focus on Python data pipelines matches work I have already delivered.",
                "Beyond my resume, my embedded firmware work on the HCS12 microcontroller in Assembly and C demonstrates the hardware-adjacent depth the posting values.",
            ],
            "closing_sentence": "I would welcome the chance to discuss how this additional background supports the team.",
        }
    )


def test_generate_with_fake_gemini_uses_validated_paragraphs(tmp_path) -> None:
    result = generate_cover_letter(
        GEMINI_SETTINGS,
        JOB,
        TEMPLATE_PATH,
        BASEINFO_PATH,
        scraper=lambda url: _scraped(),
        client_factory=_gemini_factory(_grounded_payload()),
    )
    assert result.status == "ok"
    assert result.used_provider in ("gemini", "gemini-flash")
    assert result.summary is not None and result.summary["deterministic"] is False
    assert result.latex_document is not None
    assert "Example Co" in result.latex_document
    assert "HCS12" in result.latex_document
    assert "Dear Example Co Hiring Team," in result.latex_document


def test_generate_drops_invented_numbers_and_falls_back() -> None:
    bad_payload = json.dumps(
        {
            "body_paragraphs": [
                "I boosted model accuracy by 87% at Example Co which is not a real grounded figure at all.",
                "I cut costs 55% in ways never documented anywhere in my background material either.",
            ]
        }
    )
    result = generate_cover_letter(
        GEMINI_SETTINGS,
        JOB,
        TEMPLATE_PATH,
        BASEINFO_PATH,
        scraper=lambda url: _scraped(),
        client_factory=_gemini_factory(bad_payload),
    )
    assert result.status == "ok"
    # Both paragraphs die in validation -> deterministic letter, no fake numbers.
    assert result.summary is not None and result.summary["deterministic"] is True
    assert "87" not in (result.latex_document or "")
    assert "55" not in (result.latex_document or "")


def test_generate_pairs_with_cached_resume_selection() -> None:
    import services.resumes.listing as listing_module

    scraped = _scraped()
    # A cached selection for the same listing proves the cover reused the
    # resume build's decisions rather than re-deriving its own.
    raw = json.dumps({"keywords": ["react"], "ranking": ML_FLAVOURED_RANKING})
    cache_path = listing_module._selection_cache_path(
        JOB.posting_url, CATALOG.header_block, scraped.description
    )
    listing_module._store_cached_selection(cache_path, raw, "gemini")

    result = generate_cover_letter(
        KEYLESS_SETTINGS,
        JOB,
        TEMPLATE_PATH,
        BASEINFO_PATH,
        scraper=lambda url: scraped,
    )
    assert result.status == "ok"
    assert result.summary is not None
    assert result.summary["selection_source"] == "cache"


def test_generate_legacy_profile_without_tags(tmp_path) -> None:
    template = tmp_path / "template.tex"
    template.write_text(
        "\\documentclass{article}\\begin{document}Plain legacy resume\\end{document}",
        encoding="utf-8",
    )
    baseinfo = tmp_path / "baseinfo.txt"
    baseinfo.write_text(
        "- Name: Jane Legacy\n- Built data tooling with Python for reporting.\n",
        encoding="utf-8",
    )
    result = generate_cover_letter(
        KEYLESS_SETTINGS,
        JOB,
        template,
        baseinfo,
        scraper=lambda url: _scraped(),
    )
    assert result.status == "ok"
    assert result.summary is not None and result.summary["selection_source"] == "legacy"
    assert "Jane Legacy" in (result.latex_document or "")


def test_generate_scrape_failure_is_reported() -> None:
    def _boom(url: str):
        raise RuntimeError("blocked")

    result = generate_cover_letter(KEYLESS_SETTINGS, JOB, TEMPLATE_PATH, BASEINFO_PATH, scraper=_boom)
    assert result.status == "error"
    assert "Failed to scrape posting URL" in result.message


def test_cover_telemetry_line_appended(tmp_path) -> None:
    import services.resumes.cover as cover_module

    generate_cover_letter(
        KEYLESS_SETTINGS,
        JOB,
        TEMPLATE_PATH,
        BASEINFO_PATH,
        scraper=lambda url: _scraped(),
    )
    lines = cover_module.COVER_TELEMETRY_PATH.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[-1])
    assert record["provider"] == "deterministic"
    assert record["omitted"] >= 1


# ---------------------------------------------------------------------------
# LaTeX rendering safety
# ---------------------------------------------------------------------------


def test_latex_escape_and_plain_roundtrip() -> None:
    assert latex_escape("R&D 100% #1_x{y}~z^") == (
        "R\\&D 100\\% \\#1\\_x\\{y\\}\\textasciitilde{}z\\textasciicircum{}"
    )
    assert latex_to_plain("Built \\textbf{Kafka} pipelines with 41\\% gains \\\\") == (
        "Built Kafka pipelines with 41% gains"
    )


def test_render_letter_escapes_company_and_title() -> None:
    scraped = _scraped(company="R&D_Labs #1")
    latex = render_cover_letter_latex(
        {"name": "Ernest Choi", "email": "e@example.com", "phone": "", "links": ""},
        JOB,
        scraped,
        ["Paragraph one about the role with plenty of grounded detail included.",
         "Paragraph two with more grounded detail to satisfy the minimum length."],
    )
    assert "R\\&D\\_Labs \\#1" in latex
    assert "\\documentclass[11pt]{article}" in latex
    assert latex.count("\\begin{document}") == 1
    assert latex.count("\\end{document}") == 1


def test_parse_candidate_identity_from_real_profile() -> None:
    identity = parse_candidate_identity(
        BASEINFO_PATH.read_text(encoding="utf-8"), CATALOG.header_block
    )
    assert identity["name"] == "Ernest Choi"


def test_prompt_carries_ownership_scope_rule_and_baseinfo_facts() -> None:
    """Ownership/scope reinforcement (2026-07-16): the rules block instructs
    scope-first framing, and the populated baseinfo Notes reach the prompt as
    omitted-entry facts."""
    selection = StructuredSelection(ranking=ML_FLAVOURED_RANKING, keywords=["python"])
    inventory = build_cover_inventory(CATALOG, selection, "deterministic")
    scraped = _scraped()
    identity = parse_candidate_identity(BASEINFO_PATH.read_text(encoding="utf-8"))
    points = extract_listing_points(scraped, inventory.keywords)

    prompt = build_cover_letter_prompt(JOB, scraped, inventory, identity, points)

    assert "ownership or scope" in prompt
    assert "Never claim leadership or solo credit" in prompt
    # eebot is omitted under this ranking; its Notes travel as extra facts.
    eebot = next((e for e in inventory.omitted if "eebot" in e.title.lower()), None)
    assert eebot is not None
    assert "full firmware stack" in eebot.extra_facts
    assert "full firmware stack" in prompt


def test_prompt_carries_the_candidates_owned_skill_list() -> None:
    """Recall (2026-08-24): the writer used to see skill_anchors nowhere, so a
    tool the candidate genuinely owns could only be named if some omitted
    entry's bullets happened to mention it — the listing's central requirement
    went unsaid. The audit already refuses to flag these (grounding_audit_prompt
    takes the same list), so showing them costs no safety."""
    selection = StructuredSelection(ranking=ML_FLAVOURED_RANKING, keywords=["python"])
    inventory = build_cover_inventory(CATALOG, selection, "deterministic")
    scraped = _scraped()
    identity = parse_candidate_identity(BASEINFO_PATH.read_text(encoding="utf-8"))
    points = extract_listing_points(scraped, inventory.keywords)

    prompt = build_cover_letter_prompt(JOB, scraped, inventory, identity, points)

    assert inventory.skill_anchors, "fixture profile must declare SKILL ANCHORS"
    assert "<skills>" + chr(10) in prompt
    skills_block = prompt.split("<skills>", 1)[1].split("</skills>", 1)[0]
    for anchor in inventory.skill_anchors:
        assert anchor in skills_block
    # Precision half: <job> may supply the role and employer, never a toolset.
    assert "never the candidate's own toolset" in prompt
    assert "appears ONLY in <job> must not be claimed" in prompt


def test_prompt_omits_the_skills_block_when_the_profile_declares_no_anchors() -> None:
    """Legacy/untagged profiles carry no SKILL ANCHORS; an empty block would
    read as "the candidate owns nothing" and suppress truthful mentions."""
    inventory = build_legacy_inventory("Ernest Choi\n\n- Built a thing in Python.\n")
    scraped = _scraped()
    identity = parse_candidate_identity(BASEINFO_PATH.read_text(encoding="utf-8"))

    prompt = build_cover_letter_prompt(JOB, scraped, inventory, identity, ["python"])

    assert inventory.skill_anchors == ()
    # Rule 1 names the tag; what must be absent is the block itself.
    assert "<skills>" + chr(10) not in prompt


# ---------------------------------------------------------------------------
# Deterministic-fallback pairing (2026-08-25)
#
# 3 of 8 letters in a real run fell through to the deterministic path (Gemini
# 429s) and it paired unrelated things: a Paid Search posting asking for "SEM,
# SEO or Digital Media" was answered with a General-Purpose Processor (ALU)
# bullet, because _best_omitted_match accepted ANY single shared token and the
# two texts both contain "digital".
# ---------------------------------------------------------------------------

from services.resumes.cover import (
    CoverInventory,
    OmittedEntry,
    _best_omitted_match,
    _point_tokens,
    deterministic_cover_paragraphs,
)

_ALU = OmittedEntry(
    title="General-Purpose Processor (ALU)",
    bullets=[
        "Applied RTL design practice - timing analysis, modular integration, "
        "bitwise operations - to build and test digital logic."
    ],
)


def _inventory(*entries: OmittedEntry) -> CoverInventory:
    return CoverInventory(
        omitted=list(entries), visible_titles=[], selection_source="deterministic"
    )


def test_point_tokens_drop_generic_words() -> None:
    """Overlap on ordinary prose is not evidence of a match."""
    tokens = _point_tokens("Experience with data and the ability to work with teams")
    assert "experience" not in tokens
    assert "ability" not in tokens
    assert "teams" not in tokens


def test_unrelated_requirement_is_not_paired() -> None:
    """The exact live failure: one shared word ("digital") is not enough."""
    point = "0-1 years experience in SEM, SEO or Digital Media preferred."
    assert _best_omitted_match(point, _inventory(_ALU)) is None


def test_genuinely_related_requirement_still_pairs() -> None:
    point = "Experience with RTL design, timing analysis and digital logic verification."
    match = _best_omitted_match(point, _inventory(_ALU))
    assert match is not None
    entry, bullet = match
    assert entry.title.startswith("General-Purpose Processor")
    assert "RTL design" in bullet


def test_unmatched_points_are_skipped_not_forced(monkeypatch) -> None:
    """A letter covering fewer requirements beats one asserting a connection
    the reader can see is not there."""
    job = JobContext(
        title="Paid Search Intern",
        posting_url="https://example.com/job",
        apply_url="https://example.com/job",
        source_message="[LinkedIn] Paid Search Intern",
    )
    scraped = ScrapedJobPosting(
        title="Paid Search Intern",
        company="Example Agency",
        location="Toronto, ON",
        description="We need SEM, SEO and Digital Media campaign management.",
        highlights=[],
        source_url="https://example.com/job",
    )
    paragraphs = deterministic_cover_paragraphs(
        job,
        scraped,
        _inventory(_ALU),
        ["0-1 years experience in SEM, SEO or Digital Media preferred."],
    )
    body = " ".join(paragraphs)
    # The false pairing is gone: nothing claims the ALU work answers a
    # SEM/SEO requirement.
    assert "Where the posting asks for" not in body
    # Surfacing the material generically is fine and deliberate — the
    # "Beyond my resume" fallback asserts no relevance it cannot support.
    if "RTL design practice" in body:
        assert "Beyond my resume" in body


def test_self_regard_openers_are_swapped_not_rejected() -> None:
    """Every LLM letter in a measured run closed with "I am eager to ...".
    The closing sentence is otherwise fine, so the phrase is swapped; all of
    these are followed by an infinitive, so the replacement is grammatically
    interchangeable."""
    grounding = "Ernest built pipelines in Python and Kafka for the analytics team."
    for opener in ("I am eager to", "I am excited to", "I am thrilled to"):
        text, reason = validate_cover_paragraph(
            opener + " discuss how this background supports the goals in the posting.",
            grounding,
        )
        assert reason == "" and text is not None, opener
        assert text.startswith("I would welcome the chance to"), opener


def test_self_regard_swap_does_not_touch_ordinary_prose() -> None:
    """"The team was eager to adopt" is a fact about other people, not a
    self-assessment — the pattern is anchored to "I am"."""
    grounding = "Ernest built pipelines in Python and Kafka for the analytics team."
    original = (
        "The team was eager to adopt the new pipeline once it proved stable in test."
    )
    text, reason = validate_cover_paragraph(original, grounding)
    assert reason == ""
    assert text == original
