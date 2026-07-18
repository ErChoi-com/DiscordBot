"""Tests for the structured resume pipeline (JSON decisions -> deterministic LaTeX)."""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.resumes.configkey import GeminiSettings
from services.resumes.listing import (
    ScrapedJobPosting,
    extract_job_context_from_message,
    generate_resume_rewrite,
)
from services.resumes.structured import (
    MAX_VISIBLE_BULLETS,
    MIN_VISIBLE_BULLETS,
    StructuredSelection,
    _bold_jd_tools,
    _extract_jd_tools,
    _inject_jd_tools_into_skills,
    _SKIP_JD_INJECT,
    build_structured_prompt,
    enforce_cross_bullet_consistency,
    extract_json_object,
    load_structured_profile,
    parse_structured_response,
    parse_template_catalog,
    render_structured_resume,
    validate_tailored_bullet,
)

@pytest.fixture(autouse=True)
def _isolate_structured_state(tmp_path, monkeypatch):
    """Keep selection cache, telemetry, and provider cooldowns test-local."""
    import services.resumes.listing as listing_module

    monkeypatch.setattr(listing_module, "STRUCTURED_SELECTION_CACHE_DIR", tmp_path / "sel_cache")
    monkeypatch.setattr(listing_module, "STRUCTURED_TELEMETRY_PATH", tmp_path / "telemetry.jsonl")
    listing_module._PROVIDER_COOLDOWN_UNTIL.clear()


PROFILE_DIR = Path(__file__).resolve().parents[1] / "src" / "services" / "resumes" / "resumes_cache" / "xboxsignout._"
TEMPLATE_TEXT = (PROFILE_DIR / "template.tex").read_text(encoding="utf-8")
BASEINFO_TEXT = (PROFILE_DIR / "baseinfo.txt").read_text(encoding="utf-8")

# Load via load_structured_profile so the profile's structured_config.json
# (budget, adjacent-tool leeway) applies, exactly as in production.
CATALOG = load_structured_profile(PROFILE_DIR / "template.tex", PROFILE_DIR / "baseinfo.txt")
assert CATALOG is not None


def _entry_id(fragment: str) -> str:
    assert CATALOG is not None
    return next(e.entry_id for e in CATALOG.entries if fragment in e.entry_id)


def _scraped(description: str, title: str = "Software Engineer") -> ScrapedJobPosting:
    return ScrapedJobPosting(
        title=title,
        company="Example Co",
        location="Remote",
        description=description,
        highlights=[],
        source_url="https://example.com/job",
    )


# ---------------------------------------------------------------------------
# Catalog parsing
# ---------------------------------------------------------------------------

def test_catalog_parses_all_tagged_entries() -> None:
    assert CATALOG is not None
    assert len(CATALOG.entries) == 12
    assert CATALOG.entry_sections == ("Projects", "Experience")
    projects = [e for e in CATALOG.entries if e.section == "Projects"]
    experience = [e for e in CATALOG.entries if e.section == "Experience"]
    assert len(projects) == 7
    assert len(experience) == 5


def test_catalog_preserves_headers_verbatim() -> None:
    assert CATALOG is not None
    pfs = next(e for e in CATALOG.entries if "particle" in e.entry_id)
    assert "\\hfill \\href{https://github.com/ErChoi-com/Particle-Fluid-Simulation}" in pfs.header
    eebot = next(e for e in CATALOG.entries if "eebot" in e.entry_id)
    assert "% no public repository" in eebot.header


def test_catalog_tail_keeps_skills_and_education() -> None:
    assert CATALOG is not None
    assert "\\section*{Skills}" in CATALOG.tail
    assert "\\section*{Education}" in CATALOG.tail
    assert "Toronto Metropolitan University" in CATALOG.tail


def test_catalog_returns_none_for_untagged_template() -> None:
    plain = "\\documentclass{article}\\begin{document}Hello\\end{document}"
    assert parse_template_catalog(plain) is None


# ---------------------------------------------------------------------------
# Bullet validation
# ---------------------------------------------------------------------------

CANONICAL = "Used \\textbf{LaTeX} to maintain documentation, reducing downtime by 41\\%."


def test_validate_bullet_accepts_clean_rewrite() -> None:
    text, reason = validate_tailored_bullet(
        "Maintained \\textbf{LaTeX} process documentation, reducing downtime by 41\\%.",
        CANONICAL,
    )
    assert reason is None
    assert text is not None and "41\\%" in text


def test_validate_bullet_rejects_html() -> None:
    text, reason = validate_tailored_bullet("Great bullet </ul>", CANONICAL)
    assert text is None and reason == "html/angle brackets"


def test_validate_bullet_rejects_structural_latex() -> None:
    text, reason = validate_tailored_bullet(
        "\\begin{itemize} nested 41 \\end{itemize}", CANONICAL
    )
    assert text is None and "disallowed latex command" in str(reason)


def test_validate_bullet_allows_dropping_metric() -> None:
    """Policy (2026-07-04): a rewrite may drop a metric; it may not invent one."""
    text, reason = validate_tailored_bullet(
        "Maintained documentation efficiently.", CANONICAL
    )
    assert reason is None and text is not None


def test_validate_bullet_allows_metric_moved_within_entry() -> None:
    entry_context = CANONICAL + " Performed testing, reducing errors by 20\\%."
    text, reason = validate_tailored_bullet(
        "Maintained \\textbf{LaTeX} docs, cutting downtime 41\\% and errors 20\\%.",
        CANONICAL,
        entry_context=entry_context,
    )
    assert reason is None


def test_validate_bullet_rejects_ungrounded_unbolded_tool_without_leeway() -> None:
    """A model that drops \\textbf{} must not smuggle tools past grounding on
    a conservative profile (observed: 'State Management (Riverpod)' written
    into a React bullet)."""
    text, reason = validate_tailored_bullet(
        "Maintained documentation applying State Management (Riverpod) "
        "principles, reducing downtime by 41\\%.",
        CANONICAL,
        skill_anchors=("latex", "git"),
    )
    assert text is None and "ungrounded tool token" in str(reason)


def test_validate_bullet_leeway_allows_unbolded_inserted_tool() -> None:
    """Writing-quality policy (2026-07-11): profiles with adjacent_tool_leeway
    keep unbolded skill insertions — the LLM grounding audit, not the
    deterministic guard, arbitrates claims that outright don't make sense."""
    assert CATALOG is not None and CATALOG.render_config.adjacent_tool_leeway
    text, reason = validate_tailored_bullet(
        "Maintained documentation applying State Management (Riverpod) "
        "principles, reducing downtime by 41\\%.",
        CANONICAL,
        CATALOG.render_config,
        skill_anchors=("latex", "git"),
    )
    assert reason is None and text is not None


def test_validate_bullet_allows_unbolded_tool_from_entry_grounding() -> None:
    entry_context = CANONICAL + " Built dashboards with \\textbf{Tableau}."
    text, reason = validate_tailored_bullet(
        "Maintained documentation and Tableau dashboards, reducing downtime "
        "by 41\\%.",
        CANONICAL,
        entry_context=entry_context,
    )
    assert reason is None and text is not None


def test_validate_bullet_restores_canonical_emphasis() -> None:
    """Formatting repair (2026-07-11): a rewrite that kept a canonical-bolded
    tool name but dropped its \\textbf{} gets the emphasis restored."""
    text, reason = validate_tailored_bullet(
        "Maintained LaTeX documentation of workflows, reducing downtime by 41\\%.",
        CANONICAL,
    )
    assert reason is None
    assert text is not None and "\\textbf{LaTeX}" in text


def test_validate_bullet_emphasis_restore_respects_existing_bold() -> None:
    tailored = "Maintained \\textbf{LaTeX} documentation, reducing downtime by 41\\%."
    text, reason = validate_tailored_bullet(tailored, CANONICAL)
    assert reason is None
    assert text is not None and text.count("\\textbf{LaTeX}") == 1
    assert "\\textbf{\\textbf{" not in text


def test_lowercase_concept_word_demoted_not_rejected() -> None:
    """Bolding 'performance' is bad emphasis, not a hallucinated tool."""
    from services.resumes.structured import parse_skill_anchors

    anchors = parse_skill_anchors(BASEINFO_TEXT)
    text, reason = validate_tailored_bullet(
        "Optimized \\textbf{performance} of docs, reducing downtime by 41\\%.",
        CANONICAL,
        skill_anchors=anchors,
    )
    assert reason is None
    assert text is not None and "\\textbf{performance}" not in text
    assert "performance" in text


def test_validate_bullet_demotes_bolted_posting_phrase() -> None:
    """Multi-word \\textbf phrases absent from the canonical bullet keep the
    rewrite but lose the emphasis (posting language, not a tool claim)."""
    text, reason = validate_tailored_bullet(
        "Built pipelines for robust \\textbf{computer modeling and analysis experience}, "
        "reducing downtime by 41\\%.",
        CANONICAL,
    )
    assert reason is None
    assert text is not None
    assert "\\textbf{computer modeling" not in text
    assert "computer modeling and analysis experience" in text


def test_validate_bullet_allows_bold_tech_and_canonical_phrases() -> None:
    # Short tech names are fine even when new...
    text, reason = validate_tailored_bullet(
        "Maintained \\textbf{LaTeX} docs with \\textbf{Pandas}, reducing downtime by 41\\%.",
        CANONICAL,
    )
    assert reason is None
    # ...and long phrases are fine when the canonical bullet contains them.
    long_canonical = (
        "Engineered a distributed, low-latency data pipeline for quantitative trading, "
        "reducing downtime by 41\\%."
    )
    text, reason = validate_tailored_bullet(
        "Engineered a \\textbf{low-latency data pipeline for quantitative trading}, "
        "reducing downtime by 41\\%.",
        long_canonical,
    )
    assert reason is None


def test_validate_bullet_rejects_quoted_posting_phrase() -> None:
    text, reason = validate_tailored_bullet(
        'Built pipelines, providing "computer modeling experience", reducing downtime by 41\\%.',
        CANONICAL,
    )
    assert text is None and reason == "quoted posting phrase"


def test_validate_bullet_strips_appended_filler_clause() -> None:
    """A bolted-on filler tail is stripped; the rest of the rewrite survives
    (rejecting the whole bullet threw away otherwise-good tailoring)."""
    text, reason = validate_tailored_bullet(
        "Maintained \\textbf{LaTeX} docs, reducing downtime by 41\\%, demonstrating analytical rigour.",
        CANONICAL,
    )
    assert reason is None
    assert text is not None
    assert "demonstrating" not in text
    assert "41\\%" in text
    assert text.endswith(".")


def test_validate_bullet_aggressive_skips_quote_and_filler_guards() -> None:
    """Aggressive mode is style-only: the same two inputs that a normal build
    rejects/strips (see test_validate_bullet_rejects_quoted_posting_phrase and
    test_validate_bullet_strips_appended_filler_clause) pass through whole."""
    from services.resumes.structured import RenderConfig

    text, reason = validate_tailored_bullet(
        'Built pipelines, providing "computer modeling experience", reducing downtime by 41\\%.',
        CANONICAL,
        RenderConfig(aggressive=True),
    )
    assert reason is None
    assert text is not None and '"computer modeling experience"' in text

    text, reason = validate_tailored_bullet(
        "Maintained \\textbf{LaTeX} docs, reducing downtime by 41\\%, demonstrating analytical rigour.",
        CANONICAL,
        RenderConfig(aggressive=True),
    )
    assert reason is None
    assert text is not None and "demonstrating analytical rigour" in text


def test_validate_bullet_aggressive_raises_bold_density_cap() -> None:
    """The bold-density cap itself stays a cap (never fully disabled) — but
    _effective_render_config raises it by 2 for aggressive builds."""
    from services.resumes.structured import RenderConfig, _effective_render_config, parse_skill_anchors

    anchors = parse_skill_anchors(BASEINFO_TEXT)
    bullet = (
        "Used \\textbf{Kafka}, \\textbf{FastAPI}, \\textbf{Pandas}, \\textbf{NumPy}, "
        "\\textbf{TensorFlow}, \\textbf{Docker}, and \\textbf{Git} to reduce downtime by 41\\%."
    )
    base_config = RenderConfig(max_bold_per_bullet=6)
    text, reason = validate_tailored_bullet(bullet, CANONICAL, base_config, skill_anchors=anchors)
    assert reason is None and text is not None
    assert text.count("\\textbf{") == 6  # one of the 7 demoted to plain text

    aggressive_config = _effective_render_config(RenderConfig(max_bold_per_bullet=6, aggressive=True))
    text, reason = validate_tailored_bullet(bullet, CANONICAL, aggressive_config, skill_anchors=anchors)
    assert reason is None and text is not None
    assert text.count("\\textbf{") == 7  # aggressive cap (8) fits all 7


def test_validate_bullet_aggressive_allows_fabricated_tools() -> None:
    """Aggressive mode allows bolded terms the candidate never listed."""
    from services.resumes.structured import RenderConfig

    bullet = (
        "Deployed on \\textbf{Kubernetes} and \\textbf{Solidworks}, "
        "reducing downtime by 41\\%."
    )
    text, reason = validate_tailored_bullet(bullet, CANONICAL, RenderConfig(aggressive=True))
    assert reason is None and text is not None
    assert "\\textbf{Kubernetes}" in text
    assert "\\textbf{Solidworks}" in text


def test_effective_render_config_scales_only_when_aggressive() -> None:
    from services.resumes.structured import RenderConfig, _effective_render_config

    base = RenderConfig(max_bold_per_bullet=6, max_bullet_chars=400)
    assert _effective_render_config(base) is base  # no-op, not even a copy

    aggressive = RenderConfig(
        max_bold_per_bullet=6,
        max_bullet_chars=400,
        aggressive=True,
        adjacent_tool_leeway=False,
        grounding_audit=False,
    )
    scaled = _effective_render_config(aggressive)
    assert scaled.max_bold_per_bullet == 8
    assert scaled.adjacent_tool_leeway is True
    assert scaled.grounding_audit is False
    # Never touched: max_bullet_chars looks like a style knob but live-trial
    # evidence (2026-07-12) showed raising it breaks the page-character-budget
    # calibration (a real listing went from 1 page to 2). Page-fit knobs are
    # layout limits, not tailoring freedom.
    assert scaled.max_bullet_chars == 400
    assert scaled.max_visible_bullets == aggressive.max_visible_bullets
    assert scaled.max_total_bullet_chars == aggressive.max_total_bullet_chars


def test_parse_skill_anchors_reads_real_baseinfo() -> None:
    from services.resumes.structured import parse_skill_anchors

    anchors = parse_skill_anchors(BASEINFO_TEXT)
    assert "kafka" in anchors
    assert "react" in anchors
    assert "node.js" in anchors  # split from "JavaScript/Node.js"
    assert "kubernetes" not in anchors


def test_validate_bullet_rejects_invented_tool_outside_anchors() -> None:
    from services.resumes.structured import parse_skill_anchors

    anchors = parse_skill_anchors(BASEINFO_TEXT)
    text, reason = validate_tailored_bullet(
        "Deployed docs on \\textbf{Kubernetes}, reducing downtime by 41\\%.",
        CANONICAL,
        skill_anchors=anchors,
    )
    assert text is None and "invented tool 'Kubernetes'" in str(reason)
    # A real anchor tool passes even when the canonical bullet lacks it...
    text, reason = validate_tailored_bullet(
        "Maintained \\textbf{LaTeX} docs in \\textbf{Git}, reducing downtime by 41\\%.",
        CANONICAL,
        skill_anchors=anchors,
    )
    assert reason is None
    # ...and profiles without a SKILL ANCHORS section skip the check entirely.
    text, reason = validate_tailored_bullet(
        "Deployed docs on \\textbf{Kubernetes}, reducing downtime by 41\\%.",
        CANONICAL,
        skill_anchors=(),
    )
    assert reason is None


def test_validate_bullet_adjacent_tool_leeway_skips_invented_tool_reject() -> None:
    """With adjacent_tool_leeway on, an unanchored product-shaped tool no
    longer hard-rejects; the LLM grounding audit polices plausibility instead."""
    from services.resumes.structured import RenderConfig, parse_skill_anchors

    anchors = parse_skill_anchors(BASEINFO_TEXT)
    config = RenderConfig(adjacent_tool_leeway=True)
    text, reason = validate_tailored_bullet(
        "Built an AI harness with \\textbf{LangChain}, reducing downtime by 41\\%.",
        CANONICAL,
        config,
        skill_anchors=anchors,
    )
    assert reason is None
    assert text is not None and "\\textbf{LangChain}" in text


def test_short_anchor_does_not_substring_match_everything() -> None:
    """Regression: anchor 'c' must not whitelist every phrase containing a c."""
    from services.resumes.structured import parse_skill_anchors

    anchors = parse_skill_anchors(BASEINFO_TEXT)
    assert "c" in anchors  # the C language is a real anchor
    text, reason = validate_tailored_bullet(
        "Developed \\textbf{responsive web interfaces}, reducing downtime by 41\\%.",
        CANONICAL,
        skill_anchors=anchors,
    )
    # Multi-word posting phrase: demoted to plain text, not kept bold.
    assert reason is None
    assert text is not None and "\\textbf{responsive" not in text
    assert "responsive web interfaces" in text
    # Whole-word anchors still match inside compounds ("git-based workflows").
    text, reason = validate_tailored_bullet(
        "Maintained docs via \\textbf{Git-based workflows}, reducing downtime by 41\\%.",
        CANONICAL,
        skill_anchors=anchors,
    )
    assert reason is None


def test_validate_bullet_allows_reworded_canonical_concept() -> None:
    """Bolding a phrase whose every word exists in the canonical bullet is fine."""
    long_canonical = (
        "Engineered a distributed, low-latency data pipeline for trading, "
        "reducing downtime by 41\\%."
    )
    text, reason = validate_tailored_bullet(
        "Engineered a \\textbf{distributed data pipeline} for trading, "
        "reducing downtime by 41\\%.",
        long_canonical,
        skill_anchors=("kafka",),
    )
    assert reason is None


def test_catalog_loads_skill_anchors() -> None:
    assert CATALOG is not None
    assert "kafka" in CATALOG.skill_anchors


def test_render_invented_tool_respects_leeway_config(monkeypatch) -> None:
    """End-to-end: without leeway an invented tool falls back to canonical;
    with the profile's leeway on, the deterministic gate passes it through."""
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "rewrite_scope", "full")
    goopter = _entry_id("goopter")
    selection = StructuredSelection(
        ranking=[goopter],
        bullets={
            goopter: [
                "Engineered pipelines on \\textbf{Kubernetes} using \\textbf{Kafka} and \\textbf{FastAPI}."
            ]
        },
    )
    monkeypatch.setattr(CATALOG.render_config, "adjacent_tool_leeway", False)
    latex, report = render_structured_resume(CATALOG, selection)
    assert "Kubernetes" not in latex
    assert any("invented tool" in item for item in report.canonical_fallbacks)

    monkeypatch.setattr(CATALOG.render_config, "adjacent_tool_leeway", True)
    latex, report = render_structured_resume(CATALOG, selection)
    assert "Kubernetes" in latex
    assert not any("invented tool" in item for item in report.canonical_fallbacks)


def test_render_aggressive_forces_leeway_even_when_profile_default_is_off(monkeypatch) -> None:
    """A ".resumebuild --aggressive" build must behave like adjacent_tool_leeway
    is on even for a profile whose structured_config.json has it off."""
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "rewrite_scope", "full")
    monkeypatch.setattr(CATALOG.render_config, "adjacent_tool_leeway", False)
    goopter = _entry_id("goopter")
    selection = StructuredSelection(
        ranking=[goopter],
        bullets={
            goopter: [
                "Engineered pipelines on \\textbf{Kubernetes} using \\textbf{Kafka} and \\textbf{FastAPI}."
            ]
        },
    )

    monkeypatch.setattr(CATALOG.render_config, "aggressive", False)
    latex, report = render_structured_resume(CATALOG, selection)
    assert "Kubernetes" not in latex
    assert any("invented tool" in item for item in report.canonical_fallbacks)

    monkeypatch.setattr(CATALOG.render_config, "aggressive", True)
    latex, report = render_structured_resume(CATALOG, selection)
    assert "Kubernetes" in latex
    assert not any("invented tool" in item for item in report.canonical_fallbacks)


# ---------------------------------------------------------------------------
# Cross-bullet consistency guards
# ---------------------------------------------------------------------------

def _guard_state(bullets: dict[tuple[str, int], tuple[str, str, bool]]):
    """Build (ordered_keys, canonicals, resolved, tailored) from
    {key: (canonical, resolved, tailored)} for enforce_cross_bullet_consistency."""
    ordered = list(bullets)
    canonicals = {k: v[0] for k, v in bullets.items()}
    resolved = {k: v[1] for k, v in bullets.items()}
    tailored = {k: v[2] for k, v in bullets.items()}
    return ordered, canonicals, resolved, tailored


def test_guard_reverts_metric_stated_twice_in_entry() -> None:
    ordered, canonicals, resolved, tailored = _guard_state({
        ("obotz", 0): (
            "Improved robotics scores by 35\\% through instruction.",
            "Improved robotics scores by 35\\% through instruction.",
            False,
        ),
        ("obotz", 1): (
            "Helped students place in the 75th percentile of competitions.",
            "Helped students improve completion scores by 35\\% in competitions.",
            True,
        ),
    })
    reasons = enforce_cross_bullet_consistency(ordered, canonicals, resolved, tailored)
    assert resolved[("obotz", 1)] == canonicals[("obotz", 1)]
    assert not tailored[("obotz", 1)]
    assert any("stated twice" in r for r in reasons)


def test_guard_keeps_metric_duplicated_between_canonicals() -> None:
    # Both canonical bullets legitimately state 35% — the profile's own truth
    # is exempt, even when one bullet is a tailored rewrite keeping the number.
    ordered, canonicals, resolved, tailored = _guard_state({
        ("e", 0): ("Raised scores 35\\% via drills.", "Raised scores 35\\% via drills.", False),
        ("e", 1): (
            "Sustained the 35\\% gain across terms.",
            "Sustained the 35\\% score gain across every term taught.",
            True,
        ),
    })
    reasons = enforce_cross_bullet_consistency(ordered, canonicals, resolved, tailored)
    assert tailored[("e", 1)]
    assert reasons == []


def test_guard_reverts_near_duplicate_sibling() -> None:
    ordered, canonicals, resolved, tailored = _guard_state({
        ("e", 0): (
            "Assisted students improving robotics problem completion through digital logic concepts.",
            "Assisted students improving robotics problem completion through digital logic concepts.",
            False,
        ),
        ("e", 1): (
            "Delivered clear simplified explanations of technical processes to parents.",
            "Assisted students improving robotics problem completion via digital logic concepts.",
            True,
        ),
    })
    reasons = enforce_cross_bullet_consistency(ordered, canonicals, resolved, tailored)
    assert resolved[("e", 1)] == canonicals[("e", 1)]
    assert any("near-duplicate" in r for r in reasons)


def test_guard_reverts_shared_opening_clause_even_with_low_whole_bullet_overlap() -> None:
    """Regression: two bullets that open on nearly the same clause but end on
    unrelated content dodge the whole-sentence NEAR_DUPLICATE_JACCARD check
    (their diverging endings drag the overlap down) even though a human
    reader immediately notices the repeated setup."""
    ordered, canonicals, resolved, tailored = _guard_state({
        ("mcg3d", 0): (
            "Integration work: frontend features wired to Python (Flask) backend services over secure RESTful APIs.",
            "Integrated frontend features with Python (Flask) backend services and secure RESTful APIs, contributing to AI-enabled workflows using OpenAI APIs.",
            True,
        ),
        ("mcg3d", 1): (
            "Environment: deployment and testing in a containerized Docker setup, with contributions to technical documentation.",
            "Integrated frontend features with Python backend services and RESTful APIs, applying critical thinking to resolve data flow and interface issues.",
            True,
        ),
    })
    reasons = enforce_cross_bullet_consistency(ordered, canonicals, resolved, tailored)
    assert resolved[("mcg3d", 1)] == canonicals[("mcg3d", 1)]
    assert any("shares its opening clause" in r for r in reasons)


def test_guard_reverts_restated_fact_with_reordered_opening_words() -> None:
    """Regression (live): 'Built production-ready web applications using React
    and TypeScript, focusing on...' vs 'Built production-ready web interfaces
    using React and TypeScript with a focus on...' restate the identical fact
    twice. Whole-bullet Jaccard is 0.55 here -- below even the pre-loosening
    0.6 threshold -- so only a bag-of-words check on the opening span (not
    exact sequential prefix matching) catches this."""
    ordered, canonicals, resolved, tailored = _guard_state({
        ("mcg3d", 0): (
            "Frontend features built in React and TypeScript.",
            "Built production-ready web applications using React and TypeScript, focusing on scalable component architecture, performance, and user-centric design.",
            True,
        ),
        ("mcg3d", 1): (
            "Environment: deployment and testing in a containerized Docker setup.",
            "Built production-ready web interfaces using React and TypeScript with a focus on scalable component architecture.",
            True,
        ),
    })
    reasons = enforce_cross_bullet_consistency(ordered, canonicals, resolved, tailored)
    assert resolved[("mcg3d", 1)] == canonicals[("mcg3d", 1)]
    assert any("shares its opening clause" in r for r in reasons)


def test_guard_reverts_credential_handling_restated() -> None:
    """Regression (live): both bullets open 'Implemented secure credential
    handling...' then diverge -- another same-setup restatement caught only
    by the opening-span bag-of-words check, not whole-bullet Jaccard."""
    ordered, canonicals, resolved, tailored = _guard_state({
        ("wma", 0): (
            "Security and features: secure credential handling, key generation and error recovery, machine learning autocomplete, and in-browser Python code execution.",
            "Implemented secure credential handling and key generation routines to strengthen application-level security and data integrity.",
            True,
        ),
        ("wma", 1): (
            "Infrastructure: a Raspberry Pi running Ubuntu serves as the networking backend with a secure PostgreSQL server, exposed as a layered RESTful API.",
            "Implemented secure credential handling, key generation and error recovery, machine learning autocomplete and Python code execution in-browser.",
            True,
        ),
    })
    reasons = enforce_cross_bullet_consistency(ordered, canonicals, resolved, tailored)
    assert resolved[("wma", 1)] == canonicals[("wma", 1)]
    assert any("shares its opening clause" in r for r in reasons)


def test_guard_reverts_verb_object_tool_collision_at_half_overlap() -> None:
    """Regression (live, KPMG cyber run): 'Built adaptive algorithms using
    TensorFlow and scikit-learn...' next to the canonical 'Built reinforcement
    algorithms using TensorFlow, enabling...' restates the same fact — the
    tailored bullet was supposed to rewrite the DASHBOARD canonical but
    drifted onto the RL fact instead. Opening-span Jaccard is exactly 0.5
    (verb+object+tool shared, modifiers differ), which the original 0.6
    threshold missed."""
    ordered, canonicals, resolved, tailored = _guard_state({
        ("goopter", 0): (
            "Also shipped an analytics dashboard on PyTorch Lightning and NumPy, with Tableau and Power BI integrations.",
            "Built adaptive algorithms using TensorFlow and scikit-learn to detect patterns in high-frequency trading data streams.",
            True,
        ),
        ("goopter", 1): (
            "Built reinforcement algorithms using TensorFlow, enabling adaptive, high-frequency trading signal generation.",
            "Built reinforcement algorithms using TensorFlow, enabling adaptive, high-frequency trading signal generation.",
            False,
        ),
    })
    reasons = enforce_cross_bullet_consistency(ordered, canonicals, resolved, tailored)
    assert resolved[("goopter", 0)] == canonicals[("goopter", 0)]
    assert any("shares its opening clause" in r for r in reasons)


def test_guard_allows_shared_opening_already_true_of_canonicals() -> None:
    """A shared opening clause that the CANONICAL bullets already share is the
    profile's own truth, not a tailoring defect — must not revert."""
    ordered, canonicals, resolved, tailored = _guard_state({
        ("e", 0): (
            "Improved robotics scores through hands-on instruction of digital logic concepts for students.",
            "Improved robotics scores through hands-on instruction of digital logic principles for learners.",
            True,
        ),
        ("e", 1): (
            "Improved robotics scores through hands-on instruction of embedded programming for students.",
            "Improved robotics scores through hands-on instruction of embedded firmware for learners.",
            True,
        ),
    })
    reasons = enforce_cross_bullet_consistency(ordered, canonicals, resolved, tailored)
    assert resolved[("e", 0)] != canonicals[("e", 0)]
    assert resolved[("e", 1)] != canonicals[("e", 1)]
    assert not any("shares its opening clause" in r for r in reasons)


def test_guard_allows_moderate_sibling_overlap() -> None:
    """The loosened Jaccard threshold (0.75) tolerates siblings that merely
    share topic vocabulary without being near-copies."""
    ordered, canonicals, resolved, tailored = _guard_state({
        ("e", 0): (
            "Assisted students improving robotics problem completion through digital logic concepts.",
            "Assisted students improving robotics problem completion through digital logic concepts.",
            False,
        ),
        ("e", 1): (
            "Delivered clear simplified explanations of technical processes to parents.",
            "Guided students through robotics fundamentals with hands-on digital logic exercises weekly.",
            True,
        ),
    })
    reasons = enforce_cross_bullet_consistency(ordered, canonicals, resolved, tailored)
    assert tailored[("e", 1)]
    assert reasons == []


def test_guard_aggressive_thresholds_tolerate_repetition_default_reverts() -> None:
    """Two siblings share 9/11 whole-bullet words (0.818 Jaccard, disjoint
    openings so the opening-clause guard stays quiet in both modes): default
    thresholds (0.75) revert the second one, the loosened aggressive-mode
    thresholds (0.92) that render_structured_resume passes when
    config.aggressive is set let it stand."""
    fixture = {
        ("g", 0): (
            "Wrote integration tests for the payment gateway module.",
            "Refactored using Kafka Python FastAPI Pandas NumPy pipelines for trading.",
            True,
        ),
        ("g", 1): (
            "Documented onboarding steps for new engineering hires.",
            "Migrated using Kafka Python FastAPI Pandas NumPy pipelines for trading.",
            True,
        ),
    }

    ordered, canonicals, resolved, tailored = _guard_state(fixture)
    default_reasons = enforce_cross_bullet_consistency(ordered, canonicals, resolved, tailored)
    assert resolved[("g", 1)] == canonicals[("g", 1)]
    assert any("near-duplicate" in r for r in default_reasons)
    assert not any("opening clause" in r for r in default_reasons)

    ordered, canonicals, resolved, tailored = _guard_state(fixture)
    aggressive_reasons = enforce_cross_bullet_consistency(
        ordered,
        canonicals,
        resolved,
        tailored,
        near_duplicate_jaccard=0.92,
        opening_overlap_jaccard=0.75,
        max_phrase_bullets=6,
        max_stem_bullets=8,
    )
    assert aggressive_reasons == []
    assert resolved[("g", 1)] == "Migrated using Kafka Python FastAPI Pandas NumPy pipelines for trading."


def test_guard_reverts_phrase_repeated_across_entries() -> None:
    # "functional specifications" bolted into four different entries: the
    # first three occurrences stay (loosened cap), the fourth reverts.
    entries = {}
    for n, eid in enumerate(("alpha", "beta", "gamma", "delta")):
        entries[(eid, 0)] = (
            f"Built module {n} for the platform team.",
            f"Built module {n}, defining functional specifications for delivery.",
            True,
        )
    ordered, canonicals, resolved, tailored = _guard_state(entries)
    reasons = enforce_cross_bullet_consistency(ordered, canonicals, resolved, tailored)
    assert tailored[("alpha", 0)] and tailored[("beta", 0)] and tailored[("gamma", 0)]
    assert not tailored[("delta", 0)]
    # Which of the overlapping repeated n-grams is named first is unimportant.
    assert any("repeated across" in r and "delta[0]" in r for r in reasons)


def test_guard_reverts_long_word_leaned_on_everywhere() -> None:
    # "documenting"/"documentation" stem-matched across five entries: the
    # fifth reverts (loosened cap of 4). Canonical uses don't count.
    entries = {}
    variants = ("documenting", "documentation", "documenting", "documented", "documentation")
    for n, (eid, word) in enumerate(zip(("a", "b", "c", "d", "e"), variants)):
        entries[(eid, 0)] = (
            f"Shipped feature {n} for the team.",
            f"Shipped feature {n}, {word} workflows for stakeholders.",
            True,
        )
    ordered, canonicals, resolved, tailored = _guard_state(entries)
    reasons = enforce_cross_bullet_consistency(ordered, canonicals, resolved, tailored)
    assert not tailored[("e", 0)]
    assert sum(1 for k in entries if tailored[k]) == 4
    assert any("repeated across" in r for r in reasons)


def test_render_reverts_duplicate_metric_bullet_end_to_end(monkeypatch) -> None:
    """The live-observed Obotz defect: a rewrite relocated the 35% metric into
    a sibling bullet while the canonical bullet still stated it."""
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "rewrite_scope", "full")
    obotz = _entry_id("obotz")
    selection = StructuredSelection(
        ranking=[obotz],
        bullets={
            obotz: [
                "",
                "Helped students improve robotics problem completion scores by 35\\% "
                "using custom \\textbf{Arduino} hardware.",
            ]
        },
    )
    latex, report = render_structured_resume(CATALOG, selection)
    # The canonical 75th-percentile bullet is restored; 35% appears once.
    assert "75th percentile" in latex
    assert latex.count("35\\%") == 1
    assert any("stated twice" in item for item in report.canonical_fallbacks)


def test_validate_bullet_escapes_stray_specials() -> None:
    text, reason = validate_tailored_bullet(
        "Reduced downtime by 41% via R&D documentation.", CANONICAL
    )
    assert reason is None
    assert text == "Reduced downtime by 41\\% via R\\&D documentation."


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _count_visible_bullets(latex: str) -> int:
    return latex.count("\\item")


@pytest.mark.parametrize(
    "ranking_fragments",
    [
        [],  # template order
        ["goopter", "markham", "frontend", "obotz"],  # experience first
        ["web-messaging", "frontend", "goopter"],  # web-flavoured
    ],
)
def test_render_bullet_count_within_budget(ranking_fragments: list[str]) -> None:
    assert CATALOG is not None
    selection = StructuredSelection(ranking=[_entry_id(f) for f in ranking_fragments])
    latex, report = render_structured_resume(CATALOG, selection)
    assert MIN_VISIBLE_BULLETS <= report.visible_bullet_count <= MAX_VISIBLE_BULLETS
    assert _count_visible_bullets(latex) == report.visible_bullet_count


def test_render_ranking_decides_visibility() -> None:
    """Ranking every non-embedded entry above eebot pushes eebot off the page —
    no entry has category immunity anymore."""
    assert CATALOG is not None
    ranking = [
        _entry_id(f)
        for f in (
            "web-messaging", "frontend", "goopter", "markham",
            "terrain", "bookstore", "obotz", "particle",
        )
    ]
    latex, report = render_structured_resume(CATALOG, StructuredSelection(ranking=ranking))
    assert "Web Messaging App" in latex
    assert "MCG3D" in latex
    assert _entry_id("eebot") in report.hidden_entries
    assert "eebot Mobile Robot System" not in latex


def test_render_any_combination_can_surface_together() -> None:
    """Entries from formerly-incompatible buckets (frontend + embedded + ML)
    render side by side when the ranking calls for it."""
    assert CATALOG is not None
    ranking = [_entry_id("frontend"), _entry_id("eebot"), _entry_id("goopter")]
    latex, report = render_structured_resume(CATALOG, StructuredSelection(ranking=ranking))
    for fragment in ("frontend", "eebot", "goopter"):
        assert _entry_id(fragment) in report.visible_entries
    assert "MCG3D" in latex
    assert "eebot Mobile Robot System" in latex
    assert "Goopter" in latex


def test_render_has_no_comment_environments_and_single_document() -> None:
    assert CATALOG is not None
    latex, _ = render_structured_resume(CATALOG, StructuredSelection())
    assert latex.count("\\begin{document}") == 1
    assert latex.count("\\end{document}") == 1
    assert "\\begin{comment}" not in latex
    assert latex.count("\\begin{itemize}") == latex.count("\\end{itemize}")
    assert "<" not in latex.split("\\begin{document}")[1].replace("$<$", "")


def test_render_preserves_mandatory_blocks() -> None:
    assert CATALOG is not None
    for ranking in ([], [_entry_id("goopter"), _entry_id("frontend")]):
        latex, _ = render_structured_resume(CATALOG, StructuredSelection(ranking=ranking))
        assert "Ernest Choi" in latex
        assert "ernestljchoi@gmail.com" in latex
        assert "\\section*{Skills}" in latex
        assert "\\section*{Education}" in latex
        assert "Toronto Metropolitan University" in latex


def test_render_preserves_markham_metrics_when_visible() -> None:
    assert CATALOG is not None
    latex, _ = render_structured_resume(
        CATALOG, StructuredSelection(ranking=[_entry_id("markham")])
    )
    assert "City of Markham" in latex
    assert "41\\%" in latex
    assert "14\\%" in latex
    assert "20\\%" in latex


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def test_extract_json_object_handles_fences_and_prose() -> None:
    payload = extract_json_object('Here you go:\n```json\n{"keywords": ["ML"]}\n```')
    assert payload == {"keywords": ["ML"]}
    payload = extract_json_object('prefix {"a": {"b": 1}} suffix')
    assert payload == {"a": {"b": 1}}
    assert extract_json_object("no json here") is None


def test_extract_json_object_survives_unescaped_latex_escape_collisions() -> None:
    # \textbf and \times start with valid JSON escapes (\t): a model response
    # with unescaped backslashes must not decode them to TAB + "extbf"/"imes".
    raw = '{"b": "Built \\textbf{C++} tools sized $4\\times16$."}'
    payload = extract_json_object(raw)
    assert payload is not None
    assert "\\textbf{C++}" in payload["b"]
    assert "\\times" in payload["b"]
    assert "\t" not in payload["b"]
    # Properly escaped input keeps working.
    payload = extract_json_object('{"b": "Built \\\\textbf{C++} fast."}')
    assert payload is not None
    assert "\\textbf{C++}" in payload["b"]
    # Real JSON unicode escapes are untouched.
    payload = extract_json_object('{"b": "Montr\\u00e9al"}')
    assert payload is not None
    assert payload["b"] == "Montréal"


def test_validate_tailored_bullet_repairs_tab_corrupted_textbf() -> None:
    canonical = "Built tools in \\textbf{C++} for rendering."
    corrupted = "Built rendering tools in \textbf{C++}."  # \t is a literal TAB here
    sanitized, reason = validate_tailored_bullet(corrupted, canonical)
    assert reason is None
    assert "\\textbf{C++}" in sanitized
    assert "\t" not in sanitized


def test_validate_bullet_unwraps_whole_bullet_quotes() -> None:
    sanitized, reason = validate_tailored_bullet(
        '"Maintained \\textbf{LaTeX} documentation, reducing downtime by 41\\%."',
        CANONICAL,
    )
    assert reason is None
    assert sanitized is not None
    assert not sanitized.startswith('"')
    assert not sanitized.endswith('"')
    assert "Maintained" in sanitized


def test_validate_bullet_unwraps_smart_quote_wrapping() -> None:
    sanitized, reason = validate_tailored_bullet(
        "“Maintained \\textbf{LaTeX} documentation, reducing downtime by 41\\%.”",
        CANONICAL,
    )
    assert reason is None
    assert sanitized is not None
    assert "“" not in sanitized and "”" not in sanitized
    assert '"' not in sanitized


def test_validate_bullet_normalizes_smart_apostrophe() -> None:
    sanitized, reason = validate_tailored_bullet(
        "Maintained the team’s documentation, reducing downtime by 41\\%.",
        CANONICAL,
    )
    assert reason is None
    assert sanitized is not None
    assert "’" not in sanitized
    assert "team's" in sanitized


def test_validate_bullet_still_rejects_interior_smart_quoted_phrase() -> None:
    # Interior quoting is posting-phrase padding regardless of quote style;
    # normalising smart quotes must not launder it past the straight-quote check.
    sanitized, reason = validate_tailored_bullet(
        "Maintained “world-class scalable” documentation, reducing downtime by 41\\%.",
        CANONICAL,
    )
    assert sanitized is None
    assert reason == "quoted posting phrase"


def test_validate_bullet_converts_markdown_bold_to_textbf() -> None:
    sanitized, reason = validate_tailored_bullet(
        "Maintained **LaTeX** documentation, reducing downtime by 41\\%.",
        CANONICAL,
    )
    assert reason is None
    assert sanitized is not None
    assert "\\textbf{LaTeX}" in sanitized
    assert "**" not in sanitized


def test_validate_bullet_restores_missing_backslash_on_textbf() -> None:
    sanitized, reason = validate_tailored_bullet(
        "Maintained textbf{LaTeX} documentation, reducing downtime by 41\\%.",
        CANONICAL,
    )
    assert reason is None
    assert sanitized is not None
    assert "\\textbf{LaTeX}" in sanitized
    assert " textbf{" not in sanitized


def test_validate_bullet_markdown_bold_still_subject_to_grounding() -> None:
    # Markdown-bolded posting language converts to \textbf, then the standard
    # grounding pass demotes it to plain text (multi-word non-tool phrase).
    sanitized, reason = validate_tailored_bullet(
        "Maintained **mission critical infrastructure** documentation, reducing downtime by 41\\%.",
        CANONICAL,
    )
    assert reason is None
    assert sanitized is not None
    assert "\\textbf{mission critical infrastructure}" not in sanitized
    assert "mission critical infrastructure" in sanitized


def test_parse_structured_response_ignores_legacy_family_fields() -> None:
    """Old-style payloads carrying role_family/include keys still parse; the
    unknown fields are simply ignored."""
    assert CATALOG is not None
    selection = parse_structured_response(
        '{"role_family": "BANANA", "include": ["x"], "ranking": [], "bullets": {}}',
        CATALOG,
    )
    assert selection is not None
    assert selection.ranking == []


def test_parse_structured_response_filters_unknown_ids() -> None:
    assert CATALOG is not None
    selection = parse_structured_response(
        json.dumps(
            {
                "ranking": ["nonexistent-entry", CATALOG.entries[0].entry_id],
                "bullets": {"nonexistent-entry": ["x"], CATALOG.entries[0].entry_id: ["y"]},
            }
        ),
        CATALOG,
    )
    assert selection is not None
    assert selection.ranking == [CATALOG.entries[0].entry_id]
    assert list(selection.bullets) == [CATALOG.entries[0].entry_id]


# ---------------------------------------------------------------------------
# End-to-end through generate_resume_rewrite
# ---------------------------------------------------------------------------

class _FakeModelsApi:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        response_text = self.response_text

        class _Response:
            text = response_text

        return _Response()


class _FakeGeminiClient:
    def __init__(self, models_api: _FakeModelsApi) -> None:
        self.models = models_api


def _profile_copy(tmp_path: Path) -> Path:
    target = tmp_path / "profile"
    target.mkdir()
    for name in (
        "template.tex",
        "baseinfo.txt",
        "instructions.txt",
        "structured_config.json",
        "structured_guidance.txt",
    ):
        if (PROFILE_DIR / name).exists():
            shutil.copy(PROFILE_DIR / name, target / name)
    return target


def test_generate_resume_rewrite_structured_path_with_json_response(tmp_path: Path) -> None:
    profile = _profile_copy(tmp_path)
    job = extract_job_context_from_message(
        "[LinkedIn] Machine Learning Engineer\nhttps://example.com/job"
    )
    assert job is not None

    response = json.dumps(
        {
            "keywords": ["Python", "TensorFlow"],
            "ranking": [_entry_id("goopter"), _entry_id("markham")],
            "bullets": {},
        }
    )
    models_api = _FakeModelsApi(response)

    result = generate_resume_rewrite(
        settings=GeminiSettings(api_key="test-key", model="gemini-2.5-pro"),
        job=job,
        cache_name="cachedContents/999",
        baseinfo_paths=[profile / "baseinfo.txt"],
        support_paths=[profile / "instructions.txt"],
        template_path=profile / "template.tex",
        scraper=lambda _: _scraped("Machine learning model training pipelines with Python"),
        client_factory=lambda _: _FakeGeminiClient(models_api),
    )

    assert result.status == "ok"
    assert result.used_provider == "gemini"
    assert result.latex_document is not None
    assert "\\documentclass" in result.latex_document
    assert "Goopter" in result.latex_document
    summary = json.loads(result.rewritten_resume or "{}")
    assert summary["mode"] == "structured"
    assert _entry_id("goopter") in summary["visible_entries"]
    # Structured mode must not send the legacy context cache, but does request
    # native JSON output.
    sent_config = models_api.calls[0].get("config")
    assert sent_config is not None
    cached = getattr(sent_config, "cached_content", None) or (
        sent_config.get("cached_content") if isinstance(sent_config, dict) else None
    )
    assert not cached
    mime = getattr(sent_config, "response_mime_type", None) or (
        sent_config.get("response_mime_type") if isinstance(sent_config, dict) else None
    )
    assert mime == "application/json"


def test_generate_resume_rewrite_aggressive_flag_reaches_prompt_and_summary(tmp_path: Path) -> None:
    """End-to-end: the aggressive=True kwarg on generate_resume_rewrite must
    reach the actual prompt text sent to the provider (not just an internal
    flag nobody reads) and be reported back in the summary/message."""
    profile = _profile_copy(tmp_path)
    job = extract_job_context_from_message(
        "[LinkedIn] Machine Learning Engineer\nhttps://example.com/job"
    )
    assert job is not None

    response = json.dumps(
        {
            "keywords": ["Python", "TensorFlow"],
            "ranking": [_entry_id("goopter"), _entry_id("markham")],
            "bullets": {},
        }
    )
    models_api = _FakeModelsApi(response)

    result = generate_resume_rewrite(
        settings=GeminiSettings(api_key="test-key", model="gemini-2.5-pro"),
        job=job,
        cache_name="cachedContents/999",
        baseinfo_paths=[profile / "baseinfo.txt"],
        support_paths=[profile / "instructions.txt"],
        template_path=profile / "template.tex",
        scraper=lambda _: _scraped("Machine learning model training pipelines with Python"),
        client_factory=lambda _: _FakeGeminiClient(models_api),
        aggressive=True,
    )

    assert result.status == "ok"
    sent_prompt = models_api.calls[0].get("contents")
    assert isinstance(sent_prompt, str) and "AGGRESSIVE MODE" in sent_prompt

    summary = json.loads(result.rewritten_resume or "{}")
    assert summary["aggressive"] is True
    assert "aggressive mode" in (result.message or "").lower()


def test_generate_resume_rewrite_structured_deterministic_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When every provider fails, a deterministic resume is still produced."""
    import services.resumes.listing as listing_module

    monkeypatch.setattr(listing_module, "_openrouter_api_key", lambda: None)
    monkeypatch.setattr(listing_module, "_groq_api_key", lambda: None)
    profile = _profile_copy(tmp_path)
    job = extract_job_context_from_message(
        "[JobBank] Embedded Firmware Developer\nhttps://example.com/job"
    )
    assert job is not None

    class _ExplodingModelsApi:
        def generate_content(self, **kwargs):
            raise RuntimeError("provider down")

    result = generate_resume_rewrite(
        settings=GeminiSettings(api_key="test-key", model="gemini-2.5-pro"),
        job=job,
        cache_name=None,
        baseinfo_paths=[profile / "baseinfo.txt"],
        support_paths=[profile / "instructions.txt"],
        template_path=profile / "template.tex",
        scraper=lambda _: _scraped(
            "Embedded firmware for microcontroller products, bare-metal C, digital logic",
            title="Embedded Firmware Developer",
        ),
        client_factory=lambda _: _FakeGeminiClient(_ExplodingModelsApi()),  # type: ignore[arg-type]
    )

    assert result.status == "ok"
    assert result.used_provider is None
    assert result.latex_document is not None
    assert "eebot Mobile Robot System" in result.latex_document
    summary = json.loads(result.rewritten_resume or "{}")
    assert summary["deterministic_fallback"] is True


def test_generate_resume_rewrite_untagged_template_uses_legacy_path(tmp_path: Path) -> None:
    """Profiles without % [category] tags keep the legacy free-form pipeline."""
    template = tmp_path / "template.tex"
    template.write_text(
        "\\documentclass{article}\\begin{document}Legacy\\end{document}", encoding="utf-8"
    )
    baseinfo = tmp_path / "baseinfo.txt"
    baseinfo.write_text("Candidate background.", encoding="utf-8")
    job = extract_job_context_from_message("[Indeed] Analyst\nhttps://example.com/job")
    assert job is not None

    models_api = _FakeModelsApi(
        "<latex>\\documentclass{article}\\begin{document}Hi\\end{document}</latex>"
    )
    result = generate_resume_rewrite(
        settings=GeminiSettings(api_key="test-key", model="gemini-2.5-pro"),
        job=job,
        cache_name=None,
        baseinfo_paths=[baseinfo],
        support_paths=[],
        template_path=template,
        scraper=lambda _: _scraped("General analyst role"),
        client_factory=lambda _: _FakeGeminiClient(models_api),
    )

    assert result.status == "ok"
    assert result.latex_document == "\\documentclass{article}\\begin{document}Hi\\end{document}"


def test_build_structured_prompt_mentions_every_entry() -> None:
    assert CATALOG is not None
    prompt = build_structured_prompt("Software Engineer", "Build APIs", ["Python"], CATALOG)
    for entry in CATALOG.entries:
        assert entry.entry_id in prompt
    assert "role_family" not in prompt
    assert "ONLY a JSON object" in prompt
    assert '"ranking"' in prompt
    assert '"exclude"' in prompt


def test_parse_skill_anchor_display_preserves_original_casing() -> None:
    from services.resumes.structured import parse_skill_anchor_display

    baseinfo = (
        "== SKILL ANCHORS ==\n"
        "Languages: JavaScript/Node.js, Python, C++\n"
        "Tools: PostgreSQL, Git\n"
    )
    display = parse_skill_anchor_display(baseinfo)
    assert display["postgresql"] == "PostgreSQL"
    assert display["git"] == "Git"
    assert display["python"] == "Python"
    # Slash-compound parts get their own display entry too.
    assert display["javascript"] == "JavaScript"
    assert display["node.js"] == "Node.js"
    # No SKILL ANCHORS section -> empty map, matching parse_skill_anchors' ().
    assert parse_skill_anchor_display("no anchors here") == {}


def test_build_structured_prompt_exposes_skill_anchors() -> None:
    """The model must see the candidate's real toolset to weave tools in."""
    assert CATALOG is not None
    prompt = build_structured_prompt("Software Engineer", "Build APIs", ["Python"], CATALOG)
    assert "<skill_anchors>" in prompt
    # Anchors are matched case-insensitively internally, but shown to the
    # model with their original baseinfo casing so a freshly patched-in tool
    # (no existing occurrence to copy casing from) is written correctly.
    assert "Kafka" in prompt
    assert "kafka" not in prompt.replace("Kafka", "")
    assert "PRIMARY GOAL" in prompt

    # Profiles without anchors get no section (check disabled end to end).
    import copy

    bare = copy.deepcopy(CATALOG)
    bare.skill_anchors = ()
    prompt = build_structured_prompt("Engineer", "desc", [], bare)
    assert "<skill_anchors>" not in prompt


def test_prompt_adjacent_tool_leeway_note_tracks_config(monkeypatch) -> None:
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "adjacent_tool_leeway", True)
    prompt = build_structured_prompt("Engineer", "desc", [], CATALOG)
    assert "closely" in prompt and "adjacent" in prompt
    monkeypatch.setattr(CATALOG.render_config, "adjacent_tool_leeway", False)
    prompt = build_structured_prompt("Engineer", "desc", [], CATALOG)
    assert "must never appear" in prompt


def test_prompt_aggressive_note_and_leeway_wording_track_effective_config(monkeypatch) -> None:
    """The prompt must match what the validator will actually allow: with
    aggressive on, the anchor-boundary wording switches to the leeway framing
    even though the profile's own adjacent_tool_leeway is off, and an
    AGGRESSIVE MODE note appears."""
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "adjacent_tool_leeway", False)
    monkeypatch.setattr(CATALOG.render_config, "aggressive", False)
    prompt = build_structured_prompt("Engineer", "desc", [], CATALOG)
    assert "AGGRESSIVE MODE" not in prompt
    assert "must never appear" in prompt

    monkeypatch.setattr(CATALOG.render_config, "aggressive", True)
    prompt = build_structured_prompt("Engineer", "desc", [], CATALOG)
    assert "AGGRESSIVE MODE" in prompt
    assert "KEYWORD INJECTION" in prompt
    assert "any tool from the JD" in prompt
    assert "you may name a tool ONLY when it is closely" not in prompt
    assert "Employers" in prompt and "titles" in prompt and "dates" in prompt
    assert "COHERENCE IS MANDATORY" in prompt
    assert "\\textbf{}" in prompt
    assert "coherent" in prompt.lower()


# ---------------------------------------------------------------------------
# Dissimilar template support (Jake-style macros) + render fidelity
# ---------------------------------------------------------------------------

JAKE_STYLE_TEMPLATE = r"""\documentclass[letterpaper,11pt]{article}
\usepackage{titlesec}
\newcommand{\resumeItem}[1]{\item\small{#1}}
\newcommand{\resumeSubheading}[4]{\item\textbf{#1} \hfill #2 \\ \textit{#3} \hfill #4}
\newcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=0.15in]}
\newcommand{\resumeSubHeadingListEnd}{\end{itemize}}
\newcommand{\resumeItemListStart}{\begin{itemize}}
\newcommand{\resumeItemListEnd}{\end{itemize}}
\begin{document}
\textbf{\Huge Jane Doe} \\ jane@example.com

\section{Experience}
\resumeSubHeadingListStart

% [dotnet]
\resumeSubheading{Software Engineer}{2024 -- 2026}{Contoso}{Remote}
\resumeItemListStart
  \resumeItem{Built C\# microservices handling 2M requests daily.}
  \resumeItem{Cut deployment time by 30\% with automated pipelines.}
\resumeItemListEnd

% [data]
\resumeSubheading{Data Analyst}{2022 -- 2024}{Fabrikam}{Toronto}
\resumeItemListStart
  \resumeItem{Automated SQL reporting for 14 teams.}
\resumeItemListEnd

\resumeSubHeadingListEnd

\section{Education}
\textbf{University} -- BSc \hfill 2018--2022

\end{document}
"""


def test_jake_style_template_parses_and_renders() -> None:
    """Macro-based templates (\\resumeSubheading/\\resumeItem) work end to end."""
    catalog = parse_template_catalog(JAKE_STYLE_TEMPLATE)
    assert catalog is not None
    assert [e.entry_id for e in catalog.entries] == [
        "software-engineer",
        "data-analyst",
    ]
    entry = catalog.entries[0]
    assert entry.item_style == "braced"
    assert entry.item_command == "\\resumeItem"
    assert entry.header.startswith("\\resumeSubheading{Software Engineer}")
    assert entry.bullets[0] == "Built C\\# microservices handling 2M requests daily."
    # Section wrapper macros captured as chrome, not lost.
    prefix, suffix = catalog.section_chrome["Experience"]
    assert prefix == "\\resumeSubHeadingListStart"
    assert suffix == "\\resumeSubHeadingListEnd"

    latex, report = render_structured_resume(catalog, StructuredSelection())
    # Renders with the template's own structure, and fidelity lint is clean.
    assert "\\resumeSubHeadingListStart" in latex
    assert "\\resumeSubHeadingListEnd" in latex
    assert "\\resumeItemListStart" in latex
    assert latex.count("\\resumeItemListStart") == latex.count("\\resumeItemListEnd")
    assert "\\section{Experience}" in latex  # \section, not \section*
    assert "  \\resumeItem{Built C\\# microservices handling 2M requests daily.}" in latex
    assert "\\section{Education}" in latex
    assert report.fidelity_findings == []


def test_char_budget_trims_long_tailored_bullets() -> None:
    """15 long bullets fit the count budget but not one page — chars trim too."""
    assert CATALOG is not None
    long_bullet = "Engineered a comprehensive solution " + "with extensive detail " * 12
    bullets = {
        entry.entry_id: [long_bullet] * len(entry.bullets) for entry in CATALOG.entries
    }
    selection = StructuredSelection(bullets=bullets)
    _, report = render_structured_resume(CATALOG, selection)
    # Char pressure trims below the count ceiling — and may dip up to 2 below
    # the count minimum, because a second page is the worse outcome.
    assert report.visible_bullet_count >= CATALOG.render_config.min_visible_bullets - 2
    assert report.visible_bullet_count < CATALOG.render_config.max_visible_bullets


def test_baseinfo_blocks_parsed_by_tag() -> None:
    from services.resumes.structured import parse_baseinfo_blocks

    blocks, profile = parse_baseinfo_blocks(BASEINFO_TEXT)
    assert "ml" in blocks and "Goopter" in blocks["ml"]
    # Each block is header + a user-editable "Notes:" slot for supplementary
    # facts (not a restatement of the canonical bullets).
    assert "Notes:" in blocks["ml"]
    assert "teaching" in blocks and "Notes:" in blocks["teaching"]
    # Multi-tag entries land under every tag.
    assert "hardware" in blocks and "fpga" in blocks
    assert blocks["hardware"] == blocks["fpga"]
    # Candidate-level facts come from the header.
    assert "Ernest Choi" in profile


def test_prompt_includes_baseinfo_background_per_entry() -> None:
    assert CATALOG is not None
    prompt = build_structured_prompt("ML Engineer", "train models", [], CATALOG)
    assert "verified background" in prompt
    assert "<candidate_facts>" in prompt
    assert "Toronto Metropolitan University" in prompt


def test_baseinfo_fact_not_rejected_as_invented() -> None:
    """A number that exists only in the entry's baseinfo block is legitimate."""
    assert CATALOG is not None
    obotz = next(e for e in CATALOG.entries if "obotz" in e.entry_id)
    background = " ".join(
        CATALOG.baseinfo_blocks.get(tag, "") for tag in obotz.categories
    )
    assert background  # sanity: the teaching block exists
    text, reason = validate_tailored_bullet(
        "Instructed digital logic concepts, improving completion scores by 35\\%.",
        obotz.bullets[2],  # canonical bullet 2 has no 35% figure
        skill_anchors=CATALOG.skill_anchors,
        entry_context=f"{' '.join(obotz.bullets)} {background}",
    )
    assert reason is None


def test_render_fidelity_clean_for_real_profile() -> None:
    assert CATALOG is not None
    for ranking in ([], [_entry_id("goopter"), _entry_id("frontend"), _entry_id("eebot")]):
        _, report = render_structured_resume(CATALOG, StructuredSelection(ranking=ranking))
        assert report.fidelity_findings == []


def test_load_structured_profile_roundtrip() -> None:
    catalog = load_structured_profile(
        PROFILE_DIR / "template.tex", PROFILE_DIR / "baseinfo.txt"
    )
    assert catalog is not None
    assert len(catalog.entries) == 12
    assert catalog.render_config.adjacent_tool_leeway is True


# ---------------------------------------------------------------------------
# Rewrite-scope dial
# ---------------------------------------------------------------------------


def _tailored_bullets_for_all_entries(marker: str) -> dict[str, list[str]]:
    """Grounded rewrites for every entry: canonical text + a trailing marker word."""
    assert CATALOG is not None
    return {
        entry.entry_id: [f"{bullet.rstrip('.')} {marker}." for bullet in entry.bullets]
        for entry in CATALOG.entries
    }


def test_render_config_parses_rewrite_scope_from_file(tmp_path: Path) -> None:
    from services.resumes.structured import RenderConfig

    path = tmp_path / "structured_config.json"
    path.write_text(
        json.dumps({"rewrite_scope": "LIMITED", "limited_rewrite_bullets": 3, "max_bold_per_bullet": 2}),
        encoding="utf-8",
    )
    config = RenderConfig.from_file(path)
    assert config.rewrite_scope == "limited"
    assert config.limited_rewrite_bullets == 3
    assert config.max_bold_per_bullet == 2


def test_render_config_rejects_unknown_rewrite_scope(tmp_path: Path) -> None:
    from services.resumes.structured import RenderConfig

    path = tmp_path / "structured_config.json"
    path.write_text(json.dumps({"rewrite_scope": "yolo"}), encoding="utf-8")
    config = RenderConfig.from_file(path)
    assert config.rewrite_scope == "full"


def test_render_config_parses_adjacent_tool_leeway(tmp_path: Path) -> None:
    from services.resumes.structured import RenderConfig

    path = tmp_path / "structured_config.json"
    path.write_text(json.dumps({"adjacent_tool_leeway": True}), encoding="utf-8")
    assert RenderConfig.from_file(path).adjacent_tool_leeway is True
    path.write_text(json.dumps({}), encoding="utf-8")
    assert RenderConfig.from_file(path).adjacent_tool_leeway is False
    path.write_text(json.dumps({"adjacent_tool_leeway": "yes"}), encoding="utf-8")
    assert RenderConfig.from_file(path).adjacent_tool_leeway is False  # non-bool ignored


def test_render_selection_scope_ignores_all_tailored_bullets(monkeypatch) -> None:
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "rewrite_scope", "selection")
    selection = StructuredSelection(
        bullets=_tailored_bullets_for_all_entries("zzmarker"),
    )
    latex, report = render_structured_resume(CATALOG, selection)
    assert "zzmarker" not in latex
    assert report.tailored_bullets_used == 0
    assert report.rewrite_scope == "selection"
    # Selection decisions still apply: the render is not empty.
    assert report.visible_bullet_count > 0


def test_render_limited_scope_tailors_only_first_n_bullets(monkeypatch) -> None:
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "rewrite_scope", "limited")
    monkeypatch.setattr(CATALOG.render_config, "limited_rewrite_bullets", 4)
    selection = StructuredSelection(
        bullets=_tailored_bullets_for_all_entries("zzmarker"),
    )
    latex, report = render_structured_resume(CATALOG, selection)
    assert report.tailored_bullets_used == 4
    assert latex.count("zzmarker") == 4
    assert report.rewrite_scope == "limited"
    # The tailored bullets are the FIRST ones in document order.
    bullet_lines = [
        line for line in latex.splitlines() if line.strip().startswith("\\item")
    ]
    marked = [i for i, line in enumerate(bullet_lines) if "zzmarker" in line]
    assert marked == [0, 1, 2, 3]


def test_render_full_scope_tailors_everything(monkeypatch) -> None:
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "rewrite_scope", "full")
    selection = StructuredSelection(
        bullets=_tailored_bullets_for_all_entries("zzmarker"),
    )
    latex, report = render_structured_resume(CATALOG, selection)
    assert report.tailored_bullets_used == report.visible_bullet_count
    assert report.rewrite_scope == "full"


def test_prompt_mentions_selection_scope(monkeypatch) -> None:
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "rewrite_scope", "selection")
    prompt = build_structured_prompt("Engineer", "Job description", [], CATALOG)
    assert "RENDERS ALL BULLET TEXT CANONICALLY" in prompt


def test_prompt_mentions_limited_rewrite_budget(monkeypatch) -> None:
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "rewrite_scope", "limited")
    monkeypatch.setattr(CATALOG.render_config, "limited_rewrite_bullets", 5)
    prompt = build_structured_prompt("Engineer", "Job description", [], CATALOG)
    assert "REWRITE BUDGET" in prompt
    assert "first 5" in prompt


def test_prompt_contains_prose_quality_rules() -> None:
    assert CATALOG is not None
    prompt = build_structured_prompt("Engineer", "Job description", [], CATALOG)
    assert "NO EMPTY PURPOSE CLAUSES" in prompt
    assert "NEVER REDUCE SPECIFICITY" in prompt
    assert "VARY SENTENCE SHAPE" in prompt


# ---------------------------------------------------------------------------
# Filler-tail stripping and bold-density cap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tail",
    [
        ", ensuring clean and reliable rendering",
        ", driving successful customer engagement",
        ", prioritizing application safety and data correctness",
        ", enabling seamless collaboration",
        ", streamlining operational workflows",
    ],
)
def test_validate_bullet_strips_empty_purpose_clause_tails(tail: str) -> None:
    text, reason = validate_tailored_bullet(
        f"Maintained \\textbf{{LaTeX}} documentation for the team{tail}.",
        CANONICAL,
    )
    assert reason is None
    assert text is not None
    assert text == "Maintained \\textbf{LaTeX} documentation for the team."


def test_validate_bullet_filler_strip_rejects_degenerate_stub() -> None:
    """When almost nothing remains before the filler connector, stripping would
    leave a stub — the bullet still falls back to canonical."""
    text, reason = validate_tailored_bullet(
        "Built a cache while ensuring stakeholder alignment throughout delivery.",
        "Built a cache layer, cutting latency by 40\\%.",
    )
    assert text is None
    assert "filler clause" in str(reason)


def test_validate_bullet_allows_filler_gerund_present_in_canonical() -> None:
    canonical = "Built reinforcement algorithms using \\textbf{TensorFlow}, enabling adaptive signal generation."
    text, reason = validate_tailored_bullet(
        "Built reinforcement algorithms with \\textbf{TensorFlow}, enabling adaptive signal generation.",
        canonical,
    )
    assert reason is None
    assert text is not None


def test_validate_bullet_demotes_bold_beyond_density_cap() -> None:
    canonical = (
        "Built pipelines using \\textbf{Kafka}, \\textbf{FastAPI}, \\textbf{Python}, "
        "\\textbf{Pandas}, and \\textbf{NumPy} for 41\\% faster processing."
    )
    text, reason = validate_tailored_bullet(
        "Built pipelines using \\textbf{Kafka}, \\textbf{FastAPI}, \\textbf{Python}, "
        "\\textbf{Pandas}, and \\textbf{NumPy} for 41\\% faster processing.",
        canonical,
    )
    assert reason is None
    assert text is not None
    assert text.count("\\textbf{") == 4  # default cap
    # Demoted tools remain as plain text.
    assert "NumPy" in text
    assert "\\textbf{NumPy}" not in text


def test_validate_bullet_bold_cap_keeps_listing_relevant_bolds() -> None:
    canonical = (
        "Built tools with \\textbf{Python}, \\textbf{Kafka}, \\textbf{Pandas}, "
        "and \\textbf{NumPy} plus \\textbf{Docker}."
    )
    text, reason = validate_tailored_bullet(
        canonical,
        canonical,
        listing_keywords=("Docker", "containerization"),
    )
    assert reason is None
    assert text.count("\\textbf{") == 4  # default cap
    # Docker matches the listing -> kept despite being the last bold.
    assert "\\textbf{Docker}" in text
    assert "\\textbf{NumPy}" not in text


# ---------------------------------------------------------------------------
# Skills-section reordering
# ---------------------------------------------------------------------------


def test_reorder_skills_listing_keywords_match_whole_words_only() -> None:
    from services.resumes.structured import reorder_skills_for_keywords

    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Languages:} JavaScript/Node.js, Java, Python, C++, C, CSS \\\\\n"
    )
    # None of these keywords name C — the letter c inside "react"/"recruitment"
    # must not float the C item to the front.
    result = reorder_skills_for_keywords(
        tail, ["React", "recruitment", "communication skills", "CSS"]
    )
    languages = next(l for l in result.splitlines() if "Languages" in l)
    items = [i.strip() for i in languages.split("}", 1)[1].rstrip(" \\").split(",")]
    assert items[0] == "CSS"
    # C keeps its canonical position (after C++), not promoted.
    assert items.index("C") == items.index("C++") + 1
    # A keyword that genuinely names the item as a whole word still promotes it.
    result = reorder_skills_for_keywords(tail, ["embedded C firmware"])
    languages = next(l for l in result.splitlines() if "Languages" in l)
    items = languages.split("}", 1)[1].rstrip(" \\").strip()
    assert items.startswith("C,")


def test_reorder_skills_trims_unmatched_items_beyond_cap() -> None:
    from services.resumes.structured import reorder_skills_for_keywords

    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Languages:} JavaScript/Node.js, Java, Python, C++, C, SQL, "
        "HTML, CSS, GLSL, VHDL, Verilog, MATLAB \\\\\n"
    )
    result = reorder_skills_for_keywords(
        tail,
        ["JavaScript", "HTML", "CSS"],
        max_items_per_line=6,
    )
    languages = next(l for l in result.splitlines() if "Languages" in l)
    items = [i.strip() for i in languages.split("}", 1)[1].rstrip(" \\").split(",")]
    assert len(items) == 6
    # Matched items always survive.
    for kept in ("JavaScript/Node.js", "HTML", "CSS"):
        assert kept in items
    # Unmatched tail items beyond the cap are cut.
    for cut in ("VHDL", "Verilog", "MATLAB"):
        assert cut not in items
    # Cap of 0 (default) preserves every item — reorder-only legacy behavior.
    result_off = reorder_skills_for_keywords(tail, ["HTML"])
    languages_off = next(l for l in result_off.splitlines() if "Languages" in l)
    items_off = [i.strip() for i in languages_off.split("}", 1)[1].rstrip(" \\").split(",")]
    assert len(items_off) == 12
    # Matched items always survive even when they alone exceed the cap.
    result_over = reorder_skills_for_keywords(
        tail,
        ["JavaScript", "Java", "Python", "C++", "SQL", "HTML", "CSS"],
        max_items_per_line=2,
    )
    languages_over = next(l for l in result_over.splitlines() if "Languages" in l)
    items_over = [i.strip() for i in languages_over.split("}", 1)[1].rstrip(" \\").split(",")]
    assert len(items_over) == 7


def test_skills_reorder_moves_matching_tools_first() -> None:
    from services.resumes.structured import reorder_skills_for_keywords

    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Tools:} Git, PostgreSQL, Ubuntu Linux, Docker, Kafka, Tableau \\\\\n"
        "\\textbf{Platforms:} GitHub, GitLab, Raspberry Pi, Arduino\n"
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree \\hfill 2023--2027"
    )
    result = reorder_skills_for_keywords(tail, ["Kafka", "Docker"])
    tools_line = next(l for l in result.splitlines() if "Tools:" in l)
    items = tools_line.split("}", 1)[1].rstrip(" \\").split(", ")
    assert items[0].strip() in ("Docker", "Kafka")
    assert items[1].strip() in ("Docker", "Kafka")
    # Content preserved exactly: same items, same count, trailing break intact.
    assert sorted(i.strip() for i in items) == sorted(
        ["Git", "PostgreSQL", "Ubuntu Linux", "Docker", "Kafka", "Tableau"]
    )
    assert tools_line.rstrip().endswith("\\\\")
    # Education line untouched.
    assert "\\textbf{School} -- Degree \\hfill 2023--2027" in result


def test_skills_reorder_noop_without_keywords() -> None:
    from services.resumes.structured import reorder_skills_for_keywords

    assert CATALOG is not None
    assert reorder_skills_for_keywords(CATALOG.tail, []) == CATALOG.tail


def test_render_reorders_real_skills_section() -> None:
    assert CATALOG is not None
    selection = StructuredSelection(keywords=["Kafka", "TensorFlow"])
    latex, _ = render_structured_resume(CATALOG, selection)
    tools_line = next(l for l in latex.splitlines() if l.startswith("\\textbf{Tools:}"))
    assert tools_line.index("Kafka") < tools_line.index("Git")


# ---------------------------------------------------------------------------
# Header tech retargeting + listing-aware emphasis
# ---------------------------------------------------------------------------


def test_extract_json_object_salvages_truncated_response() -> None:
    # Response dies mid-bullet (output-token limit): everything that arrived
    # intact must survive.
    truncated = (
        '{"keywords": ["Python", "PyTorch"], '
        '"bullets": {"entry-a": ["Full bullet zero.", "Full bullet one."], '
        '"entry-b": ["Partial bullet that never fini'
    )
    payload = extract_json_object(truncated)
    assert payload is not None
    assert payload["keywords"] == ["Python", "PyTorch"]
    assert payload["bullets"]["entry-a"] == ["Full bullet zero.", "Full bullet one."]
    # Truncation cut inside a key: dangling pair is dropped, prefix survives.
    truncated2 = '{"keywords": ["EMBEDDED"], "ranking": ["a", "b"], "weak_bul'
    payload2 = extract_json_object(truncated2)
    assert payload2 is not None
    assert payload2["ranking"] == ["a", "b"]


def test_extract_listing_keywords_deterministic() -> None:
    from services.resumes.structured import extract_listing_keywords

    jd = (
        "Embedded Firmware Intern\n"
        "We build sensor products. You will write firmware in C for STM32 "
        "microcontrollers, debug with logic analyzers, and work with I2C and SPI "
        "peripherals. Experience with Python scripting and Git is required. "
        "Firmware testing and sensor calibration are part of the role."
    )
    keywords = extract_listing_keywords(jd, ("C", "Python", "Git", "Kafka", "React"))
    lowered = [k.lower() for k in keywords]
    # Anchors the listing names come first; anchors it doesn't stay out.
    assert "c" in lowered and "python" in lowered and "git" in lowered
    assert "kafka" not in lowered and "react" not in lowered
    # Frequent content words appear.
    assert "firmware" in lowered
    assert len(keywords) <= 10


def test_filler_clause_and_while_forms_stripped_or_rejected() -> None:
    canonical = "Built a cache layer, cutting latency by 40\\%."
    # Long remainder: the filler tail is stripped, the rewrite survives.
    text, reason = validate_tailored_bullet(
        "Built a cache layer, cutting latency by 40\\% and demonstrating continuous improvement.",
        canonical,
    )
    assert reason is None
    assert text is not None
    assert "demonstrating" not in text
    assert "40\\%" in text
    # Short remainder: stripping would leave a stub, so it still rejects.
    text, reason = validate_tailored_bullet(
        "Built a cache layer while ensuring stakeholder alignment.",
        canonical,
    )
    assert text is None
    assert "filler" in reason


def test_retarget_header_tech_grounded_swap_and_casing() -> None:
    from services.resumes.structured import retarget_header_tech

    header = "\\textbf{Web Messaging App} | \\textit{JavaScript, Node.js, SQL} \\\\"
    grounding = (
        "Built a platform with Node.js, JavaScript, and WebRTC. "
        "Used a Raspberry Pi with PostgreSQL as a RESTful API backend."
    )
    new_header, reason = retarget_header_tech(
        header, ["webrtc", "Node.js", "PostgreSQL"], grounding
    )
    assert reason is None
    # Casing comes from the grounding text, not the model.
    assert "\\textit{WebRTC, Node.js, PostgreSQL}" in new_header
    assert new_header.startswith("\\textbf{Web Messaging App} | ")
    assert new_header.endswith("\\\\")


def test_retarget_header_tech_rejects_ungrounded_and_markup() -> None:
    from services.resumes.structured import retarget_header_tech

    header = "\\textbf{App} | \\textit{Java} \\\\"
    grounding = "Built an app in Java."
    for bad in (["Kubernetes"], ["Java", "Rust"], ["\\textbf{Java}"], []):
        new_header, reason = retarget_header_tech(header, bad, grounding)
        assert new_header is None
        assert reason
    # Header without a \textit list cannot be retargeted.
    new_header, reason = retarget_header_tech(
        "\\textbf{Intern,} {Company} -- Remote \\\\", ["Java"], grounding
    )
    assert new_header is None


def test_emphasize_listing_tools_bolds_wanted_anchors_only() -> None:
    from services.resumes.structured import emphasize_listing_tools

    text = "Built pipelines with Kafka and Pandas on \\textbf{Ubuntu Linux}."
    out = emphasize_listing_tools(
        text,
        skill_anchors=("Kafka", "Pandas", "Git"),
        keywords=["Kafka", "streaming data"],
        max_bold=3,
    )
    # Kafka is anchor + keyword -> bolded; Pandas is anchor but not asked for.
    assert "\\textbf{Kafka}" in out
    assert "\\textbf{Pandas}" not in out
    # Existing bold untouched, no nesting.
    assert "\\textbf{Ubuntu Linux}" in out
    assert "\\textbf{\\textbf" not in out


def test_emphasize_listing_tools_respects_cap_and_existing_bolds() -> None:
    from services.resumes.structured import emphasize_listing_tools

    text = "Used \\textbf{A}, \\textbf{B}, and \\textbf{D} with Kafka daily."
    out = emphasize_listing_tools(text, ("Kafka",), ["Kafka"], max_bold=3)
    assert out == text  # already at cap
    # Already-bolded tool is never double-wrapped elsewhere.
    text2 = "\\textbf{Kafka} pipelines; Kafka consumers."
    out2 = emphasize_listing_tools(text2, ("Kafka",), ["Kafka"], max_bold=3)
    assert out2 == text2


def test_render_applies_grounded_header_tech_with_clean_fidelity() -> None:
    assert CATALOG is not None
    entry = CATALOG.entry("web-messaging-app")
    assert entry is not None
    selection = StructuredSelection(
        ranking=["web-messaging-app", "particle-fluid-simulation"],
        header_tech={
            "web-messaging-app": ["PostgreSQL", "Node.js", "WebRTC"],
            # Ungrounded for this entry -> rejected, header stays canonical.
            "particle-fluid-simulation": ["Kafka"],
        },
    )
    latex, report = render_structured_resume(CATALOG, selection)
    assert report.header_tech_applied == ["web-messaging-app"]
    assert any(r.startswith("particle-fluid-simulation:") for r in report.header_tech_rejected)
    assert "\\textit{PostgreSQL, Node.js, WebRTC}" in latex
    assert entry.header not in latex  # replaced
    assert report.fidelity_findings == []


# ---------------------------------------------------------------------------
# Edge cases: parser hardening
# ---------------------------------------------------------------------------


def _mini_template(entry_block: str) -> str:
    return (
        "\\documentclass{article}\n\\begin{document}\n"
        "\\centerline{Name}\n\n"
        "\\section*{Projects}\n\n" + entry_block + "\n\n"
        "\\section*{Skills}\nstuff\n\n"
        "\\end{document}\n"
    )


def test_parser_strips_comment_wrapper_between_tag_and_header() -> None:
    """A \\begin{comment} leaking between tag and header must not reach output."""
    block = (
        "% [one]\n"
        "\\begin{comment}\n"
        "\\textbf{Entry One} \\\\\n"
        "\\begin{itemize}\n  \\item Did a thing.\n\\end{itemize}\n"
        "\\end{comment}"
    )
    catalog = parse_template_catalog(_mini_template(block))
    assert catalog is not None
    entry = catalog.entries[0]
    assert "\\begin{comment}" not in entry.header
    assert entry.header.startswith("\\textbf{Entry One}")

    latex, _ = render_structured_resume(catalog, StructuredSelection())
    assert "\\begin{comment}" not in latex
    assert latex.count("\\begin{itemize}") == latex.count("\\end{itemize}")


def test_parser_detects_smallskip_per_entry_not_per_section() -> None:
    assert CATALOG is not None
    experience = [e for e in CATALOG.entries if e.section == "Experience"]
    projects = [e for e in CATALOG.entries if e.section == "Projects"]
    assert all(e.smallskip for e in experience)
    assert not any(e.smallskip for e in projects)


def test_parser_multiline_header_preserved() -> None:
    block = (
        "% [one]\n"
        "\\textbf{Entry One} |\n\\textit{Tools} \\\\\n"
        "\\begin{itemize}\n  \\item Did a thing.\n\\end{itemize}"
    )
    catalog = parse_template_catalog(_mini_template(block))
    assert catalog is not None
    assert "\\textbf{Entry One} |" in catalog.entries[0].header
    assert "\\textit{Tools} \\\\" in catalog.entries[0].header


def test_parser_skips_entry_with_empty_itemize() -> None:
    block = (
        "% [one]\n\\textbf{Empty} \\\\\n\\begin{itemize}\n\\end{itemize}\n\n"
        "% [two]\n\\textbf{Real} \\\\\n\\begin{itemize}\n  \\item Something.\n\\end{itemize}"
    )
    catalog = parse_template_catalog(_mini_template(block))
    assert catalog is not None
    assert [e.title for e in catalog.entries] == ["Real"]


def test_parser_duplicate_titles_get_unique_ids() -> None:
    block = (
        "% [one]\n\\textbf{Same} \\\\\n\\begin{itemize}\n  \\item A.\n\\end{itemize}\n\n"
        "% [two]\n\\textbf{Same} \\\\\n\\begin{itemize}\n  \\item B.\n\\end{itemize}"
    )
    catalog = parse_template_catalog(_mini_template(block))
    assert catalog is not None
    ids = [e.entry_id for e in catalog.entries]
    assert len(ids) == len(set(ids)) == 2


# ---------------------------------------------------------------------------
# Edge cases: response payload tolerance
# ---------------------------------------------------------------------------

def test_parse_response_accepts_title_keyed_bullets() -> None:
    assert CATALOG is not None
    goopter = next(e for e in CATALOG.entries if "goopter" in e.entry_id)
    selection = parse_structured_response(
        json.dumps({"bullets": {goopter.title: ["Rewritten."]}}),
        CATALOG,
    )
    assert selection is not None
    assert goopter.entry_id in selection.bullets


def test_parse_response_accepts_dict_wrapped_bullets() -> None:
    assert CATALOG is not None
    entry = CATALOG.entries[0]
    selection = parse_structured_response(
        json.dumps({"bullets": {entry.entry_id: [{"text": "Wrapped bullet."}]}}),
        CATALOG,
    )
    assert selection is not None
    assert selection.bullets[entry.entry_id] == ["Wrapped bullet."]


def test_parse_response_deduplicates_ranking() -> None:
    assert CATALOG is not None
    entry = CATALOG.entries[0]
    selection = parse_structured_response(
        json.dumps({"ranking": [entry.entry_id, entry.entry_id]}),
        CATALOG,
    )
    assert selection is not None
    assert selection.ranking == [entry.entry_id]


# ---------------------------------------------------------------------------
# Edge cases: validation
# ---------------------------------------------------------------------------

def test_validate_bullet_rejects_caret_outside_math() -> None:
    text, reason = validate_tailored_bullet("Improved x^41 throughput", CANONICAL)
    assert text is None and reason == "caret outside math mode"
    # ...but inside math mode it is fine.
    text, reason = validate_tailored_bullet("Improved $x^{41}$ throughput", CANONICAL)
    assert reason is None


# ---------------------------------------------------------------------------
# Edge cases: per-profile config + guidance
# ---------------------------------------------------------------------------

def test_structured_config_overrides_bullet_budget(tmp_path: Path) -> None:
    profile = _profile_copy(tmp_path)
    (profile / "structured_config.json").write_text(
        json.dumps({"min_visible_bullets": 8, "max_visible_bullets": 11}),
        encoding="utf-8",
    )
    catalog = load_structured_profile(profile / "template.tex", profile / "baseinfo.txt")
    assert catalog is not None
    assert catalog.render_config.max_visible_bullets == 11
    _, report = render_structured_resume(catalog, StructuredSelection())
    assert report.visible_bullet_count <= 11


def test_structured_config_invalid_json_falls_back_to_defaults(tmp_path: Path) -> None:
    profile = _profile_copy(tmp_path)
    (profile / "structured_config.json").write_text("{not json", encoding="utf-8")
    catalog = load_structured_profile(profile / "template.tex", profile / "baseinfo.txt")
    assert catalog is not None
    assert catalog.render_config.max_visible_bullets == MAX_VISIBLE_BULLETS


def test_structured_guidance_flows_into_prompt(tmp_path: Path) -> None:
    profile = _profile_copy(tmp_path)
    (profile / "instructions.txt").write_text(
        "Voice: sample guidance marker.", encoding="utf-8"
    )
    catalog = load_structured_profile(profile / "template.tex", profile / "baseinfo.txt")
    assert catalog is not None
    prompt = build_structured_prompt(
        "Engineer", "desc", [], catalog, extra_guidance=catalog.guidance
    )
    assert "sample guidance marker" in prompt
    assert "<profile_guidance>" in prompt


# ---------------------------------------------------------------------------
# Entry exclusion + capacity rollback + weak-bullet trimming
# ---------------------------------------------------------------------------

def test_exclusion_honoured_when_page_can_still_be_filled() -> None:
    assert CATALOG is not None
    obotz = _entry_id("obotz")
    selection = StructuredSelection(exclusions=[obotz])
    latex, report = render_structured_resume(CATALOG, selection)
    assert "Obotz Robotics" not in latex
    assert obotz in report.excluded_entries
    # Page must still be filled from remaining entries.
    assert report.visible_bullet_count >= CATALOG.render_config.min_visible_bullets


def test_top_ranked_entry_can_be_excluded() -> None:
    """No entry has category immunity: excluding the highest-ranked entry
    drops it as long as the page can be filled without it."""
    assert CATALOG is not None
    goopter = _entry_id("goopter")
    selection = StructuredSelection(ranking=[goopter], exclusions=[goopter])
    latex, report = render_structured_resume(CATALOG, selection)
    assert "Goopter" not in latex
    assert goopter in report.excluded_entries
    assert report.visible_bullet_count >= CATALOG.render_config.min_visible_bullets


def test_exclusions_rolled_back_when_page_cannot_be_filled() -> None:
    """Excluding every entry must be rolled back to reach the minimum."""
    assert CATALOG is not None
    all_ids = [e.entry_id for e in CATALOG.entries]
    selection = StructuredSelection(exclusions=all_ids)
    _, report = render_structured_resume(CATALOG, selection)
    assert report.visible_bullet_count >= CATALOG.render_config.min_visible_bullets
    assert report.ignored_exclusions  # some exclusions were re-admitted
    # Honoured + ignored must account for every requested exclusion.
    assert set(report.excluded_entries) | set(report.ignored_exclusions) == set(all_ids)


def test_weak_bullet_hint_steers_trimming() -> None:
    """When over budget, the flagged weak bullet is cut instead of the last one."""
    assert CATALOG is not None
    from services.resumes.structured import RenderConfig
    import copy

    catalog = copy.deepcopy(CATALOG)
    # Force heavy trimming with a tiny budget.
    catalog.render_config = RenderConfig(min_visible_bullets=8, max_visible_bullets=10)
    pfs = next(e for e in catalog.entries if "particle" in e.entry_id)
    selection = StructuredSelection(
        weak_bullets={pfs.entry_id: [0]},  # flag the FIRST bullet as weakest
    )
    latex, report = render_structured_resume(catalog, selection)
    if pfs.entry_id in report.visible_entries:
        rendered_after_pfs = latex.split(pfs.header, 1)[1].split("\\end{itemize}", 1)[0]
        kept_first = pfs.bullets[0].split("\\", 1)[0][:40] in rendered_after_pfs
        kept_last = pfs.bullets[-1][:40] in rendered_after_pfs
        # If trimming touched this entry, bullet 0 goes before the last one.
        if not (kept_first and kept_last):
            assert kept_last and not kept_first


def test_tool_moved_within_entry_is_not_invented() -> None:
    """A tool from bullet 0 mentioned in bullet 1 of the SAME entry is grounded."""
    from services.resumes.structured import parse_skill_anchors

    anchors = parse_skill_anchors(BASEINFO_TEXT)
    entry_context = (
        "Used \\textbf{LaTeX} to maintain documentation, reducing downtime by 41\\%. "
        "Performed system validation and testing, reducing reported errors by 20\\%."
    )
    text, reason = validate_tailored_bullet(
        "Performed system validation with \\textbf{LaTeX} test docs, reducing reported errors by 20\\%.",
        "Performed system validation and testing, reducing reported errors by 20\\%.",
        skill_anchors=anchors,
        entry_context=entry_context,
    )
    assert reason is None
    # But a tool from nowhere in the entry is still rejected (default config).
    text, reason = validate_tailored_bullet(
        "Performed system validation with \\textbf{Terraform}, reducing reported errors by 20\\%.",
        "Performed system validation and testing, reducing reported errors by 20\\%.",
        skill_anchors=anchors,
        entry_context=entry_context,
    )
    assert text is None and "invented tool" in str(reason)


def test_bullet_order_front_loads_relevant_bullet() -> None:
    """bullet_order reorders an entry's bullets, most relevant first."""
    assert CATALOG is not None
    markham = next(e for e in CATALOG.entries if "markham" in e.entry_id)
    selection = StructuredSelection(
        ranking=[markham.entry_id],
        bullet_order={markham.entry_id: [2, 0, 1]},
    )
    latex, report = render_structured_resume(CATALOG, selection)
    assert markham.entry_id in report.visible_entries
    block = latex.split(markham.header, 1)[1].split("\\end{itemize}", 1)[0]
    first_bullet = block.split("\\item", 2)[1]
    assert "20\\%" in first_bullet  # canonical bullet 2 leads


def test_bullet_order_invalid_indices_ignored() -> None:
    assert CATALOG is not None
    markham = next(e for e in CATALOG.entries if "markham" in e.entry_id)
    selection = StructuredSelection(
        ranking=[markham.entry_id],
        bullet_order={markham.entry_id: [9, 7]},  # out of range
    )
    latex, report = render_structured_resume(CATALOG, selection)
    # All three canonical bullets still render, original order.
    assert "41\\%" in latex and "14\\%" in latex and "20\\%" in latex


def test_parse_response_reads_bullet_order() -> None:
    assert CATALOG is not None
    entry = CATALOG.entries[0]
    selection = parse_structured_response(
        json.dumps({"bullet_order": {entry.entry_id: [1, 0]}}),
        CATALOG,
    )
    assert selection is not None
    assert selection.bullet_order == {entry.entry_id: [1, 0]}


def test_parse_response_reads_exclude_and_weak_bullets() -> None:
    assert CATALOG is not None
    obotz = next(e for e in CATALOG.entries if "obotz" in e.entry_id)
    pfs = next(e for e in CATALOG.entries if "particle" in e.entry_id)
    selection = parse_structured_response(
        json.dumps(
            {
                "exclude": [obotz.entry_id, "unknown-entry"],
                "weak_bullets": {pfs.entry_id: [2, 0], "unknown-entry": [1]},
            }
        ),
        CATALOG,
    )
    assert selection is not None
    assert selection.exclusions == [obotz.entry_id]
    assert selection.weak_bullets == {pfs.entry_id: [2, 0]}


def test_prompt_requests_core_work_before_ranking() -> None:
    """The objects-of-work scaffold: the model must articulate what each entry
    actually built/operated BEFORE ranking, and the ranking notes must tell it
    that shared process words alone (debugging, testing) do not count."""
    assert CATALOG is not None
    prompt = build_structured_prompt("DevOps Intern", "Deploy cloud services", [], CATALOG)
    assert '"core_work"' in prompt
    # Schema order is generation order: core_work must precede ranking so the
    # model writes the phrases before it ranks.
    assert prompt.index('"core_work"') < prompt.index('"ranking"')
    assert "generic process words" in prompt
    assert "core_work FIRST" in prompt


def test_parse_response_reads_core_work() -> None:
    assert CATALOG is not None
    entry = CATALOG.entries[0]
    other = CATALOG.entries[1]
    selection = parse_structured_response(
        json.dumps(
            {
                "core_work": {
                    entry.entry_id: "  GPU particle shaders  ",
                    # Title-keyed entries resolve like every other id map.
                    other.title: "web messaging platform",
                    "unknown-entry": "nothing",
                    entry.entry_id + "-bogus": 42,
                },
            }
        ),
        CATALOG,
    )
    assert selection is not None
    assert selection.core_work == {
        entry.entry_id: "GPU particle shaders",
        other.entry_id: "web messaging platform",
    }


def test_profile_cache_purge_keeps_structured_files(tmp_path: Path) -> None:
    """ensure_profile_cache must never delete structured_config.json.

    Regression: the purge originally allowed only the three legacy files, so
    the structured pipeline's optional config file was wiped on every bot run.
    (Structured prompt guidance now lives in instructions.txt itself — an
    always-allowed file — rather than a separate optional guidance file.)
    """
    from services.resumes.resume import ensure_profile_cache

    cache_root = tmp_path / "cache"
    example = cache_root / "example"
    example.mkdir(parents=True)
    for name in ("baseinfo.txt", "instructions.txt", "template.tex"):
        (example / name).write_text("seed", encoding="utf-8")

    profile = cache_root / "user.profile"
    profile.mkdir()
    (profile / "structured_config.json").write_text("{}", encoding="utf-8")
    (profile / "junk.tmp").write_text("junk", encoding="utf-8")

    ensure_profile_cache("user.profile", cache_root)

    assert (profile / "structured_config.json").exists()
    assert not (profile / "junk.tmp").exists()  # real junk still purged


def test_profile_guidance_file_is_loaded_for_real_profile() -> None:
    catalog = load_structured_profile(
        PROFILE_DIR / "template.tex", PROFILE_DIR / "baseinfo.txt"
    )
    assert catalog is not None
    assert "Seniority" in catalog.guidance


# ---------------------------------------------------------------------------
# Portability: a completely different person's profile, same conventions
# ---------------------------------------------------------------------------

JANE_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage{enumitem}
\begin{document}

\centerline{\Huge Jane Doe}
\centerline{jane.doe@example.com | 555-0000}

\section*{Experience}

\smallskip
% [dotnet]
\textbf{Software Engineer,} {Contoso} -- Toronto \hfill 2024 -- Present \\
\begin{itemize}
  \item Built C\# microservices on .NET handling 2M requests daily.
  \item Cut deployment time by 30\% with automated Azure pipelines.
\end{itemize}

\smallskip
% [data]
\textbf{Data Analyst,} {Fabrikam} -- Remote \hfill 2022 -- 2024 \\
\begin{itemize}
  \item Automated reporting with SQL and Excel, saving 10 hours weekly.
  \item Modeled churn with logistic regression, improving retention 5\%.
\end{itemize}

\section*{Skills}
\textbf{Languages:} C\#, SQL, Python

\section*{Education}
\textbf{Some University} -- BSc \hfill 2018--2022

\end{document}
"""

JANE_BASEINFO = """
Jane Doe profile facts.
"""


def _jane_profile(tmp_path: Path) -> Path:
    profile = tmp_path / "jane"
    profile.mkdir()
    (profile / "template.tex").write_text(JANE_TEMPLATE, encoding="utf-8")
    (profile / "baseinfo.txt").write_text(JANE_BASEINFO, encoding="utf-8")
    (profile / "instructions.txt").write_text("n/a", encoding="utf-8")
    # Jane really works with C#/.NET (no skill anchors declared for her
    # profile, so the invented-tool check has nothing to compare against);
    # use a bullet budget matching her small resume.
    (profile / "structured_config.json").write_text(
        json.dumps(
            {
                "min_visible_bullets": 3,
                "max_visible_bullets": 4,
            }
        ),
        encoding="utf-8",
    )
    return profile


def test_second_profile_same_conventions_works_end_to_end(tmp_path: Path) -> None:
    profile = _jane_profile(tmp_path)
    catalog = load_structured_profile(profile / "template.tex", profile / "baseinfo.txt")
    assert catalog is not None
    assert [e.entry_id for e in catalog.entries] == [
        "software-engineer-contoso",
        "data-analyst-fabrikam",
    ]
    assert all(e.smallskip for e in catalog.entries)

    job = extract_job_context_from_message("[LinkedIn] Backend Developer\nhttps://example.com/job")
    assert job is not None
    response = json.dumps(
        {
            "ranking": ["software-engineer-contoso", "data-analyst-fabrikam"],
            "bullets": {
                "software-engineer-contoso": [
                    "Engineered C\\# microservices on .NET processing 2M requests daily.",
                    "Cut deployment time by 30\\% via automated Azure pipelines.",
                ]
            },
        }
    )
    result = generate_resume_rewrite(
        settings=GeminiSettings(api_key="test-key", model="gemini-2.5-pro"),
        job=job,
        cache_name=None,
        baseinfo_paths=[profile / "baseinfo.txt"],
        support_paths=[profile / "instructions.txt"],
        template_path=profile / "template.tex",
        scraper=lambda _: _scraped("Backend microservices with .NET and APIs", title="Backend Developer"),
        client_factory=lambda _: _FakeGeminiClient(_FakeModelsApi(response)),
    )
    assert result.status == "ok"
    latex = result.latex_document or ""
    assert "Jane Doe" in latex
    assert "jane.doe@example.com" in latex
    # C#/.NET rewrite accepted because Jane's config replaced the defaults.
    assert "Engineered C\\# microservices on .NET" in latex
    assert "Ernest" not in latex
    summary = json.loads(result.rewritten_resume or "{}")
    assert summary["tailored_bullets_used"] == 2


def test_all_selection_knobs_together_produce_coherent_report() -> None:
    """Umbrella use-case test: every selection-level modding knob exercised in
    one build — ranking, exclusion, weak-bullet steering, bullet ordering, and
    listing keywords — must produce a coherent report and a fidelity-clean
    document."""
    assert CATALOG is not None
    goopter = next(e for e in CATALOG.entries if "goopter" in e.entry_id)
    obotz = next(e for e in CATALOG.entries if "obotz" in e.entry_id)
    terrain = next(e for e in CATALOG.entries if "terrain" in e.entry_id)
    markham = next(e for e in CATALOG.entries if "markham" in e.entry_id)

    selection = StructuredSelection(
        ranking=[goopter.entry_id, terrain.entry_id, markham.entry_id],
        keywords=["python", "kafka"],
        exclusions=[obotz.entry_id],
        weak_bullets={markham.entry_id: [0]},
        bullet_order={goopter.entry_id: list(range(len(goopter.bullets)))[::-1]},
    )
    latex, report = render_structured_resume(CATALOG, selection)

    # Ranked entries lead the page; the exclusion is accounted for either way.
    assert goopter.entry_id in report.visible_entries
    assert terrain.entry_id in report.visible_entries
    assert "Goopter" in latex
    assert obotz.entry_id in set(report.excluded_entries) | set(report.ignored_exclusions)

    # Bullet order applied to Goopter: its rendered first bullet is the
    # canonical LAST bullet when the reversal order was honoured.
    if len(goopter.bullets) > 1:
        goopter_block = latex.split(goopter.header, 1)[1]
        first_positions = {
            index: goopter_block.find(bullet[:40])
            for index, bullet in enumerate(goopter.bullets)
            if goopter_block.find(bullet[:40]) != -1
        }
        if len(first_positions) > 1:
            rendered_first = min(first_positions, key=first_positions.get)
            assert rendered_first == len(goopter.bullets) - 1

    # Budget + structural coherence.
    assert report.visible_bullet_count <= CATALOG.render_config.max_visible_bullets
    assert report.visible_bullet_count >= CATALOG.render_config.min_visible_bullets - 2
    assert report.fidelity_findings == []
    assert latex.count("\\end{document}") == 1
    assert not (set(report.visible_entries) & set(report.hidden_entries))


def test_aggressive_char_trim_floor_dips_deeper_than_normal() -> None:
    """Aggressive mode sets char_trim_floor = min_visible_bullets - 5 (vs -2).

    With long-enough tailored bullets the char budget is still exceeded at the
    normal floor of 12 bullets.  A normal build stops there (second page is
    accepted); an aggressive build keeps trimming to ~9-11, preventing overflow.
    """
    import copy

    normal_cat = copy.deepcopy(CATALOG)
    aggressive_cat = copy.deepcopy(CATALOG)
    aggressive_cat.render_config.aggressive = True

    long_bullet = "Implemented distributed " + "microservice orchestration " * 9
    bullets = {
        entry.entry_id: [long_bullet] * len(entry.bullets)
        for entry in CATALOG.entries
    }
    selection = StructuredSelection(bullets=bullets)

    _, normal_report = render_structured_resume(normal_cat, selection)
    _, aggr_report = render_structured_resume(aggressive_cat, selection)

    normal_floor = normal_cat.render_config.min_visible_bullets - 2  # 12
    aggressive_floor = max(aggressive_cat.render_config.min_visible_bullets - 5, 1)  # 9

    assert normal_report.visible_bullet_count >= normal_floor
    assert aggr_report.visible_bullet_count >= aggressive_floor
    assert aggr_report.visible_bullet_count < normal_report.visible_bullet_count


def test_provider_order_gemini_first() -> None:
    """Provider order always starts with Gemini (including aggressive)."""
    from services.resumes.listing import _provider_switch_candidates

    settings = GeminiSettings(
        api_key="fake-gemini",
        model="gemini-2.5-flash",
        openrouter_api_key="fake-openrouter",
        groq_api_key="fake-groq",
    )

    candidates = _provider_switch_candidates(settings)
    names = [p.name for p in candidates]

    assert names[0] == "gemini"
    assert set(names) == {"gemini", "gemini-flash", "openrouter", "groq"}


# ---------------------------------------------------------------------------
# Template ↔ baseinfo auto-sync
# ---------------------------------------------------------------------------

from services.resumes.structured import (
    _detect_untagged_entries,
    _infer_tag,
    sync_template_baseinfo,
)


def test_detect_untagged_entry(tmp_path: Path) -> None:
    """An entry without a % [tag] line is detected as untagged."""
    template = (
        "\\documentclass{article}\n\\begin{document}\n"
        "\\section*{Experience}\n"
        "% [frontend]\n"
        "\\textbf{Frontend Intern,} {Acme} -- Remote \\hfill 2025 \\\\\n"
        "\\begin{itemize}\n  \\item Built React apps.\n\\end{itemize}\n\n"
        "\\textbf{Electrical Lead,} {RocketClub} -- Toronto \\hfill 2024 \\\\\n"
        "\\begin{itemize}\n  \\item Designed PCBs.\n\\end{itemize}\n\n"
        "\\section*{Skills}\nStuff\n"
        "\\end{document}\n"
    )
    parsed_titles = {"Frontend Intern Acme"}
    untagged = _detect_untagged_entries(template, parsed_titles)
    assert len(untagged) == 1
    assert "Electrical Lead" in untagged[0][1]
    assert untagged[0][2] == "Experience"


def test_detect_ignores_commented_entries() -> None:
    """Entries inside \\begin{comment} blocks are not flagged as untagged."""
    template = (
        "\\documentclass{article}\n\\begin{document}\n"
        "\\section*{Projects}\n"
        "% [web]\n"
        "\\textbf{Web App} | \\textit{JS} \\\\\n"
        "\\begin{itemize}\n  \\item Built it.\n\\end{itemize}\n\n"
        "\\begin{comment}\n"
        "\\textbf{Old Project} | \\textit{C} \\\\\n"
        "\\begin{itemize}\n  \\item Wrote code.\n\\end{itemize}\n"
        "\\end{comment}\n"
        "\\section*{Skills}\nStuff\n"
        "\\end{document}\n"
    )
    untagged = _detect_untagged_entries(template, {"Web App"})
    assert len(untagged) == 0


def test_infer_tag_avoids_collisions() -> None:
    """Tag inference skips generic words and appends suffixes on collision."""
    assert _infer_tag("Frontend Development Intern", set()) == "frontend"
    assert _infer_tag("Frontend Development Intern", {"frontend"}) == "frontend_development"
    assert _infer_tag("Frontend Development Intern", {"frontend", "frontend_development"}) == "frontend2"
    assert _infer_tag("Senior Lead Intern", set()) == "senior"  # all words are skip-words, falls back to words[0]
    assert _infer_tag("Senior Lead Intern", {"senior"}) == "senior2"  # collision triggers suffix


def test_sync_writes_tag_and_baseinfo_stub(tmp_path: Path) -> None:
    """sync_template_baseinfo inserts % [tag] and a baseinfo stub, then
    writes both files to disk."""
    template = (
        "\\documentclass{article}\n\\begin{document}\n"
        "\\section*{Experience}\n"
        "% [ml]\n"
        "\\textbf{ML Intern,} {BigCo} -- Remote \\hfill 2025 \\\\\n"
        "\\begin{itemize}\n  \\item Trained models.\n\\end{itemize}\n\n"
        "\\textbf{Hardware Eng,} {RocketCo} -- Toronto \\hfill 2024 \\\\\n"
        "\\begin{itemize}\n  \\item Designed \\textbf{PCB} circuits using \\textbf{KiCad}.\n\\end{itemize}\n\n"
        "\\section*{Skills}\nStuff\n"
        "\\end{document}\n"
    )
    baseinfo = (
        "Candidate profile facts:\n\n"
        "== EXPERIENCE ==\n\n"
        "[ml] ML Intern, BigCo\nNotes:\n\n"
        "== SKILL ANCHORS ==\n"
        "Languages: Python\n"
    )
    tpl_path = tmp_path / "template.tex"
    bio_path = tmp_path / "baseinfo.txt"
    tpl_path.write_text(template, encoding="utf-8")
    bio_path.write_text(baseinfo, encoding="utf-8")

    new_tpl, new_bio, msgs = sync_template_baseinfo(
        tpl_path, bio_path, template, baseinfo, {"ML Intern BigCo"},
    )

    assert "% [hardware]" in new_tpl
    assert "[hardware] Hardware Eng" in new_bio
    assert any("Auto-tagged" in m for m in msgs)
    # Files written to disk
    assert "% [hardware]" in tpl_path.read_text(encoding="utf-8")
    assert "[hardware]" in bio_path.read_text(encoding="utf-8")


def test_sync_adds_skill_anchors(tmp_path: Path) -> None:
    """Bold tools from an untagged entry are added to SKILL ANCHORS."""
    template = (
        "\\documentclass{article}\n\\begin{document}\n"
        "\\section*{Experience}\n"
        "% [web]\n"
        "\\textbf{Web Dev,} {Acme} \\\\\n"
        "\\begin{itemize}\n  \\item Built apps.\n\\end{itemize}\n\n"
        "\\textbf{EE Member,} {Club} \\\\\n"
        "\\begin{itemize}\n  \\item Used \\textbf{Altium} and \\textbf{KiCad} for \\textbf{PCB} design.\n\\end{itemize}\n\n"
        "\\section*{Skills}\nStuff\n"
        "\\end{document}\n"
    )
    baseinfo = (
        "== EXPERIENCE ==\n\n"
        "[web] Web Dev\nNotes:\n\n"
        "== SKILL ANCHORS ==\n"
        "Languages: Python\n"
    )
    tpl_path = tmp_path / "template.tex"
    bio_path = tmp_path / "baseinfo.txt"
    tpl_path.write_text(template, encoding="utf-8")
    bio_path.write_text(baseinfo, encoding="utf-8")

    _, new_bio, msgs = sync_template_baseinfo(
        tpl_path, bio_path, template, baseinfo, {"Web Dev Acme"},
    )

    assert "Altium" in new_bio
    assert "KiCad" in new_bio
    assert any("skill anchors" in m.lower() for m in msgs)


def test_sync_noop_when_all_tagged(tmp_path: Path) -> None:
    """No changes when all entries already have tags."""
    template = (
        "\\documentclass{article}\n\\begin{document}\n"
        "\\section*{Experience}\n"
        "% [ml]\n"
        "\\textbf{ML Intern,} {BigCo} \\\\\n"
        "\\begin{itemize}\n  \\item Trained models.\n\\end{itemize}\n\n"
        "\\section*{Skills}\nStuff\n"
        "\\end{document}\n"
    )
    baseinfo = "== EXPERIENCE ==\n\n[ml] ML Intern\nNotes:\n"
    tpl_path = tmp_path / "template.tex"
    bio_path = tmp_path / "baseinfo.txt"
    tpl_path.write_text(template, encoding="utf-8")
    bio_path.write_text(baseinfo, encoding="utf-8")

    new_tpl, new_bio, msgs = sync_template_baseinfo(
        tpl_path, bio_path, template, baseinfo, {"ML Intern BigCo"},
    )

    assert new_tpl == template
    assert new_bio == baseinfo
    assert msgs == []


# ---------------------------------------------------------------------------
# _bold_jd_tools tests
# ---------------------------------------------------------------------------


class TestBoldJdTools:
    def test_basic_bolding(self):
        text = r"Built a pipeline using Docker and Kubernetes."
        result = _bold_jd_tools(text, ("Docker", "Kubernetes"), max_bold=5)
        assert r"\textbf{Docker}" in result
        assert r"\textbf{Kubernetes}" in result

    def test_skips_already_bolded(self):
        text = r"Built a pipeline using \textbf{Docker} and Kubernetes."
        result = _bold_jd_tools(text, ("Docker", "Kubernetes"), max_bold=5)
        assert result.count(r"\textbf{Docker}") == 1
        assert r"\textbf{Kubernetes}" in result

    def test_max_bold_limit(self):
        text = r"Used Docker, Kubernetes, and Terraform for deployment."
        result = _bold_jd_tools(text, ("Docker", "Kubernetes", "Terraform"), max_bold=2)
        bold_count = len(re.findall(r"\\textbf\{", result))
        assert bold_count == 2

    def test_max_bold_already_at_limit(self):
        text = r"Used \textbf{Docker} and \textbf{React} plus Kubernetes."
        result = _bold_jd_tools(text, ("Kubernetes",), max_bold=2)
        assert r"\textbf{Kubernetes}" not in result

    def test_protected_span_skipping_textit(self):
        """The finditer fix: if first occurrence is inside \\textit{}, find the next one."""
        text = r"Used \textit{Kubernetes tools} for CI, then deployed Kubernetes in prod."
        result = _bold_jd_tools(text, ("Kubernetes",), max_bold=5)
        assert r"\textbf{Kubernetes}" in result
        assert result.index(r"\textbf{Kubernetes}") > result.index(r"\textit{")

    def test_protected_textit_span(self):
        text = r"Used \textit{Python, Docker} stack and Docker for CI."
        result = _bold_jd_tools(text, ("Docker",), max_bold=5)
        assert r"\textbf{Docker}" in result
        assert result.index(r"\textbf{Docker}") > result.index(r"\textit{")

    def test_no_match_returns_unchanged(self):
        text = r"Built a web application with React."
        result = _bold_jd_tools(text, ("Kubernetes",), max_bold=5)
        assert result == text

    def test_case_insensitive_match(self):
        text = r"Managed kubernetes clusters for production."
        result = _bold_jd_tools(text, ("Kubernetes",), max_bold=5)
        assert r"\textbf{kubernetes}" in result

    def test_word_boundary_no_partial_match(self):
        text = r"Used reactivity patterns in the frontend."
        result = _bold_jd_tools(text, ("React",), max_bold=5)
        assert r"\textbf{" not in result

    def test_empty_jd_tools(self):
        text = r"Built a pipeline."
        result = _bold_jd_tools(text, (), max_bold=5)
        assert result == text


# ---------------------------------------------------------------------------
# _inject_jd_tools_into_skills tests
# ---------------------------------------------------------------------------


class TestInjectJdToolsIntoSkills:
    SKILLS_TAIL = (
        "\\section*{Skills}\n"
        "\\textbf{Languages:} Python, JavaScript \\\\\n"
        "\\textbf{Frameworks:} React, Flask \\\\\n"
        "\\textbf{Tools:} Git, Docker \\\\\n"
        "\\textbf{Platforms:} GitHub, AWS \\\\\n"
        "\n"
        "\\section*{Education}\n"
        "\\textbf{TMU} -- BEng Computer Engineering\n"
    )

    def test_category_routing_tools(self):
        result = _inject_jd_tools_into_skills(
            self.SKILLS_TAIL, ("ServiceNow",), skill_anchors=(),
        )
        assert "ServiceNow" in result.split("\\textbf{Tools:}")[1].split("\n")[0]

    def test_category_routing_frameworks(self):
        result = _inject_jd_tools_into_skills(
            self.SKILLS_TAIL, ("Django",), skill_anchors=(),
        )
        assert "Django" in result.split("\\textbf{Frameworks:}")[1].split("\n")[0]

    def test_category_routing_platforms(self):
        result = _inject_jd_tools_into_skills(
            self.SKILLS_TAIL, ("Azure",), skill_anchors=(),
        )
        assert "Azure" in result.split("\\textbf{Platforms:}")[1].split("\n")[0]

    def test_category_routing_languages(self):
        result = _inject_jd_tools_into_skills(
            self.SKILLS_TAIL, ("Rust",), skill_anchors=(),
        )
        assert "Rust" in result.split("\\textbf{Languages:}")[1].split("\n")[0]

    def test_unknown_category_defaults_to_tools(self):
        result = _inject_jd_tools_into_skills(
            self.SKILLS_TAIL, ("SomeObscureTool",), skill_anchors=(),
        )
        assert "SomeObscureTool" in result.split("\\textbf{Tools:}")[1].split("\n")[0]

    def test_dedup_against_skill_anchors(self):
        result = _inject_jd_tools_into_skills(
            self.SKILLS_TAIL, ("Docker",), skill_anchors=("Docker",),
        )
        tools_line = [l for l in result.splitlines() if "\\textbf{Tools:}" in l][0]
        assert tools_line.count("Docker") == 1

    def test_dedup_against_existing_tail_text(self):
        result = _inject_jd_tools_into_skills(
            self.SKILLS_TAIL, ("Git",), skill_anchors=(),
        )
        tools_line = [l for l in result.splitlines() if "\\textbf{Tools:}" in l][0]
        assert tools_line.count("Git") == 1

    def test_empty_jd_tools_returns_unchanged(self):
        result = _inject_jd_tools_into_skills(
            self.SKILLS_TAIL, (), skill_anchors=(),
        )
        assert result == self.SKILLS_TAIL

    def test_fallback_to_last_skills_line(self):
        tail = (
            "\\section*{Skills}\n"
            "\\textbf{Languages:} Python \\\\\n"
            "\n"
            "\\section*{Education}\n"
            "\\textbf{TMU} -- BEng\n"
        )
        result = _inject_jd_tools_into_skills(
            tail, ("ServiceNow",), skill_anchors=(),
        )
        langs_line = [l for l in result.splitlines() if "\\textbf{Languages:}" in l][0]
        assert "ServiceNow" in langs_line

    def test_does_not_inject_into_education_section(self):
        result = _inject_jd_tools_into_skills(
            self.SKILLS_TAIL, ("ServiceNow",), skill_anchors=(),
        )
        edu_section = result.split("\\section*{Education}")[1]
        assert "ServiceNow" not in edu_section

    def test_multiple_tools_multiple_categories(self):
        result = _inject_jd_tools_into_skills(
            self.SKILLS_TAIL,
            ("ServiceNow", "Django", "Azure", "Rust"),
            skill_anchors=(),
        )
        assert "ServiceNow" in result.split("\\textbf{Tools:}")[1].split("\n")[0]
        assert "Django" in result.split("\\textbf{Frameworks:}")[1].split("\n")[0]
        assert "Azure" in result.split("\\textbf{Platforms:}")[1].split("\n")[0]
        assert "Rust" in result.split("\\textbf{Languages:}")[1].split("\n")[0]

    def test_line_break_preserved(self):
        result = _inject_jd_tools_into_skills(
            self.SKILLS_TAIL, ("ServiceNow",), skill_anchors=(),
        )
        tools_line = [l for l in result.splitlines() if "\\textbf{Tools:}" in l][0]
        assert tools_line.rstrip().endswith("\\\\")


# ---------------------------------------------------------------------------
# _extract_jd_tools tests
# ---------------------------------------------------------------------------


class TestExtractJdTools:
    def test_extracts_known_terms(self):
        desc = "We use Kubernetes and Terraform for our infrastructure."
        result = _extract_jd_tools(desc, skill_anchors=())
        result_lower = [t.lower() for t in result]
        assert "kubernetes" in result_lower
        assert "terraform" in result_lower

    def test_skips_skill_anchors(self):
        desc = "Must know Jenkins and Kubernetes."
        result = _extract_jd_tools(desc, skill_anchors=("Jenkins",))
        result_lower = [t.lower() for t in result]
        assert "jenkins" not in result_lower
        assert "kubernetes" in result_lower

    def test_skip_jd_inject_filter(self):
        desc = "Experience with Lean, Six Sigma, ITIL, PMP, Jenkins required."
        result = _extract_jd_tools(desc, skill_anchors=())
        result_lower = [t.lower() for t in result]
        assert "jenkins" in result_lower
        for blocked in ("lean", "six sigma", "itil", "pmp"):
            assert blocked not in result_lower

    def test_teams_filtered_out(self):
        desc = "Work with cross-functional teams and Microsoft Teams."
        result = _extract_jd_tools(desc, skill_anchors=())
        result_lower = [t.lower() for t in result]
        assert "teams" not in result_lower
        assert "microsoft teams" in result_lower

    def test_rest_filtered_out(self):
        desc = "Build REST APIs and handle the rest of the infrastructure."
        result = _extract_jd_tools(desc, skill_anchors=())
        result_lower = [t.lower() for t in result]
        assert "rest" not in result_lower

    def test_preserves_original_case(self):
        desc = "Experience with KUBERNETES clusters."
        result = _extract_jd_tools(desc, skill_anchors=())
        originals = {t for t in result if t.lower() == "kubernetes"}
        assert any(t == "KUBERNETES" for t in originals)

    def test_no_matches_returns_empty(self):
        desc = "We're looking for a passionate team player."
        result = _extract_jd_tools(desc, skill_anchors=())
        assert result == []

    def test_deduplication(self):
        desc = "Kubernetes Kubernetes Kubernetes clusters Kubernetes."
        result = _extract_jd_tools(desc, skill_anchors=())
        k8s_hits = [t for t in result if t.lower() == "kubernetes"]
        assert len(k8s_hits) == 1


# ---------------------------------------------------------------------------
# _SKIP_JD_INJECT coverage
# ---------------------------------------------------------------------------


class TestSkipJdInject:
    def test_certifications_in_skip_set(self):
        for term in ("comptia", "comptia a+", "ccna", "ccnp", "ccie"):
            assert term in _SKIP_JD_INJECT

    def test_methodologies_in_skip_set(self):
        for term in ("lean", "six sigma", "waterfall", "itil", "cobit", "prince2", "pmp"):
            assert term in _SKIP_JD_INJECT

    def test_teams_in_skip_set(self):
        assert "teams" in _SKIP_JD_INJECT

    def test_rest_in_skip_set(self):
        assert "rest" in _SKIP_JD_INJECT

    def test_real_tools_not_in_skip_set(self):
        for term in ("docker", "kubernetes", "react", "python", "aws"):
            assert term not in _SKIP_JD_INJECT

    def test_common_word_collisions_in_skip_set(self):
        """2026-07-17: found live — 'Fall 2026 and Spring 2027' (hiring
        season) got 'Spring' injected as the Java framework into unrelated
        bullets and Skills. These all double as ordinary English words that
        dominate plain JD prose over their tech sense."""
        for term in (
            "spring", "go", "workday", "spark", "express", "helm",
            "celery", "airflow", "chef", "vite", "postman",
        ):
            assert term in _SKIP_JD_INJECT

    def test_spring_not_extracted_from_hiring_season_sentence(self):
        desc = "We are seeking an Intern for Fall 2026 and Spring 2027."
        result = _extract_jd_tools(desc, skill_anchors=())
        assert "spring" not in {t.lower() for t in result}

    def test_go_not_extracted_from_ordinary_verb_usage(self):
        desc = "You'll go above and beyond, willing to go the extra mile."
        result = _extract_jd_tools(desc, skill_anchors=())
        assert "go" not in {t.lower() for t in result}

    def test_real_spring_boot_framework_still_extracted(self):
        """The multi-word 'spring boot' is unambiguous and stays matchable —
        only the bare 'spring' collision is excluded."""
        desc = "Experience building microservices with Spring Boot required."
        result = _extract_jd_tools(desc, skill_anchors=())
        assert "spring boot" in {t.lower() for t in result}


# ---------------------------------------------------------------------------
# Ownership / scope / metric-category reinforcement (2026-07-16)
# ---------------------------------------------------------------------------

from services.resumes.structured import TemplateEntry, _entry_metric_menu


def test_prompt_contains_ownership_scope_rule() -> None:
    assert CATALOG is not None
    prompt = build_structured_prompt("Engineer", "Job description", [], CATALOG)
    assert "OWNERSHIP AND SCOPE" in prompt
    assert "scale, cost, reliability, adoption" in prompt
    assert "never invent a number for a" in prompt


def test_validate_bullet_demotes_unsupported_ownership_verb() -> None:
    """Demote-not-veto: 'Led' with no leadership fact in the grounding is
    swapped for a collaborative verb; the rest of the rewrite survives."""
    text, reason = validate_tailored_bullet(
        "Led \\textbf{LaTeX} documentation efforts, reducing downtime by 41\\%.",
        CANONICAL,
    )
    assert reason is None and text is not None
    assert text.startswith("Co-led ")
    assert "41\\%" in text


def test_validate_bullet_keeps_ownership_verb_with_grounding() -> None:
    """A baseinfo 'Ownership:' line (part of entry_context) licenses the verb."""
    entry_context = CANONICAL + " Ownership: sole owner of the documentation workflow."
    text, reason = validate_tailored_bullet(
        "Led \\textbf{LaTeX} documentation efforts, reducing downtime by 41\\%.",
        CANONICAL,
        entry_context=entry_context,
    )
    assert reason is None and text is not None
    assert text.startswith("Led ")


def test_validate_bullet_ownership_verb_grounded_by_canonical_verb() -> None:
    """The canonical bullet's own opening verb is profile truth — never demoted."""
    canonical = "Managed windowing, input, and render contexts with \\textbf{GLFW}."
    text, reason = validate_tailored_bullet(
        "Managed render contexts and windowing with \\textbf{GLFW}.",
        canonical,
    )
    assert reason is None and text is not None
    assert text.startswith("Managed ")


def test_validate_bullet_rejects_ungrounded_number() -> None:
    """Numeric invention was previously policed only by prompt text; the
    deterministic guard protects the new baseinfo scope/team-size figures."""
    text, reason = validate_tailored_bullet(
        "Maintained \\textbf{LaTeX} documentation for a team of 12, reducing "
        "downtime by 41\\%.",
        CANONICAL,
    )
    assert text is None and "ungrounded number" in str(reason)


def test_validate_bullet_accepts_background_sourced_number() -> None:
    entry_context = CANONICAL + " Scope: supported a department of 12 staff."
    text, reason = validate_tailored_bullet(
        "Maintained \\textbf{LaTeX} documentation for a department of 12 staff, "
        "reducing downtime by 41\\%.",
        CANONICAL,
        entry_context=entry_context,
    )
    assert reason is None and text is not None


def test_validate_bullet_number_grounding_normalizes_commas() -> None:
    entry_context = CANONICAL + " Scale: deduplicated 206,775 job postings."
    text, reason = validate_tailored_bullet(
        "Maintained pipelines that deduplicated 206775 job postings, reducing "
        "downtime by 41\\%.",
        CANONICAL,
        entry_context=entry_context,
    )
    assert reason is None and text is not None


def test_entry_metric_menu_preserves_category_label() -> None:
    """A Notes line's category label survives the snippet window even when the
    number sits more than 40 chars into the line."""
    entry = TemplateEntry(
        entry_id="projects-widget",
        section="Projects",
        categories=("widget",),
        header="\\textbf{Widget} \\\\",
        bullets=("Built a widget for internal use.",),
        title="Widget",
    )
    background = (
        "[widget] Widget (Python)\n"
        "Reliability: documentation-driven fixes across every workflow helped "
        "cut website downtime by 41%.\n"
        "Scale: deduplicated 206,775 job postings in one month."
    )
    menu = _entry_metric_menu(entry, background)
    assert any(item.startswith("Reliability:") and "41" in item for item in menu)
    # A comma-grouped number is one menu item, shown with its commas.
    comma_items = [item for item in menu if "206,775" in item]
    assert len(comma_items) == 1
    assert not any("775 " in item and "206,775" not in item for item in menu)


def test_baseinfo_notes_flow_into_prompt_background() -> None:
    """The populated xboxsignout._ Notes reach the prompt as verified background."""
    assert CATALOG is not None
    prompt = build_structured_prompt("Engineer", "Job description", [], CATALOG)
    assert "sole designer and developer of the entire system" in prompt
    assert "206,775 job postings" in prompt


# ---------------------------------------------------------------------------
# Listing-language checklist (2026-07-16)
# ---------------------------------------------------------------------------

JD_LANG_DESC = (
    "We build observability tooling for cloud infrastructure. You will design "
    "monitoring dashboards, automate deployment pipelines, and improve "
    "incident response. Experience with Docker containers, Kafka streaming, "
    "and Python services required. Familiarity with observability platforms "
    "and monitoring practices, deployment automation, and incident response "
    "runbooks is a strong asset. Our monitoring stack ingests Kafka events "
    "from Docker services; Python automation drives deployment and incident "
    "workflows across the observability platform."
)


def test_prompt_includes_listing_language_block() -> None:
    assert CATALOG is not None
    prompt = build_structured_prompt("Platform Engineer", JD_LANG_DESC, [], CATALOG)
    assert "<listing_language>" in prompt
    assert "LISTING LANGUAGE EVERYWHERE" in prompt
    block = prompt.split("<listing_language>")[1].split("</listing_language>")[0]
    # Anchors the listing names arrive with display casing; soft JD vocabulary
    # (non-anchor repeated terms) arrives too.
    assert "Docker" in block and "Kafka" in block and "Python" in block
    assert "monitoring" in block.lower()
    assert "observability" in block.lower()


def test_prompt_listing_language_absent_in_aggressive_modes(monkeypatch) -> None:
    """Aggressive modes keep their stronger injection blocks; the truth-gated
    checklist and its rule must not double up with them."""
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "aggressive", True)
    prompt = build_structured_prompt("Platform Engineer", JD_LANG_DESC, [], CATALOG)
    assert "<listing_language>" not in prompt
    assert "LISTING LANGUAGE EVERYWHERE" not in prompt
    assert "AGGRESSIVE MODE" in prompt

    monkeypatch.setattr(CATALOG.render_config, "aggressive", False)
    monkeypatch.setattr(CATALOG.render_config, "strong_aggressive", True)
    prompt = build_structured_prompt("Platform Engineer", JD_LANG_DESC, [], CATALOG)
    assert "<listing_language>" not in prompt
    assert "LISTING LANGUAGE EVERYWHERE" not in prompt


def test_prompt_listing_language_omitted_when_no_terms() -> None:
    """A description with nothing extractable produces no empty block."""
    assert CATALOG is not None
    prompt = build_structured_prompt("Job", "help wanted", [], CATALOG)
    assert "<listing_language>" not in prompt
    assert "LISTING LANGUAGE EVERYWHERE" not in prompt


def test_rewrite_scope_notes_renumbered_after_new_rules(monkeypatch) -> None:
    """Scope notes were '14.' before rules 14/15 existed; they must not
    collide with the ownership and listing-language rules."""
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "rewrite_scope", "limited")
    prompt = build_structured_prompt("Engineer", JD_LANG_DESC, [], CATALOG)
    assert "16. REWRITE BUDGET" in prompt
    assert "14. OWNERSHIP AND SCOPE" in prompt
    assert "15. LISTING LANGUAGE EVERYWHERE" in prompt


# ---------------------------------------------------------------------------
# Ownership/number guard hardening (2026-07-16 review fixes)
# ---------------------------------------------------------------------------


def test_ownership_guard_ignores_incidental_management_word() -> None:
    """'reported progress to management' is prose about the org, not evidence
    of the candidate managing anything — the demotion must still fire."""
    canonical = (
        "Tracked project milestones and reported progress to management in "
        "\\textbf{Microsoft Suite}, reducing coordination delays by 14\\%."
    )
    text, reason = validate_tailored_bullet(
        "Managed project milestone tracking, reducing coordination delays "
        "by 14\\%.",
        canonical,
    )
    assert reason is None and text is not None
    assert text.startswith("Coordinated ")


def test_ownership_guard_exact_canonical_verb_still_licenses() -> None:
    canonical = "Managed windowing, input, and render contexts with \\textbf{GLFW}."
    text, reason = validate_tailored_bullet(
        "Managed render contexts and windowing with \\textbf{GLFW}.",
        canonical,
    )
    assert reason is None and text is not None and text.startswith("Managed ")


def test_ownership_guard_coled_grounding_does_not_license_led() -> None:
    canonical = "Co-led weekly design reviews for the rendering module."
    text, reason = validate_tailored_bullet(
        "Led weekly design reviews for the rendering module.",
        canonical,
    )
    assert reason is None and text is not None
    assert text.startswith("Co-led ")


def test_ownership_guard_unwraps_bolded_opening_verb() -> None:
    text, reason = validate_tailored_bullet(
        "\\textbf{Led} documentation efforts, reducing downtime by 41\\%.",
        CANONICAL,
    )
    assert reason is None and text is not None
    assert text.startswith("\\textbf{Co-led}")


def test_number_guard_ignores_digits_inside_identifiers() -> None:
    """'STM32' grounds the token STM32, not a free-floating '32' — and a
    rewrite naming STM32 must not need standalone-32 grounding either."""
    canonical = (
        "Designed \\textbf{PCB circuits} for a custom STM32-based avionics "
        "flight computer."
    )
    # Invented "32%" must NOT be grounded by the 32 inside STM32.
    text, reason = validate_tailored_bullet(
        "Designed PCB circuits, improving efficiency by 32\\%.",
        canonical,
        skill_anchors=("pcb", "stm32"),
    )
    assert text is None and "ungrounded number" in str(reason)
    # Naming STM32 itself stays fine.
    text, reason = validate_tailored_bullet(
        "Designed \\textbf{PCB circuits} for an STM32-based flight computer.",
        canonical,
        skill_anchors=("pcb", "stm32"),
    )
    assert reason is None and text is not None


def test_number_guard_runs_after_filler_strip_salvages_bullet() -> None:
    """A fabricated figure living only in a strippable filler tail must not
    reject the salvageable front of the rewrite."""
    text, reason = validate_tailored_bullet(
        "Maintained \\textbf{LaTeX} documentation, reducing downtime by "
        "41\\%, ensuring reliability for over 500 users.",
        CANONICAL,
    )
    assert reason is None and text is not None
    assert "41\\%" in text
    assert "500" not in text


# ---------------------------------------------------------------------------
# Strong-aggressive skills rewrite fixes (2026-07-17)
# ---------------------------------------------------------------------------

from services.resumes.structured import (
    _reconcile_skills_and_bullets,
    _rewrite_skills_for_jd,
)

SKILLS_TAIL = (
    "\\section*{Skills}\n"
    "\\textbf{Languages:} JavaScript, Java, Python, C++, C, SQL, HTML, CSS \\\\\n"
    "\\textbf{Tools:} Git, PostgreSQL, Docker \\\\\n"
    "\\section*{Education}\n"
    "\\textbf{School} -- Degree \\hfill 2023--2027"
)


def test_rewrite_skills_keeps_nonmatching_canonical_items() -> None:
    """Partial JD relevance must reorder, not gut, a skills line."""
    tail = _rewrite_skills_for_jd(SKILLS_TAIL, ("Salesforce",), ["sql", "python"], ())
    langs = next(l for l in tail.splitlines() if "Languages:" in l)
    for item in ("JavaScript", "Java", "Python", "C++", "SQL", "HTML", "CSS"):
        assert item in langs, f"{item} dropped from Languages"
    # Relevant items lead.
    assert langs.index("Python") < langs.index("JavaScript")
    tools = next(l for l in tail.splitlines() if "Tools:" in l)
    assert "Salesforce" in tools and "Git" in tools


def test_rewrite_skills_single_letter_needs_whole_token_match() -> None:
    """'C' must not rank as JD-relevant via substring noise ('css', 'excel')."""
    tail = _rewrite_skills_for_jd(SKILLS_TAIL, (), ["css", "excel"], ())
    langs = next(l for l in tail.splitlines() if "Languages:" in l)
    items = [i.strip() for i in langs.split(":}")[1].rstrip(" \\").split(",")]
    assert items[0] == "CSS"
    assert "C" in items  # kept, just not falsely promoted
    assert items.index("C") > items.index("CSS")


def test_rewrite_skills_cap_cuts_only_unmatched_tail() -> None:
    tail = _rewrite_skills_for_jd(
        SKILLS_TAIL, ("Salesforce",), ["sql", "python"], (), max_items=3
    )
    langs = next(l for l in tail.splitlines() if "Languages:" in l)
    items = [i.strip() for i in langs.split(":}")[1].rstrip(" \\").split(",")]
    assert len(items) == 3
    assert "Python" in items and "SQL" in items  # matched items survive


def test_unbold_non_jd_terms_demotes_irrelevant_bolds() -> None:
    from services.resumes.structured import _unbold_non_jd_terms

    text = (
        "Built \\textbf{Salesforce} flows and \\textbf{process maps} using "
        "\\textbf{Python} for \\textbf{stakeholder alignment}."
    )
    out = _unbold_non_jd_terms(text, ("Salesforce", "Python"), ["sql"])
    assert "\\textbf{Salesforce}" in out
    assert "\\textbf{Python}" in out
    assert "\\textbf{process maps}" not in out and "process maps" in out
    assert "\\textbf{stakeholder alignment}" not in out and "stakeholder alignment" in out


def test_unbold_non_jd_terms_containment_keeps_adjacent_forms() -> None:
    from services.resumes.structured import _unbold_non_jd_terms

    # "PostgreSQL" contains JD keyword "sql" — containment keeps it bolded,
    # consistent with _filter_keywordless_bullets' matching.
    text = "Managed \\textbf{PostgreSQL} validation."
    out = _unbold_non_jd_terms(text, (), ["sql"])
    assert "\\textbf{PostgreSQL}" in out
    # No JD terms at all -> no-op, never mass-unbold.
    assert _unbold_non_jd_terms(text, (), []) == text


def test_unbold_non_jd_terms_keeps_verbatim_jd_vocabulary() -> None:
    """A bold missing from the top-N keyword set but present verbatim in the
    listing text survives — the model read the full JD (RCFA/Excel case)."""
    from services.resumes.structured import _unbold_non_jd_terms

    jd = "Perform RCFA and condition-based monitoring; report findings in Excel."
    text = (
        "Performed \\textbf{RCFA} investigations tracked in \\textbf{Excel}, "
        "driving \\textbf{stakeholder alignment}."
    )
    out = _unbold_non_jd_terms(text, ("SAP",), ["maintenance"], jd_text=jd)
    assert "\\textbf{RCFA}" in out
    assert "\\textbf{Excel}" in out
    assert "\\textbf{stakeholder alignment}" not in out
    assert "stakeholder alignment" in out


# ---------------------------------------------------------------------------
# Strong-aggressive depth-over-breadth (2026-07-17)
# ---------------------------------------------------------------------------


def test_effective_config_sa_depth_knobs() -> None:
    from services.resumes.structured import (
        MAX_BULLET_CHARS,
        RenderConfig,
        _effective_render_config,
    )

    cfg = _effective_render_config(RenderConfig(strong_aggressive=True))
    assert cfg.max_bullet_chars == MAX_BULLET_CHARS + 120
    assert cfg.min_visible_bullets == MIN_VISIBLE_BULLETS - 5


def test_sa_prompt_offers_depth_over_breadth(monkeypatch) -> None:
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "strong_aggressive", True)
    prompt = build_structured_prompt("Engineer", "desc", [], CATALOG)
    assert "DEPTH OVER BREADTH" in prompt
    assert "write UP TO this many" in prompt
    assert "~450 characters" in prompt


def test_sa_shorter_bullets_list_drops_trailing_slots(monkeypatch) -> None:
    """In strong-aggressive, providing fewer bullets than canonical is an
    intentional depth choice — the unwritten slots are dropped, not padded
    with canonical text."""
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "strong_aggressive", True)
    web = _entry_id("web-messaging")
    others = [e.entry_id for e in CATALOG.entries if e.entry_id != web]
    selection = StructuredSelection(
        ranking=[web] + others,
        keywords=["python"],
        bullets={
            web: [
                "Built \\textbf{Python} data services covering intake, "
                "validation, and reporting for the platform.",
                "Automated \\textbf{Python} test harnesses across the full "
                "release cycle.",
            ]
        },
    )
    doc, report = render_structured_resume(CATALOG, selection)
    chunk = doc.split("Web Messaging App")[1].split("\\end{itemize}")[0]
    assert chunk.count("\\item") == 2
    assert "Raspberry Pi" not in chunk  # canonical 3rd bullet not padded in


def test_normal_mode_shorter_bullets_list_still_pads_canonical() -> None:
    """Outside strong-aggressive, a short bullets list keeps canonical text
    for the missing slots — unchanged behavior."""
    assert CATALOG is not None
    web = _entry_id("web-messaging")
    others = [e.entry_id for e in CATALOG.entries if e.entry_id != web]
    selection = StructuredSelection(
        ranking=[web] + others,
        keywords=["python"],
        bullets={
            web: [
                "Built collaborative \\textbf{Python} tooling for real-time "
                "messaging workflows.",
            ]
        },
    )
    doc, report = render_structured_resume(CATALOG, selection)
    chunk = doc.split("Web Messaging App")[1].split("\\end{itemize}")[0]
    assert chunk.count("\\item") == 3
    assert "Raspberry Pi" in chunk


def test_reconcile_ignores_header_bolds_and_preserves_case() -> None:
    """Job titles / project names bolded in entry HEADERS must not become
    'tools'; harvested bullet terms keep their original casing."""
    rendered = (
        "\\textbf{Frontend Development Intern,} {MCG3D} -- Remote \\\\\n"
        "\\begin{itemize}\n"
        "  \\item Built \\textbf{Salesforce} test flows in \\textbf{Python}.\n"
        "\\end{itemize}\n"
        "\\textbf{Web Messaging App} | \\textit{JavaScript} \\\\\n"
        "\\begin{itemize}\n"
        "  \\item Shipped \\textbf{WebRTC} messaging.\n"
        "\\end{itemize}"
    )
    plain_tail = (
        "\\section*{Skills}\n"
        "\\textbf{Tools:} Git \\\\\n"
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree"
    )
    tail = _reconcile_skills_and_bullets(plain_tail, rendered, ())
    assert "frontend development intern" not in tail.lower().replace("skills", "")
    assert "Web Messaging App" not in tail.split("Education")[0].replace(
        "\\section*{Skills}", ""
    )
    tools = next(l for l in tail.splitlines() if "Tools:" in l)
    assert "Salesforce" in tools  # original casing, not 'salesforce'
    assert "Python" in tools or "Python" in tail  # harvested with casing intact
    assert "salesforce," not in tools  # no lowercase duplicates


def test_extract_listing_keywords_handles_bilingual_postings() -> None:
    """Accented words must not shed ASCII fragments into the keyword list
    (observed live: 'veloppement' and 'quipe' from a bilingual CFIA posting),
    and common French filler must not rank as listing language."""
    from services.resumes.structured import extract_listing_keywords

    desc = (
        "Développement d'applications avec notre équipe pour le développement "
        "de pipelines. Le développement logiciel avec Python et Docker pour "
        "notre équipe. Nous cherchons un stagiaire pour le développement avec "
        "Python, Docker et des pipelines de données. Python and Docker "
        "pipelines for data engineering."
    )
    terms = extract_listing_keywords(desc, ("python", "docker"), max_keywords=15)
    lowered = [t.lower() for t in terms]
    assert "python" in lowered and "docker" in lowered
    for fragment in ("veloppement", "quipe"):
        assert fragment not in lowered
    for filler in ("avec", "pour", "notre", "équipe", "développement", "stagiaire"):
        assert filler not in lowered
    assert not any(t.endswith(".") for t in terms)


def test_extract_listing_keywords_skips_url_plumbing() -> None:
    """Domains, URL paths, and job-key hex ids must not rank as listing
    language (observed live on an Indeed posting)."""
    from services.resumes.structured import extract_listing_keywords

    desc = (
        "Apply at ca.indeed.com/viewjob key af3732d8e7f6a952 today. "
        "Apply at ca.indeed.com/viewjob key af3732d8e7f6a952 today. "
        "Apply at ca.indeed.com/viewjob key af3732d8e7f6a952 today. "
        "Kubernetes deployment and Kubernetes monitoring with Kubernetes."
    )
    terms = [t.lower() for t in extract_listing_keywords(desc, (), max_keywords=10)]
    assert "kubernetes" in terms
    assert not any("indeed" in t or "/" in t for t in terms)
    assert "af3732d8e7f6a952" not in terms


# ---------------------------------------------------------------------------
# Profile catalog memoization (load_structured_profile cache)
# ---------------------------------------------------------------------------


def _write_profile(dir_path: Path) -> tuple[Path, Path]:
    dir_path.mkdir(parents=True, exist_ok=True)
    template = dir_path / "template.tex"
    baseinfo = dir_path / "baseinfo.txt"
    template.write_text(TEMPLATE_TEXT, encoding="utf-8")
    baseinfo.write_text((PROFILE_DIR / "baseinfo.txt").read_text(encoding="utf-8"), encoding="utf-8")
    return template, baseinfo


def test_profile_cache_skips_reparse_for_unchanged_files(tmp_path, monkeypatch) -> None:
    import services.resumes.structured as structured_module

    template, baseinfo = _write_profile(tmp_path / "prof")
    calls = {"n": 0}
    real = structured_module._load_structured_profile_uncached

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(structured_module, "_load_structured_profile_uncached", counting)
    first = load_structured_profile(template, baseinfo)
    second = load_structured_profile(template, baseinfo)
    assert first is not None and second is not None
    assert calls["n"] == 1


def test_profile_cache_returns_isolated_copies(tmp_path) -> None:
    """Callers set catalog.render_config.aggressive in place; a shared cached
    instance would leak one build's mode into the next build."""
    template, baseinfo = _write_profile(tmp_path / "prof")
    first = load_structured_profile(template, baseinfo)
    assert first is not None
    first.render_config.aggressive = True
    first.render_config.strong_aggressive = True

    second = load_structured_profile(template, baseinfo)
    assert second is not None
    assert second.render_config.aggressive is False
    assert second.render_config.strong_aggressive is False
    assert second is not first


def test_profile_cache_invalidates_when_a_profile_file_changes(tmp_path) -> None:
    template, baseinfo = _write_profile(tmp_path / "prof")
    first = load_structured_profile(template, baseinfo)
    assert first is not None

    # structured_config.json is part of the stamp even when it appears later.
    (tmp_path / "prof" / "structured_config.json").write_text(
        json.dumps({"max_bold_per_bullet": 9}), encoding="utf-8"
    )
    second = load_structured_profile(template, baseinfo)
    assert second is not None
    assert second.render_config.max_bold_per_bullet == 9


# ---------------------------------------------------------------------------
# _effective_render_config mode matrix — one row per shipped mode combination,
# asserting the RESOLVED config so a future scaling-rule change that breaks an
# interaction fails loudly here instead of in a live build.
# ---------------------------------------------------------------------------


def test_effective_render_config_mode_matrix() -> None:
    from dataclasses import replace as dc_replace

    from services.resumes.structured import RenderConfig, _effective_render_config

    base = RenderConfig()

    # Normal mode: pass-through, identical object semantics.
    assert _effective_render_config(base) == base

    # Profile-level adjacent_tool_leeway survives normal mode untouched.
    leeway = dc_replace(base, adjacent_tool_leeway=True)
    assert _effective_render_config(leeway) == leeway

    # Aggressive: leeway + audit off + bold cap +2; budgets untouched.
    ag = _effective_render_config(dc_replace(base, aggressive=True))
    assert ag.adjacent_tool_leeway is True
    assert ag.grounding_audit is False
    assert ag.max_bold_per_bullet == base.max_bold_per_bullet + 2
    assert ag.max_bullet_chars == base.max_bullet_chars
    assert ag.min_visible_bullets == base.min_visible_bullets
    assert ag.max_total_bullet_chars == base.max_total_bullet_chars
    assert ag.rewrite_scope == base.rewrite_scope
    assert ag.strong_aggressive is False

    # Strong-aggressive: superset — forces aggressive, full scope, bold cap +4,
    # depth-over-breadth budgets (+120 chars, min bullets floor-capped at 6).
    sa = _effective_render_config(dc_replace(base, strong_aggressive=True))
    assert sa.aggressive is True and sa.strong_aggressive is True
    assert sa.adjacent_tool_leeway is True
    assert sa.grounding_audit is False
    assert sa.rewrite_scope == "full"
    assert sa.max_bold_per_bullet == base.max_bold_per_bullet + 4
    assert sa.max_bullet_chars == base.max_bullet_chars + 120
    assert sa.min_visible_bullets == max(6, base.min_visible_bullets - 5)

    # Strong-aggressive wins over aggressive when both are set (superset, not
    # additive: bold cap is +4, not +6).
    both = _effective_render_config(dc_replace(base, aggressive=True, strong_aggressive=True))
    assert both == sa

    # A profile grounding_audit=False stays off in every mode.
    no_audit = dc_replace(base, grounding_audit=False)
    assert _effective_render_config(no_audit).grounding_audit is False
    assert _effective_render_config(dc_replace(no_audit, aggressive=True)).grounding_audit is False

    # The shipped xboxsignout._ shape: profile overrides survive aggressive
    # scaling relative to THEIR values, not the defaults.
    profile = dc_replace(
        base,
        max_visible_bullets=15,
        max_total_bullet_chars=2650,
        max_bold_per_bullet=6,
        adjacent_tool_leeway=True,
    )
    profile_sa = _effective_render_config(dc_replace(profile, strong_aggressive=True))
    assert profile_sa.max_visible_bullets == 15
    assert profile_sa.max_total_bullet_chars == 2650
    assert profile_sa.max_bold_per_bullet == 10
