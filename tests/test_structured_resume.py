"""Tests for the structured resume pipeline (JSON decisions -> deterministic LaTeX)."""

from __future__ import annotations

import json
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
    build_structured_prompt,
    detect_role_family,
    extract_json_object,
    load_structured_profile,
    parse_role_guide,
    parse_structured_response,
    parse_template_catalog,
    render_structured_resume,
    validate_tailored_bullet,
)

PROFILE_DIR = Path(__file__).resolve().parents[1] / "src" / "services" / "resumes" / "resumes_cache" / "xboxsignout._"
TEMPLATE_TEXT = (PROFILE_DIR / "template.tex").read_text(encoding="utf-8")
BASEINFO_TEXT = (PROFILE_DIR / "baseinfo.txt").read_text(encoding="utf-8")

# Load via load_structured_profile so the profile's structured_config.json
# (forbidden terms, budget) applies, exactly as in production.
_LOADED_PROFILE = load_structured_profile(PROFILE_DIR / "template.tex", PROFILE_DIR / "baseinfo.txt")
assert _LOADED_PROFILE is not None
CATALOG, FAMILIES = _LOADED_PROFILE


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
    assert len(CATALOG.entries) == 10
    assert CATALOG.entry_sections == ("Projects", "Experience")
    projects = [e for e in CATALOG.entries if e.section == "Projects"]
    experience = [e for e in CATALOG.entries if e.section == "Experience"]
    assert len(projects) == 6
    assert len(experience) == 4


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
# Role guide parsing + detection
# ---------------------------------------------------------------------------

def test_role_guide_parses_four_families() -> None:
    keys = [family.key for family in FAMILIES]
    assert keys == ["EMBEDDED", "SOFTWARE", "FRONTEND", "ML"]


def test_role_guide_must_show_lists() -> None:
    embedded = next(f for f in FAMILIES if f.key == "EMBEDDED")
    assert {"embedded", "hardware", "fpga", "teaching"} <= embedded.must_show
    assert "fullstack" in embedded.hide

    ml = next(f for f in FAMILIES if f.key == "ML")
    assert "ml" in ml.must_show
    assert "embedded" in ml.hide


def test_detect_role_family_embedded() -> None:
    family = detect_role_family(
        "Firmware developer for microcontroller platforms, embedded C, RTOS, bare-metal drivers",
        FAMILIES,
    )
    assert family is not None and family.key == "EMBEDDED"


def test_detect_role_family_ml() -> None:
    family = detect_role_family(
        "Machine learning engineer building model training and inference pipelines, data science",
        FAMILIES,
    )
    assert family is not None and family.key == "ML"


def test_detect_role_family_keywords_match_whole_tokens_only() -> None:
    # "Registered Nurse" must not score the EMBEDDED keyword "register" via
    # substring; with no true keyword hits, detection falls back to the
    # most-inclusive-family rule, i.e. the same family a no-keyword listing gets.
    nurse = detect_role_family(
        "Registered Nurse - patient care, clinical documentation, medication administration",
        FAMILIES,
    )
    no_keywords = detect_role_family("completely unrelated listing text", FAMILIES)
    assert nurse is not None and no_keywords is not None
    assert nurse.key == no_keywords.key
    # Real token usage still routes correctly.
    real = detect_role_family("firmware engineer writing register maps for MCU drivers", FAMILIES)
    assert real is not None and real.key == "EMBEDDED"


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


def test_validate_bullet_rejects_invented_metric() -> None:
    text, reason = validate_tailored_bullet(
        "Reduced downtime by 41\\% and boosted speed by 95\\%.", CANONICAL
    )
    assert text is None and "invented number 95" in str(reason)


def test_validate_bullet_rejects_forbidden_domain_term() -> None:
    from services.resumes.structured import RenderConfig

    config = RenderConfig(forbidden_terms=("scada",))
    text, reason = validate_tailored_bullet(
        "Documented SCADA workflows, reducing downtime by 41\\%.", CANONICAL, config
    )
    assert text is None and "forbidden term" in str(reason)
    # Without a profile config there is nothing to forbid — per-profile data.
    text, reason = validate_tailored_bullet(
        "Documented SCADA workflows, reducing downtime by 41\\%.", CANONICAL
    )
    assert reason is None and text is not None


def test_forbidden_terms_match_whole_words_only() -> None:
    """Regression: 'rust' must not fire inside 'robust', 'hmi' inside 'algorithmic'."""
    from services.resumes.structured import RenderConfig

    config = RenderConfig(forbidden_terms=("rust", "hmi"))
    text, reason = validate_tailored_bullet(
        "Built robust algorithmic pipelines, reducing downtime by 41\\%.",
        CANONICAL,
        config,
    )
    assert reason is None
    text, reason = validate_tailored_bullet(
        "Built pipelines in Rust, reducing downtime by 41\\%.", CANONICAL, config
    )
    assert text is None and "forbidden term 'rust'" in str(reason)


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


def test_validate_bullet_rejects_appended_filler_clause() -> None:
    text, reason = validate_tailored_bullet(
        "Maintained \\textbf{LaTeX} docs, reducing downtime by 41\\%, demonstrating analytical rigour.",
        CANONICAL,
    )
    assert text is None and "appended filler clause 'demonstrating'" in str(reason)


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


def test_render_rejects_invented_tool_in_bullet(monkeypatch) -> None:
    """End-to-end: an invented tool in a tailored bullet falls back to canonical."""
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "rewrite_scope", "full")
    goopter = next(e for e in CATALOG.entries if "goopter" in e.entry_id)
    selection = StructuredSelection(
        role_family_key="ML",
        bullets={
            goopter.entry_id: [
                "Engineered pipelines on \\textbf{Kubernetes} using \\textbf{Kafka} and \\textbf{FastAPI}."
            ]
        },
    )
    latex, report = render_structured_resume(CATALOG, FAMILIES, selection)
    assert "Kubernetes" not in latex
    assert any("invented tool" in item for item in report.canonical_fallbacks)


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


@pytest.mark.parametrize("family_key", ["EMBEDDED", "SOFTWARE", "FRONTEND", "ML"])
def test_render_bullet_count_within_budget(family_key: str) -> None:
    assert CATALOG is not None
    latex, report = render_structured_resume(
        CATALOG, FAMILIES, StructuredSelection(role_family_key=family_key)
    )
    assert MIN_VISIBLE_BULLETS <= report.visible_bullet_count <= MAX_VISIBLE_BULLETS
    assert _count_visible_bullets(latex) == report.visible_bullet_count


def test_render_embedded_shows_hardware_hides_web() -> None:
    assert CATALOG is not None
    latex, report = render_structured_resume(
        CATALOG, FAMILIES, StructuredSelection(role_family_key="EMBEDDED")
    )
    assert "eebot Mobile Robot System" in latex
    assert "Arithmetic and Logic Unit" in latex
    assert "Obotz Robotics" in latex
    assert "Web Messaging App" not in latex
    assert "MCG3D" not in latex
    assert "Goopter" not in latex


def test_render_ml_shows_goopter_with_kafka() -> None:
    assert CATALOG is not None
    latex, _ = render_structured_resume(
        CATALOG, FAMILIES, StructuredSelection(role_family_key="ML")
    )
    assert "Goopter" in latex
    assert "Kafka" in latex
    assert "eebot" not in latex


def test_render_has_no_comment_environments_and_single_document() -> None:
    assert CATALOG is not None
    latex, _ = render_structured_resume(
        CATALOG, FAMILIES, StructuredSelection(role_family_key="SOFTWARE")
    )
    assert latex.count("\\begin{document}") == 1
    assert latex.count("\\end{document}") == 1
    assert "\\begin{comment}" not in latex
    assert latex.count("\\begin{itemize}") == latex.count("\\end{itemize}")
    assert "<" not in latex.split("\\begin{document}")[1].replace("$<$", "")


def test_render_preserves_mandatory_blocks() -> None:
    assert CATALOG is not None
    for family_key in ("EMBEDDED", "SOFTWARE", "FRONTEND", "ML"):
        latex, _ = render_structured_resume(
            CATALOG, FAMILIES, StructuredSelection(role_family_key=family_key)
        )
        assert "Ernest Choi" in latex
        assert "ernestljchoi@gmail.com" in latex
        assert "\\section*{Skills}" in latex
        assert "\\section*{Education}" in latex
        assert "Toronto Metropolitan University" in latex


def test_render_preserves_markham_metrics_when_visible() -> None:
    assert CATALOG is not None
    latex, _ = render_structured_resume(
        CATALOG, FAMILIES, StructuredSelection(role_family_key="EMBEDDED")
    )
    if "City of Markham" in latex:
        assert "41\\%" in latex
        assert "14\\%" in latex
        assert "20\\%" in latex


def test_render_must_show_survives_hostile_ranking() -> None:
    """A ranking that buries MUST SHOW entries cannot hide them."""
    assert CATALOG is not None
    web_ids = [e.entry_id for e in CATALOG.entries if "web-messaging" in e.entry_id]
    selection = StructuredSelection(role_family_key="EMBEDDED", ranking=web_ids)
    latex, report = render_structured_resume(CATALOG, FAMILIES, selection)
    assert "eebot Mobile Robot System" in latex
    assert "Web Messaging App" not in latex


def test_render_uses_valid_tailored_bullets_and_falls_back_on_bad_ones(monkeypatch) -> None:
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "rewrite_scope", "full")
    goopter = next(e for e in CATALOG.entries if "goopter" in e.entry_id)
    tailored_ok = (
        "Engineered a distributed low-latency data pipeline for quantitative trading with "
        "\\textbf{Kafka}, \\textbf{FastAPI}, and asynchronous \\textbf{Python} using "
        "\\textbf{Pandas}, \\textbf{NumPy}, and \\textbf{scikit-learn} for production analytics."
    )
    selection = StructuredSelection(
        role_family_key="ML",
        bullets={
            goopter.entry_id: [
                tailored_ok,
                "Built models with SCADA integration using \\textbf{TensorFlow}.",  # forbidden term
            ]
        },
    )
    latex, report = render_structured_resume(CATALOG, FAMILIES, selection)
    assert "production analytics" in latex  # tailored bullet 0 used
    assert "SCADA" not in latex  # bullet 1 fell back to canonical
    assert report.tailored_bullets_used == 1
    assert any("forbidden term" in item for item in report.canonical_fallbacks)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def test_extract_json_object_handles_fences_and_prose() -> None:
    payload = extract_json_object('Here you go:\n```json\n{"role_family": "ML"}\n```')
    assert payload == {"role_family": "ML"}
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


def test_parse_structured_response_falls_back_to_keyword_family() -> None:
    assert CATALOG is not None
    selection = parse_structured_response(
        '{"role_family": "BANANA", "ranking": [], "bullets": {}}',
        CATALOG,
        FAMILIES,
        "embedded firmware microcontroller RTOS",
    )
    assert selection is not None
    assert selection.role_family_key == "EMBEDDED"


def test_parse_structured_response_filters_unknown_ids() -> None:
    assert CATALOG is not None
    selection = parse_structured_response(
        json.dumps(
            {
                "role_family": "ML",
                "ranking": ["nonexistent-entry", CATALOG.entries[0].entry_id],
                "bullets": {"nonexistent-entry": ["x"], CATALOG.entries[0].entry_id: ["y"]},
            }
        ),
        CATALOG,
        FAMILIES,
        "ml job",
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
            "role_family": "ML",
            "keywords": ["Python", "TensorFlow"],
            "ranking": [],
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
    assert summary["role_family"] == "ML"
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
    assert summary["role_family"] == "EMBEDDED"


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
    prompt = build_structured_prompt(
        "Software Engineer", "Build APIs", ["Python"], CATALOG, FAMILIES
    )
    for entry in CATALOG.entries:
        assert entry.entry_id in prompt
    assert "role_family" in prompt
    assert "ONLY a JSON object" in prompt


def test_build_structured_prompt_exposes_skill_anchors() -> None:
    """The model must see the candidate's real toolset to weave tools in."""
    assert CATALOG is not None
    prompt = build_structured_prompt(
        "Software Engineer", "Build APIs", ["Python"], CATALOG, FAMILIES
    )
    assert "<skill_anchors>" in prompt
    assert "kafka" in prompt
    assert "PRIMARY GOAL" in prompt

    # Profiles without anchors get no section (check disabled end to end).
    import copy

    bare = copy.deepcopy(CATALOG)
    bare.skill_anchors = ()
    prompt = build_structured_prompt("Engineer", "desc", [], bare, FAMILIES)
    assert "<skill_anchors>" not in prompt


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

    families = parse_role_guide(JBASE := (
        "== ROLE TYPE SELECTION GUIDE ==\n\n"
        "BACKEND roles\n  Keywords: backend, microservices\n"
        "  MUST SHOW: dotnet\n  SHOW: data\n  HIDE:\n"
    ))
    latex, report = render_structured_resume(
        catalog, families, StructuredSelection(role_family_key="BACKEND")
    )
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
    selection = StructuredSelection(role_family_key="ML", bullets=bullets)
    _, report = render_structured_resume(CATALOG, FAMILIES, selection)
    # Char pressure trims below the count ceiling — and may dip up to 2 below
    # the count minimum, because a second page is the worse outcome.
    assert report.visible_bullet_count >= CATALOG.render_config.min_visible_bullets - 2
    assert report.visible_bullet_count < CATALOG.render_config.max_visible_bullets


def test_baseinfo_blocks_parsed_by_tag() -> None:
    from services.resumes.structured import parse_baseinfo_blocks

    blocks, profile = parse_baseinfo_blocks(BASEINFO_TEXT)
    assert "ml" in blocks and "Goopter" in blocks["ml"]
    assert "Kafka" in blocks["ml"]
    assert "teaching" in blocks and "35%" in blocks["teaching"]
    # Multi-tag entries land under every tag.
    assert "hardware" in blocks and "fpga" in blocks
    assert blocks["hardware"] == blocks["fpga"]
    # Candidate-level facts come from the header, guide is excluded.
    assert "Ernest Choi" in profile
    assert "MUST SHOW" not in profile
    assert all("MUST SHOW" not in text for text in blocks.values())


def test_prompt_includes_baseinfo_background_per_entry() -> None:
    assert CATALOG is not None
    prompt = build_structured_prompt(
        "ML Engineer", "train models", [], CATALOG, FAMILIES
    )
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
    for family_key in ("EMBEDDED", "SOFTWARE", "FRONTEND", "ML"):
        _, report = render_structured_resume(
            CATALOG, FAMILIES, StructuredSelection(role_family_key=family_key)
        )
        assert report.fidelity_findings == []


def test_load_structured_profile_roundtrip() -> None:
    loaded = load_structured_profile(
        PROFILE_DIR / "template.tex", PROFILE_DIR / "baseinfo.txt"
    )
    assert loaded is not None
    catalog, families = loaded
    assert len(catalog.entries) == 10
    assert len(families) == 4


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


def test_render_selection_scope_ignores_all_tailored_bullets(monkeypatch) -> None:
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "rewrite_scope", "selection")
    selection = StructuredSelection(
        role_family_key="ML",
        bullets=_tailored_bullets_for_all_entries("zzmarker"),
    )
    latex, report = render_structured_resume(CATALOG, FAMILIES, selection)
    assert "zzmarker" not in latex
    assert report.tailored_bullets_used == 0
    assert report.rewrite_scope == "selection"
    # Selection decisions still apply: the render is family-shaped, not empty.
    assert report.visible_bullet_count > 0


def test_render_limited_scope_tailors_only_first_n_bullets(monkeypatch) -> None:
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "rewrite_scope", "limited")
    monkeypatch.setattr(CATALOG.render_config, "limited_rewrite_bullets", 4)
    selection = StructuredSelection(
        role_family_key="ML",
        bullets=_tailored_bullets_for_all_entries("zzmarker"),
    )
    latex, report = render_structured_resume(CATALOG, FAMILIES, selection)
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
        role_family_key="ML",
        bullets=_tailored_bullets_for_all_entries("zzmarker"),
    )
    latex, report = render_structured_resume(CATALOG, FAMILIES, selection)
    assert report.tailored_bullets_used == report.visible_bullet_count
    assert report.rewrite_scope == "full"


def test_prompt_mentions_selection_scope(monkeypatch) -> None:
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "rewrite_scope", "selection")
    prompt = build_structured_prompt("Engineer", "Job description", [], CATALOG, FAMILIES)
    assert "RENDERS ALL BULLET TEXT CANONICALLY" in prompt


def test_prompt_mentions_limited_rewrite_budget(monkeypatch) -> None:
    assert CATALOG is not None
    monkeypatch.setattr(CATALOG.render_config, "rewrite_scope", "limited")
    monkeypatch.setattr(CATALOG.render_config, "limited_rewrite_bullets", 5)
    prompt = build_structured_prompt("Engineer", "Job description", [], CATALOG, FAMILIES)
    assert "REWRITE BUDGET" in prompt
    assert "first 5" in prompt


def test_prompt_contains_prose_quality_rules() -> None:
    assert CATALOG is not None
    prompt = build_structured_prompt("Engineer", "Job description", [], CATALOG, FAMILIES)
    assert "NO EMPTY PURPOSE CLAUSES" in prompt
    assert "NEVER REDUCE SPECIFICITY" in prompt
    assert "VARY SENTENCE SHAPE" in prompt


# ---------------------------------------------------------------------------
# Filler-tail rejection and bold-density cap
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
def test_validate_bullet_rejects_empty_purpose_clause_tails(tail: str) -> None:
    text, reason = validate_tailored_bullet(
        f"Maintained \\textbf{{LaTeX}} documentation for the team{tail}.",
        CANONICAL,
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


def test_render_config_parses_family_skill_priority(tmp_path: Path) -> None:
    from services.resumes.structured import RenderConfig

    path = tmp_path / "structured_config.json"
    path.write_text(
        json.dumps({"family_skill_priority": {"embedded": ["C", "VHDL"], "ml": ["Python"]}}),
        encoding="utf-8",
    )
    config = RenderConfig.from_file(path)
    assert config.family_skill_priority == {"EMBEDDED": ("C", "VHDL"), "ML": ("Python",)}


def test_reorder_skills_family_priority_orders_without_listing_keywords() -> None:
    from services.resumes.structured import reorder_skills_for_keywords

    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Languages:} JavaScript/Node.js, Java, Python, C++, C, SQL, VHDL \\\\\n"
    )
    result = reorder_skills_for_keywords(tail, [], ("C", "C++", "VHDL"))
    languages = next(l for l in result.splitlines() if "Languages" in l)
    items = languages.split("}", 1)[1].rstrip(" \\").strip()
    # Family priority order leads; the rest keep canonical order.
    assert items.startswith("C, C++, VHDL")
    assert "JavaScript/Node.js" in items and "Java" in items


def test_reorder_skills_listing_keywords_outrank_family_priority() -> None:
    from services.resumes.structured import reorder_skills_for_keywords

    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Languages:} Java, Python, C++, C \\\\\n"
    )
    result = reorder_skills_for_keywords(tail, ["Python"], ("C", "C++"))
    languages = next(l for l in result.splitlines() if "Languages" in l)
    items = languages.split("}", 1)[1].rstrip(" \\").strip()
    assert items.startswith("Python, C, C++")


def test_reorder_skills_single_letter_priority_does_not_match_substrings() -> None:
    from services.resumes.structured import reorder_skills_for_keywords

    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Languages:} CSS, scikit-learn, C \\\\\n"
    )
    result = reorder_skills_for_keywords(tail, [], ("C",))
    languages = next(l for l in result.splitlines() if "Languages" in l)
    items = languages.split("}", 1)[1].rstrip(" \\").strip()
    # "C" must float, but must NOT drag CSS or scikit-learn with it.
    assert items == "C, CSS, scikit-learn"


def test_render_embedded_family_leads_skills_with_c_not_javascript() -> None:
    assert CATALOG is not None
    selection = StructuredSelection(role_family_key="EMBEDDED")
    latex, _ = render_structured_resume(CATALOG, FAMILIES, selection)
    languages = next(l for l in latex.splitlines() if l.startswith("\\textbf{Languages:}"))
    items = languages.split("}", 1)[1].strip()
    assert items.startswith("C,")
    assert items.index("C,") < items.index("JavaScript")


def test_render_ml_family_leads_skills_with_python() -> None:
    assert CATALOG is not None
    selection = StructuredSelection(role_family_key="ML")
    latex, _ = render_structured_resume(CATALOG, FAMILIES, selection)
    languages = next(l for l in latex.splitlines() if l.startswith("\\textbf{Languages:}"))
    items = languages.split("}", 1)[1].strip()
    assert items.startswith("Python")


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
    assert text.count("\\textbf{") == 3
    # Demoted tools remain as plain text.
    assert "Pandas" in text and "NumPy" in text


# ---------------------------------------------------------------------------
# Edge cases: parser hardening
# ---------------------------------------------------------------------------

MINI_GUIDE = """
== ROLE TYPE SELECTION GUIDE ==

ALPHA roles
  Keywords: alpha, widget
  MUST SHOW: one
  SHOW: two
  HIDE: three
"""


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

    families = parse_role_guide(MINI_GUIDE)
    latex, _ = render_structured_resume(
        catalog, families, StructuredSelection(role_family_key="ALPHA")
    )
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
        json.dumps({"role_family": "ML", "bullets": {goopter.title: ["Rewritten."]}}),
        CATALOG,
        FAMILIES,
        "ml job",
    )
    assert selection is not None
    assert goopter.entry_id in selection.bullets


def test_parse_response_accepts_dict_wrapped_bullets() -> None:
    assert CATALOG is not None
    entry = CATALOG.entries[0]
    selection = parse_structured_response(
        json.dumps(
            {"role_family": "ML", "bullets": {entry.entry_id: [{"text": "Wrapped bullet."}]}}
        ),
        CATALOG,
        FAMILIES,
        "ml job",
    )
    assert selection is not None
    assert selection.bullets[entry.entry_id] == ["Wrapped bullet."]


def test_parse_response_deduplicates_ranking() -> None:
    assert CATALOG is not None
    entry = CATALOG.entries[0]
    selection = parse_structured_response(
        json.dumps({"role_family": "ML", "ranking": [entry.entry_id, entry.entry_id]}),
        CATALOG,
        FAMILIES,
        "ml job",
    )
    assert selection is not None
    assert selection.ranking == [entry.entry_id]


# ---------------------------------------------------------------------------
# Edge cases: validation + detection
# ---------------------------------------------------------------------------

def test_validate_bullet_rejects_caret_outside_math() -> None:
    text, reason = validate_tailored_bullet("Improved x^41 throughput", CANONICAL)
    assert text is None and reason == "caret outside math mode"
    # ...but inside math mode it is fine.
    text, reason = validate_tailored_bullet("Improved $x^{41}$ throughput", CANONICAL)
    assert reason is None


def test_validate_bullet_respects_config_extra_forbidden_terms() -> None:
    from services.resumes.structured import RenderConfig

    config = RenderConfig(extra_forbidden_terms=("cobol",))
    text, reason = validate_tailored_bullet(
        "Modernized COBOL systems, reducing downtime by 41\\%.", CANONICAL, config
    )
    assert text is None and "forbidden term 'cobol'" in str(reason)


def test_detect_role_family_generic_software_listing() -> None:
    """A listing that only says 'software' must not land on the first family."""
    family = detect_role_family(
        "Software Developer Intern working on internal tooling", FAMILIES
    )
    assert family is not None and family.key == "SOFTWARE"


def test_detect_role_family_no_signal_prefers_most_inclusive() -> None:
    family = detect_role_family("Administrative assistant position", FAMILIES)
    assert family is not None
    widest = max(FAMILIES, key=lambda f: len(f.must_show | f.show))
    assert family.key == widest.key


# ---------------------------------------------------------------------------
# Edge cases: per-profile config + guidance
# ---------------------------------------------------------------------------

def test_structured_config_overrides_bullet_budget(tmp_path: Path) -> None:
    profile = _profile_copy(tmp_path)
    (profile / "structured_config.json").write_text(
        json.dumps({"min_visible_bullets": 8, "max_visible_bullets": 11}),
        encoding="utf-8",
    )
    loaded = load_structured_profile(profile / "template.tex", profile / "baseinfo.txt")
    assert loaded is not None
    catalog, families = loaded
    assert catalog.render_config.max_visible_bullets == 11
    _, report = render_structured_resume(
        catalog, families, StructuredSelection(role_family_key="ML")
    )
    assert report.visible_bullet_count <= 11


def test_structured_config_invalid_json_falls_back_to_defaults(tmp_path: Path) -> None:
    profile = _profile_copy(tmp_path)
    (profile / "structured_config.json").write_text("{not json", encoding="utf-8")
    loaded = load_structured_profile(profile / "template.tex", profile / "baseinfo.txt")
    assert loaded is not None
    catalog, _ = loaded
    assert catalog.render_config.max_visible_bullets == MAX_VISIBLE_BULLETS


def test_structured_guidance_flows_into_prompt(tmp_path: Path) -> None:
    profile = _profile_copy(tmp_path)
    (profile / "structured_guidance.txt").write_text(
        "Voice: sample guidance marker.", encoding="utf-8"
    )
    loaded = load_structured_profile(profile / "template.tex", profile / "baseinfo.txt")
    assert loaded is not None
    catalog, families = loaded
    prompt = build_structured_prompt(
        "Engineer", "desc", [], catalog, families, extra_guidance=catalog.guidance
    )
    assert "sample guidance marker" in prompt
    assert "<profile_guidance>" in prompt


# ---------------------------------------------------------------------------
# Entry exclusion + weak-bullet trimming
# ---------------------------------------------------------------------------

def test_exclusion_of_show_entry_is_honoured_when_budget_allows() -> None:
    """FRONTEND family: excluding Obotz (SHOW, teaching) drops it — the page
    can still be filled from the remaining SHOW entries (pfs+terrain+markham)."""
    assert CATALOG is not None
    obotz = next(e for e in CATALOG.entries if "obotz" in e.entry_id)
    selection = StructuredSelection(
        role_family_key="FRONTEND", exclusions=[obotz.entry_id]
    )
    latex, report = render_structured_resume(CATALOG, FAMILIES, selection)
    assert "Obotz Robotics" not in latex
    assert obotz.entry_id in report.excluded_entries
    # Page must still be filled from remaining entries.
    assert report.visible_bullet_count >= CATALOG.render_config.min_visible_bullets


def test_exclusion_of_must_show_entry_is_ignored() -> None:
    assert CATALOG is not None
    goopter = next(e for e in CATALOG.entries if "goopter" in e.entry_id)
    selection = StructuredSelection(role_family_key="ML", exclusions=[goopter.entry_id])
    latex, report = render_structured_resume(CATALOG, FAMILIES, selection)
    assert "Goopter" in latex
    assert goopter.entry_id in report.ignored_exclusions
    assert goopter.entry_id not in report.excluded_entries


def test_exclusions_rolled_back_when_page_cannot_be_filled() -> None:
    """Excluding every SHOW entry must be rolled back to reach the minimum."""
    assert CATALOG is not None
    ml = next(f for f in FAMILIES if f.key == "ML")
    show_ids = [
        e.entry_id
        for e in CATALOG.entries
        if (set(c.lower() for c in e.categories) & ml.show)
        and not (set(c.lower() for c in e.categories) & ml.must_show)
    ]
    selection = StructuredSelection(role_family_key="ML", exclusions=show_ids)
    _, report = render_structured_resume(CATALOG, FAMILIES, selection)
    assert report.visible_bullet_count >= CATALOG.render_config.min_visible_bullets
    assert report.ignored_exclusions  # some exclusions were re-admitted
    # Honoured + ignored must account for every requested exclusion.
    assert set(report.excluded_entries) | set(report.ignored_exclusions) >= set(show_ids)


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
        role_family_key="SOFTWARE",
        weak_bullets={pfs.entry_id: [0]},  # flag the FIRST bullet as weakest
    )
    latex, report = render_structured_resume(catalog, FAMILIES, selection)
    if pfs.entry_id in report.visible_entries:
        rendered_after_pfs = latex.split(pfs.header, 1)[1].split("\\end{itemize}", 1)[0]
        kept_first = pfs.bullets[0].split("\\", 1)[0][:40] in rendered_after_pfs
        kept_last = pfs.bullets[-1][:40] in rendered_after_pfs
        # If trimming touched this entry, bullet 0 goes before the last one.
        if not (kept_first and kept_last):
            assert kept_last and not kept_first


def test_inclusion_promotes_hidden_entry_when_listing_calls_for_it() -> None:
    """ML family hides terrain (graphics); an explicit include promotes it."""
    assert CATALOG is not None
    terrain = next(e for e in CATALOG.entries if "terrain" in e.entry_id)
    selection = StructuredSelection(
        role_family_key="ML",
        ranking=[terrain.entry_id],  # model ranks its requested entry high
        inclusions=[terrain.entry_id],
    )
    latex, report = render_structured_resume(CATALOG, FAMILIES, selection)
    assert "Algorithm Terrain Visualizer" in latex
    assert terrain.entry_id in report.included_extras
    assert terrain.entry_id not in report.hidden_entries
    assert report.visible_bullet_count <= CATALOG.render_config.max_visible_bullets


def test_inclusion_ignored_when_budget_has_no_room() -> None:
    assert CATALOG is not None
    import copy
    from services.resumes.structured import RenderConfig

    catalog = copy.deepcopy(CATALOG)
    # EMBEDDED must-show already needs 8 bullets; ceiling of 9 leaves no room
    # for a 3-bullet promoted entry.
    catalog.render_config = RenderConfig(min_visible_bullets=8, max_visible_bullets=9)
    webmsg = next(e for e in catalog.entries if "web-messaging" in e.entry_id)
    selection = StructuredSelection(
        role_family_key="EMBEDDED", inclusions=[webmsg.entry_id]
    )
    latex, report = render_structured_resume(catalog, FAMILIES, selection)
    assert "Web Messaging App" not in latex
    assert webmsg.entry_id in report.ignored_inclusions


def test_inclusion_cap_limits_promotions_to_two() -> None:
    """Requesting every hidden entry only promotes MAX_INCLUSIONS of them."""
    assert CATALOG is not None
    ml = next(f for f in FAMILIES if f.key == "ML")
    hidden_ids = [
        e.entry_id
        for e in CATALOG.entries
        if not (set(c.lower() for c in e.categories) & (ml.must_show | ml.show))
    ]
    assert len(hidden_ids) > 2
    selection = StructuredSelection(role_family_key="ML", inclusions=hidden_ids)
    _, report = render_structured_resume(CATALOG, FAMILIES, selection)
    assert len(report.included_extras) <= 2
    assert set(report.included_extras) | set(report.ignored_inclusions) >= set(hidden_ids)


def test_inclusion_of_visible_entry_is_a_noop() -> None:
    """Including an entry that is already MUST SHOW/SHOW changes nothing."""
    assert CATALOG is not None
    goopter = next(e for e in CATALOG.entries if "goopter" in e.entry_id)
    selection = StructuredSelection(role_family_key="ML", inclusions=[goopter.entry_id])
    latex, report = render_structured_resume(CATALOG, FAMILIES, selection)
    assert report.included_extras == []
    assert goopter.entry_id in report.ignored_inclusions
    assert latex.count("Goopter") == 1  # no duplicate rendering


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
    # But a tool from nowhere in the entry is still rejected.
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
        role_family_key="EMBEDDED",
        bullet_order={markham.entry_id: [2, 0, 1]},
    )
    latex, report = render_structured_resume(CATALOG, FAMILIES, selection)
    if markham.entry_id in report.visible_entries:
        block = latex.split(markham.header, 1)[1].split("\\end{itemize}", 1)[0]
        first_bullet = block.split("\\item", 2)[1]
        assert "20\\%" in first_bullet  # canonical bullet 2 leads


def test_bullet_order_invalid_indices_ignored() -> None:
    assert CATALOG is not None
    markham = next(e for e in CATALOG.entries if "markham" in e.entry_id)
    selection = StructuredSelection(
        role_family_key="EMBEDDED",
        bullet_order={markham.entry_id: [9, 7]},  # out of range
    )
    latex, report = render_structured_resume(CATALOG, FAMILIES, selection)
    # All three canonical bullets still render, original order.
    assert "41\\%" in latex and "14\\%" in latex and "20\\%" in latex


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
    selection = StructuredSelection(
        role_family_key="ML", keywords=["Kafka", "TensorFlow"]
    )
    latex, _ = render_structured_resume(CATALOG, FAMILIES, selection)
    tools_line = next(l for l in latex.splitlines() if l.startswith("\\textbf{Tools:}"))
    assert tools_line.index("Kafka") < tools_line.index("Git")


def test_parse_response_reads_bullet_order() -> None:
    assert CATALOG is not None
    entry = CATALOG.entries[0]
    selection = parse_structured_response(
        json.dumps({"role_family": "ML", "bullet_order": {entry.entry_id: [1, 0]}}),
        CATALOG,
        FAMILIES,
        "ml job",
    )
    assert selection is not None
    assert selection.bullet_order == {entry.entry_id: [1, 0]}


def test_parse_response_reads_include_key() -> None:
    assert CATALOG is not None
    terrain = next(e for e in CATALOG.entries if "terrain" in e.entry_id)
    selection = parse_structured_response(
        json.dumps({"role_family": "ML", "include": [terrain.entry_id, "bogus"]}),
        CATALOG,
        FAMILIES,
        "ml job",
    )
    assert selection is not None
    assert selection.inclusions == [terrain.entry_id]


def test_parse_response_reads_exclude_and_weak_bullets() -> None:
    assert CATALOG is not None
    obotz = next(e for e in CATALOG.entries if "obotz" in e.entry_id)
    pfs = next(e for e in CATALOG.entries if "particle" in e.entry_id)
    selection = parse_structured_response(
        json.dumps(
            {
                "role_family": "SOFTWARE",
                "exclude": [obotz.entry_id, "unknown-entry"],
                "weak_bullets": {pfs.entry_id: [2, 0], "unknown-entry": [1]},
            }
        ),
        CATALOG,
        FAMILIES,
        "backend job",
    )
    assert selection is not None
    assert selection.exclusions == [obotz.entry_id]
    assert selection.weak_bullets == {pfs.entry_id: [2, 0]}


def test_profile_cache_purge_keeps_structured_files(tmp_path: Path) -> None:
    """ensure_profile_cache must never delete structured_config/guidance.

    Regression: the purge originally allowed only the three legacy files, so
    the structured pipeline's optional files were wiped on every bot run.
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
    (profile / "structured_guidance.txt").write_text("guidance", encoding="utf-8")
    (profile / "junk.tmp").write_text("junk", encoding="utf-8")

    ensure_profile_cache("user.profile", cache_root)

    assert (profile / "structured_config.json").exists()
    assert (profile / "structured_guidance.txt").exists()
    assert not (profile / "junk.tmp").exists()  # real junk still purged


def test_profile_guidance_file_is_loaded_for_real_profile() -> None:
    loaded = load_structured_profile(
        PROFILE_DIR / "template.tex", PROFILE_DIR / "baseinfo.txt"
    )
    assert loaded is not None
    catalog, _ = loaded
    assert "Seniority calibration" in catalog.guidance


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

== ROLE TYPE SELECTION GUIDE ==

BACKEND roles
  Keywords: backend, microservices, API, .NET
  MUST SHOW: dotnet
  SHOW: data
  HIDE:

DATA roles
  Keywords: analytics, SQL, reporting, data
  MUST SHOW: data
  SHOW: dotnet
  HIDE:
"""


def _jane_profile(tmp_path: Path) -> Path:
    profile = tmp_path / "jane"
    profile.mkdir()
    (profile / "template.tex").write_text(JANE_TEMPLATE, encoding="utf-8")
    (profile / "baseinfo.txt").write_text(JANE_BASEINFO, encoding="utf-8")
    (profile / "instructions.txt").write_text("n/a", encoding="utf-8")
    # Jane really works with C#/.NET: replace the default forbidden terms,
    # and use a bullet budget matching her small resume.
    (profile / "structured_config.json").write_text(
        json.dumps(
            {
                "forbidden_terms": ["scada", "hvac"],
                "min_visible_bullets": 3,
                "max_visible_bullets": 4,
                "min_must_show_bullets": 1,
            }
        ),
        encoding="utf-8",
    )
    return profile


def test_second_profile_same_conventions_works_end_to_end(tmp_path: Path) -> None:
    profile = _jane_profile(tmp_path)
    loaded = load_structured_profile(profile / "template.tex", profile / "baseinfo.txt")
    assert loaded is not None
    catalog, families = loaded
    assert [e.entry_id for e in catalog.entries] == [
        "software-engineer-contoso",
        "data-analyst-fabrikam",
    ]
    assert [f.key for f in families] == ["BACKEND", "DATA"]
    assert all(e.smallskip for e in catalog.entries)

    job = extract_job_context_from_message("[LinkedIn] Backend Developer\nhttps://example.com/job")
    assert job is not None
    response = json.dumps(
        {
            "role_family": "BACKEND",
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
    assert summary["role_family"] == "BACKEND"
    assert summary["tailored_bullets_used"] == 2


def test_second_profile_forbidden_override_still_blocks_config_terms(tmp_path: Path) -> None:
    profile = _jane_profile(tmp_path)
    loaded = load_structured_profile(profile / "template.tex", profile / "baseinfo.txt")
    assert loaded is not None
    catalog, _ = loaded
    text, reason = validate_tailored_bullet(
        "Built SCADA dashboards handling 2M requests daily.",
        "Built C\\# microservices on .NET handling 2M requests daily.",
        catalog.render_config,
    )
    assert text is None and "forbidden term 'scada'" in str(reason)
