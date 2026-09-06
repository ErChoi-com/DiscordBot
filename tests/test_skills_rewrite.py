"""Strong-aggressive skills-rewrite tests: the real LLM pass that replaces
the deterministic reorder/inject path (structured._rewrite_skills_for_jd)
with judgement calls a regex keyword matcher can't make, while re-validating
every returned item against the candidate's OWNED skill_anchors in code —
the model is never trusted to police its own "never invent a skill" rule."""
from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from services.resumes import listing
from services.resumes.configkey import GeminiSettings
from services.resumes.structured import (
    StructuredSelection,
    _reconcile_skills_and_bullets,
    apply_llm_skills_rewrite,
    build_skills_rewrite_prompt,
    gather_rewritten_bullet_text,
    normalize_skills_rewrite_payload,
    parse_template_catalog,
    render_structured_resume,
)

EXPERIENCE_ENTRY = (
    "% [software]\n"
    "\\textbf{Software Intern,} {Acme} -- Remote \\hfill 2024 -- 2025 \\\\\n"
    "\\begin{itemize}\n"
    "  \\item Built REST APIs in Python and Flask for internal tooling.\n"
    "  \\item Wrote automated pytest suites for CI.\n"
    "\\end{itemize}"
)

SKILLS_TEMPLATE = (
    "\\documentclass{article}\n\\begin{document}\n"
    "\\centerline{Name}\n\n"
    "\\section*{Experience}\n\n" + EXPERIENCE_ENTRY + "\n\n"
    "\\section*{Skills}\n"
    "\\textbf{Languages:} JavaScript, Java, Python, C++, C, SQL, HTML, CSS \\\\\n"
    "\\textbf{Tools:} Git, PostgreSQL, Docker \\\\\n"
    "\\section*{Education}\n"
    "\\textbf{School} -- Degree \\hfill 2023--2027\n"
    "\\end{document}\n"
)


@pytest.fixture()
def catalog():
    parsed = parse_template_catalog(SKILLS_TEMPLATE)
    assert parsed is not None
    parsed.skill_anchors = (
        "javascript", "java", "python", "c++", "c", "sql", "html", "css",
        "git", "postgresql", "docker", "kubernetes", "aws",
    )
    parsed.skill_anchor_display = {
        "javascript": "JavaScript", "java": "Java", "python": "Python",
        "c++": "C++", "c": "C", "sql": "SQL", "html": "HTML", "css": "CSS",
        "git": "Git", "postgresql": "PostgreSQL", "docker": "Docker",
        "kubernetes": "Kubernetes", "aws": "AWS",
    }
    parsed.render_config = replace(parsed.render_config, strong_aggressive=True, aggressive=True)
    return parsed


def _settings() -> GeminiSettings:
    return GeminiSettings(api_key="test-key", model="gemini-test")


def _entry_id(catalog) -> str:
    return catalog.entries[0].entry_id


# ── gather_rewritten_bullet_text ─────────────────────────────────────────────


def test_gather_bullet_text_uses_provided_over_canonical(catalog) -> None:
    entry_id = _entry_id(catalog)
    selection = StructuredSelection(bullets={entry_id: ["Deployed Kubernetes clusters on AWS."]})
    text = gather_rewritten_bullet_text(catalog, selection)
    assert "Kubernetes clusters on AWS" in text
    assert "REST APIs in Python and Flask" not in text


def test_gather_bullet_text_falls_back_to_canonical_when_blank(catalog) -> None:
    entry_id = _entry_id(catalog)
    selection = StructuredSelection(bullets={entry_id: [""]})
    text = gather_rewritten_bullet_text(catalog, selection)
    assert "Built REST APIs in Python and Flask" in text


def test_gather_bullet_text_uses_canonical_with_no_selection(catalog) -> None:
    text = gather_rewritten_bullet_text(catalog, StructuredSelection())
    assert "Built REST APIs in Python and Flask" in text
    assert "Wrote automated pytest suites" in text


# ── build_skills_rewrite_prompt ──────────────────────────────────────────────


def test_prompt_none_when_no_skill_anchors(catalog) -> None:
    catalog.skill_anchors = ()
    prompt = build_skills_rewrite_prompt(catalog, "Backend Engineer", "Build APIs.", "", [])
    assert prompt is None


def test_prompt_none_when_no_labeled_skills_lines(catalog) -> None:
    catalog.tail = "\\section*{Skills}\nplain text, no labels\n\\section*{Education}\nstuff"
    prompt = build_skills_rewrite_prompt(catalog, "Backend Engineer", "Build APIs.", "", [])
    assert prompt is None


def test_prompt_carries_listing_owned_skills_labels_and_bullets(catalog) -> None:
    prompt = build_skills_rewrite_prompt(
        catalog, "Backend Engineer", "Build scalable APIs with Python.",
        "Built REST APIs in Python and Flask.", ["python", "sql"],
    )
    assert prompt is not None
    assert "Backend Engineer" in prompt
    assert "Build scalable APIs with Python." in prompt
    assert "PostgreSQL" in prompt  # display casing, not lowercase anchor
    assert '"Languages:"' in prompt and '"Tools:"' in prompt
    assert "Built REST APIs in Python and Flask." in prompt
    assert "python, sql" in prompt
    # Fabrication is the point of this mode's Skills section (2026-08-28); what
    # the prompt must still carry is the listing-first ordering rule and the
    # plausibility limit that replaced the ownership rule.
    assert "Lead with the technologies, tools and platforms THIS LISTING names" in prompt
    assert "No invented product names, no tools that do not exist." in prompt
    # Coursework is never offered to this call.
    assert "Relevant Courses" not in prompt


def test_prompt_omits_bullets_block_when_no_bullet_text(catalog) -> None:
    prompt = build_skills_rewrite_prompt(catalog, "Backend Engineer", "Build APIs.", "", [])
    assert prompt is not None
    assert "Resume content this Skills section supports" not in prompt


def test_prompt_trims_long_job_description_instead_of_sending_it_whole(catalog) -> None:
    # Requirement-shaped tail sentence, placed well past a 2500-char head cut.
    filler = "This role also involves general collaboration and teamwork. " * 60
    tail = "Requirements: must know Python and SQL."
    long_description = filler + tail
    assert len(long_description) > 2500

    prompt = build_skills_rewrite_prompt(catalog, "Backend Engineer", long_description, "", [])
    assert prompt is not None
    assert len(prompt) < len(long_description) + len(SKILLS_TEMPLATE)
    assert long_description not in prompt  # not sent verbatim/whole
    assert "Requirements: must know Python and SQL." in prompt  # excerpt keeps the tail requirement


# ── apply_llm_skills_rewrite ─────────────────────────────────────────────────

SKILLS_TAIL = (
    "\\section*{Skills}\n"
    "\\textbf{Languages:} JavaScript, Java, Python, C++, C, SQL, HTML, CSS \\\\\n"
    "\\textbf{Tools:} Git, PostgreSQL, Docker \\\\\n"
    "\\section*{Education}\n"
    "\\textbf{School} -- Degree \\hfill 2023--2027"
)


def test_apply_rewrite_reorders_within_owned_items() -> None:
    tail = apply_llm_skills_rewrite(
        SKILLS_TAIL,
        {"Languages:": ["Python", "SQL", "JavaScript"]},
        ("javascript", "java", "python", "sql"),
        {"javascript": "JavaScript", "java": "Java", "python": "Python", "sql": "SQL"},
    )
    langs = next(l for l in tail.splitlines() if "Languages:" in l)
    assert langs.rstrip().endswith("\\\\")
    items = [i.strip() for i in langs.split(":}")[1].rstrip(" \\").split(",")]
    assert items == ["Python", "SQL", "JavaScript"]
    # Untouched label keeps its original line verbatim.
    tools = next(l for l in tail.splitlines() if "Tools:" in l)
    assert "Git, PostgreSQL, Docker" in tools


# ── jd_inject_tools: the deterministic path's sanctioned aggressive-mode ────
# tool injection (structured._extract_jd_tools) must survive the LLM overlay,
# not just fall under the same "never invent" filter as everything else.


def test_apply_rewrite_llm_may_directly_include_a_jd_inject_tool() -> None:
    tail = apply_llm_skills_rewrite(
        SKILLS_TAIL,
        {"Tools:": ["Kubernetes", "Docker", "Git"]},  # Kubernetes not owned
        ("git", "docker"),
        {"git": "Git", "docker": "Docker"},
        jd_inject_tools=("Kubernetes",),
    )
    tools = next(l for l in tail.splitlines() if "Tools:" in l)
    assert "Kubernetes" in tools  # sanctioned JD tool survives, unlike a genuinely invented one


def test_apply_rewrite_keeps_jd_injected_tool_even_when_llm_response_omits_it() -> None:
    """Regression: _rewrite_skills_for_jd (the deterministic baseline this
    function overlays) unconditionally injects every JD-matched tool into
    its category line and guarantees it survives the line's item cap. A
    naive 'replace the line with whatever the LLM returned' overlay would
    silently drop that guarantee whenever the model's own response for that
    label just doesn't mention the tool — even though it never invented
    anything and never had to reject it."""
    baseline_with_injection = (
        "\\section*{Skills}\n"
        "\\textbf{Tools:} Git, PostgreSQL, Docker, Kubernetes \\\\\n"  # already injected upstream
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree \\hfill 2023--2027"
    )
    tail = apply_llm_skills_rewrite(
        baseline_with_injection,
        {"Tools:": ["Docker", "Git", "PostgreSQL"]},  # model never mentions Kubernetes
        ("git", "postgresql", "docker"),
        {"git": "Git", "postgresql": "PostgreSQL", "docker": "Docker"},
        jd_inject_tools=("Kubernetes",),
    )
    tools = next(l for l in tail.splitlines() if "Tools:" in l)
    assert "Kubernetes" in tools


def test_apply_rewrite_jd_injected_tool_survives_the_max_items_cap() -> None:
    baseline_with_injection = (
        "\\section*{Skills}\n"
        "\\textbf{Tools:} Git, PostgreSQL, Docker, Kubernetes \\\\\n"
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree \\hfill 2023--2027"
    )
    tail = apply_llm_skills_rewrite(
        baseline_with_injection,
        {"Tools:": ["Docker", "Git", "PostgreSQL"]},
        ("git", "postgresql", "docker"),
        {},
        max_items_per_line=2,  # tighter than the model's own 3 items
        jd_inject_tools=("Kubernetes",),
    )
    tools = next(l for l in tail.splitlines() if "Tools:" in l)
    items = [i.strip() for i in tools.split(":}")[1].rstrip(" \\").split(",")]
    assert "Kubernetes" in items  # guaranteed item is never the one that gets cut
    assert len(items) == 2


def test_normalize_payload_allows_jd_inject_tools_item() -> None:
    result = normalize_skills_rewrite_payload(
        {"Tools:": ["Docker", "Kubernetes", "Rust"]},
        ("docker",),
        {"docker": "Docker"},
        jd_inject_tools=("Kubernetes",),
    )
    assert result["tools"] == ["Docker", "Kubernetes"]  # Rust (truly invented) still dropped


def test_prompt_includes_jd_inject_tools_block_when_present(catalog) -> None:
    prompt = build_skills_rewrite_prompt(
        catalog, "Backend Engineer", "Build APIs.", "", [], jd_inject_tools=("Kubernetes",)
    )
    assert prompt is not None
    assert "Kubernetes" in prompt
    assert "NOT in\nthe candidate's owned list" in prompt


def test_prompt_omits_jd_inject_tools_block_when_absent(catalog) -> None:
    prompt = build_skills_rewrite_prompt(catalog, "Backend Engineer", "Build APIs.", "", [])
    assert prompt is not None
    assert "NOT in\nthe candidate's owned list" not in prompt


def test_apply_rewrite_drops_invented_skill_not_in_owned_list() -> None:
    tail = apply_llm_skills_rewrite(
        SKILLS_TAIL,
        {"Languages:": ["Python", "Rust", "Go"]},  # Rust/Go never owned
        ("javascript", "java", "python", "c++", "c", "sql", "html", "css"),
        {"python": "Python"},
    )
    langs = next(l for l in tail.splitlines() if "Languages:" in l)
    assert "Rust" not in langs and "Go" not in langs
    assert "Python" in langs


def test_apply_rewrite_ignores_case_and_colon_variance_in_label_match() -> None:
    tail = apply_llm_skills_rewrite(
        SKILLS_TAIL,
        {"languages": ["SQL", "Python"]},  # lowercase, no colon
        ("python", "sql"),
        {"python": "Python", "sql": "SQL"},
    )
    langs = next(l for l in tail.splitlines() if "Languages:" in l)
    items = [i.strip() for i in langs.split(":}")[1].rstrip(" \\").split(",")]
    assert items == ["SQL", "Python"]


def test_apply_rewrite_falls_back_to_input_tail_when_label_all_invalid() -> None:
    tail = apply_llm_skills_rewrite(
        SKILLS_TAIL,
        {"Languages:": ["Rust", "Go"]},  # entirely invented, nothing survives
        ("python", "sql"),
        {},
    )
    langs = next(l for l in tail.splitlines() if "Languages:" in l)
    assert "JavaScript, Java, Python" in langs  # original line preserved


def test_apply_rewrite_caps_at_max_items_per_line() -> None:
    tail = apply_llm_skills_rewrite(
        SKILLS_TAIL,
        {"Languages:": ["Python", "SQL", "JavaScript", "Java"]},
        ("javascript", "java", "python", "sql"),
        {},
        max_items_per_line=2,
    )
    langs = next(l for l in tail.splitlines() if "Languages:" in l)
    items = [i.strip() for i in langs.split(":}")[1].rstrip(" \\").split(",")]
    assert items == ["Python", "SQL"]


def test_apply_rewrite_deduplicates_case_insensitively() -> None:
    tail = apply_llm_skills_rewrite(
        SKILLS_TAIL,
        {"Languages:": ["Python", "python", "PYTHON", "SQL"]},
        ("python", "sql"),
        {"python": "Python", "sql": "SQL"},
    )
    langs = next(l for l in tail.splitlines() if "Languages:" in l)
    items = [i.strip() for i in langs.split(":}")[1].rstrip(" \\").split(",")]
    assert items == ["Python", "SQL"]


def test_apply_rewrite_non_dict_payload_returns_tail_unchanged() -> None:
    assert apply_llm_skills_rewrite(SKILLS_TAIL, ["not", "a", "dict"], (), {}) == SKILLS_TAIL
    assert apply_llm_skills_rewrite(SKILLS_TAIL, None, (), {}) == SKILLS_TAIL


def test_apply_rewrite_empty_payload_returns_tail_unchanged() -> None:
    assert apply_llm_skills_rewrite(SKILLS_TAIL, {}, ("python",), {}) == SKILLS_TAIL


# ── render_structured_resume integration ─────────────────────────────────────


def test_render_uses_llm_rewritten_skills_over_deterministic_order(catalog) -> None:
    selection = StructuredSelection(ranking=[_entry_id(catalog)], keywords=["css"])
    latex, _ = render_structured_resume(
        catalog,
        selection,
        llm_rewritten_skills={"Languages:": ["Python", "SQL", "JavaScript", "Java", "C++", "C", "HTML", "CSS"]},
    )
    langs = next(l for l in latex.splitlines() if "Languages:" in l)
    items = [i.strip() for i in langs.split(":}")[1].rstrip(" \\").split(",")]
    assert items[0] == "Python"  # LLM order wins, not the deterministic css-first order


def test_render_falls_back_to_deterministic_when_llm_rewrite_absent(catalog) -> None:
    selection = StructuredSelection(ranking=[_entry_id(catalog)], keywords=["css"])
    latex, _ = render_structured_resume(catalog, selection, llm_rewritten_skills=None)
    langs = next(l for l in latex.splitlines() if "Languages:" in l)
    items = [i.strip() for i in langs.split(":}")[1].rstrip(" \\").split(",")]
    assert items[0] == "CSS"  # deterministic keyword-match path, unchanged behavior


def test_render_llm_rewrite_admits_unowned_skills_in_strong_aggressive(catalog) -> None:
    """Strong-aggressive fabricates Skills from the listing, like its bullets.

    This assertion was inverted on 2026-08-28 at the profile owner's direction
    ("let everything in skills be fabricated but courses"). The ownership guard
    still holds everywhere else -- see the non-strong-aggressive test below,
    which is the one that protects normal and --aggressive output.
    """
    selection = StructuredSelection(ranking=[_entry_id(catalog)])
    latex, _ = render_structured_resume(
        catalog,
        selection,
        llm_rewritten_skills={"Languages:": ["Python", "Rust", "Elixir"]},
    )
    assert "Rust" in latex and "Elixir" in latex


def test_render_llm_rewrite_never_introduces_invented_skill_outside_strong_aggressive(
    catalog,
) -> None:
    """The ownership guard, still enforced for every other mode."""
    catalog.render_config = replace(
        catalog.render_config, strong_aggressive=False, aggressive=False
    )
    selection = StructuredSelection(ranking=[_entry_id(catalog)])
    latex, _ = render_structured_resume(
        catalog,
        selection,
        llm_rewritten_skills={"Languages:": ["Python", "Rust", "Elixir"]},
    )
    assert "Rust" not in latex and "Elixir" not in latex
    assert "Python" in latex


def test_render_llm_rewrite_keeps_jd_injected_tool_the_response_omits(catalog) -> None:
    """End-to-end regression: strong-aggressive mode already unconditionally
    injects a JD-named tool it doesn't own (structured._extract_jd_tools);
    the LLM skills rewrite must not silently undo that just because its own
    response for that line doesn't happen to repeat the tool."""
    selection = StructuredSelection(ranking=[_entry_id(catalog)])
    latex, _ = render_structured_resume(
        catalog,
        selection,
        jd_inject_tools=("Terraform",),  # not in catalog.skill_anchors
        llm_rewritten_skills={"Tools:": ["Docker", "Git"]},  # LLM never mentions Terraform
    )
    tools = next(l for l in latex.splitlines() if "Tools:" in l)
    assert "Terraform" in tools


def test_render_llm_rewrite_lets_llm_place_a_jd_injected_tool_directly(catalog) -> None:
    selection = StructuredSelection(ranking=[_entry_id(catalog)])
    latex, _ = render_structured_resume(
        catalog,
        selection,
        jd_inject_tools=("Terraform",),
        llm_rewritten_skills={"Tools:": ["Terraform", "Docker", "Git"]},
    )
    tools = next(l for l in latex.splitlines() if "Tools:" in l)
    assert "Terraform" in tools


# ── listing._llm_rewrite_skills (provider loop) ──────────────────────────────


def test_llm_rewrite_skills_success_via_fake_provider(catalog, monkeypatch) -> None:
    seen_prompts: list[str] = []

    def fake_gemini(settings, prompt, cache_name, client_factory, model_override=None, json_response=False):
        seen_prompts.append(prompt)
        return json.dumps({"skills": {"Languages:": ["Python", "SQL"], "Tools:": ["Docker", "Git"]}})

    monkeypatch.setattr(listing, "_generate_with_gemini", fake_gemini)
    status, skills = listing._llm_rewrite_skills(
        _settings(), catalog, "Backend Engineer", "Build APIs with Python.",
        "Built REST APIs in Python.", ["python"], None,
    )
    assert status == "ok:gemini-flash"
    assert skills == {"Languages:": ["Python", "SQL"], "Tools:": ["Docker", "Git"]}
    assert len(seen_prompts) == 1
    assert "Backend Engineer" in seen_prompts[0]


def test_llm_rewrite_skills_skipped_when_no_skill_anchors(catalog, monkeypatch) -> None:
    catalog.skill_anchors = ()
    called = False

    def fake_gemini(*args, **kwargs):
        nonlocal called
        called = True
        return "{}"

    monkeypatch.setattr(listing, "_generate_with_gemini", fake_gemini)
    status, skills = listing._llm_rewrite_skills(
        _settings(), catalog, "Backend Engineer", "Build APIs.", "", [], None
    )
    assert status == "skipped"
    assert skills is None
    assert called is False


def test_llm_rewrite_skills_fail_open_when_all_providers_fail(catalog, monkeypatch) -> None:
    def broken_gemini(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(listing, "_generate_with_gemini", broken_gemini)
    status, skills = listing._llm_rewrite_skills(
        _settings(), catalog, "Backend Engineer", "Build APIs.", "", [], None
    )
    assert status.startswith("failed:")
    assert skills is None


def test_llm_rewrite_skills_malformed_response_fails_open(catalog, monkeypatch) -> None:
    monkeypatch.setattr(
        listing, "_generate_with_gemini", lambda *a, **k: json.dumps({"keywords": ["alpha"]})
    )
    status, skills = listing._llm_rewrite_skills(
        _settings(), catalog, "Backend Engineer", "Build APIs.", "", [], None
    )
    assert status.startswith("failed:")
    assert "no skills object" in status
    assert skills is None


def test_llm_rewrite_skills_accepts_listing_sourced_items_in_strong_aggressive(
    catalog, monkeypatch
) -> None:
    """Items outside skill_anchors are the POINT of this mode's Skills section
    (profile owner, 2026-08-28), so a response made of them is accepted on the
    first provider rather than triggering a retry.

    What still forces a retry is a response with nothing usable in it at all --
    covered by test_llm_rewrite_skills_malformed_response_fails_open. The
    ownership filter itself is still exercised at
    test_normalize_payload_filters_invented_dedupes_and_keys_by_label, which
    calls normalize_skills_rewrite_payload with allow_unowned left False.
    """
    calls: list[object] = []

    def fake_gemini(settings, prompt, cache_name, client_factory, model_override=None, json_response=False):
        calls.append(model_override)
        if model_override:  # gemini-flash is tried first (see candidate sort)
            return json.dumps({"skills": {"Languages:": ["Rust", "Go"]}})  # entirely invented
        return json.dumps({"skills": {"Languages:": ["Python", "SQL"]}})  # valid, from gemini

    monkeypatch.setattr(listing, "_generate_with_gemini", fake_gemini)
    status, skills = listing._llm_rewrite_skills(
        _settings(), catalog, "Backend Engineer", "Build APIs.", "", [], None
    )
    assert status == "ok:gemini-flash"
    assert skills == {"Languages:": ["Rust", "Go"]}
    assert len(calls) == 1  # accepted on the first provider, no retry


def test_llm_rewrite_skills_all_providers_return_only_invented_items(catalog, monkeypatch) -> None:
    # Non-strong-aggressive: the ownership union still decides acceptance, so a
    # fully invented response is rejected by every provider and the call fails.
    catalog.render_config = replace(catalog.render_config, strong_aggressive=False)
    monkeypatch.setattr(
        listing, "_generate_with_gemini",
        lambda *a, **k: json.dumps({"skills": {"Languages:": ["Rust"]}}),
    )
    status, skills = listing._llm_rewrite_skills(
        _settings(), catalog, "Backend Engineer", "Build APIs.", "", [], None
    )
    assert status.startswith("failed:")
    assert "no skill items survived validation" in status
    assert skills is None


def test_llm_rewrite_skills_caches_validated_response_and_skips_next_call(
    catalog, monkeypatch, tmp_path
) -> None:
    cache_path = tmp_path / "skills_cache.json"
    call_count = 0

    def fake_gemini(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return json.dumps({"skills": {"Languages:": ["Python", "SQL"]}})

    monkeypatch.setattr(listing, "_generate_with_gemini", fake_gemini)
    status1, skills1 = listing._llm_rewrite_skills(
        _settings(), catalog, "Backend Engineer", "Build APIs.", "", [], None, cache_path=cache_path
    )
    assert status1 == "ok:gemini-flash"
    assert skills1 == {"Languages:": ["Python", "SQL"]}
    assert call_count == 1
    assert cache_path.exists()

    def unreachable_provider(*args, **kwargs):
        raise AssertionError("provider must not be called on a cache hit")

    monkeypatch.setattr(listing, "_generate_with_gemini", unreachable_provider)
    status2, skills2 = listing._llm_rewrite_skills(
        _settings(), catalog, "Backend Engineer", "Build APIs.", "", [], None, cache_path=cache_path
    )
    assert status2 == "ok:cache"
    assert skills2 == {"Languages:": ["Python", "SQL"]}
    assert call_count == 1  # unchanged — the second call never reached the provider


def test_llm_rewrite_skills_ignores_cache_entry_with_no_valid_items(catalog, tmp_path, monkeypatch) -> None:
    """A cache poisoned by an old fully-invented response must not block a
    fresh, valid attempt — cache trust uses the same bar as live acceptance."""
    from services.resumes.listing import _store_cached_selection

    # Non-strong-aggressive: cache trust uses the ownership bar, same as live.
    catalog.render_config = replace(catalog.render_config, strong_aggressive=False)
    cache_path = tmp_path / "skills_cache.json"
    _store_cached_selection(cache_path, json.dumps({"skills": {"Languages:": ["Rust"]}}), "gemini")

    monkeypatch.setattr(
        listing, "_generate_with_gemini",
        lambda *a, **k: json.dumps({"skills": {"Languages:": ["Python"]}}),
    )
    status, skills = listing._llm_rewrite_skills(
        _settings(), catalog, "Backend Engineer", "Build APIs.", "", [], None, cache_path=cache_path
    )
    assert status == "ok:gemini-flash"
    assert skills == {"Languages:": ["Python"]}


# ── normalize_skills_rewrite_payload ─────────────────────────────────────────


def test_normalize_payload_filters_invented_dedupes_and_keys_by_label() -> None:
    result = normalize_skills_rewrite_payload(
        {"Languages:": ["Python", "python", "Rust"], "Bogus": "not-a-list", 5: ["x"]},
        ("python",),
        {"python": "Python"},
    )
    assert result == {"languages": ["Python"]}


def test_normalize_payload_non_dict_returns_empty() -> None:
    assert normalize_skills_rewrite_payload(["not", "a", "dict"], ("python",), {}) == {}
    assert normalize_skills_rewrite_payload(None, ("python",), {}) == {}


# ── _reconcile_skills_and_bullets: unbolded skills get pruned, not just ─────
# missing-bolded-terms added. A skill with nothing bolded for it anywhere in
# the actual bullets is an unsupported claim, so it's dropped.


def test_reconcile_drops_skill_item_never_bolded_in_any_bullet() -> None:
    rendered = (
        "\\begin{itemize}\n"
        "  \\item Built \\textbf{Salesforce} flows in \\textbf{Python}.\n"
        "\\end{itemize}"
    )
    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Tools:} Salesforce, Git, Docker \\\\\n"  # Git/Docker never bolded
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree"
    )
    out = _reconcile_skills_and_bullets(tail, rendered, ())
    tools = next(l for l in out.splitlines() if "Tools:" in l)
    assert "Salesforce" in tools
    assert "Git" not in tools
    assert "Docker" not in tools


def test_reconcile_falls_back_to_pre_prune_content_when_a_line_would_go_empty() -> None:
    """Floor (2026-08-20, found live: a real trial pruned a whole Skills
    section to 0 items when the skills-rewrite LLM and the bullet-bolding
    pass picked different terms on the same run). Losing an entire category
    because THIS bullet rewrite happened not to bold anything from it is a
    coverage gap, not a fabrication -- so the line falls back to its
    already-relevance-ranked pre-reconciliation content instead of vanishing."""
    rendered = (
        "\\begin{itemize}\n"
        "  \\item Built \\textbf{Salesforce} flows.\n"
        "\\end{itemize}"
    )
    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Tools:} Salesforce \\\\\n"
        "\\textbf{Languages:} C++, Rust \\\\\n"  # neither bolded anywhere
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree"
    )
    out = _reconcile_skills_and_bullets(tail, rendered, ())
    langs = next(l for l in out.splitlines() if "Languages:" in l)
    assert "C++" in langs and "Rust" in langs  # kept via the floor, not dropped
    assert "Salesforce" in out
    assert "School" in out  # unrelated sections untouched


def test_reconcile_line_still_drops_when_pre_prune_content_was_itself_empty() -> None:
    """The floor only rescues a line whose PRE-reconciliation content is
    non-empty. A label whose item list parses to nothing at all (stray
    commas, no real items) has nothing to fall back to, so it's still
    dropped rather than rendered as a bare label."""
    rendered = "\\begin{itemize}\n  \\item Built \\textbf{Salesforce} flows.\n\\end{itemize}"
    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Tools:} Salesforce \\\\\n"
        "\\textbf{Languages:} , , \\\\\n"  # parses to zero real items
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree"
    )
    out = _reconcile_skills_and_bullets(tail, rendered, ())
    assert "Languages:" not in out
    assert "Salesforce" in out
    assert "School" in out


def test_reconcile_still_adds_missing_bolded_terms_alongside_pruning() -> None:
    """The add direction (missing bolded term -> inserted into its category
    line) must keep working the same pass the new prune direction runs in."""
    rendered = (
        "\\begin{itemize}\n"
        "  \\item Used \\textbf{Kubernetes} and \\textbf{Docker} for deploys.\n"
        "\\end{itemize}"
    )
    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Platforms:} AWS, Kubernetes \\\\\n"  # AWS unbolded, Docker missing
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree"
    )
    out = _reconcile_skills_and_bullets(tail, rendered, ())
    platforms = next(l for l in out.splitlines() if "Platforms:" in l)
    assert "Kubernetes" in platforms
    assert "Docker" in platforms  # added: bolded but was missing
    assert "AWS" not in platforms  # pruned: present but never bolded


def test_reconcile_add_direction_rejects_bolded_term_outside_owned_or_jd_tools() -> None:
    """Regression (2026-08-20, found live): a bullet can bold a phrase that
    only LOOKS tool-shaped (title-case words passing the style-only
    _is_boldworthy heuristic, e.g. an open-ended model 'keyword' like
    "Agentic Harness") without it ever being a real owned skill or a
    curated jd_inject_tools term. The ADD direction must not inject that
    into Skills just because it happens to be bolded -- only PRUNE is
    allowed to be unconditional."""
    rendered = (
        "\\begin{itemize}\n"
        "  \\item Built an \\textbf{Agentic Harness} for internal tooling using "
        "\\textbf{Kubernetes}.\n"
        "\\end{itemize}"
    )
    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Tools:} Kubernetes \\\\\n"
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree"
    )
    out = _reconcile_skills_and_bullets(
        tail, rendered, jd_tools=("Kubernetes",), skill_anchors=("git",)
    )
    tools = next(l for l in out.splitlines() if "Tools:" in l)
    assert "Kubernetes" in tools  # sanctioned jd_tool: kept
    assert "Agentic Harness" not in tools  # neither owned nor jd_tools: rejected


def test_reconcile_add_direction_allows_bolded_term_that_is_an_owned_anchor() -> None:
    rendered = (
        "\\begin{itemize}\n"
        "  \\item Used \\textbf{Git} for version control.\n"
        "\\end{itemize}"
    )
    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Tools:} Docker \\\\\n"  # Docker unbolded -> pruned below
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree"
    )
    out = _reconcile_skills_and_bullets(tail, rendered, jd_tools=(), skill_anchors=("git",))
    tools = next(l for l in out.splitlines() if "Tools:" in l)
    assert "Git" in tools
    assert "Docker" not in tools


# ── Fuzzy phrasing-variant matching (2026-08-20): a bullet bolding "React.js"
# must be recognized as backing a "React" skills-line item, not treated as an
# unrelated term -- otherwise a real, truthful skill silently falls out of
# Skills just because the bullet and the skills line phrased it differently.


def test_reconcile_prune_recognizes_dotted_bullet_form_as_backing_bare_skill_item() -> None:
    rendered = (
        "\\begin{itemize}\n"
        "  \\item Built services with \\textbf{React.js} and \\textbf{Node.js}.\n"
        "\\end{itemize}"
    )
    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Frameworks:} React, Vue \\\\\n"  # Vue never bolded anywhere
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree"
    )
    out = _reconcile_skills_and_bullets(tail, rendered, jd_tools=(), skill_anchors=("react",))
    frameworks = next(l for l in out.splitlines() if "Frameworks:" in l)
    assert "React" in frameworks  # backed by the bolded "React.js" via fuzzy match
    assert "Vue" not in frameworks  # never bolded in any form, correctly pruned


def test_reconcile_add_direction_recognizes_bare_owned_anchor_as_matching_dotted_bold() -> None:
    """The ownership gate on ADD must accept a bolded 'Node.js' when the
    owned anchor is the bare 'node' -- exact-set membership would reject it
    even though it's clearly the same technology."""
    rendered = "\\begin{itemize}\n  \\item Deployed with \\textbf{Node.js}.\n\\end{itemize}"
    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Languages:} Python \\\\\n"
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree"
    )
    out = _reconcile_skills_and_bullets(tail, rendered, jd_tools=(), skill_anchors=("python", "node"))
    langs = next(l for l in out.splitlines() if "Languages:" in l)
    assert "Node.js" in langs


def test_reconcile_add_direction_skips_dotted_form_already_represented_by_bare_item() -> None:
    """A bullet bolding 'React.js' must not get ADDED as a near-duplicate
    when the skills line already carries the bare 'React' for the same
    tech -- the fuzzy 'already present' check should recognize it."""
    rendered = "\\begin{itemize}\n  \\item Built UI in \\textbf{React.js}.\n\\end{itemize}"
    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Frameworks:} React \\\\\n"
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree"
    )
    out = _reconcile_skills_and_bullets(tail, rendered, jd_tools=(), skill_anchors=("react",))
    frameworks = next(l for l in out.splitlines() if "Frameworks:" in l)
    items = [i.strip() for i in frameworks.split(":}")[1].rstrip(" \\").split(",")]
    assert items == ["React"]  # not ["React", "React.js"] -- no near-duplicate


def test_reconcile_no_bolds_anywhere_leaves_tail_unchanged() -> None:
    """Degenerate case: zero bolded terms in the whole page. Pruning
    everything down to nothing would delete the Skills section outright, so
    this stays a fail-open no-op rather than emptying the resume."""
    rendered = "\\begin{itemize}\n  \\item Built REST APIs with no bolding at all.\n\\end{itemize}"
    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Tools:} Git, Docker \\\\\n"
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree"
    )
    assert _reconcile_skills_and_bullets(tail, rendered, ()) == tail


def test_render_strong_aggressive_ranks_bolded_skills_first_end_to_end(catalog) -> None:
    """Full pipeline: what the bullets bold LEADS the line; unbolded owned
    skills follow rather than being deleted.

    This assertion was inverted on 2026-08-28. It previously required the
    prune to delete every skill no bullet bolded, which is stricter than a
    resume's Skills section actually is — it is a summary of competencies, not
    an index of the bullets. In strong-aggressive that strictness produced
    real one-item lines ("Tools: Git") on live builds, from a profile owning
    nine tools. The floor below now keeps the line healthy; the prune keeps
    deciding the ORDER, which is the part a recruiter and an ATS read first.
    Normal and --aggressive still get the strict prune (they pass no floor).
    """
    entry_id = _entry_id(catalog)
    selection = StructuredSelection(
        ranking=[entry_id],
        bullets={entry_id: ["Built \\textbf{Python} services for internal tooling."]},
    )
    latex, _ = render_structured_resume(
        catalog,
        selection,
        llm_rewritten_skills={"Languages:": ["Python", "SQL"]},
    )
    langs = next(l for l in latex.splitlines() if "Languages:" in l)
    items = [i.strip() for i in langs.split("}", 1)[1].split(",") if i.strip()]
    # The bolded skill leads.
    assert items[0].startswith("Python")
    # The line is not gutted down to that one backed item.
    assert len(items) >= 4
    # Everything on it is still a skill the profile owns.
    owned = {"JavaScript", "Java", "Python", "C++", "C", "SQL", "HTML", "CSS"}
    assert all(item.replace("\\\\", "").strip() in owned for item in items)


def test_reconcile_floor_never_leaves_a_one_item_line(catalog) -> None:
    """The defect the floor exists for: a bullet set that bolds almost nothing
    used to leave whole categories at a single item."""
    entry_id = _entry_id(catalog)
    selection = StructuredSelection(
        ranking=[entry_id],
        # Bolds nothing that appears on the Tools line at all.
        bullets={entry_id: ["Wrote \\textbf{Python} tooling for internal use."]},
    )
    latex, _ = render_structured_resume(catalog, selection)
    tools = next(l for l in latex.splitlines() if "Tools:" in l)
    items = [i.strip() for i in tools.split("}", 1)[1].split(",") if i.strip()]
    assert len(items) >= 3, f"Tools line collapsed: {tools}"


# ── Precision + recall of the reconcile matcher (2026-08-24). The prune and
# the add-dedup both used _matches_skill_anchor, whose bare substring test
# ("java" in "javascript") KEPT items no bullet actually backed and BLOCKED
# additions that were genuinely new. Both directions now go through
# _skill_item_backed, which only accepts containment at token boundaries.


def test_reconcile_does_not_keep_java_on_the_strength_of_a_javascript_bold() -> None:
    rendered = (
        "\\begin{itemize}\n"
        "  \\item Shipped a \\textbf{JavaScript} dashboard.\n"
        "\\end{itemize}"
    )
    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Languages:} JavaScript, Java \\\\\n"
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree"
    )
    out = _reconcile_skills_and_bullets(tail, rendered, ())
    langs = next(l for l in out.splitlines() if "Languages:" in l)
    assert "JavaScript" in langs
    # "Java" is a different language; the JavaScript bold never backed it.
    assert not re.search(r"(?<![a-zA-Z])Java(?![a-zA-Z])", langs)


def test_reconcile_adds_javascript_when_the_line_only_lists_java() -> None:
    rendered = (
        "\\begin{itemize}\n"
        "  \\item Shipped a \\textbf{JavaScript} dashboard.\n"
        "\\end{itemize}"
    )
    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Languages:} Java \\\\\n"
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree"
    )
    out = _reconcile_skills_and_bullets(
        tail,
        rendered,
        jd_tools=(),
        skill_anchors=("java", "javascript"),
        keywords=["java"],
    )
    langs = next(l for l in out.splitlines() if "Languages:" in l)
    # A real, separate skill: added rather than deduped away against "Java".
    assert "JavaScript" in langs
    assert re.search(r"(?<![a-zA-Z])Java(?![a-zA-Z])", langs)


def test_reconcile_still_treats_react_js_bold_as_backing_a_react_item() -> None:
    """The boundary rule must not break the intended phrasing-variant union:
    "." is a boundary, so React and React.js remain the same skill."""
    rendered = (
        "\\begin{itemize}\n"
        "  \\item Built the client in \\textbf{React.js}.\n"
        "\\end{itemize}"
    )
    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Frameworks:} React \\\\\n"
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree"
    )
    out = _reconcile_skills_and_bullets(tail, rendered, ())
    fw = next(l for l in out.splitlines() if "Frameworks:" in l)
    assert "React" in fw
    # Recognized as the same skill, not appended as a second item.
    assert "React.js" not in fw


def test_reconcile_keeps_a_listing_keyword_skill_no_bullet_bolded() -> None:
    """Recall: _rewrite_skills_for_jd promotes the items the listing asks for
    to the front of the line; the prune used to delete them right back out
    whenever the bullet-bolding pass picked different terms."""
    rendered = (
        "\\begin{itemize}\n"
        "  \\item Built \\textbf{Salesforce} flows.\n"
        "\\end{itemize}"
    )
    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Tools:} Salesforce, Docker, Figma \\\\\n"
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree"
    )
    out = _reconcile_skills_and_bullets(
        tail,
        rendered,
        jd_tools=("Docker",),
        skill_anchors=(),
        keywords=["figma"],
    )
    tools = next(l for l in out.splitlines() if "Tools:" in l)
    assert "Salesforce" in tools   # backed by a bullet bold
    assert "Docker" in tools       # JD tool: exempt from the prune
    assert "Figma" in tools        # listing keyword: exempt from the prune


def test_reconcile_prune_still_drops_an_item_with_no_listing_relevance() -> None:
    """The exemption is scoped to listing terms — an unbacked item the
    listing never asked for is still dropped."""
    rendered = (
        "\\begin{itemize}\n"
        "  \\item Built \\textbf{Salesforce} flows.\n"
        "\\end{itemize}"
    )
    tail = (
        "\\section*{Skills}\n"
        "\\textbf{Tools:} Salesforce, Blender \\\\\n"
        "\\section*{Education}\n"
        "\\textbf{School} -- Degree"
    )
    out = _reconcile_skills_and_bullets(
        tail,
        rendered,
        jd_tools=("Docker",),
        skill_anchors=(),
        keywords=["docker"],
    )
    tools = next(l for l in out.splitlines() if "Tools:" in l)
    assert "Blender" not in tools
