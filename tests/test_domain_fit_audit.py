"""Domain-fit-audit tests: the general-purpose LLM-judge companion to the
deterministic specialist_categories gate. It needs no per-domain term list
(unlike specialist_categories, which is hand-curated per category), so it
covers any narrow specialist field a profile owner never anticipated
(healthcare, construction, culinary, legal, ...)."""
from __future__ import annotations

import json

import pytest

from services.resumes import listing
from services.resumes.configkey import GeminiSettings
from services.resumes.structured import (
    RenderConfig,
    StructuredSelection,
    apply_domain_fit_verdicts,
    build_domain_fit_audit_items,
    domain_fit_audit_prompt,
    parse_template_catalog,
    render_structured_resume,
)

ELECTRICAL_ENTRY = (
    "% [electrical]\n"
    "\\textbf{Electrical Team Member,} {MARS} -- Toronto, ON \\hfill 2023 -- Present \\\\\n"
    "\\begin{itemize}\n"
    "  \\item Designed and fabricated PCB circuits for STM32-based avionics systems.\n"
    "  \\item Performed board bring-up and debugging with oscilloscopes.\n"
    "\\end{itemize}"
)
SOFTWARE_ENTRY = (
    "% [software]\n"
    "\\textbf{Software Intern,} {Acme} -- Remote \\hfill 2024 -- 2025 \\\\\n"
    "\\begin{itemize}\n"
    "  \\item Built REST APIs in Python and Flask for internal tooling.\n"
    "  \\item Wrote automated pytest suites for CI.\n"
    "\\end{itemize}"
)


def _mini_template(*entry_blocks: str) -> str:
    return (
        "\\documentclass{article}\n\\begin{document}\n"
        "\\centerline{Name}\n\n"
        "\\section*{Experience}\n\n" + "\n\n".join(entry_blocks) + "\n\n"
        "\\section*{Skills}\nstuff\n\n"
        "\\end{document}\n"
    )


@pytest.fixture()
def catalog():
    parsed = parse_template_catalog(_mini_template(ELECTRICAL_ENTRY, SOFTWARE_ENTRY))
    assert parsed is not None
    return parsed


def _settings() -> GeminiSettings:
    return GeminiSettings(api_key="test-key", model="gemini-test")


def _entry_id(catalog, category: str) -> str:
    for entry in catalog.entries:
        if category in entry.categories:
            return entry.entry_id
    raise AssertionError(f"no entry tagged {category!r}")


# ── build_domain_fit_audit_items ─────────────────────────────────────────────


def test_build_items_one_per_entry_uses_canonical_content(catalog) -> None:
    items, id_map = build_domain_fit_audit_items(catalog)
    assert len(items) == 2
    assert len(id_map) == 2
    electrical_id = _entry_id(catalog, "electrical")
    electrical_item = next(i for i in items if id_map[i["id"]] == electrical_id)
    assert "PCB" in electrical_item["summary"]
    assert "oscilloscopes" in electrical_item["summary"]


def test_prompt_carries_job_text_and_items_and_json_contract(catalog) -> None:
    items, _ = build_domain_fit_audit_items(catalog)
    prompt = domain_fit_audit_prompt("Registered Nurse", "ICU patient care", items)
    assert "Registered Nurse" in prompt
    assert "ICU patient care" in prompt
    assert '"verdicts"' in prompt
    assert "mismatch" in prompt
    assert "PCB" in prompt  # entry summaries are embedded


def test_prompt_distinguishes_employer_industry_from_job_duties(catalog) -> None:
    """Regression guard for the real gap found in the bake-off (2026-07-23):
    the audit stayed lenient on a real AI-intern listing at a construction
    company because it read the employer's industry as license, not the
    job's own duties. Confirms the prompt still carries the fix."""
    items, _ = build_domain_fit_audit_items(catalog)
    prompt = domain_fit_audit_prompt("AI Intern", "build internal AI tools", items)
    assert "employer's stated industry" in prompt
    assert "what the LISTING asks the person to actually DO" in prompt


def test_prompt_requires_matching_subdiscipline_not_just_broad_category(catalog) -> None:
    """Regression guard for the Orano mine-engineer bake-off case: sharing an
    umbrella label ("engineering", "technical") isn't enough — the specific
    sub-discipline must actually correspond."""
    items, _ = build_domain_fit_audit_items(catalog)
    prompt = domain_fit_audit_prompt("Mine Engineer", "mine equipment troubleshooting", items)
    assert "sub-discipline must actually correspond" in prompt
    assert "umbrella term" in prompt


# ── apply_domain_fit_verdicts ────────────────────────────────────────────────


def test_apply_verdicts_gates_flagged_entry_and_reports_reason(catalog) -> None:
    _, id_map = build_domain_fit_audit_items(catalog)
    electrical_id = _entry_id(catalog, "electrical")
    electrical_numeric_id = next(k for k, v in id_map.items() if v == electrical_id)
    gated, reasons = apply_domain_fit_verdicts(
        id_map,
        [{"id": electrical_numeric_id, "mismatch": True, "reason": "hardware entry on a nursing listing"}],
    )
    assert gated == [electrical_id]
    assert reasons == [f"{electrical_id}: hardware entry on a nursing listing"]


def test_apply_verdicts_ignores_malformed_input(catalog) -> None:
    _, id_map = build_domain_fit_audit_items(catalog)
    malformed = [
        "not-a-dict",
        {"mismatch": True},  # missing id
        {"id": "abc", "mismatch": True},  # unparseable id
        {"id": 99, "mismatch": True},  # unknown id
        {"id": 0, "mismatch": False},  # explicit pass
    ]
    gated, reasons = apply_domain_fit_verdicts(id_map, malformed)
    assert gated == [] and reasons == []
    gated, reasons = apply_domain_fit_verdicts(id_map, "garbage")
    assert gated == [] and reasons == []


def test_apply_verdicts_dedupes_and_reports_without_reason(catalog) -> None:
    _, id_map = build_domain_fit_audit_items(catalog)
    electrical_id = _entry_id(catalog, "electrical")
    electrical_numeric_id = next(k for k, v in id_map.items() if v == electrical_id)
    gated, reasons = apply_domain_fit_verdicts(
        id_map,
        [
            {"id": electrical_numeric_id, "mismatch": True},
            {"id": electrical_numeric_id, "mismatch": True, "reason": "dup"},
        ],
    )
    assert gated == [electrical_id]
    assert reasons == [electrical_id]


# ── render: llm_domain_gated merges into the same capacity-rollback path ────


def test_render_llm_domain_gated_hides_entry(catalog) -> None:
    from dataclasses import replace

    # Low min_visible_bullets so the 2-entry fixture doesn't trip the
    # capacity-rollback safety net (tested separately below) — this test is
    # about the plain gate-and-hide path.
    catalog.render_config = replace(catalog.render_config, min_visible_bullets=1)
    electrical_id = _entry_id(catalog, "electrical")
    selection = StructuredSelection(ranking=[electrical_id])
    latex, report = render_structured_resume(
        catalog, selection, llm_domain_gated=frozenset({electrical_id})
    )
    assert electrical_id not in report.visible_entries
    assert electrical_id in report.domain_gated_entries
    assert "MARS" not in latex


def test_render_llm_domain_gated_rolled_back_when_page_cannot_be_filled(catalog) -> None:
    electrical_id = _entry_id(catalog, "electrical")
    software_id = _entry_id(catalog, "software")
    selection = StructuredSelection(ranking=[electrical_id], exclusions=[software_id])
    _, report = render_structured_resume(
        catalog,
        selection,
        llm_domain_gated=frozenset({electrical_id}),
    )
    # Only two entries exist total; excluding both would under-fill the page,
    # so the LLM-domain-gated entry must be rolled back in just like any
    # other exclusion source.
    assert electrical_id in report.visible_entries
    assert electrical_id in report.domain_gated_entries
    assert electrical_id in report.ignored_exclusions


def test_render_config_domain_fit_gate_still_disabled_in_aggressive_mode(catalog) -> None:
    """llm_domain_gated is a caller-supplied set — render_structured_resume
    itself has no aggressive-awareness. The exemption lives at the call site
    (listing.py gates the audit call on _effective_render_config), so a
    directly-passed set still applies even under aggressive here; this test
    documents that render-level contract rather than re-testing the gating
    decision (covered by the specialist-gate aggressive test)."""
    from dataclasses import replace

    catalog.render_config = replace(catalog.render_config, aggressive=True, min_visible_bullets=1)
    electrical_id = _entry_id(catalog, "electrical")
    selection = StructuredSelection(ranking=[electrical_id])
    _, report = render_structured_resume(
        catalog, selection, llm_domain_gated=frozenset({electrical_id})
    )
    assert electrical_id not in report.visible_entries


# ── RenderConfig flag reuse ──────────────────────────────────────────────────


def test_domain_fit_audit_reuses_grounding_audit_flag(tmp_path) -> None:
    path = tmp_path / "structured_config.json"
    path.write_text(json.dumps({"grounding_audit": False}), encoding="utf-8")
    assert RenderConfig.from_file(path).grounding_audit is False


# ── _audit_domain_fit (provider loop) ────────────────────────────────────────


def test_audit_gates_entry_via_fake_provider(catalog, monkeypatch) -> None:
    electrical_id = _entry_id(catalog, "electrical")
    seen_prompts: list[str] = []

    def fake_gemini(settings, prompt, cache_name, client_factory, model_override=None, json_response=False):
        seen_prompts.append(prompt)
        items, id_map = build_domain_fit_audit_items(catalog)
        numeric_id = next(k for k, v in id_map.items() if v == electrical_id)
        return json.dumps(
            {"verdicts": [{"id": numeric_id, "mismatch": True, "reason": "hardware on a nursing listing"}]}
        )

    monkeypatch.setattr(listing, "_generate_with_gemini", fake_gemini)
    status, gated, reasons = listing._audit_domain_fit(
        _settings(), catalog, "Registered Nurse", "ICU patient care, charting, medication administration.", None
    )
    assert status == "ok:gemini-flash"
    assert gated == frozenset({electrical_id})
    assert reasons == [f"{electrical_id}: hardware on a nursing listing"]
    assert len(seen_prompts) == 1
    assert "Registered Nurse" in seen_prompts[0]


def test_audit_fail_open_when_all_providers_fail(catalog, monkeypatch) -> None:
    def broken_gemini(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(listing, "_generate_with_gemini", broken_gemini)
    status, gated, reasons = listing._audit_domain_fit(
        _settings(), catalog, "Registered Nurse", "ICU patient care.", None
    )
    assert status.startswith("failed:")
    assert gated == frozenset()
    assert reasons == []


def test_audit_malformed_response_fails_open(catalog, monkeypatch) -> None:
    monkeypatch.setattr(
        listing,
        "_generate_with_gemini",
        lambda *a, **k: json.dumps({"keywords": ["alpha"]}),  # no verdicts key
    )
    status, gated, reasons = listing._audit_domain_fit(
        _settings(), catalog, "Registered Nurse", "ICU patient care.", None
    )
    assert status.startswith("failed:")
    assert "no verdicts" in status
    assert gated == frozenset()
    assert reasons == []
