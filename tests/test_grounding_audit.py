"""Grounding-audit tests: the batched LLM-judge stage that demotes tailored
bullets claiming a domain/industry/equipment the entry's verified background
never mentions (found live: a Mechanical Designer listing produced
"ventilation simulation" / "industrial equipment control" bullets that passed
every regex validator)."""
from __future__ import annotations

import json

import pytest

from services.resumes import listing
from services.resumes.configkey import GeminiSettings
from services.resumes.structured import (
    RenderConfig,
    StructuredSelection,
    apply_grounding_verdicts,
    build_grounding_audit_items,
    grounding_audit_prompt,
    parse_template_catalog,
    render_structured_resume,
)

ENTRY_BLOCK = (
    "% [one]\n"
    "\\textbf{Widget Tool} | \\textit{C++} \\\\\n"
    "\\begin{itemize}\n"
    "  \\item Built a widget renderer in C++ improving frame times by 12\\%.\n"
    "  \\item Documented the widget pipeline for team onboarding.\n"
    "\\end{itemize}"
)


def _mini_template(entry_block: str) -> str:
    return (
        "\\documentclass{article}\n\\begin{document}\n"
        "\\centerline{Name}\n\n"
        "\\section*{Projects}\n\n" + entry_block + "\n\n"
        "\\section*{Skills}\nstuff\n\n"
        "\\end{document}\n"
    )


@pytest.fixture()
def catalog():
    parsed = parse_template_catalog(_mini_template(ENTRY_BLOCK))
    assert parsed is not None
    return parsed


def _settings() -> GeminiSettings:
    return GeminiSettings(api_key="test-key", model="gemini-test")


REWRITTEN = "Documented the widget pipeline for industrial compliance audits."


def _selection(catalog, second_bullet: str = REWRITTEN) -> StructuredSelection:
    entry = catalog.entries[0]
    return StructuredSelection(
        bullets={entry.entry_id: ["", second_bullet]},
    )


# ── build_grounding_audit_items ──────────────────────────────────────────────


def test_build_items_skips_empty_and_canonical_bullets(catalog) -> None:
    entry = catalog.entries[0]
    selection = StructuredSelection(
        bullets={
            entry.entry_id: [
                "Built a widget renderer in C++ improving frame times by 12\\%.",
                REWRITTEN,
            ]
        },
    )
    items, id_map = build_grounding_audit_items(catalog, selection)
    assert len(items) == 1
    assert id_map == {0: (entry.entry_id, 1)}
    assert items[0]["bullet"] == REWRITTEN
    # Grounding is the entry's own verified text, not the listing.
    assert "widget renderer" in items[0]["grounding"]


def test_build_items_ignores_indices_past_canonical_bullets(catalog) -> None:
    entry = catalog.entries[0]
    selection = StructuredSelection(
        bullets={entry.entry_id: ["a", "b", "c", "d"]},
    )
    items, id_map = build_grounding_audit_items(catalog, selection)
    assert len(items) == 2  # entry only has two canonical bullets
    assert set(id_map.values()) == {(entry.entry_id, 0), (entry.entry_id, 1)}


def test_build_items_unknown_entry_ids_are_skipped(catalog) -> None:
    selection = StructuredSelection(bullets={"no-such-entry": ["text"]})
    items, id_map = build_grounding_audit_items(catalog, selection)
    assert items == [] and id_map == {}


def test_prompt_carries_items_and_json_contract(catalog) -> None:
    items = [{"id": 0, "grounding": "the truth", "bullet": "the claim"}]
    prompt = grounding_audit_prompt(items)
    assert '"verdicts"' in prompt
    assert '"the claim"' in prompt
    assert "fabricated" in prompt


# ── apply_grounding_verdicts ─────────────────────────────────────────────────


def test_apply_verdicts_blanks_bullet_and_reports_reason(catalog) -> None:
    entry = catalog.entries[0]
    selection = _selection(catalog)
    _, id_map = build_grounding_audit_items(catalog, selection)
    flagged = apply_grounding_verdicts(
        selection,
        id_map,
        [{"id": 0, "fabricated": True, "phrase": "industrial compliance"}],
    )
    assert flagged == [
        f"{entry.entry_id}[1]: ungrounded domain claim 'industrial compliance'"
    ]
    assert selection.bullets[entry.entry_id][1] == ""


def test_apply_verdicts_ignores_malformed_input(catalog) -> None:
    entry = catalog.entries[0]
    selection = _selection(catalog)
    _, id_map = build_grounding_audit_items(catalog, selection)
    malformed = [
        "not-a-dict",
        {"fabricated": True},  # missing id
        {"id": "abc", "fabricated": True},  # unparseable id
        {"id": 99, "fabricated": True},  # unknown id
        {"id": 0, "fabricated": False},  # explicit pass
    ]
    assert apply_grounding_verdicts(selection, id_map, malformed) == []
    assert selection.bullets[entry.entry_id][1] == REWRITTEN
    assert apply_grounding_verdicts(selection, id_map, "garbage") == []


def test_apply_verdicts_without_phrase_still_reports(catalog) -> None:
    entry = catalog.entries[0]
    selection = _selection(catalog)
    _, id_map = build_grounding_audit_items(catalog, selection)
    flagged = apply_grounding_verdicts(
        selection, id_map, [{"id": 0, "fabricated": True}]
    )
    assert flagged == [f"{entry.entry_id}[1]: ungrounded domain claim"]


# ── render: blank slots are silent canonical fallbacks ───────────────────────


def test_render_blank_slot_renders_canonical_without_fallback_noise(catalog) -> None:
    entry = catalog.entries[0]
    selection = StructuredSelection(bullets={entry.entry_id: ["", ""]})
    latex, report = render_structured_resume(catalog, selection)
    assert "Built a widget renderer" in latex
    assert "Documented the widget pipeline for team onboarding" in latex
    assert report.canonical_fallbacks == []
    assert report.tailored_bullets_used == 0


# ── RenderConfig flag ────────────────────────────────────────────────────────


def test_render_config_reads_grounding_audit_flag(tmp_path) -> None:
    path = tmp_path / "structured_config.json"
    path.write_text(json.dumps({"grounding_audit": False}), encoding="utf-8")
    assert RenderConfig.from_file(path).grounding_audit is False
    path.write_text(json.dumps({}), encoding="utf-8")
    assert RenderConfig.from_file(path).grounding_audit is True
    path.write_text(json.dumps({"grounding_audit": "no"}), encoding="utf-8")
    assert RenderConfig.from_file(path).grounding_audit is True  # non-bool ignored


# ── _audit_selection_grounding (provider loop) ───────────────────────────────


def test_audit_strips_flagged_bullet_via_fake_provider(catalog, monkeypatch) -> None:
    entry = catalog.entries[0]
    selection = _selection(catalog)
    seen_prompts: list[str] = []

    def fake_gemini(settings, prompt, cache_name, client_factory, model_override=None, json_response=False):
        seen_prompts.append(prompt)
        return json.dumps(
            {"verdicts": [{"id": 0, "fabricated": True, "phrase": "industrial compliance"}]}
        )

    monkeypatch.setattr(listing, "_generate_with_gemini", fake_gemini)
    status, flagged = listing._audit_selection_grounding(
        _settings(), catalog, selection, None
    )
    assert status == "ok:gemini-flash"
    assert flagged == [
        f"{entry.entry_id}[1]: ungrounded domain claim 'industrial compliance'"
    ]
    assert selection.bullets[entry.entry_id][1] == ""
    assert len(seen_prompts) == 1
    assert REWRITTEN in seen_prompts[0]


def test_audit_fail_open_when_all_providers_fail(catalog, monkeypatch) -> None:
    entry = catalog.entries[0]
    selection = _selection(catalog)

    def broken_gemini(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(listing, "_generate_with_gemini", broken_gemini)
    status, flagged = listing._audit_selection_grounding(
        _settings(), catalog, selection, None
    )
    assert status.startswith("failed:")
    assert flagged == []
    # Fail-open: the rewrite survives untouched.
    assert selection.bullets[entry.entry_id][1] == REWRITTEN


def test_audit_malformed_response_fails_open(catalog, monkeypatch) -> None:
    entry = catalog.entries[0]
    selection = _selection(catalog)
    monkeypatch.setattr(
        listing,
        "_generate_with_gemini",
        lambda *a, **k: json.dumps({"keywords": ["alpha"]}),  # no verdicts key
    )
    status, flagged = listing._audit_selection_grounding(
        _settings(), catalog, selection, None
    )
    assert status.startswith("failed:")
    assert "no verdicts" in status
    assert flagged == []
    assert selection.bullets[entry.entry_id][1] == REWRITTEN


def test_audit_skipped_when_nothing_tailored(catalog) -> None:
    selection = StructuredSelection()
    status, flagged = listing._audit_selection_grounding(
        _settings(), catalog, selection, None
    )
    assert status == "skipped"
    assert flagged == []
