"""Bootstrap a freeform resume profile into a structured-pipeline profile.

Converts an untagged ``template.tex`` + ``baseinfo.txt`` pair into the format
the structured pipeline requires (``% [category]`` entry tags,
``== SKILL ANCHORS ==``, ``structured_config.json``) using the same "LLM
proposes, Python disposes" philosophy as the pipeline itself:

  1. The LLM only *points* at existing template lines (exact copies of entry
     header lines) and proposes category slugs and skill anchors as a small
     JSON object. It never rewrites template content.
  2. Python inserts the tags, renders the anchors section in the exact
     format the parsers expect, and merges it into baseinfo.
  3. Nothing is accepted until the candidate files round-trip through the
     REAL ``load_structured_profile`` and a baseline render completes with
     zero fidelity findings.
  4. On validation failure the LLM gets one retry with the exact errors.

The generator is injected as a callable so tests exercise the whole flow
without any network access.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .configkey import GeminiSettings
from .structured import (
    CATEGORY_TAG_PATTERN,
    StructuredSelection,
    extract_json_object,
    load_structured_profile,
    render_structured_resume,
)

MAX_BOOTSTRAP_ATTEMPTS = 2

CATEGORY_SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

DEFAULT_STRUCTURED_CONFIG = {
    "rewrite_scope": "limited",
    "limited_rewrite_bullets": 5,
}


@dataclass(slots=True)
class BootstrapProposal:
    """LLM-proposed structure, parsed and shape-checked (not yet applied)."""

    entries: list[tuple[str, str]]  # (header_snippet, category_slug)
    skill_anchors: dict[str, list[str]]


@dataclass(slots=True)
class BootstrapResult:
    status: str  # "ok" | "already_structured" | "error"
    messages: list[str] = field(default_factory=list)
    template_text: str | None = None
    baseinfo_text: str | None = None
    structured_config: dict[str, Any] | None = None
    provider: str | None = None
    attempts: int = 0
    categories: list[str] = field(default_factory=list)


def build_bootstrap_prompt(template_text: str, baseinfo_text: str, feedback: list[str] | None = None) -> str:
    feedback_section = ""
    if feedback:
        feedback_section = (
            "\n<previous_attempt_errors>\n"
            "Your previous proposal failed validation with these errors. Fix ALL of\n"
            "them; everything else about the task is unchanged:\n"
            + "\n".join(f"- {item}" for item in feedback)
            + "\n</previous_attempt_errors>\n"
        )

    return (
        "<task>\n"
        "You are converting a resume profile to a structured format. Analyse the\n"
        "LaTeX resume in <template> (and the background facts in <baseinfo>, when\n"
        "present) and propose:\n"
        "1. A category slug for EVERY entry in the template body. An entry is a\n"
        "   heading line introducing one job, project, volunteer role, or similar,\n"
        "   followed by its bullet items. Do NOT include section headers, the\n"
        "   candidate's name/contact block, skills lines, or education lines.\n"
        "2. The candidate's skill anchors: named tools, systems, software,\n"
        "   equipment, methods, and certifications that appear in the resume.\n"
        "This works for ANY field - engineering, healthcare, trades, retail,\n"
        "research, administration - use vocabulary from the resume itself.\n"
        "</task>\n"
        f"{feedback_section}"
        "\n<template>\n"
        f"{template_text}\n"
        "</template>\n"
        "\n<baseinfo>\n"
        f"{baseinfo_text or '(no baseinfo content)'}\n"
        "</baseinfo>\n"
        "\n<output_format>\n"
        "Return ONLY a JSON object (no prose, no markdown fences) shaped exactly:\n"
        "{\n"
        '  "entries": [\n'
        '    {"header_snippet": "<the entry\'s FIRST line copied EXACTLY,\n'
        '      character-for-character, from <template> - it is matched literally>",\n'
        '     "category": "<short lowercase slug, e.g. clinical, machining, sales>"},\n'
        "    ...\n"
        "  ],\n"
        '  "skill_anchors": {"<Label>": ["item", ...], ...}\n'
        "}\n"
        "Rules:\n"
        "1. header_snippet must be an EXACT copy of one existing template line -\n"
        "   no paraphrasing, no added/removed characters. One per entry.\n"
        "2. Category slugs: lowercase letters/digits/underscores only. Reuse the\n"
        "   same slug for entries of the same kind.\n"
        "3. skill_anchors must only contain items that appear in the resume -\n"
        "   never add skills the candidate does not claim.\n"
        "</output_format>"
    )


def parse_bootstrap_response(text: str) -> tuple[BootstrapProposal | None, list[str]]:
    """Parse + shape-check an LLM response. Returns (proposal, errors)."""
    payload = extract_json_object(text)
    if not isinstance(payload, dict):
        return None, ["response did not contain a parsable JSON object"]

    errors: list[str] = []

    entries: list[tuple[str, str]] = []
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        errors.append("'entries' must be a non-empty list")
    else:
        for index, item in enumerate(raw_entries):
            if not isinstance(item, dict):
                errors.append(f"entries[{index}] is not an object")
                continue
            snippet = str(item.get("header_snippet") or "").strip()
            category = str(item.get("category") or "").strip().lower()
            if not snippet:
                errors.append(f"entries[{index}] is missing header_snippet")
                continue
            if not CATEGORY_SLUG_PATTERN.match(category):
                errors.append(
                    f"entries[{index}] category {category!r} is not a valid slug "
                    "(lowercase letters/digits/underscores, starting with a letter)"
                )
                continue
            entries.append((snippet, category))

    skill_anchors: dict[str, list[str]] = {}
    raw_anchors = payload.get("skill_anchors")
    if isinstance(raw_anchors, dict):
        for label, values in raw_anchors.items():
            if isinstance(values, list):
                items = [str(v).strip() for v in values if str(v).strip()]
                if items:
                    skill_anchors[str(label).strip()] = items

    if errors:
        return None, errors
    return (
        BootstrapProposal(
            entries=entries,
            skill_anchors=skill_anchors,
        ),
        [],
    )


def apply_tags_to_template(template_text: str, entries: list[tuple[str, str]]) -> tuple[str | None, list[str]]:
    """Insert ``% [category]`` above each entry's header line.

    Matching is whitespace-normalized but otherwise literal, and each snippet
    must match exactly one untagged template line — anything else is an error
    fed back to the LLM. Untouched lines stay byte-identical.
    """
    lines = template_text.splitlines()
    normalized = [line.strip() for line in lines]
    insertions: dict[int, str] = {}
    errors: list[str] = []

    for snippet, category in entries:
        target = snippet.strip()
        matches = [i for i, line in enumerate(normalized) if line == target]
        if not matches:
            errors.append(f"header_snippet not found in template: {snippet!r}")
            continue
        if len(matches) > 1:
            errors.append(
                f"header_snippet matches {len(matches)} template lines (must be unique): {snippet!r}"
            )
            continue
        line_index = matches[0]
        if line_index in insertions:
            errors.append(f"two entries point at the same template line: {snippet!r}")
            continue
        if line_index > 0 and CATEGORY_TAG_PATTERN.match(lines[line_index - 1].strip()):
            errors.append(f"template line is already tagged: {snippet!r}")
            continue
        insertions[line_index] = category

    if errors:
        return None, errors

    out: list[str] = []
    for i, line in enumerate(lines):
        if i in insertions:
            out.append(f"% [{insertions[i]}]")
        out.append(line)
    tagged = "\n".join(out)
    if template_text.endswith("\n") and not tagged.endswith("\n"):
        tagged += "\n"
    return tagged, []


def render_skill_anchors(skill_anchors: dict[str, list[str]]) -> str:
    lines = ["== SKILL ANCHORS =="]
    for label, items in skill_anchors.items():
        lines.append(f"{label}: {', '.join(items)}")
    return "\n".join(lines) + "\n"


def _replace_or_append_section(baseinfo_text: str, header: str, section_text: str) -> str:
    """Replace an existing ``== header ==`` block, or append the section."""
    pattern = re.compile(
        rf"==\s*{re.escape(header)}\s*==.*?(?=\n==\s|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    if pattern.search(baseinfo_text):
        return pattern.sub(section_text.rstrip() + "\n\n", baseinfo_text).rstrip() + "\n"
    joined = baseinfo_text.rstrip() + "\n\n" + section_text.rstrip() + "\n"
    return joined


def merge_baseinfo_sections(baseinfo_text: str, proposal: BootstrapProposal) -> str:
    merged = baseinfo_text or ""
    if proposal.skill_anchors:
        merged = _replace_or_append_section(merged, "SKILL ANCHORS", render_skill_anchors(proposal.skill_anchors))
    return merged


def build_structured_config(proposal: BootstrapProposal, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(existing or {})
    for key, value in DEFAULT_STRUCTURED_CONFIG.items():
        config.setdefault(key, value)
    return config


def validate_bootstrap_output(
    template_text: str,
    baseinfo_text: str,
    structured_config: dict[str, Any],
    proposal: BootstrapProposal,
    work_dir: Path,
) -> list[str]:
    """Round-trip the candidate files through the real structured loader and
    run a baseline render. Returns a list of errors (empty = valid)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    template_path = work_dir / "template.tex"
    baseinfo_path = work_dir / "baseinfo.txt"
    template_path.write_text(template_text, encoding="utf-8")
    baseinfo_path.write_text(baseinfo_text, encoding="utf-8")
    (work_dir / "structured_config.json").write_text(
        json.dumps(structured_config, indent=2), encoding="utf-8"
    )

    catalog = load_structured_profile(template_path, baseinfo_path)
    if catalog is None:
        return ["tagged template + baseinfo did not load as a structured profile"]

    errors: list[str] = []
    if len(catalog.entries) != len(proposal.entries):
        errors.append(
            f"catalog parsed {len(catalog.entries)} entries but {len(proposal.entries)} were tagged - "
            "an entry's structure was not recognised (header + itemize bullets expected)"
        )

    try:
        document, report = render_structured_resume(catalog, StructuredSelection())
    except Exception as exc:  # pragma: no cover - defensive
        return errors + [f"baseline render raised {exc!r}"]
    if report.fidelity_findings:
        errors.append(f"baseline render: fidelity findings {report.fidelity_findings}")
    if document.count("\\begin{itemize}") != document.count("\\end{itemize}"):
        errors.append("baseline render: unbalanced itemize environments")
    if report.visible_bullet_count == 0:
        errors.append("baseline render: renders zero bullets")

    return errors


def bootstrap_structured_profile(
    profile_dir: Path,
    settings: GeminiSettings | None,
    work_dir: Path,
    generator: Callable[[str], tuple[str | None, str | None]] | None = None,
    max_attempts: int = MAX_BOOTSTRAP_ATTEMPTS,
    force: bool = False,
) -> BootstrapResult:
    """Produce validated structured-profile files for `profile_dir`.

    Does NOT write into the profile directory — the caller decides what to do
    with the returned file contents (dry-run preview vs apply)."""
    template_path = profile_dir / "template.tex"
    baseinfo_path = profile_dir / "baseinfo.txt"
    if not template_path.exists():
        return BootstrapResult(status="error", messages=[f"missing {template_path}"])

    template_text = template_path.read_text(encoding="utf-8")
    baseinfo_text = baseinfo_path.read_text(encoding="utf-8") if baseinfo_path.exists() else ""

    if CATEGORY_TAG_PATTERN.search(template_text) and not force:
        return BootstrapResult(
            status="already_structured",
            messages=["template.tex already contains % [category] tags (use force=True to redo)"],
        )

    if generator is None:
        if settings is None:
            return BootstrapResult(status="error", messages=["no generator and no LLM settings provided"])
        from .listing import generate_validated_with_providers

        def generator(prompt: str) -> tuple[str | None, str | None]:
            return generate_validated_with_providers(prompt, settings, lambda text: text)

    existing_config: dict[str, Any] | None = None
    config_path = profile_dir / "structured_config.json"
    if config_path.exists():
        try:
            parsed = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                existing_config = parsed
        except (OSError, json.JSONDecodeError):
            existing_config = None

    feedback: list[str] = []
    attempts = 0
    for attempt in range(1, max(1, max_attempts) + 1):
        attempts = attempt
        prompt = build_bootstrap_prompt(template_text, baseinfo_text, feedback or None)
        response_text, provider = generator(prompt)
        if not response_text:
            feedback = ["no provider returned a response"]
            continue

        proposal, parse_errors = parse_bootstrap_response(response_text)
        if proposal is None:
            feedback = parse_errors
            continue

        tagged_template, tag_errors = apply_tags_to_template(template_text, proposal.entries)
        if tagged_template is None:
            feedback = tag_errors
            continue

        merged_baseinfo = merge_baseinfo_sections(baseinfo_text, proposal)
        structured_config = build_structured_config(proposal, existing_config)

        validation_errors = validate_bootstrap_output(
            tagged_template,
            merged_baseinfo,
            structured_config,
            proposal,
            work_dir / f"attempt{attempt}",
        )
        if validation_errors:
            feedback = validation_errors
            continue

        return BootstrapResult(
            status="ok",
            messages=[f"validated on attempt {attempt}"],
            template_text=tagged_template,
            baseinfo_text=merged_baseinfo,
            structured_config=structured_config,
            provider=provider,
            attempts=attempt,
            categories=sorted({category for _, category in proposal.entries}),
        )

    return BootstrapResult(
        status="error",
        messages=["bootstrap failed after all attempts; last errors:"] + feedback,
        attempts=attempts,
    )
