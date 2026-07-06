"""Tests for the freeform -> structured profile bootstrap converter.

The generator is injected, so the full flow (prompt -> proposal -> tag
insertion -> baseinfo merge -> real-parser validation -> retry loop) runs
with zero network access."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.resumes.bootstrap import (
    apply_tags_to_template,
    bootstrap_structured_profile,
    parse_bootstrap_response,
)
from services.resumes.structured import load_structured_profile

UNTAGGED_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\begin{document}

\begin{center}
\textbf{Morgan Machinist} \\
morgan@example.com | 555-0199
\end{center}

\section*{Experience}

\textbf{CNC Operator,} {Precision Parts Ltd} -- Hamilton \hfill 2024 \\
\begin{itemize}
  \item Machined tight-tolerance components on 3-axis \textbf{Haas} mills, holding 0.01mm tolerances.
  \item Reduced setup scrap by 15\% by standardising fixture offsets.
\end{itemize}

\smallskip
\textbf{Shop Assistant,} {Weld-Rite Fabrication} -- Hamilton \hfill 2023 \\
\begin{itemize}
  \item Prepared and finished welded assemblies for 20 client orders per week.
  \item Maintained tooling inventory in \textbf{Excel}, cutting reorder delays by 30\%.
\end{itemize}

\section*{Projects}

\textbf{Go-Kart Frame Build} \hfill 2022 \\
\begin{itemize}
  \item Designed and welded a tubular steel frame using \textbf{SolidWorks} drawings.
  \item Fitted brakes and steering, passing a community race-day safety inspection.
\end{itemize}

\section*{Skills}
\textbf{Machines:} Haas mill, Manual lathe, MIG welder \\
\textbf{Software:} SolidWorks, Excel \\
\textbf{Certifications:} WHMIS, Forklift

\section*{Education}
Trade College -- Machining Certificate \hfill 2022--2024

\end{document}
"""

UNTAGGED_BASEINFO = """Morgan Machinist -- machining apprentice, Hamilton.

== EXPERIENCE ==

CNC Operator, Precision Parts Ltd -- 2024
Shop Assistant, Weld-Rite Fabrication -- 2023
"""

GOOD_PROPOSAL = {
    "entries": [
        {
            "header_snippet": "\\textbf{CNC Operator,} {Precision Parts Ltd} -- Hamilton \\hfill 2024 \\\\",
            "category": "machining",
        },
        {
            "header_snippet": "\\textbf{Shop Assistant,} {Weld-Rite Fabrication} -- Hamilton \\hfill 2023 \\\\",
            "category": "fabrication",
        },
        {
            "header_snippet": "\\textbf{Go-Kart Frame Build} \\hfill 2022 \\\\",
            "category": "fabrication",
        },
    ],
    "families": [
        {
            "name": "MACHINING / CNC",
            "keywords": ["cnc", "machinist", "mill", "lathe", "tolerance"],
            "must_show": ["machining"],
            "show": ["fabrication"],
            "hide": [],
        },
        {
            "name": "FABRICATION / WELDING",
            "keywords": ["welding", "fabrication", "assembly", "fitter"],
            "must_show": ["fabrication"],
            "show": ["machining"],
            "hide": [],
        },
    ],
    "skill_anchors": {
        "Machines": ["Haas mill", "Manual lathe", "MIG welder"],
        "Software": ["SolidWorks", "Excel"],
        "Certifications": ["WHMIS", "Forklift"],
    },
    "family_skill_priority": {
        "MACHINING": ["Haas mill", "Manual lathe", "SolidWorks"],
        "FABRICATION": ["MIG welder", "SolidWorks"],
    },
}


@pytest.fixture()
def freeform_profile(tmp_path: Path) -> Path:
    profile_dir = tmp_path / "morgan"
    profile_dir.mkdir()
    (profile_dir / "template.tex").write_text(UNTAGGED_TEMPLATE, encoding="utf-8")
    (profile_dir / "baseinfo.txt").write_text(UNTAGGED_BASEINFO, encoding="utf-8")
    return profile_dir


def _generator_returning(*payloads):
    """Build a fake generator yielding each payload (dict -> JSON) in turn."""
    responses = [json.dumps(p) if isinstance(p, dict) else p for p in payloads]
    calls: list[str] = []

    def generator(prompt: str):
        calls.append(prompt)
        index = min(len(calls) - 1, len(responses) - 1)
        return responses[index], "fake-provider"

    generator.calls = calls
    return generator


def test_bootstrap_happy_path_produces_valid_structured_profile(freeform_profile, tmp_path) -> None:
    generator = _generator_returning(GOOD_PROPOSAL)
    result = bootstrap_structured_profile(freeform_profile, None, tmp_path / "work", generator=generator)

    assert result.status == "ok", result.messages
    assert result.attempts == 1
    assert result.families == ["MACHINING", "FABRICATION"]
    assert result.categories == ["fabrication", "machining"]
    assert result.structured_config["rewrite_scope"] == "limited"
    assert result.structured_config["family_skill_priority"]["MACHINING"][0] == "Haas mill"

    # The produced files must load through the REAL structured loader.
    out = tmp_path / "converted"
    out.mkdir()
    (out / "template.tex").write_text(result.template_text, encoding="utf-8")
    (out / "baseinfo.txt").write_text(result.baseinfo_text, encoding="utf-8")
    loaded = load_structured_profile(out / "template.tex", out / "baseinfo.txt")
    assert loaded is not None
    catalog, families = loaded
    assert len(catalog.entries) == 3
    assert {f.key for f in families} == {"MACHINING", "FABRICATION"}
    assert "haas mill" in catalog.skill_anchors


def test_bootstrap_preserves_untouched_template_lines(freeform_profile, tmp_path) -> None:
    generator = _generator_returning(GOOD_PROPOSAL)
    result = bootstrap_structured_profile(freeform_profile, None, tmp_path / "work", generator=generator)
    assert result.status == "ok"

    original_lines = UNTAGGED_TEMPLATE.splitlines()
    tagged_lines = result.template_text.splitlines()
    # Removing exactly the inserted tag lines must reproduce the original.
    without_tags = [line for line in tagged_lines if not line.startswith("% [")]
    assert without_tags == original_lines
    assert sum(1 for line in tagged_lines if line.startswith("% [")) == 3


def test_bootstrap_retries_with_validation_feedback(freeform_profile, tmp_path) -> None:
    bad = json.loads(json.dumps(GOOD_PROPOSAL))
    bad["entries"][0]["header_snippet"] = "\\textbf{Nonexistent Line} \\\\"
    generator = _generator_returning(bad, GOOD_PROPOSAL)

    result = bootstrap_structured_profile(freeform_profile, None, tmp_path / "work", generator=generator)

    assert result.status == "ok"
    assert result.attempts == 2
    # The retry prompt must carry the exact failure reason.
    assert "header_snippet not found" in generator.calls[1]
    assert "previous_attempt_errors" in generator.calls[1]


def test_bootstrap_fails_cleanly_when_all_attempts_invalid(freeform_profile, tmp_path) -> None:
    generator = _generator_returning("this is not json at all")
    result = bootstrap_structured_profile(freeform_profile, None, tmp_path / "work", generator=generator)
    assert result.status == "error"
    assert result.attempts == 2
    assert any("JSON" in m or "json" in m for m in result.messages)


def test_bootstrap_short_circuits_already_tagged_profile(freeform_profile, tmp_path) -> None:
    template_path = freeform_profile / "template.tex"
    template_path.write_text(
        template_path.read_text(encoding="utf-8").replace(
            "\\textbf{CNC Operator,}", "% [machining]\n\\textbf{CNC Operator,}", 1
        ),
        encoding="utf-8",
    )
    called = _generator_returning(GOOD_PROPOSAL)
    result = bootstrap_structured_profile(freeform_profile, None, tmp_path / "work", generator=called)
    assert result.status == "already_structured"
    assert called.calls == []


def test_bootstrap_replaces_existing_guide_section_idempotently(freeform_profile, tmp_path) -> None:
    baseinfo_path = freeform_profile / "baseinfo.txt"
    baseinfo_path.write_text(
        UNTAGGED_BASEINFO
        + "\n== ROLE TYPE SELECTION GUIDE ==\n\nOLD STALE roles\n  MUST SHOW: stale\n",
        encoding="utf-8",
    )
    generator = _generator_returning(GOOD_PROPOSAL)
    result = bootstrap_structured_profile(freeform_profile, None, tmp_path / "work", generator=generator)
    assert result.status == "ok"
    assert "OLD STALE" not in result.baseinfo_text
    assert result.baseinfo_text.count("== ROLE TYPE SELECTION GUIDE ==") == 1
    assert result.baseinfo_text.count("== SKILL ANCHORS ==") == 1


def test_parse_bootstrap_response_rejects_unknown_family_category() -> None:
    proposal = json.loads(json.dumps(GOOD_PROPOSAL))
    proposal["families"][0]["must_show"] = ["ghost_category"]
    parsed, errors = parse_bootstrap_response(json.dumps(proposal))
    assert parsed is None
    assert any("ghost_category" in e for e in errors)


def test_apply_tags_rejects_ambiguous_snippet() -> None:
    template = "line one\nsame line\nsame line\n"
    tagged, errors = apply_tags_to_template(template, [("same line", "cat")])
    assert tagged is None
    assert any("matches 2" in e for e in errors)
