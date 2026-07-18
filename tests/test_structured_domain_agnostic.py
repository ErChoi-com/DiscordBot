"""Proof that the structured pipeline is domain-agnostic: a synthetic nursing
profile (no software content anywhere) must parse, validate, and render end to
end. Categories, skill anchors, and skills ordering all come from profile
content — nothing in the code may assume a tech resume."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.resumes.structured import (
    StructuredSelection,
    load_structured_profile,
    render_structured_resume,
    validate_tailored_bullet,
)

NURSING_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\begin{document}

\begin{center}
\textbf{Casey Cardinal} \\
casey@example.com | 555-0100
\end{center}

\section*{Clinical Experience}

% [clinical]
\textbf{Student Nurse,} {General Hospital} -- Toronto \hfill 2025 \\
\begin{itemize}
  \item Charted patient vitals in \textbf{Epic} for a 12-bed unit, cutting handoff errors by 18\%.
  \item Administered medications under supervision with zero dosing incidents across 40 shifts.
\end{itemize}

\smallskip
% [community]
\textbf{Health Outreach Volunteer,} {Community Clinic} -- Toronto \hfill 2024 \\
\begin{itemize}
  \item Screened 300 patients for blood pressure and glucose at community events.
  \item Coordinated intake scheduling with \textbf{Excel}, reducing wait times by 25\%.
\end{itemize}

\section*{Administration}

% [admin]
\textbf{Ward Clerk,} {General Hospital} -- Toronto \hfill 2023 \\
\begin{itemize}
  \item Processed admissions paperwork and bed assignments for 30 patients daily.
  \item Maintained supply inventory logs in \textbf{Meditech}, cutting stock-outs by 12\%.
\end{itemize}

\section*{Skills}
\textbf{Clinical:} Vital signs, Medication administration, Wound care, IV setup \\
\textbf{Systems:} Excel, Epic, Meditech \\
\textbf{Certifications:} CPR, First Aid

\section*{Education}
Nursing College -- BScN \hfill 2023--2027

\end{document}
"""

NURSING_BASEINFO = """Casey Cardinal -- nursing student, Toronto.

== EXPERIENCE ==

[clinical] Student Nurse, General Hospital -- 2025
- Charted patient vitals in Epic for a 12-bed unit, cutting handoff errors by 18%.
- Administered medications under supervision with zero dosing incidents across 40 shifts.

[community] Health Outreach Volunteer, Community Clinic -- 2024
- Screened 300 patients for blood pressure and glucose at community events.
- Coordinated intake scheduling with Excel, reducing wait times by 25%.

[admin] Ward Clerk, General Hospital -- 2023
- Processed admissions paperwork and bed assignments for 30 patients daily.
- Maintained supply inventory logs in Meditech, cutting stock-outs by 12%.

== SKILL ANCHORS ==
Systems: Epic, Excel, Meditech
Certifications: CPR, First Aid
Clinical: IV setup, Wound care, Vital signs
"""

NURSING_CONFIG = {
    "min_visible_bullets": 4,
    "max_visible_bullets": 8,
    "rewrite_scope": "full",
}


@pytest.fixture(scope="module")
def nursing_catalog(tmp_path_factory):
    profile_dir = tmp_path_factory.mktemp("nursing_profile")
    (profile_dir / "template.tex").write_text(NURSING_TEMPLATE, encoding="utf-8")
    (profile_dir / "baseinfo.txt").write_text(NURSING_BASEINFO, encoding="utf-8")
    (profile_dir / "structured_config.json").write_text(json.dumps(NURSING_CONFIG), encoding="utf-8")
    catalog = load_structured_profile(profile_dir / "template.tex", profile_dir / "baseinfo.txt")
    assert catalog is not None, "nursing profile failed to load as a structured profile"
    return catalog


def test_nursing_profile_parses_entries_and_anchors(nursing_catalog) -> None:
    assert len(nursing_catalog.entries) == 3
    assert "epic" in nursing_catalog.skill_anchors
    assert {c for e in nursing_catalog.entries for c in e.categories} == {
        "clinical",
        "community",
        "admin",
    }


def test_nursing_render_is_clean_and_complete(nursing_catalog) -> None:
    latex, report = render_structured_resume(nursing_catalog, StructuredSelection())
    assert report.fidelity_findings == []
    assert "Student Nurse" in latex
    assert "Casey Cardinal" in latex
    assert "18\\%" in latex
    assert latex.count("\\begin{itemize}") == latex.count("\\end{itemize}")
    assert "\\end{document}" in latex


def test_nursing_ranking_reorders_entries(nursing_catalog) -> None:
    admin_id = next(
        e.entry_id for e in nursing_catalog.entries if "admin" in e.categories
    )
    latex, report = render_structured_resume(
        nursing_catalog, StructuredSelection(ranking=[admin_id])
    )
    assert admin_id in report.visible_entries
    assert "Ward Clerk" in latex


def test_nursing_skills_lines_order_by_listing_keywords(nursing_catalog) -> None:
    latex, _ = render_structured_resume(
        nursing_catalog, StructuredSelection(keywords=["Meditech", "Epic"])
    )
    systems = next(l for l in latex.splitlines() if l.startswith("\\textbf{Systems:}"))
    items = systems.split("}", 1)[1].strip()
    # Listing-matched systems lead (stable canonical order within the match
    # tier: Epic before Meditech); canonical-first Excel drops behind them.
    assert items.startswith("Epic, Meditech, Excel")


def test_nursing_bullet_validation_grounds_on_clinical_anchors(nursing_catalog) -> None:
    clinical = next(e for e in nursing_catalog.entries if "clinical" in e.categories)
    canonical = clinical.bullets[0]

    # Bolding a real anchor the canonical bullet lacks (Meditech) is allowed.
    ok_text, ok_reason = validate_tailored_bullet(
        "Charted patient vitals in \\textbf{Epic} and \\textbf{Meditech} for a 12-bed unit, cutting handoff errors by 18\\%.",
        canonical,
        nursing_catalog.render_config,
        skill_anchors=nursing_catalog.skill_anchors,
        entry_context=" ".join(clinical.bullets),
    )
    assert ok_reason is None and ok_text is not None

    # Bolding a clinical system the candidate never used is an invented tool.
    bad_text, bad_reason = validate_tailored_bullet(
        "Charted patient vitals in \\textbf{Cerner} for a 12-bed unit, cutting handoff errors by 18\\%.",
        canonical,
        nursing_catalog.render_config,
        skill_anchors=nursing_catalog.skill_anchors,
        entry_context=" ".join(clinical.bullets),
    )
    assert bad_text is None and "invented tool" in str(bad_reason)


def test_nursing_selection_scope_renders_pure_canonical(nursing_catalog, monkeypatch) -> None:
    monkeypatch.setattr(nursing_catalog.render_config, "rewrite_scope", "selection")
    selection = StructuredSelection(
        bullets={e.entry_id: ["Totally rewritten bullet."] for e in nursing_catalog.entries},
    )
    latex, report = render_structured_resume(nursing_catalog, selection)
    assert "Totally rewritten" not in latex
    assert report.tailored_bullets_used == 0
