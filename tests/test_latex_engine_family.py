"""Engine/template family matching and unavailable-font recovery.

Two failure modes motivate these tests, both found by compiling real templates
and inspecting the resulting PDFs:

1. A pdfTeX-family template (XCharter + ``[T1]{fontenc}``) compiled under
   xelatex produces a PDF that renders perfectly and whose text extracts as
   glyph names (``/x45/x72/x6E``) because every glyph becomes a Type 3 bitmap
   with no ToUnicode map.  It is unreadable to an ATS and looks fine to a human,
   so nothing downstream notices.  Engine fallback must never cross families.

2. A font package whose font files cannot be built (MiKTeX
   "miktex-makemf did not succeed") kills the build outright, even though
   dropping that one package yields a clean, fully extractable resume in the
   default typeface.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from services.resumes import resume as resume_service
from services.resumes.resume import compile_latex_to_pdf

PDFTEX_TEMPLATE = "\\usepackage{XCharter}\n\\usepackage[T1]{fontenc}"
FONTSPEC_TEMPLATE = "\\usepackage{fontspec}\n\\setmainfont{Calibri}"

MIKTEX_FONT_ERROR = (
    "Running miktex-makemf.exe...\n"
    "!pdfTeX error: pdflatex.EXE (file SourceSans3-It-tlf-t1--base): Font SourceSans\n"
    "Sorry, but miktex-makemf did not succeed.\n"
)


# ── family classification ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("snippet", "family", "engines"),
    [
        (PDFTEX_TEMPLATE, "pdftex", ["pdflatex"]),
        (FONTSPEC_TEMPLATE, "unicode", ["xelatex", "lualatex"]),
        ("\\newfontfamily{\\hdr}{Arial}", "unicode", ["xelatex", "lualatex"]),
        ("\\usepackage{unicode-math}", "unicode", ["xelatex", "lualatex"]),
        ("\\directlua{tex.print(1)}", "luatex", ["lualatex"]),
        ("\\begin{luacode}x\\end{luacode}", "luatex", ["lualatex"]),
    ],
)
def test_engine_family_never_crosses_font_machinery(snippet, family, engines) -> None:
    assert resume_service._template_engine_family(snippet) == family
    assert resume_service._preferred_latex_engines(snippet) == engines


def test_every_stored_profile_template_is_pdftex_family() -> None:
    """Every shipped profile descends byte-identically from the example seed.

    If this ever fails, a profile has been seeded from something else and the
    Unicode-engine path has live users -- worth knowing deliberately.
    """
    templates = sorted(resume_service.RESUMES_CACHE_ROOT.glob("*/template.tex"))
    if not templates:
        # resumes_cache/ is gitignored personal data, so a clean checkout -- every
        # CI runner, and the container image, which excludes it deliberately --
        # has no profiles to check. Skipping keeps the assertion below meaningful
        # where profiles exist without failing where they cannot.
        # (Same reasoning as the module-level skip in test_cover_letter.py.)
        pytest.skip("resumes_cache profiles not present (gitignored)")
    for template in templates:
        text = template.read_text(encoding="utf-8", errors="replace")
        assert resume_service._template_engine_family(text) == "pdftex", template.name


# ── the fallback that must not happen ────────────────────────────────────────


def _environment(pdflatex: str | None, xelatex: str | None, lualatex: str | None = None):
    class _Env:
        ready = True
        pdflatex_path = pdflatex
        xelatex_path = xelatex
        lualatex_path = lualatex
        missing_files: list[str] = []

    return _Env()


def test_pdftex_template_never_falls_back_to_a_unicode_engine(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "template.tex"
    template_path.write_text("\\documentclass{article}\n", encoding="utf-8")
    monkeypatch.setattr(
        "services.resumes.resume.check_template_compile_environment",
        lambda path: _environment("C:/tex/pdflatex.exe", "C:/tex/xelatex.exe", "C:/tex/lualatex.exe"),
    )

    calls: list[str] = []

    def _runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append(command[0])
        return subprocess.CompletedProcess(command, 1, "", "! pdftex error")

    result = compile_latex_to_pdf(
        latex_document=(
            "\\documentclass{article}\n" + PDFTEX_TEMPLATE + "\n\\begin{document}Hi\\end{document}"
        ),
        job_title="Python Developer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
        command_runner=_runner,
    )

    assert result.status == "error"
    assert calls, "expected at least one compile attempt"
    assert all("pdflatex" in call for call in calls)


def test_missing_family_engine_reports_which_engine_is_needed(monkeypatch, tmp_path: Path) -> None:
    """Saying "no engine found" would be a lie when pdflatex is installed."""
    template_path = tmp_path / "template.tex"
    template_path.write_text("\\documentclass{article}\n", encoding="utf-8")
    monkeypatch.setattr(
        "services.resumes.resume.check_template_compile_environment",
        lambda path: _environment("C:/tex/pdflatex.exe", None),
    )

    result = compile_latex_to_pdf(
        latex_document=(
            "\\documentclass{article}\n" + FONTSPEC_TEMPLATE + "\n\\begin{document}Hi\\end{document}"
        ),
        job_title="Python Developer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
        command_runner=lambda command, cwd: subprocess.CompletedProcess(command, 0, "", ""),
    )

    assert result.status == "unavailable"
    assert "unicode-family" in result.message
    assert "xelatex" in result.message


# ── unavailable font packages ────────────────────────────────────────────────


def test_font_build_failure_is_classified_separately_from_missing_style() -> None:
    """No .sty is named in the error, so the missing-style path strips nothing."""
    kind = resume_service._classify_latex_compile_failure(
        "\\usepackage{sourcesanspro}", MIKTEX_FONT_ERROR, ""
    )
    assert kind == "font-unavailable"


def test_font_failure_maps_font_filename_back_to_its_package() -> None:
    """"SourceSans3-It-tlf-t1--base" must resolve to \\usepackage{sourcesanspro}."""
    document = (
        "\\documentclass{article}\n"
        "\\usepackage[letterpaper]{geometry}\n"
        "\\usepackage{sourcesanspro}\n"
        "\\begin{document}Hi\\end{document}\n"
    )
    updated, repairs = resume_service._apply_targeted_latex_auto_fix(
        document, "font-unavailable", stdout_text=MIKTEX_FONT_ERROR, stderr_text=""
    )
    assert "sourcesanspro" in " ".join(repairs)
    assert "sourcesanspro" not in updated
    # Structural packages must survive; only the font package is dropped.
    assert "geometry" in updated


def test_font_failure_leaves_unrelated_packages_alone() -> None:
    """No font package present means nothing to strip -- do not guess."""
    document = (
        "\\documentclass{article}\n"
        "\\usepackage{enumitem}\n"
        "\\usepackage{hyperref}\n"
        "\\begin{document}Hi\\end{document}\n"
    )
    updated, repairs = resume_service._apply_targeted_latex_auto_fix(
        document, "font-unavailable", stdout_text=MIKTEX_FONT_ERROR, stderr_text=""
    )
    assert "enumitem" in updated
    assert "hyperref" in updated
    assert not any("font packages" in repair for repair in repairs)
