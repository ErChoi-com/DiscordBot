"""Two ways a LaTeX compile can "succeed" and still hand the user nothing.

Both were found by pushing synthetic adversarial documents through the real
``compile_latex_to_pdf`` and auditing the resulting bytes, not by reading code.

1. ``\\usepackage[T1]{fontenc}`` with no T1 Type 1 font installed does not fail.
   MiKTeX generates EC bitmap (``.pk``) fonts on the fly, the engine exits 0,
   the PDF prints perfectly -- and every glyph is a Type 3 bitmap with no
   ToUnicode map, so an ATS extracts nothing.  There is no error text to
   classify; the only evidence is in the compiled bytes.

2. An empty document makes pdflatex report "No pages of output" and leave a
   zero-byte ``.pdf`` behind.  The file exists, so an ``exists()`` check passes
   and the caller gets ``status="ok"`` with an empty attachment.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from services.resumes import resume as resume_service
from services.resumes.ats_check import audit_pdf_ats
from services.resumes.resume import compile_latex_to_pdf

pdflatex_required = pytest.mark.skipif(
    shutil.which("pdflatex") is None, reason="pdflatex is not installed"
)

# A T1-encoded preamble naming a font package whose Type 1 files MiKTeX does
# not ship.  This is the shape that silently produces bitmaps.
BITMAP_TRAP_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[letterpaper,top=0.5in,bottom=0.5in,left=0.5in,right=0.5in]{geometry}
\usepackage{sourcesanspro}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{enumitem}
\raggedright
\pagestyle{empty}
\input{glyphtounicode}
\pdfgentounicode=1
"""

RESUME_BODY = r"""\begin{document}
\centerline{\Huge Jane Q. Applicant}
\vspace{5pt}
\centerline{jane.applicant@example.com | (416) 555-0134 | Toronto, ON}
\section*{Skills}
\textbf{Languages:} Python, C++, Rust, TypeScript, Go, SQL \\
\textbf{Tools:} Docker, Kubernetes, Terraform, PostgreSQL, Redis, Kafka
\section*{Experience}
\textbf{Senior Backend Engineer,} Acme Corp \hfill Jan 2022 -- Present
\begin{itemize}
  \item Cut p99 checkout latency from 840ms to 210ms by replacing an N+1 ORM path
        with a single materialised view refreshed on write.
  \item Led migration of 47 services off a shared Postgres instance, eliminating
        the last single point of failure in the payments path.
  \item Mentored four junior engineers; two were promoted within the year.
\end{itemize}
\textbf{Backend Engineer,} Globex \hfill Jun 2019 -- Dec 2021
\begin{itemize}
  \item Built an event ingestion pipeline sustaining 120k events per second on
        six nodes, replacing a batch job that ran once an hour.
  \item Reduced cloud spend 31\% by rightsizing instances and tiering cold data.
\end{itemize}
\section*{Education}
University of Toronto \hfill Apr 2019 \\
BASc in Computer Engineering
\end{document}
"""


# -- the lmodern injection, in isolation -------------------------------------


def test_lmodern_is_inserted_above_the_t1_fontenc_line() -> None:
    document = BITMAP_TRAP_PREAMBLE + RESUME_BODY
    updated, repairs = resume_service._ensure_type1_fallback_font(document)

    assert repairs, "a T1 document without lmodern should get lmodern"
    assert "\\usepackage{lmodern}" in updated
    # Order matters: lmodern must be seen before fontenc selects the encoding.
    assert updated.index("\\usepackage{lmodern}") < updated.index("{fontenc}")
    # Nothing else may be disturbed.
    assert updated.replace("\\usepackage{lmodern}\n", "") == document


def test_lmodern_injection_is_idempotent() -> None:
    once, first = resume_service._ensure_type1_fallback_font(
        BITMAP_TRAP_PREAMBLE + RESUME_BODY
    )
    twice, second = resume_service._ensure_type1_fallback_font(once)

    assert first and not second
    assert twice == once
    assert once.count("\\usepackage{lmodern}") == 1


def test_documents_without_t1_fontenc_are_left_alone() -> None:
    """A fontspec/xelatex document has no T1 encoding and needs no Latin Modern."""
    document = (
        "\\documentclass{article}\n"
        "\\usepackage{fontspec}\n"
        "\\setmainfont{Calibri}\n"
        "\\begin{document}Hi\\end{document}\n"
    )
    updated, repairs = resume_service._ensure_type1_fallback_font(document)

    assert not repairs
    assert updated == document


def test_commented_out_fontenc_does_not_trigger_injection() -> None:
    document = (
        "\\documentclass{article}\n"
        "% \\usepackage[T1]{fontenc}\n"
        "\\begin{document}Hi\\end{document}\n"
    )
    updated, repairs = resume_service._ensure_type1_fallback_font(document)

    assert not repairs
    assert updated == document


# -- end to end, against a real engine ---------------------------------------


@pdflatex_required
def test_silently_bitmapped_resume_is_salvaged_into_extractable_text(tmp_path: Path) -> None:
    """The compile already succeeded; the salvage is driven purely by the bytes."""
    document = BITMAP_TRAP_PREAMBLE + RESUME_BODY
    template_path = tmp_path / "template.tex"
    template_path.write_text(document, encoding="utf-8")

    result = compile_latex_to_pdf(
        latex_document=document,
        job_title="Senior Backend Engineer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
    )

    assert result.status == "ok"
    assert result.pdf_bytes

    audit = audit_pdf_ats(result.pdf_bytes)
    assert audit.status == "ok", audit.summary()
    # The whole point: the text is real characters, not bitmap outlines.
    assert audit.type3_char_ratio == 0.0
    assert audit.extracted_chars > 400
    assert audit.found_email
    assert any("lmodern" in repair for repair in result.repairs_applied)


@pdflatex_required
def test_salvage_is_reported_to_the_caller_not_applied_silently(tmp_path: Path) -> None:
    document = BITMAP_TRAP_PREAMBLE + RESUME_BODY
    template_path = tmp_path / "template.tex"
    template_path.write_text(document, encoding="utf-8")

    result = compile_latex_to_pdf(
        latex_document=document,
        job_title="Senior Backend Engineer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
    )

    # The author's chosen font really was dropped -- they must be able to see
    # that in the repair list rather than wondering why the PDF looks different.
    assert any("sourcesanspro" in repair for repair in result.repairs_applied)
    assert result.repair_attempts >= 1


@pdflatex_required
def test_a_healthy_resume_is_never_touched_by_the_salvage(tmp_path: Path) -> None:
    """XCharter is installed, so nothing about this document needs rescuing."""
    document = (
        BITMAP_TRAP_PREAMBLE.replace(
            "\\usepackage{sourcesanspro}", "\\usepackage{XCharter}"
        )
        + RESUME_BODY
    )
    template_path = tmp_path / "template.tex"
    template_path.write_text(document, encoding="utf-8")

    result = compile_latex_to_pdf(
        latex_document=document,
        job_title="Senior Backend Engineer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
    )

    assert result.status == "ok"
    assert result.ats_status == "ok"
    assert result.repairs_applied == []
    assert result.repair_attempts == 0


@pdflatex_required
def test_a_short_resume_is_not_mistaken_for_a_bitmap_failure(tmp_path: Path) -> None:
    """"Unreadable" has more than one cause, and only one of them is fonts.

    A resume too short to clear the auditor's minimum-length rule audits as
    unreadable while its fonts are perfectly fine.  Dropping the author's
    typeface would not add a single character to it.  Caught in review after an
    earlier version of the salvage gated on the status instead of the signal.
    """
    short_body = (
        "\\begin{document}\n"
        "\\centerline{\\Huge Jane Q. Applicant}\n"
        "\\centerline{jane.applicant@example.com}\n"
        "\\section*{Skills}\nPython, C++\n"
        "\\end{document}\n"
    )
    document = (
        BITMAP_TRAP_PREAMBLE.replace(
            "\\usepackage{sourcesanspro}", "\\usepackage{XCharter}"
        )
        + short_body
    )
    template_path = tmp_path / "template.tex"
    template_path.write_text(document, encoding="utf-8")

    result = compile_latex_to_pdf(
        latex_document=document,
        job_title="Senior Backend Engineer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
    )

    assert result.status == "ok"
    # The audit may well flag it as thin -- that is its job.  What must not
    # happen is the pipeline reacting by stripping fonts.
    assert result.repairs_applied == []
    assert result.repair_attempts == 0
    audit = audit_pdf_ats(result.pdf_bytes)
    assert audit.type3_char_ratio == 0.0


@pdflatex_required
def test_zero_byte_pdf_is_a_failure_not_a_successful_compile(tmp_path: Path) -> None:
    """pdflatex exits 0 and leaves an empty .pdf; "ok" would ship an empty file."""
    document = BITMAP_TRAP_PREAMBLE.replace(
        "\\usepackage{sourcesanspro}", "\\usepackage{XCharter}"
    ) + "\\begin{document}\n\\end{document}\n"
    template_path = tmp_path / "template.tex"
    template_path.write_text(document, encoding="utf-8")

    result = compile_latex_to_pdf(
        latex_document=document,
        job_title="Senior Backend Engineer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
    )

    assert result.status == "error"
    assert not result.pdf_bytes
