"""Recurring ATS-readability validation for every cached resume template.

Compiles each profile's ``template.tex`` through the real ``compile_latex_to_pdf``
pipeline, audits the produced PDF with ``audit_pdf_ats``, and compares the result
against a committed baseline.  Regressions exit non-zero.

Why this exists: a LaTeX compile exiting 0 proves the build ran, not that an ATS
can read the output.  A pdfTeX-family template rendered by a Unicode engine
produces a PDF that looks perfect and extracts as glyph names.  Nothing in the
normal build path notices, so the only way to keep it from coming back is to
compile for real and inspect the bytes on a schedule.

This script never modifies templates or source.  It reports.  Fixing anything it
finds is a human decision.

Usage:
    python scripts/validate_ats.py                 # validate against baseline
    python scripts/validate_ats.py --update-baseline
    python scripts/validate_ats.py --json          # machine-readable to stdout

Exit codes:
    0  all profiles match or improve on the baseline
    1  at least one regression
    2  harness error (no templates, unusable environment)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from services.resumes.ats_check import audit_pdf_ats  # noqa: E402
from services.resumes.resume import (  # noqa: E402
    RESUMES_CACHE_ROOT,
    _template_engine_family,
    compile_latex_to_pdf,
)

BASELINE_PATH = REPO_ROOT / "data" / "ats_baseline.json"
REPORT_DIR = REPO_ROOT / ".ats_validation"
REPORT_KEEP = 14

# A profile may legitimately not be a LaTeX document at all (users have uploaded
# plain text dumps).  Those are expected failures, not regressions, as long as
# the baseline already records them failing.
STATUS_RANK = {"error": 0, "unavailable": 0, "ok": 1}
ATS_RANK = {"unreadable": 0, "unknown": 1, "degraded": 2, "ok": 3}

# Extraction shrinking a little is normal (a template edit); falling off a cliff
# means the text stopped being extractable.
EXTRACTION_DROP_TOLERANCE = 0.20


@dataclass
class ProfileResult:
    profile: str
    family: str
    compile_status: str
    ats_status: str
    type3_ratio: float
    extracted_chars: int
    page_count: int | None
    found_email: bool
    repairs: list[str]
    findings: list[str]
    error: str = ""


def validate_profile(template: Path, log_dir: Path) -> ProfileResult:
    name = template.parent.name
    text = template.read_text(encoding="utf-8", errors="replace")
    family = _template_engine_family(text)
    result = compile_latex_to_pdf(
        latex_document=text,
        job_title="ATS Validation Run",
        template_path=template,
        log_path=log_dir / f"{name}.log",
    )
    audit = audit_pdf_ats(result.pdf_bytes)
    return ProfileResult(
        profile=name,
        family=family,
        compile_status=result.status,
        ats_status=audit.status,
        type3_ratio=round(audit.type3_char_ratio, 4),
        extracted_chars=audit.extracted_chars,
        page_count=result.page_count,
        found_email=audit.found_email,
        repairs=list(result.repairs_applied),
        findings=list(audit.findings),
        error="" if result.status == "ok" else (result.log_excerpt or result.message)[:300],
    )


def compare(current: ProfileResult, baseline: dict | None) -> list[str]:
    """Return regression messages for one profile.

    Comparison is against the profile's own recorded baseline rather than an
    absolute standard, so a template that has always failed stays a known
    failure instead of screaming every run.
    """
    if baseline is None:
        return []  # new profile: recorded, not judged

    problems: list[str] = []
    if STATUS_RANK.get(current.compile_status, 0) < STATUS_RANK.get(
        baseline.get("compile_status", "error"), 0
    ):
        problems.append(
            f"compile regressed {baseline.get('compile_status')} -> {current.compile_status}"
        )
    if ATS_RANK.get(current.ats_status, 1) < ATS_RANK.get(baseline.get("ats_status", "unknown"), 1):
        problems.append(
            f"ATS regressed {baseline.get('ats_status')} -> {current.ats_status}"
            + (f" ({'; '.join(current.findings)})" if current.findings else "")
        )
    base_chars = int(baseline.get("extracted_chars", 0) or 0)
    if base_chars and current.extracted_chars < base_chars * (1 - EXTRACTION_DROP_TOLERANCE):
        problems.append(
            f"extracted text shrank {base_chars} -> {current.extracted_chars} chars"
        )
    if baseline.get("found_email") and not current.found_email:
        problems.append("email no longer survives text extraction")
    return problems


def rotate_reports(directory: Path, keep: int) -> None:
    reports = sorted(directory.glob("run-*.json"))
    for stale in reports[:-keep] if len(reports) > keep else []:
        try:
            stale.unlink()
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit the report to stdout")
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    args = parser.parse_args()

    templates = sorted(RESUMES_CACHE_ROOT.glob("*/template.tex"))
    if not templates:
        print("ats-validate: no profile templates found", file=sys.stderr)
        return 2

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = REPORT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    baseline: dict[str, dict] = {}
    if args.baseline.exists():
        try:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8")).get("profiles", {})
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ats-validate: unusable baseline ({exc}); treating as empty", file=sys.stderr)

    started = time.time()
    results: list[ProfileResult] = []
    regressions: dict[str, list[str]] = {}
    for template in templates:
        result = validate_profile(template, log_dir)
        results.append(result)
        problems = compare(result, baseline.get(result.profile))
        if problems:
            regressions[result.profile] = problems

    elapsed = round(time.time() - started, 1)
    report = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "elapsed_seconds": elapsed,
        "profiles": {r.profile: asdict(r) for r in results},
        "regressions": regressions,
    }

    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(started))
    (REPORT_DIR / f"run-{stamp}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    rotate_reports(REPORT_DIR, REPORT_KEEP)

    if args.update_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(
            json.dumps({"updated": report["started"], "profiles": report["profiles"]}, indent=2),
            encoding="utf-8",
        )
        print(f"ats-validate: baseline updated with {len(results)} profiles")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        ok = sum(1 for r in results if r.ats_status == "ok")
        print(f"ats-validate: {len(results)} profiles in {elapsed}s | {ok} ATS-ok")
        for r in results:
            flag = "!!" if r.profile in regressions else "  "
            print(
                f"{flag} {r.profile:20} {r.family:8} compile={r.compile_status:11} "
                f"ats={r.ats_status:10} type3={r.type3_ratio:<7} chars={r.extracted_chars}"
            )
            for finding in r.findings:
                print(f"       finding: {finding}")
        for profile, problems in regressions.items():
            for problem in problems:
                print(f"REGRESSION {profile}: {problem}")

    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
