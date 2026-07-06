#!/usr/bin/env python3
"""Convert a freeform resume profile into a structured-pipeline profile.

The LLM proposes entry categories, role families, and skill anchors; Python
applies and validates them (real parser round-trip + a render of every
family). Nothing is written to the profile unless validation is green AND
--apply is passed.

Usage (from rebuilt_app/):
    python bootstrap_profile.py <profile_key>            # dry run: preview + report
    python bootstrap_profile.py <profile_key> --apply    # write, with backups
    python bootstrap_profile.py <profile_key> --force    # redo an already-tagged profile

Backups go to .resume_cache/bootstrap_backups/<profile>/<timestamp>/ -- NOT
into the profile folder, which the bot purges of unknown files.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from services.resumes.bootstrap import bootstrap_structured_profile  # noqa: E402
from services.resumes.configkey import load_gemini_settings  # noqa: E402

CACHE_ROOT = REPO / "src" / "services" / "resumes" / "resumes_cache"
PREVIEW_ROOT = REPO / ".resume_cache" / "bootstrap_preview"
BACKUP_ROOT = REPO / ".resume_cache" / "bootstrap_backups"
SEP = "-" * 64


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 1:
        print(__doc__)
        print("Available profiles:", ", ".join(sorted(p.name for p in CACHE_ROOT.iterdir() if p.is_dir())))
        return 2

    profile_key = args[0]
    profile_dir = CACHE_ROOT / profile_key
    if not profile_dir.is_dir():
        print(f"No such profile: {profile_dir}")
        return 2

    apply_changes = "--apply" in flags
    force = "--force" in flags

    settings = load_gemini_settings()
    work_dir = PREVIEW_ROOT / profile_key
    if work_dir.exists():
        shutil.rmtree(work_dir)

    print(f"Bootstrapping profile: {profile_key} ({'APPLY' if apply_changes else 'dry run'})")
    print(SEP)
    result = bootstrap_structured_profile(
        profile_dir, settings, work_dir, force=force
    )

    print(f"status:     {result.status}")
    print(f"attempts:   {result.attempts}")
    print(f"provider:   {result.provider}")
    print(f"families:   {', '.join(result.families) or '-'}")
    print(f"categories: {', '.join(result.categories) or '-'}")
    for message in result.messages:
        print(f"  {message}")

    if result.status != "ok":
        return 0 if result.status == "already_structured" else 1

    preview_dir = work_dir / "final"
    preview_dir.mkdir(parents=True, exist_ok=True)
    (preview_dir / "template.tex").write_text(result.template_text, encoding="utf-8")
    (preview_dir / "baseinfo.txt").write_text(result.baseinfo_text, encoding="utf-8")
    (preview_dir / "structured_config.json").write_text(
        json.dumps(result.structured_config, indent=2) + "\n", encoding="utf-8"
    )
    print(SEP)
    print(f"validated files written to: {preview_dir}")

    if not apply_changes:
        print("Dry run complete -- rerun with --apply to install into the profile.")
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = BACKUP_ROOT / profile_key / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ("template.tex", "baseinfo.txt", "structured_config.json"):
        source = profile_dir / name
        if source.exists():
            shutil.copy2(source, backup_dir / name)
    print(f"backed up originals to: {backup_dir}")

    (profile_dir / "template.tex").write_text(result.template_text, encoding="utf-8")
    (profile_dir / "baseinfo.txt").write_text(result.baseinfo_text, encoding="utf-8")
    (profile_dir / "structured_config.json").write_text(
        json.dumps(result.structured_config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"applied structured profile to: {profile_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
