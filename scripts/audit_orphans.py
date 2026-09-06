"""Find capabilities that exist and nothing calls.

This repo keeps producing them, and they are expensive because they look like
working features. Found and fixed in one day: `reload_company_lists` (so a
running bot could never see a newly harvested company), `ats_fleet_coverage`
and `silent_ats_platforms` (recorded per-platform health that no embed read),
and `scrape_ats_platform(company_slugs=...)` (a parameter with no caller since
it was written). Each was written, tested, and then never wired to anything.

A grep is not enough, for two reasons this script exists to handle:

  - A public-named helper used only inside its own module is fine, not an
    orphan. `geo_priority.match_keys` has no external caller and never needed
    one; `rank` and `partition` both use it.
  - Some orphans are deliberate. `ats_traversal.next_slice` is documented as
    unwired *and* carries a warning against wiring it, because at the count it
    was sized for it would cut coverage forty-fold. Silencing it needs a
    reason recorded next to the code, not a line in this script.

So: report a function only when nothing anywhere calls it -- including its own
module -- and it is not marked with an explicit `# orphan-ok:` note giving the
reason. Tests do not count as callers; a capability exercised only by its own
test is exactly the shape being hunted.

Usage:
    python scripts/audit_orphans.py            # report, exit 1 if any found
    python scripts/audit_orphans.py --list     # report, always exit 0
"""
from __future__ import annotations

import argparse
import ast
import collections
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("src", "scripts")

# Marker a maintainer puts above a def to record a deliberate orphan and why.
# Anything else is a finding. The colon is optional but the reason is not: the
# marker on its own records nothing, and a silenced finding with no reason is
# the state this script was written to end.
ORPHAN_OK = re.compile(r"#\s*orphan-ok[:\s]\s*(\S.*)")


def _sources() -> dict[Path, str]:
    out: dict[Path, str] = {}
    for d in SOURCE_DIRS:
        for p in sorted((REPO / d).rglob("*.py")):
            try:
                out[p] = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return out


def _public_functions(tree: ast.Module) -> list[tuple[str, int]]:
    """Module-level public functions, with the line their def starts on."""
    return [(n.name, n.lineno) for n in tree.body
            if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]


def _excused(text: str, lineno: int) -> str | None:
    """The reason on an `# orphan-ok:` comment above the def, if there is one.

    The whole contiguous comment block is scanned, not a fixed number of lines:
    a reason worth recording is usually several lines of why, and a marker that
    only counts when it sits within three lines of the def rewards terse notes
    over useful ones. Decorator lines are walked through for the same reason.
    """
    lines = text.splitlines()
    i = lineno - 2  # 1-indexed lineno -> the line above the def
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith("#"):
            m = ORPHAN_OK.search(stripped)
            if m:
                return m.group(1).strip()
        elif stripped.startswith("@") or not stripped:
            pass
        else:
            break
        i -= 1
    return None


def _referenced_names(tree: ast.Module) -> collections.Counter:
    """Every name this module actually uses, counted from the syntax tree.

    A regex over the raw text cannot tell a call from a mention. Comments and
    prose are where this repo records *why* something is unwired, so the moment
    a deliberate orphan is documented -- "the release half of
    acquire_browser_work_slot" -- the regex sees a second occurrence and the
    orphan stops being reported. Explaining a finding made it disappear.

    Exact string literals still count. Nothing here is dispatched by name today,
    but getattr(module, "some_function") is a real pattern and reporting one of
    those as uncalled would be a false accusation; a docstring is never exactly
    equal to a function name, so this costs nothing.
    """
    used: collections.Counter = collections.Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used[node.id] += 1
        elif isinstance(node, ast.Attribute):
            used[node.attr] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            used[node.value.strip()] += 1
    return used


def find_orphans(sources: dict[Path, str] | None = None) -> list[tuple[str, str]]:
    """(location, name) for every uncalled public function. Sorted."""
    sources = sources if sources is not None else _sources()
    trees: dict[Path, ast.Module] = {}
    for path, text in sources.items():
        try:
            trees[path] = ast.parse(text)
        except SyntaxError:
            continue

    used: collections.Counter = collections.Counter()
    for tree in trees.values():
        used.update(_referenced_names(tree))

    found: list[tuple[str, str]] = []
    for path, tree in trees.items():
        text = sources[path]
        for name, lineno in _public_functions(tree):
            if used[name] or _excused(text, lineno):
                continue
            found.append((path.relative_to(REPO).as_posix(), name))
    return sorted(found)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true",
                    help="report but always exit 0")
    args = ap.parse_args()

    orphans = find_orphans()
    if not orphans:
        print("[orphans] none — every public function has a caller")
        return 0

    print(f"[orphans] {len(orphans)} public function(s) nothing calls:")
    for location, name in orphans:
        print(f"  {location}:{name}")
    print("\nEither wire it up, delete it, or record why it is deliberate with")
    print("an `# orphan-ok: <reason>` comment on the line above its def.")
    return 0 if args.list else 1


if __name__ == "__main__":
    sys.exit(main())
