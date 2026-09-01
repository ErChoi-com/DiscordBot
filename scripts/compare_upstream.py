"""Compare our harvest against the upstream company lists.

Counting raw strings gets this badly wrong, because the two lists spell the
same company differently. iCIMS is the clearest case: upstream carries both
"careers-2u" and "2u", our harvest stores the bare form, and a set difference
reports 3,255 companies we are missing that we already have. Workday is the
mirror image -- 6,068 of upstream's entries are unresolvable
("wd1|wd1|careers"), and counting them credits upstream with companies that
do not exist.

Both mistakes push in the same direction, and together they overstated the gap
by 9,340: raw comparison said upstream had 11,738 we lacked, canonical
comparison says 2,398.

So everything is canonicalised through the harvester's own extractor first --
the same test prune and the validator use. An identifier the extractor would
rewrite counts as its rewritten form; one it would reject counts as nothing,
on either side.

Usage:
    python scripts/compare_upstream.py
    python scripts/compare_upstream.py --json
    python scripts/compare_upstream.py --platform icims --list-missing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harvest_ats as hc  # noqa: E402

COMPANY_DIR = REPO_ROOT / "data" / "ats_companies"
HARVEST_DIR = REPO_ROOT / "data" / "ats_harvest"

UPSTREAM_FILES = {
    "greenhouse": "greenhouse_companies.json",
    "lever": "lever_companies.json",
    "ashby": "ashby_companies.json",
    "workday": "workday_companies.json",
    "icims": "icims_companies.json",
    "bamboohr": "bamboohr_companies.json",
    # Upstream names this one differently and stores objects rather than strings.
    "paylocity": "paylocity_companies_clean.json",
}


def _read(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for item in data:
        if isinstance(item, dict):
            item = item.get("guid") or item.get("id") or ""
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def canonical(platform: str, identifiers: list[str]) -> set[str]:
    """Reduce identifiers to the form the extractor would produce.

    Dropping what the extractor rejects is the point as much as rewriting:
    upstream's 6,068 unresolvable Workday entries are not companies either
    side can scrape, so counting them tells you nothing about coverage.
    """
    plat = hc.PLATFORM_BY_NAME.get(platform)
    if plat is None:
        return set(identifiers)
    out: set[str] = set()
    for identifier in identifiers:
        current = hc._current_identifier(plat, identifier)
        if current is not None:
            out.add(current)
    return out


def compare(platform: str) -> dict[str, object]:
    upstream = canonical(platform, _read(COMPANY_DIR / UPSTREAM_FILES[platform]))
    ours = canonical(platform, _read(HARVEST_DIR / f"{platform}.json"))
    return {
        "upstream": len(upstream),
        "ours": len(ours),
        "we_add": len(ours - upstream),
        "they_add": len(upstream - ours),
        "union": len(upstream | ours),
        "_missing": sorted(upstream - ours),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--platform", action="append", dest="platforms",
                        choices=sorted(UPSTREAM_FILES))
    parser.add_argument("--list-missing", action="store_true",
                        help="print the identifiers upstream has and we lack")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    platforms = args.platforms or list(UPSTREAM_FILES)
    rows = {p: compare(p) for p in platforms}

    if args.json:
        print(json.dumps(
            {p: {k: v for k, v in r.items() if not k.startswith("_")}
             for p, r in rows.items()}, indent=2))
    else:
        print(f"{'platform':11} {'theirs':>8} {'ours':>8} {'we add':>8} "
              f"{'they add':>9} {'union':>8}")
        totals = dict.fromkeys(("upstream", "ours", "we_add", "they_add", "union"), 0)
        for name, row in rows.items():
            for key in totals:
                totals[key] += row[key]
            print(f"{name:11} {row['upstream']:8} {row['ours']:8} "
                  f"{row['we_add']:8} {row['they_add']:9} {row['union']:8}")
        print(f"{'TOTAL':11} {totals['upstream']:8} {totals['ours']:8} "
              f"{totals['we_add']:8} {totals['they_add']:9} {totals['union']:8}")

    if args.list_missing:
        for name, row in rows.items():
            for identifier in row["_missing"]:
                print(f"{name}\t{identifier}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
