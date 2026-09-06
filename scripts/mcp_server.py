"""A small, read-only MCP server over this bot's job archive and ATS state.

Exists so an MCP client (Claude Code, Claude Desktop, anything else that speaks
the protocol) can ask questions about the archive without a human first writing
a one-off script.  It is a thin adapter, not a second implementation: every tool
below delegates to the same service functions the bot itself calls, so there is
no separate query path that can drift from the real one.

Deliberately read-only.  Nothing here scrapes, opens a browser, spends an API
quota, or writes to the archive -- the destructive and expensive paths stay
behind the Discord commands, where a human is driving.  Adding a write tool
would mean an MCP client could mutate archive state on its own initiative;
don't, without a clear reason.

Requires the MCP SDK, which is not a bot runtime dependency -- it is deliberately
kept out of requirements.txt so it never ships in the Docker image:

    python -m pip install "mcp>=2"

Run it directly for a manual check, or wire it into a client over stdio:

    python scripts/mcp_server.py            # serve on stdio (what clients do)
    python scripts/mcp_server.py --selftest # call each tool once, print JSON

Claude Code registration (.mcp.json at the repo root):

    {"mcpServers": {"jobbot": {"command": "python",
                               "args": ["scripts/mcp_server.py"]}}}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mcp.server.mcpserver import MCPServer  # noqa: E402  (needs the path insert)
from mcp.types import ToolAnnotations  # noqa: E402

#: Every tool here only reads. Advertising that lets a client skip the
#: are-you-sure prompt it would otherwise show for an unknown-effect tool.
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)

#: Windows a caller may ask for. Each extra day is another archive day parsed
#: and deduped, and the archive holds months, so an unbounded `days` turns one
#: tool call into a multi-second scan of everything on disk.
MAX_WINDOW_DAYS = 90

#: A tool result is model context, not a file. Cap what any one call can return.
MAX_RESULTS = 100

mcp = MCPServer(
    name="jobbot",
    version="0.1.0",
    instructions=(
        "Read-only access to the Discord job bot's archive: search archived "
        "postings, summarise a window, list which days hold data, and inspect "
        "ATS company-list coverage. No scraping and no writes."
    ),
)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


@mcp.tool(
    description=(
        "Search archived job postings from the last `days` days. `query` is "
        "matched case-insensitively against title, company, location and "
        "description; omit it to get everything in the window. Results are "
        "deduped across boards, newest day first."
    ),
    annotations=READ_ONLY,
)
def search_jobs(query: str = "", days: int = 7, limit: int = 20) -> dict[str, Any]:
    from services.job_match import load_window_jobs

    days = _clamp(days, 1, MAX_WINDOW_DAYS)
    limit = _clamp(limit, 1, MAX_RESULTS)

    jobs, dates = load_window_jobs(days)
    needle = query.strip().lower()
    if needle:
        jobs = [job for job in jobs if needle in job.match_text()]

    return {
        "query": query,
        "days": days,
        "dates_scanned": dates,
        "total_matches": len(jobs),
        "returned": min(len(jobs), limit),
        "jobs": [
            {
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "link": job.link,
                "source": job.site_label,
                "date_posted": job.date_posted,
            }
            for job in jobs[:limit]
        ],
    }


@mcp.tool(
    description=(
        "Summarise the archive over the last `days` days: how many deduped "
        "postings there are, and which sources and companies they came from."
    ),
    annotations=READ_ONLY,
)
def archive_stats(days: int = 7, top: int = 10) -> dict[str, Any]:
    from services.job_match import load_window_jobs

    days = _clamp(days, 1, MAX_WINDOW_DAYS)
    top = _clamp(top, 1, 50)

    jobs, dates = load_window_jobs(days)
    by_source = Counter(job.site_label or "unknown" for job in jobs)
    by_company = Counter(job.company for job in jobs if job.company)

    return {
        "days": days,
        "dates_scanned": dates,
        "total_jobs": len(jobs),
        # A posting with no description cannot be scored on skills, so this
        # ratio is the ceiling on how well matching can work for the window.
        "with_description": sum(1 for job in jobs if job.description),
        "by_source": dict(by_source.most_common()),
        "top_companies": dict(by_company.most_common(top)),
    }


@mcp.tool(
    description=(
        "List the archive days that hold data, newest first, with the posting "
        "count for the most recent few. Use it to find a window worth querying."
    ),
    annotations=READ_ONLY,
)
def archive_days(count_recent: int = 7) -> dict[str, Any]:
    from services.jba.merge_data import daily_log_count, list_log_dates

    dates = list_log_dates()
    count_recent = _clamp(count_recent, 0, 60)
    # Counting reads each day's records, so only the head of the list is
    # counted; the rest are returned as bare dates.
    return {
        "total_days": len(dates),
        "newest": dates[0] if dates else None,
        "oldest": dates[-1] if dates else None,
        "dates": dates,
        "recent_counts": {d: daily_log_count(d) for d in dates[:count_recent]},
    }


@mcp.tool(
    description=(
        "ATS scraping coverage per platform: how many company slugs are "
        "configured, and how many are currently marked dead (404/empty, "
        "skipped until their recheck window elapses)."
    ),
    annotations=READ_ONLY,
)
def ats_coverage() -> dict[str, Any]:
    from services import ats_service

    lists = ats_service.load_company_lists()
    platforms = {}
    for platform in sorted(ats_service._PLATFORM_FILES):
        slugs = lists.get(platform, [])
        dead = ats_service._load_dead_slugs(platform)
        platforms[platform] = {
            "configured": len(slugs),
            "dead": len(dead),
            "live": max(0, len(slugs) - len(dead)),
        }

    return {
        "platforms": platforms,
        "total_configured": sum(p["configured"] for p in platforms.values()),
        "total_dead": sum(p["dead"] for p in platforms.values()),
        # load_company_lists() warns and returns {} when data/ats_companies is
        # absent (it is gitignored). Say so plainly rather than reporting zeros
        # that look like "no companies configured".
        "company_lists_present": bool(lists),
    }


def _selftest() -> int:
    """Call each registered tool once and print the result.

    A server that imports cleanly can still fail on first call -- a missing
    gitignored data directory, a schema mismatch. This exercises the real tool
    bodies so that shows up here rather than inside a client.
    """
    tools = asyncio.run(mcp.list_tools())
    calls: dict[str, dict[str, Any]] = {
        "search_jobs": {"query": "engineer", "days": 7, "limit": 3},
        "archive_stats": {"days": 7, "top": 5},
        "archive_days": {"count_recent": 3},
        "ats_coverage": {},
    }

    missing = {t.name for t in tools} - set(calls)
    if missing:
        print(f"selftest has no arguments for: {sorted(missing)}", file=sys.stderr)
        return 2

    failures = 0
    for name, kwargs in calls.items():
        try:
            result = asyncio.run(mcp.call_tool(name, kwargs))
        except Exception as exc:
            failures += 1
            print(f"--- {name} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        # isError is how a tool reports a handled failure; it is not raised.
        if getattr(result, "is_error", False):
            failures += 1
            print(f"--- {name} returned an error result", file=sys.stderr)
        payload = getattr(result, "structured_content", None)
        if payload is None:
            payload = [getattr(c, "text", c) for c in getattr(result, "content", [])]
        print(f"--- {name}")
        print(json.dumps(payload, indent=2, default=str)[:2000])
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selftest", action="store_true",
                        help="call every tool once and print the results")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()

    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
