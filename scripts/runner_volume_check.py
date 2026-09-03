"""Volume test: can a runner validate ATS boards *under load*?

The low-volume check (runner_probe_check.py) found 128/128 agreement from a
GitHub runner, including Workday and iCIMS. That ruled out a blanket block on
the address, but it did not answer the question that matters, because
rate-limiting is triggered by volume rather than by identity. A real validation
pass makes tens of thousands of requests; 128 proves nothing about the
thousandth.

So this runs the same probes the validator uses over a control set of 3,700
boards, every one of them confirmed live from the bot's own host, at the
concurrency the validator would really use.

Reading the result:

  * A slug scored "dead" here is a FALSE NEGATIVE. The board is known to exist,
    so a dead verdict is the exact failure the 2026-08-24 decision is about --
    validate_ats_slugs would write that into dead_slugs, and the bot would stop
    scraping a live company for the whole TTL.
  * A slug scored "unreachable" is a refusal or timeout. That is the honest
    answer to a block: it writes no mark and costs only a wasted request. It is
    a throughput problem, not a correctness one.

The distinction is the whole point. Being blocked is survivable; being blocked
and recording it as "this company no longer exists" is not.

Writes nothing: no marks, no files, no commits. Safe to run anywhere.
"""

from __future__ import annotations

import collections
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_ats_slugs as v  # noqa: E402

CONTROL_FILE = Path(__file__).resolve().parent / "runner_volume_control.json"


def check_platform(platform: str, slugs: list[str]) -> dict:
    probe = v.PROBES.get(platform)
    if probe is None:
        return {}
    workers = v.WORKERS.get(platform, 8)
    delay = v.DELAYS.get(platform, 0.0)
    counts: collections.Counter = collections.Counter()
    false_negatives: list[str] = []
    lock = threading.Lock()

    def one(slug: str) -> None:
        if delay:
            time.sleep(delay)
        try:
            alive = probe(slug)
        except v.Unreachable:
            with lock:
                counts["unreachable"] += 1
            return
        except Exception:  # noqa: BLE001
            with lock:
                counts["error"] += 1
            return
        with lock:
            if alive:
                counts["live"] += 1
            else:
                # Known live locally, so this is a wrong answer, not a closure.
                counts["FALSE_DEAD"] += 1
                false_negatives.append(slug)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, slugs))
    elapsed = time.monotonic() - started
    return {
        "n": len(slugs),
        "live": counts.get("live", 0),
        "false_dead": counts.get("FALSE_DEAD", 0),
        "unreachable": counts.get("unreachable", 0),
        "error": counts.get("error", 0),
        "seconds": round(elapsed, 1),
        "rate_per_s": round(len(slugs) / elapsed, 2) if elapsed else 0,
        "false_dead_examples": false_negatives[:5],
    }


def main() -> int:
    control = json.loads(CONTROL_FILE.read_text(encoding="utf-8"))
    results: dict[str, dict] = {}
    total = live = false_dead = unreachable = 0

    for platform, slugs in control.items():
        r = check_platform(platform, slugs)
        if not r:
            continue
        results[platform] = r
        total += r["n"]
        live += r["live"]
        false_dead += r["false_dead"]
        unreachable += r["unreachable"] + r["error"]
        print(f"[volume] {platform:16} n={r['n']:4} live={r['live']:4} "
              f"FALSE_DEAD={r['false_dead']:4} unreachable={r['unreachable']:4} "
              f"{r['rate_per_s']:5.2f}/s")

    print(f"[volume] TOTAL n={total} live={live} "
          f"FALSE_DEAD={false_dead} unreachable={unreachable}")
    if total:
        print(f"[volume] false-dead rate: {100 * false_dead / total:.2f}% "
              f"-- this is the number that decides whether a runner may validate")
    print(json.dumps({"total": total, "live": live, "false_dead": false_dead,
                      "unreachable": unreachable, "platforms": results}, indent=2))
    # Reports; never gates. A red build here would say nothing useful.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
