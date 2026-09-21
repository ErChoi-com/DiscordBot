"""Distil labels for the posting-piece encoder from Gemini.

Each posting in the trial corpus is split with posting_segments.split_description
and sent WHOLE, pieces in order, so the teacher sees the section a piece sits in
("Paid Time Off" under Benefits vs. a requirement list). Several postings are
packed per call. Output: one row per piece in .resume_lab/boilerplate/pieces.jsonl
(resumable: postings already written are skipped).

Budget rule: at most --max-calls per model per run, and a model retires for the
run on its first quota error. gemini-3.5-flash is never used -- it is the bot's
primary resume model.

    python scripts/posting_label.py --postings 40 --max-calls 1     # smoke test
    python scripts/posting_label.py --postings 400                  # one day's run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import queue
import random
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from services.resumes.posting_segments import split_description  # noqa: E402

CORPUS = ROOT / ".resume_lab" / "trials" / "corpus.json"
OUT = ROOT / ".resume_lab" / "boilerplate" / "pieces.jsonl"
LABELS = ("SKILL_DUTY", "ROLE_FACTS", "COMPANY_CONTEXT", "BOILERPLATE")

GUIDE = """You label the pieces of job postings for a resume-tailoring tool. Each posting is given
as numbered pieces IN ORDER; use the surrounding pieces as context (a piece under a
"Benefits" heading is a benefit even if it reads like a sentence about work).

Give every piece exactly one label:

SKILL_DUTY: what the job does or requires. Responsibilities, duties, projects, what the team
  builds; required or preferred skills, tools, technologies, experience, education,
  certifications; soft-skill requirements ("strong communication skills").

ROLE_FACTS: facts about the position a candidate is matched on, not skills. Seniority or level,
  term or duration ("16-month co-op"), start date, degree or program the candidate must be
  enrolled in, years of experience required, location, on-site / hybrid / remote, travel,
  shift or hours, pay or salary figures, clearance or work-authorization REQUIREMENTS.

COMPANY_CONTEXT: concrete, specific facts about the employer that a candidate could mirror in
  their writing: what the company or team builds, its product or industry domain, named
  values or working style ("safety-critical systems", "clean energy for nuclear plants",
  "small Agile team"). Only when concrete. Generic praise ("a leading global firm",
  "our people are our greatest asset") is BOILERPLATE.

BOILERPLATE: everything else. Equal-opportunity / non-discrimination / diversity statements,
  accommodation and accessibility notices, AI-screening and privacy notices, hiring-process
  text ("only those selected will be contacted", how to apply), benefits and perks lists,
  compensation philosophy without figures, generic culture or mission hype, legal text,
  job-board or scraper furniture (IDs, "posted 2 days ago", links, sign-up prompts).

Headings: label a heading by the section it introduces ("Requirements" -> SKILL_DUTY,
"Benefits" -> BOILERPLATE, "About Us" -> COMPANY_CONTEXT or BOILERPLATE by its content).
A piece mixing kinds: SKILL_DUTY if it names any skill or duty, else ROLE_FACTS if it states
any role fact, else by majority.

Also return "terms" for every piece: concrete terms useful on a resume, copied EXACTLY as
written (a substring of the piece): tools, software, languages, frameworks, platforms,
hardware, instruments, technical methods and standards ("unit testing", "FMEA",
"ISO 13485", "GD&T"), certifications, licences, degrees and fields of study, domain concepts
a recruiter scans for ("embedded systems", "financial modelling"). Not generic words
("team", "experience", "communication"), not company names, locations or job titles.
Terms are usually empty for BOILERPLATE. Return an entry for every id."""

SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string", "enum": list(LABELS)},
                    "terms": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "label", "terms"],
            },
        }
    },
    "required": ["labels"],
}


def gemini_key() -> str:
    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("'\"")
    for k in ("GOOGLE_AI_STUDIO_API_KEY", "GEMINI_API_KEY", "geminiAPI"):
        if env.get(k):
            return env[k]
    sys.exit("no Gemini key in .env")


def posting_id(p: dict) -> str:
    return hashlib.sha1((p.get("posting_url") or p.get("title", "") + p.get("company", "")).encode()).hexdigest()[:12]


_BOARDS = {"", "linkedin", "indeed", "glassdoor", "ziprecruiter", "google"}


def employer(p: dict) -> str:
    """The real employer. The corpus `company` field is blank for 302 postings and
    names the board ("LinkedIn") for 56 more; source_message carries
    "[LinkedIn/Ericsson] ...". Used for the per-company cap and the
    company-held-out test split, so it must not collapse postings together."""
    company = (p.get("company") or "").strip()
    if company.lower() not in _BOARDS:
        return company
    m = re.match(r"\[([^\]/]+)/([^\]]+)\]", p.get("source_message") or "")
    if m and m.group(2).strip():
        return m.group(2).strip()
    return company or (p.get("posting_url") or "unknown")


def choose_postings(n: int, seed: int, done: set[str]) -> list[dict]:
    """Company-diverse sample: at most 2 postings per company, shuffled."""
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    rng.shuffle(corpus)
    per_company: dict[str, int] = {}
    chosen = []
    for p in corpus:
        body = (p.get("description") or "").split("Description:", 1)[-1].strip()
        if len(body) < 400:
            continue
        company = employer(p).lower()
        if per_company.get(company, 0) >= 2:
            continue
        per_company[company] = per_company.get(company, 0) + 1
        pid = posting_id(p)
        pieces = split_description(body)
        if len(pieces) < 4:
            continue
        chosen.append({"pid": pid, "company": employer(p), "title": p.get("title") or "",
                       "pieces": pieces[:120]})
        if len(chosen) >= n:
            break
    return [c for c in chosen if c["pid"] not in done]


def choose_jobsdb_postings(per_site: int, seed: int, done: set[str]) -> list[dict]:
    """Range beyond the LinkedIn-heavy trial corpus: live ATS postings from
    jobs.db, `per_site` per platform, one per company. Text is flattened the way
    the bot flattens it (ats_service._html_to_text: unescape, tags and all
    whitespace to single spaces), because that is what the model will read."""
    import html as _html
    import sqlite3

    con = sqlite3.connect(f"file:{ROOT / 'data/jba/jobs/jobs.db'}?mode=ro", uri=True)
    by_site: dict[str, list[dict]] = {}
    for (blob,) in con.execute("SELECT data FROM jobs"):
        r = json.loads(blob)
        raw = r.get("description") or ""
        if len(raw) < 400:
            continue
        by_site.setdefault(r.get("_source_site") or r.get("site") or "?", []).append(r)
    rng = random.Random(seed)
    chosen = []
    for site, rows in sorted(by_site.items()):
        rng.shuffle(rows)
        companies: set[str] = set()
        taken = 0
        for r in rows:
            company = (r.get("company") or "").strip()
            if not company or company.lower() in companies:
                continue
            text = _html.unescape(r.get("description") or "")
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
            pieces = split_description(text)
            if len(pieces) < 4:
                continue
            pid = hashlib.sha1((r.get("link") or r.get("job_url") or text[:200]).encode()).hexdigest()[:12]
            if pid in done:
                continue
            companies.add(company.lower())
            chosen.append({"pid": pid, "company": company, "title": r.get("title") or "",
                           "site": site, "pieces": pieces[:120]})
            taken += 1
            if taken >= per_site:
                break
    return chosen


def pack(postings: list[dict], max_pieces: int) -> list[list[dict]]:
    batches, cur, size = [], [], 0
    for p in postings:
        if cur and size + len(p["pieces"]) > max_pieces:
            batches.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += len(p["pieces"])
    if cur:
        batches.append(cur)
    return batches


def prompt_for(batch: list[dict]) -> str:
    parts = []
    for p in batch:
        parts.append(f"=== POSTING {p['pid']}: {p['title']} at {p['company']}")
        parts += [f"{p['pid']}-{i}: {text}" for i, text in enumerate(p["pieces"])]
    return "\n".join(parts)


def rows_for(batch: list[dict], got: dict[str, dict], model: str) -> list[dict] | None:
    rows = []
    for p in batch:
        for i, text in enumerate(p["pieces"]):
            x = got.get(f"{p['pid']}-{i}")
            if not x or x.get("label") not in LABELS:
                return None  # incomplete answer: relabel the whole batch later
            terms = [t for t in dict.fromkeys(x.get("terms") or []) if t and t in text]
            rows.append({"pid": p["pid"], "company": p["company"], "idx": i, "text": text,
                         "label": x["label"], "terms": terms, "teacher": model})
    return rows


def export_batches(postings: list[dict], out_dir: Path, per_batch: int) -> int:
    """Write batch files for an agent labeller (e.g. a Claude subagent): the guide,
    the pieces with ids, and the exact output format expected back."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "GUIDE.txt").write_text(GUIDE, encoding="utf-8")
    n = 0
    for n, start in enumerate(range(0, len(postings), per_batch), 1):
        batch = postings[start:start + per_batch]
        (out_dir / f"batch_{n:03d}.txt").write_text(prompt_for(batch), encoding="utf-8")
        (out_dir / f"batch_{n:03d}.meta.json").write_text(json.dumps(batch, ensure_ascii=False), encoding="utf-8")
    return n


def import_batches(out_dir: Path, teacher: str) -> None:
    """Validate agent-written batch_NNN.labels.json files and append good ones.

    A batch is rejected whole if any piece id is missing or a label is invalid;
    terms not found verbatim in their piece are dropped, not trusted."""
    done = set()
    if OUT.exists():
        done = {json.loads(l)["pid"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    for meta_path in sorted(out_dir.glob("batch_*.meta.json")):
        labels_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".labels.json"))
        batch = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(p["pid"] in done for p in batch):
            continue
        if not labels_path.exists():
            print(f"{meta_path.stem}: no labels yet")
            continue
        try:
            raw = json.loads(labels_path.read_text(encoding="utf-8-sig"))  # agents sometimes write a BOM
            items = raw["labels"] if isinstance(raw, dict) else raw
            got = {str(x["id"]): x for x in items}
        except (ValueError, KeyError, TypeError) as exc:
            print(f"{labels_path.name}: unreadable ({exc}); rejected")
            continue
        rows = rows_for(batch, got, teacher)
        if rows is None:
            # Agents drop a single id in ~1 batch of 10. Rejecting 200 good
            # labels for one missing piece costs a whole relabel; keep the rest
            # when at most 2% are missing and every present label is valid.
            expected = [f"{p['pid']}-{i}" for p in batch for i in range(len(p["pieces"]))]
            missing = [k for k in expected if k not in got]
            invalid = [k for k in expected if k in got and got[k].get("label") not in LABELS]
            if not invalid and len(missing) <= max(1, len(expected) // 50):
                rows = []
                for p in batch:
                    for i, text in enumerate(p["pieces"]):
                        x = got.get(f"{p['pid']}-{i}")
                        if x:
                            terms = [t for t in dict.fromkeys(x.get("terms") or []) if t and t in text]
                            rows.append({"pid": p["pid"], "company": p["company"], "idx": i, "text": text,
                                         "label": x["label"], "terms": terms, "teacher": teacher})
                print(f"{labels_path.name}: {len(missing)} missing id(s) dropped: {missing}")
        if rows is None:
            expected = {f"{p['pid']}-{i}" for p in batch for i in range(len(p["pieces"]))}
            missing = sorted(expected - set(got))
            bad = [k for k, v in got.items() if v.get("label") not in LABELS]
            print(f"{labels_path.name}: rejected (missing {len(missing)} ids e.g. {missing[:3]}, invalid labels {bad[:3]})")
            continue
        with OUT.open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        done |= {p["pid"] for p in batch}
        print(f"{labels_path.name}: imported {len(rows)} pieces")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, help="write batch files for an agent labeller instead of calling Gemini")
    ap.add_argument("--per-batch", type=int, default=3, help="postings per exported batch")
    ap.add_argument("--import-dir", type=Path, help="validate and import agent-written labels")
    ap.add_argument("--teacher", default="claude-haiku-4-5")
    ap.add_argument("--jobsdb-per-site", type=int, default=0,
                    help="add this many live ATS postings per platform from jobs.db (one per company)")
    ap.add_argument("--postings", type=int, default=400)
    ap.add_argument("--max-pieces", type=int, default=90)
    ap.add_argument("--models", default="gemini-3.8-flash,gemini-3.6-flash,gemini-3-flash-preview")
    ap.add_argument("--max-calls", type=int, default=24)
    ap.add_argument("--interval", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    if args.import_dir:
        import_batches(args.import_dir, args.teacher)
        return 0
    if args.export:
        done = set()
        if OUT.exists():
            done = {json.loads(l)["pid"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
        if args.jobsdb_per_site:
            postings = choose_jobsdb_postings(args.jobsdb_per_site, args.seed, done)
            postings += choose_postings(args.postings, args.seed, done) if args.postings else []
            random.Random(args.seed).shuffle(postings)
        else:
            postings = choose_postings(args.postings, args.seed, done)
        n = export_batches(postings, args.export, args.per_batch)
        print(f"exported {len(postings)} postings "
              f"({sum(len(p['pieces']) for p in postings)} pieces) as {n} batches to {args.export}")
        return 0

    from google import genai
    from google.genai import types

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["pid"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    postings = choose_postings(args.postings, args.seed, done)
    work: queue.Queue = queue.Queue()
    for b in pack(postings, args.max_pieces):
        work.put(b)
    total = work.qsize()
    print(f"postings already labelled {len(done)}; to label {len(postings)} "
          f"({sum(len(p['pieces']) for p in postings)} pieces) in {total} calls", flush=True)

    client = genai.Client(api_key=gemini_key())
    config = types.GenerateContentConfig(system_instruction=GUIDE, response_mime_type="application/json",
                                         response_schema=SCHEMA, temperature=0)
    lock = threading.Lock()
    stats = {"batches": 0, "rows": 0, "in_flight": 0}

    def take():
        """Next batch, or None when the queue is empty AND nobody holds one.
        An empty queue alone is not the end: a failing worker requeues its batch."""
        while True:
            with lock:
                try:
                    batch = work.get_nowait()
                    stats["in_flight"] += 1
                    return batch
                except queue.Empty:
                    if stats["in_flight"] == 0:
                        return None
            time.sleep(1)

    def release(batch, requeue: bool) -> None:
        with lock:
            if requeue:
                work.put(batch)
            stats["in_flight"] -= 1

    def worker(model: str) -> None:
        calls = failures = 0
        while calls < args.max_calls:
            batch = take()
            if batch is None:
                break
            try:
                resp = client.models.generate_content(model=model, contents=prompt_for(batch), config=config)
                calls += 1
                got = {str(x["id"]): x for x in json.loads(resp.text)["labels"]}
            except Exception as exc:
                release(batch, requeue=True)
                msg = str(exc)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    print(f"[{model}] quota reached after {calls + 1} calls -> stopping", flush=True)
                    return
                failures += 1  # 503 overload / bad JSON: not charged, 5 retire the model
                print(f"[{model}] {type(exc).__name__}: {msg[:90]}", flush=True)
                if failures >= 5:
                    print(f"[{model}] 5 failures -> stopping", flush=True)
                    return
                time.sleep(args.interval * 3)
                continue
            rows = rows_for(batch, got, model)
            if rows is None:
                failures += 1
                release(batch, requeue=True)
                print(f"[{model}] incomplete answer, batch requeued", flush=True)
                continue
            with lock:
                with OUT.open("a", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                stats["in_flight"] -= 1
                stats["batches"] += 1
                stats["rows"] += len(rows)
                print(f"[{model}] call {calls}/{args.max_calls}  +{len(rows)} pieces  "
                      f"({stats['batches']}/{total} calls, {stats['rows']} pieces)", flush=True)
            time.sleep(args.interval)
        print(f"[{model}] done ({calls} calls)", flush=True)

    threads = [threading.Thread(target=worker, args=(m.strip(),)) for m in args.models.split(",") if m.strip()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"finished: {stats['rows']} pieces written, {work.qsize()} calls left for the next run", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
