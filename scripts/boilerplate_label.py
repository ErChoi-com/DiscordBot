"""Build a distilled training set for the boilerplate sentence classifier.

Splits every harvested posting description into sentences, samples them
(all cross-company repeats, all rule hits, a random slice of the rest) and
asks Gemini to label each one BOILERPLATE or CONTENT. Output is appended to
.resume_lab/boilerplate/labels.jsonl so an interrupted run resumes.

    python scripts/boilerplate_label.py --limit 6000
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / ".resume_lab" / "trials" / "corpus.json"
OUT_DIR = ROOT / ".resume_lab" / "boilerplate"
LABELS = OUT_DIR / "labels.jsonl"

GUIDE = """You label sentences taken from job postings for a resume tailoring tool.
The tool keeps CONTENT and deletes BOILERPLATE before a model reads the posting.

BOILERPLATE = anything that says nothing about the work or the skills needed:
- equal opportunity / non-discrimination / "without regard to" / diversity statements
- accommodation and accessibility notices
- AI or automated screening disclosures, privacy / data processing notices
- hiring-process text: "only those selected will be contacted", "thank you for your interest",
  "apply even if you don't meet every requirement", how to apply, recruiter agency notes
- salary / pay range / compensation philosophy, benefits, perks, PTO, wellness, stipends
- company promo: size, history, mission, values, culture, awards, "we are a leading..."
- legal/work authorization, background checks, export control
- scraper or job-board furniture: "Location: ...", "Seniority level", sign-up prompts, links

CONTENT = anything a candidate could tailor a resume to:
- duties, responsibilities, projects, what the team builds
- required or preferred skills, tools, technologies, experience, education, certifications
- soft-skill requirements ("strong communication skills") are CONTENT
- role-specific conditions that affect fit (shift work, travel, must commute, clearance needed)

When a sentence mixes both, label CONTENT if it names any skill, duty or requirement.

Also return "terms": every concrete term in the sentence that is useful for writing a
tailored resume, copied EXACTLY as it appears (same spelling and casing, a substring of the sentence):
- tools, software, languages, frameworks, libraries, platforms, cloud services, hardware, chips, instruments
- technical methods and standards (e.g. "unit testing", "FMEA", "ISO 13485", "PID control", "CAPA")
- certifications, licences, degrees and fields of study
- domain concepts a recruiter would scan for (e.g. "embedded systems", "financial modelling")
Do NOT include generic words ("team", "experience", "communication", "fast-paced"), company or
product-brand marketing names, locations, job titles, or anything in a BOILERPLATE sentence.
Return an empty list when there are none. Return one entry for every id."""

SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "label": {"type": "string", "enum": ["BOILERPLATE", "CONTENT"]},
                    "terms": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "label", "terms"],
            },
        }
    },
    "required": ["labels"],
}

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9•\-*(\"'])|\n+|\s*[•▪●◦]\s*")
_HEADER = re.compile(r"^\s*(?:Source site|Posting URL|Apply URL|Title|Company|Description)\s*:", re.I)
_RULE_HINT = re.compile(
    r"equal (?:employment )?opportunit|without regard|regardless of|accommodat|artificial intelligence"
    r"|\bAI\b|salary|benefit|only those|thank|privacy|inclusi|divers|we are an? |founded|culture",
    re.I,
)


def split_sentences(text: str) -> list[str]:
    out = []
    for piece in _SPLIT.split(text or ""):
        piece = re.sub(r"\s+", " ", piece).strip(" -*\t")
        if 25 <= len(piece) <= 600 and not _HEADER.match(piece):
            out.append(piece)
    return out


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


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


def sample(limit: int, seed: int) -> list[dict]:
    postings = json.loads(CORPUS.read_text(encoding="utf-8"))
    companies = collections.defaultdict(set)
    first: dict[str, dict] = {}
    for p in postings:
        company = (p.get("company") or "").lower()
        for s in split_sentences(p.get("description") or ""):
            n = norm(s)
            companies[n].add(company)
            first.setdefault(n, {"text": s, "company": p.get("company") or ""})
    rng = random.Random(seed)
    keys = list(first)
    rng.shuffle(keys)
    repeated = [k for k in keys if len(companies[k]) >= 2]
    hinted = [k for k in keys if k not in set(repeated) and _RULE_HINT.search(first[k]["text"])]
    taken = set(repeated) | set(hinted)
    rest = [k for k in keys if k not in taken]
    # Repeats and hints are boilerplate-heavy; the random slice keeps real
    # content well represented so the classifier learns both sides.
    budget_rest = max(limit // 2, limit - len(repeated) - len(hinted))
    chosen = (repeated + hinted)[: limit - min(budget_rest, len(rest))] + rest[:budget_rest]
    chosen = chosen[:limit]
    print(f"unique sentences {len(first)}  repeated {len(repeated)}  hinted {len(hinted)}  chosen {len(chosen)}")
    return [{"key": k, **first[k], "n_companies": len(companies[k])} for k in chosen]


def label_row(r: dict, x: dict, model: str) -> dict | None:
    if x.get("label") not in ("BOILERPLATE", "CONTENT"):
        return None
    # Keep only terms that occur verbatim: the token tagger needs exact
    # spans, and a paraphrased term has none.
    terms = [] if x["label"] == "BOILERPLATE" else [
        t for t in dict.fromkeys(x.get("terms") or []) if t and t in r["text"]
    ]
    return {**r, "label": x["label"], "terms": terms, "teacher": model}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=80)
    # One worker per model, in parallel. gemini-3.5-flash is left out on
    # purpose: it is the bot's primary resume model.
    ap.add_argument("--models", default="gemini-3.8-flash,gemini-3.6-flash,gemini-3-flash-preview")
    # Hard per-model call budget for this run, so labelling can never spend
    # more than one day's free quota even if the quota error never arrives.
    ap.add_argument("--max-calls", type=int, default=25)
    ap.add_argument("--interval", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import queue
    import threading

    from google import genai
    from google.genai import types

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if LABELS.exists():
        done = {json.loads(l)["key"] for l in LABELS.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in sample(args.limit, args.seed) if r["key"] not in done]
    work: queue.Queue = queue.Queue()
    for start in range(0, len(todo), args.batch):
        work.put(todo[start : start + args.batch])
    total_batches = work.qsize()
    print(f"already labelled {len(done)}  to label {len(todo)}  batches {total_batches}", flush=True)

    client = genai.Client(api_key=gemini_key())
    config = types.GenerateContentConfig(
        system_instruction=GUIDE, response_mime_type="application/json",
        response_schema=SCHEMA, temperature=0,
    )
    lock = threading.Lock()
    stats = {"rows": 0, "batches": 0}
    fh = LABELS.open("a", encoding="utf-8")

    def worker(model: str) -> None:
        calls = failures = 0
        while calls < args.max_calls:
            try:
                batch = work.get_nowait()
            except queue.Empty:
                break
            prompt = "\n".join(f"{i}: {r['text']}" for i, r in enumerate(batch))
            try:
                resp = client.models.generate_content(model=model, contents=prompt, config=config)
                got = {int(x["id"]): x for x in json.loads(resp.text)["labels"]}
                calls += 1
            except Exception as exc:
                work.put(batch)  # someone else (or this worker later) takes it
                msg = str(exc)
                # A 503 "high demand" is refused before it spends quota, so it
                # is not charged to the budget; five of them retire the model.
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    calls += 1
                    print(f"[{model}] quota reached after {calls} calls -> stopping this model", flush=True)
                    return
                failures += 1
                print(f"[{model}] {type(exc).__name__}: {msg[:100]}", flush=True)
                if failures >= 5:
                    print(f"[{model}] 5 failures -> stopping this model", flush=True)
                    return
                time.sleep(args.interval * 3)
                continue
            rows = [row for i, r in enumerate(batch) if (row := label_row(r, got.get(i) or {}, model))]
            with lock:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                stats["rows"] += len(rows)
                stats["batches"] += 1
                print(f"[{model}] call {calls}/{args.max_calls}  kept {len(rows)}/{len(batch)}  "
                      f"progress {stats['batches']}/{total_batches} batches, {stats['rows']} rows", flush=True)
            time.sleep(args.interval)
        print(f"[{model}] done ({calls} calls)", flush=True)

    threads = [threading.Thread(target=worker, args=(m.strip(),)) for m in args.models.split(",") if m.strip()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    fh.close()
    print(f"finished: {stats['rows']} new rows, {work.qsize()} batches left unlabelled", flush=True)


if __name__ == "__main__":
    main()
