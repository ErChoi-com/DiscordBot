"""Compile and harmonize seniority labels from all subagent batches.

Applies user directives:
1. Merge 'coop' into 'intern'
2. Eliminate 'campus': map student placements to 'intern' and rotational/grad programs to 'newgrad'
3. Validate 6-class ontology: intern, newgrad, junior, mid, senior, staff
4. Produce rebuilt_app/.resume_lab/seniority/titles.jsonl
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
BATCHES_DIR = ROOT / ".resume_lab" / "seniority" / "batches"
OUT_FILE = ROOT / ".resume_lab" / "seniority" / "titles.jsonl"

VALID_CLASSES = {"intern", "newgrad", "junior", "mid", "senior", "staff"}

def parse_prompts() -> dict[str, str]:
    """Parse id -> title from all prompt.txt files."""
    titles: dict[str, str] = {}
    pattern = re.compile(r"^\[([\w\-]+)\]\s*(.+)$")
    for prompt_file in BATCHES_DIR.glob("batch_*/prompt.txt"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                m = pattern.match(line)
                if m:
                    titles[m.group(1)] = m.group(2).strip()
    return titles

def map_label(raw_label: str, title: str) -> str:
    lbl = raw_label.strip().lower()
    if lbl == "coop":
        return "intern"
    if lbl == "campus":
        t_low = title.lower()
        if any(w in t_low for w in ["intern", "co-op", "coop", "student", "stagiaire", "placement"]):
            return "intern"
        return "newgrad"
    return lbl

def main():
    titles = parse_prompts()
    print(f"Loaded {len(titles)} titles from prompt files.")

    all_samples = []
    missing_batches = []
    
    for i in range(1, 5):
        batch_dir = BATCHES_DIR / f"batch_{i}"
        labels_file = batch_dir / "labels.json"
        if not labels_file.exists():
            missing_batches.append(f"batch_{i}")
            continue
        
        with open(labels_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            items = data if isinstance(data, list) else data.get("labels", [])
            for item in items:
                sid = item.get("id")
                raw_label = item.get("label", "")
                if sid in titles:
                    title = titles[sid]
                    final_label = map_label(raw_label, title)
                    if final_label not in VALID_CLASSES:
                        print(f"Warning: unrecognized class '{final_label}' for {sid}: {title}")
                        continue
                    all_samples.append({
                        "id": sid,
                        "title": title,
                        "label": final_label,
                        "raw_label": raw_label,
                    })

    print(f"Compiled {len(all_samples)} labeled samples.")
    if missing_batches:
        print(f"Pending batches: {', '.join(missing_batches)}")
        return

    # Deduplicate by id if needed
    seen = set()
    deduped = []
    for s in all_samples:
        if s["id"] not in seen:
            seen.add(s["id"])
            deduped.append(s)

    # Class balance stats
    counts = Counter(s["label"] for s in deduped)
    print("\nClass Distribution:")
    for cls in ["intern", "newgrad", "junior", "mid", "senior", "staff"]:
        print(f"  {cls:8s}: {counts[cls]:4d} ({counts[cls]/len(deduped)*100:5.1f}%)")

    # Save to jsonl
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for s in deduped:
            f.write(json.dumps(s) + "\n")

    print(f"\nSuccessfully wrote {len(deduped)} entries to {OUT_FILE}")

if __name__ == "__main__":
    main()

