"""Benchmark the posting boilerplate filter at several confidence thresholds.

This script is intentionally small and conservative:
- it reads labeled posting pieces from the repo's lab corpus
- it checks whether the live classifier artifact is available
- if available, it runs the classifier at several thresholds and reports the
  removal rate for labeled boilerplate versus the false-positive rate on
  non-boilerplate pieces
- if unavailable, it exits with a clear message and leaves the repo in a
  non-claiming state

Usage:
    python scripts/benchmark_posting_classifier.py --thresholds 0.99,0.98,0.95,0.90
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from services.resumes.posting_classifier import load_posting_classifier, retain_resume_relevant_text  # noqa: E402

LABELS_PATH = ROOT / ".resume_lab" / "boilerplate" / "pieces.jsonl"


def _read_labeled_pieces(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _sample_rows(limit: int | None = None) -> list[dict]:
    rows = _read_labeled_pieces(LABELS_PATH)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def _compute_predictions(
    rows: list[dict],
    loaded: tuple,
    device: Any,
    batch_size: int = 128,
) -> list[tuple[str, str, float]]:
    """Compute (gold_label, pred_label, p_boilerplate) for all rows in batches."""
    import torch
    model, tokenizer, labels, max_len = loaded
    bp_idx = labels.index("BOILERPLATE")
    model = model.to(device)
    model.eval()

    results: list[tuple[str, str, float]] = []
    texts = [str(r.get("text") or "").strip() for r in rows]
    gold_labels = [str(r.get("label") or "").strip().upper() for r in rows]

    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk_texts = texts[start : start + batch_size]
            chunk_golds = gold_labels[start : start + batch_size]
            encoded = tokenizer(
                chunk_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            res = model(**encoded)
            logits = res[0] if isinstance(res, tuple) else res
            probs = torch.softmax(logits, dim=-1)
            pred_indices = probs.argmax(-1).cpu().tolist()
            bp_probs = probs[:, bp_idx].cpu().tolist()

            for gold, pred_idx, bp_p in zip(chunk_golds, pred_indices, bp_probs):
                results.append((gold, labels[pred_idx], bp_p))
    return results


def _eval_threshold(
    predictions: list[tuple[str, str, float]],
    threshold: float,
) -> dict[str, float | int]:
    removed_boil = 0
    total_boil = 0
    removed_non_boil = 0
    total_non_boil = 0
    for gold, pred_label, bp_p in predictions:
        is_removed = (pred_label == "BOILERPLATE" and bp_p >= threshold)
        if gold == "BOILERPLATE":
            total_boil += 1
            if is_removed:
                removed_boil += 1
        else:
            total_non_boil += 1
            if is_removed:
                removed_non_boil += 1

    return {
        "threshold": threshold,
        "total_boilerplate": total_boil,
        "removed_boilerplate": removed_boil,
        "boilerplate_removal_rate": (removed_boil / total_boil) if total_boil else 0.0,
        "total_non_boilerplate": total_non_boil,
        "removed_non_boilerplate": removed_non_boil,
        "false_positive_rate": (removed_non_boil / total_non_boil) if total_non_boil else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", default="0.99,0.98,0.95,0.90")
    ap.add_argument("--limit", type=int, default=0, help="0 means all rows")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    rows = _sample_rows(args.limit if args.limit > 0 else None)
    if not rows:
        print("No labeled posting pieces found at .resume_lab/boilerplate/pieces.jsonl", flush=True)
        return 0

    loaded = load_posting_classifier()
    if loaded is None:
        print("Classifier artifact is not available in this workspace; benchmark skipped.", flush=True)
        print(f"Expected a model in {ROOT / '.resume_lab' / 'boilerplate' / 'posting_encoder'} or {ROOT / 'models' / 'posting_encoder'}", flush=True)
        return 0

    import torch
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    thresholds = [float(raw.strip()) for raw in args.thresholds.split(",")]

    print(f"Benchmarking {len(rows)} labeled pieces on {device}; model artifact present", flush=True)
    predictions = _compute_predictions(rows, loaded, device=device, batch_size=args.batch_size)

    print(f"{'threshold':>9}  {'boilerplate_remove':>18}  {'false_positive':>15}  {'boilerplate_total':>17}  {'non_boiler_total':>17}", flush=True)
    for thr in thresholds:
        stats = _eval_threshold(predictions, thr)
        print(
            f"{thr:>9.2f}  "
            f"{stats['boilerplate_removal_rate']:>18.3%}  "
            f"{stats['false_positive_rate']:>15.3%}  "
            f"{stats['total_boilerplate']:>17}  "
            f"{stats['total_non_boilerplate']:>17}",
            flush=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

