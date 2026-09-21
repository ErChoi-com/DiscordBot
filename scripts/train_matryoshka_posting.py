"""Train Matryoshka posting & boilerplate piece classifier across all dimensions.

Jointly optimizes across multiple embedding dimensionalities:
  [768, 512, 384, 256, 128, 64]

Trained on the 4-class posting structure dataset:
  - SKILL_DUTY: Core responsibilities, technical skills, daily duties
  - BOILERPLATE: EEO statements, 401k/benefits, legal notices, generic filler
  - ROLE_FACTS: Compensation, location, work terms, schedule, dates
  - COMPANY_CONTEXT: Company mission, history, organizational background

Saves master converted weights W in R^{4 x 768} (and bias b in R^4) to
models/posting_encoder/master_matryoshka_weights.pt.

Usage:
  python scripts/train_matryoshka_posting.py --max-samples 3200 --epochs 25
  python scripts/train_matryoshka_posting.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from services.semantic.engine import SemanticEngine, get_semantic_engine
from services.semantic.prompts import (
    DEFAULT_EMBEDDING_DIM,
    MASTER_EMBEDDING_DIM,
    MATRYOSHKA_DIMS,
    TASK_CLASSIFICATION,
)

TRAIN_SPLIT_PATH = ROOT / ".resume_lab" / "boilerplate" / "train_split.jsonl"
TEST_SPLIT_PATH = ROOT / ".resume_lab" / "boilerplate" / "test_split.jsonl"
HARD_NEGATIVES_DIR = ROOT / ".resume_lab" / "agent_batches" / "hard_negatives"
FLAGS_PATH = ROOT / ".resume_lab" / "skill_duty_flags.json"
CACHE_PATH = ROOT / ".resume_lab" / "boilerplate" / "nomic_embeddings_cache_768.pt"
OUT_DIR = ROOT / "models" / "posting_encoder"

CLASSES = ["SKILL_DUTY", "BOILERPLATE", "ROLE_FACTS", "COMPANY_CONTEXT"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for i, c in enumerate(CLASSES)}
BP_ID = LABEL2ID["BOILERPLATE"]


def load_disk_cache(cache_path: Path = CACHE_PATH) -> dict[str, torch.Tensor]:
    """Load precomputed 768-d embeddings from persistent disk cache."""
    if cache_path.is_file():
        try:
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                print(f"[Embedding Cache] Loaded {len(data)} cached embeddings from {cache_path}")
                return data
        except Exception as exc:
            print(f"[Embedding Cache] Failed to load cache from {cache_path}: {exc}")
    return {}


def save_disk_cache(cache: dict[str, torch.Tensor], cache_path: Path = CACHE_PATH) -> None:
    """Safely persist embedding cache to disk using atomic rename."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    try:
        torch.save(cache, tmp_path)
        if tmp_path.exists():
            tmp_path.replace(cache_path)
    except Exception as exc:
        print(f"[Embedding Cache] Failed to save cache: {exc}")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MatryoshkaPieceClassifier(nn.Module):
    """Linear classifier head trained with Matryoshka multi-dimension slicing."""

    def __init__(self, master_dim: int = MASTER_EMBEDDING_DIM, num_classes: int = len(CLASSES)):
        super().__init__()
        self.master_dim = master_dim
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.empty(num_classes, master_dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward_dim(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Forward pass for dimension d with unit L2 normalization."""
        x_d = x[:, :dim]
        x_d = nn.functional.normalize(x_d, p=2, dim=-1)
        w_d = self.weight[:, :dim]
        return torch.matmul(x_d, w_d.t()) + self.bias


class AsymmetricMatryoshkaLoss(nn.Module):
    """Joint Matryoshka multi-dimension loss with asymmetric content protection penalty.

    Applies extra penalty when true content (SKILL_DUTY, ROLE_FACTS) is misclassified
    as BOILERPLATE to strictly protect ATS tailoring.
    """

    def __init__(
        self,
        matryoshka_dims: tuple[int, ...] = MATRYOSHKA_DIMS,
        class_weights: torch.Tensor | None = None,
        asym_penalty: float = 2.0,
    ):
        super().__init__()
        self.matryoshka_dims = matryoshka_dims
        self.class_weights = class_weights
        self.asym_penalty = asym_penalty

    def forward(
        self,
        classifier: MatryoshkaPieceClassifier,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, float]]:
        total_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        losses_by_dim: dict[int, float] = {}

        for dim in self.matryoshka_dims:
            logits_d = classifier.forward_dim(embeddings, dim)
            ce = nn.functional.cross_entropy(logits_d, labels, weight=self.class_weights, reduction="none")

            # Asymmetric penalty for false positive boilerplate
            preds = logits_d.argmax(dim=-1)
            is_content = (labels == LABEL2ID["SKILL_DUTY"]) | (labels == LABEL2ID["ROLE_FACTS"])
            is_bp_pred = (preds == BP_ID)
            weighted_ce = torch.where(is_content & is_bp_pred, ce * self.asym_penalty, ce)
            dim_loss = weighted_ce.mean()

            total_loss = total_loss + dim_loss
            losses_by_dim[dim] = float(dim_loss.item())

        return total_loss, losses_by_dim


def load_all_posting_datasets(
    train_path: Path = TRAIN_SPLIT_PATH,
    test_path: Path = TEST_SPLIT_PATH,
    hard_neg_dir: Path = HARD_NEGATIVES_DIR,
    flags_path: Path = FLAGS_PATH,
    max_samples: int = 0,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load ALL possible data from train split, test split, adversarial hard negatives, and flags."""
    rng = random.Random(seed)
    train_pieces: dict[str, str] = {}  # text -> label

    # 1. Primary training split (deduplicating exact texts to avoid overfitting)
    if train_path.is_file():
        with train_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    lbl = row.get("label")
                    txt = row.get("text", "").strip()
                    if lbl in LABEL2ID and txt and len(txt) >= 10:
                        train_pieces[txt] = lbl
                except Exception:
                    continue
        print(f"Loaded {len(train_pieces)} unique training pieces from {train_path.name}")

    # 2. Hard negatives (adversarial curated pieces from batch_1 and batch_2)
    cand_path = hard_neg_dir / "candidates.json"
    b1_path = hard_neg_dir / "batch_1" / "labels.json"
    b2_path = hard_neg_dir / "batch_2" / "labels.json"

    hn_count = 0
    if cand_path.is_file() and b1_path.is_file() and b2_path.is_file():
        try:
            with cand_path.open("r", encoding="utf-8") as f:
                cands = json.load(f)
            with b1_path.open("r", encoding="utf-8") as f:
                b1_labels = json.load(f).get("labels", [])
            with b2_path.open("r", encoding="utf-8") as f:
                b2_labels = json.load(f).get("labels", [])

            all_labels = b1_labels + b2_labels
            for i, hn in enumerate(all_labels):
                if i < len(cands):
                    txt = cands[i].get("text", "").strip()
                    lbl = hn.get("label")
                    if lbl in LABEL2ID and txt and len(txt) >= 10:
                        train_pieces[txt] = lbl  # Prized adversarial labels
                        hn_count += 1
            print(f"Loaded {hn_count} adversarial hard negatives into training pool")
        except Exception as exc:
            print(f"[Warning] Failed loading hard negatives: {exc}")

    # 3. Subtle edge cases from skill_duty_flags
    if flags_path.is_file():
        try:
            with flags_path.open("r", encoding="utf-8") as f:
                flags = json.load(f)
            flags_added = 0
            for item in flags:
                txt = item.get("text", "").strip()
                if txt and len(txt) >= 10 and txt not in train_pieces:
                    # Flagged duties are true SKILL_DUTY pieces
                    train_pieces[txt] = "SKILL_DUTY"
                    flags_added += 1
            if flags_added > 0:
                print(f"Loaded {flags_added} edge-case duty pieces from flags")
        except Exception as exc:
            print(f"[Warning] Failed loading flags: {exc}")

    # 4. Disjoint holdout test split
    test_pieces: dict[str, str] = {}
    if test_path.is_file():
        with test_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    lbl = row.get("label")
                    txt = row.get("text", "").strip()
                    if lbl in LABEL2ID and txt and len(txt) >= 10:
                        if txt not in train_pieces:
                            test_pieces[txt] = lbl
                except Exception:
                    continue
        print(f"Loaded {len(test_pieces)} disjoint holdout test pieces from {test_path.name}")

    # Convert to list of dicts
    train_items = [{"text": t, "label": l} for t, l in train_pieces.items()]
    test_items = [{"text": t, "label": l} for t, l in test_pieces.items()]

    rng.shuffle(train_items)
    rng.shuffle(test_items)

    if max_samples > 0:
        by_class: dict[str, list[dict[str, Any]]] = {c: [] for c in CLASSES}
        for item in train_items:
            by_class[item["label"]].append(item)
        per_class = max_samples // len(CLASSES)
        sampled_train = []
        for c in CLASSES:
            sampled_train.extend(by_class[c][:per_class])
        rng.shuffle(sampled_train)
        train_items = sampled_train
        test_items = test_items[: max_samples // 4]
        print(f"Stratified sampling applied: {len(train_items)} train, {len(test_items)} test")

    return train_items, test_items


def compute_piece_metrics(preds: list[int], trues: list[int]) -> dict[str, Any]:
    """Compute precision, recall, F1, and critical false positive rate."""
    n = len(trues)
    if n == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "content_loss_pct": 0.0, "per_class": {}}

    acc = sum(p == t for p, t in zip(preds, trues)) / n
    per_class = {}
    f1s = []

    for cls_name, cls_id in LABEL2ID.items():
        tp = sum(p == cls_id and t == cls_id for p, t in zip(preds, trues))
        fp = sum(p == cls_id and t != cls_id for p, t in zip(preds, trues))
        fn = sum(p != cls_id and t == cls_id for p, t in zip(preds, trues))
        supp = sum(t == cls_id for t in trues)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        if supp > 0:
            f1s.append(f1)
        per_class[cls_name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": supp,
        }

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    # Content loss: true SKILL_DUTY or ROLE_FACTS dropped as BOILERPLATE
    content_samples = [i for i, t in enumerate(trues) if t in (LABEL2ID["SKILL_DUTY"], LABEL2ID["ROLE_FACTS"])]
    dropped_content = sum(1 for i in content_samples if preds[i] == BP_ID)
    content_loss_pct = (dropped_content / len(content_samples)) * 100 if content_samples else 0.0

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "content_loss_pct": round(content_loss_pct, 2),
        "per_class": per_class,
    }


def precompute_embeddings(
    items: list[dict[str, Any]],
    engine: SemanticEngine,
    batch_size: int = 128,
    cache_path: Path = CACHE_PATH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute full 768-d master embeddings with persistent disk-backed cache."""
    disk_cache = load_disk_cache(cache_path)
    initial_cached_count = len(disk_cache)

    texts = [item["text"] for item in items]
    labels = [LABEL2ID[item["label"]] for item in items]

    # Check cache for each item
    missing_indices: list[int] = []
    missing_texts: list[str] = []
    missing_hashes: list[str] = []
    text_hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]

    result_tensors: list[torch.Tensor | None] = [None] * len(texts)

    for i, (t, h) in enumerate(zip(texts, text_hashes)):
        if h in disk_cache:
            result_tensors[i] = disk_cache[h]
        else:
            missing_indices.append(i)
            missing_texts.append(t)
            missing_hashes.append(h)

    print(
        f"[Embeddings] Total items: {len(texts)} | Cache hits: {len(texts) - len(missing_texts)} | "
        f"Uncached to compute: {len(missing_texts)}"
    )

    if missing_texts:
        t0 = time.perf_counter()
        newly_computed = 0
        last_saved = 0

        for b_start in range(0, len(missing_texts), batch_size):
            b_texts = missing_texts[b_start : b_start + batch_size]
            b_indices = missing_indices[b_start : b_start + batch_size]
            b_hashes = missing_hashes[b_start : b_start + batch_size]

            vecs = engine.encode(
                b_texts,
                dim=MASTER_EMBEDDING_DIM,
                task_type=TASK_CLASSIFICATION,
                normalize=True,
                batch_size=batch_size,
            )
            if vecs is None:
                raise RuntimeError(f"Embedding generation failed for batch {b_start}..{b_start + len(b_texts)}")

            for idx, h, vec in zip(b_indices, b_hashes, vecs):
                tensor_v = torch.tensor(vec, dtype=torch.float32)
                disk_cache[h] = tensor_v
                result_tensors[idx] = tensor_v
                newly_computed += 1

            # Periodically save disk cache every 500 newly computed embeddings
            if (newly_computed - last_saved >= 500) or newly_computed == len(missing_texts):
                save_disk_cache(disk_cache, cache_path)
                last_saved = newly_computed
                elapsed = time.perf_counter() - t0
                speed = newly_computed / max(0.001, elapsed)
                print(
                    f"  Computed {newly_computed}/{len(missing_texts)} embeddings "
                    f"({speed:.1f} items/s) - Disk cache saved ({len(disk_cache)} total)",
                    flush=True,
                )

        # Final save of disk cache
        if len(disk_cache) > initial_cached_count:
            save_disk_cache(disk_cache, cache_path)

    embeddings_t = torch.stack(result_tensors, dim=0)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return embeddings_t, labels_t


def evaluate_matryoshka_dimensions(
    classifier: MatryoshkaPieceClassifier,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    dims: tuple[int, ...] = MATRYOSHKA_DIMS,
) -> dict[int, dict[str, Any]]:
    """Evaluate performance across every individual Matryoshka dimension."""
    classifier.eval()
    results = {}
    with torch.no_grad():
        trues = val_labels.cpu().tolist()
        for dim in dims:
            logits = classifier.forward_dim(val_embeddings, dim)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            metrics = compute_piece_metrics(preds, trues)
            results[dim] = metrics
    return results


def print_comparison_table(results: dict[int, dict[str, Any]]) -> None:
    """Print ASCII comparison table across all Matryoshka dimensions."""
    print("\n" + "=" * 90)
    print("POSTING PIECE CLASSIFIER: MATRYOSHKA MULTI-DIMENSION COMPARISON")
    print("=" * 90)
    print(f"{'Dimension':<10} | {'Accuracy':<9} | {'Macro F1':<9} | {'BP F1':<8} | {'Content Loss':<13} | {'Compression'}")
    print("-" * 90)

    base_dim = MASTER_EMBEDDING_DIM
    for dim in MATRYOSHKA_DIMS:
        res = results.get(dim, {})
        acc = res.get("accuracy", 0.0)
        f1 = res.get("macro_f1", 0.0)
        bp_f1 = res.get("per_class", {}).get("BOILERPLATE", {}).get("f1", 0.0)
        cl = res.get("content_loss_pct", 0.0)
        compression = f"{(1.0 - dim / base_dim) * 100:.1f}% reduction" if dim < base_dim else "baseline"
        print(f"{dim:<10} | {acc * 100:5.2f}%   | {f1:7.4f}   | {bp_f1:6.4f} | {cl:5.2f}%        | {compression}")
    print("=" * 90 + "\n")


def train_matryoshka_posting(
    max_samples: int = 0,
    epochs: int = 30,
    batch_size: int = 64,
    embed_batch_size: int = 32,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    dry_run: bool = False,
    out_dir: Path = OUT_DIR,
    seed: int = 42,
) -> dict[str, Any]:
    """Train Matryoshka piece classifier and export master weights."""
    set_seed(seed)
    print("Loading datasets across all sources (train split, test split, hard negatives, flags)...")
    train_items, val_items = load_all_posting_datasets(
        train_path=TRAIN_SPLIT_PATH,
        test_path=TEST_SPLIT_PATH,
        hard_neg_dir=HARD_NEGATIVES_DIR,
        flags_path=FLAGS_PATH,
        max_samples=max_samples,
        seed=seed,
    )
    print(
        f"Dataset summary: {len(train_items)} train pieces, {len(val_items)} holdout test pieces "
        f"across {len(CLASSES)} classes."
    )

    if dry_run:
        train_items = train_items[:32]
        val_items = val_items[:16]
        epochs = 2
        print(f"[DRY-RUN] Truncated dataset: {len(train_items)} train, {len(val_items)} val.")

    # Calculate class weights for inverse frequency
    counts = Counter(item["label"] for item in train_items)
    total_samples = len(train_items)
    weights = [
        total_samples / (len(CLASSES) * max(1, counts.get(cls_name, 1)))
        for cls_name in CLASSES
    ]
    class_weights_t = torch.tensor(weights, dtype=torch.float32)

    # Initialize SemanticEngine and precompute embeddings with persistent disk caching
    engine = get_semantic_engine()
    train_x, train_y = precompute_embeddings(train_items, engine, batch_size=embed_batch_size)
    val_x, val_y = precompute_embeddings(val_items, engine, batch_size=embed_batch_size)

    # Dataloader
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model & Loss: strictly single linear matrix W in R^{4 x 768}, b in R^4
    classifier = MatryoshkaPieceClassifier(master_dim=MASTER_EMBEDDING_DIM, num_classes=len(CLASSES))
    asym_loss_fn = AsymmetricMatryoshkaLoss(
        matryoshka_dims=MATRYOSHKA_DIMS,
        class_weights=class_weights_t,
        asym_penalty=2.0,
    )

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\nStarting Matryoshka posting structure training...")
    print(f"Jointly optimizing across dimensions: {list(MATRYOSHKA_DIMS)}")

    best_val_f1 = 0.0
    best_results: dict[int, dict[str, Any]] = {}

    for epoch in range(1, epochs + 1):
        classifier.train()
        total_epoch_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss, _ = asym_loss_fn(classifier, batch_x, batch_y)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_epoch_loss / max(1, num_batches)

        results = evaluate_matryoshka_dimensions(classifier, val_x, val_y)
        prod_f1 = results[DEFAULT_EMBEDDING_DIM]["macro_f1"]
        prod_acc = results[DEFAULT_EMBEDDING_DIM]["accuracy"]

        if prod_f1 > best_val_f1:
            best_val_f1 = prod_f1
            best_results = results

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(
                f"Epoch {epoch:2d}/{epochs:2d} | "
                f"Loss: {avg_loss:.4f} | "
                f"384-d Acc: {prod_acc * 100:.2f}%, F1: {prod_f1:.4f} | "
                f"768-d Acc: {results[768]['accuracy'] * 100:.2f}% | "
                f"128-d Acc: {results[128]['accuracy'] * 100:.2f}%"
            )

    print("\nTraining completed!")
    print_comparison_table(best_results)

    # Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "master_matryoshka_weights.pt"
    meta_path = out_dir / "matryoshka_meta.json"

    torch.save(
        {
            "weight": classifier.weight.detach().cpu(),
            "bias": classifier.bias.detach().cpu(),
            "classes": CLASSES,
            "matryoshka_dims": MATRYOSHKA_DIMS,
            "master_dim": MASTER_EMBEDDING_DIM,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        weights_path,
    )
    print(f"Saved master Matryoshka piece weights to {weights_path}")

    meta_data = {
        "labels": CLASSES,
        "backbone": "nomic-ai/nomic-embed-text-v1.5",
        "format": "unquantized-float16-gguf",
        "master_dim": MASTER_EMBEDDING_DIM,
        "default_dim": DEFAULT_EMBEDDING_DIM,
        "supported_dims": list(MATRYOSHKA_DIMS),
        "num_classes": len(CLASSES),
        "trained_samples": len(train_items),
        "val_samples": len(val_items),
        "dry_run": dry_run,
        "metrics_by_dimension": best_results,
        "production_384d_metrics": best_results.get(DEFAULT_EMBEDDING_DIM, {}),
        "full_768d_metrics": best_results.get(MASTER_EMBEDDING_DIM, {}),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)
    print(f"Saved multi-dimension metadata to {meta_path}")

    return meta_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Matryoshka posting & boilerplate piece classifier")
    parser.add_argument("--max-samples", type=int, default=0, help="Max total samples (0 for full dataset)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--embed-batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--dry-run", action="store_true", help="Quick dry run with mini dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_matryoshka_posting(
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_batch_size=args.embed_batch_size,
        lr=args.lr,
        dry_run=args.dry_run,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
