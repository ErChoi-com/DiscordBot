"""Train neural seniority classifier using MatryoshkaLoss on labeled title corpus.

Jointly optimizes across multiple embedding dimensionalities:
  [768, 512, 384, 256, 128, 64]

Uses Nomic Embed Text v1.5 representations and produces a master weight
matrix W in R^{6 x 768} (and bias b in R^6) that can be dynamically sliced
in memory (W[:, :d]) to match any swapped runtime dimensionality.

Usage:
  python scripts/train_matryoshka_seniority.py --epochs 25 --lr 1e-3
  python scripts/train_matryoshka_seniority.py --dry-run
"""
from __future__ import annotations

import argparse
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

# Optional SentenceTransformers MatryoshkaLoss
try:
    from sentence_transformers.losses import MatryoshkaLoss as STMatryoshkaLoss
except Exception:
    STMatryoshkaLoss = None

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

DATA_PATH = ROOT / ".resume_lab" / "seniority" / "titles.jsonl"
OUT_DIR = ROOT / "models" / "seniority_encoder"

CLASSES = ["intern", "newgrad", "junior", "mid", "senior", "staff"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for i, c in enumerate(CLASSES)}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MatryoshkaLinearClassifier(nn.Module):
    """Linear classifier head trained with Matryoshka multi-dimensional slicing."""

    def __init__(self, master_dim: int = MASTER_EMBEDDING_DIM, num_classes: int = len(CLASSES)):
        super().__init__()
        self.master_dim = master_dim
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.empty(num_classes, master_dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward_dim(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Forward pass for a specific Matryoshka dimension d.

        x: (batch_size, dim) or (batch_size, master_dim)
        """
        x_d = x[:, :dim]
        # Re-normalize sliced representation onto unit sphere
        x_d = nn.functional.normalize(x_d, p=2, dim=-1)
        w_d = self.weight[:, :dim]
        return torch.matmul(x_d, w_d.t()) + self.bias

    def get_sliced_weights(self, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Dynamically slice master weights down to dimension d."""
        return self.weight[:, :dim].detach().clone(), self.bias.detach().clone()


class MatryoshkaLossModule(nn.Module):
    """Joint Matryoshka multi-dimension loss module.

    Computes loss at each dimension d in matryoshka_dims and combines them:
      L_total = sum_{d} weight_d * CrossEntropy(logits_d, labels)
    """

    def __init__(
        self,
        matryoshka_dims: tuple[int, ...] = MATRYOSHKA_DIMS,
        class_weights: torch.Tensor | None = None,
        dim_weights: dict[int, float] | None = None,
    ):
        super().__init__()
        self.matryoshka_dims = matryoshka_dims
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        # Default equal weighting or custom per-dimension weights
        self.dim_weights = dim_weights or {d: 1.0 for d in matryoshka_dims}

    def forward(
        self,
        classifier: MatryoshkaLinearClassifier,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, float]]:
        """Compute joint Matryoshka multi-dimension loss."""
        total_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        losses_by_dim: dict[int, float] = {}

        for dim in self.matryoshka_dims:
            logits_d = classifier.forward_dim(embeddings, dim)
            loss_d = self.criterion(logits_d, labels)
            w_d = self.dim_weights.get(dim, 1.0)
            total_loss = total_loss + w_d * loss_d
            losses_by_dim[dim] = float(loss_d.item())

        return total_loss, losses_by_dim


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load JSONL dataset."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path}")
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                label = row.get("label")
                title = row.get("title", "").strip()
                if label in LABEL2ID and title:
                    items.append({"title": title, "label": label, "id": row.get("id", "")})
            except Exception:
                continue
    return items


def stratified_split(
    items: list[dict[str, Any]], test_ratio: float = 0.2, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified train/val split preserving class balance."""
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_class.setdefault(item["label"], []).append(item)

    train_items, test_items = [], []
    for cls, lst in by_class.items():
        rng.shuffle(lst)
        split_idx = int(len(lst) * (1.0 - test_ratio))
        train_items.extend(lst[:split_idx])
        test_items.extend(lst[split_idx:])

    rng.shuffle(train_items)
    rng.shuffle(test_items)
    return train_items, test_items


def compute_metrics(preds: list[int], trues: list[int]) -> dict[str, Any]:
    """Compute accuracy, macro F1, and per-class precision/recall/F1."""
    n = len(trues)
    if n == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}

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
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }


def precompute_embeddings(
    items: list[dict[str, Any]], engine: SemanticEngine, batch_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute full 768-d master embeddings for all items using Nomic engine."""
    titles = [item["title"] for item in items]
    labels = [LABEL2ID[item["label"]] for item in items]

    print(f"Precomputing 768-d master embeddings for {len(titles)} titles...")
    t0 = time.perf_counter()

    all_vecs = []
    for i in range(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]
        vecs = engine.encode(
            batch,
            dim=MASTER_EMBEDDING_DIM,
            task_type=TASK_CLASSIFICATION,
            normalize=True,
        )
        if vecs is None:
            raise RuntimeError(f"Embedding generation failed for batch {i}..{i+len(batch)}")
        all_vecs.append(vecs)

    elapsed = time.perf_counter() - t0
    embeddings_np = np.vstack(all_vecs)
    print(f"Precomputed {embeddings_np.shape} in {elapsed:.2f}s ({len(titles)/elapsed:.1f} titles/sec)")

    embeddings_t = torch.tensor(embeddings_np, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return embeddings_t, labels_t


def evaluate_matryoshka_dimensions(
    classifier: MatryoshkaLinearClassifier,
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
            metrics = compute_metrics(preds, trues)
            results[dim] = metrics
    return results


def print_comparison_table(results: dict[int, dict[str, Any]]) -> None:
    """Print ASCII comparison table across all Matryoshka dimensions."""
    print("\n" + "=" * 80)
    print("MATRYOSHKA MULTI-DIMENSION PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Dimension':<12} | {'Accuracy':<10} | {'Macro F1':<10} | {'Compression':<14} | {'Status'}")
    print("-" * 80)

    base_dim = MASTER_EMBEDDING_DIM
    base_acc = results[base_dim]["accuracy"]
    for dim in MATRYOSHKA_DIMS:
        res = results.get(dim, {})
        acc = res.get("accuracy", 0.0)
        f1 = res.get("macro_f1", 0.0)
        compression = f"{(1.0 - dim / base_dim) * 100:.1f}% reduction" if dim < base_dim else "baseline"
        retention = f"{(acc / base_acc) * 100:.1f}% retention" if base_acc > 0 else "N/A"
        print(f"{dim:<12} | {acc * 100:6.2f}%    | {f1:8.4f}   | {compression:<14} | {retention}")
    print("=" * 80 + "\n")


def train_matryoshka(
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dry_run: bool = False,
    data_path: Path = DATA_PATH,
    out_dir: Path = OUT_DIR,
    seed: int = 42,
) -> dict[str, Any]:
    """Train Matryoshka classifier and export master weights."""
    set_seed(seed)
    print(f"Loading data from {data_path}...")
    items = load_dataset(data_path)
    print(f"Loaded {len(items)} valid labeled items across {len(CLASSES)} classes.")

    train_items, val_items = stratified_split(items, test_ratio=0.2, seed=seed)
    print(f"Split: {len(train_items)} train, {len(val_items)} validation.")

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

    # Initialize SemanticEngine and precompute embeddings
    engine = get_semantic_engine()
    train_x, train_y = precompute_embeddings(train_items, engine, batch_size=64)
    val_x, val_y = precompute_embeddings(val_items, engine, batch_size=64)

    # Dataloader
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model & Loss
    classifier = MatryoshkaLinearClassifier(master_dim=MASTER_EMBEDDING_DIM, num_classes=len(CLASSES))
    matryoshka_loss_fn = MatryoshkaLossModule(
        matryoshka_dims=MATRYOSHKA_DIMS,
        class_weights=class_weights_t,
    )

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\nStarting Matryoshka representation learning...")
    print(f"Optimizing jointly across dimensions: {list(MATRYOSHKA_DIMS)}")

    best_val_f1 = 0.0
    best_results: dict[int, dict[str, Any]] = {}

    for epoch in range(1, epochs + 1):
        classifier.train()
        total_epoch_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss, _ = matryoshka_loss_fn(classifier, batch_x, batch_y)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_epoch_loss / max(1, num_batches)

        # Validation across dims
        results = evaluate_matryoshka_dimensions(classifier, val_x, val_y)
        # Primary tracking metric: balanced F1 at default production dimension (384-d)
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
    meta_path = out_dir / "meta.json"

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
    print(f"Saved master Matryoshka weights to {weights_path}")

    # Build comprehensive meta.json
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
    parser = argparse.ArgumentParser(description="Train Matryoshka seniority classifier")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dry-run", action="store_true", help="Quick dry run with mini dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_matryoshka(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dry_run=args.dry_run,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

