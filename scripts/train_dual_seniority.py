"""Train dual-backbone seniority classifier (BGE-small and Nomic embed).

Trains a 6-class classifier:
  intern, newgrad, junior, mid, senior, staff

Supports backbones:
  1. BAAI/bge-small-en-v1.5 (native 384-d)
  2. nomic-ai/nomic-embed-text-v1.5 (768-d Matryoshka-sliced to 384-d, L2 renormalized)

Prefix configuration:
  - Nomic: prepends 'classification: ' to input titles
  - BGE: no query prefix needed for classification

Training details:
  - AdamW optimizer with weight decay
  - Cosine annealing learning rate scheduler
  - Inverse-frequency class weighting (with smoothing)
  - Stratified train/val split
  - Comprehensive metrics: Accuracy, per-class Precision, Recall, F1, Support
  - Saves weights, tokenizer, config, and meta.json to models/seniority_encoder

Usage:
  python scripts/train_dual_seniority.py --backbone bge --epochs 5
  python scripts/train_dual_seniority.py --backbone nomic --epochs 5
  python scripts/train_dual_seniority.py --backbone bge --dry-run
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
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / ".resume_lab" / "seniority" / "titles.jsonl"
DEFAULT_OUT_DIR = ROOT / "models" / "seniority_encoder"

BACKBONE_MAP = {
    "bge": "BAAI/bge-small-en-v1.5",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
    "nomic-embed": "nomic-ai/nomic-embed-text-v1.5",
    "nomic-ai/nomic-embed-text-v1.5": "nomic-ai/nomic-embed-text-v1.5",
}

CLASSES = ["intern", "newgrad", "junior", "mid", "senior", "staff"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for i, c in enumerate(CLASSES)}

TARGET_EMBED_DIM = 384  # Matryoshka dimension


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SeniorityTitleDataset(Dataset):
    """Dataset with prefix-aware tokenization for titles."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        tokenizer: Any,
        prefix: str = "",
        max_len: int = 64,
    ) -> None:
        self.items = items
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.items[idx]
        title = str(item.get("title") or "").strip()
        formatted = f"{self.prefix}{title}"
        label_id = LABEL2ID[item["label"]]

        enc = self.tokenizer(
            formatted,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label_id, dtype=torch.long),
        }


class DualSeniorityClassifier(nn.Module):
    """Dual-backbone classifier supporting native 384-d (BGE) and 768->384 Matryoshka (Nomic)."""

    def __init__(
        self,
        backbone_id: str,
        num_classes: int = len(CLASSES),
        feature_dim: int = TARGET_EMBED_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone_id = backbone_id
        self.is_nomic = "nomic" in backbone_id.lower()
        self.feature_dim = feature_dim

        # Load transformer backbone
        self.encoder = AutoModel.from_pretrained(backbone_id, trust_remote_code=True)
        self.hidden_dim = getattr(self.encoder.config, "hidden_size", 384)

        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # Mean pooling over non-padded tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # Matryoshka slicing if hidden dimension exceeds feature_dim (e.g. 768 -> 384 for Nomic)
        if pooled.shape[-1] > self.feature_dim:
            pooled = pooled[:, :self.feature_dim]

        # L2 re-normalization (unit sphere projection)
        pooled = nn.functional.normalize(pooled, p=2, dim=-1)

        # Classification head
        logits = self.classifier(self.drop(pooled))
        return logits


def stratified_split(
    items: list[dict[str, Any]],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified train/val split preserving class balance."""
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        lbl = item.get("label")
        if lbl in LABEL2ID:
            by_class.setdefault(lbl, []).append(item)

    train_items: list[dict[str, Any]] = []
    test_items: list[dict[str, Any]] = []

    for _, lst in by_class.items():
        rng.shuffle(lst)
        split_idx = int(len(lst) * (1.0 - test_ratio))
        # Ensure at least 1 item in test if possible
        if len(lst) > 1 and split_idx == len(lst):
            split_idx = len(lst) - 1
        train_items.extend(lst[:split_idx])
        test_items.extend(lst[split_idx:])

    rng.shuffle(train_items)
    rng.shuffle(test_items)
    return train_items, test_items


def compute_metrics(preds: list[int], trues: list[int]) -> dict[str, Any]:
    """Compute overall accuracy and per-class precision, recall, f1, support."""
    correct = sum(1 for p, t in zip(preds, trues) if p == t)
    total = len(trues)
    acc = correct / max(total, 1)

    per_class = {}
    f1_list = []
    for i, cls in enumerate(CLASSES):
        tp = sum(1 for p, t in zip(preds, trues) if p == i and t == i)
        fp = sum(1 for p, t in zip(preds, trues) if p == i and t != i)
        fn = sum(1 for p, t in zip(preds, trues) if p != i and t == i)
        prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / max(prec + rec, 1e-8) if (prec + rec) > 0 else 0.0
        supp = sum(1 for t in trues if t == i)
        per_class[cls] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": supp,
        }
        if supp > 0:
            f1_list.append(f1)

    macro_f1 = sum(f1_list) / max(len(f1_list), 1)
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }


def train_dual_seniority(
    backbone_arg: str = "bge",
    data_path: Path | str = DEFAULT_DATA_PATH,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    max_len: int = 64,
    dry_run: bool = False,
    device_str: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Train seniority classifier using specified backbone and hyperparameters."""
    set_seed(seed)
    data_path = Path(data_path)
    out_dir = Path(out_dir)

    # Resolve backbone ID and prefix
    backbone_id = BACKBONE_MAP.get(backbone_arg, backbone_arg)
    is_nomic = "nomic" in backbone_id.lower()
    prefix = "classification: " if is_nomic else ""

    print("=" * 70)
    print(f"Dual-Backbone Seniority Classifier Training")
    print(f"Backbone: {backbone_id}")
    print(f"Prefix: '{prefix}'")
    print(f"Matryoshka Target Dimension: {TARGET_EMBED_DIM}")
    print(f"Data Path: {data_path}")
    print(f"Output Directory: {out_dir}")
    print(f"Epochs: {1 if dry_run else epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 70)

    if not data_path.exists():
        raise FileNotFoundError(f"Corpus dataset not found at {data_path}")

    # Load dataset
    items: list[dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                if entry.get("label") in LABEL2ID:
                    items.append(entry)

    print(f"Loaded {len(items)} labeled titles from {data_path}")

    if dry_run:
        print("[DRY RUN] Truncating dataset for rapid 1-epoch verification pipeline...")
        random.shuffle(items)
        items = items[:120]

    train_items, val_items = stratified_split(items, test_ratio=0.2, seed=seed)
    print(f"Train split: {len(train_items)} samples | Val split: {len(val_items)} samples")

    # Inverse-frequency class weights (smoothed with square root)
    train_counts = Counter(it["label"] for it in train_items)
    n_train = len(train_items)
    n_classes = len(CLASSES)
    weights = [
        math.sqrt(n_train / (n_classes * max(train_counts.get(c, 1), 1)))
        for c in CLASSES
    ]
    # Normalize weights so sum equals n_classes
    weight_sum = sum(weights)
    norm_weights = [w * (n_classes / weight_sum) for w in weights]
    class_weights_tensor = torch.tensor(norm_weights, dtype=torch.float)
    print("Class distribution & weights:")
    for c, w in zip(CLASSES, norm_weights):
        print(f"  {c:8s}: count={train_counts.get(c, 0):4d} | weight={w:.3f}")

    # Load tokenizer and model
    print(f"\nInitializing tokenizer & encoder for {backbone_id}...")
    tokenizer = AutoTokenizer.from_pretrained(backbone_id, trust_remote_code=True)
    model = DualSeniorityClassifier(
        backbone_id=backbone_id,
        num_classes=n_classes,
        feature_dim=TARGET_EMBED_DIM,
    )

    # Device selection
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model.to(device)
    class_weights_tensor = class_weights_tensor.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Data loaders
    train_ds = SeniorityTitleDataset(train_items, tokenizer, prefix=prefix, max_len=max_len)
    val_ds = SeniorityTitleDataset(val_items, tokenizer, prefix=prefix, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Optimizer & Cosine Annealing Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    actual_epochs = 1 if dry_run else epochs
    total_steps = max(len(train_loader) * actual_epochs, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6,
    )

    best_acc = 0.0
    best_metrics: dict[str, Any] = {}
    best_state: dict[str, Any] = {}

    start_time = time.time()
    for ep in range(1, actual_epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        # Validation pass
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                preds.extend(logits.argmax(dim=-1).cpu().tolist())
                trues.extend(batch["label"].tolist())

        metrics = compute_metrics(preds, trues)
        acc = metrics["accuracy"]
        macro_f1 = metrics["macro_f1"]
        print(
            f"Epoch {ep:2d}/{actual_epochs:2d} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Acc: {acc*100:5.2f}% | "
            f"Macro F1: {macro_f1*100:5.2f}%"
        )

        if acc >= best_acc or ep == 1:
            best_acc = acc
            best_metrics = metrics
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s. Best Val Accuracy: {best_acc*100:.2f}%")
    print("Per-class performance on validation set:")
    for cls, met in best_metrics["per_class"].items():
        print(
            f"  {cls:8s} -> Precision: {met['precision']:.3f} | "
            f"Recall: {met['recall']:.3f} | "
            f"F1: {met['f1']:.3f} (support={met['support']})"
        )

    # Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "seniority_encoder.pt"
    torch.save(best_state, weights_path)

    # Save tokenizer and config
    tokenizer.save_pretrained(out_dir)
    model.encoder.config.save_pretrained(out_dir)

    meta_payload = {
        "labels": CLASSES,
        "backbone": backbone_id,
        "base": backbone_id,
        "is_nomic": is_nomic,
        "prefix": prefix,
        "embedding_dim": TARGET_EMBED_DIM,
        "num_classes": n_classes,
        "max_len": max_len,
        "accuracy": best_acc,
        "macro_f1": best_metrics.get("macro_f1", 0.0),
        "metrics": best_metrics,
        "trained_samples": len(train_items),
        "val_samples": len(val_items),
        "dry_run": dry_run,
    }

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    print(f"\nModel artifacts successfully saved to {out_dir}:")
    print(f"  - Weights: {weights_path}")
    print(f"  - Metadata: {meta_path}")
    print(f"  - Tokenizer & Config: {out_dir}")

    return meta_payload


def parse_args():
    parser = argparse.ArgumentParser(description="Train dual-backbone seniority classifier.")
    parser.add_argument(
        "--backbone",
        type=str,
        default="bge",
        choices=list(BACKBONE_MAP.keys()),
        help="Backbone architecture (bge or nomic)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to titles.jsonl corpus",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Destination directory for trained model artifacts",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate (default: 3e-5)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW (default: 0.01)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=64,
        help="Maximum token sequence length (default: 64)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a fast 1-epoch validation pass for quick end-to-end verification",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force execution on device ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dual_seniority(
        backbone_arg=args.backbone,
        data_path=args.data_path,
        out_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_len=args.max_len,
        dry_run=args.dry_run,
        device_str=args.device,
        seed=args.seed,
    )
