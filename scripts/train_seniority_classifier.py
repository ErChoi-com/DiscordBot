"""Train neural seniority classifier using MiniLM on labeled title corpus.

Trains a 6-class classifier:
  intern, newgrad, junior, mid, senior, staff

Saves model weights, tokenizer, and metadata to rebuilt_app/models/seniority_encoder.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / ".resume_lab" / "seniority" / "titles.jsonl"
OUT_DIR = ROOT / "models" / "seniority_encoder"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_BASE = ROOT / "models" / "all-MiniLM-L6-v2"

CLASSES = ["intern", "newgrad", "junior", "mid", "senior", "staff"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for i, c in enumerate(CLASSES)}
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-5
SEED = 42

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TitleDataset(Dataset):
    def __init__(self, items: list[dict], tokenizer, max_len: int = 64):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        title = item["title"]
        label_id = LABEL2ID[item["label"]]
        enc = self.tokenizer(
            title,
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

class SeniorityEncoder(nn.Module):
    def __init__(self, base: str, num_classes: int = 6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        self.drop = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(self.drop(pooled))

def stratified_split(items: list[dict], test_ratio: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {}
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

def compute_metrics(preds: list[int], trues: list[int]):
    correct = sum(1 for p, t in zip(preds, trues) if p == t)
    total = len(trues)
    acc = correct / max(total, 1)
    
    per_class = {}
    for i, cls in enumerate(CLASSES):
        tp = sum(1 for p, t in zip(preds, trues) if p == i and t == i)
        fp = sum(1 for p, t in zip(preds, trues) if p == i and t != i)
        fn = sum(1 for p, t in zip(preds, trues) if p != i and t == i)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        per_class[cls] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": sum(1 for t in trues if t == i)}
        
    return {"accuracy": round(acc, 4), "per_class": per_class}

def train():
    set_seed(SEED)
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Run compile_seniority_labels.py first.")
        sys.exit(1)

    items = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"Loaded {len(items)} total samples from {DATA_PATH}")
    train_items, test_items = stratified_split(items, test_ratio=0.2, seed=SEED)
    print(f"Train split: {len(train_items)} | Test split: {len(test_items)}")

    # Class weights for imbalanced cross-entropy
    train_counts = Counter(it["label"] for it in train_items)
    total_train = len(train_items)
    weights = [
        total_train / (len(CLASSES) * max(train_counts[c], 1))
        for c in CLASSES
    ]
    # Smooth weights
    weights = [math.sqrt(w) for w in weights]
    class_weights = torch.tensor(weights, dtype=torch.float)
    print("Class weights:", {c: round(w, 2) for c, w in zip(CLASSES, weights)})

    base = str(LOCAL_BASE) if LOCAL_BASE.exists() else BASE_MODEL
    print(f"Using base model: {base}")
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = SeniorityEncoder(base, num_classes=len(CLASSES))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    model.to(device)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_ds = TitleDataset(train_items, tokenizer, max_len=MAX_LEN)
    test_ds = TitleDataset(test_items, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    best_acc = 0.0
    best_metrics = None
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
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

        avg_loss = total_loss / len(train_loader)

        # Eval
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                pred = logits.argmax(dim=-1).cpu().tolist()
                preds.extend(pred)
                trues.extend(batch["label"].tolist())

        metrics = compute_metrics(preds, trues)
        acc = metrics["accuracy"]
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Test Acc: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_metrics = metrics
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    print(f"\nTraining Complete. Best Test Accuracy: {best_acc*100:.2f}%")
    print("Per-class Metrics:")
    for cls, met in best_metrics["per_class"].items():
        print(f"  {cls:8s} -> Prec: {met['precision']:.3f}, Rec: {met['recall']:.3f}, F1: {met['f1']:.3f} (support={met['support']})")

    # Save artifact
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = OUT_DIR / "seniority_encoder.pt"
    torch.save(best_state, weights_path)
    tokenizer.save_pretrained(OUT_DIR)
    model.encoder.config.save_pretrained(OUT_DIR)

    meta = {
        "labels": CLASSES,
        "base": BASE_MODEL,
        "max_len": MAX_LEN,
        "accuracy": best_acc,
        "metrics": best_metrics,
    }
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model and tokenizer to {OUT_DIR}")

if __name__ == "__main__":
    train()

