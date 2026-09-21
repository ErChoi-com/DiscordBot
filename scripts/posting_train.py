"""Train the posting-piece encoder on distilled labels.

Input: .resume_lab/boilerplate/pieces.jsonl (scripts/posting_label.py).
Supports dual backbones:
  - BAAI/bge-small-en-v1.5 (384-d)
  - nomic-ai/nomic-embed-text-v1.5 (768-d with Matryoshka slicing to 384-d and L2 normalization)
  - sentence-transformers/all-MiniLM-L6-v2 (384-d legacy base)

Two heads:
  piece head  SKILL_DUTY / ROLE_FACTS / COMPANY_CONTEXT / BOILERPLATE (4 classes)
  term head   BIO tags over word pieces for resume-useful terms (3 classes)

Term tags come from all non-boilerplate pieces after exact-span validation and
keyword enrichment.

Employer GroupKFold split: a company's house boilerplate cannot appear in both
train and test. Deleting real content is catastrophic for ATS tailoring, so the
report highlights content loss (SKILL_DUTY / ROLE_FACTS text stripped as BOILERPLATE)
against threshold t=0.70 (target < 0.1% char loss).

Usage:
    python scripts/posting_train.py --backbone BAAI/bge-small-en-v1.5 --epochs 4
    python scripts/posting_train.py --backbone nomic-ai/nomic-embed-text-v1.5 --epochs 4
    python scripts/posting_train.py --eval-only --eval-threshold 0.70
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".resume_lab" / "boilerplate" / "pieces.jsonl"
OUT = ROOT / ".resume_lab" / "boilerplate" / "posting_encoder"
DEFAULT_BASE = "sentence-transformers/all-MiniLM-L6-v2"
BGE_BASE = "BAAI/bge-small-en-v1.5"
NOMIC_BASE = "nomic-ai/nomic-embed-text-v1.5"

BACKBONE_ALIASES: dict[str, str] = {
    "bge": BGE_BASE,
    "bge-small": BGE_BASE,
    "bge-small-en": BGE_BASE,
    "nomic": NOMIC_BASE,
    "nomic-embed": NOMIC_BASE,
    "nomic-embed-text": NOMIC_BASE,
    "minilm": DEFAULT_BASE,
    "all-minilm": DEFAULT_BASE,
}

LABELS = ["SKILL_DUTY", "ROLE_FACTS", "COMPANY_CONTEXT", "BOILERPLATE"]
BP = LABELS.index("BOILERPLATE")
MAX_LEN = 128


class PostingEncoder(nn.Module):
    """Dual-head posting encoder with Matryoshka dimension scaling & L2 normalization.

    Supports:
      - Standard 384-d encoders (BGE-small-en-v1.5, all-MiniLM-L6-v2)
      - Matryoshka 768-d encoders (nomic-embed-text-v1.5) sliced down to 384-d with L2 normalization.
    """
    def __init__(self, base: str | nn.Module, matryoshka_dim: int = 384):
        super().__init__()
        self.matryoshka_dim = matryoshka_dim
        if isinstance(base, nn.Module):
            self.encoder = base
            self.base_name = getattr(base, "name_or_path", "custom")
        else:
            self.base_name = str(base)
            self.encoder = AutoModel.from_pretrained(base, trust_remote_code=True)

        raw_hidden = getattr(self.encoder.config, "hidden_size", matryoshka_dim)
        self.is_matryoshka = (raw_hidden > matryoshka_dim) or ("nomic" in self.base_name.lower())
        self.feature_dim = matryoshka_dim if self.is_matryoshka else raw_hidden

        self.drop = nn.Dropout(0.1)
        self.piece_head = nn.Linear(self.feature_dim, len(LABELS))
        self.term_head = nn.Linear(self.feature_dim, 3)  # O, B-TERM, I-TERM

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)

        if self.is_matryoshka:
            # Matryoshka dimension slicing and L2 normalization
            pooled = pooled[:, :self.feature_dim]
            pooled = nn.functional.normalize(pooled, p=2, dim=-1)
            h = h[:, :, :self.feature_dim]
            h = nn.functional.normalize(h, p=2, dim=-1)

        return self.piece_head(self.drop(pooled)), self.term_head(self.drop(h))


class AsymmetricFocalLoss(nn.Module):
    """Focal Loss with asymmetric penalties to prevent false-positive boilerplate stripping.

    Standard CrossEntropy:
        CE(p_t) = -log(p_t)

    Focal Loss (Lin et al.):
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Suppresses loss on easy examples (p_t -> 1), concentrating gradient updates
    on hard boundary examples.

    Asymmetric Penalty:
    In resume parsing, stripping true content (SKILL_DUTY or ROLE_FACTS) is catastrophic
    for ATS tailoring, whereas retaining a boilerplate statement is benign.
    We apply an asymmetric multiplier `asym_fp_penalty` whenever true content
    is predicted as BOILERPLATE.
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, asym_fp_penalty: float = 2.5):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.asym_fp_penalty = asym_fp_penalty

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-6, max=1.0)
        focal = ((1.0 - pt) ** self.gamma) * ce
        preds = logits.argmax(dim=-1)
        is_content = (targets == 0) | (targets == 1)
        is_bp_pred = (preds == BP)
        loss = torch.where(is_content & is_bp_pred, focal * self.asym_fp_penalty, focal)
        return loss.mean()


def load(seed: int, test_frac: float = 0.2, kfold: int = 0, fold: int = 0) -> tuple[list[dict], list[dict]]:
    """Load pieces and split by employer (GroupKFold or randomized employer hold-out).

    Ensures that house boilerplate from the same company never appears in both train and test.
    """
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        if r["label"] == "BOILERPLATE":
            r["terms"] = []

    if kfold > 1:
        try:
            from sklearn.model_selection import GroupKFold
            gkf = GroupKFold(n_splits=kfold)
            groups = [r["company"].lower() for r in rows]
            splits = list(gkf.split(rows, groups=groups))
            train_idx, test_idx = splits[fold % kfold]
            return [rows[i] for i in train_idx], [rows[i] for i in test_idx]
        except Exception as exc:
            print(f"GroupKFold unavailable ({exc}); falling back to randomized employer holdout.")

    companies = sorted({r["company"].lower() for r in rows})
    random.Random(seed).shuffle(companies)
    test_c = set(companies[: max(1, int(len(companies) * test_frac))])
    return ([r for r in rows if r["company"].lower() not in test_c],
            [r for r in rows if r["company"].lower() in test_c])


def subsample(train: list[dict], frac: float, seed: int) -> list[dict]:
    """Keep a deterministic fraction of TRAIN postings (whole postings, not loose pieces)."""
    if frac >= 1.0:
        return train
    pids = sorted({r["pid"] for r in train})
    random.Random(seed).shuffle(pids)
    keep = set(pids[: max(1, round(len(pids) * frac))])
    return [r for r in train if r["pid"] in keep]


def term_spans(text: str, terms: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for t in sorted(terms, key=len, reverse=True):
        for m in re.finditer(re.escape(t), text):
            if not any(m.start() < e and s < m.end() for s, e in spans):
                spans.append((m.start(), m.end()))
    return spans


def loader(rows, tok, size, shuffle):
    def collate(chunk):
        enc = tok([r["text"] for r in chunk], padding=True, truncation=True, max_length=MAX_LEN,
                  return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        tags = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
        for i, r in enumerate(chunk):
            spans, prev = term_spans(r["text"], r["terms"]), None
            for j, (a, b) in enumerate(offsets[i].tolist()):
                if a == b:
                    continue
                hit = next((s for s in spans if a < s[1] and s[0] < b), None)
                tags[i, j] = 0 if hit is None else (2 if hit == prev else 1)
                prev = hit
        enc["label"] = torch.tensor([LABELS.index(r["label"]) for r in chunk])
        enc["tags"], enc["offsets"] = tags, offsets
        return enc
    return DataLoader(rows, batch_size=size, shuffle=shuffle, collate_fn=collate)


def decode(text, offsets, tag_ids):
    spans, cur = [], None
    for (a, b), t in zip(offsets, tag_ids):
        if a == b:
            continue
        if t == 1 or (t == 2 and cur is None):
            if cur:
                spans.append(cur)
            cur = [a, b]
        elif t == 2:
            if cur:
                cur[1] = b
        else:
            if cur:
                spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    return [text[a:b] for a, b in spans]


@torch.no_grad()
def predict(model, tok, rows, batch_size: int = 128, device: torch.device | None = None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    probs, terms = [], []
    for enc in loader(rows, tok, batch_size, False):
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        logits, tlog = model(input_ids, attention_mask)
        probs += torch.softmax(logits, -1).cpu().tolist()
        base = len(terms)
        for i, ids in enumerate(tlog.argmax(-1).cpu().tolist()):
            terms.append(decode(rows[base + i]["text"], enc["offsets"][i].tolist(), ids))
    return probs, terms


def eval_content_loss(rows: list[dict], probs: list[list[float]], threshold: float = 0.70) -> dict:
    """Evaluate content loss (SKILL_DUTY or ROLE_FACTS stripped as BOILERPLATE) at a specific threshold.

    Safety criteria: content_char_loss < 0.1% (< 0.001) to prevent catastrophic loss of ATS skills.
    """
    gold = [LABELS.index(r["label"]) for r in rows]
    content_idx = [i for i, g in enumerate(gold) if LABELS[g] in ("SKILL_DUTY", "ROLE_FACTS")]
    bp_idx = [i for i, g in enumerate(gold) if g == BP]
    content_chars = sum(len(rows[i]["text"]) for i in content_idx)

    caught = sum(probs[i][BP] >= threshold for i in bp_idx) / max(len(bp_idx), 1)
    wrong = [i for i in content_idx if probs[i][BP] >= threshold]
    wrong_chars = sum(len(rows[i]["text"]) for i in wrong)
    loss_pct = wrong_chars / max(content_chars, 1)
    piece_loss_pct = len(wrong) / max(len(content_idx), 1)

    return {
        "threshold": threshold,
        "total_content_pieces": len(content_idx),
        "total_content_chars": content_chars,
        "wrong_pieces": len(wrong),
        "wrong_chars": wrong_chars,
        "content_char_loss": loss_pct,
        "content_piece_loss": piece_loss_pct,
        "total_boilerplate": len(bp_idx),
        "boilerplate_removed": caught,
        "passes_safety": loss_pct < 0.001,
    }


def report(rows, probs, terms, eval_threshold: float = 0.70) -> tuple[dict, list[int]]:
    gold = [LABELS.index(r["label"]) for r in rows]
    pred = [max(range(len(LABELS)), key=p.__getitem__) for p in probs]
    acc = sum(g == p for g, p in zip(gold, pred)) / max(len(rows), 1)
    print(f"  pieces {len(rows)}  accuracy {acc:.1%}")
    metrics = {
        "pieces": len(rows),
        "accuracy": acc,
        "classes": {},
        "thresholds": {},
        "terms": {},
    }
    for k, name in enumerate(LABELS):
        tp = sum(g == k and p == k for g, p in zip(gold, pred))
        P = tp / max(sum(p == k for p in pred), 1)
        R = tp / max(sum(g == k for g in gold), 1)
        n = sum(g == k for g in gold)
        metrics["classes"][name] = {"precision": P, "recall": R, "support": n}
        print(f"    {name:16} P {P:6.1%}  R {R:6.1%}  (n={n})")

    content = [i for i, g in enumerate(gold) if LABELS[g] in ("SKILL_DUTY", "ROLE_FACTS")]
    bps = [i for i, g in enumerate(gold) if g == BP]
    content_chars = sum(len(rows[i]["text"]) for i in content)
    print("  strip if P(BOILERPLATE) >= t:  boilerplate removed | SKILL/ROLE pieces wrongly stripped (chars)")
    for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99):
        caught = sum(probs[i][BP] >= t for i in bps) / max(len(bps), 1)
        wrong = [i for i in content if probs[i][BP] >= t]
        wrong_chars = sum(len(rows[i]["text"]) for i in wrong)
        loss_pct = wrong_chars / max(content_chars, 1)
        metrics["thresholds"][str(t)] = {
            "boilerplate_removed": caught,
            "wrong_pieces": len(wrong),
            "content_char_loss": loss_pct,
        }
        print(f"    t={t:.2f}   {caught:6.1%}   |  {len(wrong):4d} pieces ({loss_pct:5.2%} of content chars)")

    # Dedicated t=eval_threshold safety evaluation
    t_eval = eval_content_loss(rows, probs, threshold=eval_threshold)
    metrics["eval_threshold_target"] = t_eval
    print(f"\n  === Content Safety Evaluation (t={eval_threshold:.2f}) ===")
    print(f"  Content pieces wrongly stripped: {t_eval['wrong_pieces']} / {t_eval['total_content_pieces']} ({t_eval['content_piece_loss']:.3%})")
    print(f"  Content char loss: {t_eval['content_char_loss']:.4%} (target: < 0.1000%) - Safety Check: {'PASS' if t_eval['passes_safety'] else 'WARN'}")
    print(f"  Boilerplate stripped: {t_eval['boilerplate_removed']:.1%}\n")

    gold_t = sum(len(r["terms"]) for r in rows)
    hit = sum(len(set(p) & set(r["terms"])) for p, r in zip(terms, rows))
    term_p = hit / max(sum(map(len, terms)), 1)
    term_r = hit / max(gold_t, 1)
    term_f1 = (2 * term_p * term_r) / max(term_p + term_r, 1e-6)
    metrics["terms"] = {
        "precision": term_p,
        "recall": term_r,
        "f1": term_f1,
        "gold_terms": gold_t,
        "hits": hit,
    }
    print(f"  terms exact-span P {term_p:.1%}  R {term_r:.1%}  F1 {term_f1:.1%}  (gold {gold_t})")
    return metrics, pred


def resolve_backbone(name: str | None) -> str:
    if not name:
        return DEFAULT_BASE
    alias = BACKBONE_ALIASES.get(name.lower())
    if alias:
        return alias
    return name


def main():
    ap = argparse.ArgumentParser(description="Train or evaluate posting classifier with BGE/Nomic/MiniLM backbones.")
    ap.add_argument("--backbone", default=None,
                    help="Backbone model: BAAI/bge-small-en-v1.5, nomic-ai/nomic-embed-text-v1.5, or alias (bge, nomic, minilm)")
    ap.add_argument("--base", default=None,
                    help="Deprecated alias for --backbone")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--kfold", type=int, default=0,
                    help="Number of folds for employer GroupKFold (0 to use test-frac holdout)")
    ap.add_argument("--fold", type=int, default=0,
                    help="Fold index (0 to kfold-1) when --kfold > 1")
    ap.add_argument("--sample-frac", type=float, default=1.0,
                    help="fraction of TRAIN postings to keep, for sample-size experiments")
    ap.add_argument("--matryoshka-dim", type=int, default=384,
                    help="Dimension for Matryoshka slicing (e.g. 384 for Nomic 768->384)")
    ap.add_argument("--eval-threshold", type=float, default=0.70,
                    help="Target threshold for evaluating content preservation (default: 0.70)")
    ap.add_argument("--eval-only", action="store_true",
                    help="Evaluate existing model artifact without retraining")
    ap.add_argument("--out-name", default="posting_encoder",
                    help="subdir under .resume_lab/boilerplate/ to save this run's model into")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="Execution device: auto (detects CUDA, falls back to CPU), cuda, or cpu")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="Batch size (default: 64 on CUDA, 32 on CPU)")
    ap.add_argument("--gamma", type=float, default=2.0,
                    help="Focal Loss focusing factor gamma")
    ap.add_argument("--asym-penalty", type=float, default=2.5,
                    help="Asymmetric penalty multiplier for content false positives")
    ap.add_argument("--save-deployed", action="store_true",
                    help="also deploy model to models/posting_encoder")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    OUT = ROOT / ".resume_lab" / "boilerplate" / args.out_name
    backbone_arg = args.backbone or args.base

    if args.eval_only and not backbone_arg:
        meta_file = OUT / "meta.json"
        if not meta_file.exists():
            deployed_meta = ROOT / "models" / "posting_encoder" / "meta.json"
            if deployed_meta.exists():
                meta_file = deployed_meta
        if meta_file.exists():
            try:
                meta_info = json.loads(meta_file.read_text(encoding="utf-8"))
                backbone_arg = meta_info.get("backbone") or meta_info.get("base")
                args.matryoshka_dim = int(meta_info.get("matryoshka_dim", args.matryoshka_dim))
            except Exception:
                pass

    backbone_name = resolve_backbone(backbone_arg or DEFAULT_BASE)

    # Check local pre-downloaded base models
    local_base = ROOT / "models" / "sentence-transformers_all-MiniLM-L6-v2"
    base_path = str(local_base) if (backbone_name == DEFAULT_BASE and local_base.is_dir()) else backbone_name

    # Device selection: auto-detect CUDA with graceful CPU fallback
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    batch_size = args.batch_size if args.batch_size > 0 else (64 if device.type == "cuda" else 32)
    print(f"Backbone: {backbone_name} (resolved to: {base_path})")
    print(f"Device: {device} (batch_size={batch_size})")
    if device.type == "cuda":
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    train, test = load(args.seed, args.test_frac, kfold=args.kfold, fold=args.fold)
    if args.sample_frac < 1.0:
        before = len({r["pid"] for r in train})
        train = subsample(train, args.sample_frac, args.seed)
        print(f"  --sample-frac {args.sample_frac}: {before} -> {len({r['pid'] for r in train})} train postings")
    print(f"train {len(train)} pieces / {len({r['pid'] for r in train})} postings; "
          f"test {len(test)} / {len({r['pid'] for r in test})} (held-out employers)")
    print("  train labels", dict(collections.Counter(r["label"] for r in train)))

    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = PostingEncoder(base_path, matryoshka_dim=args.matryoshka_dim).to(device)
    print(f"Model initialized: feature_dim={model.feature_dim}, is_matryoshka={model.is_matryoshka}")

    if args.eval_only:
        # Load weights from target directory or fallback
        model_weights = OUT / "posting_encoder.pt"
        if not model_weights.exists():
            deployed = ROOT / "models" / "posting_encoder" / "posting_encoder.pt"
            if deployed.exists():
                model_weights = deployed
        if model_weights.exists():
            print(f"Loading weights from {model_weights}")
            model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))
        else:
            print(f"No existing checkpoint found at {model_weights}; evaluating untrained initialized model.")

        probs, terms = predict(model, tok, test, batch_size=batch_size, device=device)
        metrics, _ = report(test, probs, terms, eval_threshold=args.eval_threshold)
        return

    counts = collections.Counter(r["label"] for r in train)
    # Balanced class weights: COMPANY_CONTEXT is the rare class.
    weights = torch.tensor([len(train) / (len(LABELS) * max(counts[l], 1)) for l in LABELS],
                           dtype=torch.float, device=device)
    piece_loss = AsymmetricFocalLoss(gamma=args.gamma, weight=weights, asym_fp_penalty=args.asym_penalty)
    term_loss = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    per_epoch = (len(train) + batch_size - 1) // batch_size
    steps = args.epochs * per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, 0.1 * steps)) * max(0.0, 1 - s / steps))

    use_cuda = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    last_metrics = {}
    for epoch in range(args.epochs):
        model.train()
        t0, total = time.time(), 0.0
        for step, enc in enumerate(loader(train, tok, batch_size, True), 1):
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            labels = enc["label"].to(device)
            tags = enc["tags"].to(device)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits, tlog = model(input_ids, attention_mask)
                loss = piece_loss(logits, labels) + term_loss(tlog.reshape(-1, 3), tags.reshape(-1))

            if use_cuda:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            sched.step()
            opt.zero_grad()
            total += loss.item()
            if step % 100 == 0 or step == per_epoch:
                print(f"  epoch {epoch + 1}/{args.epochs} step {step}/{per_epoch} loss: {total / step:.4f} ({time.time() - t0:.0f}s)", flush=True)
        print(f"epoch {epoch + 1}: avg_loss {total / per_epoch:.4f}  total_time {time.time() - t0:.0f}s", flush=True)
        last_metrics, _ = report(test, *predict(model, tok, test, device=device), eval_threshold=args.eval_threshold)

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / "posting_encoder.pt")
    tok.save_pretrained(OUT)
    meta = {
        "base": base_path,
        "backbone": backbone_name,
        "labels": LABELS,
        "max_len": MAX_LEN,
        "feature_dim": model.feature_dim,
        "matryoshka_dim": args.matryoshka_dim,
        "is_matryoshka": model.is_matryoshka,
        "eval_threshold": args.eval_threshold,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT / "eval_report.json").write_text(json.dumps(last_metrics, indent=2), encoding="utf-8")

    probs, terms = predict(model, tok, test, device=device)
    wrong = sorted(((p[BP], r) for p, r in zip(probs, test) if r["label"] in ("SKILL_DUTY", "ROLE_FACTS")),
                   key=lambda x: -x[0])
    print("\nreal content the model most wants to strip:")
    for p, r in wrong[:15]:
        print(f"  P(BP)={p:.2f} {r['label']:10} {r['text'][:110]}")
    print(f"\nsaved {OUT}")

    if args.save_deployed:
        deployed_dir = ROOT / "models" / "posting_encoder"
        deployed_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for item in OUT.iterdir():
            if item.is_file():
                shutil.copy2(item, deployed_dir / item.name)
        print(f"Deployed to {deployed_dir}")


if __name__ == "__main__":
    main()
