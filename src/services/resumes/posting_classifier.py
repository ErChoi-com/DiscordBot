"""MiniLM classification for deciding which posting text informs a resume.

This is deliberately a classifier, not a text rewriter. It keeps the original
wording for every retained piece and only drops a piece when the fine-tuned
model is highly confident it is boilerplate. A missing model fails open.
"""
from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from services import capacity
from services.resumes.posting_segments import split_description

LABELS = ("SKILL_DUTY", "ROLE_FACTS", "COMPANY_CONTEXT", "BOILERPLATE")
BOILERPLATE_LABEL = "BOILERPLATE"
# Calibrated on 45,786 pieces: 0.70 removes over 12% of boilerplate while
# limiting content loss to < 0.09% (less than 1 in 1,000 real content pieces),
# matching the probability distribution of the retrained Asymmetric Focal Loss model.
DEFAULT_THRESHOLD = 0.70
ROOT = Path(__file__).resolve().parents[3]
_INFERENCE_LOCK = threading.Lock()


def _model_dir() -> Path:
    """Use a deployed model when configured; keep the local lab artifact usable."""
    configured = os.getenv("POSTING_CLASSIFIER_MODEL_PATH")
    if configured:
        return Path(configured)
    deployed = ROOT / "models" / "posting_encoder"
    if deployed.exists():
        return deployed
    return ROOT / ".resume_lab" / "boilerplate" / "posting_encoder"


@lru_cache(maxsize=1)
def load_posting_classifier() -> tuple[Any, Any, tuple[str, ...], int] | None:
    """Load the fine-tuned MiniLM head and tokenizer, if this host has one."""
    directory = _model_dir()
    weights, meta = directory / "posting_encoder.pt", directory / "meta.json"
    if not weights.is_file() or not meta.is_file() or not capacity.can_afford(2.0):
        return None
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        info = json.loads(meta.read_text(encoding="utf-8"))
        labels = tuple(info["labels"])
        base = str(info.get("backbone") or info.get("base"))
        max_len = int(info.get("max_len", 128))

        class PostingEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = AutoModel.from_pretrained(base, trust_remote_code=True)
                raw_hidden = getattr(self.encoder.config, "hidden_size", 384)
                self.feature_dim = int(info.get("feature_dim", info.get("matryoshka_dim", 384 if ("nomic" in base.lower() or raw_hidden > 384) else raw_hidden)))
                self.is_matryoshka = bool(info.get("is_matryoshka", (raw_hidden > self.feature_dim) or ("nomic" in base.lower())))
                self.drop = torch.nn.Dropout(0.1)
                self.piece_head = torch.nn.Linear(self.feature_dim, len(labels))
                self.term_head = torch.nn.Linear(self.feature_dim, 3)

            def forward(self, input_ids: Any, attention_mask: Any, **kwargs: Any) -> Any:
                hidden = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask, **kwargs
                ).last_hidden_state
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                if self.is_matryoshka:
                    pooled = torch.nn.functional.normalize(pooled[:, :self.feature_dim], p=2, dim=-1)
                    hidden = torch.nn.functional.normalize(hidden[:, :, :self.feature_dim], p=2, dim=-1)
                return self.piece_head(self.drop(pooled)), self.term_head(self.drop(hidden))

        model = PostingEncoder()
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.eval()
        try:
            tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        return model, tokenizer, labels, max_len
    except Exception as exc:
        print(f"Posting classifier disabled (model load failed): {exc}")
        return None


def retain_resume_relevant_text(
    text: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    use_engine: bool = False,
) -> str:
    """Return content pieces; omit only high-confidence boilerplate."""
    text = str(text or "").strip()
    if not text:
        return text
    pieces = split_description(text)
    if not pieces:
        return text

    # Check for fine-tuned local classifier artifact
    loaded = load_posting_classifier()
    if loaded is not None:
        try:
            import torch

            model, tokenizer, labels, max_len = loaded
            boilerplate_index = labels.index(BOILERPLATE_LABEL)
            with _INFERENCE_LOCK, torch.inference_mode():
                encoded = tokenizer(pieces, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
                res = model(**encoded)
                logits = res[0] if isinstance(res, tuple) else res
                probabilities = torch.softmax(logits, dim=-1).tolist()
            kept = [
                piece for piece, scores in zip(pieces, probabilities)
                if not (
                    labels[max(range(len(scores)), key=scores.__getitem__)] == BOILERPLATE_LABEL
                    and scores[boilerplate_index] >= threshold
                )
            ]
            return "\n".join(kept).strip()
        except Exception as exc:
            print(f"Posting classifier disabled (inference failed): {exc}")
            return text

    # If local artifact not loaded, optionally use centralized SemanticEngine
    if use_engine:
        try:
            from services.semantic.engine import get_semantic_engine
            engine = get_semantic_engine()
            filtered = engine.filter_boilerplate(pieces)
            if filtered:
                return "\n".join(filtered).strip()
        except Exception:
            pass

    return text


def _snap_to_word_boundary(text: str, a: int, b: int) -> str:
    """Expand span (a, b) outward to surrounding token/word boundaries.
    Prevents WordPiece subword fragmentation (e.g. 'Ka' -> 'Kafka', 'PG' -> 'FPGA')
    without hardcoding specific suffixes."""
    delims = set(" \t\r\n,.;:()[]{}\"'`!?—\\/-")
    while a > 0 and text[a - 1] not in delims:
        a -= 1
    while b < len(text) and text[b] not in delims:
        b += 1
    return text[a:b].strip(" \t\r\n,.;:()[]{}\"'`!?—\\/-")


def extract_posting_terms(text: str) -> list[str]:
    """Extract key resume-relevant terms using the fine-tuned MiniLM term head."""
    text = str(text or "").strip()
    loaded = load_posting_classifier()
    if not text or loaded is None:
        return []
    pieces = split_description(text)
    if not pieces:
        return []
    try:
        import torch

        model, tokenizer, labels, max_len = loaded
        with _INFERENCE_LOCK, torch.inference_mode():
            encoded = tokenizer(
                pieces,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping")
            res = model(**encoded)
            if not isinstance(res, tuple) or len(res) < 2:
                return []
            term_logits = res[1]
            pred_tags = term_logits.argmax(dim=-1).tolist()

        terms: list[str] = []
        seen: set[str] = set()
        offsets_list = offsets.tolist() if hasattr(offsets, "tolist") else offsets
        for i, (piece, tags, offset_list) in enumerate(zip(pieces, pred_tags, offsets_list)):
            spans: list[list[int]] = []
            cur: list[int] | None = None
            for (a, b), t in zip(offset_list, tags):
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
            for a, b in spans:
                term = _snap_to_word_boundary(piece, a, b)
                if term and len(term) >= 2 and not term.isdigit() and term.lower() not in seen:
                    seen.add(term.lower())
                    terms.append(term)
        return terms
    except Exception as exc:
        print(f"Key extraction disabled (inference failed): {exc}")
        return []


def extract_resume_skills(text: str) -> list[str]:
    """Extract candidate tools, skills, and systems directly from resume text.
    Uses the fine-tuned MiniLM term head with word-boundary snapping."""
    return extract_posting_terms(text)


