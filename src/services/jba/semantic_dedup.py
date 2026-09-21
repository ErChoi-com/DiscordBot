"""Semantic job deduplication using local MiniLM embeddings.

Catches cross-ATS reposts and session-URL duplicates that bypass exact URL hashes
(e.g., Oracle Cloud, Workday, Greenhouse URLs with transient session tokens).
"""
from __future__ import annotations

import functools
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from services import capacity
from services.resumes.posting_classifier import retain_resume_relevant_text

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "sentence-transformers_all-MiniLM-L6-v2"
if not MODEL_DIR.exists():
    MODEL_DIR = ROOT / "models" / "posting_encoder"

_MODEL_LOCK = threading.Lock()


@functools.lru_cache(maxsize=1)
def load_dedup_encoder() -> tuple[Any, Any, Any] | None:
    """Load MiniLM model and tokenizer on CUDA (if available) or CPU."""
    if not capacity.can_afford(1.5):
        return None
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_path = str(MODEL_DIR) if MODEL_DIR.exists() else "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as exc:
        print(f"Semantic dedup model load failed: {exc}")
        return None


def job_text_for_embedding(job: dict[str, Any], max_desc_chars: int = 400) -> str:
    """Construct canonical text representation for semantic deduplication."""
    title = str(job.get("title") or "").strip()
    company = str(job.get("company") or "").strip()
    location = str(job.get("location") or "").strip()
    raw_desc = str(
        job.get("description")
        or job.get("snippet")
        or job.get("summary")
        or job.get("job_description")
        or ""
    ).strip()
    # Strip EEO/legal boilerplate before embedding so vectors compare technical duties
    clean_desc = retain_resume_relevant_text(raw_desc, threshold=0.70)[:max_desc_chars]
    parts = [p for p in [title, company, location, clean_desc] if p]
    return " | ".join(parts)


def compute_job_embeddings(
    jobs: list[dict[str, Any]],
    batch_size: int = 128,
    max_length: int = 128,
    dim: int = 128,
) -> np.ndarray | None:
    """Compute normalized L2 embeddings for a list of job dicts using SemanticEngine (default 128-d Matryoshka)."""
    if not jobs:
        return None

    try:
        from services.semantic.engine import get_semantic_engine
        engine = get_semantic_engine()
        engine_vecs = engine.compute_job_embeddings(jobs, dims=dim)
        if engine_vecs is not None:
            return engine_vecs
    except Exception:
        pass

    # Fallback to local MiniLM encoder if SemanticEngine is unavailable
    loaded = load_dedup_encoder()
    if not loaded:
        return None

    import torch

    model, tokenizer, device = loaded
    texts = [job_text_for_embedding(j) for j in jobs]
    all_embeddings = []

    with _MODEL_LOCK, torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            encoded = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(normed.cpu().numpy())

    return np.vstack(all_embeddings) if all_embeddings else None


def normalize_title_for_dedup(title: str) -> str:
    """Normalize title for exact or near-exact deduplication.
    Lowercases, strips punctuation, and normalizes whitespace without
    stripping distinct employment categories (preserving full-time vs part-time)."""
    import re
    t = str(title or "").lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def find_semantic_duplicates(
    jobs: list[dict[str, Any]],
    threshold: float = 0.92,
) -> list[tuple[int, int, float]]:
    """Identify duplicate pairs (idx1, idx2, similarity_score) within the same employer.
    Requires matching normalized titles to prevent merging distinct roles (e.g. Server vs Host)."""
    embeddings = compute_job_embeddings(jobs)
    if embeddings is None:
        return []

    # Group by normalized company
    companies: dict[str, list[int]] = {}
    for idx, j in enumerate(jobs):
        comp = str(j.get("company") or "").strip().lower()
        if comp:
            companies.setdefault(comp, []).append(idx)

    duplicates: list[tuple[int, int, float]] = []
    for comp, indices in companies.items():
        if len(indices) < 2:
            continue
        sub_X = embeddings[indices]
        sim_matrix = np.dot(sub_X, sub_X.T)
        m = len(indices)
        for i in range(m):
            idx_i = indices[i]
            t_i = normalize_title_for_dedup(jobs[idx_i].get("title", ""))
            loc_i = str(jobs[idx_i].get("location") or "").strip().lower()
            for j in range(i + 1, m):
                idx_j = indices[j]
                t_j = normalize_title_for_dedup(jobs[idx_j].get("title", ""))
                loc_j = str(jobs[idx_j].get("location") or "").strip().lower()

                # Title guard: titles must be identical after normalization
                if t_i != t_j:
                    continue
                # Location guard: if both specify a location, they must match
                if loc_i and loc_j and loc_i != loc_j:
                    continue

                score = float(sim_matrix[i, j])
                if score >= threshold:
                    duplicates.append((idx_i, idx_j, score))

    duplicates.sort(key=lambda x: -x[2])
    return duplicates

