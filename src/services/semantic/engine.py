"""Pure Nomic Semantic Engine (nomic-embed-text-v1.5) with Dynamic Matryoshka Slicing.

Backed by Nomic's official local runtime (`nomic[local]` / Embed4All) with
unquantized float16 weights (`nomic-embed-text-v1.5.f16.gguf`) and built-in
SHA-256 checksum verification.

Supports dynamic swapping of Matryoshka dimensions (768, 512, 384, 256, 128, 64)
in and out at runtime with zero model reloads, enabling lightweight vector
storage for batch ingestion alongside full-precision representations for final
match scoring.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from services import capacity
from services.semantic.prompts import (
    BOILERPLATE_ANCHOR,
    DEFAULT_EMBEDDING_DIM,
    DUTY_ANCHOR,
    GGUF_MODEL_FILE,
    HF_MODEL_ID,
    MASTER_EMBEDDING_DIM,
    MATRYOSHKA_DIMS,
    MODEL_ID,
    PREFIX_CLASSIFY,
    PREFIX_SEARCH_DOC,
    PREFIX_SEARCH_QUERY,
    SENIORITY_ANCHORS,
    SENIORITY_LABELS,
    SKILL_ANCHOR,
    TASK_CLASSIFICATION,
    TASK_SEARCH_DOC,
    TASK_SEARCH_QUERY,
)

# Minimum memory in GB required for Nomic engine
MIN_MEMORY_GB = 1.0


def get_gpt4all_cache_dir() -> Path:
    """Return the default cache directory for gpt4all / nomic models."""
    return Path.home() / ".cache" / "gpt4all"


def is_nomic_weight_cached() -> bool:
    """Check if the verified nomic-embed-text-v1.5 f16 GGUF exists locally."""
    target = get_gpt4all_cache_dir() / GGUF_MODEL_FILE
    return target.is_file() and target.stat().st_size > 200_000_000


class SemanticEngine:
    """Pure Nomic semantic inference engine with dynamic Matryoshka dimension and weight swapping."""

    _instance: SemanticEngine | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._current_dim: int = DEFAULT_EMBEDDING_DIM
        self._active_backend: str = "none"
        self._active_task: str = "seniority"
        self._transformer_model: Any | None = None
        self._inference_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._local = threading.local()

        # In-memory LRU embedding cache: (text, dim, task_type, normalize) -> np.ndarray
        self._embedding_cache: dict[tuple[str, int, str, bool], np.ndarray] = {}
        self._max_cache_entries: int = 4096

        # Master anchor embeddings computed at 768-d, sliced dynamically for any d in MATRYOSHKA_DIMS
        self._master_seniority_anchors: tuple[np.ndarray, list[str]] | None = None
        self._master_bp_anchors: np.ndarray | None = None
        self._master_skill_anchor: np.ndarray | None = None

        # Sliced anchor cache: (anchor_name, dim) -> np.ndarray
        self._cached_anchor_slices: dict[tuple[str, int], Any] = {}

        # Converted Matryoshka Task Weights Registry
        # Stores master weights W in R^{C x 768}, bias b in R^C, and class labels
        self._task_weights: dict[str, dict[str, Any]] = {}
        # Sliced task weights cache: (task_name, dim) -> (W_d, b, labels)
        self._sliced_task_weights_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, list[str]]] = {}

        self._load_default_task_weights()

    @classmethod
    def instance(cls) -> SemanticEngine:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ========================================================================
    # Dynamic Task Weights & Swapping Mechanism
    # ========================================================================
    def _load_default_task_weights(self) -> None:
        """Auto-discover and register master Matryoshka weights from both datasets."""
        root = Path(__file__).resolve().parents[3]
        seniority_weights = root / "models" / "seniority_encoder" / "master_matryoshka_weights.pt"
        if seniority_weights.is_file():
            self.load_task_weights("seniority", seniority_weights)

        posting_weights = root / "models" / "posting_encoder" / "master_matryoshka_weights.pt"
        if posting_weights.is_file():
            self.load_task_weights("posting", posting_weights)

    def load_task_weights(self, task_name: str, path: str | Path) -> bool:
        """Register converted master Matryoshka weights (W in R^{C x 768}, b in R^C)."""
        pt_path = Path(path)
        if not pt_path.is_file():
            return False
        try:
            import torch
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            w = data["weight"].float().numpy()
            b = data.get("bias")
            b_np = b.float().numpy() if b is not None else np.zeros(w.shape[0], dtype=np.float32)
            classes = list(data.get("classes", []))
            self._task_weights[task_name] = {
                "weight": w,
                "bias": b_np,
                "classes": classes,
                "master_dim": int(data.get("master_dim", w.shape[-1])),
            }
            # Clear slice cache for this task
            for k in list(self._sliced_task_weights_cache.keys()):
                if k[0] == task_name:
                    del self._sliced_task_weights_cache[k]
            return True
        except Exception:
            return False

    def has_task_weights(self, task_name: str) -> bool:
        """Check if converted weights are registered for a task."""
        return task_name in self._task_weights

    def get_registered_tasks(self) -> list[str]:
        """List all currently registered Matryoshka tasks with converted weights."""
        return list(self._task_weights.keys())

    def get_task_weights(
        self, task_name: str, dim: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
        """Dynamically slice task weights down to target Matryoshka dimension."""
        if task_name not in self._task_weights:
            return None
        target_dim = dim if dim is not None else self._current_dim
        cached = self._sliced_task_weights_cache.get((task_name, target_dim))
        if cached is not None:
            return cached

        entry = self._task_weights[task_name]
        w_master = entry["weight"]
        b = entry["bias"]
        classes = entry["classes"]

        # Slice to target dimension
        w_sliced = w_master[:, :target_dim].copy()
        result = (w_sliced, b, classes)
        self._sliced_task_weights_cache[(task_name, target_dim)] = result
        return result

    def predict_task(
        self,
        texts: str | list[str],
        task_name: str,
        dim: int | None = None,
    ) -> list[tuple[str, float]] | tuple[str, float] | None:
        """Perform classification using dynamically sliced task weights."""
        target_dim = dim if dim is not None else self.get_dimension()
        weights_data = self.get_task_weights(task_name, dim=target_dim)
        if weights_data is None:
            return None

        w_d, b, classes = weights_data
        single = isinstance(texts, str)
        text_list = [texts] if single else texts
        if not text_list:
            return [] if not single else None

        vecs = self.encode(text_list, dim=target_dim, task_type=TASK_CLASSIFICATION, normalize=True)
        if vecs is None:
            return None
        if vecs.ndim == 1:
            vecs = vecs[np.newaxis, :]

        # logits = vecs @ w_d.T + b
        logits = np.dot(vecs, w_d.T) + b
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        results: list[tuple[str, float]] = []
        for p_row in probs:
            best_idx = int(np.argmax(p_row))
            results.append((classes[best_idx], float(p_row[best_idx])))

        return results[0] if single else results

    @contextmanager
    def using_task(self, task_name: str, dim: int | None = None) -> Iterator[SemanticEngine]:
        """Context manager allowing temporary thread-isolated task weight and dimension swapping."""
        old_task = getattr(self._local, "task", self._active_task)
        old_dim = self.get_dimension()
        self._local.task = task_name
        if dim is not None:
            self.set_dimension(dim)
        try:
            yield self
        finally:
            self._local.task = old_task
            self.set_dimension(old_dim)

    # ========================================================================
    # Matryoshka Dimension Swapping & Dynamic Resizing
    # ========================================================================
    DEFAULT_WORKLOAD_DIMENSIONS: dict[str, int] = {
        "dedup": 128,
        "watcher": 384,
        "search": 384,
        "neural_judge": 384,
        "bestjobs": 384,
        "classification": 384,
        "posting": 384,
        "seniority": 384,
        "high_precision": 768,
        "ultra_fast": 64,
    }

    def get_dimension(self) -> int:
        """Return the current active Matryoshka embedding dimension for this thread."""
        return getattr(self._local, "dim", self._current_dim)

    def set_dimension(self, dim: int) -> None:
        """Swap the active Matryoshka dimension for this thread."""
        if dim not in MATRYOSHKA_DIMS:
            raise ValueError(
                f"Invalid dimension {dim}. Supported Matryoshka dimensions: {MATRYOSHKA_DIMS}"
            )
        self._local.dim = dim

    def set_global_default_dimension(self, dim: int) -> None:
        """Update the process-wide default Matryoshka dimension."""
        if dim not in MATRYOSHKA_DIMS:
            raise ValueError(
                f"Invalid dimension {dim}. Supported Matryoshka dimensions: {MATRYOSHKA_DIMS}"
            )
        self._current_dim = dim

    def resolve_dynamic_dimension(
        self,
        workload: str | None = None,
        batch_size: int = 1,
        requested_dim: int | None = None,
        auto_scale: bool = True,
    ) -> int:
        """Dynamically resolve the optimal Matryoshka dimension for a request.

        Resolution hierarchy:
        1. Explicitly requested dimension (if specified)
        2. Thread-local dimension (if set via using_dimension / set_dimension)
        3. Workload-specific dimension (e.g. dedup -> 128, watcher -> 384, high_precision -> 768)
        4. Global default dimension (self._current_dim)

        Auto-scaling (when auto_scale=True):
        - If batch_size >= 128 or host memory is constrained, dynamically downscales
          to a more compact dimension tier to maximize throughput and prevent OOM.
        """
        # 1. Explicit request
        if requested_dim is not None:
            if requested_dim not in MATRYOSHKA_DIMS:
                raise ValueError(
                    f"Invalid dimension {requested_dim}. Supported Matryoshka dimensions: {MATRYOSHKA_DIMS}"
                )
            target = requested_dim
        # 2. Thread-local override
        elif hasattr(self._local, "dim"):
            target = self._local.dim
        # 3. Workload default
        elif workload and workload.lower() in self.DEFAULT_WORKLOAD_DIMENSIONS:
            target = self.DEFAULT_WORKLOAD_DIMENSIONS[workload.lower()]
        # 4. Global process default
        else:
            target = self._current_dim

        # 5. Dynamic auto-scaling under load
        if auto_scale:
            # Massive batch: downscale high dimensions to conserve throughput
            if batch_size >= 256 and target > 128:
                target = 128
            elif batch_size >= 128 and target > 256:
                target = 256
            # Low host headroom: downscale one tier
            elif not capacity.can_afford(MIN_MEMORY_GB + 1.0) and target > 128:
                idx = MATRYOSHKA_DIMS.index(target) if target in MATRYOSHKA_DIMS else 2
                target = MATRYOSHKA_DIMS[min(len(MATRYOSHKA_DIMS) - 1, idx + 1)]

        return target

    def get_active_dimensions(self) -> dict[str, Any]:
        """Return diagnostic dictionary of active dimensions across threads and workloads."""
        return {
            "current_thread_dim": self.get_dimension(),
            "global_default_dim": self._current_dim,
            "workload_defaults": dict(self.DEFAULT_WORKLOAD_DIMENSIONS),
            "supported_dims": list(MATRYOSHKA_DIMS),
        }

    @contextmanager
    def using_dimension(self, dim: int) -> Iterator[SemanticEngine]:
        """Context manager allowing temporary dimension swapping without modifying global state.

        Example:
            with engine.using_dimension(128):
                embeddings = engine.compute_job_embeddings(large_batch)
        """
        old_dim = self.get_dimension()
        self.set_dimension(dim)
        try:
            yield self
        finally:
            self.set_dimension(old_dim)

    # ========================================================================
    # Backend Encoding Pipeline
    # ========================================================================
    def _ensure_backend_available(self) -> bool:
        """Verify host memory capacity before launching inference."""
        if not capacity.can_afford(MIN_MEMORY_GB):
            return False
        return True

    def _encode_via_nomic_sdk(
        self,
        texts: list[str],
        dim: int,
        task_type: str = TASK_SEARCH_DOC,
    ) -> np.ndarray | None:
        """Encode texts using official nomic.embed local engine."""
        try:
            from nomic import embed

            res = embed.text(
                texts,
                model=MODEL_ID,
                dimensionality=dim,
                task_type=task_type,
                inference_mode="local",
            )
            raw = res.get("embeddings")
            if raw is None or len(raw) == 0:
                return None
            vecs = np.asarray(raw, dtype=np.float32)
            self._active_backend = "nomic-local"
            return vecs
        except Exception:
            return None

    def _encode_via_sentence_transformer(
        self,
        texts: list[str],
        dim: int,
        task_type: str = TASK_SEARCH_DOC,
        batch_size: int = 64,
    ) -> np.ndarray | None:
        """Portable fallback using SentenceTransformer with explicit Matryoshka slicing."""
        try:
            from sentence_transformers import SentenceTransformer

            if self._transformer_model is None:
                import torch
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._transformer_model = SentenceTransformer(
                    HF_MODEL_ID, trust_remote_code=True, device=device
                )
                try:
                    num_threads = max(1, (os.cpu_count() or 4) - 2)
                    torch.set_num_threads(num_threads)
                except Exception:
                    pass

            prefix = ""
            if task_type == TASK_SEARCH_DOC:
                prefix = PREFIX_SEARCH_DOC
            elif task_type == TASK_SEARCH_QUERY:
                prefix = PREFIX_SEARCH_QUERY
            elif task_type == TASK_CLASSIFICATION:
                prefix = PREFIX_CLASSIFY

            prefixed = [f"{prefix}{t}" for t in texts]

            # Try encoding with given batch_size; if OOM occurs, fall back to sub-batches of 16
            try:
                raw = self._transformer_model.encode(
                    prefixed,
                    batch_size=batch_size,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
            except Exception as oom_exc:
                if "out of memory" in str(oom_exc).lower():
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    sub_batches = []
                    sub_sz = max(8, batch_size // 4)
                    for s_idx in range(0, len(prefixed), sub_sz):
                        chunk = prefixed[s_idx : s_idx + sub_sz]
                        sub_batches.append(
                            self._transformer_model.encode(
                                chunk,
                                batch_size=sub_sz,
                                normalize_embeddings=False,
                                show_progress_bar=False,
                            )
                        )
                    raw = np.vstack(sub_batches)
                else:
                    raise oom_exc

            vecs = np.asarray(raw, dtype=np.float32)
            if vecs.ndim == 1:
                vecs = vecs[np.newaxis, :]

            # Matryoshka dimension truncation
            if dim is not None and vecs.shape[-1] > dim:
                vecs = vecs[:, :dim]

            self._active_backend = "nomic-transformer"
            return vecs
        except Exception as exc:
            print(f"[SemanticEngine] Transformer fallback failed: {exc}")
            return None

    def encode(
        self,
        texts: str | list[str],
        dim: int | None = None,
        task_type: str = TASK_SEARCH_DOC,
        normalize: bool = True,
        batch_size: int = 64,
        normalize_embeddings: bool | None = None,
        **kwargs: Any,
    ) -> np.ndarray | None:
        """Encode texts into normalized L2 embeddings, sliced to dim via Matryoshka."""
        if normalize_embeddings is not None:
            normalize = normalize_embeddings

        if not self._ensure_backend_available():
            return None

        target_dim = dim if dim is not None else self.get_dimension()
        single = isinstance(texts, str)
        text_list = [texts] if single else texts
        if not text_list:
            return np.empty((0, target_dim), dtype=np.float32)

        # LRU cache optimization for single text queries (thread-safe)
        if single:
            cache_key = (texts, target_dim, task_type, normalize)
            with self._cache_lock:
                cached_vec = self._embedding_cache.get(cache_key)
                if cached_vec is not None:
                    return cached_vec.copy()

        with self._inference_lock:
            # 1. Primary: Nomic SDK local engine (gpt4all / f16.gguf)
            vecs = self._encode_via_nomic_sdk(text_list, dim=target_dim, task_type=task_type)
            # 2. Fallback: SentenceTransformer with manual Matryoshka slicing
            if vecs is None:
                vecs = self._encode_via_sentence_transformer(
                    text_list, dim=target_dim, task_type=task_type, batch_size=batch_size
                )

            if vecs is None:
                return None

            # Always project to unit sphere for cosine distance
            if normalize:
                norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                vecs = vecs / norms

            if single:
                with self._cache_lock:
                    if len(self._embedding_cache) >= self._max_cache_entries:
                        keys_to_evict = list(self._embedding_cache.keys())[: self._max_cache_entries // 4]
                        for k in keys_to_evict:
                            self._embedding_cache.pop(k, None)
                    self._embedding_cache[(texts, target_dim, task_type, normalize)] = vecs[0].copy()
                return vecs[0]
            return vecs

    # ========================================================================
    # Dynamic Anchor Slicing (Zero re-computation across swapped dimensions)
    # ========================================================================
    def _get_master_seniority_anchors(self) -> tuple[np.ndarray, list[str]] | None:
        if self._master_seniority_anchors is not None:
            return self._master_seniority_anchors

        labels = list(SENIORITY_LABELS)
        anchor_texts = [SENIORITY_ANCHORS[l] for l in labels]
        # Embed at full master dimension (768-d)
        vecs = self.encode(anchor_texts, dim=MASTER_EMBEDDING_DIM, task_type=TASK_CLASSIFICATION, normalize=False)
        if vecs is not None:
            self._master_seniority_anchors = (vecs, labels)
        return self._master_seniority_anchors

    def _get_seniority_anchors_for_dim(self, dim: int) -> tuple[np.ndarray, list[str]] | None:
        """Dynamically slice master seniority anchors to target dimension with unit normalization."""
        cache_key = ("seniority", dim)
        if cache_key in self._cached_anchor_slices:
            return self._cached_anchor_slices[cache_key]

        master = self._get_master_seniority_anchors()
        if master is None:
            return None
        vecs, labels = master
        sliced = vecs[:, :dim]
        norms = np.linalg.norm(sliced, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        res = (sliced / norms, labels)
        self._cached_anchor_slices[cache_key] = res
        return res

    def _get_master_bp_anchors(self) -> np.ndarray | None:
        if self._master_bp_anchors is not None:
            return self._master_bp_anchors
        vecs = self.encode(
            [BOILERPLATE_ANCHOR, DUTY_ANCHOR],
            dim=MASTER_EMBEDDING_DIM,
            task_type=TASK_CLASSIFICATION,
            normalize=False,
        )
        if vecs is not None:
            self._master_bp_anchors = vecs
        return self._master_bp_anchors

    def _get_bp_anchors_for_dim(self, dim: int) -> np.ndarray | None:
        """Dynamically slice master boilerplate anchors to target dimension with unit normalization."""
        cache_key = ("bp", dim)
        if cache_key in self._cached_anchor_slices:
            return self._cached_anchor_slices[cache_key]

        master = self._get_master_bp_anchors()
        if master is None:
            return None
        sliced = master[:, :dim]
        norms = np.linalg.norm(sliced, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        res = sliced / norms
        self._cached_anchor_slices[cache_key] = res
        return res

    # ========================================================================
    # Task 1: Seniority Classification (Dynamic Swapped Weights or Anchors)
    # ========================================================================
    def classify_seniority(self, title: str, dim: int | None = None) -> tuple[str, float] | None:
        """Classify job title into one of the 6 canonical seniority tiers at dimension dim."""
        if not title or not title.strip():
            return None

        target_dim = dim if dim is not None else self.get_dimension()

        # 1. Primary: Use trained Matryoshka seniority weights if available (and not in legacy mock test mode)
        if "seniority" in self._task_weights and not hasattr(self, "_cached_seniority_anchors"):
            pred = self.predict_task(title.strip(), "seniority", dim=target_dim)
            if pred is not None and isinstance(pred, tuple):
                return pred

        # 2. Fallback to anchor cosine similarity
        if hasattr(self, "_cached_seniority_anchors"):
            anchors_data = self._cached_seniority_anchors
        else:
            anchors_data = self._get_seniority_anchors_for_dim(target_dim)
        if anchors_data is None:
            return None
        anchor_vecs, labels = anchors_data

        title_vec = self.encode(title.strip(), dim=target_dim, task_type=TASK_CLASSIFICATION)
        if title_vec is None:
            return None

        scores = np.squeeze(np.dot(anchor_vecs, title_vec))
        best_idx = int(np.argmax(scores))
        best_val = scores[best_idx]
        best_score = float(best_val.item() if hasattr(best_val, "item") else best_val)
        return labels[best_idx], best_score

    # ========================================================================
    # Task 2: Scraper Noise & Boilerplate Filtering
    # ========================================================================
    def classify_posting_piece(self, piece: str, dim: int | None = None) -> tuple[str, float] | None:
        """Classify a posting piece into SKILL_DUTY, BOILERPLATE, ROLE_FACTS, or COMPANY_CONTEXT."""
        if not piece or not piece.strip():
            return None
        target_dim = dim if dim is not None else self.get_dimension()
        if "posting" in self._task_weights:
            pred = self.predict_task(piece.strip(), "posting", dim=target_dim)
            if pred is not None and isinstance(pred, tuple):
                return pred
        return None

    def filter_boilerplate(
        self,
        chunks: list[str],
        threshold: float = 0.15,
        dim: int | None = None,
    ) -> list[str]:
        """Filter out boilerplate/EEO chunks at dimension dim using converted weights or anchors."""
        if not chunks:
            return []

        target_dim = dim if dim is not None else self.get_dimension()

        # 1. Primary: Use trained Matryoshka posting piece weights if available (and not in legacy mock test mode)
        if "posting" in self._task_weights and not hasattr(self, "_cached_bp_anchors"):
            preds = self.predict_task([c.strip() for c in chunks], "posting", dim=target_dim)
            if preds is not None and isinstance(preds, list):
                kept = []
                for orig, (label, prob) in zip(chunks, preds):
                    # Only drop if explicitly predicted as BOILERPLATE with sufficient confidence
                    if label == "BOILERPLATE" and prob >= 0.60:
                        continue
                    kept.append(orig)
                return kept

        # 2. Fallback to anchor cosine similarity
        if hasattr(self, "_cached_bp_anchors"):
            bp_anchors = self._cached_bp_anchors
        else:
            bp_anchors = self._get_bp_anchors_for_dim(target_dim)
        if bp_anchors is None:
            return chunks
        bp_vec, duty_vec = bp_anchors[0], bp_anchors[1]

        chunk_vecs = self.encode([c.strip() for c in chunks], dim=target_dim, task_type=TASK_CLASSIFICATION)
        if chunk_vecs is None:
            return chunks

        kept: list[str] = []
        for orig, vec in zip(chunks, chunk_vecs):
            bp_sim = float(np.dot(bp_vec, vec))
            duty_sim = float(np.dot(duty_vec, vec))
            if (bp_sim - duty_sim) < threshold:
                kept.append(orig)
        return kept

    # ========================================================================
    # Task 3: Cross-ATS Semantic Deduplication Embeddings
    # ========================================================================
    def compute_job_embeddings(
        self,
        jobs: list[dict[str, Any]],
        dims: int | None = None,
    ) -> np.ndarray | None:
        """Embed full job postings for cross-board deduplication at dimension dims."""
        if not jobs:
            return None

        target_dim = self.resolve_dynamic_dimension(
            workload="dedup", batch_size=len(jobs), requested_dim=dims
        )
        texts = []
        for j in jobs:
            title = str(j.get("title") or "").strip()
            company = str(j.get("company") or "").strip()
            location = str(j.get("location") or "").strip()
            desc = str(j.get("description") or j.get("snippet") or j.get("summary") or "").strip()
            canonical = " | ".join(p for p in [title, company, location, desc] if p)
            texts.append(canonical)

        return self.encode(texts, dim=target_dim, task_type=TASK_SEARCH_DOC, normalize=True)

    # ========================================================================
    # Task 4: Resume-to-Job Semantic Match Scoring
    # ========================================================================
    def score_resume_fit(
        self,
        resume_text: str,
        job_text: str,
        dim: int | None = None,
    ) -> float:
        """Score alignment between resume and job description at dimension dim."""
        if not resume_text.strip() or not job_text.strip():
            return 0.5

        target_dim = self.resolve_dynamic_dimension(
            workload="neural_judge", batch_size=1, requested_dim=dim
        )
        r_vec = self.encode(resume_text.strip(), dim=target_dim, task_type=TASK_SEARCH_QUERY)
        j_vec = self.encode(job_text.strip(), dim=target_dim, task_type=TASK_SEARCH_DOC)
        if r_vec is None or j_vec is None:
            return 0.5

        score = float(np.dot(r_vec, j_vec))
        return max(0.0, min(1.0, score))

    def score_resume_fit_batch(
        self,
        resume_text: str,
        job_texts: list[str],
        dim: int | None = None,
    ) -> list[float]:
        """Score alignment between one resume and multiple job descriptions in one pass."""
        if not resume_text.strip() or not job_texts:
            return [0.5] * len(job_texts)

        target_dim = self.resolve_dynamic_dimension(
            workload="neural_judge", batch_size=len(job_texts), requested_dim=dim
        )
        r_vec = self.encode(resume_text.strip(), dim=target_dim, task_type=TASK_SEARCH_QUERY, normalize=True)
        if r_vec is None:
            return [0.5] * len(job_texts)

        clean_jobs = [j.strip() if j.strip() else "Job Position" for j in job_texts]
        j_vecs = self.encode(clean_jobs, dim=target_dim, task_type=TASK_SEARCH_DOC, normalize=True)
        if j_vecs is None:
            return [0.5] * len(job_texts)

        scores = np.dot(j_vecs, r_vec)
        return [float(max(0.0, min(1.0, s))) for s in scores]

    def active_backend(self) -> str:
        return self._active_backend


def get_semantic_engine() -> SemanticEngine:
    return SemanticEngine.instance()
