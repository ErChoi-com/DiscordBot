"""Tests for dynamic Nomic Matryoshka dimension resizing, resolution, and CLI integration."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.semantic.engine import MATRYOSHKA_DIMS, SemanticEngine, get_semantic_engine
from commands.handlers import extract_best_jobs_dimension, parse_best_jobs_payload
from services.job_match import ArchivedJob, ProfileSignal, judge_jobs_neural, best_jobs
from config import load_config, AppConfig


def test_config_semantic_dimension(tmp_path: Path, monkeypatch):
    """Verify semantic_dimension loads from env var and config."""
    monkeypatch.setenv("SEMANTIC_DIMENSION", "512")
    cfg = load_config()
    assert cfg.semantic_dimension == 512

    monkeypatch.delenv("SEMANTIC_DIMENSION", raising=False)
    cfg_default = load_config()
    assert cfg_default.semantic_dimension == 384


def test_dynamic_dimension_resolution_hierarchy():
    """Verify explicit > thread-local > workload > global default hierarchy."""
    engine = get_semantic_engine()

    # 1. Global default
    assert engine.resolve_dynamic_dimension() == 384

    # 2. Workload defaults
    assert engine.resolve_dynamic_dimension(workload="dedup") == 128
    assert engine.resolve_dynamic_dimension(workload="neural_judge") == 384
    assert engine.resolve_dynamic_dimension(workload="high_precision") == 768
    assert engine.resolve_dynamic_dimension(workload="ultra_fast") == 64

    # 3. Thread-local override via using_dimension
    with engine.using_dimension(256):
        assert engine.resolve_dynamic_dimension(workload="neural_judge") == 256
        assert engine.get_dimension() == 256

    # 4. Explicit requested_dim overrides everything
    assert engine.resolve_dynamic_dimension(workload="dedup", requested_dim=512) == 512

    # 5. Invalid dimensions raise ValueError
    with pytest.raises(ValueError, match="Invalid dimension"):
        engine.resolve_dynamic_dimension(requested_dim=999)


def test_dynamic_dimension_auto_scaling_under_batch_load():
    """Verify large batches automatically scale down to prevent OOM."""
    engine = get_semantic_engine()

    # Small batch maintains requested / default precision
    assert engine.resolve_dynamic_dimension(requested_dim=768, batch_size=10, auto_scale=True) == 768

    # Large batch (>=128) scales down to max 256
    assert engine.resolve_dynamic_dimension(requested_dim=768, batch_size=128, auto_scale=True) == 256
    assert engine.resolve_dynamic_dimension(requested_dim=384, batch_size=128, auto_scale=True) == 256

    # Huge batch (>=256) scales down to 128
    assert engine.resolve_dynamic_dimension(requested_dim=768, batch_size=256, auto_scale=True) == 128

    # When auto_scale=False, stays strictly at requested
    assert engine.resolve_dynamic_dimension(requested_dim=768, batch_size=256, auto_scale=False) == 768


def test_get_active_dimensions_diagnostics():
    """Verify get_active_dimensions returns full diagnostic state."""
    engine = get_semantic_engine()
    diag = engine.get_active_dimensions()

    assert "current_thread_dim" in diag
    assert "global_default_dim" in diag
    assert "workload_defaults" in diag
    assert diag["workload_defaults"]["dedup"] == 128
    assert set(diag["supported_dims"]) == set(MATRYOSHKA_DIMS)


def test_extract_best_jobs_dimension_cli_parsing():
    """Verify --dim flag is cleanly parsed without breaking argument order."""
    # 1. Standard --dim 768
    clean, dim, err = extract_best_jobs_dimension("week 10 ricky --dim 768 --fast")
    assert dim == 768
    assert err is None
    w, l, p, f, pe = parse_best_jobs_payload(clean)
    assert (w, l, p, f, pe) == ("week", 10, "ricky", False, None)

    # 2. Syntax with equals --dim=128
    clean, dim, err = extract_best_jobs_dimension("--dim=128 5")
    assert dim == 128
    assert err is None
    w, l, p, f, pe = parse_best_jobs_payload(clean)
    assert (w, l, p, f, pe) == ("day", 5, None, True, None)

    # 3. Invalid dimension returns clear error
    _, dim_bad, err_bad = extract_best_jobs_dimension("week --dim 999")
    assert dim_bad is None
    assert "Invalid dimension '999'" in err_bad

    # 4. No dimension specified leaves payload intact
    clean_none, dim_none, err_none = extract_best_jobs_dimension("week 15")
    assert dim_none is None
    assert err_none is None
    assert clean_none == "week 15"


def test_neural_judge_explicit_dimensions():
    """Verify judge_jobs_neural executes properly across different dynamically passed dimensions."""
    signal = ProfileSignal(
        profile_key="test",
        anchors=("python", "fastapi"),
        role_terms=("backend engineer",),
        seniority="mid",
        locations=(),
        document_text="Mid-level backend engineer specializing in Python, FastAPI, and Docker.",
    )
    jobs = [
        ArchivedJob(
            title="Senior Python Backend Developer",
            company="Tech Corp",
            location="Remote",
            link="https://example.com/1",
            site_label="LinkedIn",
            date_posted="2026-08-25",
            description="Build scalable FastAPI microservices with PostgreSQL and Docker.",
        )
    ]

    # Test at 128-d (fast/compact)
    v128, r128 = judge_jobs_neural(signal, jobs, dim=128)
    assert r128 == "ok:nomic-neural"
    assert len(v128) == 1
    assert v128[0].score > 0.50

    # Test at 768-d (high-precision master)
    v768, r768 = judge_jobs_neural(signal, jobs, dim=768)
    assert r768 == "ok:nomic-neural"
    assert len(v768) == 1
    assert v768[0].score > 0.50
