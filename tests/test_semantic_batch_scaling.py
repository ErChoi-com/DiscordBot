"""Semantic encode batch size scales to the host the same way every other
pool in this codebase does -- see services/capacity.py's module docstring.
Unlike the I/O pools, this one is a local torch allocation, so it is allowed
to scale up on bigger hardware too.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import job_service  # noqa: E402

KEYWORDS, LOCATION = "software intern", "Toronto"
SEARCH = job_service.search_text_for_semantic_match(KEYWORDS, LOCATION, [])


def _item(title="", description="", link="https://x/1", company="Acme"):
    return {"title": title, "description": description, "link": link, "company": company, "location": "Toronto"}


class _FakeModel:
    """Records the batch_size kwarg every encode call actually received."""

    def __init__(self) -> None:
        self.batch_sizes: list[int | None] = []

    def encode(self, texts, normalize_embeddings=True, batch_size=None):
        self.batch_sizes.append(batch_size)
        return np.array([[1.0, 0.0] for _ in texts])


@pytest.fixture
def model(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(job_service, "load_semantic_plugin_model", lambda: fake)
    return fake


def _fake_hardware(monkeypatch, cpus: float, memory_gb: float | None):
    monkeypatch.setattr(job_service.capacity, "cpu_limit", lambda: cpus)
    monkeypatch.setattr(
        job_service.capacity, "memory_limit_bytes",
        lambda: None if memory_gb is None else int(memory_gb * 1024 ** 3),
    )


# ---------------------------------------------------------------------------
# semantic_encode_batch() itself
# ---------------------------------------------------------------------------

def test_reference_hardware_leaves_the_tuned_batch_untouched(monkeypatch):
    _fake_hardware(monkeypatch, job_service.capacity.REFERENCE_CPUS,
                    job_service.capacity.REFERENCE_MEMORY_GB)
    assert job_service.semantic_encode_batch() == job_service.SEMANTIC_ENCODE_BATCH


def test_small_vps_scales_the_batch_down(monkeypatch):
    _fake_hardware(monkeypatch, 1, 1.0)  # 1 core, 1 GB
    batch = job_service.semantic_encode_batch()
    assert batch < job_service.SEMANTIC_ENCODE_BATCH


def test_batch_never_drops_below_the_floor(monkeypatch):
    _fake_hardware(monkeypatch, 0.1, 0.1)  # about as small as it gets
    assert job_service.semantic_encode_batch() >= 4


def test_large_box_scales_the_batch_up(monkeypatch):
    _fake_hardware(monkeypatch, 96, 256.0)
    batch = job_service.semantic_encode_batch()
    assert batch > job_service.SEMANTIC_ENCODE_BATCH


def test_batch_never_exceeds_the_ceiling(monkeypatch):
    _fake_hardware(monkeypatch, 10_000, 4_000.0)
    assert job_service.semantic_encode_batch() <= 128


# ---------------------------------------------------------------------------
# The encode call site actually uses the scaled value
# ---------------------------------------------------------------------------

def test_the_encode_call_uses_the_scaled_batch_not_the_bare_constant(model, monkeypatch):
    _fake_hardware(monkeypatch, 1, 1.0)
    expected = job_service.semantic_encode_batch()
    assert expected != job_service.SEMANTIC_ENCODE_BATCH

    items = [_item("Software Intern", "Software internship at Acme")]
    job_service.semantic_filter_items(items, KEYWORDS, LOCATION, [])

    assert model.batch_sizes == [expected]
