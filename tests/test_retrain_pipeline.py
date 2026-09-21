from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import pipeline_data_sync, relabel_pieces
from services.resumes import posting_classifier


def test_clean_text_mojibake():
    dirty = "That\ufffd\u2019s why you\u2019ll love working here."
    cleaned = relabel_pieces.clean_text(dirty)
    assert cleaned == "That's why you'll love working here."


def test_extract_terms_for_piece():
    text = "We require 3+ years experience with Python, PyTorch, and Docker Compose for CI/CD pipelines."
    terms = relabel_pieces.extract_terms_for_piece(text, "SKILL_DUTY")
    assert "Python" in terms or "python" in [t.lower() for t in terms]
    assert "PyTorch" in terms or "pytorch" in [t.lower() for t in terms]
    assert "Docker Compose" in terms or "docker compose" in [t.lower() for t in terms]
    assert "CI/CD" in terms or "ci/cd" in [t.lower() for t in terms]
    
    # Verify stopwords are NOT extracted
    for sw in ["experience", "team", "work"]:
        assert sw not in [t.lower() for t in terms]

    # Verify BOILERPLATE returns empty
    bp_terms = relabel_pieces.extract_terms_for_piece(text, "BOILERPLATE")
    assert bp_terms == []


def test_validate_dataset_pass(tmp_path: Path):
    sample_file = tmp_path / "pieces_sample.jsonl"
    rows = [
        {
            "pid": "pid_1",
            "company": "Company A",
            "idx": 0,
            "text": "Develop microservices using Go and Kubernetes.",
            "label": "SKILL_DUTY",
            "terms": ["Go", "Kubernetes"],
        },
        {
            "pid": "pid_1",
            "company": "Company A",
            "idx": 1,
            "text": "Equal Opportunity Employer.",
            "label": "BOILERPLATE",
            "terms": [],
        },
    ]
    sample_file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    report = pipeline_data_sync.validate_dataset(sample_file)
    assert report["status"] == "PASS"
    assert report["total_pieces"] == 2
    assert report["pieces_with_terms"] == 1
    assert report["total_terms"] == 2
    assert report["term_span_mismatches"] == 0


def test_validate_dataset_detects_mismatch(tmp_path: Path):
    sample_file = tmp_path / "pieces_bad.jsonl"
    rows = [
        {
            "pid": "pid_2",
            "company": "Company B",
            "idx": 0,
            "text": "Develop microservices.",
            "label": "SKILL_DUTY",
            "terms": ["Rust"],  # Rust is NOT in text!
        },
    ]
    sample_file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    report = pipeline_data_sync.validate_dataset(sample_file)
    assert report["status"] == "FAIL"
    assert report["term_span_mismatches"] == 1


def test_check_employer_split(tmp_path: Path):
    sample_file = tmp_path / "pieces_split.jsonl"
    rows = [
        {"pid": f"p_{i}", "company": f"Comp_{i % 10}", "idx": 0, "text": f"Task {i}", "label": "SKILL_DUTY", "terms": []}
        for i in range(100)
    ]
    sample_file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    split = pipeline_data_sync.check_employer_split(sample_file, seed=42, test_frac=0.3)
    assert split["zero_leakage"] is True
    assert split["overlap_count"] == 0
    assert split["train_pieces"] + split["test_pieces"] == 100


def test_extract_posting_terms_mocked(monkeypatch):
    class Tokenizer:
        def __call__(self, pieces, **kwargs):
            return {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "offset_mapping": [[[0, 6], [7, 13], [14, 20]]],
            }

    class Model:
        def __call__(self, **kwargs):
            import torch
            piece_logits = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
            # Predict B-TERM (1) on first word, O (0) elsewhere
            term_logits = torch.tensor([[[0.0, 10.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
            return piece_logits, term_logits

    monkeypatch.setattr(
        posting_classifier,
        "load_posting_classifier",
        lambda: (Model(), Tokenizer(), posting_classifier.LABELS, 128),
    )
    terms = posting_classifier.extract_posting_terms("Python coding skills.")
    assert "Python" in terms


def test_asymmetric_focal_loss():
    import torch
    from scripts.posting_train import AsymmetricFocalLoss, LABELS, BP

    loss_fn = AsymmetricFocalLoss(gamma=2.0, asym_fp_penalty=3.0)
    
    # 1. Easy example: high probability on target class
    logits_easy = torch.tensor([[10.0, -5.0, -5.0, -5.0]])
    target_easy = torch.tensor([0])
    loss_easy = loss_fn(logits_easy, target_easy)
    
    # 2. Hard example: uncertain prediction
    logits_hard = torch.tensor([[0.5, 0.4, 0.1, 0.0]])
    target_hard = torch.tensor([0])
    loss_hard = loss_fn(logits_hard, target_hard)
    
    # Focal loss property: hard example loss >> easy example loss
    assert loss_hard > loss_easy * 50

    # 3. Asymmetric penalty test: content wrongly predicted as BOILERPLATE
    # Target is SKILL_DUTY (0), but logits predict BOILERPLATE (3)
    logits_wrong_bp = torch.tensor([[-5.0, -5.0, -5.0, 10.0]])
    loss_wrong_bp = loss_fn(logits_wrong_bp, torch.tensor([0]))
    
    # Target is SKILL_DUTY (0), but logits predict ROLE_FACTS (1)
    logits_wrong_role = torch.tensor([[-5.0, 10.0, -5.0, -5.0]])
    loss_wrong_role = loss_fn(logits_wrong_role, torch.tensor([0]))
    
    # False boilerplate prediction should have higher loss due to asym_fp_penalty
    assert loss_wrong_bp > loss_wrong_role * 2.0
