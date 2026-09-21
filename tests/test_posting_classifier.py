from services.resumes import posting_classifier


def test_classifier_fails_open_without_an_artifact(monkeypatch):
    monkeypatch.setattr(posting_classifier, "load_posting_classifier", lambda: None)
    text = "Build Python services. Equal opportunity employer."
    assert posting_classifier.retain_resume_relevant_text(text) == text


def test_classifier_removes_only_high_confidence_boilerplate(monkeypatch):
    class Tokenizer:
        def __call__(self, pieces, **kwargs):
            return {"input_ids": pieces, "attention_mask": pieces}

    class Model:
        def __call__(self, **kwargs):
            import torch
            return torch.tensor([[8.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 8.0]])

    monkeypatch.setattr(
        posting_classifier,
        "load_posting_classifier",
        lambda: (Model(), Tokenizer(), posting_classifier.LABELS, 128),
    )
    text = "Build Python services. Equal opportunity employer."
    assert posting_classifier.retain_resume_relevant_text(text) == "Build Python services."


def test_classifier_drops_all_boilerplate_when_every_piece_is_confidently_boilerplate(monkeypatch):
    class Tokenizer:
        def __call__(self, pieces, **kwargs):
            return {"input_ids": pieces, "attention_mask": pieces}

    class Model:
        def __call__(self, **kwargs):
            import torch
            return torch.tensor([[0.0, 0.0, 0.0, 10.0], [0.0, 0.0, 0.0, 10.0]])

    monkeypatch.setattr(
        posting_classifier,
        "load_posting_classifier",
        lambda: (Model(), Tokenizer(), posting_classifier.LABELS, 128),
    )
    text = "Equal opportunity employer. We value work-life balance."
    assert posting_classifier.retain_resume_relevant_text(text, threshold=0.5) == ""


def test_default_threshold_is_calibrated_to_point_seven():
    assert posting_classifier.DEFAULT_THRESHOLD == 0.70


def test_snap_to_word_boundary_expands_subwords():
    text = "Streamed continuous Kafka data to Altera FPGA with custom STM32-based controller."
    # 'Ka' at 20:22 should expand to 'Kafka'
    assert posting_classifier._snap_to_word_boundary(text, 20, 22) == "Kafka"
    # 'PG' at 41:43 in FPGA should expand to 'FPGA'
    assert posting_classifier._snap_to_word_boundary(text, 41, 43) == "FPGA"
    # 'STM32' should strip -based
    assert posting_classifier._snap_to_word_boundary(text, 58, 63) == "STM32"


def test_extract_posting_terms_and_skills(monkeypatch):
    import torch

    class Tokenizer:
        def __call__(self, pieces, **kwargs):
            # 6 tokens: [CLS, Python, and, Kafka, ., SEP]
            offsets = torch.tensor([
                [[0, 0], [0, 6], [7, 10], [11, 16], [16, 17], [0, 0]]
            ])
            return {
                "input_ids": torch.zeros((1, 6), dtype=torch.long),
                "attention_mask": torch.ones((1, 6), dtype=torch.long),
                "offset_mapping": offsets,
            }

    class Model:
        def __call__(self, **kwargs):
            # Piece logits: SKILL_DUTY
            piece_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
            # Term tags: [0, B-TERM (1), 0, B-TERM (1), 0, 0]
            term_logits = torch.zeros((1, 6, 3))
            term_logits[0, 1, 1] = 10.0  # Python -> B-TERM
            term_logits[0, 3, 1] = 10.0  # Kafka -> B-TERM
            return piece_logits, term_logits

    monkeypatch.setattr(
        posting_classifier,
        "load_posting_classifier",
        lambda: (Model(), Tokenizer(), posting_classifier.LABELS, 128),
    )
    sample_text = "Python and Kafka."
    terms = posting_classifier.extract_posting_terms(sample_text)
    assert "Python" in terms
    assert "Kafka" in terms

    # extract_resume_skills calls extract_posting_terms
    skills = posting_classifier.extract_resume_skills(sample_text)
    assert skills == terms


def test_asymmetric_focal_loss_multiplier():
    import torch
    from scripts.posting_train import AsymmetricFocalLoss

    loss_fn = AsymmetricFocalLoss(gamma=2.0, asym_fp_penalty=2.5)

    # 1. Target is SKILL_DUTY (0), but prediction is BOILERPLATE (3)
    content_wrong_logits = torch.tensor([[-5.0, -5.0, -5.0, 5.0]])
    target_content = torch.tensor([0])
    loss_penalized = loss_fn(content_wrong_logits, target_content).item()

    # 2. Same logits and loss without penalty (asym_fp_penalty=1.0)
    loss_fn_baseline = AsymmetricFocalLoss(gamma=2.0, asym_fp_penalty=1.0)
    loss_baseline = loss_fn_baseline(content_wrong_logits, target_content).item()

    assert abs(loss_penalized - 2.5 * loss_baseline) < 1e-4

    # 3. If prediction is correct (not BP), penalty is NOT applied
    content_correct_logits = torch.tensor([[5.0, -5.0, -5.0, -5.0]])
    loss_correct = loss_fn(content_correct_logits, target_content).item()
    loss_correct_base = loss_fn_baseline(content_correct_logits, target_content).item()
    assert abs(loss_correct - loss_correct_base) < 1e-6


def test_dual_backbone_posting_encoder_forward():
    import torch
    import torch.nn as nn
    from scripts.posting_train import PostingEncoder

    # Mock BGE 384-d base encoder
    class MockBGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.name_or_path = "BAAI/bge-small-en-v1.5"
            self.config = type("Config", (), {"hidden_size": 384})()

        def forward(self, input_ids, attention_mask):
            b, s = input_ids.shape
            return type("Out", (), {"last_hidden_state": torch.randn(b, s, 384)})()

    # Mock Nomic 768-d base encoder
    class MockNomic(nn.Module):
        def __init__(self):
            super().__init__()
            self.name_or_path = "nomic-ai/nomic-embed-text-v1.5"
            self.config = type("Config", (), {"hidden_size": 768})()

        def forward(self, input_ids, attention_mask):
            b, s = input_ids.shape
            return type("Out", (), {"last_hidden_state": torch.randn(b, s, 768)})()

    # Test BGE model (384-d, no Matryoshka slicing)
    encoder_bge = PostingEncoder(MockBGE(), matryoshka_dim=384)
    assert encoder_bge.feature_dim == 384
    assert not encoder_bge.is_matryoshka

    ids = torch.randint(0, 100, (2, 8))
    mask = torch.ones((2, 8), dtype=torch.long)
    p_bge, t_bge = encoder_bge(ids, mask)
    assert p_bge.shape == (2, 4)
    assert t_bge.shape == (2, 8, 3)

    # Test Nomic model (768-d sliced down to 384-d with L2 normalization)
    encoder_nom = PostingEncoder(MockNomic(), matryoshka_dim=384)
    assert encoder_nom.feature_dim == 384
    assert encoder_nom.is_matryoshka

    p_nom, t_nom = encoder_nom(ids, mask)
    assert p_nom.shape == (2, 4)
    assert t_nom.shape == (2, 8, 3)


def test_content_loss_evaluation_at_threshold_point_seven():
    from scripts.posting_train import eval_content_loss

    # 10 rows: 8 content pieces (SKILL_DUTY, ROLE_FACTS), 2 boilerplate
    rows = [
        {"text": "Lead distributed systems development with Python and Kafka.", "label": "SKILL_DUTY"},
        {"text": "Deploy services to AWS ECS using Terraform.", "label": "SKILL_DUTY"},
        {"text": "Manage Postgres databases and Redis caching layer.", "label": "SKILL_DUTY"},
        {"text": "Collaborate with cross-functional product design team.", "label": "SKILL_DUTY"},
        {"text": "Bachelor's degree in Computer Science or equivalent experience.", "label": "ROLE_FACTS"},
        {"text": "Minimum 5 years of backend engineering experience required.", "label": "ROLE_FACTS"},
        {"text": "Experience with high-throughput distributed microservices.", "label": "ROLE_FACTS"},
        {"text": "Strong communication and technical leadership skills.", "label": "ROLE_FACTS"},
        {"text": "We are an equal opportunity employer committed to diversity.", "label": "BOILERPLATE"},
        {"text": "Comprehensive health, dental, and 401(k) retirement benefits.", "label": "BOILERPLATE"},
    ]

    # Probs: BP is index 3.
    # Content rows have very low P(BP) (0.01), except row 0 slightly elevated (0.20), none >= 0.70.
    # Boilerplate rows have P(BP) = 0.95.
    probs = [
        [0.70, 0.05, 0.05, 0.20],
        [0.85, 0.05, 0.05, 0.05],
        [0.85, 0.05, 0.05, 0.05],
        [0.85, 0.05, 0.05, 0.05],
        [0.10, 0.80, 0.05, 0.05],
        [0.10, 0.80, 0.05, 0.05],
        [0.10, 0.80, 0.05, 0.05],
        [0.10, 0.80, 0.05, 0.05],
        [0.01, 0.01, 0.03, 0.95],
        [0.01, 0.01, 0.03, 0.95],
    ]

    res = eval_content_loss(rows, probs, threshold=0.70)
    assert res["threshold"] == 0.70
    assert res["wrong_pieces"] == 0
    assert res["content_char_loss"] == 0.0
    assert res["passes_safety"] is True
    assert res["boilerplate_removed"] == 1.0


