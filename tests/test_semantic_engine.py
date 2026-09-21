"""Unit tests for the Pure Nomic SemanticEngine (nomic-embed-text-v1.5)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from services.semantic.engine import (
    SemanticEngine,
    get_semantic_engine,
    is_nomic_weight_cached,
)
from services.semantic.prompts import (
    DEFAULT_EMBEDDING_DIM,
    MODEL_ID,
    SENIORITY_LABELS,
)


class TestPureNomicSemanticEngine(unittest.TestCase):
    """Test suite verifying multi-task pure Nomic semantic engine functionality."""

    def setUp(self):
        self.engine = SemanticEngine()

    def test_singleton_accessor(self):
        engine1 = get_semantic_engine()
        engine2 = get_semantic_engine()
        self.assertIs(engine1, engine2)

    def test_weight_cache_status(self):
        """Verify that local weight cache check accurately detects verified GGUF weights."""
        is_cached = is_nomic_weight_cached()
        self.assertTrue(is_cached, "nomic-embed-text-v1.5.f16.gguf should be cached in ~/.cache/gpt4all")

    def test_matryoshka_dimension_slicing(self):
        """Verify that embeddings can be sliced to 384 and 256 dimensions natively."""
        mock_vec = np.random.randn(2, 768).astype(np.float32)
        with patch.object(self.engine, "_encode_via_nomic_sdk", return_value=mock_vec[:, :384]):
            vecs = self.engine.encode(["Software Engineer", "Staff Architect"], dim=384)
            self.assertIsNotNone(vecs)
            self.assertEqual(vecs.shape, (2, 384))

            # Check normalization
            norms = np.linalg.norm(vecs, axis=-1)
            np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

    def test_seniority_classification(self):
        """Verify seniority tier prediction logic against anchor embeddings."""
        # Preload fake anchors
        anchor_vecs = np.zeros((6, 384), dtype=np.float32)
        anchor_vecs[0, 0] = 1.0  # intern
        anchor_vecs[1, 3] = 1.0  # newgrad
        anchor_vecs[2, 3] = 1.0  # junior
        anchor_vecs[3, 3] = 1.0  # mid
        anchor_vecs[4, 1] = 1.0  # senior
        anchor_vecs[5, 2] = 1.0  # staff
        self.engine._cached_seniority_anchors = (anchor_vecs, list(SENIORITY_LABELS))

        def fake_encode(texts, **kwargs):
            texts_list = [texts] if isinstance(texts, str) else texts
            result = []
            for t in texts_list:
                vec = np.zeros(384, dtype=np.float32)
                t_lower = t.lower()
                if "intern" in t_lower:
                    vec[0] = 1.0
                elif "senior" in t_lower:
                    vec[1] = 1.0
                elif "staff" in t_lower or "principal" in t_lower:
                    vec[2] = 1.0
                else:
                    vec[3] = 1.0
                result.append(vec)
            arr = np.array(result, dtype=np.float32)
            return arr[0] if isinstance(texts, str) else arr

        with patch.object(self.engine, "encode", side_effect=fake_encode):
            tier, conf = self.engine.classify_seniority("Software Engineer Intern (Summer 2026)")
            self.assertEqual(tier, "intern")
            self.assertGreater(conf, 0.90)

            tier, conf = self.engine.classify_seniority("Senior Backend Systems Engineer")
            self.assertEqual(tier, "senior")
            self.assertGreater(conf, 0.90)

    def test_boilerplate_filtering(self):
        """Verify boilerplate chunks are detected and removed while core duties are kept."""
        bp_vec = np.zeros(384, dtype=np.float32)
        bp_vec[0] = 1.0
        duty_vec = np.zeros(384, dtype=np.float32)
        duty_vec[1] = 1.0
        self.engine._cached_bp_anchors = np.array([bp_vec, duty_vec])

        def fake_encode(texts, **kwargs):
            texts_list = [texts] if isinstance(texts, str) else texts
            result = []
            for t in texts_list:
                vec = np.zeros(384, dtype=np.float32)
                if "equal opportunity" in t.lower() or "401k" in t.lower():
                    vec[0] = 1.0  # matches bp
                else:
                    vec[1] = 1.0  # matches duty
                result.append(vec)
            return np.array(result, dtype=np.float32)

        with patch.object(self.engine, "encode", side_effect=fake_encode):
            chunks = [
                "We are an Equal Opportunity Employer offering comprehensive 401k plans.",
                "Design and implement high-throughput REST APIs using Python and FastAPI.",
                "Maintain PostgreSQL database clusters and Kubernetes deployments.",
            ]
            kept = self.engine.filter_boilerplate(chunks)
            self.assertEqual(len(kept), 2)
            self.assertIn("Design and implement high-throughput REST APIs using Python and FastAPI.", kept)
            self.assertNotIn("We are an Equal Opportunity Employer offering comprehensive 401k plans.", kept)

    def test_dedup_job_embeddings(self):
        """Verify job dict list is embedded into a 2D numpy array."""
        jobs = [
            {"title": "Dev 1", "company": "Co A", "location": "NYC", "description": "Writing code"},
            {"title": "Dev 2", "company": "Co B", "location": "Remote", "description": "Writing tests"},
            {"title": "Dev 3", "company": "Co C", "location": "SF", "description": "Deploying apps"},
        ]
        mock_vecs = np.random.randn(3, 384).astype(np.float32)
        with patch.object(self.engine, "encode", return_value=mock_vecs):
            vecs = self.engine.compute_job_embeddings(jobs, dims=384)
            self.assertIsNotNone(vecs)
            self.assertEqual(vecs.shape, (3, 384))

    def test_resume_fit_scoring(self):
        """Verify asymmetric scoring between resume and job description."""
        r_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        j_vec = np.array([0.8, 0.6, 0.0], dtype=np.float32)

        with patch.object(self.engine, "encode", side_effect=[r_vec, j_vec]):
            score = self.engine.score_resume_fit("Python Kubernetes SWE", "Senior Python Infrastructure Role")
            self.assertAlmostEqual(score, 0.8, places=3)


    def test_dynamic_dimension_swapping(self):
        """Verify dynamic dimension swapping via set_dimension and using_dimension."""
        # Initial dimension should be DEFAULT_EMBEDDING_DIM (384)
        self.assertEqual(self.engine.get_dimension(), 384)

        # Swapping via set_dimension
        self.engine.set_dimension(128)
        self.assertEqual(self.engine.get_dimension(), 128)

        # Context manager swapping
        with self.engine.using_dimension(64):
            self.assertEqual(self.engine.get_dimension(), 64)
            # Anchor slicing works dynamically at 64
            fake_768 = np.ones((6, 768), dtype=np.float32)
            with patch.object(self.engine, "_get_master_seniority_anchors", return_value=(fake_768, list(SENIORITY_LABELS))):
                sliced, labels = self.engine._get_seniority_anchors_for_dim(64)
                self.assertEqual(sliced.shape, (6, 64))
                np.testing.assert_allclose(np.linalg.norm(sliced, axis=-1), np.ones(6), atol=1e-5)

        # Exiting context manager restores previous dimension (128)
        self.assertEqual(self.engine.get_dimension(), 128)

        # Restore to default
        self.engine.set_dimension(384)
        self.assertEqual(self.engine.get_dimension(), 384)

    def test_all_functions_under_swapped_dimensions(self):
        """Verify all engine functions execute cleanly with explicit dim= parameter."""
        test_dims = [768, 512, 384, 256, 128, 64]

        for d in test_dims:
            # 1. Encoding at dimension d
            mock_vec = np.ones((2, d), dtype=np.float32)
            with patch.object(self.engine, "_encode_via_nomic_sdk", return_value=mock_vec):
                vecs = self.engine.encode(["SWE", "Manager"], dim=d)
                self.assertEqual(vecs.shape, (2, d))

            # 2. Seniority classification at dimension d
            with patch.object(self.engine, "encode", return_value=np.ones(d, dtype=np.float32) / np.sqrt(d)):
                fake_anchors = (np.ones((6, d), dtype=np.float32) / np.sqrt(d), list(SENIORITY_LABELS))
                with patch.object(self.engine, "_get_seniority_anchors_for_dim", return_value=fake_anchors):
                    res = self.engine.classify_seniority("Staff Engineer", dim=d)
                    self.assertIsNotNone(res)
                    self.assertIn(res[0], SENIORITY_LABELS)

            # 3. Boilerplate filtering at dimension d
            with patch.object(self.engine, "encode", return_value=np.ones((2, d), dtype=np.float32) / np.sqrt(d)):
                fake_bp = np.ones((2, d), dtype=np.float32) / np.sqrt(d)
                with patch.object(self.engine, "_get_bp_anchors_for_dim", return_value=fake_bp):
                    kept = self.engine.filter_boilerplate(["Chunk 1", "Chunk 2"], dim=d)
                    self.assertIsInstance(kept, list)

            # 4. Job embeddings at dimension d
            jobs = [{"title": "Dev", "company": "Co", "location": "Remote", "description": "Python"}]
            with patch.object(self.engine, "encode", return_value=np.ones((1, d), dtype=np.float32) / np.sqrt(d)):
                job_vecs = self.engine.compute_job_embeddings(jobs, dims=d)
                self.assertEqual(job_vecs.shape, (1, d))

            # 5. Resume fit scoring at dimension d
            v1 = np.ones(d, dtype=np.float32) / np.sqrt(d)
            v2 = np.ones(d, dtype=np.float32) / np.sqrt(d)
            with patch.object(self.engine, "encode", side_effect=[v1, v2]):
                score = self.engine.score_resume_fit("Resume", "Job Description", dim=d)
                self.assertAlmostEqual(score, 1.0, places=3)

    def test_dual_task_weights_registration(self):
        """Verify that converted weights from both datasets are registered and dynamically sliced."""
        tasks = self.engine.get_registered_tasks()
        self.assertIn("seniority", tasks)
        self.assertIn("posting", tasks)

        # Check seniority weights slicing
        sen_data = self.engine.get_task_weights("seniority", dim=384)
        self.assertIsNotNone(sen_data)
        w_sen, b_sen, labels_sen = sen_data
        self.assertEqual(w_sen.shape, (6, 384))
        self.assertEqual(len(b_sen), 6)
        self.assertEqual(labels_sen, ["intern", "newgrad", "junior", "mid", "senior", "staff"])

        # Check posting weights slicing
        post_data = self.engine.get_task_weights("posting", dim=256)
        self.assertIsNotNone(post_data)
        w_post, b_post, labels_post = post_data
        self.assertEqual(w_post.shape, (4, 256))
        self.assertEqual(len(b_post), 4)
        self.assertEqual(labels_post, ["SKILL_DUTY", "BOILERPLATE", "ROLE_FACTS", "COMPANY_CONTEXT"])

    def test_dynamic_task_swapping(self):
        """Verify dynamic weight swapping using using_task context manager."""
        with self.engine.using_task("posting", dim=128):
            self.assertEqual(self.engine.get_dimension(), 128)
            mock_vec = np.ones((1, 128), dtype=np.float32) / np.sqrt(128)
            with patch.object(self.engine, "encode", return_value=mock_vec):
                res = self.engine.predict_task("Build Python services", "posting")
                self.assertIsNotNone(res)
                self.assertIn(res[0], ["SKILL_DUTY", "BOILERPLATE", "ROLE_FACTS", "COMPANY_CONTEXT"])

        # Reverts to default dimension
        self.assertEqual(self.engine.get_dimension(), DEFAULT_EMBEDDING_DIM)

    def test_lru_embedding_cache(self):
        """Verify that identical single-text queries hit the LRU cache without re-encoding."""
        text = "Unique Test Job Title for Cache Check"
        mock_vec = np.ones(384, dtype=np.float32) / np.sqrt(384)

        with patch.object(self.engine, "_encode_via_nomic_sdk", return_value=mock_vec[np.newaxis, :]) as mock_sdk:
            v1 = self.engine.encode(text, dim=384)
            self.assertEqual(mock_sdk.call_count, 1)

            # Second call should hit in-memory cache directly
            v2 = self.engine.encode(text, dim=384)
            self.assertEqual(mock_sdk.call_count, 1)
            np.testing.assert_array_equal(v1, v2)


if __name__ == "__main__":
    unittest.main()


