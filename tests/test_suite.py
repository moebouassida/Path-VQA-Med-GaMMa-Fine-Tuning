"""
Unit tests — run in CI without GPU or HuggingFace access.
    pytest tests/ -v
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Metrics ───────────────────────────────────────────────────────────────────
class TestMetrics:
    def test_exact_match_identical(self):
        from src.metrics import exact_match

        assert exact_match("yes", "yes") == 1.0

    def test_exact_match_case_insensitive(self):
        from src.metrics import exact_match

        assert exact_match("Yes", "yes") == 1.0
        assert exact_match("NO", "no") == 1.0

    def test_exact_match_different(self):
        from src.metrics import exact_match

        assert exact_match("yes", "no") == 0.0

    def test_exact_match_normalized(self):
        from src.metrics import exact_match

        assert exact_match("  yes. ", "yes") == 1.0

    def test_is_yes_no_detection(self):
        from src.metrics import is_yes_no

        assert is_yes_no("yes") is True
        assert is_yes_no("Yes") is True
        assert is_yes_no("no") is True
        assert is_yes_no("gastrointestinal") is False
        assert is_yes_no("tumor core present") is False

    def test_bleu_identical(self):
        from src.metrics import bleu_score

        score = bleu_score("the tumor is malignant", "the tumor is malignant")
        assert score > 0.9

    def test_bleu_partial(self):
        from src.metrics import bleu_score

        score = bleu_score("the tumor", "the tumor is malignant")
        assert 0.0 < score < 1.0

    def test_bleu_empty(self):
        from src.metrics import bleu_score

        assert bleu_score("", "something") == 0.0

    def test_vqa_score_yes_no(self):
        from src.metrics import vqa_score

        result = vqa_score("yes", "yes")
        assert result["exact_match"] == 1.0
        assert result["is_yes_no"] is True

    def test_vqa_score_open_ended(self):
        from src.metrics import vqa_score

        result = vqa_score("gastrointestinal tissue", "gastrointestinal")
        assert result["is_yes_no"] is False
        assert result["bleu"] > 0.0

    def test_aggregate_scores(self):
        from src.metrics import vqa_score, aggregate_scores

        results = [
            vqa_score("yes", "yes"),
            vqa_score("no", "no"),
            vqa_score("gastrointestinal tissue", "gastrointestinal"),
        ]
        agg = aggregate_scores(results)
        assert "yes_no_accuracy" in agg
        assert "open_ended_bleu" in agg
        assert "overall_exact_match" in agg
        assert agg["yes_no_count"] == 2
        assert agg["open_ended_count"] == 1

    def test_quality_gates_pass(self):
        from src.metrics import check_quality_gates

        scores = {"yes_no_accuracy": 0.70, "open_ended_bleu": 0.30}
        cfg = {"gate_exact_match": 0.55, "gate_bleu": 0.20}
        passed, failures = check_quality_gates(scores, cfg)
        assert passed
        assert len(failures) == 0

    def test_quality_gates_fail(self):
        from src.metrics import check_quality_gates

        scores = {"yes_no_accuracy": 0.30, "open_ended_bleu": 0.05}
        cfg = {"gate_exact_match": 0.55, "gate_bleu": 0.20}
        passed, failures = check_quality_gates(scores, cfg)
        assert not passed
        assert len(failures) == 2


# ── Data Processing ───────────────────────────────────────────────────────────
class TestDataProcessing:
    def test_convert_to_conversation_structure(self):
        from src.data_processing import convert_to_conversation
        from PIL import Image

        sample = {
            "image": Image.new("RGB", (64, 64)),
            "question": "Is this malignant?",
            "answer": "yes",
            "enhanced_answer": "yes. Explanation: The tissue shows malignant features.",
        }

        conv = convert_to_conversation(sample, use_enhanced=True)

        assert "messages" in conv
        assert len(conv["messages"]) == 2
        assert conv["messages"][0]["role"] == "user"
        assert conv["messages"][1]["role"] == "assistant"

    def test_convert_uses_enhanced_answer(self):
        from src.data_processing import convert_to_conversation
        from PIL import Image

        sample = {
            "image": Image.new("RGB", (64, 64)),
            "question": "What is present?",
            "answer": "colon",
            "enhanced_answer": "colon. Explanation: Colonic crypts are present.",
        }

        conv = convert_to_conversation(sample, use_enhanced=True)
        answer_text = conv["messages"][1]["content"][0]["text"]
        assert "Explanation" in answer_text

    def test_convert_fallback_on_empty_enhanced(self):
        from src.data_processing import convert_to_conversation
        from PIL import Image

        sample = {
            "image": Image.new("RGB", (64, 64)),
            "question": "What is present?",
            "answer": "colon",
            "enhanced_answer": "",
        }

        conv = convert_to_conversation(sample, use_enhanced=True)
        answer_text = conv["messages"][1]["content"][0]["text"]
        assert answer_text == "colon"

    def test_conversation_has_image(self):
        from src.data_processing import convert_to_conversation
        from PIL import Image

        sample = {
            "image": Image.new("RGB", (64, 64)),
            "question": "Is edema present?",
            "answer": "no",
            "enhanced_answer": "no. Explanation: No edema observed.",
        }

        conv = convert_to_conversation(sample)
        types = [c["type"] for c in conv["messages"][0]["content"]]
        assert "image" in types
        assert "text" in types


# ── Config ────────────────────────────────────────────────────────────────────
class TestConfig:
    @pytest.fixture(autouse=True)
    def load_cfg(self):
        import yaml

        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "config.yaml"
        )
        if os.path.exists(cfg_path):
            self.cfg = yaml.safe_load(open(cfg_path))
        else:
            self.cfg = None

    def test_config_exists(self):
        assert self.cfg is not None, "config/config.yaml not found"

    def test_required_keys(self):
        required = [
            "pretrained_model",
            "lora_r",
            "lora_alpha",
            "learning_rate",
            "num_train_epochs",
            "output_dir",
        ]
        for key in required:
            assert key in self.cfg, f"Missing key: {key}"

    def test_lora_r_positive(self):
        assert self.cfg["lora_r"] > 0

    def test_learning_rate_sensible(self):
        assert 1e-6 < float(self.cfg["learning_rate"]) < 1.0

    def test_quality_gates_present(self):
        assert "gate_exact_match" in self.cfg
        assert "gate_bleu" in self.cfg

    def test_dataset_name_correct(self):
        assert self.cfg.get("dataset_name") == "moebouassida/path-vqa-enhanced"
