"""Fast CPU checks for the error-prone implementation details."""
from __future__ import annotations

import unittest
import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

try:
    from . import benchmark, checkpointing, colab_automation, dataset, final_inference, inference, losses, model, train
except ImportError:
    import benchmark
    import checkpointing
    import colab_automation
    import dataset
    import final_inference
    import inference
    import losses
    import model
    import train


class TestLosses(unittest.TestCase):
    def test_focal_gamma_zero_equals_cross_entropy(self):
        torch.manual_seed(0)
        logits = torch.randn(16, 9)
        labels = torch.randint(0, 9, (16,))
        actual = losses.FocalLoss(gamma=0)(logits, labels)
        expected = F.cross_entropy(logits, labels)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)

    def test_label_smoothing_zero_equals_cross_entropy(self):
        torch.manual_seed(1)
        logits = torch.randn(8, 9)
        labels = torch.randint(0, 9, (8,))
        torch.testing.assert_close(
            losses.LabelSmoothingCE(0)(logits, labels),
            F.cross_entropy(logits, labels), atol=1e-7, rtol=1e-6,
        )

    def test_cutmix_lambda_matches_changed_area(self):
        torch.manual_seed(2)
        images = torch.arange(4 * 3 * 16 * 16, dtype=torch.float32).reshape(4, 3, 16, 16)
        labels = torch.arange(4)
        mixed, (_, _, lam) = losses.mix_batch(images, labels, alpha=1.0, mode="cutmix")
        self.assertEqual(mixed.shape, images.shape)
        self.assertGreaterEqual(lam, 0.0)
        self.assertLessEqual(lam, 1.0)

    def test_class_weights_have_mean_one(self):
        weights = losses.class_weights([10, 20, 40])
        torch.testing.assert_close(weights.mean(), torch.tensor(1.0))


class TestInference(unittest.TestCase):
    def test_views_and_aggregation(self):
        x = torch.randn(2, 3, 256, 256)
        self.assertEqual(len(inference.views_multicrop(x, 224)), 5)
        logits = [np.zeros((2, 9)), np.ones((2, 9))]
        probs = inference.aggregate_views(logits, "prob")
        np.testing.assert_allclose(probs.sum(1), 1.0)

    def test_temperature_is_positive_and_preserves_argmax(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(64, 9))
        labels = rng.integers(0, 9, size=64)
        temperature = inference.fit_temperature(logits, labels)
        self.assertGreater(temperature, 0)
        probs = inference.apply_temperature(logits, temperature)
        np.testing.assert_array_equal(probs.argmax(1), logits.argmax(1))

    def test_conv_bn_fusion(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4), nn.ReLU()).eval()
        x = torch.randn(2, 3, 8, 8)
        with torch.inference_mode():
            before = model(x)
            fused = inference.fuse_conv_bn(model)
            after = fused(x)
        torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)


class TestHelpers(unittest.TestCase):
    def test_split_csv_does_not_require_species_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_subset0.csv"
            pd.DataFrame({"Filename": ["a.jpg"], "Label": [0]}).to_csv(path, index=False)
            frame = dataset._read_csv(path)
            self.assertEqual(list(frame.columns), ["Filename", "Label"])

    def test_detect_images_dir_supports_nested_and_flat_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.jpg").touch()
            self.assertEqual(colab_automation._detect_images_dir(root), root)
            (root / "images").mkdir()
            (root / "images" / "nested.jpg").touch()
            self.assertEqual(colab_automation._detect_images_dir(root), root / "images")

    def test_overrides(self):
        parsed = train.parse_overrides(["seed=123", "amp=false", "ema_decay=0.999", "mix=none"])
        self.assertEqual(parsed, {"seed": 123, "amp": False, "ema_decay": 0.999, "mix": None})

    def test_benchmark_contract(self):
        report = benchmark.bench(lambda: 1 + 1, warmup=1, iters=50)
        self.assertEqual(report["n"], 50)
        self.assertLessEqual(report["p50"], report["p95"])
        self.assertLessEqual(report["p95"], report["p99"])

    def test_atomic_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "last.pt"
            mirror = Path(directory) / "drive" / "last.pt"
            checkpointing.atomic_torch_save({"epoch": 3, "tensor": torch.arange(4)}, source)
            digest = checkpointing.atomic_copy(source, mirror)
            self.assertEqual(digest, checkpointing.sha256_file(source))
            restored = checkpointing.torch_load(mirror)
            self.assertEqual(restored["epoch"], 3)
            torch.testing.assert_close(restored["tensor"], torch.arange(4))

    def test_profile_helpers_are_removed_from_old_checkpoints(self):
        network = nn.Sequential(nn.Linear(3, 2))
        polluted = network.state_dict()
        polluted["total_ops"] = torch.tensor([1.0])
        polluted["0.total_params"] = torch.tensor([8.0])
        result = checkpointing.load_model_state_dict(network, polluted)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    def test_gmac_profile_does_not_mutate_training_model(self):
        network = nn.Sequential(nn.Conv2d(3, 4, 1))

        def fake_profile(profiled_model, inputs, verbose):
            profiled_model.register_buffer("total_ops", torch.tensor([123.0]))
            return 2_000_000_000, 0

        with mock.patch.dict(sys.modules, {"thop": SimpleNamespace(profile=fake_profile)}):
            self.assertEqual(model.count_gmacs(network, 8), 2.0)
        self.assertNotIn("total_ops", network.state_dict())

    def test_manifests_keep_test_locked(self):
        with tempfile.TemporaryDirectory() as directory:
            screening = Path(directory) / "screening.json"
            final_training = Path(directory) / "final_training.json"
            final = Path(directory) / "final_test.json"
            colab_automation.create_screening_manifest(screening)
            screen_data = __import__("json").loads(screening.read_text(encoding="utf-8"))
            self.assertFalse(screen_data["test_access"])
            self.assertTrue(all(not item["config"].get("save_test_predictions", False)
                                for item in screen_data["runs"]))
            colab_automation.create_final_training_manifest(
                final_training, train.Config(exp_id="F01"), "Chọn bằng validation"
            )
            train_data = __import__("json").loads(final_training.read_text(encoding="utf-8"))
            self.assertFalse(train_data["test_access"])
            frozen_spec = Path(directory) / "spec.json"
            spec = final_inference.InferenceSpec(name="I01", views=("identity", "hflip"))
            checkpointing.atomic_json_dump({
                "spec": __import__("dataclasses").asdict(spec),
                "spec_sha256": final_inference.spec_hash(spec),
                "selected_from": "validation_only",
                "selection_note": "I01 thắng trên validation",
            }, frozen_spec)
            colab_automation.create_final_test_manifest(final, directory, frozen_spec)
            final_data = __import__("json").loads(final.read_text(encoding="utf-8"))
            self.assertTrue(final_data["test_access"])
            self.assertTrue(final_data["locked"])
            colab_automation.unlock_final_manifest(final, "Chốt hoàn toàn theo validation")
            final_data = __import__("json").loads(final.read_text(encoding="utf-8"))
            self.assertFalse(final_data["locked"])
            self.assertIn("plan_sha256", final_data)


if __name__ == "__main__":
    unittest.main()
