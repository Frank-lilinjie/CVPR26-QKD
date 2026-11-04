import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from qkd_mindspore.config import normalize_config

try:
    import mindspore as ms
    from mindspore import Tensor

    from qkd_mindspore.model import QuanAdapterNetwork
    from qkd_mindspore.quantum import FeaturePreprocessor

    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False


def tiny_config():
    return normalize_config(
        {
            "dataset": "synthetic",
            "img_size": 32,
            "patch_size": 16,
            "embed_dim": 32,
            "depth": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "ffn_num": 4,
            "init_cls": 2,
            "increment": 2,
            "max_tasks": 2,
            "synthetic_classes": 4,
            "n_qubits": 2,
            "svd_dim": 4,
            "quantum_layers": 1,
            "batch_size": 2,
            "shuffle": False,
            "drop": 0.0,
            "attn_drop": 0.0,
            "drop_path": 0.0,
            "device_target": "cpu",
            "device": "cpu",
            "save_checkpoint": False,
        }
    )


@unittest.skipUnless(MINDSPORE_AVAILABLE, "MindSpore is not installed")
class FeaturePreprocessorTests(unittest.TestCase):
    def test_fit_matches_reference_standard_scaler_and_svd(self):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(11)
        values = rng.normal(size=(16, 8)).astype(np.float32)
        values[:, 0] = 3.0
        features = Tensor(values)
        preprocessor = FeaturePreprocessor(input_dim=8, output_dim=4, random_state=1993)

        scaler = StandardScaler()
        expected_scaled = scaler.fit_transform(values)
        expected_scaled /= np.linalg.norm(expected_scaled, axis=1, keepdims=True) + 1e-8
        expected = TruncatedSVD(n_components=4, random_state=1993).fit_transform(expected_scaled)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True) + 1e-8

        projected = preprocessor.fit(features).asnumpy()
        np.testing.assert_allclose(preprocessor.scaler_mean.asnumpy(), scaler.mean_, atol=1e-6)
        np.testing.assert_allclose(preprocessor.scaler_std.asnumpy(), scaler.scale_, atol=1e-6)
        np.testing.assert_allclose(projected, expected, atol=1e-5)

    def test_fit_uses_truncated_svd_and_normalises_rows(self):
        rng = np.random.default_rng(7)
        features = Tensor(rng.normal(size=(12, 32)).astype(np.float32))
        preprocessor = FeaturePreprocessor(input_dim=32, output_dim=4, random_state=1993)

        with mock.patch("numpy.linalg.svd", side_effect=AssertionError("full SVD used")):
            projected = preprocessor.fit(features).asnumpy()

        self.assertTrue(preprocessor.is_fitted)
        self.assertEqual(projected.shape, (12, 4))
        norms = np.linalg.norm(projected, axis=1)
        np.testing.assert_allclose(norms, np.ones(12), atol=1e-5)
        self.assertTrue(np.isfinite(projected).all())

    def test_small_batch_is_zero_padded_when_rank_is_insufficient(self):
        features = Tensor(np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, 3.0]], dtype=np.float32))
        preprocessor = FeaturePreprocessor(input_dim=4, output_dim=3, random_state=1993)
        projected = preprocessor.fit(features).asnumpy()
        self.assertEqual(projected.shape, (2, 3))
        self.assertTrue(np.isfinite(projected).all())
        np.testing.assert_allclose(np.linalg.norm(projected, axis=1), 1.0, atol=1e-5)


@unittest.skipUnless(MINDSPORE_AVAILABLE, "MindSpore is not installed")
class CheckpointRoundTripTests(unittest.TestCase):
    def test_network_and_pqk_prototype_round_trip(self):
        config = tiny_config()
        rng = np.random.default_rng(19)
        features = Tensor(rng.normal(size=(8, config["embed_dim"])).astype(np.float32))
        network = QuanAdapterNetwork(config, (2, 2))
        expected_projection = network.quantum.preprocessor.fit(features).asnumpy()
        network.quantum.begin_task_prototype(0)
        network.quantum.update_task_prototype(features)
        network.quantum.end_task_prototype()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "task_00.ckpt"
            prototypes = root / "task_00_prototypes.npz"
            network.save_mindspore_checkpoint(checkpoint)
            network.quantum.save_prototypes_to_file(prototypes)

            restored = QuanAdapterNetwork(config, (2, 2))
            restored.load_mindspore_checkpoint(checkpoint)
            restored.quantum.load_prototypes_from_file(prototypes)

            self.assertTrue(restored.quantum.preprocessor.is_fitted)
            np.testing.assert_allclose(
                restored.quantum.preprocessor.transform(features).asnumpy(),
                expected_projection,
                atol=1e-5,
            )
            self.assertIn(0, restored.quantum.task_reduced_density_matrices)

            images = Tensor(rng.normal(size=(2, 3, 32, 32)).astype(np.float32))
            logits, predicted_tasks, weights = restored.routed_logits(
                images,
                [0],
                soft_routing=False,
            )
            self.assertEqual(tuple(logits.shape), (2, 4))
            self.assertEqual(tuple(predicted_tasks.shape), (2,))
            self.assertEqual(tuple(weights.shape), (2, 1))
            self.assertTrue(np.isfinite(logits.asnumpy()).all())
            np.testing.assert_allclose(weights.asnumpy(), 1.0, atol=1e-6)

    def test_timm_style_npz_loading_splits_qkv_and_strips_prefixes(self):
        config = tiny_config()
        source = QuanAdapterNetwork(config, (2, 2))
        backbone = source.backbone
        arrays = {
            "module.vit.patch_embed.proj.weight": backbone.patch_embed.proj.weight.asnumpy(),
            "module.vit.patch_embed.proj.bias": backbone.patch_embed.proj.bias.asnumpy(),
            "module.vit.cls_token": backbone.cls_token.asnumpy(),
            "module.vit.pos_embed": backbone.pos_embed.asnumpy(),
            "module.vit.norm.weight": backbone.norm.gamma.asnumpy(),
            "module.vit.norm.bias": backbone.norm.beta.asnumpy(),
        }
        for index, block in enumerate(backbone.blocks):
            prefix = "module.vit.blocks.{}.".format(index)
            arrays[prefix + "attn.qkv.weight"] = np.concatenate(
                [
                    block.attn.q_proj.weight.asnumpy(),
                    block.attn.k_proj.weight.asnumpy(),
                    block.attn.v_proj.weight.asnumpy(),
                ],
                axis=0,
            )
            arrays[prefix + "attn.qkv.bias"] = np.concatenate(
                [
                    block.attn.q_proj.bias.asnumpy(),
                    block.attn.k_proj.bias.asnumpy(),
                    block.attn.v_proj.bias.asnumpy(),
                ],
                axis=0,
            )
            arrays.update(
                {
                    prefix + "attn.proj.weight": block.attn.proj.weight.asnumpy(),
                    prefix + "attn.proj.bias": block.attn.proj.bias.asnumpy(),
                    prefix + "norm1.weight": block.norm1.gamma.asnumpy(),
                    prefix + "norm1.bias": block.norm1.beta.asnumpy(),
                    prefix + "norm2.weight": block.norm2.gamma.asnumpy(),
                    prefix + "norm2.bias": block.norm2.beta.asnumpy(),
                    prefix + "mlp.fc1.weight": block.mlp.fc1.weight.asnumpy(),
                    prefix + "mlp.fc1.bias": block.mlp.fc1.bias.asnumpy(),
                    prefix + "mlp.fc2.weight": block.mlp.fc2.weight.asnumpy(),
                    prefix + "mlp.fc2.bias": block.mlp.fc2.bias.asnumpy(),
                }
            )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vit.npz"
            np.savez(path, **arrays)
            restored = QuanAdapterNetwork(config, (2, 2))
            loaded = restored.load_pretrained_npz(path, strict=True)

        expected_loaded = 6 + len(backbone.blocks) * (6 + 10)
        self.assertEqual(loaded, expected_loaded)
        np.testing.assert_allclose(
            restored.backbone.blocks[0].attn.q_proj.weight.asnumpy(),
            backbone.blocks[0].attn.q_proj.weight.asnumpy(),
        )
        np.testing.assert_allclose(
            restored.backbone.blocks[0].attn.v_proj.bias.asnumpy(),
            backbone.blocks[0].attn.v_proj.bias.asnumpy(),
        )


if __name__ == "__main__":
    unittest.main()
