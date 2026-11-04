import argparse
import unittest

from qkd_mindspore.config import apply_cli_overrides, normalize_config


class ConfigTests(unittest.TestCase):
    def test_defaults_match_legacy_adapter_settings(self):
        config = normalize_config({})
        self.assertEqual(config["framework"], "mindspore")
        self.assertEqual(config["ffn_adapter_layernorm_option"], "none")
        self.assertEqual(config["ffn_adapter_scalar"], "0.1")
        self.assertEqual(config["device_target"], "auto")
        self.assertEqual(config["ms_mode"], "pynative")

    def test_legacy_scalar_lists_are_normalised(self):
        config = normalize_config({"device": ["npu:0"], "seed": [1993]})
        self.assertEqual(config["device"], "npu:0")
        self.assertEqual(config["seed"], 1993)

    def test_legacy_backend_selects_ascend(self):
        config = normalize_config({"backend": "npu", "device": ["npu:0"]})
        self.assertEqual(config["device_target"], "ascend")

    def test_cli_overrides(self):
        config = normalize_config(
            {"dataset": "synthetic", "batch_size": 4},
            {"batch_size": 2, "epochs": 3, "n_qubits": 4},
        )
        self.assertEqual(config["batch_size"], 2)
        self.assertEqual(config["epochs"], 3)
        self.assertEqual(config["n_qubits"], 4)

        namespace = argparse.Namespace(
            num_workers=0,
            epochs=None,
            quantum_epochs=2,
            crct_epochs=None,
            max_tasks=1,
            max_samples_per_class=8,
            checkpoint_path=None,
            checkpoint_dir=None,
            data_root=None,
            n_qubits=None,
            svd_dim=None,
            batch_size=None,
            device_target="ascend",
            ms_mode="pynative",
            init_cls=None,
            increment=None,
            dataset=None,
            exp="smoke",
            output_dir=None,
            log_dir=None,
            pretrained=False,
            device="npu:0",
            seed=7,
        )
        apply_cli_overrides(config, namespace)
        self.assertEqual(config["exp"], "smoke")
        self.assertEqual(config["seed"], 7)
        self.assertEqual(config["device"], "npu:0")
        self.assertEqual(config["max_tasks"], 1)
        self.assertFalse(config["pretrained"])

    def test_invalid_values_are_rejected(self):
        for values in (
            {"n_qubits": 0},
            {"svd_dim": 0},
            {"batch_size": 0},
            {"epochs": 0},
            {"img_size": 65},
            {"clip_norm": -1},
            {"framework": "pytorch"},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    normalize_config(values)


if __name__ == "__main__":
    unittest.main()
