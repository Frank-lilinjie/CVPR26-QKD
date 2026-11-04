import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from qkd_mindspore.config import normalize_config
from qkd_mindspore.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DataManager,
    _safe_extract_tar,
    dataset_transform_profile,
    transform_image,
)


class SyntheticDataTests(unittest.TestCase):
    def make_config(self, **overrides):
        values = {
            "dataset": "synthetic",
            "img_size": 32,
            "patch_size": 16,
            "synthetic_classes": 6,
            "synthetic_train_per_class": 3,
            "synthetic_test_per_class": 2,
            "shuffle": False,
            "init_cls": 3,
            "increment": 3,
            "max_tasks": 2,
            "max_samples_per_class": 2,
            "batch_size": 2,
            "num_workers": 0,
        }
        values.update(overrides)
        return normalize_config(values)

    def test_two_task_split_and_limits(self):
        manager = DataManager(self.make_config())
        self.assertEqual(manager.task_sizes, (3, 3))
        self.assertEqual(manager.nb_tasks, 2)
        self.assertEqual(manager.nb_classes, 6)
        self.assertEqual(manager.class_range(0), (0, 3))
        self.assertEqual(manager.class_range(1), (3, 6))

        loader = manager.get_loader(
            [0, 1, 2],
            source="train",
            mode="train",
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )
        indexes, images, labels = next(iter(loader))
        self.assertEqual(images.shape[1:], (3, 32, 32))
        self.assertEqual(images.dtype, "float32")
        self.assertEqual(labels.shape[0], 2)
        self.assertEqual(indexes.shape[0], 2)
        self.assertLessEqual(int(labels.max()), 2)

    def test_max_tasks_limits_classes(self):
        manager = DataManager(self.make_config(max_tasks=1))
        self.assertEqual(manager.task_sizes, (3,))
        self.assertEqual(manager.nb_classes, 3)
        self.assertEqual(manager.class_range(0), (0, 3))

    def test_test_transform_keeps_pixel_range(self):
        image = Image.new("RGB", (256, 256), color=(255, 255, 255))
        array = transform_image(image, train=False, size=32)
        self.assertEqual(array.shape, (3, 32, 32))
        self.assertEqual(array.dtype, np.float32)
        np.testing.assert_allclose(array, 1.0, atol=1e-6)

    def test_imagenet_normalisation_matches_reference_transform(self):
        image = Image.new("RGB", (256, 256), color=(255, 255, 255))
        array = transform_image(
            image,
            train=False,
            size=32,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
        expected = (np.ones(3, dtype=np.float32) - np.asarray(IMAGENET_MEAN)) / np.asarray(
            IMAGENET_STD
        )
        expected = np.broadcast_to(expected[:, None, None], array.shape)
        np.testing.assert_allclose(array, expected, atol=1e-5)

    def test_transform_profiles_match_reference_dataset_classes(self):
        self.assertEqual(dataset_transform_profile("cifar10"), "cifar32")
        self.assertEqual(dataset_transform_profile("cifar100"), "cifar32")
        self.assertEqual(dataset_transform_profile("cifar224"), "vit")
        self.assertEqual(dataset_transform_profile("imagenet1000"), "imagenet1000")
        self.assertEqual(dataset_transform_profile("imagenet100"), "vit")
        self.assertEqual(dataset_transform_profile("cub"), "vit")

    def test_cifar_test_transform_does_not_resize_native_image(self):
        values = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        image = Image.fromarray(values, mode="RGB")
        array = transform_image(
            image,
            train=False,
            size=32,
            profile="cifar32",
        )
        expected = values.astype(np.float32).transpose(2, 0, 1) / 255.0
        np.testing.assert_allclose(array, expected, atol=1e-7)

    def test_cifar_train_transform_keeps_native_output_size(self):
        image = Image.fromarray(
            np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3), mode="RGB"
        )
        array = transform_image(
            image,
            train=True,
            rng=np.random.default_rng(11),
            size=32,
            profile="cifar32",
        )
        self.assertEqual(array.shape, (3, 32, 32))
        self.assertEqual(array.dtype, np.float32)

    def test_unknown_transform_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            transform_image(
                Image.new("RGB", (32, 32)),
                train=False,
                size=32,
                profile="unknown",
            )


class DataSafetyTests(unittest.TestCase):
    def test_tar_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "malicious.tar.gz"
            payload = b"must not escape"
            with tarfile.open(archive, "w:gz") as tar:
                member = tarfile.TarInfo("../evil.txt")
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))

            destination = root / "extract"
            with self.assertRaises(ValueError):
                _safe_extract_tar(archive, destination)
            self.assertFalse((root / "evil.txt").exists())

    def test_tar_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "links.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                member = tarfile.TarInfo("dataset/link")
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                tar.addfile(member)

            with self.assertRaises(ValueError):
                _safe_extract_tar(archive, root / "extract")


if __name__ == "__main__":
    unittest.main()
