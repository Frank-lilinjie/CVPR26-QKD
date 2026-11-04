"""Dataset and incremental-task management without torchvision.

Images are decoded with Pillow and yielded as NCHW ``float32`` NumPy arrays.
The training loop converts batches to MindSpore tensors.  A small threaded
prefetcher is provided because the original image pipeline used PyTorch
workers; it avoids introducing a second, incompatible dataset abstraction.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import shutil
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .paths import resolve_dataset_dir


IMAGE_SIZE = 224
CIFAR_IMAGE_SIZE = 32
CIFAR_BRIGHTNESS = 63.0 / 255.0
SUPPORTED_TRANSFORM_PROFILES = {"cifar32", "imagenet1000", "vit"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Match the reference transforms exactly. Most QuanAdapter datasets use
# ``ToTensor`` without mean/std normalisation; ImageNet-1K is the exception.
DATASET_NORMALIZATION = {
    "cifar10": (CIFAR10_MEAN, CIFAR10_STD),
    "cifar100": (CIFAR100_MEAN, CIFAR100_STD),
    "imagenet1000": (IMAGENET_MEAN, IMAGENET_STD),
}


def dataset_transform_profile(dataset_name: str) -> str:
    """Return the augmentation profile used by the reference dataset class."""
    name = str(dataset_name).strip().lower()
    if name in {"cifar10", "cifar100"}:
        return "cifar32"
    if name == "imagenet1000":
        return "imagenet1000"
    return "vit"


def _to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def _load_image(value: Any, use_path: bool) -> Image.Image:
    if use_path:
        with open(str(value), "rb") as handle:
            return _to_rgb(Image.open(handle))
    return _to_rgb(Image.fromarray(np.asarray(value, dtype=np.uint8)))


def _random_resized_crop(
    image: Image.Image,
    size: int,
    rng: np.random.Generator,
    scale: tuple[float, float] = (0.05, 1.0),
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
) -> Image.Image:
    width, height = image.size
    area = width * height
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * float(rng.uniform(scale[0], scale[1]))
        aspect_ratio = math.exp(float(rng.uniform(log_ratio[0], log_ratio[1])))
        crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            left = int(rng.integers(0, width - crop_w + 1))
            top = int(rng.integers(0, height - crop_h + 1))
            return image.crop((left, top, left + crop_w, top + crop_h)).resize(
                (size, size), Image.BILINEAR
            )

    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize(
        (size, size), Image.BILINEAR
    )


def _random_crop_with_padding(
    image: Image.Image,
    size: int,
    rng: np.random.Generator,
    padding: int = 4,
) -> Image.Image:
    """Equivalent to torchvision RandomCrop(size, padding=4, fill=0)."""
    padded = ImageOps.expand(image, border=int(padding), fill=0)
    width, height = padded.size
    if width < size or height < size:
        raise ValueError(
            "Padded image {}x{} is smaller than crop size {}".format(width, height, size)
        )
    left = int(rng.integers(0, width - size + 1))
    top = int(rng.integers(0, height - size + 1))
    return padded.crop((left, top, left + size, top + size))


def _random_horizontal_flip(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    if float(rng.random()) < 0.5:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def _random_brightness(
    image: Image.Image,
    rng: np.random.Generator,
    brightness: float = CIFAR_BRIGHTNESS,
) -> Image.Image:
    """Apply torchvision ColorJitter(brightness=brightness) semantics."""
    delta = float(brightness)
    factor = float(rng.uniform(max(0.0, 1.0 - delta), 1.0 + delta))
    return ImageEnhance.Brightness(image).enhance(factor)


def _test_resize_crop(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    width, height = image.size
    resize_side = int(round((256.0 / 224.0) * size))
    if width <= height:
        new_w, new_h = resize_side, int(round(height * resize_side / width))
    else:
        new_w, new_h = int(round(width * resize_side / height)), resize_side
    image = image.resize((new_w, new_h), Image.BICUBIC)
    left = int(round((new_w - size) / 2.0))
    top = int(round((new_h - size) / 2.0))
    return image.crop((left, top, left + size, top + size))


def transform_image(
    image: Image.Image,
    *,
    train: bool,
    rng: np.random.Generator | None = None,
    size: int = IMAGE_SIZE,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    profile: str = "vit",
) -> np.ndarray:
    """Reproduce the dataset-specific transforms from ``utils/data.py``."""
    profile = str(profile).strip().lower()
    if profile not in SUPPORTED_TRANSFORM_PROFILES:
        raise ValueError(
            "Unsupported transform profile {!r}; expected one of {}".format(
                profile, sorted(SUPPORTED_TRANSFORM_PROFILES)
            )
        )

    if train:
        if rng is None:
            raise ValueError("A NumPy Generator is required for training transforms")
        if profile == "cifar32":
            image = _random_crop_with_padding(image, size=size, rng=rng, padding=4)
            image = _random_horizontal_flip(image, rng)
            image = _random_brightness(image, rng)
        else:
            scale = (0.08, 1.0) if profile == "imagenet1000" else (0.05, 1.0)
            image = _random_resized_crop(image, size=size, rng=rng, scale=scale)
            image = _random_horizontal_flip(image, rng)
            if profile == "imagenet1000":
                image = _random_brightness(image, rng)
    else:
        # The reference CIFAR transforms evaluate on the original 32x32 image.
        if profile != "cifar32":
            image = _test_resize_crop(image, size=size)

    array = (np.asarray(image, dtype=np.float32) / 255.0).transpose(2, 0, 1)
    if mean is not None or std is not None:
        if mean is None or std is None:
            raise ValueError("mean and std must be provided together")
        mean_array = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        std_array = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        if np.any(std_array <= 0):
            raise ValueError("Normalisation standard deviations must be positive")
        array = (array - mean_array) / std_array
    return np.ascontiguousarray(array)


def _dataset_normalization(
    dataset_name: str, config: dict[str, Any]
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """Resolve optional per-channel normalisation for the selected dataset."""
    requested = config.get("normalize", "auto")
    if requested is False or str(requested).strip().lower() in {"false", "none", "off"}:
        return None

    mean = config.get("image_mean")
    std = config.get("image_std")
    if mean is not None or std is not None:
        if mean is None or std is None:
            raise ValueError("image_mean and image_std must be configured together")
        mean_values = tuple(float(value) for value in mean)
        std_values = tuple(float(value) for value in std)
        if len(mean_values) != 3 or len(std_values) != 3:
            raise ValueError("image_mean and image_std must each contain three values")
        return mean_values, std_values

    return DATASET_NORMALIZATION.get(dataset_name.lower())


class IncrementalDataset:
    """A lightweight view over a CIFAR array or an ImageFolder path list."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        *,
        use_path: bool,
        train: bool,
        seed: int,
        image_size: int = IMAGE_SIZE,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        transform_profile: str = "vit",
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.use_path = bool(use_path)
        self.train = bool(train)
        self.seed = int(seed)
        self.image_size = int(image_size)
        self.mean = mean
        self.std = std
        self.transform_profile = str(transform_profile)
        self.epoch = 0

    def __len__(self) -> int:
        return int(self.indices.size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, position: int):
        data_index = int(self.indices[position])
        item_rng = np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + position * 97 + int(data_index)
        )
        image = _load_image(self.images[data_index], self.use_path)
        tensor = transform_image(
            image,
            train=self.train,
            rng=item_rng,
            size=self.image_size,
            mean=self.mean,
            std=self.std,
            profile=self.transform_profile,
        )
        return data_index, tensor, int(self.labels[data_index])


class BatchLoader:
    """Small deterministic batch loader with optional threaded prefetch."""

    def __init__(
        self,
        dataset: IncrementalDataset,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        num_workers: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_workers = max(0, int(num_workers))
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def __len__(self) -> int:
        total = len(self.dataset)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size

    def _batch(self, positions: Sequence[int]):
        items = [self.dataset[int(position)] for position in positions]
        indexes = np.asarray([item[0] for item in items], dtype=np.int64)
        images = np.stack([item[1] for item in items], axis=0).astype(np.float32, copy=False)
        labels = np.asarray([item[2] for item in items], dtype=np.int64)
        return indexes, images, labels

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(self.seed + self.epoch * 7_919)
        order = rng.permutation(len(self.dataset)) if self.shuffle else np.arange(len(self.dataset))
        batches = [
            order[start : start + self.batch_size]
            for start in range(0, len(order), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()

        self.dataset.set_epoch(self.epoch)
        if self.num_workers == 0:
            for positions in batches:
                yield self._batch(positions)
            return

        with ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix="qkd-data") as pool:
            for batch in pool.map(self._batch, batches):
                yield batch


def _download_file(url: str, destination: Path) -> None:
    """Download to a sibling temporary file, then atomically publish it."""
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    try:
        logging.info("Downloading %s", url)
        with urllib.request.urlopen(url, timeout=120) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded archive is empty: {url}")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _safe_tar_members(tar: tarfile.TarFile, root: Path) -> list[tarfile.TarInfo]:
    """Validate a tar before extraction and reject links or path escapes."""
    root = root.resolve()
    members: list[tarfile.TarInfo] = []
    for member in tar.getmembers():
        if not member.name:
            raise ValueError("Tar member has an empty path")
        member_path = Path(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe tar member path: {member.name!r}")
        if member.issym() or member.islnk():
            raise ValueError(f"Tar links are not allowed: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"Unsupported tar member type: {member.name!r}")

        target = (root / member_path).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Tar member escapes extraction root: {member.name!r}")
        members.append(member)
    return members


def _safe_extract_tar(archive: Path, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=root, members=_safe_tar_members(tar, root))


def _download_and_extract(url: str, root: Path, expected_dir: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / Path(url).name
    if not archive.exists():
        _download_file(url, archive)
    _safe_extract_tar(archive, root)
    if not (root / expected_dir).exists():
        raise RuntimeError(f"Could not find extracted dataset directory {root / expected_dir}")


def _load_cifar(root: Path, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if kind == "cifar100":
        directory = root / "cifar-100-python"
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        if not directory.exists():
            _download_and_extract(url, root, directory.name)
        with (directory / "train").open("rb") as handle:
            train = pickle.load(handle, encoding="bytes")
        with (directory / "test").open("rb") as handle:
            test = pickle.load(handle, encoding="bytes")
        train_x = np.asarray(train[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        test_x = np.asarray(test[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        train_y = np.asarray(train[b"fine_labels"], dtype=np.int64)
        test_y = np.asarray(test[b"fine_labels"], dtype=np.int64)
        return train_x, train_y, test_x, test_y

    if kind == "cifar10":
        directory = root / "cifar-10-batches-py"
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        if not directory.exists():
            _download_and_extract(url, root, directory.name)
        train_images, train_labels = [], []
        for index in range(1, 6):
            with (directory / f"data_batch_{index}").open("rb") as handle:
                batch = pickle.load(handle, encoding="bytes")
            train_images.append(np.asarray(batch[b"data"], dtype=np.uint8))
            train_labels.append(np.asarray(batch[b"labels"], dtype=np.int64))
        with (directory / "test_batch").open("rb") as handle:
            test = pickle.load(handle, encoding="bytes")
        train_x = np.concatenate(train_images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        test_x = np.asarray(test[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        train_y = np.concatenate(train_labels)
        test_y = np.asarray(test[b"labels"], dtype=np.int64)
        return train_x, train_y, test_x, test_y

    raise ValueError(f"Unknown CIFAR variant: {kind}")


def _collect_image_folder(split_dir: Path, class_to_index: dict[str, int]) -> tuple[list[str], list[int]]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"ImageFolder split does not exist: {split_dir}")
    paths: list[str] = []
    labels: list[int] = []
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        label = class_to_index[class_dir.name]
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                paths.append(str(path))
                labels.append(label)
    return paths, labels


def _load_image_folder(
    root: Path,
    train_name: str = "train",
    test_name: str = "test",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dir = root / train_name
    test_dir = root / test_name
    class_names = sorted(
        {
            path.name
            for parent in (train_dir, test_dir)
            if parent.is_dir()
            for path in parent.iterdir()
            if path.is_dir()
        }
    )
    if not class_names:
        raise RuntimeError(f"No class directories found under {root}")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    train_paths, train_labels = _collect_image_folder(train_dir, class_to_index)
    test_paths, test_labels = _collect_image_folder(test_dir, class_to_index)
    return (
        np.asarray(train_paths, dtype=object),
        np.asarray(train_labels, dtype=np.int64),
        np.asarray(test_paths, dtype=object),
        np.asarray(test_labels, dtype=np.int64),
    )


def _generate_synthetic(
    n_classes: int,
    train_per_class: int,
    test_per_class: int,
    seed: int,
    image_size: int = IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    palette = rng.uniform(0.15, 0.85, size=(n_classes, 3, 1, 1)).astype(np.float32)
    train_x: list[np.ndarray] = []
    train_y: list[int] = []
    test_x: list[np.ndarray] = []
    test_y: list[int] = []
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    stripe = (np.sin((xx + yy) / 19.0) * 0.08).astype(np.float32)[None, :, :]
    for class_id in range(n_classes):
        pattern = palette[class_id] + stripe
        for label, count, target_x, target_y in (
            (class_id, train_per_class, train_x, train_y),
            (class_id, test_per_class, test_x, test_y),
        ):
            for _ in range(count):
                noise = rng.normal(0.0, 0.035, size=(3, image_size, image_size)).astype(np.float32)
                image = np.clip(pattern + noise, 0.0, 1.0)
                target_x.append((image.transpose(1, 2, 0) * 255.0).round().astype(np.uint8))
                target_y.append(label)
    return (
        np.stack(train_x),
        np.asarray(train_y, dtype=np.int64),
        np.stack(test_x),
        np.asarray(test_y, dtype=np.int64),
    )


def _load_dataset(
    name: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    name = name.lower()
    if name == "synthetic":
        n_classes = int(config.get("synthetic_classes", 6))
        train_per_class = int(config.get("synthetic_train_per_class", 8))
        test_per_class = int(config.get("synthetic_test_per_class", 4))
        arrays = _generate_synthetic(
            n_classes,
            train_per_class,
            test_per_class,
            int(config.get("seed", 1993)) + 17,
            image_size=int(config.get("img_size", IMAGE_SIZE)),
        )
        return (*arrays, False)

    root = resolve_dataset_dir(name, config)
    if name in {"cifar10", "cifar100", "cifar224"}:
        arrays = _load_cifar(root, "cifar10" if name == "cifar10" else "cifar100")
        return (*arrays, False)

    train_name, test_name = "train", "test"
    if name == "imagenet1000":
        test_name = "val"
    elif name == "imagenet100":
        test_name = "eval"
    supported = {
        "imagenet100",
        "imagenet1000",
        "imagenetr",
        "imagenet-r",
        "imageneta",
        "imagenet-a",
        "cub",
        "objectnet",
        "omnibenchmark",
        "vtab",
    }
    if name not in supported:
        raise NotImplementedError(f"Unsupported dataset for the MindSpore path: {name}")
    arrays = _load_image_folder(root, train_name=train_name, test_name=test_name)
    return (*arrays, True)


def _limit_per_class(
    images: np.ndarray,
    labels: np.ndarray,
    limit: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[np.ndarray] = []
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) > limit:
            indices = np.sort(rng.choice(indices, size=limit, replace=False))
        selected.append(indices)
    final = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    return images[final], labels[final]


class DataManager:
    """Task split and class-order logic matching ``utils/data_manager.py``."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.dataset_name = str(config["dataset"])
        self.transform_profile = dataset_transform_profile(self.dataset_name)
        self.seed = int(config["seed"])
        if self.transform_profile == "cifar32" and int(config.get("img_size", IMAGE_SIZE)) != CIFAR_IMAGE_SIZE:
            raise ValueError(
                "cifar10/cifar100 use native 32x32 transforms; set img_size=32, "
                "or use dataset=cifar224 for the resized 224x224 variant"
            )
        normalization = _dataset_normalization(self.dataset_name, config)
        self.image_mean = normalization[0] if normalization else None
        self.image_std = normalization[1] if normalization else None
        (
            self.train_images,
            raw_train_labels,
            self.test_images,
            raw_test_labels,
            self.use_path,
        ) = _load_dataset(self.dataset_name, config)
        if len(self.train_images) == 0 or len(self.test_images) == 0:
            raise RuntimeError("Dataset split is empty")

        n_available = int(len(np.unique(raw_train_labels)))
        if isinstance(raw_train_labels, np.ndarray):
            n_available = max(n_available, int(raw_train_labels.max()) + 1)

        base_order = list(range(n_available))
        if bool(config.get("shuffle", True)):
            rng = np.random.RandomState(self.seed)
            class_order = rng.permutation(n_available).tolist()
        else:
            class_order = base_order

        inverse = np.empty(n_available, dtype=np.int64)
        for new_index, old_index in enumerate(class_order):
            inverse[int(old_index)] = int(new_index)
        self.train_labels = inverse[raw_train_labels]
        self.test_labels = inverse[raw_test_labels]
        self._class_order = class_order

        init_cls = int(config["init_cls"])
        increment = int(config["increment"])
        if init_cls > n_available:
            raise ValueError(f"init_cls={init_cls} exceeds available classes={n_available}")
        increments = [init_cls]
        while sum(increments) + increment < n_available:
            increments.append(increment)
        offset = n_available - sum(increments)
        if offset > 0:
            increments.append(offset)

        max_tasks = int(config.get("max_tasks") or 0)
        if max_tasks > 0:
            increments = increments[:max_tasks]
            keep_classes = sum(increments)
            self._class_order = self._class_order[:keep_classes]
            train_mask = self.train_labels < keep_classes
            test_mask = self.test_labels < keep_classes
            self.train_images = self.train_images[train_mask]
            self.train_labels = self.train_labels[train_mask]
            self.test_images = self.test_images[test_mask]
            self.test_labels = self.test_labels[test_mask]
            logging.info("Limited run to %d task(s), %d classes", len(increments), keep_classes)

        limit = int(config.get("max_samples_per_class") or 0)
        if limit > 0:
            rng = np.random.default_rng(self.seed)
            self.train_images, self.train_labels = _limit_per_class(
                self.train_images, self.train_labels, limit, rng
            )
            self.test_images, self.test_labels = _limit_per_class(
                self.test_images, self.test_labels, limit, rng
            )
            logging.info("Limited dataset to at most %d sample(s) per class", limit)

        self.task_sizes = tuple(int(size) for size in increments)
        # Kept as a private alias for code that used the pre-release field.
        self._increments = self.task_sizes
        self.class_to_task = np.zeros(len(self._class_order), dtype=np.int64)
        start = 0
        for task_id, size in enumerate(self.task_sizes):
            self.class_to_task[start : start + size] = task_id
            start += size

        logging.info("Class order: %s", self._class_order)
        logging.info("Task sizes: %s", list(self.task_sizes))

    @property
    def nb_tasks(self) -> int:
        return len(self.task_sizes)

    @property
    def nb_classes(self) -> int:
        return len(self._class_order)

    def get_task_size(self, task_id: int) -> int:
        return int(self.task_sizes[task_id])

    def class_range(self, task_id: int) -> tuple[int, int]:
        start = 0 if task_id == 0 else int(sum(self.task_sizes[:task_id]))
        return start, start + int(self.task_sizes[task_id])

    def get_dataset(self, indices: Sequence[int], source: str, mode: str) -> IncrementalDataset:
        if source == "train":
            images, labels = self.train_images, self.train_labels
        elif source == "test":
            images, labels = self.test_images, self.test_labels
        else:
            raise ValueError(f"Unknown data source: {source}")
        if mode not in {"train", "test"}:
            raise ValueError(f"Unknown data mode: {mode}")
        selected = np.flatnonzero(np.isin(labels, np.asarray(list(indices), dtype=np.int64)))
        return IncrementalDataset(
            images,
            labels,
            selected,
            use_path=self.use_path,
            train=(mode == "train"),
            seed=self.seed,
            image_size=int(self.config.get("img_size", IMAGE_SIZE)),
            mean=self.image_mean,
            std=self.image_std,
            transform_profile=self.transform_profile,
        )

    def get_loader(
        self,
        indices: Sequence[int],
        source: str,
        mode: str,
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
    ) -> BatchLoader:
        dataset = self.get_dataset(indices, source, mode)
        return BatchLoader(
            dataset,
            batch_size,
            shuffle=shuffle,
            seed=self.seed,
            num_workers=num_workers,
        )
