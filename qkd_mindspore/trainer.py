"""End-to-end MindSpore trainer for QKD/QuanAdapter on Ascend 910B."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, ops, value_and_grad

from .config import normalize_config
from .data import BatchLoader, DataManager
from .device import describe_device, setup_context, synchronize, tensor_to_numpy
from .model import QuanAdapterNetwork
from .paths import resolve_checkpoint_path


def _parameter_count(parameters: Iterable[ms.Parameter]) -> int:
    return int(sum(int(np.prod(tuple(parameter.shape))) for parameter in parameters))


def _learning_rate(config: dict, epoch: int, epochs: int, base_lr: float) -> float:
    scheduler = config["scheduler"]
    if scheduler == "constant":
        return float(base_lr)
    if scheduler == "steplr":
        milestones = config.get("init_milestones") or [max(1, epochs // 2)]
        decay = float(config.get("init_lr_decay", 0.1))
        return float(base_lr) * (decay ** sum(epoch >= int(item) for item in milestones))
    if epochs <= 1:
        return float(base_lr)
    progress = min(1.0, max(0.0, float(epoch) / float(epochs - 1)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(config["min_lr"] + (base_lr - config["min_lr"]) * cosine)


def _set_optimizer_lr(optimizer: nn.Optimizer, learning_rate: float) -> None:
    value = Tensor(float(learning_rate), ms.float32)
    for attribute in ("learning_rate", "_learning_rate"):
        candidate = getattr(optimizer, attribute, None)
        if candidate is None:
            continue
        if isinstance(candidate, ms.Parameter) or hasattr(candidate, "set_data"):
            try:
                candidate.set_data(value)
                return
            except (TypeError, ValueError, RuntimeError):
                pass
    try:
        optimizer.learning_rate = value
    except (AttributeError, TypeError, ValueError, RuntimeError):
        logging.debug("Optimizer does not expose a mutable learning_rate tensor")


def _clip_gradients(grads, clip_norm: float, parameters: Sequence[ms.Parameter]):
    """Clip a complete gradient tuple, replacing unused slots with zeros."""
    grads = tuple(grads)
    if len(grads) != len(parameters):
        raise ValueError("gradient and parameter counts do not match")
    grads = tuple(
        ops.zeros_like(parameter) if gradient is None else gradient
        for gradient, parameter in zip(grads, parameters)
    )
    if not grads or float(clip_norm) <= 0.0:
        return grads
    squared_norm = None
    for gradient in grads:
        term = ops.sum(gradient * gradient)
        squared_norm = term if squared_norm is None else squared_norm + term
    if squared_norm is None:
        return grads
    total_norm = ops.sqrt(squared_norm)
    scale = ops.minimum(
        Tensor(1.0, ms.float32),
        Tensor(float(clip_norm), ms.float32) / (total_norm + Tensor(1e-12, ms.float32)),
    )
    return tuple(gradient * scale for gradient in grads)


def _scalar_int(value: Tensor) -> int:
    return int(np.asarray(value.asnumpy()).item())


def _scalar_float(value: Tensor) -> float:
    return float(np.asarray(value.asnumpy()).item())


def _make_optimizer(config: dict, parameters: Sequence[ms.Parameter], learning_rate: float):
    if not parameters:
        raise ValueError("Cannot create an optimizer without trainable parameters")
    weight_decay = float(config["weight_decay"])
    name = str(config["optimizer"]).lower()
    if name == "sgd":
        return nn.SGD(
            parameters,
            learning_rate=float(learning_rate),
            momentum=0.9,
            weight_decay=weight_decay,
        )
    if name == "adam":
        return nn.Adam(
            parameters,
            learning_rate=float(learning_rate),
            weight_decay=weight_decay,
        )
    return nn.AdamWeightDecay(
        parameters,
        learning_rate=float(learning_rate),
        weight_decay=weight_decay,
    )


def _to_device_batch(images: np.ndarray, labels: np.ndarray):
    return Tensor(images, ms.float32), Tensor(labels.astype(np.int32, copy=False), ms.int32)


class _Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.data_manager = DataManager(config)
        self.num_tasks = self.data_manager.nb_tasks
        self.task_sizes = tuple(self.data_manager.task_sizes)
        self.network = QuanAdapterNetwork(config, self.task_sizes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.cls_mean: Dict[int, List[np.ndarray]] = {}
        self.cls_cov: Dict[int, List[np.ndarray]] = {}
        self.cls_task: Dict[int, int] = {}
        self._rng = np.random.default_rng(int(config["seed"]) + 991)
        self._task_rows: List[List[float]] = []
        self._metrics_history: List[Dict[str, Any]] = []
        self.output_dir = Path(str(config.get("output_dir") or "results")).expanduser() / str(
            config["exp"]
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load_initial_weights()

    def _load_initial_weights(self) -> None:
        checkpoint = self.config.get("checkpoint_path")
        if checkpoint:
            path = Path(str(checkpoint)).expanduser()
            if path.is_dir():
                path = resolve_checkpoint_path("in21k", self.config)
        else:
            path = resolve_checkpoint_path("in21k", self.config)

        if Path(path).is_file():
            if Path(path).suffix.lower() == ".ckpt":
                self.network.load_mindspore_checkpoint(path)
            else:
                self.network.load_pretrained_npz(path)
            return
        if bool(self.config.get("pretrained")):
            raise FileNotFoundError(
                "pretrained=true but no ViT checkpoint was found. Set "
                "--checkpoint-path/--checkpoint-dir or run synthetic smoke with "
                "pretrained=false. Searched: {}".format(path)
            )
        logging.warning(
            "No ViT checkpoint found at %s; using random initialisation. "
            "This is suitable for a software smoke test, not paper accuracy.",
            path,
        )

    def run(self) -> None:
        logging.info("Starting QKD MindSpore run: %s", self.config["exp"])
        logging.info("Tasks: %s", list(self.data_manager.task_sizes))
        logging.info("Classes: %d", self.data_manager.nb_classes)

        for task_id in range(self.num_tasks):
            start_time = time.time()
            class_start, class_end = self.data_manager.class_range(task_id)
            logging.info(
                "Task %d/%d classes %d-%d", task_id + 1, self.num_tasks, class_start, class_end - 1
            )
            train_loader = self.data_manager.get_loader(
                list(range(class_start, class_end)),
                source="train",
                mode="train",
                batch_size=int(self.config["batch_size"]),
                shuffle=True,
                num_workers=int(self.config["num_workers"]),
            )
            self._train_adapter_stage(task_id, train_loader, class_start)
            self._train_quantum_stage(task_id, train_loader)
            self._save_task_prototype(task_id, train_loader)
            self._compute_class_statistics(task_id, class_start, class_end)
            if task_id > 0 and int(self.config["crct_epochs"]) > 0:
                self._classifier_alignment(task_id)

            metrics = self._evaluate(task_id, class_end)
            self._record_task_row(metrics)
            record = dict(metrics)
            record["task"] = task_id + 1
            self._metrics_history.append(record)
            self._save_artifacts(task_id)
            logging.info(
                "Task %d done in %.1fs | top1=%.2f top5=%.2f task_id_acc=%.2f",
                task_id + 1,
                time.time() - start_time,
                metrics["top1"],
                metrics["top5"],
                metrics.get("task_id_accuracy", 0.0),
            )

        self._log_forgetting()
        self._save_run_summary()

    def _train_adapter_stage(
        self, task_id: int, loader: BatchLoader, class_start: int
    ) -> None:
        self.network.set_train(True)
        adapter_parameters = list(self.network.adapter_parameters(task_id))
        head_parameters = list(self.network.head_parameters())
        adapter_optimizer = _make_optimizer(
            self.config, adapter_parameters, float(self.config["init_lr"])
        )
        head_optimizer = _make_optimizer(
            self.config, head_parameters, float(self.config["init_lr"]) * 0.1
        )
        all_parameters = adapter_parameters + head_parameters
        adapter_count = len(adapter_parameters)
        epochs = int(self.config["epochs"])
        old_tasks = list(range(task_id))
        class_end = class_start + int(self.task_sizes[task_id])

        logging.info(
            "Stage 1 adapter: %d trainable tensors, %d parameters, classes %d-%d",
            len(all_parameters),
            _parameter_count(all_parameters),
            class_start,
            class_end - 1,
        )

        for epoch in range(epochs):
            lr_adapter = _learning_rate(
                self.config, epoch, epochs, float(self.config["init_lr"])
            )
            _set_optimizer_lr(adapter_optimizer, lr_adapter)
            _set_optimizer_lr(head_optimizer, lr_adapter * 0.1)
            loader.epoch = epoch
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for _, images_np, labels_np in loader:
                images, labels = _to_device_batch(images_np, labels_np)
                teacher_logits: Optional[Tensor] = None
                pqk_weights: Optional[Tensor] = None
                if task_id > 0 and bool(self.config["use_distillation"]):
                    teacher_features = self.network.task_feature_stack(
                        images, old_tasks
                    )
                    teacher_logits = ops.stop_gradient(
                        self.network.head_logits(teacher_features)
                    )
                    pqk_weights = ops.stop_gradient(
                        self.network.task_weights(images, old_tasks)
                    )
                    if pqk_weights is None:
                        raise RuntimeError(
                            "PQK prototypes are missing for historical tasks"
                        )

                def forward_fn(images_batch, labels_batch):
                    logits, features = self.network.train_logits(images_batch, task_id)
                    current_logits = logits[:, class_start:class_end]
                    current_targets = labels_batch - class_start
                    loss = self.loss_fn(current_logits, current_targets)
                    kd_loss = Tensor(0.0, ms.float32)
                    if teacher_logits is not None and pqk_weights is not None:
                        temperature = float(self.config["kd_temperature"])
                        teacher_probs = ops.softmax(
                            teacher_logits / temperature, axis=-1
                        )
                        student_logits = self.network.head_logits(features)
                        student_log_probs = ops.log_softmax(
                            student_logits / temperature, axis=-1
                        )
                        # KL(teacher || student) for every historical adapter.
                        kl = ops.sum(
                            teacher_probs
                            * (
                                ops.log(teacher_probs + 1e-8)
                                - ops.expand_dims(student_log_probs, 1)
                            ),
                            dim=-1,
                        )
                        kd_loss = ops.mean(ops.sum(pqk_weights * kl, dim=1)) * (
                            temperature ** 2
                        )
                    return loss + float(self.config["lambda_kd"]) * kd_loss

                grad_fn = value_and_grad(forward_fn, None, weights=all_parameters)
                loss, grads = grad_fn(images, labels)
                grads = _clip_gradients(
                    grads, float(self.config["clip_norm"]), all_parameters
                )
                adapter_optimizer(grads[:adapter_count])
                head_optimizer(grads[adapter_count:])

                training_logits, _ = self.network.train_logits(images, task_id)
                predictions = (
                    ops.argmax(training_logits[:, class_start:class_end], 1)
                    + class_start
                )
                total_correct += _scalar_int(
                    ops.sum(ops.cast(predictions == labels, ms.int32))
                )
                total_samples += int(labels.shape[0])
                total_loss += _scalar_float(loss)

            accuracy = 100.0 * total_correct / max(1, total_samples)
            logging.info(
                "Adapter epoch %d/%d loss=%.4f train_acc=%.2f%% lr=%.5f",
                epoch + 1,
                epochs,
                total_loss / max(1, len(loader)),
                accuracy,
                lr_adapter,
            )

    def _train_quantum_stage(self, task_id: int, loader: BatchLoader) -> None:
        self.network.set_train(False)
        fit_limit = int(self.config.get("quantum_fit_samples", 8192))
        fit_features = self._collect_base_features(loader, limit=fit_limit)
        if fit_features.size == 0:
            raise RuntimeError("No features available to fit the quantum preprocessor")
        self.network.quantum.preprocessor.fit(Tensor(fit_features, ms.float32))
        self.network.quantum.set_task_id(task_id)

        parameters = list(self.network.quantum_parameters())
        optimizer = _make_optimizer(
            self.config, parameters, float(self.config["init_lr"]) * 0.1
        )
        epochs = int(self.config["quantum_epochs"])
        logging.info(
            "Stage 2 quantum: %d parameters, %d fitted samples",
            _parameter_count(parameters),
            fit_features.shape[0],
        )

        for epoch in range(epochs):
            lr = _learning_rate(
                self.config, epoch, epochs, float(self.config["init_lr"]) * 0.1
            )
            _set_optimizer_lr(optimizer, lr)
            loader.epoch = epoch
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for _, images_np, _ in loader:
                images = Tensor(images_np, ms.float32)
                task_targets = Tensor(
                    np.full((images_np.shape[0],), task_id, dtype=np.int32), ms.int32
                )

                def forward_fn(images_batch, task_labels):
                    base = self.network.base_features(images_batch)
                    task_logits, _, _ = self.network.quantum.forward_training(base)
                    return self.loss_fn(task_logits[:, : task_id + 1], task_labels)

                grad_fn = value_and_grad(forward_fn, None, weights=parameters)
                loss, grads = grad_fn(images, task_targets)
                grads = _clip_gradients(
                    grads, float(self.config["clip_norm"]), parameters
                )
                optimizer(grads)

                task_logits, _, _ = self.network.quantum.forward_training(
                    self.network.base_features(images)
                )
                predictions = ops.argmax(task_logits[:, : task_id + 1], 1)
                total_correct += _scalar_int(
                    ops.sum(ops.cast(predictions == task_targets, ms.int32))
                )
                total_samples += int(task_targets.shape[0])
                total_loss += _scalar_float(loss)

            logging.info(
                "Quantum epoch %d/%d loss=%.4f task_acc=%.2f%%",
                epoch + 1,
                epochs,
                total_loss / max(1, len(loader)),
                100.0 * total_correct / max(1, total_samples),
            )

    def _save_task_prototype(self, task_id: int, loader: BatchLoader) -> None:
        self.network.set_train(False)
        self.network.quantum.begin_task_prototype(task_id)
        for _, images_np, _ in loader:
            images = Tensor(images_np, ms.float32)
            self.network.quantum.update_task_prototype(self.network.base_features(images))
        self.network.quantum.end_task_prototype()

    def _collect_base_features(self, loader: BatchLoader, limit: int) -> np.ndarray:
        chunks: List[np.ndarray] = []
        collected = 0
        for _, images_np, _ in loader:
            if collected >= limit:
                break
            images = Tensor(images_np, ms.float32)
            features = tensor_to_numpy(self.network.base_features(images)).astype(
                np.float32, copy=False
            )
            remaining = limit - collected
            chunks.append(features[:remaining])
            collected += min(remaining, features.shape[0])
        if chunks:
            return np.concatenate(chunks, axis=0)
        return np.empty((0, int(self.config["embed_dim"])), dtype=np.float32)

    def _compute_class_statistics(self, task_id: int, start: int, end: int) -> None:
        self.network.set_train(False)
        method = str(self.config["ca_storage_efficient_method"])
        for class_id in range(start, end):
            loader = self.data_manager.get_loader(
                [class_id],
                source="train",
                mode="test",
                batch_size=max(int(self.config["batch_size"]) * 3, 8),
                shuffle=False,
                num_workers=int(self.config["num_workers"]),
            )
            chunks = []
            for _, images_np, _ in loader:
                chunks.append(
                    tensor_to_numpy(
                        self.network.task_features(Tensor(images_np, ms.float32), task_id)
                    )
                )
            features = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
            mean = features.mean(axis=0)
            if method == "covariance":
                dimension = int(features.shape[1])
                if features.shape[0] < 2:
                    covariance = np.eye(dimension, dtype=np.float32) * 1e-3
                else:
                    covariance = np.atleast_2d(np.cov(features, rowvar=False)).astype(
                        np.float32
                    )
                    if covariance.shape != (dimension, dimension):
                        logging.warning(
                            "Unexpected covariance shape for class %d: %s; using diagonal fallback",
                            class_id,
                            covariance.shape,
                        )
                        covariance = np.diag(
                            features.var(axis=0).astype(np.float32) + 1e-3
                        )
                covariance = covariance + np.eye(dimension, dtype=np.float32) * 1e-3
                self.cls_mean[class_id] = [mean.astype(np.float32)]
                self.cls_cov[class_id] = [covariance]
            elif method == "variance":
                variance = features.var(axis=0).astype(np.float32) + 1e-3
                self.cls_mean[class_id] = [mean.astype(np.float32)]
                self.cls_cov[class_id] = [variance]
            else:
                self._store_multi_centroid(class_id, features)
            self.cls_task[class_id] = task_id
        logging.info("Stored feature statistics for classes %d-%d", start, end - 1)

    def _store_multi_centroid(self, class_id: int, features: np.ndarray) -> None:
        n_clusters = max(1, min(int(self.config["n_centroids"]), features.shape[0]))
        try:
            from sklearn.cluster import KMeans

            labels = KMeans(
                n_clusters=n_clusters,
                n_init=10,
                random_state=int(self.config["seed"]),
            ).fit_predict(features)
        except Exception as error:
            logging.warning("KMeans unavailable for class %d (%s); using one centroid", class_id, error)
            labels = np.zeros(features.shape[0], dtype=np.int32)

        means: List[np.ndarray] = []
        variances: List[np.ndarray] = []
        for cluster in sorted(np.unique(labels)):
            cluster_features = features[labels == cluster]
            if cluster_features.shape[0] == 0:
                continue
            means.append(cluster_features.mean(axis=0).astype(np.float32))
            variances.append(cluster_features.var(axis=0).astype(np.float32) + 1e-3)
        if not means:
            means = [features.mean(axis=0).astype(np.float32)]
            variances = [features.var(axis=0).astype(np.float32) + 1e-3]
        self.cls_mean[class_id] = means
        self.cls_cov[class_id] = variances

    def _sample_features(self, class_id: int) -> np.ndarray:
        samples_per_centroid = int(self.config["ca_samples_per_class"])
        chunks: List[np.ndarray] = []
        method = str(self.config["ca_storage_efficient_method"])
        for mean, covariance in zip(self.cls_mean[class_id], self.cls_cov[class_id]):
            if method == "covariance":
                try:
                    sampled = self._rng.multivariate_normal(
                        mean, covariance, size=samples_per_centroid, check_valid="ignore"
                    )
                except Exception as error:
                    logging.warning("Covariance sampling failed for class %d: %s", class_id, error)
                    sampled = mean[None, :] + self._rng.normal(
                        0.0, np.sqrt(covariance.diagonal()), size=(samples_per_centroid, mean.shape[0])
                    )
            else:
                sampled = mean[None, :] + self._rng.normal(
                    0.0,
                    np.sqrt(covariance),
                    size=(samples_per_centroid, mean.shape[0]),
                )
            chunks.append(sampled.astype(np.float32, copy=False))
        return np.concatenate(chunks, axis=0)

    def _classifier_alignment(self, task_id: int) -> None:
        self.network.set_train(False)
        epochs = int(self.config["crct_epochs"])
        parameters = list(self.network.head_parameters())
        optimizer = _make_optimizer(self.config, parameters, float(self.config["ca_lr"]))
        classes = list(range(sum(self.data_manager.task_sizes[: task_id + 1])))
        alignment_batch = max(1, int(self.config["ca_samples_per_class"]))
        logging.info(
            "Classifier alignment: %d classes, %d epochs, %d parameters",
            len(classes),
            epochs,
            _parameter_count(parameters),
        )

        for epoch in range(epochs):
            sampled_features: List[np.ndarray] = []
            sampled_labels: List[np.ndarray] = []
            for class_id in classes:
                sampled = self._sample_features(class_id)
                if sampled.shape[0] == 0:
                    continue
                sampled_features.append(sampled)
                sampled_labels.append(
                    np.full((sampled.shape[0],), class_id, dtype=np.int32)
                )
            if not sampled_features:
                logging.warning(
                    "Classifier alignment epoch %d/%d skipped: no sampled features",
                    epoch + 1,
                    epochs,
                )
                continue

            all_features = np.concatenate(sampled_features, axis=0).astype(
                np.float32, copy=False
            )
            all_labels = np.concatenate(sampled_labels, axis=0)
            lr = _learning_rate(
                self.config, epoch, epochs, float(self.config["ca_lr"])
            )
            _set_optimizer_lr(optimizer, lr)
            order = self._rng.permutation(all_features.shape[0])
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            valid_batches = 0
            for start in range(0, len(order), alignment_batch):
                batch_indices = order[start : start + alignment_batch]
                features = Tensor(all_features[batch_indices], ms.float32)
                labels = Tensor(all_labels[batch_indices], ms.int32)
                if not np.isfinite(all_features[batch_indices]).all():
                    logging.warning(
                        "Classifier alignment epoch %d: skipping a non-finite batch",
                        epoch + 1,
                    )
                    continue

                def forward_fn():
                    logits = self.network.head_logits(features)[:, : len(classes)]
                    return self.loss_fn(logits, labels)

                grad_fn = value_and_grad(forward_fn, None, weights=parameters)
                loss, grads = grad_fn()
                grads = _clip_gradients(
                    grads, float(self.config["clip_norm"]), parameters
                )
                optimizer(grads)
                predictions = ops.argmax(
                    self.network.head_logits(features)[:, : len(classes)], 1
                )
                total_correct += _scalar_int(
                    ops.sum(ops.cast(predictions == labels, ms.int32))
                )
                total_samples += int(labels.shape[0])
                total_loss += _scalar_float(loss)
                valid_batches += 1
            logging.info(
                "Classifier alignment epoch %d/%d loss=%.4f acc=%.2f%% lr=%.5f",
                epoch + 1,
                epochs,
                total_loss / max(1, valid_batches),
                100.0 * total_correct / max(1, total_samples),
                lr,
            )

    def _evaluate(self, task_id: int, total_classes: int) -> Dict[str, Any]:
        self.network.set_train(False)
        loader = self.data_manager.get_loader(
            list(range(total_classes)),
            source="test",
            mode="test",
            batch_size=int(self.config["batch_size"]),
            shuffle=False,
            num_workers=int(self.config["num_workers"]),
        )
        all_predictions: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        all_task_predictions: List[np.ndarray] = []
        seen_tasks = list(range(task_id + 1))
        topk = max(1, min(int(self.config["topk"]), total_classes))

        for _, images_np, labels_np in loader:
            images = Tensor(images_np, ms.float32)
            logits, predicted_tasks, _ = self.network.routed_logits(
                images,
                seen_tasks,
                soft_routing=bool(self.config["use_soft_routing"]),
                temperature=float(self.config["soft_routing_temperature"]),
            )
            logits = logits[:, :total_classes]
            _, indices = ops.topk(logits, k=topk, dim=1, largest=True, sorted=True)
            all_predictions.append(tensor_to_numpy(indices))
            all_targets.append(labels_np.astype(np.int64, copy=False))
            all_task_predictions.append(tensor_to_numpy(predicted_tasks).astype(np.int64, copy=False))

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        predicted_tasks = np.concatenate(all_task_predictions, axis=0)
        true_tasks = self.data_manager.class_to_task[targets]
        accuracy = 100.0 * float(np.mean(predictions[:, 0] == targets))
        top5_accuracy = 100.0 * float(
            np.mean((predictions == targets[:, None]).any(axis=1))
        )
        task_accuracy = 100.0 * float(np.mean(predicted_tasks == true_tasks))

        grouped: Dict[str, float] = {}
        for seen_task in seen_tasks:
            start, end = self.data_manager.class_range(seen_task)
            mask = (targets >= start) & (targets < end)
            if np.any(mask):
                grouped["{}-{}".format(start, end - 1)] = 100.0 * float(
                    np.mean(predictions[mask, 0] == targets[mask])
                )
        logging.info("Evaluation top1=%.2f top5=%.2f task_id=%.2f", accuracy, top5_accuracy, task_accuracy)
        return {
            "top1": accuracy,
            "top5": top5_accuracy,
            "task_id_accuracy": task_accuracy,
            "grouped": grouped,
        }

    def _record_task_row(self, metrics: Dict[str, Any]) -> None:
        row = [float(value) for value in metrics["grouped"].values()]
        self._task_rows.append(row)
        logging.info("Grouped accuracy: %s", metrics["grouped"])

    def _log_forgetting(self) -> None:
        if not self._task_rows or not bool(self.config.get("print_forget")):
            return
        width = max(len(row) for row in self._task_rows)
        matrix = np.full((len(self._task_rows), width), np.nan, dtype=np.float64)
        for index, row in enumerate(self._task_rows):
            matrix[index, : len(row)] = row
        logging.info("Accuracy matrix:\n%s", np.array2string(matrix, precision=2, suppress_small=True))
        if len(matrix) < 2:
            return
        forgetting_values = []
        for task_index in range(len(matrix) - 1):
            best = np.nanmax(matrix[task_index, : task_index + 1])
            current = matrix[-1, task_index]
            if np.isfinite(current):
                forgetting_values.append(best - current)
        if forgetting_values:
            logging.info("Mean forgetting: %.2f", float(np.mean(forgetting_values)))

    def _save_artifacts(self, task_id: int) -> None:
        if not bool(self.config.get("save_checkpoint")):
            return
        checkpoint_path = self.output_dir / "task_{:02d}.ckpt".format(task_id)
        self.network.save_mindspore_checkpoint(checkpoint_path)
        self.network.quantum.save_prototypes_to_file(
            self.output_dir / "task_{:02d}_prototypes.npz".format(task_id)
        )
        payload = {
            "task_id": task_id,
            "config": self.config,
            "class_means": {str(key): [value.tolist() for value in values] for key, values in self.cls_mean.items()},
            "class_covariances": {str(key): [value.tolist() for value in values] for key, values in self.cls_cov.items()},
            "class_tasks": {str(key): value for key, value in self.cls_task.items()},
        }
        with (self.output_dir / "task_{:02d}_stats.json".format(task_id)).open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, indent=2, default=str)
        logging.info("Saved task checkpoint and statistics to %s", self.output_dir)

    def _save_run_summary(self) -> None:
        payload = {
            "framework": "mindspore",
            "dataset": self.config["dataset"],
            "experiment": self.config["exp"],
            "task_sizes": list(self.data_manager.task_sizes),
            "tasks": self._metrics_history,
        }
        path = self.output_dir / "metrics.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        logging.info("Saved run summary to %s", path)


def _configure_logging(config: dict) -> None:
    log_dir = Path(str(config.get("log_dir") or "logs")).expanduser() / str(config["exp"])
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = log_dir / "{}_{}_{}.log".format(
        config["dataset"], config["seed"], config["framework"]
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        handlers=[logging.FileHandler(filename), logging.StreamHandler()],
        force=True,
    )


def train(raw_config: dict) -> None:
    config = normalize_config(raw_config)
    config.setdefault("framework", "mindspore")
    _configure_logging(config)
    logging.info("Effective configuration: %s", json.dumps(config, indent=2, default=str))
    info = setup_context(config)
    logging.info(describe_device(info))
    trainer = _Trainer(config)
    trainer.run()
    synchronize()
