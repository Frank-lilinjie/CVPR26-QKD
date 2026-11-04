"""Quantum task identifier and PQK prototype logic in pure MindSpore.

Complex amplitudes are represented by a pair of real-valued tensors. This is
deliberate: it avoids depending on CANN complex-dtype operator coverage while
preserving the statevector, Pauli-Z readout and reduced-density-matrix
mathematics used by the PyTorch reference implementation.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import Normal, Zero
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


class FeaturePreprocessor(nn.Cell):
    """StandardScaler + L2 normalisation + SVD fitted once with NumPy.

    The fitted projection is stored in non-trainable MindSpore parameters so
    that transform and all downstream circuit computation stay on the active
    device. Fitting is a one-off host-side operation and avoids a dependence on
    CANN SVD kernels.
    """

    def __init__(self, input_dim: int = 768, output_dim: int = 12, random_state: int = 1993):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.random_state = int(random_state)
        self._is_fitted = False

        self.scaler_mean = Parameter(
            Tensor(np.zeros(self.input_dim, dtype=np.float32)),
            requires_grad=False,
            name="scaler_mean",
        )
        self.scaler_std = Parameter(
            Tensor(np.ones(self.input_dim, dtype=np.float32)),
            requires_grad=False,
            name="scaler_std",
        )
        self.svd_components = Parameter(
            Tensor(np.zeros((self.output_dim, self.input_dim), dtype=np.float32)),
            requires_grad=False,
            name="svd_components",
        )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def mark_fitted(self) -> None:
        """Mark parameters loaded from a checkpoint as ready for transform."""
        self._is_fitted = True

    def fit(self, features: Tensor) -> Tensor:
        if self._is_fitted:
            return self.transform(features)
        if features.ndim != 2 or int(features.shape[1]) != self.input_dim:
            raise ValueError(
                "Expected features [N, {}], got {}".format(self.input_dim, tuple(features.shape))
            )

        values = features.asnumpy().astype(np.float32, copy=False)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(values)
        mean = np.asarray(scaler.mean_, dtype=np.float32)
        std = np.asarray(scaler.scale_, dtype=np.float32)
        scaled = scaled / (np.linalg.norm(scaled, axis=1, keepdims=True) + 1e-8)

        # Match the reference implementation exactly: TruncatedSVD is fitted
        # without centring after StandardScaler. For the paper setting this is
        # much cheaper than a full SVD of an [8192, 768] feature matrix.
        n_components = min(self.output_dim, self.input_dim)
        svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        svd.fit(scaled)
        fitted_components = np.asarray(svd.components_, dtype=np.float32)
        components = np.zeros((self.output_dim, self.input_dim), dtype=np.float32)
        components[: fitted_components.shape[0]] = fitted_components

        self.scaler_mean.set_data(Tensor(mean))
        self.scaler_std.set_data(Tensor(std))
        self.svd_components.set_data(Tensor(components))
        self._is_fitted = True
        return self.transform(features)

    def transform(self, features: Tensor) -> Tensor:
        if not self._is_fitted:
            raise RuntimeError("FeaturePreprocessor must be fitted before transform")
        scaled = (features - self.scaler_mean) / (self.scaler_std + 1e-8)
        norm = ops.sqrt(ops.sum(scaled * scaled, dim=1, keepdim=True)) + 1e-8
        normalized = scaled / norm
        projected = ops.matmul(normalized, ops.transpose(self.svd_components, (1, 0)))
        projected_norm = ops.sqrt(ops.sum(projected * projected, dim=1, keepdim=True)) + 1e-8
        return projected / projected_norm

    def construct(self, features: Tensor) -> Tensor:
        return self.transform(features)


class QuantumFeatureEncoder(nn.Cell):
    """Pad/truncate SVD features and convert them to data-reuploading angles."""

    def __init__(self, input_dim: int, n_qubits: int, data_reuploading: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_qubits = int(n_qubits)
        self.data_reuploading = int(data_reuploading)
        self.expected_dim = self.n_qubits * self.data_reuploading

    def construct(self, features: Tensor) -> Tensor:
        batch = features.shape[0]
        if self.input_dim < self.expected_dim:
            padding = ops.zeros((batch, self.expected_dim - self.input_dim), features.dtype)
            features = ops.concat((features, padding), axis=1)
        elif self.input_dim > self.expected_dim:
            features = features[:, : self.expected_dim]
        angles = ops.reshape(features, (batch, self.data_reuploading, self.n_qubits))
        angles = ops.transpose(angles, (0, 2, 1))
        return angles * math.pi


class QuantumTaskIdentifier(nn.Cell):
    """Angle-encoded variational circuit with Pauli-Z and PQK readouts."""

    def __init__(
        self,
        feature_dim: int = 12,
        n_qubits: int = 6,
        n_layers: int = 3,
        max_tasks: int = 10,
        data_reuploading: int = 2,
        n_readout_qubits: int = 3,
        svd_random_state: int = 1993,
        backbone_dim: int = 768,
        pqk_temperature: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.max_tasks = int(max_tasks)
        self.data_reuploading = int(data_reuploading)
        self.n_readout_qubits = min(int(n_readout_qubits), self.n_qubits)
        self.pqk_temperature = float(pqk_temperature)
        self.current_task_id = -1

        if self.feature_dim != self.n_qubits * self.data_reuploading:
            logging.warning(
                "feature_dim (%d) != n_qubits (%d) * data_reuploading (%d); "
                "the encoder will pad/truncate automatically",
                self.feature_dim,
                self.n_qubits,
                self.data_reuploading,
            )

        self.preprocessor = FeaturePreprocessor(
            input_dim=int(backbone_dim),
            output_dim=self.feature_dim,
            random_state=int(svd_random_state),
        )
        self.encoder = QuantumFeatureEncoder(
            input_dim=self.feature_dim,
            n_qubits=self.n_qubits,
            data_reuploading=self.data_reuploading,
        )
        self.theta = Parameter(
            Tensor((np.random.default_rng(int(svd_random_state)).normal(0.0, 0.1, size=(self.n_layers * self.data_reuploading, self.n_qubits))).astype(np.float32)),
            name="quantum_theta",
        )
        self.initial_state = Parameter(
            Tensor(np.eye(1 << self.n_qubits, dtype=np.float32)[0]),
            requires_grad=False,
            name="initial_state",
        )
        self.task_head = nn.Dense(
            self.n_readout_qubits,
            self.max_tasks,
            weight_init=Normal(0.02),
            bias_init=Zero(),
        )
        self.task_reduced_density_matrices: Dict[int, Tuple[Tensor, Tensor]] = {}
        self._prototype_sum: Optional[Tuple[Tensor, Tensor]] = None
        self._prototype_count = 0
        self._prototype_task_id = -1
        self._cnot_permutations = [
            Tensor(_cnot_permutation(self.n_qubits, control, control + 1))
            for control in range(self.n_qubits - 1)
        ]

    def trainable_parameters(self) -> List[Parameter]:
        return [self.theta, self.task_head.weight, self.task_head.bias]

    def set_task_id(self, task_id: int) -> None:
        self.current_task_id = int(task_id)

    @staticmethod
    def _apply_ry(real: Tensor, imag: Tensor, angle: Tensor, qubit: int) -> Tuple[Tensor, Tensor]:
        batch = real.shape[0]
        n_qubits = int(round(math.log2(int(real.shape[1]))))
        left = 1 << qubit
        right = 1 << (n_qubits - qubit - 1)
        real = ops.reshape(real, (batch, left, 2, right))
        imag = ops.reshape(imag, (batch, left, 2, right))
        cosine = ops.cos(angle / 2.0)
        sine = ops.sin(angle / 2.0)
        if cosine.ndim == 1:
            cosine = ops.reshape(cosine, (batch, 1, 1))
            sine = ops.reshape(sine, (batch, 1, 1))

        real_zero = cosine * real[:, :, 0, :] - sine * real[:, :, 1, :]
        real_one = sine * real[:, :, 0, :] + cosine * real[:, :, 1, :]
        imag_zero = cosine * imag[:, :, 0, :] - sine * imag[:, :, 1, :]
        imag_one = sine * imag[:, :, 0, :] + cosine * imag[:, :, 1, :]
        real = ops.reshape(ops.stack((real_zero, real_one), axis=2), (batch, -1))
        imag = ops.reshape(ops.stack((imag_zero, imag_one), axis=2), (batch, -1))
        return real, imag

    def run_circuit(self, angles: Tensor) -> Tuple[Tensor, Tensor]:
        batch = angles.shape[0]
        real = ops.tile(ops.expand_dims(self.initial_state, 0), (batch, 1))
        imag = ops.zeros_like(real)
        sublayer = 0
        for _layer in range(self.n_layers):
            for reupload in range(self.data_reuploading):
                for qubit in range(self.n_qubits):
                    effective = angles[:, qubit, reupload] + self.theta[sublayer, qubit]
                    real, imag = self._apply_ry(real, imag, effective, qubit)
                for permutation in self._cnot_permutations:
                    real = ops.gather(real, permutation, axis=1)
                    imag = ops.gather(imag, permutation, axis=1)
                sublayer += 1
        return real, imag

    def measure_pauli_z(self, real: Tensor, imag: Tensor) -> Tensor:
        probabilities = real * real + imag * imag
        batch = probabilities.shape[0]
        values = []
        for qubit in range(self.n_readout_qubits):
            reshaped = ops.reshape(
                probabilities,
                (batch, 1 << qubit, 2, 1 << (self.n_qubits - qubit - 1)),
            )
            value = ops.sum(reshaped[:, :, 0, :] - reshaped[:, :, 1, :], dim=(1, 2))
            values.append(value)
        return ops.stack(values, axis=1)

    def reduced_density_matrix(
        self, real: Tensor, imag: Tensor, target_qubit: int = 0
    ) -> Tuple[Tensor, Tensor]:
        batch = real.shape[0]
        tensor_shape = (batch,) + (2,) * self.n_qubits
        order = [0, target_qubit + 1] + [
            index + 1 for index in range(self.n_qubits) if index != target_qubit
        ]
        a = ops.reshape(
            ops.transpose(ops.reshape(real, tensor_shape), order), (batch, 2, -1)
        )
        b = ops.reshape(
            ops.transpose(ops.reshape(imag, tensor_shape), order), (batch, 2, -1)
        )
        rho_real = ops.matmul(a, ops.transpose(a, (0, 2, 1))) + ops.matmul(
            b, ops.transpose(b, (0, 2, 1))
        )
        rho_imag = ops.matmul(b, ops.transpose(a, (0, 2, 1))) - ops.matmul(
            a, ops.transpose(b, (0, 2, 1))
        )
        return rho_real, rho_imag

    def _quantum_state(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.preprocessor.is_fitted:
            projected = self.preprocessor.fit(features)
        else:
            projected = self.preprocessor.transform(features)
        angles = self.encoder(projected)
        return self.run_circuit(angles)

    def readout_and_logits(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Run the circuit once and return the Pauli-Z readout and task logits."""
        real, imag = self._quantum_state(features)
        readout = self.measure_pauli_z(real, imag)
        return readout, self.task_head(readout)

    def _circuit_outputs(self, features: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        real, imag = self._quantum_state(features)
        readout = self.measure_pauli_z(real, imag)
        rho_real, rho_imag = self.reduced_density_matrix(real, imag, target_qubit=0)
        return readout, real, rho_real, rho_imag

    def pqk_scores(
        self,
        current_real: Tensor,
        current_imag: Tensor,
        task_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if task_ids is None:
            selected_ids = sorted(self.task_reduced_density_matrices)
        else:
            selected_ids = [int(task_id) for task_id in task_ids if int(task_id) in self.task_reduced_density_matrices]
        if not selected_ids:
            return None, None

        prototypes_real = ops.stack(
            [self.task_reduced_density_matrices[task_id][0] for task_id in selected_ids], axis=0
        )
        prototypes_imag = ops.stack(
            [self.task_reduced_density_matrices[task_id][1] for task_id in selected_ids], axis=0
        )
        batch = current_real.shape[0]
        current_real_flat = ops.reshape(current_real, (batch, 4))
        current_imag_flat = ops.reshape(current_imag, (batch, 4))
        prototypes_real_flat = ops.reshape(prototypes_real, (len(selected_ids), 4))
        prototypes_imag_flat = ops.reshape(prototypes_imag, (len(selected_ids), 4))

        # Tr((A + iB)(C + iD)).real = sum(A*C) - sum(B*D).
        scores = ops.matmul(
            current_real_flat, ops.transpose(prototypes_real_flat, (1, 0))
        ) - ops.matmul(
            current_imag_flat, ops.transpose(prototypes_imag_flat, (1, 0))
        )
        weights = ops.stop_gradient(ops.softmax(scores / self.pqk_temperature, axis=-1))
        return scores, weights

    def task_logits(self, features: Tensor) -> Tensor:
        readout, _, _, _ = self._circuit_outputs(features)
        return self.task_head(readout)

    def predict_weights(self, features: Tensor, task_ids: Sequence[int]) -> Tensor:
        """Return normalized task weights restricted to ``task_ids``."""
        selected_ids = [int(task_id) for task_id in task_ids]
        if not selected_ids:
            raise ValueError("predict_weights requires at least one task id")
        _, _, rho_real, rho_imag = self._circuit_outputs(features)
        _, weights = self.pqk_scores(rho_real, rho_imag, selected_ids)
        if weights is not None:
            return weights

        logits = self.task_head(self.measure_pauli_z(*self.run_circuit(self.encoder(self.preprocessor.transform(features)))))
        selected_logits = ops.stack([logits[:, task_id] for task_id in selected_ids], axis=1)
        return ops.softmax(selected_logits, axis=-1)

    def forward_training(self, features: Tensor):
        readout, _, rho_real, rho_imag = self._circuit_outputs(features)
        task_logits = self.task_head(readout)
        _, pqk_weights = self.pqk_scores(rho_real, rho_imag)
        return task_logits, readout, pqk_weights

    def begin_task_prototype(self, task_id: int) -> None:
        self._prototype_task_id = int(task_id)
        self._prototype_sum = None
        self._prototype_count = 0

    def update_task_prototype(self, features: Tensor) -> None:
        if self._prototype_task_id < 0:
            raise RuntimeError("Call begin_task_prototype before update_task_prototype")
        _, _, rho_real, rho_imag = self._circuit_outputs(features)
        batch_sum_real = ops.sum(rho_real, dim=0)
        batch_sum_imag = ops.sum(rho_imag, dim=0)
        if self._prototype_sum is None:
            self._prototype_sum = (batch_sum_real, batch_sum_imag)
        else:
            self._prototype_sum = (
                self._prototype_sum[0] + batch_sum_real,
                self._prototype_sum[1] + batch_sum_imag,
            )
        self._prototype_count += int(features.shape[0])

    def end_task_prototype(self) -> None:
        if self._prototype_sum is None or self._prototype_count <= 0:
            raise RuntimeError("No features were accumulated for the task prototype")
        denominator = float(self._prototype_count)
        self.task_reduced_density_matrices[self._prototype_task_id] = (
            ops.stop_gradient(self._prototype_sum[0] / denominator),
            ops.stop_gradient(self._prototype_sum[1] / denominator),
        )
        logging.info(
            "Saved PQK prototype for task %d from %d samples",
            self._prototype_task_id,
            self._prototype_count,
        )
        self._prototype_sum = None
        self._prototype_count = 0
        self._prototype_task_id = -1

    def save_prototypes_to_file(self, filepath: str | Path) -> None:
        path = Path(filepath)
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")
        payload = {}
        for task_id, (rho_real, rho_imag) in self.task_reduced_density_matrices.items():
            payload["task_{}_real".format(task_id)] = rho_real.asnumpy()
            payload["task_{}_imag".format(task_id)] = rho_imag.asnumpy()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)
        logging.info("Saved %d PQK prototypes to %s", len(self.task_reduced_density_matrices), path)

    def load_prototypes_from_file(self, filepath: str | Path) -> None:
        path = Path(filepath)
        if not path.is_file():
            logging.warning("PQK prototype file does not exist: %s", path)
            return
        with np.load(path, allow_pickle=False) as payload:
            loaded: Dict[int, Tuple[Tensor, Tensor]] = {}
            task_ids = sorted(
                int(key.split("_")[1])
                for key in payload.files
                if key.startswith("task_") and key.endswith("_real")
            )
            for task_id in task_ids:
                real = np.asarray(payload["task_{}_real".format(task_id)], dtype=np.float32)
                imag = np.asarray(payload["task_{}_imag".format(task_id)], dtype=np.float32)
                loaded[task_id] = (Tensor(real), Tensor(imag))
        self.task_reduced_density_matrices = loaded
        logging.info("Loaded %d PQK prototypes from %s", len(loaded), path)

    def construct(self, features: Tensor):
        return self.forward_training(features)


def _cnot_permutation(n_qubits: int, control: int, target: int) -> np.ndarray:
    dim = 1 << n_qubits
    permutation = np.arange(dim, dtype=np.int32)
    control_mask = 1 << (n_qubits - 1 - control)
    target_mask = 1 << (n_qubits - 1 - target)
    active = (permutation & control_mask) != 0
    permutation[active] ^= target_mask
    return permutation
