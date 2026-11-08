"""Pure-NumPy reference implementation of the QKD quantum circuit.

The MindSpore implementation stores real and imaginary amplitudes separately.
These helpers provide a complex-valued reference for the circuit mathematics,
independent of MindSpore and the device runtime.
"""

from __future__ import annotations

import math

import numpy as np


def cnot_permutation(n_qubits: int, control: int, target: int) -> np.ndarray:
    """Permutation mapping input basis indices to CNOT output indices."""
    dim = 1 << n_qubits
    permutation = np.arange(dim, dtype=np.int64)
    control_mask = 1 << (n_qubits - 1 - control)
    target_mask = 1 << (n_qubits - 1 - target)
    active = (permutation & control_mask) != 0
    permutation[active] ^= target_mask
    return permutation


def apply_ry(state: np.ndarray, angle, qubit: int) -> np.ndarray:
    """Apply an RY rotation to a batch of complex state vectors."""
    n_qubits = int(round(math.log2(state.shape[-1])))
    batch = state.shape[0]
    left = 1 << qubit
    right = 1 << (n_qubits - qubit - 1)
    reshaped = np.asarray(state).reshape(batch, left, 2, right)
    cosine = np.cos(np.asarray(angle) / 2.0)
    sine = np.sin(np.asarray(angle) / 2.0)
    if np.ndim(cosine) == 1:
        cosine = cosine.reshape(batch, 1, 1)
        sine = sine.reshape(batch, 1, 1)
    new_zero = cosine * reshaped[:, :, 0, :] - sine * reshaped[:, :, 1, :]
    new_one = sine * reshaped[:, :, 0, :] + cosine * reshaped[:, :, 1, :]
    return np.stack((new_zero, new_one), axis=2).reshape(batch, -1)


def run_reference_circuit_literal(
    angles: np.ndarray,
    theta: np.ndarray,
    n_qubits: int,
    n_layers: int,
    data_reuploading: int,
) -> np.ndarray:
    """Complex reference that applies data and trainable RY gates separately."""
    angles = np.asarray(angles, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    batch = angles.shape[0]
    state = np.zeros((batch, 1 << n_qubits), dtype=np.complex128)
    state[:, 0] = 1.0
    sublayer = 0
    for _layer in range(n_layers):
        for reupload in range(data_reuploading):
            for qubit in range(n_qubits):
                state = apply_ry(state, angles[:, qubit, reupload], qubit)
                state = apply_ry(state, np.full(batch, theta[sublayer, qubit]), qubit)
            for control in range(n_qubits - 1):
                permutation = cnot_permutation(n_qubits, control, control + 1)
                state = state[:, permutation]
            sublayer += 1
    return state


def run_reference_circuit(
    angles: np.ndarray,
    theta: np.ndarray,
    n_qubits: int,
    n_layers: int,
    data_reuploading: int,
) -> np.ndarray:
    """Complex reference using the equivalent combined RY angle."""
    angles = np.asarray(angles, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    batch = angles.shape[0]
    state = np.zeros((batch, 1 << n_qubits), dtype=np.complex128)
    state[:, 0] = 1.0
    sublayer = 0
    for _layer in range(n_layers):
        for reupload in range(data_reuploading):
            for qubit in range(n_qubits):
                state = apply_ry(state, angles[:, qubit, reupload] + theta[sublayer, qubit], qubit)
            for control in range(n_qubits - 1):
                state = state[:, cnot_permutation(n_qubits, control, control + 1)]
            sublayer += 1
    return state


def reduced_density_matrix(state: np.ndarray, target_qubit: int = 0) -> np.ndarray:
    """Return the one-qubit reduced density matrix for each state vector."""
    state = np.asarray(state)
    n_qubits = int(round(math.log2(state.shape[-1])))
    tensor = state.reshape((state.shape[0],) + (2,) * n_qubits)
    order = [0, target_qubit + 1] + [
        index + 1 for index in range(n_qubits) if index != target_qubit
    ]
    matrix = np.transpose(tensor, order).reshape(state.shape[0], 2, -1)
    return matrix @ matrix.conj().transpose(0, 2, 1)


def pauli_z_expectations(state: np.ndarray, n_qubits_to_measure: int) -> np.ndarray:
    state = np.asarray(state)
    n_qubits = int(round(math.log2(state.shape[-1])))
    probabilities = np.abs(state) ** 2
    values = []
    for qubit in range(n_qubits_to_measure):
        reshaped = probabilities.reshape(
            state.shape[0], 1 << qubit, 2, 1 << (n_qubits - qubit - 1)
        )
        values.append((reshaped[:, :, 0, :] - reshaped[:, :, 1, :]).sum(axis=(1, 2)))
    return np.stack(values, axis=1)


def pqk_trace(current_rho: np.ndarray, task_rho: np.ndarray) -> np.ndarray:
    product = current_rho @ task_rho[None, :, :]
    return np.trace(product, axis1=1, axis2=2).real


def real_imag_reduced_density(
    real: np.ndarray,
    imag: np.ndarray,
    target_qubit: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy mirror of the real/imaginary MindSpore trace-out operation."""
    n_qubits = int(round(math.log2(real.shape[-1])))
    tensor_order = [0, target_qubit + 1] + [
        index + 1 for index in range(n_qubits) if index != target_qubit
    ]
    a = np.transpose(
        real.reshape((real.shape[0],) + (2,) * n_qubits), tensor_order
    ).reshape(real.shape[0], 2, -1)
    b = np.transpose(
        imag.reshape((imag.shape[0],) + (2,) * n_qubits), tensor_order
    ).reshape(imag.shape[0], 2, -1)
    rho_real = a @ np.transpose(a, (0, 2, 1)) + b @ np.transpose(b, (0, 2, 1))
    rho_imag = b @ np.transpose(a, (0, 2, 1)) - a @ np.transpose(b, (0, 2, 1))
    return rho_real, rho_imag


def real_imag_pqk_trace(
    current_real: np.ndarray,
    current_imag: np.ndarray,
    task_real: np.ndarray,
    task_imag: np.ndarray,
) -> np.ndarray:
    """Real part of ``Trace(rho * sigma)`` for Hermitian 2x2 matrices.

    If ``rho = A + iB`` and ``sigma = C + iD``, the real part is
    ``Trace(AC - BD)``. This is the quantity used by the PQK readout.
    """
    current_real = np.asarray(current_real).reshape(-1, 4)
    current_imag = np.asarray(current_imag).reshape(-1, 4)
    return current_real @ np.asarray(task_real).reshape(4) - current_imag @ np.asarray(
        task_imag
    ).reshape(4)
