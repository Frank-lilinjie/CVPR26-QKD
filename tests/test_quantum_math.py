import unittest

import numpy as np

from qkd_mindspore.quantum_math import (
    pqk_trace,
    real_imag_pqk_trace,
    real_imag_reduced_density,
    reduced_density_matrix,
    run_reference_circuit,
    run_reference_circuit_literal,
)


class QuantumMathTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.n_qubits = 3
        self.n_layers = 2
        self.reupload = 2
        self.angles = rng.normal(0.0, 0.3, size=(4, self.n_qubits, self.reupload))
        self.theta = rng.normal(0.0, 0.2, size=(self.n_layers * self.reupload, self.n_qubits))

    def test_reference_circuit_is_normalised(self):
        state = run_reference_circuit(
            self.angles,
            self.theta,
            self.n_qubits,
            self.n_layers,
            self.reupload,
        )
        np.testing.assert_allclose(np.sum(np.abs(state) ** 2, axis=1), 1.0, atol=1e-10)

    def test_combined_and_literal_circuit_agree(self):
        combined = run_reference_circuit(
            self.angles,
            self.theta,
            self.n_qubits,
            self.n_layers,
            self.reupload,
        )
        literal = run_reference_circuit_literal(
            self.angles,
            self.theta,
            self.n_qubits,
            self.n_layers,
            self.reupload,
        )
        np.testing.assert_allclose(combined, literal, atol=1e-10)

    def test_reduced_density_matrix_has_unit_trace(self):
        state = run_reference_circuit(
            self.angles,
            self.theta,
            self.n_qubits,
            self.n_layers,
            self.reupload,
        )
        rho = reduced_density_matrix(state, target_qubit=0)
        np.testing.assert_allclose(
            np.trace(rho, axis1=1, axis2=2), np.ones(state.shape[0]), atol=1e-10
        )

    def test_real_imag_pqk_trace_matches_complex_trace(self):
        first = run_reference_circuit(
            self.angles,
            self.theta,
            self.n_qubits,
            self.n_layers,
            self.reupload,
        )
        second = run_reference_circuit(
            self.angles[::-1],
            self.theta * 0.5,
            self.n_qubits,
            self.n_layers,
            self.reupload,
        )
        rho = reduced_density_matrix(first)
        sigma = reduced_density_matrix(second)
        expected = pqk_trace(rho, sigma[1])

        real, imag = real_imag_reduced_density(first.real, first.imag)
        got = real_imag_pqk_trace(
            real,
            imag,
            sigma.real[1],
            sigma.imag[1],
        )
        np.testing.assert_allclose(got, expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
