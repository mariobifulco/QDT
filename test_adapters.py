import importlib.util
import os
import subprocess
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from qdt.adapters import get_adapter
from qdt.adapters.qiskit_run import Adapter as AerAdapter
from qdt.circuit import Circuit
from qdt.config import Config
from test_qdt import TEST_CIRCUIT

ANGLES = np.vstack(
    [np.random.default_rng(5).uniform(-4, 4, (5, 3)), [0, 0, 0], [np.pi, 0, 0]]
)
NOISE = {
    "depolarizing": 0.08,
    "amplitude_damping": 0.12,
    "phase_damping": 0.17,
    "readout_error": 0.04,
}


class AdapterTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("cudaq"), "Install the cudaq extra")
    @patch.dict(os.environ, CUDAQ_TARGET="density-matrix-cpu")
    def test_cudaq_randomized_circuits_match_aer(self):
        rng = np.random.default_rng(125)
        for size in (1, 2, 3, 4):
            gates = [("ry", q, f"x[{q}]") for q in range(size)]
            gates += [("rz", q, f"theta[{q}]") for q in range(size)]
            gates += [("ry", q, f"theta[{size + q}]") for q in range(size)]
            gates += [("cx", q, q + 1) for q in range(size - 1)]
            gates += [
                ("x", 0),
                ("x", 0),
                ("h", 0),
                ("s", 0),
                ("t", 0),
                ("rx", 0, 0.125),
                ("ry", size - 1, "x[0]"),
            ]
            if size > 1:
                gates += [("cz", 0, size - 1), ("swap", 0, size - 1)]
            circuit = Circuit(num_qubits=size, readout_qubit=size - 1, gates=gates)
            rows = rng.uniform(-3, 3, (3, circuit.num_inputs))
            rows[0] = 0
            for rates in ({}, {"readout_error": 0.13}, NOISE):
                with self.subTest(qubits=size, rates=rates):
                    config = Config(backend="cudaq", circuit=circuit, **rates)
                    actual = get_adapter(config).probabilities(rows, 2)
                    expected = AerAdapter(config).probabilities(rows, 2)
                    np.testing.assert_allclose(actual, expected, atol=1e-11)

    @unittest.skipUnless(importlib.util.find_spec("cudaq"), "Install the cudaq extra")
    @patch.dict(os.environ, CUDAQ_TARGET="qpp-cpu")
    def test_cudaq_exact_observations_are_batched(self):
        adapter = get_adapter(Config(circuit=TEST_CIRCUIT, backend="cudaq"))
        with patch.object(
            adapter.runtime, "observe", wraps=adapter.runtime.observe
        ) as observe:
            actual = adapter.probabilities(ANGLES, 3)
            self.assertEqual(observe.call_count, 3)
        np.testing.assert_allclose(
            actual,
            AerAdapter(Config(circuit=TEST_CIRCUIT)).probabilities(ANGLES, 3),
            atol=1e-12,
        )

    @unittest.skipUnless(importlib.util.find_spec("cudaq"), "Install the cudaq extra")
    @patch.dict(os.environ, CUDAQ_TARGET="density-matrix-cpu")
    def test_cudaq_noisy_gate_cancellations_keep_all_channels(self):
        circuit = Circuit(
            num_qubits=1,
            readout_qubit=0,
            gates=(("ry", 0, "x[0]"), ("rz", 0, "theta[0]"), ("x", 0), ("x", 0)),
        )
        config = Config(backend="cudaq", circuit=circuit, depolarizing=0.2)
        actual = get_adapter(config).probabilities([[0, 0]], 64)
        np.testing.assert_allclose(actual, [(1 + 0.8**4) / 2], atol=1e-12)

    @unittest.skipUnless(importlib.util.find_spec("cudaq"), "Install the cudaq extra")
    @patch.dict(os.environ, CUDAQ_TARGET="density-matrix-cpu")
    def test_cudaq_entanglement_and_gate_noise_match_aer(self):
        circuit = Circuit(
            num_qubits=3,
            readout_qubit=1,
            gates=(
                ("ry", 0, "x[0]"),
                ("rx", 1, "x[1]"),
                ("h", 2),
                ("x", 0),
                ("y", 1),
                ("z", 2),
                ("s", 0),
                ("t", 1),
                ("rz", 0, "theta[0]"),
                ("cx", 0, 1),
                ("cz", 1, 2),
                ("ry", 1, "theta[1]"),
                ("swap", 0, 2),
                ("ry", 1, "x[0]"),
            ),
        )
        rows = np.random.default_rng(8).uniform(-3, 3, (5, 4))
        for rates in ({}, NOISE):
            config = Config(backend="cudaq", circuit=circuit, **rates)
            adapter = get_adapter(config)
            kernel = adapter.kernel
            expected = AerAdapter(config).probabilities(rows, 2)
            np.testing.assert_allclose(
                adapter.probabilities(rows, 64), expected, atol=1e-11
            )
            np.testing.assert_allclose(
                adapter.probabilities(rows, 2), expected, atol=1e-11
            )
            self.assertIs(adapter.kernel, kernel)
            sampled = replace(config, shots=8192)
            actual = get_adapter(sampled).probabilities(rows, 64)
            np.testing.assert_allclose(actual, expected, atol=0.03)
            np.testing.assert_array_equal(
                actual, get_adapter(sampled).probabilities(rows, 64)
            )
        circuit = Circuit(
            num_qubits=2,
            readout_qubit=1,
            gates=(
                ("ry", 0, "x[0]"),
                ("ry", 1, "x[1]"),
                ("rz", 0, "theta[0]"),
            ),
        )
        config = Config(backend="cudaq", circuit=circuit, shots=64)
        np.testing.assert_array_equal(
            get_adapter(config).probabilities([[0, np.pi, 0], [np.pi, 0, 0]], 64),
            [0, 1],
        )

    def check_inputs(self, adapter):
        self.assertEqual(adapter.probabilities([], 64).size, 0)
        for angles, batch in (
            ([(0, 0)], 1),
            ([(0, np.nan, 0)], 1),
            (ANGLES, 0),
            (ANGLES, 1.5),
        ):
            with self.assertRaises(ValueError):
                adapter.probabilities(angles, batch)
        self.assertEqual((adapter.calls, adapter.circuits), (0, 0))

    def test_aer_factory_and_optional_imports(self):
        self.check_inputs(get_adapter(Config(circuit=TEST_CIRCUIT)))
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from qdt.adapters import get_adapter; from qdt.config import Config; get_adapter(Config.load('configs/aer_ideale.json')); assert 'cudaq' not in sys.modules; assert 'iqm.qiskit_iqm' not in sys.modules",
            ],
            check=True,
            capture_output=True,
        )

    @unittest.skipUnless(importlib.util.find_spec("cudaq"), "Install the cudaq extra")
    @patch.dict(os.environ, CUDAQ_TARGET="qpp-cpu")
    def test_cudaq_ideal_noise_and_sampling(self):
        ideal = get_adapter(Config(circuit=TEST_CIRCUIT, backend="cudaq"))
        self.check_inputs(ideal)
        np.testing.assert_allclose(
            ideal.probabilities(ANGLES, 64),
            AerAdapter(Config(circuit=TEST_CIRCUIT)).probabilities(ANGLES, 2),
            atol=1e-12,
        )
        with patch.dict(os.environ, CUDAQ_TARGET="density-matrix-cpu"):
            for rates in (
                NOISE,
                {"depolarizing": 1},
                {"amplitude_damping": 1},
                {"phase_damping": 1},
                {"readout_error": 1},
            ):
                config = Config(circuit=TEST_CIRCUIT, backend="cudaq", **rates)
                adapter = get_adapter(config)
                np.testing.assert_allclose(
                    adapter.probabilities(ANGLES, 64),
                    AerAdapter(config).probabilities(ANGLES, 2),
                    atol=1e-11,
                )
            config = Config(circuit=TEST_CIRCUIT, backend="cudaq", shots=8192, **NOISE)
            sampled = get_adapter(config).probabilities(ANGLES, 64)
            np.testing.assert_array_equal(
                sampled, get_adapter(config).probabilities(ANGLES, 2)
            )
            np.testing.assert_allclose(
                sampled,
                AerAdapter(replace(config, shots=0)).probabilities(ANGLES, 2),
                atol=0.03,
            )
        np.testing.assert_allclose(
            ideal.probabilities(ANGLES),
            AerAdapter(Config(circuit=TEST_CIRCUIT)).probabilities(ANGLES),
            atol=1e-12,
        )
        config = Config(
            circuit=TEST_CIRCUIT, backend="cudaq", shots=512, readout_error=1
        )
        np.testing.assert_array_equal(
            get_adapter(config).probabilities(ANGLES[-2:], 64), [0, 1]
        )
        with self.assertRaises(ValueError):
            get_adapter(Config(circuit=TEST_CIRCUIT, backend="cudaq", depolarizing=0.1))


if __name__ == "__main__":
    unittest.main()
