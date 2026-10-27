import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from qdt.adapters import get_adapter
from qdt.adapters.iqm_run import Adapter as IQMAdapter
from qdt.adapters.qiskit_run import Adapter as AerAdapter
from qdt.circuit import Circuit
from qdt.config import Config

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
    @unittest.skipUnless(importlib.util.find_spec("iqm"), "Install the iqm extra")
    @patch.dict(os.environ, IQM_BACKEND="fake")
    def test_iqm_entangled_samples_compile_once_per_batch_shape(self):
        from iqm.qiskit_iqm import transpile_to_IQM

        circuit = Circuit(
            num_qubits=2,
            readout_qubit=1,
            gates=(
                ("ry", 0, "x[0]"),
                ("ry", 1, "x[1]"),
                ("cx", 0, 1),
                ("ry", 1, "theta[0]"),
                ("rz", 0, "theta[1]"),
            ),
        )
        config = Config(backend="iqm", circuit=circuit, shots=256)
        rows = np.tile(
            [[0, np.pi, 0, 0], [np.pi, 0, 0, 0], [np.pi, np.pi, 0, 0], [0, 0, 0, 0]],
            (3, 1),
        )
        with patch(
            "iqm.qiskit_iqm.transpile_to_IQM", wraps=transpile_to_IQM
        ) as compile_circuit:
            adapter = get_adapter(config)
            with patch.object(
                adapter.simulator, "run", wraps=adapter.simulator.run
            ) as run:
                actual = adapter.probabilities(rows, 64)
                self.assertEqual(
                    [call.args[0].num_clbits for call in run.call_args_list], [10, 2]
                )
            self.assertEqual(compile_circuit.call_count, 2)
            templates = dict(adapter.templates)
            adapter.probabilities(rows, 64)
            self.assertEqual(compile_circuit.call_count, 2)
            for size in templates:
                self.assertIs(templates[size][0], adapter.templates[size][0])
        np.testing.assert_allclose(actual, [0, 0, 1, 1] * 3, atol=0.2)
        np.testing.assert_array_equal(
            actual, get_adapter(config).probabilities(rows, 64)
        )

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
        # Explicit asymmetric basis states detect reversed bitstring order.
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
        self.check_inputs(get_adapter(Config()))
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from qdt.adapters import get_adapter; from qdt.config import Config; get_adapter(Config()); assert 'cudaq' not in sys.modules; assert 'iqm.qiskit_iqm' not in sys.modules",
            ],
            check=True,
            capture_output=True,
        )

    @patch.dict(os.environ, IQM_BACKEND="sim")
    def test_iqm_sim_matches_aer_and_cli(self):
        for config in (
            Config(backend="iqm"),
            Config(backend="iqm", **NOISE),
            Config(backend="iqm", shots=512, **NOISE),
        ):
            adapter = get_adapter(config)
            self.check_inputs(adapter)
            np.testing.assert_array_equal(
                adapter.probabilities(ANGLES, 64),
                AerAdapter(config).probabilities(ANGLES, 64),
            )
            self.assertEqual((adapter.calls, adapter.circuits), (1, len(ANGLES)))
        # Exercise backend selection through the complete ANOVA/COBYLA CLI.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    asdict(
                        Config(
                            backend="iqm",
                            max_train_samples=30,
                            top_k=1,
                            max_depth=1,
                            max_evals=4,
                            restarts=1,
                        )
                    )
                )
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qdt",
                    "--config",
                    str(config),
                    "--output",
                    str(root / "result.json"),
                ],
                check=True,
                capture_output=True,
            )
            result = json.loads((root / "result.json").read_text())
            self.assertEqual(result["qdt"]["device"], "iqm:sim:statevector")
            self.assertEqual(result["qdt"]["optimizer"], "cobyla")
            self.assertGreater(result["train_probability_calls"], 0)

    def test_iqm_rejects_ambiguous_noise_and_exact_hardware(self):
        for mode in ("fake", "qpu"):
            with patch.dict(os.environ, IQM_BACKEND=mode):
                for config in (Config(), Config(shots=128, **NOISE)):
                    with self.assertRaises(ValueError):
                        IQMAdapter(config)
        with patch.dict(os.environ, IQM_BACKEND="invalid"):
            with self.assertRaises(ValueError):
                IQMAdapter(Config())

    @unittest.skipUnless(importlib.util.find_spec("iqm"), "Install the iqm extra")
    @patch.dict(os.environ, IQM_BACKEND="fake")
    def test_iqm_fake_calibration_batch_cap_and_seed(self):
        config = Config(backend="iqm", shots=128)
        adapter = get_adapter(config)
        self.check_inputs(adapter)
        angles = np.array([[0, 0, 0], [np.pi, 0, 0]] * 11 + [[0, 0, 0]])
        with patch.object(adapter.simulator, "run", wraps=adapter.simulator.run) as run:
            actual = adapter.probabilities(angles, 64)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(
                [call.args[0].num_clbits for call in run.call_args_list], [20, 3]
            )
        np.testing.assert_array_equal(
            actual, get_adapter(config).probabilities(angles, 64)
        )
        np.testing.assert_allclose(actual, [1, 0] * 11 + [1], atol=0.15)
        self.assertEqual((adapter.calls, adapter.circuits), (1, len(angles)))

    @unittest.skipUnless(importlib.util.find_spec("iqm"), "Install the iqm extra")
    @patch.dict(os.environ, IQM_BACKEND="qpu", IQM_SERVER_URL="https://example.invalid")
    def test_iqm_qpu_submission_contract_without_network(self):
        from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

        backend = IQMFakeGarnet()
        # Counts use little-endian classical bits: first sample always zero.
        result = SimpleNamespace(success=True, get_counts=lambda: {"10": 10})
        with (
            patch("iqm.qiskit_iqm.IQMProvider") as provider,
            patch.object(backend, "run") as run,
        ):
            provider.return_value.get_backend.return_value = backend
            run.return_value.result.return_value = result
            actual = get_adapter(Config(backend="iqm", shots=10)).probabilities(
                ANGLES[:2], 64
            )
            np.testing.assert_array_equal(actual, [1, 0])
            self.assertEqual(run.call_args.kwargs, {"shots": 10})

    @unittest.skipUnless(importlib.util.find_spec("cudaq"), "Install the cudaq extra")
    @patch.dict(os.environ, CUDAQ_TARGET="qpp-cpu")
    def test_cudaq_ideal_noise_and_sampling(self):
        ideal = get_adapter(Config(backend="cudaq"))
        self.check_inputs(ideal)
        np.testing.assert_allclose(
            ideal.probabilities(ANGLES, 64),
            AerAdapter(Config()).probabilities(ANGLES, 2),
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
                config = Config(backend="cudaq", **rates)
                adapter = get_adapter(config)
                np.testing.assert_allclose(
                    adapter.probabilities(ANGLES, 64),
                    AerAdapter(config).probabilities(ANGLES, 2),
                    atol=1e-11,
                )
            config = Config(backend="cudaq", shots=8192, **NOISE)
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
            AerAdapter(Config()).probabilities(ANGLES),
            atol=1e-12,
        )
        config = Config(backend="cudaq", shots=512, readout_error=1)
        np.testing.assert_array_equal(
            get_adapter(config).probabilities(ANGLES[-2:], 64), [0, 1]
        )
        with self.assertRaises(ValueError):
            get_adapter(Config(backend="cudaq", depolarizing=0.1))


if __name__ == "__main__":
    unittest.main()
