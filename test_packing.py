import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from qiskit_aer import AerSimulator

from qdt.adapters.qiskit_run import Adapter
from qdt.circuit import Circuit
from qdt.config import Config
from qdt.packing import compose_copies, marginal_zeros

CIRCUIT = Circuit(
    num_qubits=2,
    readout_qubit=1,
    gates=(
        ("ry", 0, "x[0]"),
        ("rx", 1, "x[1]"),
        ("cx", 0, 1),
        ("ry", 1, "theta[0]"),
        ("rz", 0, "theta[1]"),
    ),
)
ROWS = np.random.default_rng(41).uniform(-3, 3, (11, 4))


class PackingTests(unittest.TestCase):
    def test_marginals_batch_tails_noise_and_copy_isolation(self):
        self.assertEqual(marginal_zeros({"01": 3, "10": 1}, 2).tolist(), [0.25, 0.75])
        for noise in (
            {},
            {"depolarizing": 0.02, "amplitude_damping": 0.03, "readout_error": 0.08},
        ):
            config = Config(circuit=CIRCUIT, **noise)
            reference = Adapter(config).probabilities(ROWS, 4)
            for packing in (1, 2, 4):
                for batch in (1, 3, 64):
                    adapter = Adapter(replace(config, packing_factor=packing))
                    np.testing.assert_allclose(
                        adapter.probabilities(ROWS, batch), reference, atol=1e-12
                    )
                sampled = Adapter(replace(config, packing_factor=packing, shots=8192))
                np.testing.assert_allclose(
                    sampled.probabilities(ROWS, 3), reference, atol=0.03
                )
        qc, parameters, readouts = compose_copies(CIRCUIT, 3)
        self.assertEqual(readouts, [1, 3, 5])
        self.assertEqual(len(set(parameters)), 12)
        for inst in qc.data:
            self.assertEqual(len({qc.find_bit(q).index // 2 for q in inst.qubits}), 1)

    def test_resource_accounting_and_reconstructible_counts(self):
        config = Config(circuit=CIRCUIT, packing_factor=4, shots=512)
        adapter = Adapter(config)
        p = adapter.probabilities(ROWS, 3)
        self.assertEqual(adapter.statistics["backend_requests"], 2)
        self.assertEqual(adapter.statistics["composite_circuits"], 3)
        self.assertEqual(adapter.statistics["composite_shots"], 3 * 512)
        self.assertEqual(adapter.statistics["logical_shots"], 11 * 512)
        decoded = []
        for request in adapter.records:
            for experiment in request["experiments"]:
                decoded.extend(
                    marginal_zeros(experiment["counts"], request["active_packing"])
                )
                self.assertEqual(experiment["metadata"]["device"], "CPU")
        np.testing.assert_allclose(decoded, p)

    def test_gpu_request_never_falls_back_to_cpu(self):
        with patch.object(AerSimulator, "available_devices", return_value=("CPU",)):
            with self.assertRaisesRegex(ValueError, "GPU is unavailable"):
                Adapter(Config(circuit=CIRCUIT, aer_device="GPU"))
        for fields in (
            {"packing_factor": 0},
            {"packing_factor": True},
            {"aer_device": "auto"},
            {"aer_method": "unknown"},
            {"aer_method": "statevector", "depolarizing": 0.1},
        ):
            with self.assertRaises(ValueError):
                Config(circuit=CIRCUIT, **fields)

    @unittest.skipUnless(
        "GPU" in AerSimulator().available_devices(),
        "Requires GPU-enabled Aer and a compatible GPU",
    )
    def test_gpu_matches_cpu_and_reports_gpu(self):
        config = Config(circuit=CIRCUIT, packing_factor=2)
        gpu = Adapter(replace(config, aer_device="GPU"))
        np.testing.assert_allclose(
            gpu.probabilities(ROWS), Adapter(config).probabilities(ROWS), atol=1e-10
        )
        self.assertTrue(gpu.last_metadata)
        self.assertTrue(all(record["device"] == "GPU" for record in gpu.last_metadata))
