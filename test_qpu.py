import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import chdir
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from qdt.adapters import get_adapter
from qdt.circuit import Circuit
from qdt.config import Config
from qdt.qpu import (
    IQM_DEVICES,
    Profile,
    _load_credentials,
    fetch_profile,
    list_online_ibm,
    list_qpus,
    load_profile,
    write_profile,
)

PAIR = Circuit(
    num_qubits=2,
    readout_qubit=1,
    gates=(("ry", 0, "x[0]"), ("cx", 0, 1), ("ry", 1, "theta[0]")),
)
ROWS = np.array([[0, 0], [np.pi, 0], [0.4, 0.8]])


def ibm_test_profile():
    backend = GenericBackendV2(2, basis_gates=["ry", "rz", "cx"], seed=7)
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.003, 1), "ry")
    noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), "cx")
    return Profile(
        backend.target,
        noise,
        {"source": "ibm:test", "kind": "downloaded calibration"},
    )


class QPUProfileTests(unittest.TestCase):
    def test_local_catalog_has_only_iqm(self):
        with patch("qdt.qpu._ibm_service", side_effect=AssertionError("No network")):
            self.assertEqual(
                {entry["name"] for entry in list_qpus()},
                {f"iqm:{name}" for name in IQM_DEVICES},
            )
        with self.assertRaisesRegex(ValueError, "snapshots are disabled"):
            load_profile("ibm:torino")

    @unittest.skipUnless(importlib.util.find_spec("dotenv"), "Install ibm extra")
    def test_dotenv_skips_empty_values_and_preserves_environment(self):
        with tempfile.TemporaryDirectory() as folder, chdir(folder):
            Path(".env").write_text(
                'QISKIT_IBM_TOKEN="example-token"\nQISKIT_IBM_INSTANCE=example-instance\nQISKIT_IBM_CHANNEL=\nUNUSED_PROVIDER_TOKEN=unused\n'
            )
            with patch.dict(
                os.environ, {"QISKIT_IBM_TOKEN": "existing-token"}, clear=True
            ):
                _load_credentials()
                self.assertEqual(os.environ["QISKIT_IBM_TOKEN"], "existing-token")
                self.assertEqual(os.environ["QISKIT_IBM_INSTANCE"], "example-instance")
                self.assertNotIn("QISKIT_IBM_CHANNEL", os.environ)
                self.assertNotIn("UNUSED_PROVIDER_TOKEN", os.environ)

    def test_online_ibm_listing_and_fetch_filter_status(self):
        backend = GenericBackendV2(
            20,
            basis_gates=["rz", "sx", "x", "cx"],
            coupling_map=[[q, q + 1] for q in range(19)],
            seed=7,
        )
        properties = SimpleNamespace(
            last_update_date="2026-10-28T00:00:00Z",
            qubits=[
                [
                    SimpleNamespace(name="prob_meas1_prep0", value=0.07),
                    SimpleNamespace(name="prob_meas0_prep1", value=0.21),
                ]
                for _ in range(20)
            ],
        )
        backend.properties = lambda: properties
        backend.configuration = lambda: SimpleNamespace(simulator=False)
        backend.status = lambda: SimpleNamespace(operational=True)
        with (
            patch("qdt.qpu._ibm_service") as service,
            patch("qdt.qpu.importlib.metadata.version", return_value="test-version"),
            patch.object(backend, "run") as run,
        ):
            service.return_value.backends.return_value = [
                SimpleNamespace(name="ibm_current", num_qubits=20)
            ]
            self.assertEqual(
                list_online_ibm(), [{"name": "ibm:ibm_current", "qubits": 20}]
            )
            service.return_value.backends.assert_called_once_with(
                operational=True, simulator=False
            )
            service.return_value.backend.return_value = backend
            profile = fetch_profile("ibm:ibm_current")
            self.assertEqual(profile.metadata["kind"], "downloaded calibration")
            service.return_value.backend.assert_called_once_with("ibm_current")
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "ibm.json"
                write_profile(profile, path)
                config = Config(
                    fake_qpu=f"file:{path}", circuit=PAIR, qpu_layout=[10, 11]
                )
                with patch(
                    "qdt.qpu._ibm_service",
                    side_effect=AssertionError("No network during simulation"),
                ):
                    adapter = get_adapter(config)
                    self.assertLess(adapter.template.num_qubits, 5)
                    np.testing.assert_allclose(
                        adapter.readout, [[0.93, 0.07], [0.21, 0.79]]
                    )
                    self.assertTrue(np.isfinite(adapter.probabilities(ROWS, 3)).all())
                    with self.assertRaises(ValueError):
                        get_adapter(replace(config, qpu_layout=[20, 21]))
                payload = json.loads(path.read_text())
                payload["metadata"]["kind"] = "library snapshot"
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "snapshots are disabled"):
                    load_profile(f"file:{path}")
            for simulator, operational in ((False, False), (True, True)):
                backend.configuration = lambda simulator=simulator: SimpleNamespace(
                    simulator=simulator
                )
                backend.status = lambda operational=operational: SimpleNamespace(
                    operational=operational
                )
                with self.assertRaisesRegex(ValueError, "operational IBM QPUs"):
                    fetch_profile("ibm:ibm_unavailable")
            run.assert_not_called()

    def test_config_boundaries(self):
        for values in (
            {"fake_qpu": ""},
            {"fake_qpu": True},
            {"fake_qpu": "iqm:adonis", "backend": "unsupported"},
            {"fake_qpu": "iqm:adonis", "readout_error": 0.01},
            {"qpu_layout": [0]},
            {"fake_qpu": "iqm:adonis", "qpu_layout": [-1]},
            {"fake_qpu": "iqm:adonis", "circuit": PAIR, "qpu_layout": [0, 0]},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config(**{"circuit": PAIR, **values})
        for selector in (
            "unknown:qpu",
            "aqt:offline",
            "aqt:offline_simulator_noise",
            "ibm:no_such_device",
            "iqm:unknown",
            "bad",
        ):
            with self.assertRaises((ValueError, ImportError)):
                load_profile(selector)

    def test_ibm_snapshot_roundtrip_exact_asymmetric_readout_and_sampling(self):
        profile = ibm_test_profile()
        profile.noise.add_all_qubit_readout_error(
            ReadoutError([[0.93, 0.07], [0.21, 0.79]])
        )
        config = Config(fake_qpu="ibm:test", circuit=PAIR)
        with patch("qdt.adapters.qpu_run.load_profile", return_value=profile):
            exact_adapter = get_adapter(config)
        exact = exact_adapter.probabilities(ROWS, 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ibm.json"
            write_profile(profile, path)
            with self.assertRaises(FileExistsError):
                write_profile(profile, path)
            restored = replace(config, fake_qpu=f"file:{path}")
            np.testing.assert_allclose(
                get_adapter(restored).probabilities(ROWS, 3), exact, atol=1e-12
            )
            sampled = replace(restored, shots=8192)
            first = get_adapter(sampled).probabilities(ROWS, 2)
            np.testing.assert_allclose(first, exact, atol=0.025)
            np.testing.assert_array_equal(
                first, get_adapter(sampled).probabilities(ROWS, 2)
            )
            self.assertEqual(
                len(get_adapter(restored).profile_metadata["snapshot_sha256"]), 64
            )
            payload = json.loads(path.read_text())
            for source in ("other:qpu", "aqt:offline"):
                payload["metadata"]["source"] = source
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "Only IBM and IQM"):
                    load_profile(f"file:{path}")

    @unittest.skipUnless(importlib.util.find_spec("iqm"), "Install iqm extra")
    def test_all_iqm_profiles_local_multiqubit_and_move_noise(self):
        with patch(
            "iqm.qiskit_iqm.IQMProvider",
            side_effect=AssertionError("Remote provider forbidden"),
        ):
            for name in IQM_DEVICES:
                config = Config(fake_qpu=f"iqm:{name}", circuit=PAIR)
                adapter = get_adapter(config)
                exact = adapter.probabilities(ROWS, 2)
                np.testing.assert_allclose(exact[:2], [1, 0], atol=0.2)
                self.assertLessEqual(adapter.template.num_qubits, 3)
                if name == "deneb":
                    self.assertGreater(
                        adapter.profile_metadata["native_gate_counts"]["move"], 0
                    )
                changed_seed = get_adapter(replace(config, seed=999))
                self.assertEqual(
                    adapter.profile_metadata["physical_qubits"],
                    changed_seed.profile_metadata["physical_qubits"],
                )
                np.testing.assert_allclose(
                    exact, changed_seed.probabilities(ROWS, 1), atol=1e-12
                )
        with self.assertRaisesRegex(ValueError, "Routing leaves"):
            get_adapter(Config(circuit=PAIR, fake_qpu="iqm:deneb", packing_factor=2))
        profile = load_profile("iqm:deneb")
        profile.noise = NoiseModel()
        profile.noise.add_all_qubit_quantum_error(depolarizing_error(1, 2), "move")
        with patch("qdt.adapters.qpu_run.load_profile", return_value=profile):
            noisy = get_adapter(Config(fake_qpu="iqm:deneb", circuit=PAIR))
        np.testing.assert_allclose(
            noisy.probabilities(ROWS[:2]), [0.5, 0.5], atol=1e-12
        )

    def test_packed_profile_marginals_and_explicit_partition_constraints(self):
        backend = GenericBackendV2(
            4,
            basis_gates=["ry", "rz", "cx"],
            coupling_map=[[0, 1], [1, 0], [2, 3], [3, 2]],
            seed=7,
        )
        noise = NoiseModel()
        noise.add_readout_error(ReadoutError([[0.9, 0.1], [0.2, 0.8]]), [1])
        noise.add_readout_error(ReadoutError([[0.8, 0.2], [0.3, 0.7]]), [3])
        profile = Profile(backend.target, noise, {"source": "ibm:partition-test"})
        rows = np.vstack([ROWS, ROWS[:2]])
        config = Config(
            circuit=PAIR, fake_qpu="ibm:test", packing_factor=2, qpu_layout=[0, 1, 2, 3]
        )
        with patch("qdt.adapters.qpu_run.load_profile", return_value=profile):
            packed = get_adapter(config)
            p = packed.probabilities(rows, 2)
            for i in range(2):
                isolated = get_adapter(
                    replace(config, packing_factor=1, qpu_layout=[2 * i, 2 * i + 1])
                )
                np.testing.assert_allclose(
                    p[i::2], isolated.probabilities(rows[i::2]), atol=1e-12
                )
            sampled = get_adapter(replace(config, shots=8192))
            np.testing.assert_allclose(sampled.probabilities(rows, 2), p, atol=0.03)
            self.assertEqual(packed.profile_metadata["physical_readouts"], [1, 3])
            automatic = get_adapter(replace(config, qpu_layout=None))
            explicit = get_adapter(
                replace(config, qpu_layout=automatic.selected_layout)
            )
            np.testing.assert_allclose(
                automatic.probabilities(rows), explicit.probabilities(rows), atol=1e-12
            )
            self.assertEqual(
                automatic.profile_metadata["partition_selection"], "greedy"
            )

    def test_cli_local_profile_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            profile_path = Path(folder) / "ibm.json"
            write_profile(ibm_test_profile(), profile_path)
            output = Path(folder) / "result.json"
            config = Path(folder) / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "circuit": asdict(PAIR),
                        "fake_qpu": f"file:{profile_path}",
                        "max_train_samples": 20,
                        "max_evals": 4,
                        "restarts": 1,
                        "max_depth": 1,
                    }
                )
            )
            env = dict(os.environ, IQM_BACKEND="qpu")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qdt",
                    "--config",
                    str(config),
                    "--output",
                    str(output),
                ],
                env=env,
                check=True,
                capture_output=True,
            )
            result = json.loads(output.read_text())
            self.assertEqual(result["qdt"]["device"], "local:ibm:test")
            self.assertEqual(
                result["qdt"]["noise_profile"]["simulation"], "local Aer density_matrix"
            )


if __name__ == "__main__":
    unittest.main()
