import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from qiskit.primitives import BackendSamplerV2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator

from qdt.adapters import get_adapter
from qdt.config import Config
from qdt.experiment import train_model
from tests.test_packing import CIRCUIT, ROWS


@unittest.skipUnless(
    importlib.util.find_spec("qiskit_ibm_runtime"), "Install the ibm extra"
)
class IBMTests(unittest.TestCase):
    def services(self, folder, *, timeout=False, submission_error=False):
        backend = GenericBackendV2(
            4,
            basis_gates=["rz", "sx", "x", "cx"],
            coupling_map=[[0, 1], [1, 0], [2, 3], [3, 2]],
            noise_info=False,
            seed=7,
        )
        backend.configuration = lambda: SimpleNamespace(simulator=False)
        backend.status = lambda: SimpleNamespace(operational=True)
        backend.properties = lambda: SimpleNamespace(
            to_dict=lambda: {"calibration": "local-test"}
        )
        jobs = {}
        service = SimpleNamespace(
            backend=Mock(return_value=backend),
            job=Mock(side_effect=lambda key: jobs[key]),
        )
        sampler = Mock()

        def submit(pubs, *, shots):
            if submission_error:
                raise ConnectionError("Injected ambiguous submission")
            local = BackendSamplerV2(backend=AerSimulator(max_parallel_threads=1)).run(
                pubs, shots=shots
            )
            job_id = f"local-{len(jobs)}"
            remaining = [timeout]

            def result(timeout=None):
                if remaining[0]:
                    remaining[0] = False
                    raise TimeoutError("Injected wait timeout")
                return local.result()

            job = SimpleNamespace(
                job_id=lambda: job_id,
                result=result,
                metrics=lambda: {"usage": {"quantum_seconds": 1.0}},
            )
            jobs[job_id] = job
            return job

        sampler.run.side_effect = submit
        config = Config(
            circuit=CIRCUIT,
            backend="ibm",
            ibm_backend=backend.name,
            job_journal=folder,
            packing_factor=2,
            shots=1024,
            max_qpu_jobs=1,
            max_qpu_shots=2048,
        )
        return config, service, sampler

    def test_timeout_retrieves_accepted_job_and_cached_results_do_not_resubmit(self):
        with tempfile.TemporaryDirectory() as folder:
            config, service, sampler = self.services(folder, timeout=True)
            with (
                patch("qdt.adapters.ibm_run._ibm_service", return_value=service),
                patch("qiskit_ibm_runtime.SamplerV2", return_value=sampler),
            ):
                first = get_adapter(config)
                with self.assertRaises(TimeoutError):
                    first.probabilities(ROWS[:4], 4)
                self.assertTrue(
                    (Path(folder) / "request-000001/accepted.json").exists()
                )
                service.backend.return_value.status = lambda: SimpleNamespace(
                    operational=False
                )
                restored = get_adapter(config)
                actual = restored.probabilities(ROWS[:4], 4)
                expected = get_adapter(
                    replace(config, backend="aer", shots=0)
                ).probabilities(ROWS[:4], 4)
                np.testing.assert_allclose(actual, expected, atol=0.05)
                self.assertEqual(sampler.run.call_count, 1)
                self.assertEqual(restored.statistics["qpu_recovered_jobs"], 1)
                lookups = service.job.call_count
                cached = get_adapter(config)
                np.testing.assert_array_equal(cached.probabilities(ROWS[:4], 4), actual)
                self.assertEqual(service.job.call_count, lookups)
                self.assertEqual(cached.statistics["qpu_submitted_shots"], 0)
                with self.assertRaisesRegex(ValueError, "budget exhausted"):
                    cached.probabilities(ROWS[:4], 4)
                altered = ROWS[:4].copy()
                altered[0, 0] += 0.1
                with self.assertRaisesRegex(ValueError, "differs from this workload"):
                    get_adapter(config).probabilities(altered, 4)
                self.assertEqual(sampler.run.call_count, 1)

    def test_partial_packing_uses_a_frozen_tail_template(self):
        with tempfile.TemporaryDirectory() as folder:
            config, service, sampler = self.services(folder)
            config = replace(config, max_qpu_jobs=2, max_qpu_shots=3072)
            with (
                patch("qdt.adapters.ibm_run._ibm_service", return_value=service),
                patch("qiskit_ibm_runtime.SamplerV2", return_value=sampler),
            ):
                adapter = get_adapter(config)
                actual = adapter.probabilities(ROWS[:5], 4)
                reference = get_adapter(
                    replace(config, backend="aer", shots=0)
                ).probabilities(ROWS[:5])
                np.testing.assert_allclose(actual, reference, atol=0.05)
                self.assertEqual(sampler.run.call_count, 2)
                self.assertTrue((Path(folder) / "template-1.json").exists())
                self.assertEqual(adapter.statistics["qpu_submitted_shots"], 3072)
                np.testing.assert_array_equal(
                    get_adapter(config).probabilities(ROWS[:5], 4), actual
                )
                self.assertEqual(sampler.run.call_count, 2)

    def test_ambiguous_submission_never_retries_automatically(self):
        with tempfile.TemporaryDirectory() as folder:
            config, service, sampler = self.services(folder, submission_error=True)
            with (
                patch("qdt.adapters.ibm_run._ibm_service", return_value=service),
                patch("qiskit_ibm_runtime.SamplerV2", return_value=sampler),
            ):
                with self.assertRaises(ConnectionError):
                    get_adapter(config).probabilities(ROWS[:4], 4)
                with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                    get_adapter(config).probabilities(ROWS[:4], 4)
                self.assertEqual(sampler.run.call_count, 1)
                with self.assertRaisesRegex(ValueError, "Training uses local"):
                    train_model(config)

    def test_env_credentials_are_passed_without_replacing_saved_account_defaults(self):
        from qdt.qpu import _ibm_service

        with (
            patch("qdt.qpu._load_credentials"),
            patch.dict(
                "os.environ",
                {"QISKIT_IBM_TOKEN": "example", "QISKIT_IBM_INSTANCE": "instance"},
                clear=True,
            ),
            patch("qiskit_ibm_runtime.QiskitRuntimeService") as service,
        ):
            _ibm_service()
            service.assert_called_once_with(token="example", instance="instance")
        with self.assertRaises(ValueError):
            Config(circuit=CIRCUIT, backend="ibm")
