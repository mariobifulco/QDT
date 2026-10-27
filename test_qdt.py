import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from qdt.adapters.qiskit_run import Adapter
from qdt.circuit import Circuit
from qdt.config import Config
from qdt.data import DATASET_KEYS, prepare_dataset, select_features
from qdt.optimization import gini, optimize_split, soft_gini, split_score
from qdt.quantum_tree import fit, predict, run_probabilities


class QDTTests(unittest.TestCase):
    def test_circuit_validation_roundtrip_and_parameter_dimensions(self):
        circuit = Circuit(
            num_qubits=2,
            readout_qubit=1,
            gates=(
                ("ry", 0, "x[0]"),
                ("rx", 1, "x[1]"),
                ("cx", 0, 1),
                *(("ry", i % 2, f"theta[{i}]") for i in range(5)),
                ("rz", 0, "x[0]"),
                ("h", 0),
                ("cz", 0, 1),
                ("rx", 1, 0.3),
            ),
        )
        self.assertEqual((circuit.num_features, circuit.num_parameters), (2, 5))
        self.assertIs(circuit.template, circuit.template)
        config = Config(circuit=circuit, max_evals=14, restarts=2)
        decoded = Config(**json.loads(json.dumps(asdict(config))))
        self.assertEqual(config, decoded)
        calls = []
        labels = np.tile([0, 1], 5)

        def probabilities(theta):
            calls.append(theta.copy())
            return np.linspace(0.1, 0.9, len(labels))

        optimize_split(probabilities, labels, config, np.random.default_rng(4))
        self.assertTrue(calls)
        self.assertLessEqual(len(calls), config.max_evals)
        self.assertTrue(all(len(theta) == 5 for theta in calls))
        with self.assertRaises(ValueError):
            replace(config, max_evals=13)
        for definition in (
            {"num_qubits": 0},
            {"readout_qubit": 1},
            {"gates": [["ry", 0, "x[1]"], ["rz", 0, "theta[0]"]]},
            {"gates": [["ry", 0, "x[0]"], ["rz", 0, "theta[1]"]]},
            {"gates": [["ry", 0, "x[0]"], ["cx", 0, 0]]},
            {"gates": [["ry", 0, "x[0]+theta[0]"]]},
            {"gates": [["ry", 0, float("nan")]]},
            {"gates": [["unknown", 0]]},
        ):
            with self.subTest(definition=definition), self.assertRaises(ValueError):
                Circuit(**definition)
        with self.assertRaises(ValueError):
            Config(circuit=circuit, top_k=1)

    def test_multiqubit_entanglement_readout_and_template_reuse(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        circuit = Circuit(
            num_qubits=3,
            readout_qubit=2,
            gates=(
                ("ry", 0, "x[0]"),
                ("rx", 1, "x[1]"),
                ("h", 2),
                ("cx", 0, 1),
                ("cz", 1, 2),
                ("ry", 2, "theta[0]"),
                ("rz", 0, "theta[1]"),
                ("cx", 1, 2),
                ("ry", 0, "x[0]"),
            ),
        )
        rows = np.random.default_rng(81).uniform(-3, 3, (9, 4))
        expected = []
        for x0, x1, t0, t1 in rows:
            qc = QuantumCircuit(3)
            qc.ry(x0, 0)
            qc.rx(x1, 1)
            qc.h(2)
            qc.cx(0, 1)
            qc.cz(1, 2)
            qc.ry(t0, 2)
            qc.rz(t1, 0)
            qc.cx(1, 2)
            qc.ry(x0, 0)
            expected.append(Statevector.from_instruction(qc).probabilities([2])[0])
        config = Config(circuit=circuit)
        adapter = Adapter(config)
        template = adapter.template
        for batch in (1, 4, 64):
            np.testing.assert_allclose(
                adapter.probabilities(rows, batch), expected, atol=1e-12
            )
            self.assertIs(adapter.template, template)
            self.assertEqual(len(adapter.template.parameters), 4)
        noisy = replace(
            config,
            depolarizing=0.03,
            amplitude_damping=0.02,
            phase_damping=0.01,
            readout_error=0.04,
        )
        exact = Adapter(noisy).probabilities(rows, 4)
        sampled = replace(noisy, shots=8192)
        actual = Adapter(sampled).probabilities(rows, 4)
        np.testing.assert_allclose(actual, exact, atol=0.03)
        np.testing.assert_array_equal(actual, Adapter(sampled).probabilities(rows, 4))

    def test_multifeature_tree_groups_and_saved_parameters(self):
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
        config = Config(circuit=circuit, max_depth=2, max_evals=8, restarts=1)
        X = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (5, 1))
        y = (X[:, 0] != X[:, 1]).astype(int)
        X = np.column_stack((X, np.linspace(0, 0.1, len(X)), np.zeros(len(X))))
        adapter = Adapter(config)
        tree = fit(X, y, config, adapter=adapter)
        self.assertIsNone(tree.value)
        self.assertEqual(len(tree.feature_indices), 2)
        self.assertEqual(len(tree.theta), 2)
        np.testing.assert_array_equal(predict(X, tree, config), y)
        self.assertLessEqual(
            adapter.calls, (X.shape[1] - circuit.num_features + 1) * config.max_evals
        )
        with self.assertRaises(ValueError):
            fit(X[:, :1], y, config)
        # A different architecture also supports more than two fitted angles.
        expanded = replace(
            circuit,
            gates=(*circuit.gates, ("rx", 1, "theta[2]"), ("ry", 1, "theta[3]")),
        )
        config = replace(config, circuit=expanded)
        tree = fit(X[:, :2], y, config)
        self.assertIsNone(tree.value)
        self.assertEqual(len(tree.theta), 4)
        np.testing.assert_array_equal(predict(X[:, :2], tree, config), y)

    def test_aer_ideal_matches_formula_and_batching(self):
        angles = np.random.default_rng(4).uniform(-4, 4, (9, 3))
        a, z, b = angles.T
        expected = (1 + np.cos(a) * np.cos(b) - np.sin(a) * np.cos(z) * np.sin(b)) / 2
        adapter = Adapter(Config())
        self.assertEqual(adapter.device_name, "statevector")
        for batch in (1, 4, 20):
            np.testing.assert_allclose(
                adapter.probabilities(angles, batch), expected, atol=1e-12
            )
        self.assertEqual(adapter.probabilities([]).size, 0)

    def test_noise_limits_and_shot_reproducibility(self):
        angles = [(np.pi, 0, 0), (0, 0, 0), (0.4, 0.6, 0.2)]
        for noise, expected in (
            ({"depolarizing": 1}, [0.5, 0.5, 0.5]),
            ({"amplitude_damping": 1}, [1, 1, 1]),
        ):
            adapter = Adapter(Config(**noise))
            self.assertEqual(adapter.device_name, "density_matrix")
            np.testing.assert_allclose(
                adapter.probabilities(angles, 2), expected, atol=1e-12
            )
        ideal = Adapter(Config()).probabilities(angles, 2)
        inverted = Adapter(Config(readout_error=1)).probabilities(angles, 2)
        np.testing.assert_allclose(inverted, 1 - ideal, atol=1e-12)
        noise = Config(
            depolarizing=0.02,
            amplitude_damping=0.03,
            phase_damping=0.04,
            readout_error=0.05,
        )
        exact = Adapter(noise).probabilities(angles, 2)
        sampled = replace(noise, shots=8192)
        actual = Adapter(sampled).probabilities(angles, 2)
        np.testing.assert_array_equal(actual, Adapter(sampled).probabilities(angles, 2))
        np.testing.assert_allclose(actual, exact, atol=0.03)
        np.testing.assert_array_equal(
            Adapter(Config(shots=128)).probabilities(angles[:2], 2), [0, 1]
        )

    def test_soft_gini_and_cobyla_budget(self):
        values = np.linspace(0, 1, 40)
        labels = (values > 0.5).astype(int)
        mask = values < 0.5
        self.assertAlmostEqual(
            soft_gini(mask.astype(float), labels), split_score(mask, labels)
        )
        self.assertAlmostEqual(soft_gini(np.full(40, 0.5), labels), gini(labels))
        config = Config(max_evals=9, restarts=2)
        adapter = Adapter(config)

        def probabilities(theta):
            return run_probabilities(values, theta, adapter, config)

        result = optimize_split(
            probabilities, labels, config, np.random.default_rng(42)
        )
        self.assertIsNotNone(result)
        self.assertLessEqual(adapter.calls, config.max_evals)
        theta, mask = result
        np.testing.assert_array_equal(mask, probabilities(theta) >= 0.5)

    def test_anova_rejects_constants_and_redundancy(self):
        rng = np.random.default_rng(13)
        y = np.repeat([0, 1], 100)
        signal = y + rng.normal(0, 0.05, len(y))
        X = np.column_stack([signal, -signal, np.ones(len(y)), rng.normal(size=len(y))])
        features = select_features(X, y, Config(top_k=2))
        self.assertIn(features[0], (0, 1))
        self.assertEqual(features[1], 3)

    def test_initial_filter_is_training_only_and_not_repeated_at_nodes(self):
        config = Config(top_k=2, max_evals=8, restarts=1)
        with patch("qdt.data.select_features", wraps=select_features) as selector:
            data = prepare_dataset(config)
            first = fit(data.X_train, data.y_train, config)
            self.assertEqual(selector.call_count, 1)
        second = fit(data.X_train, data.y_train, config)
        self.assertEqual(asdict(first), asdict(second))
        np.testing.assert_array_equal(
            predict(data.X_test, first, config),
            predict(data.X_test, second, replace(config, batch_size=3)),
        )
        train = np.column_stack([np.repeat([0.1, 0.9], 10), np.tile([0.2, 0.3], 10)])
        train[:, 0] += np.tile(np.linspace(-0.001, 0.001, 10), 2)
        labels = np.repeat([0, 1], 10)
        for test, test_labels in (
            (np.ones((4, 2)), np.array([0, 1, 0, 1])),
            (np.full((4, 2), 1e6), np.array([1, 1, 0, 0])),
        ):
            with patch(
                "qdt.data.train_test_split",
                return_value=(train, test, labels, test_labels),
            ):
                prepared = prepare_dataset(replace(config, top_k=1))
                np.testing.assert_array_equal(prepared.feature_indices, [0])
                np.testing.assert_allclose(
                    prepared.X_train[:, 0], train[:, 0] / np.abs(train[:, 0]).max()
                )
        adapter = Adapter(config)
        leaf = fit(np.ones((10, 2)), np.tile([0, 1], 5), config, adapter=adapter)
        self.assertEqual(leaf.value, 0)
        self.assertEqual(adapter.calls, 0)

    def test_preparation_scaling_and_original_feature_indices(self):
        config = Config(top_k=None)
        data = prepare_dataset(config)
        raw = load_iris()
        train, test, _, _ = train_test_split(
            raw.data,
            raw.target,
            test_size=config.test_size,
            stratify=raw.target,
            random_state=config.seed,
        )
        scales = np.abs(train).max(axis=0)
        np.testing.assert_allclose(data.X_train, train / scales)
        np.testing.assert_allclose(data.X_test, test / scales)
        data = prepare_dataset(Config(dataset="breast_cancer", top_k=4))
        self.assertEqual(data.original_dim, 30)
        self.assertEqual(data.X_train.shape[1], len(data.feature_indices))
        self.assertLessEqual(len(data.feature_indices), 4)
        for key in DATASET_KEYS:
            if key not in {"ionosphere", "sonar"}:
                prepared = prepare_dataset(replace(config, dataset=key))
                self.assertTrue(np.isfinite(prepared.X_train).all())

    def test_invalid_configs_and_example_files(self):
        for values in (
            {"top_k": 0},
            {"shots": -1},
            {"readout_error": float("nan")},
            {"amplitude_damping": 1.1},
            {"max_evals": 7},
        ):
            with self.assertRaises(ValueError):
                Config(**values)
        repository = Path(__file__).resolve().parent
        for filename, device in (
            ("default.json", "statevector"),
            ("noise.json", "density_matrix"),
        ):
            config = Config.load(repository / "configs" / filename)
            self.assertEqual(Adapter(config).device_name, device)
        with self.assertRaises(ValueError):
            Config(backend="pennylane")

    def test_slurm_and_cli_output_overwrite_protection(self):
        repository = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "run.json"
            config.write_text(
                json.dumps(
                    asdict(Config(max_train_samples=40, max_evals=8, restarts=1))
                )
            )
            srun = root / "srun"
            srun.write_text('#!/bin/sh\nexec "$@"\n')
            srun.chmod(0o755)
            env = dict(
                os.environ,
                PATH=str(root) + os.pathsep + os.environ["PATH"],
                QDT_PYTHON=sys.executable,
                SLURM_ARRAY_TASK_ID="3",
            )
            command = [
                "bash",
                str(repository / "slurm/run.sbatch"),
                str(config),
                str(root / "results"),
            ]
            subprocess.run(
                command, env=env, cwd=repository, check=True, capture_output=True
            )
            result = json.loads((root / "results/3.json").read_text())
            self.assertEqual(result["config"]["seed"], 3)
            self.assertEqual(result["qdt"]["optimizer"], "cobyla")
            self.assertEqual(result["qdt"]["device"], "statevector")
            self.assertEqual(result["selected_dim"], len(result["feature_indices"]))
            self.assertLessEqual(result["selected_dim"], result["config"]["top_k"])
            self.assertIn("macro_f1", result["cart"])
            self.assertNotEqual(
                subprocess.run(
                    command, env=env, cwd=repository, capture_output=True
                ).returncode,
                0,
            )


if __name__ == "__main__":
    unittest.main()
