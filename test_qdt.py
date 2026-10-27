"""Small regression suite: python -m unittest -v test_qdt."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from qdt.adapters import get_adapter
from qdt.config import Config
from qdt.data import DATASET_KEYS, prepare_dataset
from qdt.optimization import loss_and_gradient, optimize_split, soft_gini
from qdt.quantum_tree import fit, predict, run_probabilities, select_features


class CoreTests(unittest.TestCase):
    def test_parameter_shift_chain_rule_matches_finite_differences(self):
        values = np.linspace(-0.9, 0.9, 15)
        labels = np.tile([0, 1, 2], 5)
        theta, epsilon = np.array([0.7, -0.9]), 1e-6
        for config in (
            Config(),
            Config(
                backend="aer",
                depolarizing=0.02,
                amplitude_damping=0.03,
                phase_damping=0.04,
                readout_error=0.05,
            ),
        ):
            adapter = get_adapter(config)

            def probabilities(parameters, adapter=adapter, config=config):
                return run_probabilities(values, parameters, adapter, config)

            loss, gradient = loss_and_gradient(probabilities, theta, labels)
            numerical = []
            for parameter in range(2):
                shift = np.zeros(2)
                shift[parameter] = epsilon
                numerical.append(
                    (
                        soft_gini(probabilities(theta + shift), labels)[0]
                        - soft_gini(probabilities(theta - shift), labels)[0]
                    )
                    / (2 * epsilon)
                )
            self.assertAlmostEqual(loss, soft_gini(probabilities(theta), labels)[0])
            np.testing.assert_allclose(gradient, numerical, atol=1e-8)

    def test_optimizers_respect_probability_budget_and_preserve_partition(self):
        values = np.linspace(0, 1, 40)
        labels = (values > 0.5).astype(int)
        for method in ("cobyla", "lbfgs"):
            config = Config(optimizer=method, max_evals=30)
            adapter = get_adapter(config)

            def probabilities(theta, adapter=adapter, config=config):
                return run_probabilities(values, theta, adapter, config)

            result = optimize_split(
                probabilities, labels, config, np.random.default_rng(42)
            )
            self.assertIsNotNone(result)
            self.assertLessEqual(adapter.calls, config.max_evals)
            theta, mask = result
            np.testing.assert_array_equal(mask, probabilities(theta) >= 0.5)

    def test_noise_factorization_and_shot_bit_order(self):
        angles = np.random.default_rng(12).uniform(-2, 2, (5, 3)).tolist()
        config = Config(
            backend="aer", depolarizing=0.02, amplitude_damping=0.03, readout_error=0.04
        )
        expected = get_adapter(config).probabilities(angles, 3)
        actual = get_adapter(replace(config, backend="aer_stitched")).probabilities(
            angles, 3
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        basis = [(0, 0, 0), (np.pi, 0, 0), (0, 0, 0)]
        config = Config(backend="aer_stitched", shots=128)
        np.testing.assert_array_equal(
            get_adapter(config).probabilities(basis, 2), [1, 0, 1]
        )
        config = replace(config, readout_error=0.1)
        np.testing.assert_array_equal(
            get_adapter(config).probabilities(basis, 2),
            get_adapter(config).probabilities(basis, 2),
        )

    def test_ideal_formula_matches_aer(self):
        angles = np.random.default_rng(4).uniform(-4, 4, (9, 3)).tolist()
        expected = get_adapter(Config()).probabilities(angles, 4)
        actual = get_adapter(Config(backend="aer")).probabilities(angles, 4)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_filter_and_deterministic_tree(self):
        rng = np.random.default_rng(13)
        y = np.repeat([0, 1], 100)
        signal = y + rng.normal(0, 0.05, len(y))
        X = np.column_stack([signal, -signal, np.ones(len(y)), rng.normal(size=len(y))])
        features = select_features(X, y, Config(top_k=2))
        self.assertIn(features[0], (0, 1))
        self.assertEqual(features[1], 3)
        data = prepare_dataset(Config())
        for method in ("cobyla", "lbfgs"):
            config = Config(optimizer=method, max_evals=20)
            first = fit(data.X_train, data.y_train, config)
            second = fit(data.X_train, data.y_train, config)
            self.assertEqual(asdict(first), asdict(second))
            np.testing.assert_array_equal(
                predict(data.X_test, first, config),
                predict(data.X_test, second, replace(config, batch_size=3)),
            )
        self.assertEqual(fit(np.ones((10, 2)), np.tile([0, 1], 5)).value, 0)

    def test_training_only_scaling_and_all_builtin_datasets(self):
        config = Config()
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
        for dataset in DATASET_KEYS:
            if dataset not in {"ionosphere", "sonar"}:
                prepared = prepare_dataset(replace(config, dataset=dataset))
                self.assertTrue(np.isfinite(prepared.X_train).all())

    def test_config_validation(self):
        for values in (
            {"optimizer": "spsa"},
            {"top_k": 0},
            {"depolarizing": 0.1},
            {"backend": "aer", "readout_error": float("nan")},
            {"optimizer": "lbfgs", "max_evals": 8},
        ):
            with self.assertRaises(ValueError):
                Config(**values)

    def test_slurm_entry_point_and_overwrite_protection(self):
        repository = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "run.json"
            config.write_text(
                json.dumps(asdict(Config(max_train_samples=40, max_evals=10)))
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
            self.assertIn("accuracy", result["qdt"])
            self.assertIn("macro_f1", result["cart"])
            self.assertNotEqual(
                subprocess.run(
                    command, env=env, cwd=repository, capture_output=True
                ).returncode,
                0,
            )


if __name__ == "__main__":
    unittest.main()
