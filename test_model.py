import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from qdt.config import Config
from qdt.data import load_dataset, prepare_dataset
from qdt.model import Model
from qdt.quantum_tree import fit
from test_qdt import TEST_CIRCUIT


class ModelTests(unittest.TestCase):
    def test_train_predict_cli_and_trace_budgets(self):
        config = Config(
            circuit=TEST_CIRCUIT,
            validation_size=0.2,
            max_train_samples=40,
            max_depth=1,
            max_evals=8,
            restarts=1,
            top_k=1,
            split_seed=7,
            optimizer_seed=11,
            simulator_seed=13,
            transpiler_seed=17,
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path, model_path, output = (
                root / "config.json",
                root / "model.json",
                root / "out.json",
            )
            config_path.write_text(json.dumps(asdict(config)))
            command = [sys.executable, "-m", "qdt"]
            subprocess.run(
                [
                    *command,
                    "train",
                    "--config",
                    str(config_path),
                    "--output",
                    str(model_path),
                ],
                check=True,
                capture_output=True,
            )
            original = model_path.read_bytes()
            model = Model.load(model_path)
            trace = model.data["training"]["trace"]
            self.assertTrue(any(event["event"] == "evaluation" for event in trace))
            for event in trace:
                if event["event"] == "restart_end":
                    self.assertLessEqual(event["evaluations"], event["budget"])
                    self.assertIn("message", event)
            self.assertEqual(
                model.data["training"]["seeds"],
                {"split": 7, "optimizer": 11, "simulator": 13, "transpiler": 17},
            )
            subprocess.run(
                [
                    *command,
                    "predict",
                    "--model",
                    str(model_path),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
            )
            result = json.loads(output.read_text())
            self.assertEqual(result["split"], "validation")
            self.assertEqual(result["model_id"], model.model_id)
            self.assertEqual(len(result["predictions"]), 30)
            self.assertEqual(model_path.read_bytes(), original)
            self.assertNotEqual(
                subprocess.run(
                    [
                        *command,
                        "predict",
                        "--model",
                        str(model_path),
                        "--output",
                        str(output),
                    ],
                    capture_output=True,
                ).returncode,
                0,
            )

    def test_frozen_model_roundtrip_and_training_only_preprocessing(self):
        config = Config(
            circuit=TEST_CIRCUIT, validation_size=0.2, max_evals=8, restarts=1
        )
        data = prepare_dataset(config)
        X, _ = load_dataset(config)
        splits = data.split_indices
        self.assertEqual(
            [len(splits[k]) for k in ("train", "validation", "test")], [90, 30, 30]
        )
        self.assertEqual(len(set(sum(splits.values(), []))), len(X))
        changed = X.copy()
        changed[splits["validation"] + splits["test"]] *= 1e6
        with patch(
            "qdt.data.load_dataset", return_value=(changed, load_dataset(config)[1])
        ):
            self.assertEqual(prepare_dataset(config).preprocessing, data.preprocessing)
        self.assertEqual(
            prepare_dataset(replace(config, seed=123, split_seed=42)).split_indices,
            splits,
        )
        model = Model(
            config,
            data.preprocessing,
            fit(data.X_train, data.y_train, config),
            [0, 1, 2],
            {"split_indices": splits, "dataset_sha256": data.dataset_sha256},
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            model.save(path)
            restored = Model.load(path)
            np.testing.assert_array_equal(
                model.predict(X), restored.predict(X, execution={"batch_size": 3})
            )
            self.assertEqual(model.model_id, restored.model_id)
            check = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from qdt.model import Model; from qdt.data import load_dataset; import sys; "
                    "m=Model.load(sys.argv[1]); print(m.predict(load_dataset(m.config)[0]).tolist())",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(check.stdout), model.predict(X).tolist())
            with self.assertRaises(FileExistsError):
                model.save(path)
            with self.assertRaises(ValueError):
                model.predict(X, execution={"max_depth": 0})
            with self.assertRaises(ValueError):
                model.predict(X[:, :1])
            payload = json.loads(path.read_text())
            payload["tree"]["theta"] = [0]
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                Model.load(path)
