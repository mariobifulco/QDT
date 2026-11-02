import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from qdt.analysis import analyze, paired_summary
from qdt.campaign import completed_result, execute, load_plan, plan
from qdt.config import Config
from qdt.experiment import train_model
from qdt.model import write_json
from test_qdt import TEST_CIRCUIT


class CampaignTests(unittest.TestCase):
    def prepare(self, folder):
        root = Path(folder)
        config = Config(
            circuit=TEST_CIRCUIT,
            validation_size=0.2,
            split_seed=42,
            max_train_samples=40,
            max_depth=1,
            max_evals=8,
            restarts=1,
            top_k=1,
        )
        model = train_model(config)
        model.save(root / "model.json")
        manifest = {
            "schema_version": 1,
            "models": ["model.json"],
            "split": "validation",
            "executions": [
                {"name": "isolated", "overrides": {"packing_factor": 1}},
                {"name": "packed", "overrides": {"packing_factor": 3}},
            ],
        }
        path = root / "manifest.json"
        write_json(path, manifest)
        return root, path, manifest

    def test_plan_interruption_resume_and_result_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root, manifest_path, _ = self.prepare(folder)
            with patch(
                "qiskit_aer.AerSimulator.run",
                side_effect=AssertionError("No execution during planning"),
            ):
                campaign = plan(manifest_path)
            write_json(root / "plan.json", campaign)
            self.assertEqual(
                load_plan(root / "plan.json")["campaign_id"], campaign["campaign_id"]
            )
            with patch("qdt.campaign.evaluate", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    execute(root / "plan.json")
            self.assertFalse(list(root.glob("runs/*/claim.json")))
            self.assertEqual(len(list(root.glob("runs/*/attempt-*/failure.json"))), 1)
            self.assertEqual(execute(root / "plan.json")["completed"], 2)
            self.assertEqual(execute(root / "plan.json")["skipped"], 2)
            for run in campaign["runs"]:
                self.assertIsNotNone(completed_result(root, run))
            with patch(
                "qiskit_aer.AerSimulator.run",
                side_effect=AssertionError("Analysis must not execute circuits"),
            ):
                report = analyze(root / "plan.json", root / "analysis")
            self.assertEqual(report["completed_runs"], 2)
            self.assertEqual(report["comparisons"][0]["paired_models"], 1)
            self.assertIsNone(report["comparisons"][0]["quality_difference_ci95"])
            self.assertTrue((root / "analysis/runs.csv").exists())
            output = next(root.glob("runs/*/attempt-*/result.json"))
            output.write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum"):
                execute(root / "plan.json")

    def test_failed_attempts_reserve_shot_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root, manifest_path, manifest = self.prepare(folder)
            for execution in manifest["executions"]:
                execution["overrides"]["shots"] = 128
            manifest_path.write_text(json.dumps(manifest))
            campaign = plan(manifest_path)
            bound = campaign["logical_shots_upper_bound"]
            self.assertGreater(bound, 0)
            manifest["limits"] = {"max_logical_shots": bound}
            manifest_path.write_text(json.dumps(manifest))
            write_json(root / "plan.json", plan(manifest_path))
            with patch(
                "qdt.campaign.evaluate", side_effect=RuntimeError("Injected failure")
            ):
                with self.assertRaises(RuntimeError):
                    execute(root / "plan.json")
            resumed = execute(root / "plan.json")
            self.assertEqual(resumed["completed"], 1)
            self.assertTrue(resumed["budget_stop"])

    def test_paired_summary_uses_models_as_units(self):
        result = paired_summary([0.1, 0.1], [2, 2])
        self.assertEqual(result["paired_models"], 2)
        self.assertEqual(result["quality_difference_ci95"], [0.1, 0.1])
        self.assertEqual(result["speedup_ci95"], [2.0, 2.0])

    def test_unfinished_plans_reject_source_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root, manifest_path, _ = self.prepare(folder)
            campaign = plan(manifest_path)
            write_json(root / "plan.json", campaign)
            with patch(
                "qdt.campaign.provenance",
                return_value={**campaign["provenance"], "source_sha256": "changed"},
            ):
                with self.assertRaisesRegex(ValueError, "Code or environment changed"):
                    execute(root / "plan.json")
            self.assertFalse(list(root.glob("runs/*/attempt-*")))

    def test_limits_and_atomic_json(self):
        with tempfile.TemporaryDirectory() as folder:
            root, manifest_path, manifest = self.prepare(folder)
            manifest["limits"] = {"max_runs": 1}
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "max_runs"):
                plan(manifest_path)
            manifest["limits"] = {"max_logical_shots": 1}
            manifest["executions"][0]["overrides"]["shots"] = 128
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "max_logical_shots"):
                plan(manifest_path)
            path = root / "atomic.json"
            with self.assertRaises(ValueError):
                write_json(path, {"value": float("nan")})
            self.assertFalse(path.exists())
            write_json(path, asdict(Config(circuit=TEST_CIRCUIT)))
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_json(path, {})
            self.assertEqual(path.read_bytes(), original)
