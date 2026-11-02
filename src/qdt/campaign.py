"""Small, immutable local campaigns; no remote job submission."""

import argparse
import hashlib
import json
import os
import socket
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .adapters import get_adapter
from .data import dataset_checksum, load_dataset
from .experiment import evaluate, provenance
from .model import Model, write_json


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def environment_id(value):
    return identity(
        {
            key: value.get(key)
            for key in (
                "source_sha256",
                "versions",
                "python",
                "platform",
                "lockfile_sha256",
            )
        }
    )


def file_checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def plan(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported campaign schema")
    if set(manifest) - {
        "schema_version",
        "models",
        "executions",
        "repetitions",
        "split",
        "limits",
        "max_samples",
    }:
        raise ValueError("Unknown campaign fields")
    repetitions = positive_integer(manifest.get("repetitions", 1), "repetitions")
    limits = manifest.get("limits", {})
    if set(limits) - {"max_runs", "max_logical_shots", "max_seconds"}:
        raise ValueError("Unknown campaign limits")
    limits = {
        key: positive_integer(limits.get(key, default), key)
        for key, default in [
            ("max_runs", 100),
            ("max_logical_shots", 10000000),
            ("max_seconds", 3600),
        ]
    }
    maximum = manifest.get("max_samples")
    if maximum is not None:
        positive_integer(maximum, "max_samples")
    environment = provenance()
    split = manifest.get("split", "validation")
    if split not in {"validation", "test"}:
        raise ValueError("Campaign split must be validation or test")
    models, executions = manifest["models"], manifest["executions"]
    if (
        not models
        or not executions
        or not isinstance(models, list)
        or not isinstance(executions, list)
    ):
        raise ValueError("Campaign needs nonempty model and execution lists")
    if len(models) * len(executions) * repetitions > limits["max_runs"]:
        raise ValueError("Planned run count exceeds max_runs")
    names = [item["name"] for item in executions]
    if any(not isinstance(name, str) or not name for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("Execution names must be nonempty and unique")
    runs = []
    model_ids = set()

    def depth(node):
        return (
            0
            if node.value is not None
            else 1 + max(depth(node.left), depth(node.right))
        )

    for name in models:
        path = (manifest_path.parent / name).resolve()
        model = Model.load(path)
        model_id = model.model_id
        if model_id in model_ids:
            raise ValueError("Duplicate model artifact in campaign")
        model_ids.add(model_id)
        X, y = load_dataset(model.config)
        if dataset_checksum(X, y) != model.data["dataset_sha256"]:
            raise ValueError("Dataset differs from the frozen model")
        indices = model.data["split_indices"][split][:maximum]
        if not indices:
            raise ValueError(f"Model has no {split} samples")
        logical_bound = len(indices) * depth(model.tree)
        for execution in executions:
            if set(execution) - {"name", "overrides"}:
                raise ValueError("Unknown execution fields")
            initial_overrides = dict(execution.get("overrides", {}))
            selector = initial_overrides.get("fake_qpu", model.config.fake_qpu)
            profile_file = None
            if isinstance(selector, str) and selector.startswith("file:"):
                profile_file = str(Path(selector.removeprefix("file:")).resolve())
                initial_overrides["fake_qpu"] = f"file:{profile_file}"
            base = model.execution_config(initial_overrides)
            if base.backend != "aer":
                raise ValueError(
                    "Campaigns currently run local simulations; use predict with a unique IBM journal"
                )
            # Validate feasibility and device support without calling backend.run.
            get_adapter(base)
            for repetition in range(repetitions):
                overrides = {
                    **initial_overrides,
                    "simulator_seed": base.seed_for("simulator") + repetition,
                }
                resolved = model.execution_config(overrides)
                spec = {
                    "model_id": model_id,
                    "environment_id": environment_id(environment),
                    "max_samples": maximum,
                    "sample_indices": indices,
                    "model_sha256": file_checksum(path),
                    "profile_file": profile_file,
                    "profile_sha256": file_checksum(profile_file)
                    if profile_file
                    else None,
                    "name": execution["name"],
                    "execution": overrides,
                    "resolved_config": asdict(resolved),
                    "repetition": repetition,
                    "split": split,
                    "logical_circuits_upper_bound": logical_bound,
                    "logical_shots_upper_bound": logical_bound * resolved.shots,
                }
                runs.append(
                    {**spec, "run_id": identity(spec)[:20], "model_path": str(path)}
                )
    shots = sum(run["logical_shots_upper_bound"] for run in runs)
    if shots > limits["max_logical_shots"]:
        raise ValueError("Planned upper bound exceeds max_logical_shots")
    return {
        "schema_version": 1,
        "campaign_id": identity({"manifest": manifest, "runs": runs, "limits": limits})[
            :20
        ],
        "manifest": manifest,
        "limits": limits,
        "runs": runs,
        "logical_shots_upper_bound": shots,
        "provenance": environment,
    }


def load_plan(path):
    value = json.loads(Path(path).read_text())
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported plan schema")
    expected = identity(
        {
            "manifest": value["manifest"],
            "runs": value["runs"],
            "limits": value["limits"],
        }
    )[:20]
    if value["campaign_id"] != expected:
        raise ValueError("Campaign plan checksum mismatch")
    for run in value["runs"]:
        if "environment_id" in run and run["environment_id"] != environment_id(
            value["provenance"]
        ):
            raise ValueError("Plan environment checksum mismatch")
        spec = {
            key: item
            for key, item in run.items()
            if key not in {"run_id", "model_path"}
        }
        if identity(spec)[:20] != run["run_id"]:
            raise ValueError("Run checksum mismatch")
    return value


def completed_result(root, run):
    folder = Path(root) / "runs" / run["run_id"]
    completed = sorted(folder.glob("attempt-*/complete.json"))
    if len(completed) > 1:
        raise ValueError(f"Multiple completed attempts for {run['run_id']}")
    if not completed:
        return None
    marker = json.loads(completed[0].read_text())
    result_path = completed[0].parent / "result.json"
    if marker["result_sha256"] != file_checksum(result_path):
        raise ValueError(f"Result checksum mismatch: {result_path}")
    result = json.loads(result_path.read_text())
    if result["model_id"] != run["model_id"] or result["run_id"] != run["run_id"]:
        raise ValueError("Completed result identity mismatch")
    if (
        identity(result["resolved_config"]) != identity(run["resolved_config"])
        or result["split"] != run["split"]
    ):
        raise ValueError("Completed result configuration mismatch")
    if (
        "environment_id" in run
        and environment_id(result["provenance"]) != run["environment_id"]
    ):
        raise ValueError("Completed result environment mismatch")
    if "sample_indices" in run and result["sample_indices"] != run["sample_indices"]:
        raise ValueError("Completed result sample indices mismatch")
    return result


def execute(plan_path, *, recover_stale=False):
    plan_path = Path(plan_path).resolve()
    campaign = load_plan(plan_path)
    start = time.monotonic()
    attempts = list(plan_path.parent.glob("runs/*/attempt-*"))
    reserved_shots = sum(
        json.loads((attempt / "spec.json").read_text())["logical_shots_upper_bound"]
        for attempt in attempts
        if (attempt / "spec.json").exists()
    )
    spent_seconds = 0.0
    for attempt in attempts:
        duration = attempt / "duration.json"
        if duration.exists():
            spent_seconds += json.loads(duration.read_text())["elapsed_s"]
        elif (attempt / "started.json").exists():
            # ponytail: an unmeasured hard interruption reserves elapsed wall time, including downtime.
            started = json.loads((attempt / "started.json").read_text())["utc"]
            spent_seconds += max(
                0.0,
                (datetime.now(UTC) - datetime.fromisoformat(started)).total_seconds(),
            )
    summary = {"completed": 0, "skipped": 0, "failed": 0, "budget_stop": False}
    current_environment = None
    for run in campaign["runs"]:
        if file_checksum(run["model_path"]) != run["model_sha256"]:
            raise ValueError("Frozen model file changed since planning")
        if (
            run.get("profile_file")
            and file_checksum(run["profile_file"]) != run["profile_sha256"]
        ):
            raise ValueError("Frozen profile file changed since planning")
        if completed_result(plan_path.parent, run) is not None:
            summary["skipped"] += 1
            continue
        if current_environment is None:
            current_environment = provenance()
        keys = ("source_sha256", "versions", "python", "platform", "lockfile_sha256")
        if any(
            current_environment.get(key) != campaign["provenance"].get(key)
            for key in keys
        ):
            raise ValueError(
                "Code or environment changed since planning; create a new plan in a new output directory"
            )
        if (
            spent_seconds + time.monotonic() - start
            >= campaign["limits"]["max_seconds"]
            or reserved_shots + run["logical_shots_upper_bound"]
            > campaign["limits"]["max_logical_shots"]
        ):
            summary["budget_stop"] = True
            break
        folder = plan_path.parent / "runs" / run["run_id"]
        folder.mkdir(parents=True, exist_ok=True)
        claim = folder / "claim.json"
        if claim.exists() and recover_stale:
            owner = json.loads(claim.read_text())
            if owner["host"] != socket.gethostname():
                raise ValueError("Cannot recover a claim from another host")
            try:
                os.kill(owner["pid"], 0)
            except ProcessLookupError:
                claim.rename(folder / f"recovered-claim-{time.time_ns()}.json")
            else:
                raise ValueError("Claim owner is still running")
        write_json(claim, {"host": socket.gethostname(), "pid": os.getpid()})
        try:
            attempts = list(folder.glob("attempt-*"))
            attempt = folder / f"attempt-{len(attempts) + 1:04d}"
            attempt.mkdir()
            write_json(attempt / "spec.json", run)
            write_json(
                attempt / "started.json",
                {"utc": datetime.now(UTC).isoformat(), "provenance": provenance()},
            )
            attempt_start = time.monotonic()
            reserved_shots += run["logical_shots_upper_bound"]
            try:
                result = evaluate(
                    Model.load(run["model_path"]),
                    execution=run["execution"],
                    split=run["split"],
                    records_output=attempt / "backend.json",
                    max_samples=run.get("max_samples"),
                )
                result.update(
                    campaign_id=campaign["campaign_id"],
                    run_id=run["run_id"],
                    attempt_id=attempt.name,
                    name=run["name"],
                    repetition=run["repetition"],
                )
                write_json(attempt / "result.json", result)
                write_json(
                    attempt / "complete.json",
                    {
                        "result_sha256": file_checksum(attempt / "result.json"),
                        "utc": datetime.now(UTC).isoformat(),
                    },
                )
                summary["completed"] += 1
            except (Exception, KeyboardInterrupt) as error:
                write_json(
                    attempt / "failure.json",
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "utc": datetime.now(UTC).isoformat(),
                    },
                )
                summary["failed"] += 1
                raise
            finally:
                write_json(
                    attempt / "duration.json",
                    {"elapsed_s": time.monotonic() - attempt_start},
                )
        finally:
            claim.unlink(missing_ok=True)
    return summary


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("--manifest", type=Path, required=True)
    planning.add_argument("--output", type=Path, required=True)
    running = sub.add_parser("run")
    running.add_argument("--plan", type=Path, required=True)
    running.add_argument("--recover-stale", action="store_true")
    checking = sub.add_parser("verify")
    checking.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.operation == "plan":
        value = plan(args.manifest)
        write_json(args.output, value)
        print(
            f"Planned {len(value['runs'])} runs; logical shot upper bound {value['logical_shots_upper_bound']}"
        )
    elif args.operation == "run":
        print(json.dumps(execute(args.plan, recover_stale=args.recover_stale)))
    else:
        campaign = load_plan(args.plan)
        done = sum(
            completed_result(args.plan.parent, run) is not None
            for run in campaign["runs"]
        )
        print(f"Verified {done}/{len(campaign['runs'])} completed runs")
        if done != len(campaign["runs"]):
            raise SystemExit(1)
