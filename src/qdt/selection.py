"""Validation-selected fixed execution settings: a baseline, not an adaptive policy."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .campaign import completed_result, file_checksum, identity, load_plan
from .experiment import provenance
from .model import Model, write_json


def select(plan_path, *, reference, delta=0.01):
    start = time.perf_counter()
    if not np.isfinite(delta) or not 0 <= delta <= 1:
        raise ValueError("delta must be a finite absolute quality tolerance in [0, 1]")
    plan_path = Path(plan_path)
    campaign = load_plan(plan_path)
    if any(run["split"] != "validation" for run in campaign["runs"]):
        raise ValueError("Settings may only be selected on validation data")
    groups = defaultdict(lambda: defaultdict(list))
    paths = {}
    for run in campaign["runs"]:
        result = completed_result(plan_path.parent, run)
        if result is None:
            raise ValueError("Selection requires a complete validation campaign")
        groups[run["model_id"]][run["name"]].append((run, result))
        paths[run["model_id"]] = run["model_path"]
    decisions = []
    for model_id, conditions in sorted(groups.items()):
        if reference not in conditions:
            raise ValueError("Reference is missing for a model")
        reference_quality = float(
            np.mean(
                [result["balanced_accuracy"] for _, result in conditions[reference]]
            )
        )
        eligible = []
        for name, values in conditions.items():
            quality = float(
                np.mean([result["balanced_accuracy"] for _, result in values])
            )
            duration = float(np.median([result["total_s"] for _, result in values]))
            if quality >= reference_quality - delta:
                eligible.append((duration, name, quality))
        if not eligible:
            raise ValueError("No execution satisfies the validation constraint")
        duration, name, quality = min(eligible)
        run = min(conditions[name], key=lambda pair: pair[0]["repetition"])[0]
        model = Model.load(paths[model_id])
        if model.model_id != model_id:
            raise ValueError("Frozen model changed since collection")
        model.execution_config(run["execution"])
        decisions.append(
            {
                "model_id": model_id,
                "name": name,
                "execution": run["execution"],
                "validation_quality": quality,
                "reference_quality": reference_quality,
                "median_validation_total_s": duration,
            }
        )
    durations = []
    for path in plan_path.parent.glob("runs/*/attempt-*/started.json"):
        duration_path = path.parent / "duration.json"
        if not duration_path.exists():
            durations = None
            break
        durations.append(json.loads(duration_path.read_text())["elapsed_s"])
    result = {
        "schema_version": 1,
        "kind": "validation_selected_fixed_setting",
        "selection_split": "validation",
        "metric": "balanced_accuracy",
        "delta": delta,
        "reference": reference,
        "source_campaign_id": campaign["campaign_id"],
        "source_plan_sha256": file_checksum(plan_path),
        "source_provenance": campaign["provenance"],
        "selection_provenance": provenance(),
        "decisions": decisions,
        "selection_attempt_wall_s": float(sum(durations))
        if durations is not None
        else None,
        "selection_bookkeeping_s": time.perf_counter() - start,
        "limitation": "Feasibility uses observed validation means, not a statistical "
        "non-inferiority guarantee. Selection cost excludes model training "
        "and planning. Deployment test quality has not been evaluated.",
    }
    result["policy_id"] = identity(result)
    return result


def policy_execution(path, model):
    value = json.loads(Path(path).read_text())
    expected = identity(
        {key: item for key, item in value.items() if key != "policy_id"}
    )
    if (
        value.get("schema_version") != 1
        or value.get("policy_id") != expected
        or value.get("kind") != "validation_selected_fixed_setting"
        or value.get("selection_split") != "validation"
    ):
        raise ValueError("Invalid policy schema, checksum or selection split")
    decisions = [
        item for item in value["decisions"] if item["model_id"] == model.model_id
    ]
    if len(decisions) != 1:
        raise ValueError("Policy does not uniquely identify this frozen model")
    execution = decisions[0]["execution"]
    model.execution_config(execution)
    return execution, value["policy_id"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = select(args.plan, reference=args.reference, delta=args.delta)
    write_json(args.output, result)
    print(f"Saved {len(result['decisions'])} fixed model policies to {args.output}")
