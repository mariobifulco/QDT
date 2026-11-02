"""Portable, JSON-only frozen models; no pickle or hidden preprocessing state."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .config import Config
from .data import Preprocessing
from .qpu import _json_default
from .quantum_tree import QNode, predict

EXECUTION_FIELDS = {
    "max_memory_mb",
    "parallel_experiments",
    "parallel_shots",
    "packing_factor",
    "aer_device",
    "aer_method",
    "simulator_seed",
    "transpiler_seed",
    "backend",
    "batch_size",
    "shots",
    "depolarizing",
    "amplitude_damping",
    "phase_damping",
    "readout_error",
    "fake_qpu",
    "qpu_layout",
    "threads",
}


@dataclass
class Model:
    config: Config
    preprocessing: Preprocessing
    tree: QNode
    classes: list[int]
    data: dict

    def execution_config(self, overrides=None):
        overrides = overrides or {}
        if set(overrides) - EXECUTION_FIELDS:
            raise ValueError("Inference overrides may only change execution settings")
        return replace(self.config, **overrides)

    def predict(self, X, *, execution=None, adapter=None):
        return predict(
            self.preprocessing.transform(X),
            self.tree,
            self.execution_config(execution),
            adapter=adapter,
        )

    def payload(self):
        return {
            "schema_version": 1,
            "config": asdict(self.config),
            "preprocessing": asdict(self.preprocessing),
            "tree": asdict(self.tree),
            "classes": self.classes,
            "data": self.data,
        }

    @property
    def model_id(self):
        return hashlib.sha256(
            json.dumps(
                self.payload(), sort_keys=True, allow_nan=False, default=_json_default
            ).encode()
        ).hexdigest()

    def save(self, path):
        payload = self.payload()
        payload["model_id"] = self.model_id
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False, default=_json_default)
            handle.write("\n")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported model schema version")
        config = Config(**payload["config"])
        preprocessing = Preprocessing(**payload["preprocessing"])
        classes = payload["classes"]
        if (
            not classes
            or any(type(c) is not int for c in classes)
            or len(set(classes)) != len(classes)
        ):
            raise ValueError("Invalid model classes")
        width = len(preprocessing.feature_indices)
        if width < config.circuit.num_features:
            raise ValueError("Preprocessing has fewer features than the circuit")

        def node(value):
            result = QNode(**value)
            if result.value is not None:
                if result.value not in classes or any(
                    x is not None
                    for x in (
                        result.left,
                        result.right,
                        result.theta,
                        result.feature_indices,
                    )
                ):
                    raise ValueError("Invalid leaf")
            else:
                if (
                    not isinstance(result.feature_indices, list)
                    or len(result.feature_indices) != config.circuit.num_features
                    or any(
                        type(i) is not int or not 0 <= i < width
                        for i in result.feature_indices
                    )
                    or len(set(result.feature_indices)) != len(result.feature_indices)
                    or not isinstance(result.theta, list)
                    or len(result.theta) != config.circuit.num_parameters
                    or not np.isfinite(result.theta).all()
                    or not isinstance(result.left, dict)
                    or not isinstance(result.right, dict)
                ):
                    raise ValueError("Invalid branch dimensions or children")
                result.left, result.right = node(result.left), node(result.right)
            return result

        model = cls(
            config, preprocessing, node(payload["tree"]), classes, payload["data"]
        )
        if model.model_id != payload.get("model_id"):
            raise ValueError("Model checksum mismatch")
        return model
