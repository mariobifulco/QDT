import argparse
import importlib
import importlib.metadata
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.transpiler import InstructionProperties, Target
from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError

IQM_DEVICES = {"adonis": 5, "apollo": 20, "aphrodite": 54, "garnet": 20, "deneb": 6}


@dataclass
class Profile:
    target: Target
    noise: NoiseModel
    metadata: dict
    iqm_backend: object = None


def _ibm_profile(backend, metadata):
    from qiskit_aer.noise.device import basic_device_readout_errors

    target = backend.target
    noise = NoiseModel.from_backend(backend)
    for qubits, error in basic_device_readout_errors(backend.properties()):
        noise.add_readout_error(error, qubits, warnings=False)
    return Profile(target, noise, metadata)


def _load_credentials():
    from dotenv import dotenv_values

    for key, value in dotenv_values(".env").items():
        if (
            key in {"QISKIT_IBM_TOKEN", "QISKIT_IBM_INSTANCE", "QISKIT_IBM_CHANNEL"}
            and value
        ):
            os.environ.setdefault(key, value)


def _ibm_service():
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as error:
        raise ImportError("IBM downloads require: uv sync --extra ibm") from error
    _load_credentials()
    return QiskitRuntimeService()


def list_online_ibm():
    return [
        {"name": f"ibm:{backend.name}", "qubits": backend.num_qubits}
        for backend in sorted(
            _ibm_service().backends(operational=True, simulator=False),
            key=lambda backend: backend.name,
        )
    ]


def list_qpus():
    return [
        {"name": f"iqm:{name}", "qubits": size, "kind": "library profile"}
        for name, size in IQM_DEVICES.items()
    ]


def load_profile(selector):
    provider, separator, name = selector.partition(":")
    if not separator or not name:
        raise ValueError("Use iqm:name or file:path")
    if provider == "file":
        return read_profile(name)
    if provider == "ibm":
        raise ValueError(
            "IBM library snapshots are disabled; fetch an online QPU with qdt.qpu fetch, then use file:path"
        )
    if provider == "iqm":
        name = name.lower()
        if name not in IQM_DEVICES:
            raise ValueError(
                f"Unknown IQM fake QPU {name!r}; choose {', '.join(IQM_DEVICES)}"
            )
        try:
            module = importlib.import_module(
                f"iqm.qiskit_iqm.fake_backends.fake_{name}"
            )
        except ImportError as error:
            raise ImportError("IQM profiles require: uv sync --extra iqm") from error
        backend = getattr(module, f"IQMFake{name.title()}")()
        return Profile(
            backend.target,
            backend.noise_model,
            {
                "source": f"iqm:{name}",
                "kind": "device profile"
                if name in {"garnet", "deneb"}
                else "representative architecture",
                "library_version": importlib.metadata.version("iqm-client"),
                "profile_name": backend.error_profile.name,
            },
            backend,
        )
    raise ValueError(
        f"Unsupported local profile {selector!r}; use --list-qpus or file:path"
    )


def noise_from_dict(data, physical_qubits=None):
    mapping = (
        None
        if physical_qubits is None
        else {q: i for i, q in enumerate(physical_qubits)}
    )
    noise = NoiseModel()
    for record in data["errors"]:
        loci = record.get("gate_qubits")
        if mapping is not None and loci is not None:
            loci = [
                tuple(mapping[q] for q in locus)
                for locus in loci
                if all(q in mapping for q in locus)
            ]
            if not loci:
                continue
        if record["type"] == "qerror":
            error = QuantumError.from_dict(record)
            if loci is None:
                noise.add_all_qubit_quantum_error(error, record["operations"])
            else:
                for locus in loci:
                    noise.add_quantum_error(error, record["operations"], locus)
        elif record["type"] == "roerror":
            error = ReadoutError(record["probabilities"])
            if loci is None:
                noise.add_all_qubit_readout_error(error)
            else:
                for locus in loci:
                    noise.add_readout_error(error, locus)
        else:
            raise ValueError(f"Unsupported noise record: {record['type']}")
    return noise


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _json_hook(value):
    return complex(*value["__complex__"]) if set(value) == {"__complex__"} else value


def write_profile(profile, path):
    target = profile.target
    instructions = []
    for name in target.operation_names:
        if name in {
            "if_else",
            "while_loop",
            "for_loop",
            "switch_case",
            "break_loop",
            "continue_loop",
            "store",
        }:
            continue
        for locus, properties in target[name].items():
            instructions.append(
                {
                    "name": name,
                    "qubits": locus,
                    "duration": properties.duration if properties else None,
                    "error": properties.error if properties else None,
                }
            )
    payload = {
        "format": "qdt-qpu-v1",
        "metadata": profile.metadata,
        "num_qubits": target.num_qubits,
        "instructions": instructions,
        "noise_model": profile.noise.to_dict(),
        "iqm_compiler": profile.metadata["source"] if profile.iqm_backend else None,
    }
    with Path(path).open("x") as handle:
        json.dump(payload, handle, default=_json_default, indent=2, allow_nan=False)
        handle.write("\n")


def read_profile(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw, object_hook=_json_hook)
    if data.get("format") != "qdt-qpu-v1":
        raise ValueError("Expected a qdt-qpu-v1 snapshot")
    metadata = data["metadata"]
    if not metadata.get("source", "").startswith(("ibm:", "iqm:")):
        raise ValueError("Only IBM and IQM profiles are supported")
    if (
        metadata.get("source", "").startswith("ibm:")
        and metadata.get("kind") != "downloaded calibration"
    ):
        raise ValueError(
            "IBM library snapshots are disabled; fetch a profile from an online QPU"
        )
    if data.get("iqm_compiler"):
        profile = load_profile(data["iqm_compiler"])
    else:
        target = Target(num_qubits=data["num_qubits"])
        gates = get_standard_gate_name_mapping()
        grouped = {}
        for entry in data["instructions"]:
            name = entry["name"]
            if name not in gates:
                raise ValueError(f"Snapshot contains unsupported native gate {name!r}")
            locus = None if entry["qubits"] is None else tuple(entry["qubits"])
            grouped.setdefault(name, {})[locus] = InstructionProperties(
                duration=entry["duration"], error=entry["error"]
            )
        for name, loci in grouped.items():
            target.add_instruction(gates[name], loci)
        profile = Profile(target, None, {})
    profile.noise = noise_from_dict(data["noise_model"])
    import hashlib

    profile.metadata = dict(
        data["metadata"], snapshot_sha256=hashlib.sha256(raw).hexdigest()
    )
    return profile


def fetch_profile(source):
    if source.startswith("ibm:"):
        backend = _ibm_service().backend(source.removeprefix("ibm:"))
        if backend.configuration().simulator or not backend.status().operational:
            raise ValueError("Only operational IBM QPUs can supply noise profiles")
        return _ibm_profile(
            backend,
            {
                "source": source,
                "kind": "downloaded calibration",
                "retrieved_at": datetime.now(UTC).isoformat(),
                "calibration_time": str(backend.properties().last_update_date),
                "library_version": importlib.metadata.version("qiskit-ibm-runtime"),
            },
        )
    raise ValueError("Calibration downloads support only ibm:REAL_BACKEND")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser(
        "list", help="List local profiles, or online IBM QPUs"
    )
    listing.add_argument(
        "--online",
        action="store_true",
        help="Read operational IBM QPUs accessible to your account",
    )
    for command in ("export", "fetch"):
        sub = commands.add_parser(command)
        sub.add_argument("source")
        sub.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(list_online_ibm() if args.online else list_qpus(), indent=2))
        return
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    if args.command == "fetch":
        profile = fetch_profile(args.source)
    else:
        profile = load_profile(args.source)
    write_profile(profile, args.output)
    print(f"Saved local profile: {args.output}")


if __name__ == "__main__":
    main()
