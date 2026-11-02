"""Budgeted IBM SamplerV2 inference with durable, nonduplicating job recovery."""

import base64
import hashlib
import io
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from qiskit import qpy

from ..model import write_json
from ..packing import marginal_zeros, packed_batches
from ..partitions import compile_packed, select_layout
from ..qpu import _ibm_service
from . import circuit_array
from .qiskit_run import Adapter as AerAdapter


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def native_json(value):
    from qiskit_ibm_runtime import RuntimeEncoder

    return json.loads(json.dumps(value, cls=RuntimeEncoder))


class Adapter(AerAdapter):
    def __init__(self, config):
        from qiskit_ibm_runtime import SamplerV2

        self._initialize(config)
        self.new_jobs = self.recovered_jobs = self.new_shots = 0
        self.device_name = f"ibm:{config.ibm_backend}"
        self.service = _ibm_service()
        self.backend = self.service.backend(config.ibm_backend)
        if self.backend.configuration().simulator:
            raise ValueError("Select an IBM hardware backend")
        self.journal = Path(config.job_journal)
        self.journal.mkdir(parents=True, exist_ok=True)
        requested = {
            key: getattr(config, key)
            for key in (
                "ibm_backend",
                "packing_factor",
                "qpu_layout",
                "shots",
                "batch_size",
                "transpiler_seed",
                "max_qpu_jobs",
                "max_qpu_shots",
                "qpu_max_execution_time_s",
            )
        }
        requested["circuit"] = asdict(config.circuit)
        path = self.journal / "context.json"
        if path.exists():
            context = json.loads(path.read_text())
            if digest(context["requested"]) != digest(requested):
                raise ValueError("Journal belongs to another execution configuration")
        else:
            properties = self.backend.properties()
            layout = config.qpu_layout or select_layout(
                self.backend.target, self.circuit.num_qubits, config.packing_factor
            )
            context = {
                "schema_version": 1,
                "requested": requested,
                "layout": list(layout),
                "calibration_properties": native_json(properties.to_dict())
                if properties is not None
                else None,
            }
            write_json(path, context)
        self.context = context
        self.selected_layout = context["layout"]
        self.sampler = SamplerV2(
            mode=self.backend,
            options={
                "max_execution_time": config.qpu_max_execution_time_s,
                "twirling": {"enable_gates": False, "enable_measure": False},
                "dynamical_decoupling": {"enable": False},
            },
        )
        self.template = self._template(config.packing_factor)[0]

    @property
    def statistics(self):
        return {
            **super().statistics,
            "qpu_submitted_jobs": self.new_jobs,
            "qpu_recovered_jobs": self.recovered_jobs,
            "qpu_submitted_shots": self.new_shots,
        }

    def _template(self, copies):
        if copies not in self.templates:
            start = time.perf_counter()
            path = self.journal / f"template-{copies}.json"
            if path.exists():
                stored = json.loads(path.read_text())
                if stored["template_id"] != digest(
                    {
                        key: value
                        for key, value in stored.items()
                        if key != "template_id"
                    }
                ):
                    raise ValueError("Journal template checksum mismatch")
                raw = base64.b64decode(stored["qpy"])
                if hashlib.sha256(raw).hexdigest() != stored["sha256"]:
                    raise ValueError("Journal template checksum mismatch")
                circuit = qpy.load(io.BytesIO(raw))[0]
                names = stored["parameter_names"]
            else:
                circuit, parameters = compile_packed(
                    self.config, copies, self.backend.target, self.selected_layout
                )
                names = [p.name for p in parameters]
                stream = io.BytesIO()
                qpy.dump(circuit, stream)
                raw = stream.getvalue()
                stored = {
                    "qpy": base64.b64encode(raw).decode(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "parameter_names": names,
                }
                stored["template_id"] = digest(stored)
                write_json(path, stored)
            measured = {
                circuit.find_bit(inst.clbits[0]).index: circuit.find_bit(
                    inst.qubits[0]
                ).index
                for inst in circuit.data
                if inst.operation.name == "measure"
            }
            if set(measured) != set(range(copies)):
                raise ValueError("Compiled IBM template lost readout measurements")
            self.template_metadata[copies] = {
                "template_id": stored["template_id"],
                "physical_readouts": [measured[i] for i in range(copies)],
                "native_gate_counts": dict(circuit.count_ops()),
                "compiled_depth": circuit.depth(),
            }
            self.templates[copies] = circuit, names
            self.template_s += time.perf_counter() - start
        return self.templates[copies]

    def _request(self, batch):
        groups, copies, width = batch.shape
        circuit, names = self._template(copies)
        values = batch.reshape(groups, copies * width)
        request = {
            "context": digest(self.context),
            "template_id": self.template_metadata[copies]["template_id"],
            "inputs": batch.tolist(),
            "copies": copies,
            "shots": self.config.shots,
        }
        folder = self.journal / f"request-{self.requests:06d}"
        folder.mkdir(exist_ok=True)
        intent, accepted, cached = (
            folder / "intent.json",
            folder / "accepted.json",
            folder / "result.json",
        )
        submitted_now = False
        if intent.exists():
            if digest(json.loads(intent.read_text())) != digest(request):
                raise ValueError(
                    "Journal request differs from this workload; use a new journal"
                )
        else:
            prior = [
                json.loads(path.read_text())
                for path in self.journal.glob("request-*/intent.json")
            ]
            if (
                len(prior) >= self.config.max_qpu_jobs
                or sum(len(item["inputs"]) * item["shots"] for item in prior)
                + groups * self.config.shots
                > self.config.max_qpu_shots
            ):
                raise ValueError("QPU journal job/shot budget exhausted")
            if not self.backend.status().operational:
                raise ValueError("IBM hardware backend is not operational")
            write_json(
                intent, request
            )  # Reserve before submission; ambiguous outcomes are never retried.
            if circuit.parameters:
                columns = {name: i for i, name in enumerate(names)}
                ordered = np.column_stack(
                    [values[:, columns[p.name]] for p in circuit.parameters]
                )
                pubs = [(circuit, ordered)]
            else:
                pubs = [circuit] * groups
            job = self.sampler.run(pubs, shots=self.config.shots)
            write_json(accepted, {"job_id": job.job_id()})
            self.new_jobs += 1
            self.new_shots += groups * self.config.shots
            submitted_now = True
        if not accepted.exists():
            raise RuntimeError(
                "Submission outcome is ambiguous; locate the existing job before any retry"
            )
        job_id = json.loads(accepted.read_text())["job_id"]
        if cached.exists():
            value = json.loads(cached.read_text())
            if value["checksum"] != digest(value["data"]):
                raise ValueError("Cached IBM result checksum mismatch")
            self.recovered_jobs += 1
            return value["data"], job_id
        job = self.service.job(job_id)
        self.recovered_jobs += int(not submitted_now)
        result = job.result(timeout=self.config.qpu_timeout_s)
        counts = (
            [result[0].data.readout.get_counts(i) for i in range(groups)]
            if circuit.parameters
            else [pub.data.readout.get_counts() for pub in result]
        )
        try:
            usage = native_json(job.metrics())
        except Exception as error:
            usage = {"unavailable": type(error).__name__}
        data = {
            "counts": counts,
            "metadata": native_json(result.metadata),
            "pub_metadata": [native_json(pub.metadata) for pub in result],
            "job_metrics": usage,
        }
        write_json(cached, {"data": data, "checksum": digest(data)})
        return data, job_id

    def probabilities(self, circuits, batch_size=1):
        angles = circuit_array(circuits, batch_size, self.circuit.num_inputs)
        if not len(angles):
            return np.empty(0)
        self.calls += 1
        self.circuits += len(angles)
        probabilities = np.empty(len(angles))
        for start, batch in packed_batches(
            angles, self.config.packing_factor, batch_size
        ):
            groups, copies, _ = batch.shape
            self._template(copies)
            self.requests += 1
            self.composite_circuits += groups
            self.physical_shots += groups * self.config.shots
            self.logical_shots += groups * copies * self.config.shots
            begin = time.perf_counter()
            record = {
                "request": self.requests,
                "probability_call": self.calls,
                "logical_start": start,
                "active_packing": copies,
                "requested_packing": self.config.packing_factor,
                "shots": self.config.shots,
                "compiled_template": self.template_metadata[copies],
                "layout": self.selected_layout[: copies * self.circuit.num_qubits],
            }
            self.records.append(record)
            data, job_id = self._request(batch)
            record.update(
                data, job_id=job_id, submit_wait_s=time.perf_counter() - begin
            )
            if len(data["counts"]) != groups:
                raise RuntimeError("Unexpected IBM result size")
            for index, counts in enumerate(data["counts"]):
                if sum(counts.values()) != self.config.shots:
                    raise RuntimeError("Unexpected IBM shot count")
                offset = start + index * copies
                probabilities[offset : offset + copies] = marginal_zeros(counts, copies)
        return probabilities
