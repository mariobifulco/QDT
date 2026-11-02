"""Local profile simulation using the same packed execution loop as Aer."""

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

from ..partitions import compile_packed, select_layout
from ..qpu import load_profile, noise_from_dict
from .qiskit_run import Adapter as AerAdapter
from .qiskit_run import aer_backend


class Adapter(AerAdapter):
    def __init__(self, config):
        self._initialize(config)
        self.profile = load_profile(config.fake_qpu)
        self.noise_data = self.profile.noise.to_dict()
        self.device_name = f"local:{self.profile.metadata['source']}"
        self.backends = {}
        self.readout_matrices = {}
        self.selected_layout = config.qpu_layout
        if config.packing_factor > 1 and self.selected_layout is None:
            self.selected_layout = select_layout(
                self.profile.target, self.circuit.num_qubits, config.packing_factor
            )
        if config.aer_method == "statevector" and not config.shots:
            raise ValueError(
                "Exact hardware-profile simulation requires density_matrix"
            )
        self.template = self._template(config.packing_factor)[0]
        self.backend = self.backends[config.packing_factor]

    def _template(self, copies):
        if copies in self.templates:
            return self.templates[copies]
        start = time.perf_counter()
        config, profile = self.config, self.profile
        layout = (
            list(self.selected_layout[: copies * self.circuit.num_qubits])
            if self.selected_layout
            else None
        )
        compiled, parameters = compile_packed(
            config, copies, profile.target, layout, profile.iqm_backend
        )
        physical = sorted(
            {compiled.find_bit(q).index for inst in compiled.data for q in inst.qubits}
        )
        mapping = {q: i for i, q in enumerate(physical)}
        template = QuantumCircuit(len(physical), copies)
        template.global_phase = compiled.global_phase
        measured = {}
        for inst in compiled.data:
            wires = [mapping[compiled.find_bit(q).index] for q in inst.qubits]
            if inst.operation.name == "measure":
                bit = compiled.find_bit(inst.clbits[0]).index
                measured[bit] = wires[0]
                if config.shots:
                    template.measure(wires[0], bit)
            elif inst.operation.name == "move":
                from iqm.qiskit_iqm.move_gate import MOVE_GATE_UNITARY

                template.unitary(MOVE_GATE_UNITARY, wires, label="move")
            else:
                template.append(inst.operation, wires)
        if set(measured) != set(range(copies)):
            raise RuntimeError("Compiled device circuit lost readout measurements")
        matrices = []
        for bit in range(copies):
            physical_readout = physical[measured[bit]]
            default, local = np.eye(2), None
            for error in self.noise_data["errors"]:
                if error["type"] != "roerror":
                    continue
                loci = error.get("gate_qubits")
                if loci is None:
                    default = np.asarray(error["probabilities"])
                elif any(physical_readout in locus for locus in loci):
                    if any(len(locus) != 1 for locus in loci):
                        raise ValueError(
                            "Correlated readout needs joint measurement support"
                        )
                    local = np.asarray(error["probabilities"])
            matrices.append(default if local is None else local)
            if not config.shots:
                template.save_expectation_value(
                    Pauli("Z"), [measured[bit]], label=f"z{bit}"
                )
        self.readout_matrices[copies] = matrices
        self.readout = matrices[0]  # Existing single-copy inspection API.
        method = (
            "density_matrix" if config.aer_method == "automatic" else config.aer_method
        )
        self.backends[copies] = aer_backend(
            config, method, noise_from_dict(self.noise_data, physical)
        )
        metadata = dict(
            profile.metadata,
            physical_qubits=physical,
            physical_readout=physical[measured[0]],
            physical_readouts=[physical[measured[i]] for i in range(copies)],
            simulation=f"local Aer {method}",
            native_gate_counts=dict(compiled.count_ops()),
            compiled_depth=compiled.depth(),
            requested_layout=layout,
            partition_selection="explicit"
            if config.qpu_layout
            else ("greedy" if self.selected_layout else "transpiler"),
        )
        self.template_metadata[copies] = metadata
        if copies == config.packing_factor:
            self.profile_metadata = metadata
        self.templates[copies] = template, parameters
        self.template_s += time.perf_counter() - start
        return self.templates[copies]

    def _get_backend(self, copies):
        return self.backends[copies]

    def _correct_readout(self, p, copies):
        return np.array(
            [
                value * matrix[0, 0] + (1 - value) * matrix[1, 0]
                for value, matrix in zip(p, self.readout_matrices[copies], strict=True)
            ]
        )
