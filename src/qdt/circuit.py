import re
from dataclasses import dataclass
from functools import cached_property
from numbers import Real

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

ROTATIONS = {"rx", "ry", "rz"}
SINGLE_GATES = ROTATIONS | {"h", "x", "y", "z", "s", "t"}
TWO_GATES = {"cx", "cz", "swap"}
REFERENCE = re.compile(r"(x|theta)\[(0|[1-9][0-9]*)\]")


@dataclass(frozen=True)
class Circuit:
    num_qubits: int
    readout_qubit: int
    gates: tuple

    def __post_init__(self):
        if type(self.num_qubits) is not int or self.num_qubits < 1:
            raise ValueError("num_qubits must be a positive integer")
        if (
            type(self.readout_qubit) is not int
            or not 0 <= self.readout_qubit < self.num_qubits
        ):
            raise ValueError("readout_qubit is outside the circuit")
        references = {"x": set(), "theta": set()}
        if not isinstance(self.gates, (list, tuple)):
            raise ValueError("gates must be a list")
        for gate in self.gates:
            if not isinstance(gate, (list, tuple)) or not gate:
                raise ValueError("Each gate must be a nonempty list")
            name = gate[0]
            if not isinstance(name, str) or name not in SINGLE_GATES | TWO_GATES:
                raise ValueError(f"Unsupported gate: {name}")
            size = 3 if name in ROTATIONS | TWO_GATES else 2
            if len(gate) != size:
                raise ValueError(f"Invalid arguments for gate {name}")
            wires = gate[1:3] if name in TWO_GATES else gate[1:2]
            if any(
                type(q) is not int or not 0 <= q < self.num_qubits for q in wires
            ) or len(set(wires)) != len(wires):
                raise ValueError(f"Invalid qubits in {gate}")
            if name in ROTATIONS:
                angle = gate[2]
                if isinstance(angle, str) and (match := REFERENCE.fullmatch(angle)):
                    references[match[1]].add(int(match[2]))
                elif (
                    not isinstance(angle, Real)
                    or isinstance(angle, bool)
                    or not np.isfinite(angle)
                ):
                    raise ValueError("Angles must be finite numbers, x[i], or theta[i]")
        for name, indices in references.items():
            if not indices or indices != set(range(len(indices))):
                raise ValueError(f"{name} references must start at 0 with no gaps")
        object.__setattr__(self, "gates", tuple(tuple(gate) for gate in self.gates))

    @cached_property
    def num_features(self):
        return len(
            {
                g[2]
                for g in self.gates
                if g[0] in ROTATIONS and isinstance(g[2], str) and g[2].startswith("x[")
            }
        )

    @cached_property
    def num_parameters(self):
        return len(
            {
                g[2]
                for g in self.gates
                if g[0] in ROTATIONS
                and isinstance(g[2], str)
                and g[2].startswith("theta[")
            }
        )

    @property
    def num_inputs(self):
        return self.num_features + self.num_parameters

    @cached_property
    def input_indices(self):
        return {
            f"{name}[{i}]": offset + i
            for name, count, offset in (
                ("x", self.num_features, 0),
                ("theta", self.num_parameters, self.num_features),
            )
            for i in range(count)
        }

    @cached_property
    def parameters(self):
        return (
            *ParameterVector("x", self.num_features),
            *ParameterVector("theta", self.num_parameters),
        )

    @cached_property
    def operations(self):
        operations = []
        for gate in self.gates:
            if gate[0] == "swap":
                _, a, b = gate
                operations.extend((("cx", a, b), ("cx", b, a), ("cx", a, b)))
            else:
                operations.append(gate)
        return tuple(operations)

    @cached_property
    def template(self):
        qc = QuantumCircuit(self.num_qubits)
        for name, *args in self.operations:
            if name in ROTATIONS:
                wire, angle = args
                angle = (
                    self.parameters[self.input_indices[angle]]
                    if isinstance(angle, str)
                    else float(angle)
                )
                getattr(qc, name)(angle, wire)
            else:
                getattr(qc, name)(*args)
        return qc
