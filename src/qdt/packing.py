"""Disjoint Qiskit copies and explicit little-endian measurement decoding."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def compose_copies(circuit, copies):
    qc = QuantumCircuit(copies * circuit.num_qubits)
    parameters = []
    for copy in range(copies):
        inputs = (
            circuit.parameters
            if copies == 1
            else tuple(ParameterVector(f"copy_{copy}", circuit.num_inputs))
        )
        parameters.extend(inputs)
        block = circuit.template.assign_parameters(
            dict(zip(circuit.parameters, inputs, strict=True))
        )
        qc.compose(
            block,
            qubits=range(copy * circuit.num_qubits, (copy + 1) * circuit.num_qubits),
            inplace=True,
        )
    readouts = [i * circuit.num_qubits + circuit.readout_qubit for i in range(copies)]
    return qc, parameters, readouts


def packed_batches(angles, copies, batch_size):
    for start in range(0, len(angles), copies * batch_size):
        chunk = angles[start : start + copies * batch_size]
        full = len(chunk) // copies * copies
        if full:
            yield start, chunk[:full].reshape(-1, copies, angles.shape[1])
        if full < len(chunk):
            yield (
                start + full,
                chunk[full:].reshape(1, len(chunk) - full, angles.shape[1]),
            )


def marginal_zeros(counts, copies):
    total = sum(counts.values())
    if total <= 0:
        raise RuntimeError("Backend returned no measurement counts")
    probabilities = np.zeros(copies)
    for bits, count in counts.items():
        text = bits.replace(" ", "")
        outcome = int(text, 16 if text.startswith("0x") else 2)
        if outcome >= 2**copies or count < 0:
            raise RuntimeError("Invalid measurement counts")
        for bit in range(copies):
            if not (outcome >> bit) & 1:
                probabilities[bit] += count / total
    return probabilities
