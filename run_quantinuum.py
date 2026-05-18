import numpy as np
from pytket.circuit import OpType
from pytket import Circuit
from pytket.extensions.qiskit import AerBackend
from pytket.extensions.qiskit import qiskit_to_tk

SHOTS = 100
MAX_STITCH = 40


def stitch_and_run(circuits: list) -> np.ndarray:
    n_total = len(circuits)
    n_qubits = MAX_STITCH
    stitched = Circuit(n_qubits, n_total)

    for i, qiskit_circ in enumerate(circuits):
        target_qubit = i % n_qubits
        target_bit = i
        circ = qiskit_to_tk(qiskit_circ)
        for command in circ.get_commands():
            if command.op.type != OpType.Measure:
                stitched.add_gate(command.op.type, command.op.params, [
                                  stitched.qubits[target_qubit]])

        stitched.Measure(
            stitched.qubits[target_qubit], stitched.bits[target_bit])

        if (i + n_qubits) < n_total:
            stitched.add_gate(OpType.Reset, [stitched.qubits[target_qubit]])

    backend = AerBackend()
    backend._qiskit_backend.set_options(method="matrix_product_state")
    compiled = backend.get_compiled_circuit(stitched)
    result = backend.run_circuit(compiled, n_shots=SHOTS)
    counts = result.get_counts()
    bit0_counts = [0] * n_total
    for bitstring, count in counts.items():
        for i in range(n_total):
            if bitstring[i] == 0:
                bit0_counts[i] += count

    return np.array([c >= SHOTS / 2 for c in bit0_counts])
