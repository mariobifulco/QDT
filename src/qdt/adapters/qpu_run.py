import numpy as np
from qiskit import QuantumCircuit, transpile

from ..qpu import load_profile, noise_from_dict
from . import circuit_array


class Adapter:
    def __init__(self, config):
        from qiskit_aer import AerSimulator

        self.config = config
        self.circuit = config.circuit
        self.rng = np.random.default_rng(config.seed)
        self.calls = self.circuits = 0
        profile = load_profile(config.fake_qpu)
        noise_data = profile.noise.to_dict()
        self.device_name = f"local:{profile.metadata['source']}"
        qc = self.circuit.template.copy()
        from qiskit import ClassicalRegister

        qc.add_register(ClassicalRegister(1))
        qc.measure(self.circuit.readout_qubit, 0)
        if config.qpu_layout and max(config.qpu_layout) >= profile.target.num_qubits:
            raise ValueError("qpu_layout exceeds the device qubit count")
        if qc.num_qubits > profile.target.num_qubits:
            raise ValueError("Circuit exceeds the device qubit count")
        options = {
            "initial_layout": list(config.qpu_layout) if config.qpu_layout else None,
            "seed_transpiler": 0,
            "optimization_level": 0,
        }
        if profile.iqm_backend is not None:
            from iqm.qiskit_iqm import transpile_to_IQM

            compiled = transpile_to_IQM(qc, backend=profile.iqm_backend, **options)
        else:
            compiled = transpile(qc, target=profile.target, **options)
        physical = sorted(
            {compiled.find_bit(q).index for inst in compiled.data for q in inst.qubits}
        )
        mapping = {q: i for i, q in enumerate(physical)}
        self.template = QuantumCircuit(len(physical), 1)
        self.template.global_phase = compiled.global_phase
        readout_qubit = None
        for inst in compiled.data:
            wires = [mapping[compiled.find_bit(q).index] for q in inst.qubits]
            if inst.operation.name == "measure":
                readout_qubit = wires[0]
                if config.shots:
                    self.template.measure(wires[0], 0)
            elif inst.operation.name == "move":
                from iqm.qiskit_iqm.move_gate import MOVE_GATE_UNITARY

                self.template.unitary(MOVE_GATE_UNITARY, wires, label="move")
            else:
                self.template.append(inst.operation, wires)
        if readout_qubit is None:
            raise RuntimeError("Compiled device circuit lost the readout measurement")
        self.readout = np.eye(2)
        physical_readout = physical[readout_qubit]
        local_readout = None
        for error in noise_data["errors"]:
            if error["type"] != "roerror":
                continue
            loci = error.get("gate_qubits")
            if loci is None:
                self.readout = np.asarray(error["probabilities"])
            elif any(physical_readout in locus for locus in loci):
                if any(len(locus) != 1 for locus in loci):
                    raise ValueError(
                        "Correlated readout needs joint measurement support"
                    )
                local_readout = np.asarray(error["probabilities"])
        if local_readout is not None:
            self.readout = local_readout
        if not config.shots:
            self.template.save_probabilities([readout_qubit], label="p")
        self.backend = AerSimulator(
            method="density_matrix",
            noise_model=noise_from_dict(noise_data, physical),
            max_parallel_threads=config.threads,
            fusion_enable=False,
            runtime_parameter_bind_enable=True,
        )
        self.profile_metadata = dict(
            profile.metadata,
            physical_qubits=physical,
            physical_readout=physical_readout,
            simulation="local Aer density_matrix",
            native_gate_counts=dict(compiled.count_ops()),
        )

    def probabilities(self, circuits, batch_size=1):
        angles = circuit_array(circuits, batch_size, self.circuit.num_inputs)
        if not len(angles):
            return np.empty(0)
        self.calls += 1
        self.circuits += len(angles)
        probabilities = []
        parameters = self.template.parameters
        for start in range(0, len(angles), batch_size):
            batch = angles[start : start + batch_size]
            bindings = {
                parameter: batch[:, column].tolist()
                for column, parameter in enumerate(self.circuit.parameters)
                if parameter in parameters
            }
            circuit = self.template
            options = {"parameter_binds": [bindings]} if bindings else {}
            if not bindings:
                circuit = [circuit] * len(batch)
            result = self.backend.run(
                circuit,
                **options,
                shots=self.config.shots or 1,
                seed_simulator=int(self.rng.integers(0, 2**31 - 1)),
            ).result()
            if not result.success:
                raise RuntimeError(f"Local QPU simulation failed: {result.status}")
            for index in range(len(batch)):
                if self.config.shots:
                    counts = result.get_counts(index)
                    total = sum(counts.values())
                    if not total:
                        raise RuntimeError(
                            "Local simulator returned no measurement counts"
                        )
                    p = counts.get("0", 0) / total
                else:
                    p = np.asarray(result.data(index)["p"]) @ self.readout[:, 0]
                probabilities.append(p)
        return np.clip(probabilities, 0, 1)
