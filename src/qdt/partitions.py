"""A deterministic greedy baseline, not an optimal partitioning solver."""


def select_layout(target, width, copies):
    if width * copies > target.num_qubits:
        raise ValueError("Packed circuits exceed the device qubit count")
    nodes = set(range(target.num_qubits))
    graph = {q: set() for q in nodes}
    coupling = target.build_coupling_map()
    if coupling is None:
        for q in nodes:
            graph[q] = nodes - {q}
    else:
        for a, b in coupling.get_edges():
            graph[a].add(b)
            graph[b].add(a)
    # Missing calibration is penalized rather than treated as perfect fidelity.
    readout = dict.fromkeys(nodes, 1.0)
    if "measure" in target.operation_names:
        for locus, properties in target["measure"].items():
            if (
                locus is not None
                and properties is not None
                and properties.error is not None
            ):
                readout[locus[0]] = properties.error
    errors = {}
    for name in target.operation_names:
        if target.operation_from_name(name).num_qubits != 2:
            continue
        for locus, properties in target[name].items():
            if (
                locus is not None
                and properties is not None
                and properties.error is not None
            ):
                edge = frozenset(locus)
                errors[edge] = min(errors.get(edge, 1.0), properties.error)

    def edge_error(a, b):
        return errors.get(frozenset((a, b)), 1.0)

    def score(group):
        edges = [edge_error(a, b) for a in group for b in graph[a] & group if a < b]
        return sum(readout[q] for q in group) / len(group) + (
            sum(edges) / len(edges) if edges else 0
        )

    available = nodes.copy()
    layout = []
    for _ in range(copies):
        candidates = set()
        for root in sorted(available):
            group = {root}
            while len(group) < width:
                frontier = set().union(*(graph[q] for q in group)) & (available - group)
                if not frontier:
                    break
                choice = min(
                    frontier,
                    key=lambda q: (
                        readout[q] + min(edge_error(q, v) for v in graph[q] & group),
                        q,
                    ),
                )
                group.add(choice)
            if len(group) == width:
                candidates.add(tuple(sorted(group)))
        if not candidates:
            raise ValueError(
                "Greedy partition selection cannot place all copies; try an explicit qpu_layout"
            )
        chosen = min(candidates, key=lambda group: (score(set(group)), group))
        layout.extend(chosen)
        available.difference_update(chosen)
    return tuple(layout)


def compile_packed(config, copies, target, layout, iqm_backend=None):
    from qiskit import ClassicalRegister, transpile

    from .packing import compose_copies

    qc, parameters, readouts = compose_copies(config.circuit, copies)
    qc.add_register(ClassicalRegister(copies, "readout"))
    qc.measure(readouts, range(copies))
    layout = list(layout[: qc.num_qubits]) if layout else None
    if layout and max(layout) >= target.num_qubits:
        raise ValueError("qpu_layout exceeds the device qubit count")
    if qc.num_qubits > target.num_qubits:
        raise ValueError("Circuit exceeds the device qubit count")
    options = {
        "initial_layout": layout,
        "seed_transpiler": config.transpiler_seed,
        "optimization_level": 0,
    }
    if iqm_backend is not None:
        from iqm.qiskit_iqm import transpile_to_IQM

        compiled = transpile_to_IQM(qc, backend=iqm_backend, **options)
    else:
        compiled = transpile(qc, target=target, **options)
    if copies > 1:
        owners = {q: i // config.circuit.num_qubits for i, q in enumerate(layout)}
        for inst in compiled.data:
            wires = [compiled.find_bit(q).index for q in inst.qubits]
            if (
                any(q not in owners for q in wires)
                or len({owners[q] for q in wires}) > 1
            ):
                raise ValueError(
                    "Routing leaves the declared disjoint partitions or shares resources"
                )
            if (
                inst.operation.name == "measure"
                and owners[wires[0]] != compiled.find_bit(inst.clbits[0]).index
            ):
                raise ValueError("Compiled readout moved to another partition")
    return compiled, parameters
