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
