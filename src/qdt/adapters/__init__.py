CircuitSpec = tuple[float, float, float]


def get_adapter(config):
    if config.backend == "iqm":
        from .iqm_run import Adapter
    elif config.backend == "cudaq":
        from .cudaq_run import Adapter
    elif config.backend == "quantinuum":
        from .quantinuum_run import Adapter
    else:
        from .qiskit_run import Adapter
    return Adapter(config)
