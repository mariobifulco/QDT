from importlib import import_module

import numpy as np


def get_adapter(config):
    module = {"aer": "qiskit_run", "iqm": "iqm_run", "cudaq": "cudaq_run"}
    return import_module(f".{module[config.backend]}", __name__).Adapter(config)


def circuit_array(circuits, batch_size, num_inputs=3):
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be an integer >= 1")
    if not len(circuits):
        return np.empty((0, num_inputs))
    angles = np.asarray(circuits, dtype=float)
    if angles.shape != (len(circuits), num_inputs) or not np.isfinite(angles).all():
        raise ValueError(
            f"Expected finite input rows of width {num_inputs}: data angles, then variational angles"
        )
    return angles
