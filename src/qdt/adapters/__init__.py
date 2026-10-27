from importlib import import_module

import numpy as np


def get_adapter(config):
    module = {"aer": "qiskit_run", "iqm": "iqm_run", "cudaq": "cudaq_run"}
    return import_module(f".{module[config.backend]}", __name__).Adapter(config)


def circuit_array(circuits, batch_size):
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be an integer >= 1")
    if not len(circuits):
        return np.empty((0, 3))
    angles = np.asarray(circuits, dtype=float)
    if angles.shape != (len(circuits), 3) or not np.isfinite(angles).all():
        raise ValueError("Expected finite (feature_angle, theta_z, theta_y) triples")
    return angles
