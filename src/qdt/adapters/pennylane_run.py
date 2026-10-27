import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class Adapter:
    def __init__(self, config):
        self.config = config
        self.calls = self.circuits = 0
        noisy = any(
            (
                config.depolarizing,
                config.amplitude_damping,
                config.phase_damping,
                config.readout_error,
            )
        )
        if config.backend == "lightning":
            self.device_name, self.diff_method = "lightning.qubit", "adjoint"
        else:
            self.device_name = "default.mixed" if noisy else "default.qubit"
            self.diff_method = "backprop"
        if config.shots:
            self.diff_method = "parameter-shift"
        device = qml.device(self.device_name, wires=1, seed=config.seed)

        @qml.qnode(device, interface="autograd", diff_method=self.diff_method)
        def circuit(angles, theta):
            for gate, angle in (
                (qml.RY, angles),
                (qml.RZ, theta[..., 0]),
                (qml.RY, theta[..., 1]),
            ):
                gate(angle, wires=0)
                if config.depolarizing:
                    qml.DepolarizingChannel(3 * config.depolarizing / 4, wires=0)
                if config.amplitude_damping:
                    qml.AmplitudeDamping(config.amplitude_damping, wires=0)
                if config.phase_damping:
                    qml.PhaseDamping(config.phase_damping, wires=0)
            if config.readout_error:
                qml.BitFlip(config.readout_error, wires=0)
            return qml.expval(qml.PauliZ(0))

        self._circuit = qml.set_shots(circuit, shots=config.shots or None)
        self._jacobian = qml.jacobian(self._probabilities, argnums=1)

    def _probabilities(self, angles, theta):
        return (1 + self._circuit(angles, theta)) / 2

    def probabilities(self, circuits, batch_size=1):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not len(circuits):
            return np.empty(0)
        angles = np.asarray(circuits, dtype=float)
        if angles.shape != (len(circuits), 3) or not np.isfinite(angles).all():
            raise ValueError(
                "Expected finite (feature_angle, theta_z, theta_y) triples"
            )
        self.calls += 1
        self.circuits += len(circuits)
        results = []
        for start in range(0, len(circuits), batch_size):
            batch = angles[start : start + batch_size]
            results.append(
                self._probabilities(
                    batch[:, 0], pnp.array(batch[:, 1:], requires_grad=False)
                )
            )
        return np.clip(np.concatenate([np.atleast_1d(p) for p in results]), 0, 1)

    def jacobian(self, values, theta, batch_size=1):
        if self.config.shots:
            raise ValueError("Native backprop/adjoint requires shots=0")
        values, theta = np.asarray(values, dtype=float), np.asarray(theta, dtype=float)
        if (
            batch_size < 1
            or values.ndim != 1
            or theta.shape != (2,)
            or not np.isfinite(values).all()
            or not np.isfinite(theta).all()
        ):
            raise ValueError(
                "Expected finite feature values, two parameters and batch_size >= 1"
            )
        if not len(values):
            return np.empty((0, 2))
        self.calls += 1
        self.circuits += len(values)
        parameters = pnp.array(theta, requires_grad=True)
        return np.concatenate(
            [
                np.asarray(
                    self._jacobian(
                        np.pi * values[start : start + batch_size], parameters
                    )
                ).reshape(-1, 2)
                for start in range(0, len(values), batch_size)
            ]
        )
