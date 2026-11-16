import subprocess
import sys
import unittest

import numpy as np

from qdt.adapters import get_adapter
from qdt.config import Config
from tests.test_qdt import TEST_CIRCUIT

ANGLES = np.random.default_rng(5).uniform(-4, 4, (7, 3))


class AdapterTests(unittest.TestCase):
    def check_inputs(self, adapter):
        self.assertEqual(adapter.probabilities([], 64).size, 0)
        for angles, batch in (
            ([(0, 0)], 1),
            ([(0, np.nan, 0)], 1),
            (ANGLES, 0),
            (ANGLES, 1.5),
        ):
            with self.assertRaises(ValueError):
                adapter.probabilities(angles, batch)
        self.assertEqual((adapter.calls, adapter.circuits), (0, 0))

    def test_aer_factory_and_optional_imports(self):
        self.check_inputs(get_adapter(Config(circuit=TEST_CIRCUIT)))
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from qdt.adapters import get_adapter; from qdt.config import Config; get_adapter(Config.load('configs/aer_noiseless.json')); assert 'iqm.qiskit_iqm' not in sys.modules",
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
