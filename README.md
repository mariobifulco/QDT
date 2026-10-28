# QDT

Quantum decision tree experiments for classification, with a classical decision
tree (CART) as a baseline. Supports noiseless and noisy quantum circuit simulation.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). From the project directory:

```sh
uv sync --locked
```

## Run an experiment

```sh
uv run qdt --config configs/aer_noiseless.json --output results/experiment.json
```

This trains both models on the breast cancer dataset and prints their accuracy.
The JSON file includes metrics, predictions, timings, the quantum tree and the
experiment configuration. Choose a new output filename for each run; existing
files are never overwritten.

To change the dataset or random seed:

```sh
uv run qdt --config configs/aer_noiseless.json --dataset iris --seed 7
```

Edit the JSON configuration to change the circuit, tree depth, training budget,
sampling (`shots`) or noise rates. With `shots: 0`, the simulator uses exact
probabilities.

Available datasets: `iris`, `wine`, `breast_cancer`, `digits`, `ionosphere`, `sonar`,
`synthetic` and `synthetic_xor`. If the Ionosphere or Sonar data files are missing,
download them with `uv run qdt --download-data`.

Run `uv run qdt --help` for all command-line options.
