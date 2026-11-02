# QDT

For the HPC–QPU circuit batching research plan, see the
[research roadmap](docs/RESEARCH_ROADMAP.md), including implementation milestones,
experiment protocols, result collection and paper preparation.

QDT compares a quantum decision tree with a classical decision tree (CART) on
classification datasets. Aer supports CPU/GPU simulation, including local
IQM/IBM device profiles.
Frozen-model inference also supports budgeted IBM hardware execution with durable
job recovery; see the [experiment guide](docs/experiment_guide.md).

Install Python 3.11+ and [uv](https://docs.astral.sh/uv/), then run from the project
directory:

```sh
uv sync --locked
uv run qdt --config configs/aer_noiseless.json --output results/experiment.json
```

For separate training and frozen-model inference, see the
[experiment guide](docs/experiment_guide.md).

This trains both models on the breast cancer dataset, prints their accuracy and
saves metrics, predictions, timings, the quantum tree and its configuration.
Use a new output filename for each run; existing files are never overwritten.

Choose an example in `configs/`: `aer_noiseless.json` for ideal simulation,
`custom_noisy.json` for user-defined noise, or `iqm.json` / `ibm.json` for
device profiles. Edit the JSON to set the circuit,
tree depth, training budget and sampling: `shots: 0` uses exact probabilities;
positive values sample measurement outcomes. Override the dataset or seed at
launch with, for example, `--dataset iris --seed 7`.

For custom noise, keep `fake_qpu: null` and adjust `depolarizing`,
`amplitude_damping`, `phase_damping` and `readout_error` between 0 and 1.
The example values are illustrative, not hardware calibrations. With an IQM or
IBM profile, keep these four rates at zero: noise comes from the profile.

IQM requires its optional dependency. Run its example with
`uv run --extra iqm qdt --config configs/iqm.json`.
Use `uv run qdt --list-qpus` to list bundled IQM models.

For IBM, copy `.env.example` to `.env`, fill in your credentials and download a
calibration before running the experiment:

```sh
mkdir -p profiles
uv run --extra ibm python -m qdt.qpu list --online
uv run --extra ibm python -m qdt.qpu fetch ibm:ibm_fez profiles/ibm_fez.json
uv run qdt --config configs/ibm.json --output results/ibm.json
```

Choose a QPU listed as accessible to your account. If using a device other than
`ibm_fez`, update the download command and the `fake_qpu` file path in
`configs/ibm.json`. The downloaded profile is then used locally in Aer.

Available datasets: `iris`, `wine`, `breast_cancer`, `digits`, `ionosphere`, `sonar`,
`synthetic` and `synthetic_xor`. Download missing Ionosphere and Sonar data with
`uv run qdt --download-data`. See `uv run qdt --help` for all command-line options.
