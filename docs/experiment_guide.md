# Experiment guide

Use a new output path for every artifact. All commands below are implemented; hardware and GPU availability are separate requirements.

## Train once and reuse the model

```sh
uv sync --locked
uv run qdt train --config configs/aer_noiseless.json --output results/model.json
uv run qdt predict --model results/model.json --output results/validation.json
uv run qdt predict --model results/model.json --split test --output results/test.json
```

`predict` defaults to validation. Run the test evaluation only after selecting settings on validation. Example configurations now use a fixed split seed and 60/20/20 train/validation/test partitions. API configurations with `validation_size=0` retain legacy train/test splits and cannot evaluate a missing validation set.

Inference loads the saved preprocessing, tree, and angles. It does not train or select features. Models contain the dataset checksum and exact split indices; dataset changes are rejected. To evaluate external data, pass `--inputs path.json` containing a matrix of finite raw features in the original column order. No labels are needed for external predictions.

To change execution settings, pass `--execution path.json` containing only permitted overrides, for example:

```json
{"shots": 2048, "batch_size": 32, "simulator_seed": 123}
```

Model/circuit/training changes are rejected as inference overrides. Train a new model for those changes. `split_seed`, `optimizer_seed`, `simulator_seed`, and `transpiler_seed` are recorded independently. Unspecified split/optimizer/simulator seeds inherit `seed`; the transpiler defaults to zero.

Training records each objective evaluation and restart termination, with node/candidate context and declared budgets. Timing includes setup and training; inference reports setup, prediction, and total time. Detailed backend instrumentation is a subsequent implementation step.

The original `qdt --config ... --output ...` comparison command remains available for exploratory compatibility. It trains both QDT and CART and is distinct from the frozen-model workflow.

## Spatial packing and Aer devices

`packing_factor` is the number of disjoint copies in a composite circuit. `batch_size` retains its existing name and controls composite circuits per backend request. They are independent settings. Partial batches use a smaller template; unused copies are not executed. Finite shots are shared by all active copies in a composite circuit.

Example inference overrides:

```json
{"packing_factor": 4, "batch_size": 16, "shots": 2048, "aer_device": "CPU", "aer_method": "automatic"}
```

Methods are `automatic`, `statevector`, and `density_matrix`. Automatic selection uses density matrices for gate noise and statevectors otherwise. Exact gate-noise references require density matrices. GPU requests must be supported by the installed build and are checked against actual result metadata.

## GPU environment

Use a compatible Linux/NVIDIA cluster. Keep its environment separate from local CPU development. The CPU lockfile does not provide GPU support. Follow the official Aer installation documentation linked in the roadmap and verify the distribution against cluster drivers and runtime.

1. Create a separate environment, for example `.venv-gpu`.
2. Install the project and its CPU-pinned dependencies, then replace the standard `qiskit-aer` distribution with a compatible, explicitly pinned `qiskit-aer-gpu` package (or a GPU-enabled source build).
3. Check the effective environment rather than installing both overlapping Aer distributions. Save the resolved versions/build and GPU runtime details with the campaign.
4. Run `.venv-gpu/bin/python -m unittest -v test_packing` and confirm the GPU test runs instead of being skipped.
5. Use `aer_device: GPU` in execution overrides and preserve result metadata. Do not run `uv sync` against the CPU project environment as a way to prepare GPU support.

GPU correctness and performance have not been measured on the current CPU-only Mac. Multi-GPU scaling remains a later pilot.

## Hardware-profile simulation and execution records

Profiles still run locally in Aer. A profile does not submit a QPU job. For packing greater than one, provide `qpu_layout` with one disjoint physical-qubit group per copy, flattened in copy order. For a two-qubit circuit and packing two, an example is `[0, 1, 4, 5]`, provided those groups are valid for the selected device. Tail batches reuse the first active partitions.

The compiler must keep operations inside the declared partitions. Routing outside them, interactions across copies, or shared resonator operations are rejected for packed profile simulations. Automatic partition search and hardware crosstalk characterization are not yet implemented. Readout corrections support independent asymmetric errors; correlated readout requires a later extension.

Frozen-model training and inference outputs include backend request records with inputs, raw counts for finite shots, marginal probabilities, actual Aer metadata, and layout information. Counters distinguish logical evaluations, composite circuits, backend requests, composite shots, and logical marginal samples. Exact-probability references record zero physical shots, although Aer internally receives a dummy shot argument.

`max_memory_mb` caps Aer memory; `parallel_experiments` and `parallel_shots` expose native execution limits alongside `threads`. CPU/GPU memory reporting currently includes process-lifetime peak RSS; GPU peak memory is explicitly unavailable rather than estimated. Raw training records currently remain inside the model JSON, which is simple but can become large; campaigns should archive these files and budget storage accordingly.
