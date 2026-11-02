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

Profiles still run locally in Aer. A profile does not submit a QPU job. For packing greater than one, an omitted `qpu_layout` selects a deterministic greedy layout using target readout and two-qubit error data. Missing errors are penalized, not treated as perfect. You may instead provide `qpu_layout` with one disjoint physical-qubit group per copy, flattened in copy order. For a two-qubit circuit and packing two, an example is `[0, 1, 4, 5]`, provided those groups are valid for the selected device. Tail batches reuse the first active partitions.

The compiler must keep operations inside the declared partitions. Routing outside them, interactions across copies, or shared resonator operations are rejected for packed profile simulations. The greedy selector is a baseline, not an optimal solver or proof of maximum feasible packing. A proposed layout can still fail compilation checks; use an explicit layout to investigate alternatives. Hardware crosstalk characterization is not implemented. Readout corrections support independent asymmetric errors; correlated readout requires a later extension.

Frozen-model training and inference outputs include backend request records with inputs, raw counts for finite shots, marginal probabilities, actual Aer metadata, and layout information. Counters distinguish logical evaluations, composite circuits, backend requests, composite shots, and logical marginal samples. Exact-probability references record zero physical shots, although Aer internally receives a dummy shot argument.

`max_memory_mb` caps Aer memory; `parallel_experiments` and `parallel_shots` expose native execution limits alongside `threads`. CPU/GPU memory reporting currently includes process-lifetime peak RSS; GPU peak memory is explicitly unavailable rather than estimated. Raw training records currently remain inside the model JSON, which is simple but can become large; campaigns should archive these files and budget storage accordingly.

## Resumable local campaigns

Train frozen models first. A manifest uses paths relative to its own location:

```json
{
  "schema_version": 1,
  "models": ["model.json"],
  "split": "validation",
  "repetitions": 2,
  "executions": [
    {"name": "isolated", "overrides": {"packing_factor": 1, "batch_size": 32}},
    {"name": "packed", "overrides": {"packing_factor": 4, "batch_size": 32}}
  ],
  "limits": {"max_runs": 20, "max_logical_shots": 1000000, "max_seconds": 3600}
}
```

```sh
uv run qdt campaign plan --manifest results/manifest.json --output results/pilot/plan.json
uv run qdt campaign run --plan results/pilot/plan.json
uv run qdt campaign verify --plan results/pilot/plan.json
```

Planning validates artifacts, datasets, configuration, device support, and profile layouts without calling `backend.run`. It resolves immutable run IDs and upper bounds from sample count and tree depth. Model contents and plans are checked for changes. Each repetition increments the resolved simulator seed; batching can still change random-number consumption, so counts across strategies are not expected to be identical.

Run the same `campaign run` command to resume. Completed results are skipped only after checksum and identity checks. Each attempt keeps its specification, start/provenance, raw backend records where available, result or failure, duration, and completion checksum. Invalid or corrupted completed results raise an error instead of being silently accepted or overwritten.

A campaign uses one coordinator. Per-run claims prevent duplicate execution. After a hard process interruption, `--recover-stale` can recover a claim only from the same host after checking that its recorded PID is absent. Claims from another host or a live PID are not removed automatically.

The shot budget conservatively reserves each attempt's logical-shot upper bound, including failed attempts, before retrying. Time limits include recorded prior attempt durations; an unmeasured hard interruption conservatively reserves wall time since its start, including downtime. The deadline is checked between runs and does not interrupt an in-progress Aer call; use scheduler wall-time limits for a hard external cutoff. These are local simulation campaigns, not live QPU orchestration. Parallel campaign workers and global multi-GPU scheduling are not implemented.

## Rebuild analysis without running circuits

```sh
uv run qdt analyze --plan results/pilot/plan.json --output results/pilot/analysis --reference isolated
uv run --extra analysis qdt analyze --plan results/pilot/plan.json --output results/pilot/analysis-with-plots --reference isolated --plots
```

Use a new or empty output directory. Analysis checks completion/result integrity and recomputes metrics from stored predictions. It produces `runs.csv`, `comparisons.json`, and `report.md`; `--plots` additionally creates standard quality–time PDF/PNG scatter plots using the optional Matplotlib dependency. No model training, circuit execution, or QPU access is needed.

Timing repetitions are collapsed within each model. Quality differences and time ratios are paired by frozen model, separately for each dataset. Confidence intervals use 2,000 model-level bootstrap samples with a fixed seed; one-model comparisons have no confidence interval. Small exploratory samples do not establish non-inferiority. Incomplete campaigns are rejected unless explicitly requested with `--allow-incomplete`; missing runs and failure counts remain in the report.

## Reproduce the local CPU smoke campaign

These definitions deliberately use two model initializations, one shallow two-qubit QDT, 40 training samples, a fixed split, and two timing repetitions. They validate the pipeline and provide exploratory observations, not final scientific evidence. The manifest includes sequential/native-batch/packed exact execution and finite-shot isolated/packed execution.

Use a fresh `results/cpu-pilot` directory; existing artifacts are never overwritten:

```sh
uv run qdt train --config configs/campaigns/cpu_pilot_training.json --seed 0 --output results/cpu-pilot/models/seed-0.json
uv run qdt train --config configs/campaigns/cpu_pilot_training.json --seed 1 --output results/cpu-pilot/models/seed-1.json
uv run qdt campaign plan --manifest configs/campaigns/cpu_pilot.json --output results/cpu-pilot/campaign/plan.json
uv run qdt campaign run --plan results/cpu-pilot/campaign/plan.json
uv run qdt campaign verify --plan results/cpu-pilot/campaign/plan.json
uv run --extra analysis qdt analyze --plan results/cpu-pilot/campaign/plan.json --output results/cpu-pilot/analysis --reference sequential --plots
```

Preserve raw results and model artifacts locally. They are ignored by Git and must be included explicitly in a later publication archive. To run elsewhere, train new models there and resolve a new plan from the manifest; resolved plans contain absolute artifact paths.

On systems with slow system-font discovery, the tested Matplotlib version can use bundled fonts by setting `MPL_IGNORE_SYSTEM_FONTS=1`; set `MPLCONFIGDIR` to a writable cache directory. These settings affect figure generation, not simulation measurements. The first CPU pilot is documented in `docs/pilots/cpu_pilot_20261102.md`.
