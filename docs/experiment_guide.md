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

Use a compatible Linux/NVIDIA cluster. The wheel-based `gpu` extra currently targets Linux x86-64. The default `uv sync --locked` installs the CPU dependency group; CPU and GPU Aer distributions are mutually exclusive. IQM's current Qiskit extra requires the CPU Aer wheel, so use a separate CPU environment for IQM profiles and hardware clients.

On the cluster, create the GPU environment without the default CPU group:

```sh
UV_PROJECT_ENVIRONMENT="$PWD/.venv-gpu" uv sync --locked --no-group cpu --extra gpu --python 3.11
.venv-gpu/bin/python -m unittest -v test_packing
```

Check the cluster GPU, driver, and CUDA compatibility first. Confirm that the GPU test actually runs rather than being skipped. This route was dependency-resolved for Linux in a dry run; GPU execution has not been validated on this Mac. For other architectures, use a compatible GPU-enabled Aer source build and record its environment separately.

The available GPU wheel uses Aer 0.18.2 while the current IQM-compatible CPU environment uses Aer 0.17.2. Run CPU and GPU reference comparisons inside the same GPU-enabled cluster environment, selecting `aer_device: CPU` or `GPU`, so package versions do not confound the comparison. Do not compare local Mac CPU timing with cluster GPU timing as an accelerator speedup.

Invoke `.venv-gpu/bin/python` directly in Slurm jobs. A default `uv run`/`uv sync` would re-enable the CPU group; when using uv in that environment, preserve `--no-group cpu --extra gpu`. Save the resolved lockfile, installed distribution versions, build/runtime information, and `nvidia-smi` output with the campaign.


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

## Scalable synthetic data and compatibility

For `synthetic` or `synthetic_xor`, set `synthetic_samples` (at least 20) and `data_seed` in the training configuration. These settings control data generation independently of optimizer initialization. Save the resulting checksum and exact split indices with each model. Changing the number of training samples alone is not the same experiment as changing the generated dataset size.

Saved schema-v1 model identities preserve their original configuration fields when newer defaults are added. The earlier CPU pilot models still load with their original IDs. This supports reading old artifacts, not silently mixing code revisions in an unfinished campaign: starting new work from a plan checks source, dependency lock, package versions, Python, and platform. If those change, create a new plan in a fresh output directory. Completed old results remain verifiable and analyzable without rerunning them.

## Select a fixed setting using validation only

```sh
uv run qdt select --plan results/cpu-pilot/campaign/plan.json --reference sequential --delta 0.01 --output results/cpu-pilot/fixed-policy.json
uv run qdt predict --model results/cpu-pilot/models/seed-0.json --policy results/cpu-pilot/fixed-policy.json --output results/cpu-pilot/policy-validation.json
```

Selection requires a complete validation campaign and rejects test campaigns. For each frozen model, it selects the lowest median measured total time among settings whose observed mean balanced accuracy is within `delta` of the reference. The policy records model IDs, execution overrides, source/selection provenance, measured selection-attempt time, and bookkeeping time. Missing attempt timing is null. This is the validation-selected fixed-setting baseline, not an adaptive scheduler or a non-inferiority guarantee.

`predict --policy` verifies the policy checksum and model ID. It cannot be combined with `--execution`, and still defaults to validation. An eventual held-out evaluation requires explicit `--split test` after the study protocol and settings are frozen. Include selection cost and test uncertainty when reporting any benefit.

## IBM hardware inference and durable job recovery

IBM is the first live provider. IQM hardware submission remains a later extension; its local profiles continue to work. IBM execution uses SamplerV2, ISA compilation, and the same copy mapping and partition checks as local profile simulation. Twirling and dynamical decoupling are explicitly disabled in this first implementation.

Install the IBM extra and configure a saved IBM account or the existing `.env` variables. Tokens are passed to the service, not written into model artifacts or journals. Choose an actually accessible device; no backend name is assumed to be available.

Create an execution JSON file with your device name and fresh journal path:

```json
{
  "backend": "ibm",
  "ibm_backend": "YOUR_ACCESSIBLE_BACKEND",
  "job_journal": "results/ibm-control/journal",
  "shots": 128,
  "packing_factor": 2,
  "batch_size": 8,
  "max_qpu_jobs": 4,
  "max_qpu_shots": 4096,
  "qpu_timeout_s": 300,
  "qpu_max_execution_time_s": 300
}
```

The numbers are illustrative control-run limits, not a final study budget or a monetary cost guarantee. Set instance/provider spending limits as appropriate. Clear any simulated noise and `fake_qpu` when transferring a model trained with them to hardware. An explicit `qpu_layout` is optional; otherwise a deterministic greedy layout is selected and frozen in the journal.

```sh
uv run --extra ibm qdt predict --model results/model.json --execution configs/execution/ibm-control.json --max-samples 4 --output results/ibm-control/predictions.json
```

Create that execution file first; the path above is an example. `--max-samples` limits the first rows of the selected split or external input matrix and records their indices. The default split remains validation. This is a control run, not a full generalization estimate. Training and the legacy train-and-compare command reject live IBM execution.

Each independent model/repetition/temporal block needs a new journal. Retrying the same interrupted workload must reuse its journal. The journal freezes configuration, selected layout, calibration properties, and compiled QPY templates; it records submission intent before sending a request and persists the returned job ID before waiting for results. A timeout can resume by retrieving that job, even if the backend is temporarily unavailable for new work. Completed results are cached and checked for corruption.

If submission acceptance is uncertain and no job ID was saved, execution stops rather than submitting again. Locate and verify the existing job through the provider before resolving that journal entry; do not delete the journal or rerun with a new one as an automatic retry. Failed or ambiguous intents conservatively consume the job/shot reservation.

`max_qpu_shots` counts composite-circuit shots, including all parameter rows; `max_qpu_jobs` counts submission intents. Runtime execution-time limits are per job. Counters distinguish workload shots from newly submitted jobs/shots and recovered jobs. Raw counts, Runtime metadata, and provider job metrics are retained. Archive the journal with results. Local `campaign` orchestration deliberately rejects live IBM runs until per-run remote campaign coordination is implemented; use standalone frozen-model inference for the initial pilot.

Offline regression tests use local BackendSamplerV2 results and mocked IBM services. They validate parameter ordering, recovery, cached results, limits, and ambiguous-submission handling, but do not establish real device compatibility or performance. Relevant IBM references: [Sampler inputs/outputs](https://quantum.cloud.ibm.com/docs/en/guides/sampler-input-output) and [job retrieval](https://docs.quantum.ibm.com/guides/save-jobs).

## NVIDIA GPU jobs on Slurm

`slurm/qdt.sbatch` forwards any implemented qdt command without reconstructing its arguments. It sets native thread environment variables from the CPU allocation and uses `QDT_PYTHON` to select the environment. The existing `slurm/run.sbatch` remains the legacy seeded comparison array.

Create `logs` before submission. Replace the partition/account placeholders with site settings. Example GPU training:

```sh
mkdir -p logs
QDT_PYTHON="$PWD/.venv-gpu/bin/python" sbatch --partition GPU_PARTITION --account PROJECT --gpus=1 --cpus-per-task=4 slurm/qdt.sbatch train --config configs/campaigns/cpu_pilot_training.json --aer-device GPU --threads 4 --seed 0 --output results/gpu-pilot/models/seed-0.json
```

`--aer-device` and `--threads` override only those training execution settings. Keep the model circuit, data, split, and optimizer budget fixed for CPU/GPU comparisons. Choose different output paths for independent initializations.

Run GPU numerical checks inside an allocated GPU job, for example using the site's interactive allocation followed by:

```sh
srun .venv-gpu/bin/python -m unittest -v test_packing
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

Preserve the hardware output and scheduler logs. Provenance records selected Slurm allocation identifiers and `CUDA_VISIBLE_DEVICES`; actual Aer device metadata remains the execution check.

For a campaign, create a site-specific manifest pointing to models on the cluster's filesystem, with CPU/GPU overrides and explicit limits. Generate its plan on the compute environment, then execute there using the same code/environment:

```sh
QDT_PYTHON="$PWD/.venv-gpu/bin/python" sbatch --wait --partition GPU_PARTITION --account PROJECT --gpus=1 --cpus-per-task=4 slurm/qdt.sbatch campaign plan --manifest configs/campaigns/YOUR_GPU_MANIFEST.json --output results/gpu-pilot/campaign/plan.json
QDT_PYTHON="$PWD/.venv-gpu/bin/python" sbatch --partition GPU_PARTITION --account PROJECT --gpus=1 --cpus-per-task=4 slurm/qdt.sbatch campaign run --plan results/gpu-pilot/campaign/plan.json
```

Create that manifest first. It is not a supplied site configuration. Set `threads` explicitly in its overrides to fit the CPU allocation. A plan resolved on the Mac cannot be used for new cluster runs because source/environment checks and absolute model paths differ. Do not change source or package resolution between planning and execution. Resume by submitting the same run command after a normal interruption; inspect stale claims and budgets as documented earlier.

These commands currently use one coordinator and one allocated GPU. Independent multi-GPU scaling, GPU peak-memory telemetry, and actual site validation remain pending. Run CPU-only IBM inference clients without reserving a GPU unnecessarily. The user currently requests preparation only for these paths: execute experiments locally on CPU and do not submit Slurm or QPU jobs.

## Small CPU profile studies

Campaign manifests may specify `max_samples` to limit the first validation/test rows for a control study. The exact selected indices and limit are frozen in each run specification, used in budget estimates, and verified against completed results. Small prefixes are exploratory control inputs, not a full-dataset generalization estimate.

New plans include an execution-environment fingerprint in each run ID. This distinguishes identical settings collected with different source/dependency environments. Old completed plans remain readable for analysis; unfinished work must still match its original source and environment.


Current IBM calibration snapshots may also advertise `xslow`, `measure_2`, `reset_2`, `measure_reset`, and `measure_reset_2`. The local reader explicitly excludes these specialized instructions from its static simulation target and records `excluded_native_instructions` in profile metadata. Standard gates retain their original calibration/noise information; the raw snapshot is unchanged. Unknown unsupported instructions still raise an error. These simulations do not evaluate the excluded operations or hardware crosstalk.

File-based profile paths are resolved from the planning working directory and frozen with their byte checksums. Execute planning from the repository root for the supplied examples. Changing a snapshot requires a new plan; completed results remain analyzable from their stored records.

## CPU-only noise/profile pilot definitions

The `profile_training_ideal.json` and `profile_training_noisy.json` configurations use the same small training budget and fixed split; only controlled noise differs. Train each with seeds 0 and 1, saving to `results/local-profile-pilot/models/ideal-0.json`, `ideal-1.json`, `noisy-0.json`, and `noisy-1.json`.

Six supplied manifests named `profile_<training>_<regime>.json` cover `training=ideal/noisy` and `regime=custom/ibm/iqm`. Each contains 12 CPU validation runs: two frozen models, packing 1/2/4, and exact or 512-shot measurements. The first 16 validation rows are explicitly limited; no test set is evaluated. The sample is too small for generalization or non-inferiority claims.

The IBM manifest needs `results/local-profile-pilot/profiles/ibm_fez.json`. This is a calibration snapshot, not execution on that QPU. The snapshot was obtained using the existing read-only calibration download command; all circuit calculations run in Aer CPU. Preserve its original bytes: replacing it requires regenerating layouts and plans. IQM uses the bundled Garnet profile. Profile manifests explicitly freeze prefixes of the same selected four-partition layout.

For each manifest, use the standard campaign plan/run/verify/analyze commands with a distinct directory under `results/local-profile-pilot/campaigns/`. Run training, planning, and collection from one committed revision and environment. Keep timing comparisons within a noise regime and training family; profile simulations do not estimate QPU runtime.

`analysis/check_profile_packing.py` provides a stronger numerical control: it compares each packed root-node probability with an isolated circuit using that same physical partition and input. Example:

```sh
.venv/bin/python analysis/check_profile_packing.py --model results/local-profile-pilot/models/ideal-0.json --profile file:results/local-profile-pilot/profiles/ibm_fez.json --packing 4 --max-samples 16 --output results/local-profile-pilot/matched-ibm-4.json
```

The check saves probabilities, layouts, raw records, provenance, and its source checksum, and defaults to an absolute error tolerance of `1e-10`, with an explicit `--atol` override. It also reports the minimum decision margin and requires no threshold-decision disagreements. Repeat for packing 2 and for `iqm:garnet`. Agreement is expected for independent-noise models. It validates composition and decoding, not physical crosstalk or the absence of a hardware packing penalty. Comparisons that assign inputs to different partitions can still differ because of calibration heterogeneity.


In the first downloaded IBM-profile control, a strict `1e-10` check failed at roughly `5e-10`. Native operation sequences/parameters matched, changing fusion and zero-threshold settings did not change the discrepancy, and reconstructed noise channels were trace-preserving to about `6.3e-15`. The exact source of the remaining simulator-level numerical difference is unresolved. For these exploratory profile controls, an explicit `--atol 1e-8` is used, on the scale of the SDK's channel-validation tolerance; this is an engineering acceptance criterion, not a formal error bound. The original failed artifact is retained, and threshold decisions must still agree. Ideal/simple-noise unit tests retain tighter tolerances.
