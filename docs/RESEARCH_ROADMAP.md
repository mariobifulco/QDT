# Research roadmap: circuit batching for variational HPC–QPU workflows

Reference date: November 2, 2026. Target: prepare a submission to IEEE Quantum Week 2027.

This document describes the path from the current repository to a reproducible research artifact for running experiments, collecting results, and writing the paper. Milestones describe work to be done; listing a capability here does not mean it has already been implemented.

## Implementation checkpoint

All implementation notes and project documentation are maintained in English. Each completed step receives a local commit; never push. Resume from the first pending item below, read the linked implementation notes, and inspect `git status` before editing.

| Step | Status | Local commit subject / evidence |
|---|---|---|
| S00 — Record the approved roadmap | Complete | `docs: checkpoint the approved research roadmap` |
| S01 — Freeze an initial protocol and remove CUDA-Q | Complete | `chore: remove CUDA-Q and record the exploratory protocol` |
| S02 — Save preprocessing, validation splits, and complete models | Complete | `feat: persist preprocessing splits and frozen QDT models` |
| S03 — Separate training/inference and capture optimizer traces | Complete | `feat: separate training and frozen inference with optimizer traces` |
| S04 — Implement Aer spatial packing and CPU/GPU selection | Implemented; GPU validation pending | `feat: add disjoint circuit packing and explicit Aer devices` |
| S05 — Instrument execution and support packed hardware profiles | Complete offline; hardware validation pending | `feat: record execution costs and simulate packed hardware profiles` |
| S06a — Implement resumable local campaigns | Complete | `feat: add immutable local campaigns and safe resumption` |
| S06b — Add paired analysis | Complete | `feat: rebuild paired campaign analysis and optional figures` |
| S06c — Define the reproducible local CPU pilot | Complete | `test: define a small reproducible CPU pilot` |
| S06d — Execute and document the CPU pilot | Complete | `docs: record the verified exploratory CPU pilot` |
| S07 — Add deterministic profile partition selection | Complete offline | `feat: select deterministic calibration-aware profile partitions` |
| S08 — Preserve old model IDs and add scalable synthetic data | Complete | `feat: scale synthetic data and preserve frozen artifact compatibility` |
| S09 — Add validation-only fixed-setting selection | Complete offline | `feat: select and apply validation-only fixed execution policies` |
| S10 — Implement the IBM live QPU adapter and durable job recovery | Implemented; live validation pending | `feat: add budgeted IBM inference with durable job recovery` |
| S10b — Prepare the NVIDIA Slurm launch path | Implemented; cluster validation pending | `feat: prepare isolated Aer GPU environments and Slurm workflows` |
| S11a — Freeze CPU control inputs and execution identity | Complete | `feat: freeze campaign sample limits and execution identity` |
| S11b — Load current IBM snapshots for static CPU simulation | Complete | `fix: support static simulation of current IBM profiles` |
| S11c1 — Freeze external profile inputs | Complete | `fix: bind local campaigns to immutable noise profiles` |
| S11c2 — Define CPU noise/profile campaigns | Complete | `test: define CPU noise-profile transfer and partition controls` |
| S11d — Collect CPU profile pilots | Complete | 72/72 local validation runs passed integrity checks |
| S11e — Make numerical control tolerance explicit | Complete | `test: expose numerical tolerance in matched-partition controls` |
| S11f — Complete controls and commit CPU study report | Complete | `docs: checkpoint verified CPU profile study and resume instructions` |
| S12a — Extend CPU tracing and benchmark rigor | Next | Node-path traces, randomized repetitions, warm/cold measurements |
| S11-remote — GPU/HPC and real-QPU pilots | Deferred by user | Prepare code only; no remote jobs |
| S12 — Confirmatory policy study, final campaign, and paper | Pending scientific evidence | M8–M11 |

Current environment: local macOS, Qiskit 2.1.2, Aer 0.17.2; Aer reports CPU only. No GPU benchmark or real QPU execution has been performed. Hardware campaigns and empirical claims must remain pending until measured. The pre-existing README and roadmap edits are included in S00.

### Latest implementation notes

- S00: `d773f3a` records the approved roadmap.
- S01: CUDA-Q adapter/config/dependencies/CI removed; 24 checks passed (one optional check skipped). Dependency resolution required registry metadata after the offline cache proved incomplete; existing retained versions were preserved. Initial protocol and related-work comparison are in `docs/research_protocol.md` and `docs/related_work.md`; references are in `paper/references.bib`. Final hardware budget and full-text novelty audit remain pending, so M0 is exploratory rather than fully frozen.
- S02: complete JSON model artifacts include preprocessing, classes, split indices, dataset checksum, and integrity ID. Fresh-process inference roundtrip and training-only preprocessing verified. Examples now use 60/20/20 splits with fixed split seed; `validation_size=0` retains legacy API splits. 25 tests passed (one optional skip).
- S03: `qdt train` and `qdt predict` implemented; prediction defaults to validation and reuses a frozen model. Separate split/optimizer/simulator/transpiler seeds and per-node optimizer traces are recorded. CLI roundtrip, non-overwrite, and budget checks added; 26 tests passed (one optional skip). See `docs/experiment_guide.md`.
- S04: cached disjoint-copy templates, explicit little-endian marginals, partial batches, independent request/packing settings, and Aer CPU/GPU/method selection implemented. Fusion enabled for fair native baselines. 29 tests passed with two skips; actual GPU smoke validation remains pending on compatible hardware. See the GPU environment instructions in the experiment guide.
- S05: Aer and profile adapters share packed execution and accounting. Raw counts, parameter rows, metadata, resolved configurations, checksums, timing, process peak RSS, and exact/shot counters are persisted. Packed profiles require explicit layouts and reject cross-partition/shared-resource routing. Independent asymmetric readout is supported. Automatic partition selection, correlated readout, GPU peak-memory measurement, and live QPU integration remain pending.
- S05 verification: installed the existing IBM/IQM extras and ran all 31 tests; only the GPU check is skipped. This includes all bundled IQM profile/MOVE checks and explicit two-partition IBM-profile comparisons. Lint and formatting pass. No remote provider was contacted by these tests.
- S06a: immutable plans, deterministic run IDs, dry validation, atomic non-overwriting JSON, per-attempt records, integrity checks, same-host stale-claim recovery, conservative retry shot reservations, and between-run time limits implemented for one local coordinator. Interruption/resumption and corruption tests pass. Live QPU orchestration, parallel campaign workers, and hard within-run deadlines are not implemented.
- S06b: analysis validates raw predictions, writes CSV/JSON/Markdown, and optionally generates PDF/PNG figures. Comparisons pair frozen models within datasets and collapse timing repetitions before model-level bootstrap; no CI is emitted for a single model. Analysis tests forbid circuit execution. Matplotlib is an optional analysis extra, not a simulation dependency.
- S06c: committed pilot training/campaign manifests and exact reproduction commands. The run matrix has two frozen models, six execution conditions, and two repetitions (24 runs), using validation only.
- S06d: 24/24 CPU validation runs completed from clean commit `9fbbbc8`; resumption skipped all completed runs. PDF/PNG generated and visually checked. Native request batching already captures most observed speedup in this tiny pilot; no general packing, GPU, or hardware claim is established. Summary and checksums: `docs/pilots/cpu_pilot_20261102.md` and `.json`. Raw data remains in ignored `results/cpu-pilot/`. Full regression: 35 tests, one GPU skip; lint/format/Slurm syntax pass.
- S07: optional deterministic greedy connected partitions use target readout/two-qubit errors and penalize missing calibration. Explicit layouts remain supported; compiled-circuit validation rejects infeasible routing and shared resonators. Tests verify deterministic calibration ranking and automatic/explicit profile equivalence. No optimal packing or crosstalk claim is made.
- S08: synthetic sample count/data seed are configurable and independently recorded. Saved schema-v1 model IDs remain stable across additive Config defaults; the real S06 CPU pilot models were reloaded successfully. Unfinished campaigns now reject code/environment changes instead of silently mixing revisions; completed old results remain readable.
- Final S09 verification: 39 tests passed with one GPU skip; lint and formatting pass.
- S09: validation-only per-model fixed-setting selection, policy integrity/model matching, measured attempt/bookkeeping cost, and `predict --policy` implemented. Test campaigns and incomplete validation campaigns are rejected. This implements a required baseline, not the proposed adaptive research contribution.
- S10: IBM SamplerV2 inference, shared ISA/partition compilation, frozen QPY templates, explicit job/shot/runtime limits, durable submission intents/job IDs, result caching, and ambiguous-submission refusal implemented. Offline tests use local sampler results and mocked services; no account was contacted and no QPU job was submitted. Saved-account and environment credential loading are supported. IQM live execution and remote campaign orchestration remain pending.
- S10 verification: 42 tests passed with one GPU skip; IBM tests submit only to a local sampler behind mocked Runtime services. A wait timeout and temporarily unavailable backend recover the accepted job; cached reads submit no new jobs. Shared compiler regression checks cover IBM and all IQM profiles.
- S10b: default CPU and mutually exclusive GPU dependency routes, Linux GPU dry resolution, generic Slurm command forwarding, train CPU/GPU/thread overrides, and allocation provenance implemented. GPU wheel resolution selects Aer 0.18.2; the IQM-compatible CPU route remains 0.17.2. Compare CPU/GPU inside the same GPU-enabled cluster build, not across these environments. IQM and GPU-wheel extras are intentionally exclusive because IQM pulls the CPU wheel.
- Read-only IBM discovery on November 2, 2026 confirmed access to `ibm_fez`, `ibm_kingston`, and `ibm_marrakesh`; no QPU jobs were submitted. A specific backend/budget and cluster SSH alias/partition/account/GPU model were requested and remain pending.
- S10b verification: 44 tests passed with one GPU skip; Linux GPU dependency dry resolution succeeded without installing GPU packages on the Mac. Slurm argument/thread forwarding, shell syntax, lint, formatting, and the earlier 24-run artifact verification pass.
- Current user instruction: prepare GPU/IBM code but execute computations locally on CPU only. Remote GPU/HPC and QPU jobs are deferred; no further hardware-access or job-budget decision is needed for the current CPU phase.
- S11a: campaign sample limits, exact input indices, budget bounds, and environment fingerprints are frozen in run identities. Completed-result checks validate indices/environment as well as configuration and file checksums. Seven campaign tests pass.
- S11b: downloaded IBM calibration metadata for local Aer use only; no quantum circuit was submitted. The current snapshot advertises specialized instructions beyond the static simulator gate subset. The loader now reports explicit exclusions and still rejects unknown unsupported gates. Nine profile tests pass, including current-instruction projection.
- S11c1: file-based noise-profile selectors are resolved to absolute paths and their bytes are included in run identity; changed snapshots are rejected before execution. Eight campaign tests pass.
- S11c2: ideal/noisy training configurations, six explicit CPU campaign manifests (72 validation runs total), fixed profile layouts, sample limits, and a matched-partition root-probability diagnostic are prepared. Raw IBM snapshot is local/ignored and its checksum is frozen by planning; IQM uses the bundled Garnet profile.
- S11d: four models and all six CPU campaigns were collected from clean commit `82686fb`; 72/72 validation runs completed without failed attempts and analysis/figures regenerated.
- S11e: the initial IBM matched-partition check failed its strict `1e-10` threshold at `5e-10`. Fusion/zero-threshold ablations did not change it; reconstructed channels have trace residual about `6.3e-15`, and native operation sequences/parameters match. Its precise simulator-level cause remains unresolved. Diagnostic tolerance is now explicit; profile controls will use `--atol 1e-8` and must also preserve threshold decisions. The failed artifact remains archived.
- S11f: four matched-partition controls passed at explicit `1e-8`, with zero threshold-decision changes. Maximum IBM differences were `5e-10`/`6.84e-10`; IQM differences were below `1.3e-15`. Original strict failure is retained. CPU report and checksums: `docs/pilots/local_profiles_20261102.md` and `.json`; raw data/figures: ignored `results/local-profile-pilot/` (about 43 MB).
- Final verification: 47 tests discovered, 46 passed and one GPU check skipped; lint, formatting, shell syntax, and integrity checks pass. All circuit computations were local CPU simulation. No GPU/Slurm/QPU circuit jobs were submitted.
- Next: S12a node-path inference traces, randomized/repeated CPU benchmarking, and separate warm/cold costs. Use larger validation/split/model banks before evaluating a quality-constrained contribution. Independent multi-GPU scheduling, GPU memory telemetry, remote campaign coordination, final study collection, adaptive scheduling, full-text novelty review, and manuscript results remain unfinished.

### Resume instructions

1. Read this checkpoint and `docs/pilots/local_profiles_20261102.md`; inspect `git status` and recent local commits. Never push.
2. Keep execution CPU-only until the user changes that instruction. The IBM/Slurm paths are preparation, not permission to launch hardware jobs.
3. Use the default CPU group: `uv sync --locked --extra ibm --extra iqm --extra analysis`. Regression command: `.venv/bin/python -m unittest discover -q`.
4. Preserve ignored raw data in `results/cpu-pilot/` and `results/local-profile-pilot/`. The committed reports contain checksums, not the complete raw archive. Preserve the original failed strict numerical control as well.
5. Resume at S12a. Create a new manifest/plan/output directory for new code or experimental settings; do not overwrite old models or results. Existing results can be analyzed without new circuit execution.
6. Keep the test set held out. The completed studies use validation prefixes and cannot establish paper-level performance or non-inferiority.

## 1. Research objective and scope

The main research question is:

> How should batching be selected across a variational HPC–QPU workflow to reduce execution time and resource consumption while maintaining a predefined level of application quality?

The initial workflow is training with a simulator on CPU/GPU resources, saving the complete model, and running inference on a superconducting QPU with multiple independent circuits executed simultaneously on distinct hardware regions. Training and inference are separate stages; this workflow does not require a low-latency connection between an optimizer and a QPU.

The quantum decision tree (QDT) is the primary workload. Evaluate both individual node circuits and inference through the complete tree. If the title and conclusions make general claims about variational quantum circuits (VQCs), add at least one variational classifier without a tree structure.

The candidate contribution is a procedure for selecting packing, layout, and shots using application quality and total execution cost. An adaptive policy for QDT nodes is a possible later extension. Novelty must be checked against related work; using HPC or a different dataset does not establish it by itself.

Do not assume in advance that:

- The optimal simulator batch size is small.
- The largest physically feasible packing factor is the best QPU configuration.
- Parameters trained in simulation retain their predictive quality on hardware.
- Lower QPU execution time implies lower end-to-end time.
- An improvement over another execution of the same quantum workload establishes an advantage over classical computing.

### 1.1 Required terminology

| Term | Meaning | Proposed variable |
|---|---|---|
| Logical circuit | One evaluation with identified inputs and parameters | `logical_circuit_id` |
| Request batching | Evaluations grouped into one backend request | `request_batch_size` |
| Spatial packing | Independent copies placed on disjoint qubits within one composite circuit | `packing_factor` |
| HPC parallelism | Resources processing independent simulations concurrently | Process/GPU count and execution strategy |
| Training mini-batch | A subset of training examples used for a model update | `training_minibatch_size`, only if introduced |
| Shots | Repetitions of a single or composite physical circuit | `shots` |

The current `batch_size` mainly controls request grouping. Decide explicitly how to rename or migrate it. Do not silently reinterpret historical results as spatial packing experiments.

### 1.2 Research questions

| ID | Question | Required evidence |
|---|---|---|
| RQ1 | Does packing preserve marginal probabilities in ideal simulation? | Numerical comparisons with separate circuits and readout checks |
| RQ2 | Which Aer execution strategy uses CPU/GPU resources most efficiently? | Throughput, memory, scaling, and training time with declared resources |
| RQ3 | What quality penalty does packing introduce on superconducting hardware? | Isolated/simultaneous comparisons using the same qubits and inputs |
| RQ4 | How well does a model transfer from simulation to hardware? | Frozen models evaluated under different noise and packing conditions |
| RQ5 | Does quality-constrained selection improve the cost–accuracy tradeoff? | Comparisons with unit packing, maximum feasible packing, and the best fixed setting selected on validation data |
| RQ6 | Do tree dependencies limit the benefits? | Batch occupancy, errors by tree level, latency, and complete QDT quality |

### 1.3 Technology scope

Use a Qiskit-based stack throughout the planned experiments:

| Execution target | Planned stack | Scope |
|---|---|---|
| Local and HPC CPUs | Qiskit circuits and Qiskit Aer | Required reference and simulation platform |
| HPC GPUs | Qiskit circuits and a GPU-enabled Aer installation | Required GPU simulation path |
| IBM superconducting QPU | Qiskit and IBM Runtime | Candidate first hardware provider |
| IQM superconducting QPU | Qiskit and IQM's Qiskit integration | Alternative first hardware provider |
| Local hardware-profile simulation | Aer with IBM/IQM-derived profiles | Controlled experiments and preparation for hardware |
| CUDA-Q | Existing implementation, outside the research scope | No new development or campaign dependency; evaluate removal |

Choose IBM or IQM according to actual access. Supporting and evaluating both providers is an extension, not a requirement for the first complete study. Do not make cross-provider claims from a single device.

Use Qiskit circuit representations, parameter handling, measurement decoding, and experiment definitions across CPU, GPU, and hardware. Keep provider-specific submission, compilation, and metadata handling explicit: a shared circuit SDK does not make the IBM and IQM execution services interchangeable.

For the initial GPU platform, use a compatible Linux/NVIDIA environment with GPU-enabled Aer. Document the exact supported GPU, driver, runtime, and package/build combination on the available cluster. The local Mac can provide CPU checks; do not assume it provides the same Aer GPU path. Aer GPU installation and device selection are documented in the [installation guide](https://qiskit.github.io/qiskit-aer/getting_started.html) and [AerSimulator reference](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator).

Prioritize single-GPU simulation and independent circuits distributed across CPU workers or GPUs. Distributed simulation of one large statevector through MPI/cache blocking is optional unless circuit sizes require it. Distinguish this from distributing independent circuit evaluations.

## 2. Original repository baseline (before implementation)

The initial inspection identified:

- Configurable parameterized circuits with `x` inputs and `theta` parameters.
- A QDT trained with COBYLA and a CART baseline.
- Aer simulation with exact probabilities and finite shots.
- A CUDA-Q adapter, optional dependencies, example configuration, and CI checks that are outside the revised research scope.
- Configurable noise and IBM/IQM profiles used locally in Aer.
- JSON output containing configuration, tree, predictions, and some metrics.
- A Slurm script distributing independent experiments across seeds.
- Tests and CI. The initial check passed the 16 tests in `test_qdt`; this does not certify optional dependencies, GPU execution, or real hardware.

Main gaps relative to the research plan:

| Area | Current baseline | Required work |
|---|---|---|
| Spatial packing | Not implemented by the current adapters | Compose disjoint copies and reconstruct their results |
| Aer CPU/GPU selection | No explicit device selection exposed by the current configuration | Add device/method options and verify the execution device |
| Aer environment | Standard CPU-oriented dependency setup | Document and validate a compatible GPU installation |
| CUDA-Q | Present but outside the selected stack | Evaluate removal without making it a dependency of other milestones |
| Training/inference | Essentially the same configuration in the CLI | Separate model artifacts from execution configuration |
| Deployable model | Tree and selected features saved; preprocessing incomplete | Save a complete artifact that loads in a fresh process |
| Real QPU execution | Hardware profiles used for local simulation | Implement actual submission and metadata collection for the chosen provider |
| Timing | Adapter initialization outside the main timers | Include setup, compilation, communication, and waiting |
| Counters | Probability-function calls and sample circuits | Also count backend requests, composite circuits, and shots |
| HPC | Array of independent seeded runs | Explicit within-workload CPU/GPU parallelism benchmarks |
| Analysis | Preliminary and historical results | Versioned campaigns, statistical analysis, and reproducible figures |

Do not automatically combine historical benchmarks with new results. Available metadata includes local Mac runs and earlier implementations. Check provenance, compatibility, and experimental purpose first. CUDA-Q results from related work remain relevant literature even though CUDA-Q is excluded from this implementation plan.

## 3. Milestones and dependencies

| Milestone | Outcome | Dependencies |
|---|---|---|
| M0 | Frozen scope and research protocol | None |
| M1 | Complete model artifact and separate training/inference | M0 |
| M2 | Correct packing in Aer on CPU/GPU | M1 |
| M3 | Layout and noise handling for packed circuits | M2 |
| M4 | Measurements, provenance, and result format | M1; extend through M2–M3 |
| M5 | Resumable campaigns and an operational interface | M2–M4 |
| M6 | CPU/GPU pilot and HPC feasibility | M5 |
| M7 | Superconducting QPU pilot and simultaneous-execution characterization | M3–M6 |
| M8 | Batching selection and QDT integration | M6–M7 |
| M9 | Frozen final campaign | M8 |
| M10 | Verified analysis and figures | M9 |
| M11 | Paper and reproducibility package | M10 |

Develop M4 instrumentation alongside the experiments. Do not leave it until collection is complete. Start the final campaign only after correctness, resource accounting, and recovery have been checked.

### 3.1 CUDA-Q removal assessment

Implementation status: removed in S01. Aer-only adapter checks were retained; the lockfile no longer contains CUDA-Q. The list below records the cleanup scope.

CUDA-Q is not needed to complete any milestone. No active research configuration requires it. The implementation has been removed; historical results remain untouched.

Perform the assessment early, before introducing new Aer options:

1. Check whether an active experiment still requires `backend="cudaq"`. Preserve the provenance of existing results even if their execution path is removed.
2. If removal is appropriate, remove `src/qdt/adapters/cudaq_run.py`, `configs/cudaq_noiseless.json`, the CUDA-Q dependency extra, and its CI step.
3. Update backend validation in `src/qdt/config.py`, adapter selection in `src/qdt/adapters/__init__.py`, and references in the README and tests.
4. Preserve useful Aer-only checks from `test_adapters.py`. Remove CUDA-Q comparisons and obsolete assertions rather than deleting unrelated coverage.
5. Regenerate the lockfile and verify installation, core tests, profile tests, lint, and formatting without CUDA-Q.
6. If a concrete active requirement prevents immediate removal, leave it isolated and exclude it from the research campaign and required research checks.

Do not introduce a new multi-framework abstraction to replace CUDA-Q. Aer CPU and GPU should use the same adapter where practical, with explicit execution options.

## 4. M0 — Define the contribution and protocol

### Steps

1. Build a related-work table covering workloads, packing, layout selection, noise, real hardware, HPC, model transfer, total cost, and code availability.
2. Read the initial references and relevant citations. Update the search before freezing the campaign and again before submission.
3. Write one sentence describing the candidate contribution and a verifiable difference from each closest paper.
4. Freeze the minimum scope: QDT, Aer CPU/GPU, one HPC platform, and one accessible superconducting QPU through IBM or IQM. A second hardware provider is optional.
5. Select a primary quality metric, such as balanced accuracy, and an absolute tolerance `delta` expressed in percentage points.
6. Select a primary performance metric: end-to-end inference time under the quality constraint. Retain QPU time and time excluding queueing separately.
7. Define train/validation/test splits, seeds, repetitions, budgets, and rules for handling failed runs.
8. Define baselines before designing the proposed policy.
9. Inventory CPU/GPU resources, scheduler, Aer GPU installation options, QPU access, job/shot limits, and time or monetary budget.
10. Define negative results that would justify narrowing or changing the research claim.

### Expected artifacts

- `docs/research_protocol.md`: hypotheses, RQs, primary metrics, tolerances, and statistical protocol.
- `docs/related_work.md`: comparison table with sources and comparison limits.
- `paper/references.bib`: verified bibliographic references.
- A documented decision on CUDA-Q removal or isolation.

### Completion criteria

- [ ] Candidate novelty is stated without claiming unobserved results.
- [ ] Test data will not select packing, layouts, shots, or hyperparameters.
- [ ] Budget, baselines, and success criteria are explicit.
- [ ] The CPU/GPU and hardware execution stack is fixed, with no CUDA-Q requirement.

## 5. M1 — Separate training, model artifacts, and inference

### Steps

1. Separate model configuration from training and execution configuration. Changing the inference backend, device, shots, or packing must not change the learned tree, selected features, or parameters.
2. Make preprocessing serializable: column order, retained features, training-derived scales, and encoding transformation.
3. Introduce validation data. Fit preprocessing and feature selection only on training data; apply the same transformations to validation and test data.
4. Save a versioned artifact containing the circuit, tree, parameters per node, preprocessing, classes, dataset/split identifiers, and training provenance.
5. Implement loading in a fresh process, checking dimensions, format, and circuit compatibility.
6. Support repeated inference from one artifact without retraining or repeating feature selection.
7. Separate seeds for data splitting, initialization/optimization, shot simulation, and transpilation. Record each seed.
8. Save training traces: objective evaluations, objective values, best solution, timing, and stopping reason per node/candidate/restart.

### Checks

- [x] Training, saving, process termination, and reloading preserve exact-simulation predictions.
- [x] New data passes through the saved preprocessing without label access.
- [x] Changing inference configuration does not modify the model artifact.
- [x] COBYLA budgets are explicitly defined per candidate and node, accounting for restarts.

### Expected artifact

A model trained once and reused across inference comparisons. This is the experimental unit for isolating backend and packing effects.

## 6. M2 — Implement spatial packing and validate Aer CPU/GPU execution

### Steps

1. Define an ordered list of logical evaluations, each with an ID, inputs, parameters, and requested readout.
2. Construct a composite Qiskit circuit with `B` copies on disjoint qubit sets. Support different inputs and parameters per copy, including cases where training shares some parameters.
3. Avoid parameter and classical-bit collisions. Maintain an explicit map from logical circuit to copy, logical qubits, physical qubits, and measurement bits.
4. Reconstruct marginal probabilities from multi-bit measurements. Do not infer result positions from backend bit-string ordering alone.
5. Handle partial batches, `B=1`, empty inputs, and nonzero readout indices. Specify whether the final batch uses a smaller template or inactive slots.
6. Reuse compiled templates when semantically valid, timing initial compilation and parameter binding separately.
7. Keep `request_batch_size` and `packing_factor` separate: one request may contain multiple composite circuits.
8. Start with a common shot count across copies. If different budgets are later required, group compatible requests and account for acquired shots.
9. Expose explicit Aer device and simulation-method selection. Start with `statevector` for ideal references and `density_matrix` for exact noisy references where feasible; distinguish these from finite-shot noisy trajectories.
10. Check available devices and record the effective device and method from configuration and result metadata. An explicitly requested GPU run must fail clearly if unsupported rather than silently becoming a CPU benchmark.
11. Prepare separate, reproducible CPU and GPU installation instructions. GPU-enabled Aer distributions can replace the standard Aer installation: do not assume that installing overlapping distributions together is a valid environment. Verify package compatibility and record the resolved environment.

### Required checks

- [x] Packed and separate marginal probabilities agree in ideal simulation within a stated numerical tolerance.
- [x] Cases include different parameters per copy, intra-copy entanglement, and nontrivial readout.
- [ ] Parameter/bit ordering changes after compilation are covered.
- [x] Results are correctly restored to input order after grouping.
- [x] Finite-shot estimates are compared statistically with exact references, without requiring identical counts across batching strategies.
- [x] No unintended connections exist between logical copies.
- [ ] A GPU smoke test verifies real GPU execution, not merely GPU visibility.
- [ ] CPU and GPU results agree within tolerances appropriate to method, precision, and shots.
- [x] GPU tests are skipped explicitly when hardware is unavailable; CPU CI remains usable without CUDA-Q or GPU packages.

### Completion criteria

A small benchmark reconstructs every logical evaluation for several packing factors in Aer. CPU checks pass, and a compatible GPU environment passes the device and numerical checks.

## 7. M3 — Handle layout, compilation, and noise

### Steps

1. Represent partitions as explicit sets of physical qubits. Check disjointness, connectivity, supported gates, and readout.
2. Implement a simple partition-selection baseline: a feasible fixed set and calibration-based ordering. Reuse existing tools where appropriate.
3. Distinguish the nominal bound `floor(Q/n)` from feasible packing after topology, routing, excluded qubits, and separation constraints.
4. Compile while preserving partition constraints. Inspect final mappings and operations for interactions between copies. Record infeasible cases explicitly.
5. Save calibration data, assigned qubits, transpiler seed/level, native gates, SWAP count, depth, and estimated duration where available.
6. Prepare ideal, controlled independent-noise, and hardware-profile regimes in Aer. Add an explicit crosstalk model only with documented parameters and limitations.
7. Compare each partition alone with that same partition active alongside others. Separately include sequential execution on the best available qubits.
8. Record scheduling, mitigation, twirling, and dynamical-decoupling settings where supported. Match settings or evaluate their effects explicitly.
9. For IBM, use the selected backend's target and native operations. For IQM, use its Qiskit transpilation integration and verify any shared computational resonator or MOVE operations required by the device architecture. Disjoint logical qubits alone do not prove independent physical resource use.

The IQM integration supports Qiskit circuits and provider-specific transpilation; device architectures may require additional routing steps. Check these against the selected device and installed version. See the [IQM Qiskit user guide](https://docs.iqm.tech/iqm-client/user_guide_qiskit.html).

### Checks and limitations

- [x] Copy-to-result correspondence survives transpilation.
- [ ] Poorer-qubit effects are distinguished from simultaneous-execution effects.
- [ ] Noise profiles are not presented as automatic crosstalk measurements.
- [ ] Noisy simulations fit the available qubit and memory budget.
- [x] Shared hardware resources and routing constraints are included in feasibility checks.

Dense simulation of a composite circuit may use memory proportional to `2^(B*n)` for a statevector and `4^(B*n)` for a density matrix. Independent states instead require `B*2^n` or `B*4^n`, before overheads. Use independent-state simulation as an HPC baseline. Monolithic simulation can characterize a limit, but must not be the only comparison. Factorization is not generally exact when noise couples partitions.

## 8. M4 — Make every result traceable and measurable

### Steps

1. Define versioned formats for resolved configuration, events, raw results, and run summaries. JSON/JSONL and CSV are sufficient initially.
2. Assign `campaign_id`, `run_id`, `attempt_id`, `model_id`, `split_id`, `logical_circuit_id`, and `backend_job_id` when applicable.
3. Record the commit and checkout modifications. Use a clean revision for final campaigns or archive the exact diff. Save dependency resolution, versions, OS, CPU/GPU, memory, threads, processes, and Slurm allocation.
4. Save dataset checksums, split indices, and resolved configurations. A seed alone does not stably identify a split if data or libraries change.
5. Measure setup, construction, compilation, binding, submission, waiting, execution, transfer, and postprocessing where observable.
6. Distinguish client measurements, provider measurements, and compiler estimates. Store unavailable values as missing, not zero.
7. Count logical evaluations, composite circuits, requests, shots per copy, and composite-circuit shots. `S` shots of `B` copies produce `B*S` marginal samples, but not `B*S` physical composite-circuit shots.
8. Measure CPU/GPU peak memory and resource allocation; distinguish reserved resources from measured utilization.
9. Preserve raw counts/measurements and metadata needed to reconstruct probabilities and metrics. Exclude credentials and tokens from archived configuration.
10. Separate first use from template/cache reuse, including warm-up in total cost where relevant.
11. Record Aer method, effective CPU/GPU device, precision, thread limits, worker count, GPU assignment, parameter-binding settings, and enabled shot/experiment optimizations. Save raw Aer metadata to substantiate parallelism claims.

### Minimum run summary

| Group | Required fields |
|---|---|
| Identity | Format version and campaign/run/attempt/model/split IDs |
| Provenance | Commit, checkout state, versions, machine, inputs, checksums |
| Execution | Real/simulated backend, CPU/GPU device, method, requested/effective packing, request batch, shots, seeds |
| Hardware | Device, partitions, calibration, compilation and mitigation settings |
| Resources | Circuits, requests, shots, timing, memory, reserved resources |
| Quality | Predictions, metrics, reference, paired differences |
| Status | Success/failure/partial state, reason, job ID, raw-data paths |

### Checks

- [x] Counters match a small example that can be checked manually.
- [x] Overlapping asynchronous intervals are not double-counted.
- [ ] GPU timing waits for relevant operations to finish.
- [x] End-to-end time is measured directly in addition to component timings.
- [x] Metrics can be recomputed from raw results.
- [ ] Metadata distinguishes CPU, single-GPU, and multi-GPU execution.

## 9. M5 — Prepare resumable campaigns and an operational interface

### Steps

1. Introduce a campaign manifest containing models, datasets, splits, seeds, Aer CPU/GPU strategies, hardware settings, packing, shots, resources, repetitions, and budget limits.
2. Resolve the manifest into a finite run list before execution. Show run count, predicted circuits/shots, and resource estimates with assumptions.
3. Implement validation without job submission, checking configuration, artifacts, data, layouts, GPU availability, and backend limits.
4. Implement execution, resumption, and summaries. Skip only completed, validated runs; retain failed and partial attempts.
5. Persist QPU job IDs immediately. Before retrying an uncertain submission, query the existing job to avoid duplicate hardware execution.
6. Use atomic writes and distinct run/attempt outputs. Prevent two workers from claiming the same run; deterministic run-list partitioning is sufficient initially.
7. Enforce concurrency, shot, and time/cost limits. Stop at the configured limits and preserve campaign state.
8. Extend Slurm scripts for Aer CPU/GPU experiments. Assign each worker its intended resources and record affinity, GPU assignment, and thread counts to avoid oversubscription.
9. Implement aggregation and figure generation from completed runs, explicitly reporting missing runs.
10. Update the README and CPU CI with a small training → save → inference → analysis example. Add a separate GPU validation procedure for a compatible runner or cluster.
11. Remove CUDA-Q from required research setup and checks. If its removal was selected, complete the cleanup in Section 3.1 before freezing campaign environments.

### Expected operational interface

These are operations to implement, not existing commands. Choose final command names during implementation.

| Operation | Input | Output |
|---|---|---|
| Train | Training configuration and split | Complete model and training trace |
| Predict | Frozen model, data, execution configuration | Raw measurements, predictions, metrics |
| Plan | Campaign manifest | Resolved run list and resource estimate |
| Validate | Resolved run list | Errors and summary without submission |
| Execute/resume | Run list and persisted state | Results and updated state |
| Analyze | Complete or partial campaign | Tables, confidence intervals, figures |
| Verify artifact | Campaign and associated files | Integrity and completeness report |

### Completion criteria

- [x] A small local campaign can be interrupted and resumed without losing or overwriting results.
- [x] Mocked hardware jobs cover submission, retrieval, failure, and resumption without credentials.
- [ ] The documented CPU example works from a clean checkout with resolved dependencies.
- [ ] GPU installation and validation instructions work on the intended HPC platform.

## 10. M6 — Run the Aer CPU/GPU and HPC pilot

### Steps

1. Fix one or a few models for initial inference comparisons; do not retrain them when changing packing.
2. First measure probability evaluation on a frozen input/parameter list, then complete training and tree inference.
3. Compare sequential Aer, native request/parameter batching, independent simulations in parallel workers, and monolithic packing only where feasible.
4. On CPU, vary thread count and experiment/shot parallelism using compatible settings. On GPU, compare basic execution and applicable parameter/shot optimizations. Validate support in the pinned Aer version; do not assume every optimization applies to every method or noise regime.
5. Start with one CPU worker or GPU. Extend independent-workload execution to 2 and 4 workers/GPUs where available. Separate strong scaling at fixed workload from weak scaling with increasing workload.
6. Vary circuit width and depth independently of dataset size. Make synthetic-data size configurable and record checksums.
7. Compare strategies at equal resources and separately account for added resources. Report CPU/GPU hours and memory alongside speedup.
8. Record warm-up, timing repetitions, and variability. Use fair Aer optimization settings; disable an optimization only for a documented reason or controlled ablation.
9. Estimate final-campaign cost from pilot measurements, including training and analysis preparation.

Aer exposes CPU experiment/shot parallelism, runtime parameter binding, and GPU shot optimizations. Their applicability and compatibility depend on method and build. Use the pinned version's [AerSimulator reference](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator) to select valid combinations. Keep independent-workload distribution separate from optional MPI distribution of a single statevector.

### Proposed initial matrix

| Dimension | Starting values |
|---|---|
| Simulator | Aer CPU and Aer GPU |
| Qubits per circuit | 2, 4, 8 initially; larger widths if needed to investigate GPU crossover |
| Depth | At least two explicitly defined levels |
| Packing | 1, 2, 4, 8, then larger values only where feasible |
| Request batch | At least a small value and one that amortizes overhead |
| Shots | Exact reference; 128, 512, 2048, 8192 for sampling |
| Data | One real dataset and one scalable synthetic dataset |
| Resources | One worker/GPU, then 2 and 4 where available; CPU thread counts recorded |

Do not run the full Cartesian product automatically. Use screening to choose representative overhead-, compute-, and memory-limited cases. GPU execution may lose to CPU for very small circuits; measure the crossover instead of assuming acceleration.

### Pilot decision

- [ ] The strongest HPC comparison does not unnecessarily simulate a huge monolithic state of independent copies.
- [ ] CPU/GPU comparisons use matching logical workloads, methods, precision, and quality requirements, or explicitly report differences.
- [ ] Any benefit to complete training is distinguished from faster circuit evaluation alone.
- [ ] Hardware candidates fit the budget.
- [ ] Hypotheses are updated even if optimal simulation batching is large or CPU wins for small circuits.

## 11. M7 — Run the superconducting QPU pilot

### Steps

1. Implement and validate actual execution for the first accessible provider: IBM Runtime or IQM's Qiskit integration. Keep it distinct from local Aer profile simulation.
2. Check current access, limits, execution modes, and device-specific compilation requirements. Implement the second provider only if needed for the study.
3. Run a small control circuit and verify mappings, counts, shots, and job metadata.
4. Select a few feasible partitions and at least three packing levels, including `B=1`.
5. Execute every partition alone and concurrently with the others, using the same inputs and parameters. For shared-resource architectures, verify that the intended parallel schedule is physically meaningful.
6. Include sequential execution on the best qubits and an efficient request-batching baseline. Improvements must not depend on an artificially slow submission strategy.
7. Alternate and randomize conditions within temporal blocks. Repeat across calibration windows; three windows are an operational starting point, to be adjusted using pilot variability.
8. Save calibration, timestamps, raw counts, layouts, runtime settings, client timings, and provider-reported usage where available.
9. Estimate probability errors, changed QDT decisions, and predictive-quality loss. Measure inter-copy readout correlations where observable, without treating them alone as causal proof of crosstalk.
10. Separate partition heterogeneity, simultaneous-execution effects, and temporal drift.

### Completion criteria

- [ ] A paired hardware comparison exists for isolated versus simultaneous execution on the same qubits.
- [ ] QPU time and end-to-end time are both measured where observable; unavailable provider measurements are identified.
- [ ] Simulation-to-hardware quality loss is quantified.
- [ ] The final-campaign budget is updated using real measurements.
- [ ] Claims are restricted to superconducting hardware and the tested device(s).

If packing degrades quality excessively or overheads remove its benefit, preserve the result. Narrow the contribution to limits or a policy that identifies beneficial cases. Do not retain only favorable hardware runs.

## 12. M8 — Select batching and integrate the QDT workflow

### 12.1 Minimum selection policy

1. Generate a small feasible set of packing/layout/shot combinations.
2. Evaluate them on validation data using frozen models and a declared budget.
3. Reject configurations that violate the quality tolerance relative to the predefined reference.
4. Select the lowest-cost/time feasible configuration. If none meets the constraint, report the failure and apply an explicit fallback.
5. Include selection cost. Report one-time cost and amortized cost over a declared number of inference workloads separately.
6. Freeze the policy before test evaluation.

### 12.2 Tree execution

1. Group samples requiring evaluation at the same node.
2. Consider grouping available work from independent nodes or models only after defining identifiers and result reconstruction.
3. Respect dependencies: a sample's next node depends on its current result. Account for extra work if a strategy evaluates all branches rather than only the selected path.
4. Record requested/effective packing, unused slots, samples per node, tree level, waiting, and total logical evaluations.
5. Measure disagreement by depth and how errors at upper nodes affect final predictions.
6. Distinguish offline inference on an available dataset from online serving. Batch-filling delay is an additional metric only when studying online inference.

### 12.3 Optional extension: decision margins and adaptive shots

QDT decisions use `p >= 0.5`. Small probability errors may preserve a decision far from the threshold while changing one close to it.

Investigate an inexpensive initial evaluation followed by more shots or lower packing for uncertain cases. The policy must use information available during execution, not the ideal probability of the sample. A method using ideal probabilities is a diagnostic oracle, not a deployable algorithm.

Shot uncertainty intervals do not automatically cover hardware bias. Validate the uncertainty criterion empirically and include reevaluation cost. Implement this extension only if pilots indicate room to improve over a validation-selected fixed setting.

### Baselines and ablations

| Comparison | Question |
|---|---|
| `B=1` with efficient requests | What does packing add to a well-implemented isolated execution? |
| Maximum feasible physical packing | Why is filling the device insufficient? |
| Best fixed packing selected on validation data | Does a more complex policy improve over selecting a constant? |
| Proposed policy without layout selection | How much does layout matter? |
| Proposed policy without adaptive shots | How much does shot allocation matter? |
| Basic QDT scheduler versus available-work grouping | How much does workload structure matter? |

Include ablations only for components present in the final implementation.

### Completion criteria

- [ ] The policy uses only information available when a decision is made.
- [ ] Selection, fallback, and reevaluation costs are included.
- [ ] Its utility over the best fixed setting is tested rather than assumed.

## 13. M9 — Freeze and run the final campaign

### Steps

1. Freeze the commit, CPU/GPU environments, protocol, models, splits, primary metrics, and resolved manifest.
2. Separate exploratory and confirmatory results. If the protocol changes after examining test results, disclose this and design an independent evaluation.
3. Select at least two real datasets with different characteristics and one scalable synthetic dataset within budget. Add a simple VQC classifier if the claim remains general.
4. Prepare a model bank. Around ten independent training runs are a starting point; distinguish split and initialization seeds and adjust using pilot variability.
5. Reuse each frozen model across inference conditions. Do not retrain for every `B` when isolating packing effects.
6. Compare ideal and noisy training under declared, comparable budgets. Hardware fine-tuning, if introduced, is a separate condition with its own cost.
7. Schedule paired hardware comparisons within temporal blocks and across calibration windows.
8. Monitor integrity, budget, and job status. Do not alter selection criteria in response to intermediate test outcomes.
9. Preserve failures, timeouts, infeasible configurations, and retry attempts. Apply predefined recovery rules.
10. Check campaign completeness before closing data collection.

### Experimental groups

| Group | Purpose | Workload |
|---|---|---|
| E1 — Correctness | Ideal equivalence, CPU/GPU consistency, accounting | Control circuits |
| E2 — CPU/GPU HPC | Throughput, memory, scaling, training | Aer circuits and QDT |
| E3 — Hardware | Packing and partition effects | Frozen models on IBM or IQM superconducting QPU |
| E4 — Transfer | Ideal/noisy simulation → isolated/packed hardware | Same models across execution targets |
| E5 — Policy | Quality constraint and selection cost | Frozen models with separate validation data |
| E6 — Application | Complete inference and dependencies | QDT; additional VQC if planned |

### Completion criteria

- [ ] Every planned run is completed or has a recorded explanation.
- [ ] Results identify the correct model, split, configuration, execution device, and calibration.
- [ ] Test information has not selected configurations.
- [ ] Raw results are archived before final analysis.

## 14. M10 — Analyze results and produce figures

### Metrics

| Area | Metrics |
|---|---|
| Application quality | Primary balanced accuracy if selected in M0; accuracy, macro-F1, confusion matrix |
| Probabilities | Absolute error against the reference, error distributions, decision disagreement |
| QDT | Disagreement by level, changed paths, visited nodes/circuits, effective packing |
| Training | Objective traces, evaluations, time, final quality, variation across seeds |
| Performance | End-to-end, queue-excluded and QPU time; throughput and latency in the declared regime |
| Resources | Peak memory, CPU/GPU hours, reserved resources, requests, circuits, shots |
| Policy | Selection cost, constraint satisfaction rate, reevaluations, amortization point |
| Reliability | Failed runs, timeouts, repeated jobs, infeasible conditions |

### Statistical analysis steps

1. Validate data and units before aggregation. Preserve execution and queueing components separately.
2. Define the independent comparison unit: model/split and hardware block or window. Thousands of shots from one job are not thousands of independent campaign repetitions.
3. Compute paired quality/performance differences on the same models and inputs. State the confidence-interval method, such as paired bootstrap respecting experimental groups.
4. Report training and hardware-window variability separately where possible. Acknowledge remaining uncertainty when few windows are available.
5. Specify speedup aggregation in advance: per-pair ratios, median, or another stated method. Do not switch methods to favor the outcome.
6. Report confidence intervals for quality differences. A small observed difference alone does not establish non-inferiority.
7. Select settings on validation and evaluate the selection on test data. Do not label the best test-set outcome selected retrospectively as a deployable policy.
8. Report missing data, failures, and sensitivity to protocol-defined exclusions.
9. Distinguish acceleration of the same quantum workload from comparisons with classical baselines. Include efficient Aer CPU/GPU simulation for small circuits. CART contextualizes classification quality, not packing performance.

### Planned figures and tables

| Artifact | Content | Purpose |
|---|---|---|
| Figure 1 | Aer CPU/GPU training → model artifact → superconducting QPU inference | Explain the system |
| Figure 2 | CPU/GPU throughput and memory by strategy, width, and resources | RQ2 |
| Figure 3 | Quality and time versus packing factor | RQ3 |
| Figure 4 | Quality–time/cost frontier with baselines and policy | RQ5 |
| Figure 5 | Ideal/noisy simulation-to-hardware transfer | RQ4 |
| Figure 6 | Timing breakdown for first use and reuse | Identify the source of benefit |
| Figure 7 | Batch occupancy and errors by QDT level | RQ6 |
| Table 1 | Aer environments, hardware platform, versions, calibrations, resources | Reproducibility |
| Table 2 | Main quality/cost comparisons with uncertainty | Central result |
| Table 3 | Ablations, selection cost, and limitations | Explain necessary components |

Adapt the number of figures to the paper's space limit and move details to supplementary material. Every figure must be generated from explicit inputs; do not manually edit numbers or silently select runs.

### Completion criteria

- [ ] Every RQ has an evidence-backed answer or an explicit limitation.
- [ ] Conclusions do not depend on one model or a single favorable hardware run.
- [ ] Tables and figures regenerate from the results archive.
- [ ] Cases where CPU beats GPU or packing is not beneficial remain visible.

## 15. M11 — Write the paper and prepare the artifact

### Writing sequence

1. Draft the method and protocol during M1–M8, separating intended design from measured results.
2. After M10, state at most three contributions grounded in the evidence.
3. Write results and discussion first; then align the introduction, abstract, and title with the findings.
4. Compare explicitly with the closest work on variational concurrency, noise-aware packing, and HPC–QPU workflows. Related work using CUDA-Q is still relevant even though this repository uses Aer.
5. State limitations: classically simulable small circuits, dependence on the tested superconducting device, incomplete noise models, temporal drift, calibration cost, workload availability, and platform count.
6. Distinguish ideal correctness, empirical hardware evidence, and future applicability hypotheses.
7. Check the actual 2027 call for track, format, page limit, anonymization, artifact policy, and deadlines. The 2026 call is preliminary context only; QSYS is a possible systems-oriented track.
8. Verify every number in the manuscript against generated tables.

### Proposed paper structure

1. Introduction: operational problem, gap, measured contributions.
2. Background and related work: batching terminology and prior results.
3. Workflow model: dependencies, resources, quality, cost.
4. Method: implemented packing, layout, selection, and QDT scheduling.
5. Setup: Aer CPU/GPU environments, superconducting QPU, data, models, baselines, statistics.
6. Results organized by RQ.
7. Discussion, limitations, threats to validity.
8. Conclusions and artifact availability.

### Reproducibility package

- [ ] Code revision/tag and exact dependency resolution used for results.
- [ ] Separate CPU, GPU, HPC, and selected-provider setup instructions.
- [ ] Final manifests, frozen models, splits, dataset checksums.
- [ ] Shareable raw results, calibrations, metadata, summaries.
- [ ] Scripts rebuilding every figure/table from collected results without QPU access.
- [ ] A small credential-free CPU example with estimated time/resources.
- [ ] A GPU validation procedure for a compatible environment.
- [ ] Identification of hardware-dependent results and expected replication variability.
- [ ] Verified code license and dataset/calibration redistribution conditions.
- [ ] A persistent archive when the package is ready for publication.

The current `.gitignore` excludes `results`, CSV, PNG, and PDF files. Decide where final artifacts will be archived and verify their inclusion in the package. Local presence does not make them available to reviewers.

## 16. Minimum repository organization at completion

The following structure is a proposal. Create components when their milestone requires them and reuse existing code. A general-purpose experiment framework is not required.

```text
README.md
docs/
  RESEARCH_ROADMAP.md
  research_protocol.md
  related_work.md
  experiment_guide.md
  [CPU/GPU environment instructions]
configs/
  ...                         # Existing Aer and hardware-profile examples
  campaigns/                  # Versioned exploratory and final manifests
src/qdt/
  ...                         # Model, preprocessing, Qiskit adapters, CLI
  [minimal model artifact, packing, and campaign components]
slurm/
  run.sbatch
  [required Aer CPU/GPU scripts]
tests or test_*.py             # Initially retain the project's testing style
analysis/
  [aggregation, statistics, and figure scripts]
paper/
  [manuscript sources and references.bib]
results/                      # Local data; explicit final archiving required
  campaigns/<campaign_id>/
    manifest.json
    models/
    runs/<run_id>/<attempt_id>/
    tables/
    figures/
```

The target research setup does not require CUDA-Q files, dependencies, or configurations. If temporarily retained, they remain outside the supported campaign path.

## 17. Operational sequence for running experiments

### 17.1 Checks already available

These commands belong to the current repository and cover the CPU path. The example simulation may take time; use a new output filename because existing results are not overwritten.

```sh
uv sync --locked
uv run python -m unittest -v test_qdt
uv run ruff check .
uv run ruff format --check .
uv run qdt --config configs/aer_noiseless.json --output results/initial-baseline.json
```

Existing optional checks relevant to the revised scope:

```sh
uv run --extra ibm --extra iqm python -m unittest -v test_qpu
bash -n slurm/run.sbatch
```

CUDA-Q tests and its CI step have been removed; the retained adapter checks run without CUDA-Q. The commands above do not validate GPU execution or real QPU access.

### 17.2 Aer GPU setup to implement and document

1. Inspect the available cluster GPU, OS, driver, and runtime versions.
2. Select a compatible GPU-enabled Aer package or build and pin its environment. Follow official installation guidance; do not assume the CPU environment created by `uv sync --locked` is GPU-ready.
3. Validate the device exposed by Aer and execute a small simulation with GPU explicitly requested.
4. Compare its numerical output with the CPU reference and inspect execution metadata.
5. Run a representative workload and measure memory/timing with proper completion synchronization.
6. Save the environment specification and working Slurm launch instructions.
7. Only then enable the GPU section of the campaign manifest.

A new GPU CLI flag or configuration field is not yet available in the current repository. Document its final name and usage when M2 implements it.

### 17.3 Procedure to enable by M5

1. Install the campaign revision and the appropriate CPU/GPU/provider environment.
2. Prepare data offline and verify checksums and splits.
3. Run correctness checks and a small local campaign.
4. Train or load the frozen models listed in the manifest.
5. Resolve the run list and check feasibility, resources, and budget.
6. Run the CPU/GPU HPC pilot and revise estimates.
7. Run the selected IBM or IQM QPU pilot and verify mappings, accounting, and quality.
8. Freeze the manifest and policy for the final campaign.
9. Launch runs, monitor state/budget, and resume incomplete work according to the protocol.
10. Verify raw-result completeness and integrity.
11. Regenerate analysis, figures, and tables without additional hardware execution.
12. Archive shareable code, data, and instructions with the paper.

## 18. Overall readiness criteria

### Ready for local/HPC campaigns

- [ ] M0–M5 completed.
- [ ] Correct packing and an independent-state Aer baseline available.
- [ ] Loadable models, complete metadata, resumable campaigns.
- [ ] Finite-shot execution matches the declared strategy.
- [ ] Aer GPU execution verified on the target platform, with a reproducible environment.
- [ ] CUDA-Q is not required by the research setup.

### Ready for an extended QPU campaign

- [ ] M6–M7 pilots completed.
- [ ] Layouts and decoding checked through small real hardware jobs.
- [ ] Budget and run estimates updated from measurements.
- [ ] Policy and comparisons selected using only training/validation data.
- [ ] The chosen superconducting device's shared-resource constraints are accounted for.

### Ready for final writing

- [ ] M9 collection closed and M10 analysis verified.
- [ ] Improvements quantified at comparable quality, with uncertainty.
- [ ] Setup/selection costs and unfavorable cases reported.
- [ ] General claims match the tested workloads and execution platforms.

### Ready for submission

- [ ] Manuscript follows the actual 2027 call.
- [ ] Every numerical claim is traceable to results.
- [ ] Another person can regenerate the paper outputs, at least from collected data.
- [ ] Data/hardware access restrictions are documented.

## 19. Initial bibliography and operational sources

This is a starting point for M0, not a complete systematic review. Verify versions, bibliographic metadata, and new work before submission. Software documentation describes available mechanisms; actual experiment settings must be checked against the installed versions.

1. Resch et al., *Accelerating Variational Quantum Algorithms Using Circuit Concurrency*, 2021. [arXiv:2109.01714](https://arxiv.org/abs/2109.01714). Prior work on circuit concurrency for variational algorithms.
2. Mineh and Montanaro, *Accelerating the variational quantum eigensolver using parallelism*, 2023. [arXiv:2209.03796](https://arxiv.org/abs/2209.03796). Parallel VQE evaluation on real hardware.
3. Vandromme and Tsuji, *Post-Variational Quantum Neural Networks on a Hybrid HPC-QC System*, 2025. [DOI:10.1145/3731599.3767540](https://doi.org/10.1145/3731599.3767540). HPC–QPU QML with circuit stacking; post-variational training differs from VQC training.
4. Chatterjee et al., *Parallel Circuit Execution for Scalable Quantum Computation*, 2026. [arXiv:2609.12172](https://arxiv.org/abs/2609.12172). Central comparison for QPU packing, error-aware layouts, and GPU parallelism; its CUDA-Q implementation does not require adopting that framework here.
5. IBM Quantum, [Introduction to Quantum Compute execution modes](https://quantum.cloud.ibm.com/docs/en/guides/execution-modes). Distinguishes job/session/batch execution modes from spatial packing.
6. IEEE Quantum Week, [Call for Technical Papers 2026](https://qce.quantum.ieee.org/2026/call-for-technical-papers/). Preliminary positioning reference, not 2027 requirements or deadlines.
7. Qiskit Aer, [Getting started](https://qiskit.github.io/qiskit-aer/getting_started.html). CPU/GPU installation and optional build capabilities.
8. Qiskit Aer, [AerSimulator API reference](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator). Device selection, simulation methods, and execution options.
9. IQM, [Qiskit on IQM user guide](https://docs.iqm.tech/iqm-client/user_guide_qiskit.html). Qiskit circuit execution and device-specific transpilation.
