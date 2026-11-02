# Research protocol — exploratory version 0.1

Date: November 2, 2026. Scope: QDT, Qiskit Aer CPU/GPU, and one superconducting IBM or IQM QPU. CUDA-Q is removed. This is an implementation and pilot protocol, not a preregistration of a final hardware campaign.

## Candidate contribution

Select spatial packing, layout, and shots to minimize end-to-end inference time while preserving predictive quality of a frozen model trained in simulation. Compare with efficient unit packing, maximum feasible packing, and the best validation-selected fixed setting. Novelty remains provisional until the related-work review and pilots are complete.

## Data and models

Use stratified train/validation/test partitions of 60/20/20 percent. Fit scaling and feature selection on training only. Preserve split indices and dataset checksums. Use validation to choose execution settings; do not inspect test results during selection. Treat legacy train/test-only runs as exploratory and label them separately.

Start with Iris for smoke tests, Breast Cancer and Wine for real-data pilots, and scalable synthetic data for throughput. Start with the existing two-qubit circuit, then add four/eight-qubit circuits. Run ten model initializations in final experiments if the pilot supports this budget. Keep split, optimizer, simulator, and transpiler seeds distinct and recorded.

Save models once and reuse them across inference conditions. Compare ideal and noisy training as separate conditions. Record objective evaluations and stopping reasons. Parameters and preprocessing must not change when packing, shots, or device changes.

## Primary metrics and references

Primary quality: balanced accuracy. Initial non-inferiority tolerance: 0.01 absolute (one percentage point), subject to revision on exploratory validation data before freezing the final protocol. Reference: the same frozen model's isolated hardware execution for packing effects; exact ideal Aer for simulation-to-hardware transfer. Keep these references distinct.

Primary performance: measured end-to-end inference time at the chosen quality constraint. Also report setup, selection, queue-excluded time, provider-reported QPU time when available, throughput, peak memory, CPU/GPU hours, requests, composite circuits, and shots. Missing provider measurements are null, never zero.

## Pilot design and budgets

CPU pilots may run locally; cap each initial smoke run at 40 training samples, depth one, and eight objective evaluations per candidate/node. Use a 20 percent validation fraction for new artifact workflows. Start packing at 1/2/4 and exact probabilities; investigate finite shots afterward. A local pilot is not an HPC scaling result.

No final GPU or hardware campaign is frozen yet. Before hardware submission, record device access, feasible partitions, maximum circuits/shots, concurrency, time/cost budget, and calibration-window schedule in the campaign manifest. Hardware resource availability and final budget remain external inputs, not assumed facts.

## Statistics and failure handling

Use paired comparisons of the same model, split, input set, and temporal hardware block. Report confidence intervals on quality differences and speedups. Bootstrap independent model/block units, not individual shots as independent experiment repetitions. Separate training-seed variation from calibration-window variation. Start with at least three hardware windows, then adjust using pilot variability.

Preserve failed, partial, and infeasible attempts. Retry only explicitly recoverable work and preserve job IDs before retrying. No silent exclusion or replacement of unfavorable runs. Do not infer non-inferiority from a small point estimate alone.

## Decisions before a confirmatory campaign

- Verify novelty against full texts and new related work.
- Select an available GPU platform and one IBM/IQM superconducting device.
- Freeze data, model bank, metric tolerance, paired analysis, manifest, code, dependencies, and budgets before test evaluation.
- If CPU outperforms GPU or packing loses quality/cost efficiency, report the crossover or limit rather than suppressing the condition.
