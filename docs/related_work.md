# Related work — initial comparison

This initial table summarizes the sources identified during planning. It is not a systematic review. Check full-text methods, code availability, and follow-up work before making novelty claims or freezing final experiments.

| Work | Relevant mechanism | Evidence / scope | Proposed distinction to verify |
|---|---|---|---|
| [Resch et al. (2021)](https://arxiv.org/abs/2109.01714) | Concurrent copies for variational sampling | Variational-algorithm acceleration | Frozen-model transfer and application-quality-constrained inference selection |
| [Mineh and Montanaro (2023)](https://arxiv.org/abs/2209.03796) | Parallel VQE circuits | Real Rigetti hardware and optimization | QDT dependencies and inference after simulator training |
| [Vandromme and Tsuji (2025)](https://doi.org/10.1145/3731599.3767540) | Circuit stacking in HPC–QPU QML | Post-variational workload on Fugaku and IBM | Variationally trained frozen models and explicit packing/shot selection |
| [Chatterjee et al. (2026)](https://arxiv.org/abs/2609.12172) | Error/topology-aware partitions and GPU circuit parallelism | QPU benchmarks and Hamiltonian observable estimation | Predictive-quality constraints, selection overhead, transfer, and QDT scheduling |

A different SDK or dataset alone is not a contribution. The closest comparison is the 2026 parallel-execution work. Reproduce or implement efficient fixed baselines before claiming a benefit for selection. Real-hardware results from other superconducting devices are relevant even when their software stack differs from ours.

Pending review fields: exact partition-selection algorithm, crosstalk controls, shot allocation, timing boundaries, baseline optimizations, statistical units, available scripts/data, and any direct treatment of simulator-trained classifier inference.
