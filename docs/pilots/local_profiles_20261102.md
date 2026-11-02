# CPU noise/profile pilot — November 2, 2026

All quantum computations ran locally on CPU in Aer. IBM supplied calibration metadata only; no quantum circuit was submitted to IBM or IQM. No GPU or Slurm job was launched.

## Design

Four frozen models were trained from clean commit `82686fb`: ideal/noisy training, each with seeds 0 and 1. All use the same two-qubit depth-one QDT, 40 training samples, 16 objective evaluations, and fixed data split. Six campaigns cover controlled noise, an IBM Fez calibration snapshot, and the bundled IQM Garnet profile. Each campaign evaluates two models at packing 1/2/4 with exact probabilities or 512 shots, on the first 16 validation rows. The test set was not evaluated.

All 72 runs completed without failed campaign attempts and passed identity, checksum, configuration, input-index, and metric checks. Every run executed 16 logical circuits on CPU. Training and collection checkouts were clean. PDF/PNG figures were generated; IBM and IQM examples were visually inspected.

## Observations

Changes count predicted-label differences against exact B1 on the same frozen model and rows. Ratios compare median measured cold end-to-end simulator times against exact B1; they are not QPU speedups.

| Campaign | Mean exact B1 balanced accuracy | Changes at exact B2 | Changes at exact B4 | CPU time ratio B1/B2 | CPU time ratio B1/B4 |
|---|---:|---:|---:|---:|---:|
| ideal-custom | 0.979167 | 0 | 0 | 1.227 | 0.831 |
| ideal-ibm | 0.979167 | 0 | 0 | 1.305 | 1.272 |
| ideal-iqm | 0.979167 | 0 | 0 | 2.132 | 1.643 |
| noisy-custom | 0.979167 | 0 | 0 | 1.248 | 0.842 |
| noisy-ibm | 0.979167 | 0 | 0 | 1.290 | 1.256 |
| noisy-iqm | 0.979167 | 0 | 0 | 2.245 | 1.692 |

Exact packed predictions were unchanged on this small prefix. This does not establish robustness across datasets, training seeds, near-threshold inputs, or hardware noise. Shot-based variation occasionally changes the observed score; it is not evidence that sampling improves the model.

Packing four copies was slower than B1 in the controlled density-matrix cases. Profile-case ratios include substantial snapshot reconstruction/compilation and first-use costs. Timing order was fixed, there was only one repetition per setting, and two models provide very weak inferential evidence. These numbers validate the collection pipeline and motivate larger randomized, warm/cold, and memory-controlled studies.

## Matched-partition controls and numerical investigation

Four root-node controls compare each packed marginal to isolated execution on the same physical partition/input, avoiding a layout-quality confound. All four preserve threshold decisions and pass an explicitly selected absolute tolerance of `1e-8`. The controls use diagnostic commit `184c001` and record their script checksum.

| Profile | Packing | Maximum absolute difference | Minimum isolated decision margin | Threshold disagreements |
|---|---:|---:|---:|---:|
| IBM Fez snapshot | 2 | 5e-10 | 0.048179 | 0 |
| IBM Fez snapshot | 4 | 6.84e-10 | 0.048596 | 0 |
| iqm:garnet | 2 | 3.89e-16 | 0.047644 | 0 |
| iqm:garnet | 4 | 1.28e-15 | 0.047798 | 0 |

The original strict `1e-10` IBM check failed at approximately `5e-10`; its failed artifact is preserved. Native operation sequences and parameters matched exactly. Disabling fusion or reducing the zero threshold did not change that discrepancy. Reconstructed channels had a worst trace-preservation residual around `6.3e-15`. Its precise simulator-level cause remains unresolved. The explicit `1e-8` engineering tolerance is on the scale of the SDK channel-validation tolerance, not a formal expectation-value error guarantee. Ideal and simple-noise unit tests retain tighter tolerances.

## Profile limitations

The IBM snapshot is frozen by byte checksum and records its retrieval/calibration timestamps. Its specialized `xslow`, `measure_2`, `reset_2`, `measure_reset`, and `measure_reset_2` instructions are explicitly excluded from the static target. Standard calibrated gates and independent noise remain represented. IQM uses its library-provided Garnet profile. Neither case measures simultaneous hardware crosstalk, temporal drift, or QPU billing/runtime.

## Artifacts and resumption

Raw models, calibration snapshot, plans, attempts, counts, controls, and figures are in ignored `results/local-profile-pilot/` (about 43 MB). The committed JSON summary includes campaign/model/profile/control/figure checksums. Include the complete raw directory in a publication archive after checking redistribution conditions. Commands, manifest paths, and recovery rules are in `docs/experiment_guide.md`.

Next CPU work: larger validation sets and split/model banks, wider/deeper workloads, randomized timing repetitions, warm-template measurements, per-run memory measurements, and explicit node-path traces. Establish a useful quality–cost frontier before claiming a research contribution. GPU and real-QPU execution remain deferred by the user.
