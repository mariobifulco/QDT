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
