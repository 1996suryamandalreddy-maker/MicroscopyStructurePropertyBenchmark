# EM Benchmark

Minimal benchmarks for active learning of structure-property relationships in electron microscopy.

The project is intentionally small and modular: a configurable active-learning loop, simple datasets, swappable representations/models/acquisition functions, scalarizer rewards, and saved benchmark artifacts that are easy to inspect or extend.

## Quick Start

```bash
uv sync --extra dev
uv run em-benchmark --config configs/pca_gp_ei.yaml
```

To run on the real DTMicroscope STEM test file, first place the file at `data/raw/test_stem.h5`:

```bash
curl -L https://github.com/pycroscopy/DTMicroscope/raw/utk/data/STEM/SI/test_stem.h5 -o data/raw/test_stem.h5
uv run em-benchmark --config configs/stem_pca_gp_ei.yaml
```

To compare the main methods and random baselines for 100 BO steps and write a single CSV:

```bash
uv run em-benchmark-sweep --config configs/stem_all_methods.yaml
```

## Current Shape

```text
dataset -> representation -> model -> acquisition -> reward/metric -> log
```

## Implemented Pieces

Datasets:

- `synthetic`: small microscopy-like toy dataset for fast tests
- `stem_h5`: real STEM H5 file with overview image, spectrum image, and energy axis

Representations:

- `pca`: `sklearn.decomposition.PCA` over flattened patches
- `patches`: raw image patches for DKL

Models:

- `gpytorch_gp`: exact GP baseline from `gpytorch`
- `dkl`: lightweight deep-kernel GP with a CNN feature extractor and variational GP

Acquisition:

- `expected_improvement`: BoTorch `LogExpectedImprovement`
- `upper_confidence_bound`: BoTorch `UpperConfidenceBound`; alias for the MU-style high-beta exploration baseline
- `beacon`: lightweight BEACON-style value plus novelty score
- `random`: simple baseline

Rewards/scalarizers live in `src/em_benchmark/rewards.py`:

- `dipole`: 0.35-0.55 eV
- `edge`: 0.60-0.75 eV
- `bulk`: 0.80-1.00 eV
- `zero`: zero scalarizer for control/debug runs

## Configs

- `configs/pca_gp_ei.yaml`: synthetic PCA-GP with BoTorch expected improvement
- `configs/dkl_ei.yaml`: DKL model with BoTorch expected improvement
- `configs/dkl_beacon.yaml`: DKL model with BEACON-style acquisition
- `configs/stem_pca_gp_ei.yaml`: real STEM H5 PCA-GP with expected improvement
- `configs/stem_dkl_mu.yaml`: real STEM H5 DKL with high-beta UCB/MU acquisition
- `configs/stem_dkl_beacon.yaml`: real STEM H5 DKL with BEACON-style acquisition
- `configs/stem_all_methods.yaml`: real STEM H5 sweep over PCA-GP/EI, PCA-GP/MU, PCA-GP/random, DKL/EI, DKL/MU, DKL/BEACON, and DKL/random for 100 BO steps

## Outputs

When `output.enabled: true`, runs write to `outputs/<run_name>_seed<seed>_<timestamp>/`:

- `predictions_BO_step<N>.png`: original image, predicted mean, predicted variance, true scalarizer
- `predictions_BO_step<N>.pkl`: per-step arrays and metrics
- `Active_learning_statistics.pkl`: acquisition order, seed indices, traces, features, coordinates
- `AL_traj.png`: active-learning trajectory over the image
- `run.log`: human-readable run/debug log
- `training_log.jsonl`: structured per-run and per-step logs, including training loss traces and model diagnostics
- `checkpoints/model_step<N>.pt`: optional model checkpoint when `checkpoint.save_model: true`
- `checkpoints/latest.pt`: optional latest model checkpoint

Checkpoint options:

```yaml
checkpoint:
  save_model: true
  load_model_path: outputs/some_run/checkpoints/latest.pt
```

`load_model_path` warm-starts model weights before each BO-step training call. It is useful for debugging and model-state reuse; BO state resume is intentionally separate and not implemented yet.

Sweep runs write a compact CSV with one row per method per BO step:

```text
method,step,selected_index,mse,mae,nlpd,coverage,mean_prediction,mean_variance,loss_initial,loss_final
```

Sweep runs also write:

- `outputs/stem_all_methods_100_steps.log`: human-readable sweep log
- `outputs/stem_all_methods_100_steps_log.jsonl`: structured sweep log with method start/end/error events

Generate comparison plots from a sweep CSV:

```bash
uv run python scripts/postprocess_csv.py --csv outputs/stem_all_methods_100_steps.csv
```

The notebook exports in `llm-context/` are reference material for the DKL-EI, DKL-BEACON, and plotting/metric workflows.

## Contributing A Method

Add the method behind a small interface:

- representation transforms raw patches into model inputs
- model exposes `fit(X, y)` and `predict(X)`
- acquisition scores unacquired candidate points
- runner handles the active-learning loop and logging

Start with a config in `configs/` and a smoke test in `tests/`.
