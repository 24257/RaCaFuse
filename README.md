# SEER

This repository contains the implementation of SEER for medication recommendation, including data preparation, training, and evaluation utilities.

## Public package boundary

This repository excludes clinical prediction arrays, trained
checkpoints and frozen graph/mask binaries. Keep those files outside the public
folder and use only artifacts authorized under the source data agreements.
See [security and local evaluation](docs/security.md) for paths and trust settings.

For the public synthetic self-check:

```bash
python scripts/verify_public.py
```

This checks software behavior; it does not reproduce clinical results.

## Overview

SEER combines two learned branches:

1. a longitudinal medication-recommendation branch with a shared three-dataset training protocol and a retrieval-readout refinement stage;
2. a SEER auxiliary branch bundled in `src/seer_auxiliary_branch/` and evaluated through `baselines/seer_auxiliary_adapter.py`.

The released final prediction is produced by fixed validation-selected fusion:

- ranking score: `0.3 * z(longitudinal score) + 0.7 * z(logit(AUXILIARY probability))`;
- prescription count: `round(0.4 * calibrated longitudinal count + 0.6 * calibrated AUXILIARY count) + 1`;
- DDI-aware greedy decode: penalty `0.04`, candidate window `60`.

Count calibration is fit on training data only. To avoid shipping hundreds of megabytes of intermediate train components, the exact fitted affine calibrators are packaged in `configs/count_calibration.json`. They were independently checked against the original train-only components and reproduce the released final arrays exactly.

## Expected Paper Results

The primary acceptance target is the three-seed arithmetic mean ± sample standard deviation (`ddof=1`), reported in percent:

| Dataset | Jaccard | F1 | PRAUC |
|---|---:|---:|---:|
| MIMIC-III | 57.73 ± 0.09 | 72.48 ± 0.08 | 81.77 ± 0.05 |
| MIMIC-IV | 53.20 ± 0.06 | 68.41 ± 0.03 | 77.97 ± 0.06 |
| eICU | 54.52 ± 0.16 | 68.89 ± 0.13 | 82.58 ± 0.20 |

The nine authorized local prediction files, supplied separately, are sufficient to recompute and regression-check this table using authorized local files without a GPU:

```bash
python scripts/reproduce_results.py --predictions-root /path/to/authorized/predictions/seer
```

The checker independently recomputes Jaccard, F1, PRAUC and count MAE from all nine NPZ files, checks each seed against the packaged CSV, recomputes arithmetic means and sample SDs, checks `FINAL_TEST_RESULTS.json`, and fails if any of the displayed paper targets changes.

For a convenience rendering of the main table:

```bash
python scripts/reproduce_table1.py
```

For per-seed values:

```bash
python scripts/reproduce_table2.py
```

## Real-Data Fresh-Training Acceptance

In addition to artifact replay, the release was exercised end-to-end on **MIMIC-III seed1203** using the protected real processed data and fresh random initialization. This is a one-seed acceptance run, not a claim that all three datasets and all three seeds were freshly retrained.

The acceptance covered:

- longitudinal Stage A from random initialization through 60 epochs, with the final-lineage epoch55 checkpoint reproduced exactly;
- longitudinal Stage B initialized from the fresh epoch55 Stage A checkpoint, with all 121 model-state tensors matching the historical final Stage B weights exactly;
- the SEER auxiliary branch trained for the full 40 epochs, with best epoch 39 and historical training metrics reproduced;
- train-only count calibration refit from fresh branch outputs;
- final fusion and DDI-aware decoding on the held-out MIMIC-III seed1203 test set.

The fresh train-only calibrators recovered the released values for seed1203:

```text
longitudinal: slope=0.95, intercept=-0.35
AUXILIARY expert:   slope=0.90, intercept=-1.15
```

The final fresh seed1203 metrics were:

```text
Jaccard = 0.5778873796077955
F1      = 0.7255168981185356
PRAUC   = 0.8178497618992944
```

The final `y_true`, discrete recommendation set `y_pred`, `predicted_count`, `patient_ids`, and `visit_ids` matched the released seed1203 prediction arrays exactly. The only observed non-bit-exact quantity was the AUXILIARY test probability/ranking path, with a maximum AUXILIARY probability difference of about `1.67e-6`; it did not change the final recommendation set or predicted cardinality.

## Requirements

The historical experiment environment recorded in the workspace used Python 3.9 and a CUDA-enabled PyTorch runtime. A representative historical software stack was:

- Python 3.9.23
- PyTorch 2.6.0 + CUDA 11.8
- NumPy 2.0.1
- pandas 2.3.2
- SciPy 1.13.1
- scikit-learn 1.6.1
- dill 0.4.0
- matplotlib 3.9.4

The historical Python 3.9 environment is no longer supported. New installations should use Python 3.11 or newer and a security-patched PyTorch build. `requirements.txt` and `environment.yml` are installation specifications, not fully locked environments; GPU reproducibility must be validated separately.

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate seer
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


## Dataset Preparation

Raw MIMIC-III, MIMIC-IV, and eICU clinical data are not redistributed. Prepare the processed files locally under:

```text
data/
├── mimic-iii/
├── mimic-iv/
└── eicu/
```

Each processed dataset used by the training/evaluation stack contains the project equivalents of:

```text
records_final.pkl
voc_final.pkl
ehr.pkl / ehr_adj_final.pkl
ddi_A_final.pkl
```

### MIMIC-III

```bash
python scripts/preprocess.py \
  --dataset mimic-iii \
  --raw-root /path/to/mimic-iii \
  --output-root data/mimic-iii
```

### MIMIC-IV

For the literature-aligned cohort, use the locally obtained Carmen `data_final_4.pkl` as the source of truth:

```bash
python scripts/preprocess.py \
  --dataset mimic-iv \
  --source /path/to/data_final_4.pkl \
  --output-root data/mimic-iv
```

Optional aligned EHR/DDI matrices can be supplied with `--ehr-source` and `--ddi-source`.

The stock MIMIC-IV raw release alone does not reconstruct the exact literature cohort used here. Do not silently replace the Carmen-aligned processed frame with a different raw replay and call it the same cohort.

### eICU

```bash
python scripts/preprocess.py \
  --dataset eicu \
  --raw-root /path/to/eicu-crd \
  --output-root data/eicu
```

The reference eICU DDI matrix is a zero placeholder. Therefore eICU DDI is reported as **N/A**, not as evidence of drug-interaction safety.

See `data/README.md` and `docs/datasets.md` for details.

## Quick Start

### 1. Recompute reference metrics from authorized local predictions

The nine prediction files must be supplied separately under their access terms; no GPU is required:

```bash
python scripts/reproduce_results.py --predictions-root /path/to/authorized/predictions/seer
```

### 2. Run all release-level verification utilities

```bash
python scripts/reproduce_all.py
```

This defaults to the public synthetic check. To rerun all nine final inferences after configuring authorized local checkpoints, frozen inputs, clinical data and the explicit legacy-loading trust setting (see `docs/security.md`):

```bash
python scripts/reproduce_all.py --with-inference --predictions-root artifacts/local_predictions --cuda 0
```

### 3. Inspect a final inference command without opening protected data

```bash
python scripts/evaluate.py --dataset mimic-iii --seed 1203 --dry-run
python scripts/evaluate.py --dataset eicu --seed 1203 --dry-run
```

`--dry-run` checks bundled runtime and command construction; it does not check external checkpoint existence or open clinical data.

## Shared Longitudinal Training Protocol

The public from-scratch longitudinal protocol is one shared configuration across MIMIC-III, MIMIC-IV and eICU:

| Parameter | Value |
|---|---:|
| hidden dimension | 256 |
| batch size | 32 |
| visit setting | 3 |
| weight decay | 1e-6 |
| BCE weight | 0.7 |
| soft-Jaccard weight | 0.2 |
| pairwise-rank weight | 0.02 |
| epochs | 60 |
| learning rate | 2e-4 |
| early-stop patience | 10 |
| temporal encoder | Mamba |
| medication temporal encoder | GRU |
| patient fusion | sum |
| clinical feature mode | none |
| EHR graph | directed weighted |

The values are stored in `configs/final_model.json` and consumed by `scripts/protocol.py`.

For eICU, training additionally passes `--disable_procedure_branch` because eICU does not provide the same procedure input structure. This is a structural dataset override, not dataset-specific hyperparameter tuning.

Training is performed only on train/validation manifests under `configs/splits/train_validation/`; `--validation_only_no_test` prevents test-set use during selection.

### Train Stage A and Stage B

```bash
python scripts/train.py --dataset mimic-iii --seed 1203
python scripts/train.py --dataset mimic-iv  --seed 1203
python scripts/train.py --dataset eicu      --seed 1203
```

Outputs are written under `artifacts/retrained/<dataset>/seed<seed>/` by default and do not overwrite the released checkpoints.

Stage A trains the shared longitudinal backbone. Stage B initializes from Stage A and trains the retrieval-readout path with the fixed settings stored in `configs/final_model.json`.

To inspect commands only:

```bash
python scripts/train.py --dataset mimic-iii --seed 1203 --dry-run
```

### MIMIC-III Final-Checkpoint Lineage

The shared protocol above is the public cross-dataset training protocol and should remain described as one common configuration. Separately, the **released MIMIC-III final checkpoint** came from a later validation-only corrective/selection lineage used to freeze the final paper checkpoint. For MIMIC-III seed1203, that lineage used Stage A loss coefficients `BCE=1.0`, `soft-Jaccard=0.3`, `pairwise-rank=0.02` with `weight_decay=1e-4`, trained Stage A for 60 epochs, selected the fixed epoch55 teacher using validation-only evidence, and trained the Stage B retrieval-readout from that teacher.

This later final-checkpoint lineage must not be rewritten as evidence that the original shared three-dataset protocol used different batch sizes or per-dataset training schedules. In particular, the public shared training batch size remains **32 for all three datasets**.

## SEER Auxiliary Branch Training

The final SEER model also uses `seer_auxiliary_branch.model`. This is the checkpoint of the **SEER auxiliary branch**, used by final SEER fusion. The checkpoint is supplied separately through an authorized local path.

The required runtime is bundled in:

```text
src/seer_auxiliary_branch/
```

The adapter is:

```text
baselines/seer_auxiliary_adapter.py
```

Retraining command:

```bash
python scripts/train_seer_auxiliary.py --dataset mimic-iii --seed 1203 --cuda 0
```

The historical expert settings represented by the release are `dim=64`, `dropout=0.7`, `lr=5e-4`, `coef=2.5`, `target_ddi=0.06`, `epochs=40`, `early_stop=30`, `target_start_visit=1`.

For exact historical MIMIC-III reproduction, the adapter preserves the original SetTransformer MAB path and the historical CPU/Python DDI-rate semantics used by the locally adapted training code instead of silently substituting a numerically different optimized attention path.

`SEER_AUXILIARY_SRC` / `--seer_auxiliary-src` may be used only as an explicit debugging override. The default release path is fully local.

## Released-Checkpoint Final Inference

For one dataset/seed:

```bash
python scripts/inference.py --dataset mimic-iii --seed 1203 --cuda 0
```

or equivalently:

```bash
python scripts/evaluate.py --dataset mimic-iii --seed 1203 --cuda 0
```

The released-checkpoint inference protocol is deliberately distinct from from-scratch training configuration:

- MIMIC-III and MIMIC-IV final inference use the paper protocol without an explicit final split manifest;
- eICU final inference uses the paper protocol **plus** the packaged filtered final split manifest;
- the eICU training-only `--disable_procedure_branch` flag is not injected into released-checkpoint final inference;
- MIMIC-III final component export uses runtime `batch=64`; this is an inference/export setting, **not** a MIMIC-III training batch size;
- final-checkpoint runtime/lineage loss arguments are kept separate from the shared public training protocol in `scripts/protocol.py`;
- final count fusion uses the frozen train-only affine calibrators in `configs/count_calibration.json`.

When `scripts/evaluate.py --refit-calibration` is used with fresh checkpoints, both branch calibrators are refit from train-only outputs. `--calibration-data-root` applies to the train-only export for both branches, and the target/test retrieval cache is kept separate from the calibration retrieval cache.

This separation prevents export/runtime flags or later experimental branches from being mistaken for public training hyperparameters.

## Diagnostic Ablations

`run_ablation.py` is a transparent final-fusion diagnostic utility. It does **not** relabel arbitrary settings as manuscript-reported ablations.

Given target components from both branches and an aligned DDI matrix:

```bash
python scripts/run_ablation.py \
  --dataset mimic-iii \
  --seed 1203 \
  --longitudinal /path/to/longitudinal_test.npz \
  --seer_auxiliary /path/to/seer_auxiliary_predictions.npz \
  --ddi data/mimic-iii/ddi_A_final.pkl \
  --score-long-weight 0.3 \
  --count-long-weight 0.4 \
  --count-offset 1 \
  --ddi-penalty 0.04
```

## Baselines

Third-party comparison implementations are not copied wholesale into this repository. Their locally recorded source links are listed in `configs/baseline_registry.json` and `baselines/README.md`.

Inspect the registry:

```bash
python scripts/run_baselines.py --list
```

The bundled SEER auxiliary branch is the only comparison/expert runtime intentionally included because it is part of the final SEER computation graph.

## Synthetic Smoke Fixture

No clinical data is needed to generate the tiny synthetic metric fixture:

```bash
python data/make_synthetic_fixture.py \
  --output data/sample/synthetic_predictions.npz
```

The synthetic fixture is for software smoke tests only and has no clinical or paper-result meaning.

## Repository Layout

```text
SEER-code/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── src/                 # SEER model/training/fusion implementation
├── scripts/             # preprocessing, training, inference, reproduction
├── configs/             # shared protocol, fusion, calibrators, splits
├── baselines/           # local AUXILIARY-derived expert adapter and registry support
├── data/                # data instructions and synthetic fixture tooling
├── checkpoints/         # documentation only; weights excluded
├── predictions/         # documentation only; clinical arrays excluded
├── results/             # per-seed and aggregated release results
├── tests/               # CPU/no-protected-data regression tests
└── docs/                # model, data, reproducibility, troubleshooting
```

## Tests

The release tests use the Python standard-library `unittest` runner:

```bash
python -m unittest discover -s tests -v
```

A stronger release smoke is:

```bash
python scripts/reproduce_results.py --predictions-root /path/to/authorized/predictions/seer
python -m compileall -q src scripts baselines tests data
python -m unittest discover -s tests -v
python scripts/train.py --dataset mimic-iii --seed 1203 --dry-run
python scripts/train_seer_auxiliary.py --dataset mimic-iii --seed 1203 --dry-run
python scripts/evaluate.py --dataset mimic-iii --seed 1203 --dry-run
python scripts/evaluate.py --dataset eicu --seed 1203 --dry-run
```

## Hardware and Runtime

Artifact verification and unit tests are CPU-friendly. Full preprocessing, training and nine-seed inference are substantially more expensive and depend on protected clinical datasets. The historical and fresh-acceptance runs used a CUDA-enabled PyTorch environment; the completed fresh acceptance covered **MIMIC-III seed1203 only** and should not be generalized into a claim that all nine dataset/seed combinations were freshly retrained.

## Reproducibility Boundary

The release now has two separately validated claims. First, `scripts/reproduce_results.py` exactly reproduces the stated three-dataset, three-seed paper table from the nine authorized local final prediction artifacts. Second, a protected-data real-GPU acceptance run freshly retrained **MIMIC-III seed1203** through both learned branches, refit train-only calibration, and reproduced the released discrete final recommendation and cardinality arrays exactly. The repository does not claim that MIMIC-IV, eICU, seed1204, or seed1205 were freshly retrained during this acceptance pass.
