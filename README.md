# SEER

Implementation of SEER for medication recommendation, including data preparation,
training, and evaluation utilities.

SEER combines a longitudinal branch with a medication recommendation auxiliary
branch. Stage A trains the longitudinal backbone; Stage B refines its retrieval
readout. Final predictions combine both branches with count calibration and
DDI-aware decoding. Model settings are in [configs/final_model.json](configs/final_model.json).

## Installation

Use Python 3.11 and install the dependencies:

```bash
conda env create -f environment.yml
conda activate seer
```

Alternatively, install [requirements.txt](requirements.txt) in a Python environment:

```bash
python -m pip install -r requirements.txt
```

GPU training requires a compatible CUDA-enabled PyTorch installation. The dependency
files are not fully locked; full GPU reproducibility in a new environment must be
validated separately.

## Data preparation

Obtain MIMIC-III, MIMIC-IV, and eICU through their official access procedures.
Clinical data, individual predictions, trained weights, and frozen graph inputs
are excluded from this repository. See [data/README.md](data/README.md) for dataset
preparation, required local files, and trusted legacy-file loading.

The included synthetic fixture can be checked without clinical data:

```bash
python scripts/verify_public.py
```

## Training

After preparing authorized local data and configuring legacy loading as described
in the data guide, train the two branches:

```bash
python scripts/train.py --dataset mimic-iii --seed 1203 --data-root /path/to/data/mimic-iii
python scripts/train_seer_auxiliary.py --dataset mimic-iii --seed 1203 --data-root /path/to/data/mimic-iii --cuda 0
```

Supported datasets are `mimic-iii`, `mimic-iv`, and `eicu`; reference seeds are
`1203`, `1204`, and `1205`. Outputs default to `artifacts/retrained/` and
`artifacts/retrained_seer_auxiliary/`. Add `--dry-run` to inspect generated commands.

Shared longitudinal training uses batch size 32 and the settings in
[configs/final_model.json](configs/final_model.json). eICU disables the procedure
branch during longitudinal training. Training uses the train/validation manifests
in `configs/splits/train_validation/`.

Reference MIMIC-III weights came from a separate validation-selected run using
Stage A BCE/soft-Jaccard/ranking weights `1.0/0.3/0.02`, weight decay `1e-4`, and
the epoch55 teacher from 60 epochs. Reference inference uses batch size 64.
These differ from the default shared training configuration; running the default
training command alone is not a claim of exact reference-weight reproduction.

## Evaluation

Configure the authorized external artifact tree described in the data guide, then run:

```bash
python scripts/evaluate.py --dataset mimic-iii --seed 1203 --data-root /path/to/data/mimic-iii --cuda 0
```

`--cuda N` selects visible GPU N for both branches and calibration; `--cuda -1`
selects CPU. `scripts/inference.py` provides the same entry point. `--dry-run`
checks command construction without loading clinical data or checking external
checkpoint existence.

Use `--stage-a`, `--stage-b`, and `--auxiliary-checkpoint` to supply alternative
weights, with `--refit-calibration` to fit counts from their training outputs.
`--calibration-data-root` optionally selects a separate calibration data root.
The current evaluation entry point still requires the external frozen graph/mask
inputs, including when alternative weights are supplied.

Reference evaluation uses the protocol in [scripts/protocol.py](scripts/protocol.py)
and train-only calibrators in [configs/count_calibration.json](configs/count_calibration.json).
Content-based graph/retrieval caches are rebuilt when input data or vocabularies
change; legacy caches are not reused and the cache directory must be writable.

To recompute reference metrics from nine authorized local prediction files:

```bash
python scripts/reproduce_results.py --predictions-root /path/to/private-root/predictions/seer
```

## Reference results

Mean ± sample standard deviation over three seeds, in percent:

| Dataset | Jaccard | F1 | PRAUC |
|---|---:|---:|---:|
| MIMIC-III | 57.73 ± 0.09 | 72.48 ± 0.08 | 81.77 ± 0.05 |
| MIMIC-IV | 53.20 ± 0.06 | 68.41 ± 0.03 | 77.97 ± 0.06 |
| eICU | 54.52 ± 0.16 | 68.89 ± 0.13 | 82.58 ± 0.20 |

Per-seed metrics are in [results/](results/README.md). The reference eICU DDI
matrix is a placeholder, so its DDI rate is N/A. Prediction replay checks existing
outputs; historical fresh-training validation covered only MIMIC-III seed1203.

## Tests and additional utilities

```bash
python -m unittest discover -s tests
```

The clinical prediction regression test requires `SEER_TEST_PREDICTIONS_ROOT`.
Tests requiring unavailable training dependencies or private predictions are skipped.
`reproduce_table1.py` and `reproduce_table2.py` format saved metrics;
`run_ablation.py` provides fusion diagnostics; use `python scripts/run_ablation.py --help` for its inputs.

## Baselines and license

Comparison sources are listed in [configs/baseline_registry.json](configs/baseline_registry.json).
Use `python scripts/run_baselines.py --list` to inspect availability. External
comparison methods require their own implementations and aligned data protocols.
Code licensing is described in [LICENSE](LICENSE); clinical data and derived
artifacts remain subject to their source access terms.
