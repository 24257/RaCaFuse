# Data preparation

Obtain [MIMIC-III](https://physionet.org/content/mimiciii/1.4/),
[MIMIC-IV](https://physionet.org/content/mimiciv/3.1/), and
[eICU](https://physionet.org/content/eicu-crd/2.0/) under their access agreements.
Patient records, individual labels/predictions, weights, and frozen graph/mask
binaries are not redistributed here. Keep them outside the public repository.
Internal split indices are cohort positions; do not publish an identifier map.

## Processed inputs

Each local dataset directory must contain:

```text
<dataset-root>/
  records_final.pkl
  voc_final.pkl
  ddi_A_final.pkl
  ehr.pkl
```

Records, vocabulary indices, graph matrices, and checkpoint dimensions must agree.
Reference results require the same cohort and vocabulary as the reference model.

## Preprocessing

The supplied MIMIC-III and eICU raw-data entry points are:

```bash
python scripts/preprocess.py --dataset mimic-iii --raw-root /path/to/raw/mimic-iii --output-root /path/to/processed/mimic-iii
python scripts/preprocess.py --dataset eicu --raw-root /path/to/raw/eicu --output-root /path/to/processed/eicu
```

The MIMIC-III raw builder uses an NDC vocabulary and a zero DDI placeholder;
its output is not verified as identical to the reference experiment input.
The reference eICU DDI matrix is also a placeholder, and eICU DDI is reported as N/A.
Do not interpret a placeholder matrix as evidence of interaction safety.

For the literature-aligned MIMIC-IV cohort, import a locally authorized Carmen
`data_final_4.pkl`:

```bash
python scripts/preprocess.py --dataset mimic-iv --source /path/to/data_final_4.pkl --output-root /path/to/processed/mimic-iv
```

The MIMIC-IV importer accepts `--ddi-source`, `--ddi-source-vocab`, and
`--ehr-source` for aligned inputs. A stock MIMIC-IV v3.1 raw rebuild is not
verified to reconstruct this cohort exactly. See [Carmen](https://github.com/bit1029public/Carmen)
for the upstream processing context.

## External reference artifacts

Set `SEER_PRIVATE_ARTIFACT_ROOT` to the authorized tree below:

```text
<private-root>/
  checkpoints/<dataset>/seed<seed>/stage_a.pth
  checkpoints/<dataset>/seed<seed>/stage_b.pth
  checkpoints/<dataset>/seed<seed>/seer_auxiliary_branch.model
  configs/frozen_inputs/ehr_graph/<dataset>/seed<seed>/*.pkl
  configs/frozen_inputs/ddi_mask/<dataset>/seed<seed>.pkl
  predictions/seer/<dataset>/seed<seed>.npz
```

Predictions are only needed for metric replay. Checkpoints and frozen inputs are
required by the reference evaluation command. Use `--data-root` for the processed
clinical dataset. Follow the source agreements when obtaining or sharing derived
artifacts; see [PhysioNet guidance](https://physionet.org/content/mimiciv/3.1/).

## Loading trusted local files

NPZ readers reject object arrays. Checkpoints use `weights_only=True` without an
unsafe fallback; unsupported legacy checkpoints need a pure tensor-state export
from a trusted environment.

Dill/Pickle can execute code and is disabled by default. Only for independently
verified local inputs and caches, enable it for the current process:

```bash
SEER_PRIVATE_ARTIFACT_ROOT=/path/to/private-root SEER_ALLOW_UNSAFE_PICKLE=1 python scripts/evaluate.py --dataset mimic-iii --seed 1203 --data-root /path/to/processed/mimic-iii --cuda 0
```

In PowerShell, set `$env:SEER_PRIVATE_ARTIFACT_ROOT='/path/to/private-root'` and
`$env:SEER_ALLOW_UNSAFE_PICKLE='1'` before the run, then remove the trust setting
with `Remove-Item Env:SEER_ALLOW_UNSAFE_PICKLE`. The same trust setting is required
by training or preprocessing paths that read legacy Pickle files. Do not enable
it globally or for unverified downloads.

`sample/` contains synthetic data only. Before sharing a directory or archive,
run `python scripts/verify_public.py` from the repository root. It checks known
file types and locations; it cannot detect every form of sensitive data. Do not
include private outputs or backups, and remember that ignore rules do not filter
archives or remove previously committed files.
