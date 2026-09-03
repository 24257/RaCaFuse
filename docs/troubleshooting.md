# Troubleshooting

## Clinical data file is missing

The repository does not redistribute MIMIC/eICU patient data. Prepare the processed dataset under `data/<dataset>/` or pass the documented local data/source arguments. Use `--dry-run` when only command inspection is needed.

## MIMIC-IV cohort mismatch

For the literature-aligned cohort, use Carmen `data_final_4.pkl`. A stock MIMIC-IV v3.1 raw replay is not guaranteed to produce the identical cohort and should not be treated as such.

## SEERAuxiliary source error

The release bundles the adapted runtime in `src/seer_auxiliary_branch/`; an external checkout is not required. `SEER_AUXILIARY_SRC` or `--seer_auxiliary-src` is only an explicit debugging override. Remove stale environment overrides if the adapter unexpectedly points elsewhere.

## AUXILIARY fresh training drifts from the historical path

For exact MIMIC-III reproduction, use the bundled adapter unchanged. The historical locally adapted training path uses the original SetTransformer MAB computation and CPU/Python DDI-rate semantics. Replacing that path with scaled-dot-product attention or other optimized numerical shortcuts can change the training trajectory even when the high-level equations look equivalent.

## eICU DDI is N/A

This is expected. The reference eICU DDI matrix has no nonzero interactions. A numerical zero would be misleading as a safety conclusion, so the release reports N/A.

## Artifact replay passes but fresh training differs

Artifact replay and fresh training are separate claims. The nine authorized local final predictions are the exact three-dataset/three-seed regression target. A real-data fresh acceptance was completed for MIMIC-III seed1203, where the final discrete prediction set and predicted cardinality matched the released arrays exactly. Tiny floating-point variation can still occur in continuous scores; the observed AUXILIARY test-probability maximum difference in that acceptance was about `1.67e-6` and did not change the final recommendation set.

Do not infer from the seed1203 acceptance that MIMIC-IV, eICU, seed1204, or seed1205 were freshly retrained during the same pass.

## MIMIC-III batch size looks inconsistent

The shared longitudinal **training** batch size is 32. MIMIC-III final **component export/inference** uses runtime batch 64. The latter is not a dataset-specific training hyperparameter.

## Fresh calibration fails with retrieval-cache index errors

When `--calibration-data-root` differs from the target/test `--data-root`, use the current `scripts/evaluate.py`. It applies the calibration root to train-only exports from both branches and keeps calibration retrieval caches separate from target/test retrieval caches. Reusing a cache built against a different data surface can produce retrieval-prior length mismatches.

## Missing Python module

Create the release environment with `conda env create -f environment.yml` or install `requirements.txt`. The historically verified experiment environment used Python 3.9.
