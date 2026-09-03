# Reproduction Guide

`scripts/evaluate.py --cuda N` and its inference alias use visible GPU index N
for both model branches, including train-only calibration. Use `--cuda -1` for
CPU execution. Indices are relative to `CUDA_VISIBLE_DEVICES` when it is set.

Graph and retrieval caches now include SHA-256 content identities for their
inputs. Changes to records or vocabulary mappings invalidate cached results,
even when shapes and patient indices stay the same. Legacy cache names are not
reused; the first run rebuilds caches and needs write access to the cache directory.

SEER keeps three reproducibility levels separate.

## Public synthetic check

Run `python scripts/verify_public.py` without clinical data. This is a software smoke test only.

## Authorized local artifact replay

Authorized local prediction files are required; a GPU is not needed:

```bash
python scripts/reproduce_results.py --predictions-root /path/to/authorized/predictions/seer
```

The checker loads all nine authorized local NPZ files, independently recomputes metrics, checks per-seed values, recomputes arithmetic means and sample SDs, and verifies the displayed paper targets. During release preparation, the authorized local final arrays were also checked against the original final-test component fusion at array level.

## Released-checkpoint inference

After preparing authorized processed data, supplying external checkpoints/frozen inputs, and configuring the trust settings in `docs/security.md`:

```bash
python scripts/evaluate.py --dataset mimic-iii --seed 1203 --cuda 0
```

Use `--dry-run` first to inspect split/checkpoint/runtime commands without opening clinical data. Final inference consumes `stage_a.pth`, `stage_b.pth`, and `seer_auxiliary_branch.model`, plus the frozen train-only affine parameters in `configs/count_calibration.json`.

Final inference/export settings are intentionally separated from the shared training protocol. In particular, MIMIC-III final component export uses runtime `batch=64`; the public shared training batch remains `32` for MIMIC-III, MIMIC-IV, and eICU.

## Shared fresh retraining

The public longitudinal entry point uses one shared three-dataset protocol:

```bash
python scripts/train.py --dataset mimic-iii --seed 1203
python scripts/train.py --dataset mimic-iv --seed 1203
python scripts/train.py --dataset eicu --seed 1203
```

The SEER auxiliary branch is retrained separately:

```bash
python scripts/train_seer_auxiliary.py --dataset mimic-iii --seed 1203 --cuda 0
```

Fresh GPU training depends on protected data, CUDA kernels, and library versions, so ordinary tiny floating-point variation can occur. It is a different reproducibility claim from exact authorized-artifact replay.

## MIMIC-III final-checkpoint lineage

The public shared longitudinal protocol is not the same statement as the later checkpoint-freezing lineage. The released MIMIC-III final checkpoint was frozen after a validation-only corrective/selection branch. For seed1203, the final-lineage Stage A used `BCE=1.0`, `soft-Jaccard=0.3`, `pairwise-rank=0.02`, and `weight_decay=1e-4`, trained for 60 epochs, and used the fixed epoch55 teacher before Stage B retrieval-readout training. Test data were not used for this teacher selection.

These later final-checkpoint coefficients do not redefine the original public shared cross-dataset protocol and should not be described as evidence that MIMIC-III trained with a different batch size. The shared training batch is 32; the batch-64 value belongs to final inference/component export.

## Completed real-data acceptance

A full protected-data acceptance was completed for **MIMIC-III seed1203 only**. It covered fresh random initialization, Stage A, final-lineage epoch55 selection, Stage B, a full 40-epoch SEER auxiliary branch retraining, train-only calibration refit, and final fusion.

Acceptance observations:

- fresh Stage A epoch55 reproduced the historical final-lineage checkpoint;
- fresh Stage B matched the historical final model state tensor-by-tensor;
- the fresh AUXILIARY expert completed all 40 epochs with best epoch39 and reproduced the historical training metrics;
- longitudinal train and test core component arrays matched the historical components exactly;
- AUXILIARY train arrays matched exactly; AUXILIARY test probabilities differed by at most about `1.67e-6` while discrete AUXILIARY predictions matched;
- refit seed1203 calibrators were `longitudinal=(0.95,-0.35)` and `AUXILIARY=(0.90,-1.15)`;
- final `y_true`, `y_pred`, `predicted_count`, `patient_ids`, and `visit_ids` matched the released seed1203 prediction exactly.

The resulting seed1203 metrics were Jaccard `0.5778873796077955`, F1 `0.7255168981185356`, and PRAUC `0.8178497618992944`.

This acceptance does **not** claim fresh retraining of MIMIC-IV, eICU, seed1204, or seed1205. The three-dataset/three-seed table remains validated by exact artifact replay.

## Fresh-checkpoint calibration refit

`scripts/evaluate.py --refit-calibration` exports train-only outputs from both supplied checkpoint branches and refits the affine count calibrators before target/test fusion. If the calibration data surface differs from the target/test data surface, pass `--calibration-data-root`; it applies to both learned branches. Target/test and calibration retrieval caches are kept separate so cached retrieval priors cannot be mixed across data surfaces.
