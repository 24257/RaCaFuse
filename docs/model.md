# Model and Final Fusion

SEER combines a longitudinal branch and a SEER auxiliary branch.

## Longitudinal branch

Stage A trains the shared backbone with the common three-dataset protocol in `configs/final_model.json`. Stage B initializes from Stage A and trains the retrieval-readout path with the base frozen as specified by the same configuration.

The public shared training loss uses BCE weight `0.7`, soft-Jaccard weight `0.2`, pairwise-rank weight `0.02`, and weight decay `1e-6`. The hidden dimension is 256 and batch size is 32 for all three datasets. eICU adds a structural `disable_procedure_branch` training flag; this is not a dataset-specific hyperparameter retune.

### Final MIMIC-III checkpoint lineage

The shared protocol above and the lineage of the released MIMIC-III final checkpoint are related but distinct reproducibility statements. A later validation-only corrective/selection branch froze the final MIMIC-III checkpoint. For seed1203, that branch used Stage A coefficients `BCE=1.0`, `soft-Jaccard=0.3`, `pairwise-rank=0.02`, `weight_decay=1e-4`, trained Stage A for 60 epochs, fixed epoch55 as the teacher, and then trained Stage B retrieval-readout from that teacher. This does not change the shared public training batch size of 32.

Final component export is also separate from training. MIMIC-III final export uses runtime `batch=64`; that value is an inference/export setting and must not be described as a MIMIC-III training hyperparameter.

## SEER auxiliary branch

The adapted expert runtime is bundled under `src/seer_auxiliary_branch/` and invoked through `baselines/seer_auxiliary_adapter.py`. Its released checkpoint is `seer_auxiliary_branch.model`. Normal use loads the bundled SEER auxiliary runtime.

For exact historical MIMIC-III reproduction, the adapter preserves the original SetTransformer MAB computation and the historical CPU/Python DDI-rate path used by the locally adapted training code. An optimized attention path that changes the numerical trajectory is not substituted into the normal exact-reproduction path.

## Ranking fusion

The longitudinal final-decoder score is row-wise z-normalized. AUXILIARY probabilities are converted to logits and then row-wise z-normalized:

```text
0.3 * z(longitudinal_score) + 0.7 * z(logit(auxiliary_probability))
```

## Cardinality fusion

Both branch counts use train-only affine calibrators from `configs/count_calibration.json`:

```text
round(0.4 * calibrated_longitudinal_count
    + 0.6 * calibrated_auxiliary_count) + 1
```

The result is clipped to medication vocabulary size.

When fresh checkpoints are evaluated with `scripts/evaluate.py --refit-calibration`, both branch calibrators are refit from train-only outputs. `--calibration-data-root` applies to both branches, and calibration retrieval caches are isolated from target/test retrieval caches.

## DDI-aware decode

Final decoding is greedy with DDI penalty `0.04` and candidate window `60`. eICU DDI is N/A because its reference matrix has no nonzero interactions.

The auditable fusion implementation is `src/final_fusion.py`; `scripts/evaluate.py` orchestrates the released checkpoint roles, runtime/export semantics, fresh calibration refit, and frozen calibrators.
