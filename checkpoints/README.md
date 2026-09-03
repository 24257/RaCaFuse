# Authorized Local Checkpoints

Weights are excluded from the public package. Set `SEER_PRIVATE_ARTIFACT_ROOT` to the external authorized artifact tree described in `docs/security.md`. The paths below are relative to that private root.

The final SEER release uses three checkpoint roles for every dataset and seed:

```text
checkpoints/<dataset>/seed<seed>/
├── stage_a.pth
├── stage_b.pth
└── seer_auxiliary_branch.model
```

Datasets are `mimic-iii`, `mimic-iv`, and `eicu`; seeds are `1203`, `1204`, and `1205`.

`stage_a.pth` is the shared longitudinal backbone checkpoint. `stage_b.pth` is the retrieval-readout checkpoint initialized from Stage A. `seer_auxiliary_branch.model` is the SEER auxiliary branch used by final SEER fusion; it is not an unused auxiliary artifact.

Final released-checkpoint inference requires all three roles for the requested dataset/seed. Public retraining writes to `artifacts/retrained/` by default so the released checkpoint tree is not overwritten.
