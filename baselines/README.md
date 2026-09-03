# Baselines and SEER Auxiliary Branch

Third-party comparison repositories are not vendored wholesale because they have their own licenses and dependency stacks. The locally recorded project links and runner status are stored in `configs/baseline_registry.json`.

Inspect the registry with:

```bash
python scripts/run_baselines.py --list
```

The **SEER auxiliary branch** is part of the final SEER computation graph. Its runtime is bundled in `src/seer_auxiliary_branch/`, its adapter is `baselines/seer_auxiliary_adapter.py`, and its released checkpoint role is `seer_auxiliary_branch.model`.

`SEER_AUXILIARY_SRC`/`--seer_auxiliary-src` is only an explicit debugging override; normal use loads the bundled SEER auxiliary runtime.

When reproducing third-party comparisons, use the same processed cohort, split policy, target-visit policy, and metric definitions as SEER. Do not claim a comparison has been reproduced unless that method was actually rerun under the aligned protocol.
