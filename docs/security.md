# Public release and local artifacts

This package provides the implementation and reproducibility materials accompanying
the SEER manuscript submission. The public package contains source code, aggregate reference metrics, configuration
and a synthetic fixture. It excludes individual clinical labels/predictions,
trained checkpoints and frozen graph/mask binaries. Internal split indices are
cohort positions, not original clinical identifiers; do not add an identifier map.
No publication venue, acceptance, or DOI is claimed.

Before distributing a directory or archive, run:

```bash
python scripts/verify_public.py
```

This checks known file types and locations, not arbitrary disguised clinical data.
Do not upload local runtime outputs, backups, model files or original clinical data.
Ignore rules do not remove files from an archive or already tracked Git history.

## Authorized local evaluation

Obtain clinical data and derived artifacts only through their applicable access
agreements. A code license does not grant permission to redistribute clinical data
or trained models. PhysioNet's MIMIC guidance covers derived datasets and models:
https://physionet.org/content/mimiciv/3.1/

Keep authorized artifacts outside the public directory in this layout:

```text
<private-root>/
  checkpoints/<dataset>/seed<seed>/stage_a.pth
  checkpoints/<dataset>/seed<seed>/stage_b.pth
  checkpoints/<dataset>/seed<seed>/seer_auxiliary_branch.model
  configs/frozen_inputs/ehr_graph/<dataset>/seed<seed>/*.pkl
  configs/frozen_inputs/ddi_mask/<dataset>/seed<seed>.pkl
  predictions/seer/<dataset>/seed<seed>.npz
```

`SEER_PRIVATE_ARTIFACT_ROOT` selects this tree. `--data-root` selects separately
authorized processed clinical data. `--dry-run` prints planned commands without
checking external checkpoint existence or reading clinical data.

```bash
python scripts/reproduce_results.py --predictions-root /path/to/private-root/predictions/seer
```

This recomputes clinical metrics from authorized local predictions. In contrast,
`verify_public.py` uses synthetic data and `reproduce_table1.py`/`reproduce_table2.py`
only format the provided aggregate/per-seed metric files.

## File-loading trust boundary

NPZ readers disable Pickle. Object arrays are rejected. Model checkpoints use
`torch.load(weights_only=True)` with no automatic unsafe fallback. Use a maintained,
security-patched PyTorch build. Legacy checkpoints containing unsupported objects
must be exported as pure tensor state dictionaries in a trusted environment first.

Legacy processed clinical data and caches still use Dill/Pickle, which can execute
arbitrary code. Their readers refuse to load by default. Only after verifying the
origin of every input/cache, explicitly enable loading for the current process:

```bash
SEER_PRIVATE_ARTIFACT_ROOT=/path/to/private-root SEER_ALLOW_UNSAFE_PICKLE=1 python scripts/evaluate.py --dataset mimic-iii --seed 1203 --data-root /path/to/authorized/data
```

In PowerShell, set `$env:SEER_PRIVATE_ARTIFACT_ROOT` and
`$env:SEER_ALLOW_UNSAFE_PICKLE='1'` for the authorized run, then remove the latter
with `Remove-Item Env:SEER_ALLOW_UNSAFE_PICKLE`. This opt-in is not a sandbox or a
malware check. Do not set it globally or enable it for unverified downloaded files.

The historical Python 3.9 environment is for provenance only and is end-of-life.
Use a supported Python version for new installations. Full GPU reproducibility
still requires validating the chosen dependency versions on authorized data.
