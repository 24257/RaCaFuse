# Released Results

This directory contains the compact reference results used to regression-check the SEER release.

- `test_results_per_seed.csv`: metrics for all 3 datasets × 3 seeds.
- `test_results_mean_sd.csv`: arithmetic means and sample standard deviations (`ddof=1`).
- `FINAL_TEST_RESULTS.json`: machine-readable aggregate targets.

The final arrays are excluded from this public package and must be supplied through an authorized local path. `scripts/reproduce_results.py` recomputes Jaccard, F1, PRAUC, and count MAE directly from those arrays rather than trusting the CSV values.

```bash
python scripts/reproduce_results.py --predictions-root /path/to/authorized/predictions/seer
```

Expected paper-display values are:

| Dataset | Jaccard | F1 | PRAUC |
|---|---:|---:|---:|
| MIMIC-III | 57.73 ± 0.09 | 72.48 ± 0.08 | 81.77 ± 0.05 |
| MIMIC-IV | 53.20 ± 0.06 | 68.41 ± 0.03 | 77.97 ± 0.06 |
| eICU | 54.52 ± 0.16 | 68.89 ± 0.13 | 82.58 ± 0.20 |

The reference eICU DDI matrix contains no nonzero interactions, so eICU DDI is reported as N/A rather than interpreted as a safety result.
