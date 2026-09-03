# Dataset Notes

Clinical source data are not redistributed. Users must obtain MIMIC-III, MIMIC-IV, and eICU under the source datasets' own access terms and licenses.

## Processed surface

Each dataset directory under `data/` must provide the record/vocabulary objects and aligned graph matrices expected by the public loaders, including equivalents of `records_final.pkl`, `voc_final.pkl`, `ddi_A_final.pkl`, and the corresponding EHR adjacency artifact.

## MIMIC-III

Use `scripts/preprocess.py --dataset mimic-iii` with a local raw-data root. The packaged raw rebuild path does not claim to reconstruct the paper DDI matrix directly from raw MIMIC tables; use the aligned DDI input required by the experiment when applicable.

## MIMIC-IV

The literature-aligned release uses Carmen `data_final_4.pkl` as the cohort source of truth:

```bash
python scripts/preprocess.py --dataset mimic-iv --source /path/to/data_final_4.pkl --output-root data/mimic-iv
```

A replay from stock MIMIC-IV v3.1 raw tables does not exactly reconstruct the same literature cohort and must not be reported as identical. Optional aligned EHR/DDI sources can be supplied explicitly.

## eICU

Use `scripts/preprocess.py --dataset eicu` with a local eICU root. The final release uses explicit packaged split manifests. The reference eICU DDI matrix is a zero placeholder, so DDI is reported as N/A.

## Synthetic fixture

`data/make_synthetic_fixture.py` creates a tiny non-clinical NPZ for software smoke testing only. It has no scientific-result meaning.
