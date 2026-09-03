# Data Preparation

MIMIC-III, MIMIC-IV, and eICU patient records are not redistributed in this repository. Obtain them through their official access procedures and licenses, then preprocess them locally into the loader format used by SEER.

Expected processed layout:

```text
data/
  mimic-iii/
    records_final.pkl
    voc_final.pkl
    ddi_A_final.pkl
    ehr.pkl
  mimic-iv/
    records_final.pkl
    voc_final.pkl
    ddi_A_final.pkl
    ehr.pkl
  eicu/
    records_final.pkl
    voc_final.pkl
    ddi_A_final.pkl
    ehr.pkl
```

MIMIC-III and eICU use local raw-data roots:

```bash
python scripts/preprocess.py --dataset mimic-iii --raw-root /path/to/raw/mimic-iii --output-root data/mimic-iii
python scripts/preprocess.py --dataset eicu --raw-root /path/to/raw/eicu --output-root data/eicu
```

For the literature-aligned MIMIC-IV cohort, use the locally obtained Carmen `data_final_4.pkl` as the source of truth:

```bash
python scripts/preprocess.py --dataset mimic-iv --source /path/to/data_final_4.pkl --output-root data/mimic-iv
```

A replay from stock MIMIC-IV v3.1 raw tables alone does not exactly reconstruct the same cohort and must not be reported as identical. Optional aligned EHR/DDI sources can be supplied explicitly.

The reference eICU DDI matrix is a zero placeholder. eICU DDI is therefore reported as N/A rather than as a safety result.

`data/sample/` contains only a synthetic non-clinical software fixture. See `docs/datasets.md` for protocol caveats.
