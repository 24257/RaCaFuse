from __future__ import annotations

import argparse
import json
from pathlib import Path

import dill
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "mimic-iii"


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            word = str(word)
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


def open_csv(path, **kwargs):
    if str(path).endswith(".gz"):
        return pd.read_csv(path, compression="gzip", **kwargs)
    return pd.read_csv(path, **kwargs)


def normalize_code(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def normalize_ndc(value):
    text = normalize_code(value)
    if text is None:
        return None
    text = text.replace("-", "")
    try:
        text = str(int(float(text)))
    except ValueError:
        pass
    if text == "0":
        return None
    return text


def preprocess_medications(raw_root, top_med):
    med = open_csv(raw_root / "PRESCRIPTIONS.csv.gz", dtype={"NDC": "string"}, low_memory=False)
    raw_rows = len(med)
    med = med[["SUBJECT_ID", "HADM_ID", "STARTDATE", "NDC"]].copy()
    med["NDC"] = med["NDC"].map(normalize_ndc)
    med = med.dropna(subset=["SUBJECT_ID", "HADM_ID", "STARTDATE", "NDC"])
    med["STARTDATE"] = pd.to_datetime(med["STARTDATE"])
    med = med.drop_duplicates()
    med = med.sort_values(["SUBJECT_ID", "HADM_ID", "STARTDATE"]).reset_index(drop=True)

    visit_counts = med[["SUBJECT_ID", "HADM_ID"]].groupby("SUBJECT_ID")["HADM_ID"].nunique()
    valid_patients = set(int(subject_id) for subject_id in visit_counts[visit_counts > 1].index)
    med = med[med["SUBJECT_ID"].astype(int).isin(valid_patients)].reset_index(drop=True)

    top_codes = med["NDC"].value_counts().head(top_med).index
    med = med[med["NDC"].isin(top_codes)].reset_index(drop=True)
    return med, {
        "prescription_rows_raw": int(raw_rows),
        "prescription_rows_after_raw_filter": int(len(med)),
        "patients_with_gt1_prescription_visits": int(len(valid_patients)),
        "medication_code_source": "PRESCRIPTIONS.NDC",
        "medication_top_k": int(top_med),
    }


def process_diagnoses(raw_root, top_diag):
    diag = open_csv(raw_root / "DIAGNOSES_ICD.csv.gz", dtype={"ICD9_CODE": "string"})
    raw_rows = len(diag)
    diag = diag[["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]].copy()
    diag["ICD9_CODE"] = diag["ICD9_CODE"].map(normalize_code)
    diag = diag.dropna(subset=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
    diag = diag.drop_duplicates()
    diag = diag.sort_values(["SUBJECT_ID", "HADM_ID"]).reset_index(drop=True)
    top_codes = diag["ICD9_CODE"].value_counts().head(top_diag).index
    diag = diag[diag["ICD9_CODE"].isin(top_codes)].reset_index(drop=True)
    return diag, {
        "diagnosis_rows_raw": int(raw_rows),
        "diagnosis_rows_after_raw_filter": int(len(diag)),
        "diagnosis_code_source": "DIAGNOSES_ICD.ICD9_CODE",
        "diagnosis_top_k": int(top_diag),
    }


def process_procedures(raw_root):
    proc = open_csv(raw_root / "PROCEDURES_ICD.csv.gz", dtype={"ICD9_CODE": "string"})
    raw_rows = len(proc)
    keep_cols = ["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"] if "SEQ_NUM" in proc.columns else ["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]
    proc = proc[keep_cols].copy()
    proc["ICD9_CODE"] = proc["ICD9_CODE"].map(normalize_code)
    proc = proc.dropna(subset=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
    proc = proc.drop_duplicates()
    sort_cols = ["SUBJECT_ID", "HADM_ID"]
    if "SEQ_NUM" in proc.columns:
        sort_cols.append("SEQ_NUM")
    proc = proc.sort_values(sort_cols)
    if "SEQ_NUM" in proc.columns:
        proc = proc.drop(columns=["SEQ_NUM"])
    proc = proc.drop_duplicates().reset_index(drop=True)
    return proc, {
        "procedure_rows_raw": int(raw_rows),
        "procedure_rows_after_raw_filter": int(len(proc)),
        "procedure_code_source": "PROCEDURES_ICD.ICD9_CODE",
        "procedure_top_k": None,
    }


def combine_tables(med, diag, proc):
    med_key = med[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    diag_key = diag[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    proc_key = proc[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    combined_key = med_key.merge(diag_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    combined_key = combined_key.merge(proc_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")

    med = med.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    diag = diag.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    proc = proc.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")

    diag_grouped = diag.groupby(["SUBJECT_ID", "HADM_ID"], sort=True)["ICD9_CODE"].unique().reset_index()
    med_grouped = med.groupby(["SUBJECT_ID", "HADM_ID"], sort=True)["NDC"].unique().reset_index()
    proc_grouped = (
        proc.groupby(["SUBJECT_ID", "HADM_ID"], sort=True)["ICD9_CODE"]
        .unique()
        .reset_index()
        .rename(columns={"ICD9_CODE": "PRO_CODE"})
    )
    diag_grouped["ICD9_CODE"] = diag_grouped["ICD9_CODE"].map(list)
    med_grouped["NDC"] = med_grouped["NDC"].map(list)
    proc_grouped["PRO_CODE"] = proc_grouped["PRO_CODE"].map(list)

    data = diag_grouped.merge(med_grouped, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    data = data.merge(proc_grouped, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    data["NDC_Len"] = data["NDC"].map(len)
    return data.sort_values(["SUBJECT_ID", "HADM_ID"]).reset_index(drop=True)


def create_vocab(data):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()
    for _, row in data.iterrows():
        diag_voc.add_sentence(row["ICD9_CODE"])
        med_voc.add_sentence(row["NDC"])
        pro_voc.add_sentence(row["PRO_CODE"])
    return {"diag_voc": diag_voc, "med_voc": med_voc, "pro_voc": pro_voc}


def create_records_and_visit_keys(data, voc):
    records = []
    visit_keys = []
    for subject_id in data["SUBJECT_ID"].unique():
        item_df = data[data["SUBJECT_ID"] == subject_id]
        patient = []
        patient_keys = []
        for _, row in item_df.iterrows():
            patient.append(
                [
                    [voc["diag_voc"].word2idx[str(code)] for code in row["ICD9_CODE"]],
                    [voc["pro_voc"].word2idx[str(code)] for code in row["PRO_CODE"]],
                    [voc["med_voc"].word2idx[str(code)] for code in row["NDC"]],
                ]
            )
            patient_keys.append((int(row["SUBJECT_ID"]), int(row["HADM_ID"])))
        records.append(patient)
        visit_keys.append(patient_keys)
    return records, visit_keys


def create_binary_ehr_adj(records, med_voc_size):
    ehr_adj = np.zeros((med_voc_size, med_voc_size), dtype=np.float32)
    for patient in records:
        for visit in patient:
            meds = list(dict.fromkeys(int(med) for med in visit[2]))
            for i, med_i in enumerate(meds):
                for med_j in meds[i + 1 :]:
                    ehr_adj[med_i, med_j] = 1.0
                    ehr_adj[med_j, med_i] = 1.0
    return ehr_adj


def create_directed_weighted_ehr_adj(records, med_voc_size):
    co_counts = np.zeros((med_voc_size, med_voc_size), dtype=np.float32)
    med_counts = np.zeros(med_voc_size, dtype=np.float32)
    for patient in records:
        for visit in patient:
            meds = list(dict.fromkeys(int(med) for med in visit[2]))
            if not meds:
                continue
            med_counts[meds] += 1.0
            visit_vector = np.zeros(med_voc_size, dtype=np.float32)
            visit_vector[meds] = 1.0
            co_counts += np.outer(visit_vector, visit_vector)
    np.fill_diagonal(co_counts, 0.0)
    denom = np.maximum(med_counts.reshape(-1, 1), 1.0)
    return co_counts / denom


def create_raw_only_ddi_adj(raw_root, med_voc_size):
    candidate_files = [
        "drug-DDI.csv",
        "drug-DDI.csv.gz",
        "DDI.csv",
        "DDI.csv.gz",
    ]
    found = [str(raw_root / name) for name in candidate_files if (raw_root / name).exists()]
    if found:
        raise NotImplementedError(
            "A DDI raw file exists, but its schema must be reviewed before building ddi_A_final.pkl: "
            + ", ".join(found)
        )
    ddi_adj = np.zeros((med_voc_size, med_voc_size), dtype=np.float32)
    return ddi_adj, {
        "ddi_source": "none_in_mimic_iii_raw",
        "ddi_nonzero": 0,
        "ddi_note": (
            "MIMIC-III raw tables do not contain a drug-drug-interaction fact table. "
            "To keep every generated file raw-only and avoid author processed data, "
            "ddi_A_final.pkl is an all-zero placeholder with the raw NDC medication vocabulary."
        ),
        "searched_ddi_paths": [str(raw_root / name) for name in candidate_files],
    }


def dump_pickle(path, payload):
    with open(path, "wb") as fh:
        dill.dump(payload, fh)


def dataset_statistics(data, records, voc):
    return {
        "patient_count": int(len(records)),
        "visit_count": int(sum(len(patient) for patient in records)),
        "diagnosis_voc_size": int(len(voc["diag_voc"].idx2word)),
        "procedure_voc_size": int(len(voc["pro_voc"].idx2word)),
        "medication_voc_size": int(len(voc["med_voc"].idx2word)),
        "dataframe_patient_count": int(data["SUBJECT_ID"].nunique()),
        "dataframe_visit_count": int(len(data)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Strictly rebuild MIMIC-III SEER processed data from raw MIMIC-III tables.")
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--top_diag", type=int, default=2000)
    parser.add_argument("--top_med", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_root} already exists and is not empty; pass --overwrite to replace generated files")
    output_root.mkdir(parents=True, exist_ok=True)

    audit = {
        "dataset": "mimic-iii",
        "source": "strict_raw_mimic_iii",
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "source_tables": [
            "PRESCRIPTIONS.csv.gz",
            "DIAGNOSES_ICD.csv.gz",
            "PROCEDURES_ICD.csv.gz",
            "LABEVENTS.csv.gz",
            "CHARTEVENTS.csv.gz",
            "ICUSTAYS.csv.gz",
            "PATIENTS.csv.gz",
        ],
        "selection": {
            "med_patient_filter": "patients with more than one HADM_ID in filtered PRESCRIPTIONS",
            "diagnosis_top_k": int(args.top_diag),
            "medication_top_k": int(args.top_med),
            "procedure_top_k": None,
            "required_visit_tables": ["medications", "diagnoses", "procedures"],
            "record_order": ["SUBJECT_ID", "HADM_ID"],
        },
        "author_processed_data_used": False,
    }

    print("[INFO] reading medications from raw PRESCRIPTIONS.NDC")
    med, med_audit = preprocess_medications(raw_root, args.top_med)
    audit.update(med_audit)

    print("[INFO] reading diagnoses from raw DIAGNOSES_ICD")
    diag, diag_audit = process_diagnoses(raw_root, args.top_diag)
    audit.update(diag_audit)

    print("[INFO] reading procedures from raw PROCEDURES_ICD")
    proc, proc_audit = process_procedures(raw_root)
    audit.update(proc_audit)

    print("[INFO] combining visits")
    data = combine_tables(med, diag, proc)
    voc = create_vocab(data)
    records, visit_keys = create_records_and_visit_keys(data, voc)
    stats = dataset_statistics(data, records, voc)
    audit.update(stats)

    print("[INFO] building raw-derived EHR graphs")
    med_voc_size = len(voc["med_voc"].idx2word)
    ehr_binary = create_binary_ehr_adj(records, med_voc_size)
    ehr_directed = create_directed_weighted_ehr_adj(records, med_voc_size)
    audit["ehr_graph_source"] = "computed_from_raw_rebuilt_records"
    audit["ehr_nonzero"] = int(np.count_nonzero(ehr_directed))

    print("[INFO] building raw-only DDI placeholder")
    ddi_adj, ddi_audit = create_raw_only_ddi_adj(raw_root, med_voc_size)
    audit.update(ddi_audit)

    print("[INFO] writing files")
    data.to_pickle(output_root / "data_final.pkl")
    dump_pickle(output_root / "records_final.pkl", records)
    dump_pickle(output_root / "voc_final.pkl", voc)
    dump_pickle(output_root / "visit_keys.pkl", visit_keys)
    dump_pickle(output_root / "ehr_adj_final.pkl", ehr_binary)
    dump_pickle(output_root / "weighted_confidence_directed_ehr_graph.pkl", ehr_directed)
    dump_pickle(output_root / "ehr.pkl", ehr_directed)
    dump_pickle(output_root / "ddi_A_final.pkl", ddi_adj)
    with open(output_root / "raw_rebuild_audit.json", "w") as fh:
        json.dump(audit, fh, indent=2, sort_keys=True)

    print(
        "[OK] wrote strict raw MIMIC-III dataset "
        f"patients={stats['patient_count']} visits={stats['visit_count']} "
        f"voc=({stats['diagnosis_voc_size']}, {stats['procedure_voc_size']}, {stats['medication_voc_size']}) "
        f"to {output_root}"
    )


if __name__ == "__main__":
    main()
