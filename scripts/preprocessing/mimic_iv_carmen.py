#!/usr/bin/env python3
"""Build package-ready MIMIC-IV artifacts from Carmen's data_final_4.pkl."""

from __future__ import annotations

import sys as _seer_sys
from pathlib import Path as _SeerPath
_seer_sys.path.insert(0, str(_SeerPath(__file__).resolve().parents[2] / "src"))
import seer_security

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import dill
import numpy as np
import pandas as pd

from ddi_vocab_alignment import align_square_matrix_by_codes, ordered_vocabulary_codes


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def dump_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        dill.dump(payload, fh)


def load_pickle(path: Path):
    with path.open("rb") as fh:
        return seer_security.load_pickle(fh)


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def load_carmen_frame(path: Path) -> pd.DataFrame:
    frame = seer_security.read_pandas_pickle(path)
    required = {"subject_id", "hadm_id", "admittime", "icd_code", "pro_code", "ndc"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing Carmen columns: {missing}")
    return frame.copy()


def audit_source_frame(frame: pd.DataFrame) -> dict:
    patient_count = int(frame["subject_id"].nunique())
    visit_count = int(len(frame))
    duplicate_visit_rows = int(frame.duplicated(["subject_id", "hadm_id"]).sum())
    visits_per_patient = frame.groupby("subject_id")["hadm_id"].nunique()
    return {
        "patient_count": patient_count,
        "visit_count": visit_count,
        "unique_visit_keys": int(frame[["subject_id", "hadm_id"]].drop_duplicates().shape[0]),
        "duplicate_visit_rows": duplicate_visit_rows,
        "single_visit_patients": int((visits_per_patient < 2).sum()),
        "min_visits_per_patient": int(visits_per_patient.min()) if not visits_per_patient.empty else 0,
        "max_visits_per_patient": int(visits_per_patient.max()) if not visits_per_patient.empty else 0,
    }


def validate_source_frame(
    frame: pd.DataFrame,
    expected_patients: int | None,
    expected_visits: int | None,
    allow_single_visit_patients: bool,
) -> dict:
    audit = audit_source_frame(frame)
    if audit["duplicate_visit_rows"]:
        raise ValueError(f"Carmen source has duplicate (subject_id, hadm_id) rows: {audit['duplicate_visit_rows']}")
    if audit["single_visit_patients"] and not allow_single_visit_patients:
        raise ValueError(f"Carmen source has single-visit patients: {audit['single_visit_patients']}")
    if expected_patients is not None and audit["patient_count"] != expected_patients:
        raise ValueError(f"patient count mismatch: got {audit['patient_count']}, expected {expected_patients}")
    if expected_visits is not None and audit["visit_count"] != expected_visits:
        raise ValueError(f"visit count mismatch: got {audit['visit_count']}, expected {expected_visits}")
    return audit


def create_vocab(frame: pd.DataFrame) -> dict[str, Voc]:
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()
    for row in frame.itertuples(index=False):
        diag_voc.add_sentence(getattr(row, "icd_code"))
        med_voc.add_sentence(getattr(row, "ndc"))
        pro_voc.add_sentence(getattr(row, "pro_code"))
    return {"diag_voc": diag_voc, "med_voc": med_voc, "pro_voc": pro_voc}


def create_records_and_visit_keys(frame: pd.DataFrame, voc: dict[str, Voc]):
    records = []
    visit_keys = []
    frame = frame.sort_values(["subject_id", "admittime", "hadm_id"]).reset_index(drop=True)
    for _, patient_frame in frame.groupby("subject_id", sort=False):
        patient = []
        patient_keys = []
        patient_frame = patient_frame.sort_values(["admittime", "hadm_id"])
        for row in patient_frame.itertuples(index=False):
            patient.append(
                [
                    [int(voc["diag_voc"].word2idx[str(code)]) for code in getattr(row, "icd_code")],
                    [int(voc["pro_voc"].word2idx[str(code)]) for code in getattr(row, "pro_code")],
                    [int(voc["med_voc"].word2idx[str(code)]) for code in getattr(row, "ndc")],
                ]
            )
            patient_keys.append((int(getattr(row, "subject_id")), int(getattr(row, "hadm_id"))))
        records.append(patient)
        visit_keys.append(patient_keys)
    return records, visit_keys


def audit_records(records, visit_keys) -> dict:
    visit_key_rows = [tuple(key) for patient in visit_keys for key in patient]
    visit_key_counts = Counter(visit_key_rows)
    patient_key_counts = Counter(tuple(tuple(key) for key in patient) for patient in visit_keys)
    first_subject_counts = Counter(patient[0][0] for patient in visit_keys if patient)
    return {
        "record_patient_count": int(len(records)),
        "record_visit_count": int(sum(len(patient) for patient in records)),
        "visit_key_patient_count": int(len(visit_keys)),
        "visit_key_visit_count": int(len(visit_key_rows)),
        "duplicate_visit_keys": int(sum(count - 1 for count in visit_key_counts.values() if count > 1)),
        "duplicate_patient_key_sequences": int(sum(count - 1 for count in patient_key_counts.values() if count > 1)),
        "subjects_split_across_patient_blocks": int(sum(1 for count in first_subject_counts.values() if count > 1)),
        "extra_patient_blocks_from_subject_split": int(sum(count - 1 for count in first_subject_counts.values() if count > 1)),
    }


def validate_records(records, visit_keys) -> dict:
    audit = audit_records(records, visit_keys)
    if audit["record_patient_count"] != audit["visit_key_patient_count"]:
        raise ValueError("records and visit keys have different patient counts")
    if audit["record_visit_count"] != audit["visit_key_visit_count"]:
        raise ValueError("records and visit keys have different visit counts")
    if audit["duplicate_visit_keys"]:
        raise ValueError(f"duplicate visit keys detected: {audit['duplicate_visit_keys']}")
    if audit["duplicate_patient_key_sequences"]:
        raise ValueError(f"duplicate patient key sequences detected: {audit['duplicate_patient_key_sequences']}")
    if audit["subjects_split_across_patient_blocks"]:
        raise ValueError(f"subjects split across patient blocks: {audit['subjects_split_across_patient_blocks']}")
    return audit


def build_ehr_adjacency(records, med_voc_size: int) -> np.ndarray:
    ehr_adj = np.zeros((med_voc_size, med_voc_size), dtype=np.float64)
    for patient in records:
        for visit in patient:
            meds = visit[2]
            for i, med_i in enumerate(meds):
                for med_j in meds[i + 1 :]:
                    ehr_adj[med_i, med_j] = 1.0
                    ehr_adj[med_j, med_i] = 1.0
    return ehr_adj


def copy_or_build_graphs(args: argparse.Namespace, output_root: Path, records, voc: dict[str, Voc]) -> dict:
    graph_audit = {}
    med_voc_size = len(voc["med_voc"].idx2word)
    if args.ddi_source:
        ddi_source = resolve_path(args.ddi_source)
        ddi_arr = np.asarray(load_pickle(ddi_source))
        graph_audit["ddi_source"] = str(ddi_source)
        if args.ddi_source_vocab:
            ddi_source_vocab = resolve_path(args.ddi_source_vocab)
            source_voc = load_pickle(ddi_source_vocab)
            ddi_arr, alignment = align_square_matrix_by_codes(
                ddi_arr,
                ordered_vocabulary_codes(source_voc),
                ordered_vocabulary_codes(voc),
            )
            dump_pickle(output_root / "ddi_A_final.pkl", ddi_arr)
            graph_audit["ddi_source_vocab"] = str(ddi_source_vocab)
            graph_audit["ddi_alignment"] = alignment
            graph_audit["ddi_alignment_status"] = "aligned_by_medication_code"
        else:
            shutil.copy2(ddi_source, output_root / "ddi_A_final.pkl")
            graph_audit["ddi_alignment_status"] = "not_checked_no_source_vocabulary"
    else:
        ddi_arr = np.zeros((med_voc_size, med_voc_size), dtype=np.float64)
        dump_pickle(output_root / "ddi_A_final.pkl", ddi_arr)
        graph_audit["ddi_source"] = "zero_matrix"
        graph_audit["ddi_alignment_status"] = "not_applicable_zero_matrix"
    if ddi_arr.shape != (med_voc_size, med_voc_size):
        raise ValueError(f"DDI shape {ddi_arr.shape} does not match medication vocab size {med_voc_size}")

    if args.ehr_source:
        ehr_source = resolve_path(args.ehr_source)
        shutil.copy2(ehr_source, output_root / "ehr.pkl")
        ehr_arr = np.asarray(load_pickle(output_root / "ehr.pkl"))
        graph_audit["ehr_source"] = str(ehr_source)
    else:
        ehr_arr = build_ehr_adjacency(records, med_voc_size)
        dump_pickle(output_root / "ehr.pkl", ehr_arr)
        graph_audit["ehr_source"] = "rebuilt_binary_cooccurrence"
    if ehr_arr.shape != (med_voc_size, med_voc_size):
        raise ValueError(f"EHR shape {ehr_arr.shape} does not match medication vocab size {med_voc_size}")

    graph_audit["ddi_shape"] = list(ddi_arr.shape)
    graph_audit["ehr_shape"] = list(ehr_arr.shape)
    graph_audit["ddi_nonzero"] = int(np.count_nonzero(ddi_arr))
    graph_audit["ehr_nonzero"] = int(np.count_nonzero(ehr_arr))
    return graph_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Carmen data_final_4.pkl into current package MIMIC-IV pkl artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="local Carmen data_final_4.pkl")
    parser.add_argument("--output_root", default="data/mimic-iv")
    parser.add_argument("--expected_patients", type=int, default=9036)
    parser.add_argument("--expected_visits", type=int, default=20616)
    parser.add_argument(
        "--allow_single_visit_patients",
        action="store_true",
        help="allow paper-style raw cohorts that keep single-visit patients before train-time paper_protocol filtering",
    )
    parser.add_argument("--ddi_source", default=None, help="optional DDI matrix to copy as ddi_A_final.pkl")
    parser.add_argument(
        "--ddi_source_vocab",
        default=None,
        help="voc_final.pkl defining --ddi_source row/column order; aligns DDI by medication code",
    )
    parser.add_argument("--ehr_source", default=None, help="optional EHR matrix to copy as ehr.pkl")
    parser.add_argument("--audit_name", default="carmen_mimic_iv_audit.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = resolve_path(args.source)
    output_root = resolve_path(args.output_root)
    frame = load_carmen_frame(source)
    source_audit = validate_source_frame(
        frame,
        args.expected_patients,
        args.expected_visits,
        args.allow_single_visit_patients,
    )
    voc = create_vocab(frame)
    records, visit_keys = create_records_and_visit_keys(frame, voc)
    record_audit = validate_records(records, visit_keys)
    med_voc_size = len(voc["med_voc"].idx2word)

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_root} already exists and is not empty; pass --overwrite")
    output_root.mkdir(parents=True, exist_ok=True)
    dump_pickle(output_root / "records_final.pkl", records)
    dump_pickle(output_root / "voc_final.pkl", voc)
    dump_pickle(output_root / "visit_keys.pkl", visit_keys)
    graph_audit = copy_or_build_graphs(args, output_root, records, voc)

    audit = {
        "source": str(source),
        "source_audit": source_audit,
        "record_audit": record_audit,
        "vocab_sizes": {
            "diag": len(voc["diag_voc"].idx2word),
            "proc": len(voc["pro_voc"].idx2word),
            "med": med_voc_size,
        },
        "graph_audit": graph_audit,
    }
    (output_root / args.audit_name).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[OK] built Carmen MIMIC-IV artifacts "
        f"patients={record_audit['record_patient_count']} visits={record_audit['record_visit_count']} "
        f"output={output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
