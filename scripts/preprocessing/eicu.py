#!/usr/bin/env python3
"""Build a literature-aligned eICU medication-recommendation dataset.

The default recipe follows recent eICU medication-recommendation practice:
diagnosis labels from diagnosis.icd9code, medication labels from medication.gtc,
cancelled orders removed, GTC 0 removed, no real procedure feature, and only
patients with at least two valid ICU stays retained.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import dill
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
NO_PROC_TOKEN = "<NO_PROC>"

PAPER_STATS = {
    "DKINet": {"patients": 9583, "visits": 21018, "diag": 1371, "med": 45},
    "NLA-MMR": {"patients": 9539, "visits": 20912, "diag": 1371, "med": 31},
    "HypeMed": {"patients": 7855, "visits": 16869, "diag": 692, "med": 46},
}


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


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def dump_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        dill.dump(payload, fh)


def open_gzip_csv(path: Path):
    return gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace")


def clean_text(value: str | None) -> str:
    return (value or "").strip()


def parse_int(value: str | None, default: int = 0) -> int:
    value = clean_text(value)
    if not value:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def normalize_icd_code(code: str) -> str:
    return clean_text(code).upper().replace(".", "").replace(" ", "")


def split_icd_codes(value: str | None) -> list[str]:
    codes = []
    for part in re.split(r"[,;|]", value or ""):
        code = normalize_icd_code(part)
        if code and code not in {"NA", "NAN", "NULL", "NONE"}:
            codes.append(code)
    return codes


def normalize_gtc(value: str | None) -> str:
    value = clean_text(value)
    if not value:
        return ""
    try:
        numeric = float(value)
    except ValueError:
        return value
    if numeric.is_integer():
        return str(int(numeric))
    return value


def is_cancelled(value: str | None) -> bool:
    return clean_text(value).lower() in {"yes", "y", "true", "1"}


def select_top(counter: Counter, top_k: int) -> list[str]:
    items = sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    if top_k and top_k > 0:
        items = items[:top_k]
    return [str(code) for code, _ in items]


def read_patients(raw_root: Path) -> tuple[dict[int, str], dict[str, list[tuple[tuple[int, int, int, int], int]]], dict]:
    stay_to_patient = {}
    patient_stays = defaultdict(list)
    hospital_ids = set()
    unit_ids = set()
    row_count = 0
    missing_pid = 0
    with open_gzip_csv(raw_root / "patient.csv.gz") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_count += 1
            stay_id = int(row["patientunitstayid"])
            patient_id = clean_text(row.get("uniquepid"))
            if not patient_id:
                patient_id = clean_text(row.get("patienthealthsystemstayid")) or str(stay_id)
                missing_pid += 1
            order_key = (
                parse_int(row.get("hospitaldischargeyear")),
                parse_int(row.get("hospitaladmitoffset")),
                parse_int(row.get("unitvisitnumber")),
                stay_id,
            )
            stay_to_patient[stay_id] = patient_id
            patient_stays[patient_id].append((order_key, stay_id))
            if clean_text(row.get("hospitalid")):
                hospital_ids.add(clean_text(row.get("hospitalid")))
            if clean_text(row.get("wardid")):
                unit_ids.add(clean_text(row.get("wardid")))

    for patient_id in patient_stays:
        patient_stays[patient_id].sort()

    audit = {
        "patient_rows": row_count,
        "raw_unique_patients": len(patient_stays),
        "raw_stays": len(stay_to_patient),
        "raw_hospitals": len(hospital_ids),
        "raw_units": len(unit_ids),
        "missing_uniquepid_rows": missing_pid,
        "raw_multi_stay_patients": sum(1 for stays in patient_stays.values() if len(stays) >= 2),
        "raw_max_stays_per_patient": max((len(stays) for stays in patient_stays.values()), default=0),
    }
    return stay_to_patient, patient_stays, audit


def read_diagnoses(raw_root: Path) -> tuple[dict[int, set[str]], dict]:
    stay_diags = defaultdict(set)
    row_count = 0
    empty_rows = 0
    code_rows = 0
    with open_gzip_csv(raw_root / "diagnosis.csv.gz") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_count += 1
            stay_id = int(row["patientunitstayid"])
            codes = split_icd_codes(row.get("icd9code"))
            if not codes:
                empty_rows += 1
                continue
            code_rows += 1
            stay_diags[stay_id].update(codes)
    all_codes = {code for codes in stay_diags.values() for code in codes}
    audit = {
        "diagnosis_rows": row_count,
        "diagnosis_rows_with_icd": code_rows,
        "diagnosis_rows_without_icd": empty_rows,
        "diagnosis_stays_with_icd": len(stay_diags),
        "diagnosis_unique_codes_before_topk": len(all_codes),
        "diagnosis_code_source": "diagnosis.icd9code",
        "diagnosis_code_normalization": "upper-case, strip dots and spaces, split comma/semicolon/pipe lists",
    }
    return stay_diags, audit


def read_medications(raw_root: Path, keep_gtc_zero: bool) -> tuple[dict[int, set[str]], dict]:
    stay_meds = defaultdict(set)
    row_count = 0
    cancelled_rows = 0
    missing_gtc_rows = 0
    zero_gtc_rows = 0
    used_rows = 0
    all_gtc = set()
    with open_gzip_csv(raw_root / "medication.csv.gz") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_count += 1
            if is_cancelled(row.get("drugordercancelled")):
                cancelled_rows += 1
                continue
            gtc = normalize_gtc(row.get("gtc"))
            if not gtc:
                missing_gtc_rows += 1
                continue
            all_gtc.add(gtc)
            if gtc == "0":
                zero_gtc_rows += 1
                if not keep_gtc_zero:
                    continue
            stay_id = int(row["patientunitstayid"])
            stay_meds[stay_id].add(gtc)
            used_rows += 1
    used_gtc = {code for codes in stay_meds.values() for code in codes}
    audit = {
        "medication_rows": row_count,
        "medication_cancelled_rows_removed": cancelled_rows,
        "medication_missing_gtc_rows_removed": missing_gtc_rows,
        "medication_gtc_zero_rows": zero_gtc_rows,
        "medication_rows_after_filter": used_rows,
        "medication_stays_after_filter": len(stay_meds),
        "medication_unique_gtc_before_zero_filter": len(all_gtc),
        "medication_unique_gtc_after_filter": len(used_gtc),
        "medication_code_source": "medication.gtc",
        "gtc_zero_policy": "kept" if keep_gtc_zero else "removed",
    }
    return stay_meds, audit


def build_records(
    patient_stays: dict[str, list[tuple[tuple[int, int, int, int], int]]],
    stay_diags: dict[int, set[str]],
    stay_meds: dict[int, set[str]],
    diag_keep: set[str],
    med_keep: set[str],
    voc: dict[str, Voc],
    min_visits: int,
) -> tuple[list, list, dict]:
    records = []
    visit_keys = []
    one_valid_visit_patients = 0
    patients_with_any_valid_visit = 0
    valid_stays_before_min_visits = 0
    dropped_valid_stays_from_single_visit_patients = 0

    for patient_id in sorted(patient_stays, key=str):
        patient = []
        patient_keys = []
        for _, stay_id in patient_stays[patient_id]:
            diag_codes = sorted(stay_diags.get(stay_id, set()) & diag_keep)
            med_codes = sorted(stay_meds.get(stay_id, set()) & med_keep)
            if not diag_codes or not med_codes:
                continue
            visit = [
                [int(voc["diag_voc"].word2idx[code]) for code in diag_codes],
                [int(voc["pro_voc"].word2idx[NO_PROC_TOKEN])],
                [int(voc["med_voc"].word2idx[code]) for code in med_codes],
            ]
            patient.append(visit)
            patient_keys.append((str(patient_id), int(stay_id)))

        if patient:
            patients_with_any_valid_visit += 1
            valid_stays_before_min_visits += len(patient)
        if len(patient) == 1:
            one_valid_visit_patients += 1
            dropped_valid_stays_from_single_visit_patients += 1
        if len(patient) >= min_visits:
            records.append(patient)
            visit_keys.append(patient_keys)

    audit = {
        "patients_with_any_valid_visit_after_topk": patients_with_any_valid_visit,
        "valid_stays_before_min_visit_filter": valid_stays_before_min_visits,
        "patients_removed_by_min_visit_filter": one_valid_visit_patients if min_visits >= 2 else 0,
        "valid_stays_removed_by_min_visit_filter": dropped_valid_stays_from_single_visit_patients if min_visits >= 2 else 0,
        "record_patient_count": len(records),
        "record_visit_count": sum(len(patient) for patient in records),
        "target_visit_count_with_history": sum(max(0, len(patient) - 1) for patient in records),
        "min_visits_per_patient": min((len(patient) for patient in records), default=0),
        "max_visits_per_patient": max((len(patient) for patient in records), default=0),
        "avg_visits_per_patient": float(np.mean([len(patient) for patient in records])) if records else 0.0,
    }
    return records, visit_keys, audit


def build_ehr_adjacency(records: list, med_voc_size: int) -> np.ndarray:
    ehr_adj = np.zeros((med_voc_size, med_voc_size), dtype=np.float32)
    for patient in records:
        for visit in patient:
            meds = list(dict.fromkeys(int(med) for med in visit[2]))
            for i, med_i in enumerate(meds):
                for med_j in meds[i + 1 :]:
                    ehr_adj[med_i, med_j] = 1.0
                    ehr_adj[med_j, med_i] = 1.0
    return ehr_adj


def build_directed_ehr_graph(records: list, med_voc_size: int) -> np.ndarray:
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
    return co_counts / np.maximum(med_counts.reshape(-1, 1), 1.0)


def record_feature_stats(records: list) -> dict:
    visit_count = sum(len(patient) for patient in records)
    diag_lens = [len(visit[0]) for patient in records for visit in patient]
    med_lens = [len(visit[2]) for patient in records for visit in patient]
    target_med_lens = [len(visit[2]) for patient in records for visit in patient[1:]]
    return {
        "avg_diagnoses_per_visit": float(np.mean(diag_lens)) if diag_lens else 0.0,
        "avg_medications_per_visit": float(np.mean(med_lens)) if med_lens else 0.0,
        "avg_medications_per_target_visit": float(np.mean(target_med_lens)) if target_med_lens else 0.0,
        "record_visit_count_check": visit_count,
    }


def make_split_manifest(records: list, voc: dict[str, Voc], data_root: Path, seed: int) -> dict:
    patient_count = len(records)
    indices = list(range(patient_count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_count = int(patient_count * 2 / 3)
    valid_count = int(patient_count * 1 / 6)
    train = indices[:train_count]
    valid = indices[train_count : train_count + valid_count]
    test = indices[train_count + valid_count :]
    try:
        data_root_text = str(data_root.relative_to(REPO_ROOT))
    except ValueError:
        data_root_text = str(data_root)
    return {
        "dataset": "eicu",
        "data_root": data_root_text,
        "seed": seed,
        "split_policy": "seeded_shuffle",
        "holdout_order": "valid_first",
        "ratios": {"train": 2 / 3, "valid": 1 / 6, "test": 1 / 6},
        "summary": {
            "patient_count": patient_count,
            "visit_count": sum(len(patient) for patient in records),
            "target_visit_count_with_history": sum(max(0, len(patient) - 1) for patient in records),
            "vocab_sizes": {
                "diag": len(voc["diag_voc"].idx2word),
                "proc": len(voc["pro_voc"].idx2word),
                "med": len(voc["med_voc"].idx2word),
            },
        },
        "train_patient_indices": train,
        "valid_patient_indices": valid,
        "test_patient_indices": test,
    }


def write_splits(records: list, voc: dict[str, Voc], output_root: Path, seeds: list[int]) -> list[str]:
    written = []
    split_dir = output_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        manifest = make_split_manifest(records, voc, output_root, seed)
        output = split_dir / f"eicu_seed{seed}.json"
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(display_path(output))
    return written


def literature_comparison(stats: dict, voc: dict[str, Voc]) -> dict:
    current = {
        "patients": int(stats["record_patient_count"]),
        "visits": int(stats["record_visit_count"]),
        "diag": len(voc["diag_voc"].idx2word),
        "med": len(voc["med_voc"].idx2word),
    }
    comparison = {}
    for paper, target in PAPER_STATS.items():
        comparison[paper] = {
            metric: {"current": current[metric], "paper": target[metric], "delta": current[metric] - target[metric]}
            for metric in ["patients", "visits", "diag", "med"]
        }
    return comparison


def write_report(path: Path, audit: dict) -> None:
    stats = audit["record_audit"]
    vocab = audit["vocab_sizes"]
    feature = audit["feature_stats"]
    lines = [
        "# eICU Build Trial",
        "",
        "This build uses diagnosis ICD codes and medication GTC labels. eICU has no MIMIC-style procedure table, so `pro_voc` contains a single `<NO_PROC>` compatibility token.",
        "",
        "## Recipe",
        "",
        f"- Raw root: `{audit['raw_root']}`",
        f"- Diagnosis: `diagnosis.icd9code`, top {audit['selection']['diag_top_k']} codes",
        f"- Medication: `medication.gtc`, top {audit['selection']['med_top_k']} codes",
        f"- Cancelled medication orders removed: {audit['medication_audit']['medication_cancelled_rows_removed']}",
        f"- GTC 0 policy: {audit['medication_audit']['gtc_zero_policy']}",
        f"- Minimum valid visits per patient: {audit['selection']['min_visits_per_patient']}",
        f"- Split seeds: {', '.join(str(seed) for seed in audit['split_seeds'])}",
        "",
        "## Built Scale",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| patients | {stats['record_patient_count']} |",
        f"| visits | {stats['record_visit_count']} |",
        f"| target visits with history | {stats['target_visit_count_with_history']} |",
        f"| diagnoses | {vocab['diag']} |",
        f"| procedure compatibility tokens | {vocab['proc']} |",
        f"| medications | {vocab['med']} |",
        f"| avg visits / patient | {stats['avg_visits_per_patient']:.4f} |",
        f"| avg diagnoses / visit | {feature['avg_diagnoses_per_visit']:.4f} |",
        f"| avg medications / visit | {feature['avg_medications_per_visit']:.4f} |",
        f"| avg medications / target visit | {feature['avg_medications_per_target_visit']:.4f} |",
        "",
        "## Literature Comparison",
        "",
        "| Paper | patients | visits | diag | med |",
        "|---|---:|---:|---:|---:|",
    ]
    for paper, comp in audit["literature_comparison"].items():
        lines.append(
            "| {paper} | {cur_pat} vs {paper_pat} ({d_pat:+d}) | {cur_vis} vs {paper_vis} ({d_vis:+d}) | "
            "{cur_diag} vs {paper_diag} ({d_diag:+d}) | {cur_med} vs {paper_med} ({d_med:+d}) |".format(
                paper=paper,
                cur_pat=comp["patients"]["current"],
                paper_pat=comp["patients"]["paper"],
                d_pat=comp["patients"]["delta"],
                cur_vis=comp["visits"]["current"],
                paper_vis=comp["visits"]["paper"],
                d_vis=comp["visits"]["delta"],
                cur_diag=comp["diag"]["current"],
                paper_diag=comp["diag"]["paper"],
                d_diag=comp["diag"]["delta"],
                cur_med=comp["med"]["current"],
                paper_med=comp["med"]["paper"],
                d_med=comp["med"]["delta"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The public preprocessing descriptions are enough to reproduce the vocabulary convention, but not enough to reproduce the smaller DKINet/NLA-MMR patient and visit counts exactly. With the explicit rules above, the retained cohort is larger, which implies those papers likely used an additional unpublished DrugRec/SafeDrug-style cohort restriction or a preprocessed eICU release.",
            "",
            "For fair experiments, report this as an independently rebuilt eICU-GTC cohort unless an exact released preprocessing script is obtained.",
            "",
            "## Files",
            "",
        ]
    )
    for rel_path in audit["written_files"]:
        lines.append(f"- `{rel_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_seed_list(value: str) -> list[int]:
    if not clean_text(value):
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build package-ready eICU pkl artifacts for medication recommendation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--output_root", default="data/eicu")
    parser.add_argument("--diag_top_k", type=int, default=1371, help="0 or negative keeps all diagnosis codes")
    parser.add_argument("--med_top_k", type=int, default=45, help="0 or negative keeps all medication GTC codes")
    parser.add_argument("--keep_gtc_zero", action="store_true", help="keep GTC 0 instead of treating it as an invalid class")
    parser.add_argument("--min_visits", type=int, default=2)
    parser.add_argument("--split_seeds", default="1203,1204,1205")
    parser.add_argument("--audit_name", default="eicu_build_audit.json")
    parser.add_argument("--report", default="", help="optional report path; defaults inside --output_root")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_root = resolve_path(args.raw_root)
    output_root = resolve_path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_root} already exists and is not empty; pass --overwrite")
    output_root.mkdir(parents=True, exist_ok=True)

    print("[INFO] reading patient.csv.gz", flush=True)
    stay_to_patient, patient_stays, patient_audit = read_patients(raw_root)
    print("[INFO] reading diagnosis.csv.gz", flush=True)
    stay_diags, diagnosis_audit = read_diagnoses(raw_root)
    print("[INFO] reading medication.csv.gz", flush=True)
    stay_meds, medication_audit = read_medications(raw_root, keep_gtc_zero=args.keep_gtc_zero)

    base_stays = [stay_id for stay_id in stay_to_patient if stay_diags.get(stay_id) and stay_meds.get(stay_id)]
    diag_counts = Counter()
    med_counts = Counter()
    for stay_id in base_stays:
        diag_counts.update(stay_diags[stay_id])
        med_counts.update(stay_meds[stay_id])

    diag_codes = select_top(diag_counts, args.diag_top_k)
    med_codes = select_top(med_counts, args.med_top_k)
    diag_keep = set(diag_codes)
    med_keep = set(med_codes)

    voc = {"diag_voc": Voc(), "med_voc": Voc(), "pro_voc": Voc()}
    voc["diag_voc"].add_sentence(diag_codes)
    voc["med_voc"].add_sentence(med_codes)
    voc["pro_voc"].add_sentence([NO_PROC_TOKEN])

    print("[INFO] building records", flush=True)
    records, visit_keys, record_audit = build_records(
        patient_stays=patient_stays,
        stay_diags=stay_diags,
        stay_meds=stay_meds,
        diag_keep=diag_keep,
        med_keep=med_keep,
        voc=voc,
        min_visits=args.min_visits,
    )
    med_voc_size = len(voc["med_voc"].idx2word)

    print("[INFO] building graph placeholders", flush=True)
    ehr_adj = build_ehr_adjacency(records, med_voc_size)
    directed_ehr = build_directed_ehr_graph(records, med_voc_size)
    ddi_adj = np.zeros((med_voc_size, med_voc_size), dtype=np.float32)

    written_files = []
    for name, payload in [
        ("records_final.pkl", records),
        ("voc_final.pkl", voc),
        ("visit_keys.pkl", visit_keys),
        ("ehr.pkl", ehr_adj),
        ("ehr_adj_final.pkl", ehr_adj),
        ("weighted_confidence_directed_ehr_graph.pkl", directed_ehr),
        ("ddi_A_final.pkl", ddi_adj),
    ]:
        dump_pickle(output_root / name, payload)
        written_files.append(display_path(output_root / name))

    split_seeds = parse_seed_list(args.split_seeds)
    split_files = write_splits(records, voc, output_root, split_seeds)
    written_files.extend(split_files)

    audit = {
        "dataset": "eicu",
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "selection": {
            "diag_top_k": args.diag_top_k,
            "med_top_k": args.med_top_k,
            "keep_gtc_zero": bool(args.keep_gtc_zero),
            "min_visits_per_patient": args.min_visits,
            "procedure_policy": "single compatibility token; no real eICU procedure feature used",
            "base_stays_with_diagnosis_and_medication_before_topk": len(base_stays),
        },
        "patient_audit": patient_audit,
        "diagnosis_audit": diagnosis_audit,
        "medication_audit": medication_audit,
        "record_audit": record_audit,
        "feature_stats": record_feature_stats(records),
        "vocab_sizes": {
            "diag": len(voc["diag_voc"].idx2word),
            "proc": len(voc["pro_voc"].idx2word),
            "med": med_voc_size,
        },
        "graph_audit": {
            "ehr_source": "binary medication co-occurrence from rebuilt records",
            "ehr_shape": list(ehr_adj.shape),
            "ehr_nonzero": int(np.count_nonzero(ehr_adj)),
            "weighted_ehr_source": "directed confidence graph from rebuilt records",
            "weighted_ehr_nonzero": int(np.count_nonzero(directed_ehr)),
            "ddi_source": "zero placeholder; eICU DDI is not a reported metric",
            "ddi_shape": list(ddi_adj.shape),
            "ddi_nonzero": int(np.count_nonzero(ddi_adj)),
        },
        "literature_comparison": literature_comparison(record_audit, voc),
        "split_seeds": split_seeds,
        "written_files": written_files,
    }
    audit_path = output_root / args.audit_name
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written_files.append(display_path(audit_path))
    audit["written_files"] = written_files
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = resolve_path(args.report) if args.report else output_root / "eicu_build_report.md"
    write_report(report_path, audit)

    print(
        "[OK] built eICU artifacts "
        f"patients={record_audit['record_patient_count']} "
        f"visits={record_audit['record_visit_count']} "
        f"voc=({len(voc['diag_voc'].idx2word)}, {len(voc['pro_voc'].idx2word)}, {med_voc_size}) "
        f"output={output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
