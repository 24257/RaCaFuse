"""Shared IO helpers for baseline comparison scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def load_split_manifest(path: str) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"split manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def split_patients(
    data: list,
    seed: int,
    split_path: str = "",
    paper_protocol: bool = False,
) -> tuple[list, list, list]:
    splits, _ = split_patients_with_ids(data, seed, split_path, paper_protocol)
    return splits


def split_patients_with_ids(
    data: list,
    seed: int,
    split_path: str = "",
    paper_protocol: bool = False,
) -> tuple[tuple[list, list, list], tuple[list[int], list[int], list[int]]]:
    if paper_protocol:
        filtered = [patient for patient in data if len(patient) >= 2]
        if split_path:
            manifest = load_split_manifest(split_path)
            index_space = manifest.get("patient_index_space")
            if index_space != "paper_filtered":
                raise ValueError(
                    "--paper_protocol with --split_path requires "
                    "patient_index_space='paper_filtered'"
                )
            ids = _manifest_split_ids(manifest, len(filtered))
            splits = tuple([filtered[idx] for idx in split_ids] for split_ids in ids)
            return splits, ids
        split_point = int(len(filtered) * 2 / 3)
        eval_len = int(len(filtered[split_point:]) / 2)
        train = filtered[:split_point]
        valid = filtered[split_point + eval_len :]
        test = filtered[split_point : split_point + eval_len]
        train_ids = list(range(split_point))
        test_ids = list(range(split_point, split_point + eval_len))
        valid_ids = list(range(split_point + eval_len, len(filtered)))
        return (train, valid, test), (train_ids, valid_ids, test_ids)

    if split_path:
        manifest = load_split_manifest(split_path)
        index_space = manifest.get("patient_index_space", "raw")
        if index_space != "raw":
            raise ValueError(
                "split manifest patient_index_space must be 'raw' when paper protocol is disabled"
            )
        ids = _manifest_split_ids(manifest, len(data))
        splits = tuple([data[idx] for idx in split_ids] for split_ids in ids)
        return splits, ids

    rng = np.random.RandomState(seed)
    order = np.arange(len(data))
    rng.shuffle(order)
    shuffled = [data[idx] for idx in order]
    split_point = int(len(shuffled) * 2 / 3)
    eval_len = int(len(shuffled[split_point:]) / 2)
    train = shuffled[:split_point]
    test = shuffled[split_point : split_point + eval_len]
    valid = shuffled[split_point + eval_len :]
    train_ids = order[:split_point].astype(int).tolist()
    test_ids = order[split_point : split_point + eval_len].astype(int).tolist()
    valid_ids = order[split_point + eval_len :].astype(int).tolist()
    return (train, valid, test), (train_ids, valid_ids, test_ids)


def _manifest_split_ids(
    manifest: Mapping[str, Any],
    patient_count: int,
) -> tuple[list[int], list[int], list[int]]:
    keys = ("train_patient_indices", "valid_patient_indices", "test_patient_indices")
    ids = tuple([int(idx) for idx in manifest[key]] for key in keys)
    if not ids[0]:
        raise ValueError("split manifest train_patient_indices is empty")
    if not ids[1]:
        raise ValueError("split manifest valid_patient_indices is empty")
    all_ids = [idx for split_ids in ids for idx in split_ids]
    bad = [idx for idx in all_ids if idx < 0 or idx >= patient_count]
    if bad:
        raise ValueError(
            f"split manifest references patient index {bad[0]}, "
            f"but loaded data has only {patient_count} patients"
        )
    return ids


def output_dir_for(args: Any) -> Path | None:
    output_dir = getattr(args, "output_dir", "")
    if not output_dir:
        return None
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_id_for(args: Any) -> str:
    run_id = getattr(args, "run_id", "")
    if run_id:
        return run_id
    return f"{getattr(args, 'model_name', 'baseline')}_seed{getattr(args, 'seed', 'unknown')}"


SUPPORTED_DATASETS = {
    "mimic-iii",
    "mimic-iv",
    "eicu",
    "tempt_hicl_treatment",
    "resmr_drugname_treatment",
    "pyhealth_physicalexam_drugname",
}


def configure_dataset_args(args: Any, default_dataset: str) -> Any:
    dataset = getattr(args, "dataset", "") or default_dataset
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    args.dataset = dataset
    args.MIMIC = 4 if dataset == "mimic-iv" else 3
    return args


def target_start_visit_for(args: Any) -> int:
    explicit = getattr(args, "target_start_visit", -1)
    if explicit is not None and int(explicit) >= 0:
        return int(explicit)
    return 1 if getattr(args, "paper_protocol", False) else 0


def records_filename(args: Any) -> str:
    return "records_final_4.pkl" if getattr(args, "MIMIC", 3) == 4 else "records_final.pkl"


def voc_filename(args: Any) -> str:
    return "voc_final_4.pkl" if getattr(args, "MIMIC", 3) == 4 else "voc_final.pkl"


def ddi_filename(args: Any) -> str:
    return "ddi_A_final_4.pkl" if getattr(args, "MIMIC", 3) == 4 else "ddi_A_final.pkl"


def existing_dataset_file(datadir: str, *names: str) -> str:
    for name in names:
        if (Path(datadir) / name).exists():
            return name
    raise FileNotFoundError(f"none of {names} found under {datadir}")


def checkpoint_path_for(args: Any, filename: str = "best.model") -> Path:
    output_dir = output_dir_for(args)
    if output_dir is not None:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir / filename
    path = Path("saved") / getattr(args, "model_name", "baseline")
    path.mkdir(parents=True, exist_ok=True)
    return path / filename


def save_command_metadata(args: Any) -> None:
    output_dir = output_dir_for(args)
    if output_dir is None:
        return
    payload = {
        "argv": sys.argv,
        "args": vars(args),
        "run_id": run_id_for(args),
    }
    phase = (
        "validation_only"
        if getattr(args, "validation_only_no_test", False)
        else ("test" if getattr(args, "Test", False) else "train")
    )
    (output_dir / f"command_{phase}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command_path = output_dir / "command.json"
    if not command_path.exists():
        command_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_prediction_artifact(
    args: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metadata: dict[str, Any] | None = None,
    extra_arrays: Mapping[str, np.ndarray] | None = None,
) -> Path | None:
    if not getattr(args, "save_predictions", False):
        return None
    output_dir = output_dir_for(args)
    if output_dir is None:
        raise ValueError("--save_predictions requires --output_dir")
    payload_metadata = {
        "schema_version": "baseline-prediction-v1",
        "model": getattr(args, "model_name", ""),
        "seed": getattr(args, "seed", None),
        "dataset": getattr(args, "dataset", ""),
        "split_path": getattr(args, "split_path", ""),
        "target_start_visit": target_start_visit_for(args),
        "run_id": run_id_for(args),
    }
    if metadata:
        payload_metadata.update(metadata)
    path = output_dir / "predictions.npz"
    arrays = {
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "y_prob": np.asarray(y_prob),
        "metadata": np.array(payload_metadata, dtype=object),
        "dataset": np.array(payload_metadata.get("dataset", ""), dtype=object),
    }
    for key, value in (extra_arrays or {}).items():
        if key in arrays:
            raise ValueError(f"extra prediction array collides with reserved field: {key}")
        arrays[key] = np.asarray(value)
    np.savez_compressed(path, **arrays)
    return path
