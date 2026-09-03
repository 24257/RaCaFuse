"""Canonical SEER training and released-checkpoint evaluation protocol."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ARTIFACT_ROOT = Path(os.environ.get("SEER_PRIVATE_ARTIFACT_ROOT", str(ROOT))).expanduser().resolve()
ENGINE = ROOT / "src" / "trainer.py"
DATASET_IDS = {"mimic-iii": "3", "mimic-iv": "4", "eicu": "5"}
MODEL_CONFIG = json.loads((ROOT / "configs" / "final_model.json").read_text(encoding="utf-8"))
_shared = MODEL_CONFIG["shared_longitudinal_training"]
SHARED_TRAINING = {
    "dim": _shared["dim"],
    "batch": _shared["batch"],
    "visit": _shared["visit"],
    "weight_decay": _shared["weight_decay"],
    "bce": _shared["bce_weight"],
    "soft_jaccard": _shared["soft_jaccard_weight"],
    "pairwise_rank": _shared["pairwise_rank_weight"],
    "epochs": _shared["epochs"],
    "lr": _shared["lr"],
    "patience": _shared["early_stop_patience"],
}
STAGE_B = MODEL_CONFIG["longitudinal_stage_b"]

# Frozen final-test inference/export arguments. These are runtime/lineage
# arguments for evaluating final checkpoints; they are intentionally separate
# from SHARED_TRAINING and are not cross-dataset training hyperparameters.
FINAL_EXPORT_RUNTIME = {
    "mimic-iii": {"batch": 64, "weight_decay": 1e-4, "bce": 1.0, "soft_jaccard": 0.3, "pairwise_rank": 0.02},
    "mimic-iv": {"batch": 32, "weight_decay": 1e-6, "bce": 1.0, "soft_jaccard": 0.3, "pairwise_rank": 0.04},
    "eicu": {"batch": 32, "weight_decay": 1e-6, "bce": 0.7, "soft_jaccard": 0.2, "pairwise_rank": 0.02},
}


def _replace_option(command: list[str], option: str, value: object) -> None:
    index = command.index(option)
    command[index + 1] = str(value)


def training_split_path(dataset: str, seed: int) -> Path:
    return ROOT / "configs" / "splits" / "train_validation" / f"{dataset}_seed{seed}.json"


def final_split_path(dataset: str, seed: int) -> Path | None:
    return ROOT / "configs" / "splits" / f"eicu_seed{seed}_paper_filtered.json" if dataset == "eicu" else None


def ehr_cache_path(dataset: str, seed: int) -> Path:
    return PRIVATE_ARTIFACT_ROOT / "configs" / "frozen_inputs" / "ehr_graph" / dataset / f"seed{seed}"


def ddi_mask_path(dataset: str, seed: int) -> Path:
    return PRIVATE_ARTIFACT_ROOT / "configs" / "frozen_inputs" / "ddi_mask" / dataset / f"seed{seed}.pkl"


def _base_longitudinal_command(
    dataset: str,
    seed: int,
    data_root: Path,
    *,
    eval_split: str,
    ehr_cache: Path | None,
    training_structure: bool = False,
) -> list[str]:
    p = SHARED_TRAINING
    command = [
        sys.executable, str(ENGINE), "--dataset", DATASET_IDS[dataset],
        "--data_root_override", str(data_root), "--paper_protocol", "--eval_split", eval_split, "--seed", str(seed),
        "--dim", str(p["dim"]), "--batch", str(p["batch"]), "--visit", str(p["visit"]),
        "--metric_aggregation", "visit", "--best_metric", "ja", "--temporal_type", "mamba", "--med_temporal_type", "gru",
        "--patient_fusion_type", "sum", "--clinical_feature_mode", "none", "--ehr_graph_mode", "directed_weighted",
        "--weight_decay", str(p["weight_decay"]), "--bce_weight", str(p["bce"]),
        "--soft_jaccard_weight", str(p["soft_jaccard"]), "--pairwise_rank_weight", str(p["pairwise_rank"]),
        "--action_transition_mode", "none", "--action_transition_weight", "0.0",
        "--action_ddi_loss_weight", "0.0", "--soft_ddi_weight", "0.0", "--fusion_ddi_soft_weight", "0.0",
    ]
    if dataset == "eicu" and training_structure:
        command.append("--disable_procedure_branch")
    if ehr_cache is not None:
        command.extend(["--ehr_graph_cache_dir", str(ehr_cache)])
    return command


def _add_training_partition(command: list[str], dataset: str, seed: int) -> None:
    command.extend(["--split_path", str(training_split_path(dataset, seed)), "--validation_only_no_test"])


def stage_a_command(dataset: str, seed: int, data_root: Path, output: Path) -> list[str]:
    command = _base_longitudinal_command(
        dataset, seed, data_root, eval_split="eval",
        ehr_cache=output.parent / "cache" / "ehr_graph", training_structure=True,
    )
    _add_training_partition(command, dataset, seed)
    p = SHARED_TRAINING
    command.extend([
        "--retrieval_mode", "none", "--decode_mode", "threshold", "--eval_threshold", "0.3",
        "--epoches", str(p["epochs"]), "--lr", str(p["lr"]), "--early_stop_patience", str(p["patience"]),
        "--save_model", "1", "--save_model_selection", "best", "--save_model_path", str(output),
    ])
    return command


def stage_b_command(dataset: str, seed: int, data_root: Path, stage_a: Path, output: Path, cache_dir: Path) -> list[str]:
    command = _base_longitudinal_command(
        dataset, seed, data_root, eval_split="eval",
        ehr_cache=output.parent / "cache" / "ehr_graph", training_structure=True,
    )
    _add_training_partition(command, dataset, seed)
    command.extend([
        "--init_model_checkpoint", str(stage_a), "--retrieval_mode", str(STAGE_B["retrieval_mode"]),
        "--retrieval_candidate_scope", str(STAGE_B["retrieval_candidate_scope"]),
        "--retrieval_context_source", str(STAGE_B["retrieval_context_source"]), "--retrieval_weight", str(STAGE_B["retrieval_weight"]),
        "--retrieval_copy_mode", str(STAGE_B["retrieval_copy_mode"]), "--retrieval_copy_weight", str(STAGE_B["retrieval_copy_weight"]),
        "--retrieval_readout_mode", str(STAGE_B["retrieval_readout_mode"]), "--retrieval_readout_weight", str(STAGE_B["retrieval_readout_weight"]), "--freeze_base_for_retrieval_readout",
        "--paper_m3_checkpoint", str(stage_a), "--paper_m3_topk", str(STAGE_B["paper_m3_topk"]), "--paper_m3_cache_dir", str(cache_dir),
        "--paper_m3_clinical_weight", str(STAGE_B["paper_m3_clinical_weight"]), "--paper_m3_dxproc_weight", str(STAGE_B["paper_m3_dxproc_weight"]),
        "--decode_config_json", str(ROOT / "configs" / "decode_config.json"),
        "--epoches", str(STAGE_B["epochs"]), "--lr", str(STAGE_B["lr"]), "--early_stop_patience", str(STAGE_B["early_stop_patience"]),
        "--save_model", "1", "--save_model_selection", "best", "--save_model_path", str(output),
    ])
    return command


def longitudinal_export_command(dataset: str, seed: int, data_root: Path, stage_a: Path, stage_b: Path, split: str, output: Path, cache_dir: Path, cuda: int = 0) -> list[str]:
    command = _base_longitudinal_command(dataset, seed, data_root, eval_split=split, ehr_cache=ehr_cache_path(dataset, seed))
    command.extend(["--cuda", str(cuda)])
    runtime = FINAL_EXPORT_RUNTIME[dataset]
    _replace_option(command, "--batch", runtime["batch"])
    _replace_option(command, "--weight_decay", runtime["weight_decay"])
    _replace_option(command, "--bce_weight", runtime["bce"])
    _replace_option(command, "--soft_jaccard_weight", runtime["soft_jaccard"])
    _replace_option(command, "--pairwise_rank_weight", runtime["pairwise_rank"])
    if split == "train":
        _add_training_partition(command, dataset, seed)
    else:
        manifest = final_split_path(dataset, seed)
        if manifest is not None:
            command.extend(["--split_path", str(manifest)])
    command.extend([
        "--retrieval_mode", str(STAGE_B["retrieval_mode"]), "--retrieval_candidate_scope", str(STAGE_B["retrieval_candidate_scope"]),
        "--retrieval_context_source", str(STAGE_B["retrieval_context_source"]), "--retrieval_weight", str(STAGE_B["retrieval_weight"]),
        "--retrieval_copy_mode", str(STAGE_B["retrieval_copy_mode"]), "--retrieval_copy_weight", str(STAGE_B["retrieval_copy_weight"]),
        "--retrieval_readout_mode", str(STAGE_B["retrieval_readout_mode"]), "--retrieval_readout_weight", str(STAGE_B["retrieval_readout_weight"]),
        "--paper_m3_checkpoint", str(stage_a), "--paper_m3_topk", str(STAGE_B["paper_m3_topk"]), "--paper_m3_cache_dir", str(cache_dir),
        "--paper_m3_clinical_weight", str(STAGE_B["paper_m3_clinical_weight"]), "--paper_m3_dxproc_weight", str(STAGE_B["paper_m3_dxproc_weight"]),
        "--decode_config_json", str(ROOT / "configs" / "decode_config.json"),
        "--eval_only", "--eval_model_checkpoint", str(stage_b),
        "--fixed_component_export_no_metric_selection", "--component_output", str(output),
    ])
    return command
