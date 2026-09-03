"""Treatment episode retrieval utilities.

The existing static retrieval path ranks historical visits by state similarity.
This module ranks clinical treatment episodes: state, action, subsequent change,
and safety burden are scored together before producing medication priors or
Top-K retrieval context tensors.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix


DEFAULT_EPISODE_CONFIG = {
    "state_weight": 1.0,
    "dxproc_weight": 1.0,
    "clinical_weight": 0.0,
    "history_weight": 0.25,
    "action_weight": 0.0,
    "outcome_weight": 0.25,
    "safety_weight": 0.25,
    "score_transform": "minmax",
    "min_outcome_mask": 1.0,
    "chunk_size": 256,
    "history_mode": "cumulative",
    "history_similarity_mode": "current",
    "action_similarity_mode": "transition_jaccard",
}

EPISODE_EXPERIENCE_SCALAR_DIM = 8


def episode_config(**overrides: Any) -> Dict[str, Any]:
    config = dict(DEFAULT_EPISODE_CONFIG)
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def resolve_patient_id_offsets(data_subset, patient_id_offsets=None):
    if patient_id_offsets is None:
        return list(range(len(data_subset)))
    if len(patient_id_offsets) != len(data_subset):
        raise ValueError("patient_id_offsets must align with data_subset")
    return list(patient_id_offsets)


def _empty_matrix(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols), dtype=np.float32)


def _infer_payload_dim(payload, part_idx: int = 0) -> int:
    if payload is None:
        return 0
    for patient in payload[part_idx]:
        for visit_value in patient:
            return int(np.asarray(visit_value, dtype=np.float32).reshape(-1).shape[0])
    return 0


def _payload_vector(payload, part_idx: int, patient_idx: int, visit_idx: int, dim: int) -> np.ndarray:
    if payload is None or dim <= 0:
        return np.zeros(dim, dtype=np.float32)
    patient_part = payload[part_idx][patient_idx]
    if visit_idx >= len(patient_part):
        return np.zeros(dim, dtype=np.float32)
    value = np.asarray(patient_part[visit_idx], dtype=np.float32).reshape(-1)
    if value.shape[0] == dim:
        return value
    padded = np.zeros(dim, dtype=np.float32)
    width = min(dim, value.shape[0])
    padded[:width] = value[:width]
    return padded


def _as_ddi_matrix(ddi_adj):
    if ddi_adj is None:
        return None
    if hasattr(ddi_adj, "toarray"):
        ddi_adj = ddi_adj.toarray()
    return np.asarray(ddi_adj, dtype=np.float32)


def _ddi_burden(med_vector: np.ndarray, ddi_matrix: Optional[np.ndarray]) -> float:
    if ddi_matrix is None:
        return 0.0
    med_indexes = np.flatnonzero(med_vector > 0)
    if med_indexes.shape[0] < 2:
        return 0.0
    submatrix = ddi_matrix[np.ix_(med_indexes, med_indexes)]
    pair_count = float(med_indexes.shape[0] * (med_indexes.shape[0] - 1))
    if pair_count <= 0:
        return 0.0
    return float(submatrix.sum() / pair_count)


def _normalise_candidate_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    finite = np.isfinite(scores)
    clean = np.where(finite, scores, 0.0).astype(np.float32)
    if mode == "positive":
        normalised = np.maximum(clean, 0.0)
    elif mode == "sigmoid":
        clipped = np.clip(clean, -30.0, 30.0)
        normalised = 1.0 / (1.0 + np.exp(-clipped))
    elif mode == "minmax":
        normalised = np.zeros_like(clean, dtype=np.float32)
        for row_idx in range(clean.shape[0]):
            row_finite = finite[row_idx]
            if not row_finite.any():
                continue
            row_values = clean[row_idx, row_finite]
            row_min = float(row_values.min())
            row_max = float(row_values.max())
            if row_max - row_min <= 1e-6:
                normalised[row_idx, row_finite] = 1.0
            else:
                normalised[row_idx, row_finite] = (row_values - row_min) / (row_max - row_min)
    else:
        raise ValueError(f"Unsupported treatment episode score transform: {mode}")
    normalised[~finite] = 0.0
    return normalised.astype(np.float32)


def _select_topk(scores: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    candidate_count = scores.shape[1]
    if topk <= 0:
        raise ValueError("retrieval_topk must be positive when treatment episode retrieval is enabled")
    if candidate_count <= 0:
        raise ValueError("treatment episode retrieval requires at least one candidate episode")
    topk = min(topk, candidate_count)
    kth = max(candidate_count - topk, 0)
    top_indexes = np.argpartition(scores, kth, axis=1)[:, kth:]
    top_scores = np.take_along_axis(scores, top_indexes, axis=1)
    order = np.argsort(-top_scores, axis=1)
    top_indexes = np.take_along_axis(top_indexes, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)
    return top_indexes.astype(np.int64), top_scores.astype(np.float32)


def build_episode_inputs(
    data_subset,
    voc_size,
    clinical_payload=None,
    clinical_state_payload=None,
    ddi_adj=None,
    patient_id_offsets=None,
    start_visit: int = 0,
    history_mode: str = "cumulative",
):
    if history_mode not in {"cumulative", "immediate"}:
        raise ValueError(f"Unsupported treatment episode history mode: {history_mode}")
    d_voc_size, p_voc_size, m_voc_size = voc_size
    patient_id_offsets = resolve_patient_id_offsets(data_subset, patient_id_offsets)
    clinical_dim = _infer_payload_dim(clinical_payload, 0)
    state_dim = _infer_payload_dim(clinical_state_payload, 0)
    ddi_matrix = _as_ddi_matrix(ddi_adj)

    dxproc_rows = []
    dxproc_cols = []
    meds = []
    history_meds = []
    added_meds = []
    continued_meds = []
    removed_meds = []
    clinical_values = []
    clinical_masks = []
    outcome_delta = []
    outcome_masks = []
    outcome_scores = []
    outcome_mask_counts = []
    clinical_worsening_rates = []
    ddi_risks = []
    patient_ids = []
    visit_ids = []
    row_idx = 0

    for patient_idx, patient in enumerate(data_subset):
        global_patient_idx = patient_id_offsets[patient_idx]
        cumulative_history_vector = np.zeros(m_voc_size, dtype=np.float32)
        immediate_history_vector = np.zeros(m_voc_size, dtype=np.float32)
        for previous_visit in patient[:start_visit]:
            previous_vector = np.zeros(m_voc_size, dtype=np.float32)
            if previous_visit[2]:
                previous_vector[np.asarray(previous_visit[2], dtype=np.int64)] = 1.0
                cumulative_history_vector[np.asarray(previous_visit[2], dtype=np.int64)] = 1.0
            immediate_history_vector = previous_vector

        for visit_idx in range(start_visit, len(patient)):
            diag_codes, proc_codes, med_codes = patient[visit_idx]
            dxproc_rows.extend([row_idx] * (len(diag_codes) + len(proc_codes)))
            dxproc_cols.extend(list(diag_codes) + [d_voc_size + code for code in proc_codes])

            med_vector = np.zeros(m_voc_size, dtype=np.float32)
            if med_codes:
                med_vector[np.asarray(med_codes, dtype=np.int64)] = 1.0
            meds.append(med_vector)
            history_vector = (
                immediate_history_vector.copy()
                if history_mode == "immediate"
                else cumulative_history_vector.copy()
            )
            history_meds.append(history_vector.copy())
            added_meds.append(np.maximum(med_vector - history_vector, 0.0))
            continued_meds.append(med_vector * history_vector)
            removed_meds.append(np.maximum(history_vector - med_vector, 0.0))
            patient_ids.append(global_patient_idx)
            visit_ids.append(visit_idx)

            clinical_values.append(_payload_vector(clinical_payload, 0, patient_idx, visit_idx, clinical_dim))
            clinical_masks.append(_payload_vector(clinical_payload, 1, patient_idx, visit_idx, clinical_dim))

            current_state = _payload_vector(clinical_state_payload, 0, patient_idx, visit_idx, state_dim)
            current_mask = _payload_vector(clinical_state_payload, 1, patient_idx, visit_idx, state_dim)
            if visit_idx + 1 < len(patient):
                next_state = _payload_vector(clinical_state_payload, 0, patient_idx, visit_idx + 1, state_dim)
                next_mask = _payload_vector(clinical_state_payload, 1, patient_idx, visit_idx + 1, state_dim)
            else:
                next_state = np.zeros(state_dim, dtype=np.float32)
                next_mask = np.zeros(state_dim, dtype=np.float32)
            common_mask = current_mask * next_mask
            next_only_mask = next_mask * (1.0 - common_mask)
            transition_delta = (current_state - next_state) * common_mask
            # If the initial state is missing, the follow-up abnormal state is still
            # useful as a conservative negative outcome signal.
            followup_status_delta = -next_state * next_only_mask
            delta = transition_delta + followup_status_delta
            worsening = (
                np.maximum(next_state - current_state, 0.0) * common_mask
                + next_state * next_only_mask
            )
            mask_count = float((common_mask + next_only_mask).sum())
            outcome_delta.append(delta)
            outcome_masks.append(common_mask)
            outcome_mask_counts.append(mask_count)
            if mask_count > 0:
                outcome_scores.append(float(delta.sum() / mask_count))
                clinical_worsening_rates.append(float(worsening.sum() / mask_count))
            else:
                outcome_scores.append(0.0)
                clinical_worsening_rates.append(0.0)
            ddi_risks.append(_ddi_burden(med_vector, ddi_matrix))

            if med_codes:
                cumulative_history_vector[np.asarray(med_codes, dtype=np.int64)] = 1.0
            immediate_history_vector = med_vector.copy()
            row_idx += 1

    dxproc = csr_matrix(
        (np.ones(len(dxproc_rows), dtype=np.float32), (dxproc_rows, dxproc_cols)),
        shape=(row_idx, d_voc_size + p_voc_size),
        dtype=np.float32,
    )
    if row_idx == 0:
        med_matrix = _empty_matrix(0, m_voc_size)
        clinical_matrix = _empty_matrix(0, clinical_dim)
        state_matrix = _empty_matrix(0, state_dim)
    else:
        med_matrix = np.stack(meds, axis=0).astype(np.float32)
        clinical_matrix = np.stack(clinical_values, axis=0).astype(np.float32)
        state_matrix = np.stack(outcome_delta, axis=0).astype(np.float32)

    return {
        "dxproc": dxproc,
        "dxproc_count": np.asarray(dxproc.sum(axis=1)).reshape(-1).astype(np.float32),
        "meds": med_matrix,
        "history_meds": np.stack(history_meds, axis=0).astype(np.float32) if row_idx else _empty_matrix(0, m_voc_size),
        "added_meds": np.stack(added_meds, axis=0).astype(np.float32) if row_idx else _empty_matrix(0, m_voc_size),
        "continued_meds": np.stack(continued_meds, axis=0).astype(np.float32) if row_idx else _empty_matrix(0, m_voc_size),
        "removed_meds": np.stack(removed_meds, axis=0).astype(np.float32) if row_idx else _empty_matrix(0, m_voc_size),
        "clinical_values": clinical_matrix,
        "clinical_masks": np.stack(clinical_masks, axis=0).astype(np.float32) if row_idx else _empty_matrix(0, clinical_dim),
        "outcome_delta": state_matrix,
        "outcome_masks": np.stack(outcome_masks, axis=0).astype(np.float32) if row_idx else _empty_matrix(0, state_dim),
        "outcome_score": np.asarray(outcome_scores, dtype=np.float32),
        "outcome_mask_count": np.asarray(outcome_mask_counts, dtype=np.float32),
        "clinical_worsening_rate": np.asarray(clinical_worsening_rates, dtype=np.float32),
        "ddi_risk": np.asarray(ddi_risks, dtype=np.float32),
        "patient_ids": np.asarray(patient_ids, dtype=np.int64),
        "visit_ids": np.asarray(visit_ids, dtype=np.int64),
    }


def compute_dxproc_similarity(query_inputs, candidate_inputs, start: int, end: int) -> np.ndarray:
    intersections = query_inputs["dxproc"][start:end].dot(candidate_inputs["dxproc"].T).toarray().astype(np.float32)
    query_counts = query_inputs["dxproc_count"][start:end].reshape(-1, 1)
    candidate_counts = candidate_inputs["dxproc_count"].reshape(1, -1)
    unions = query_counts + candidate_counts - intersections
    return np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float32),
        where=unions > 0,
    )


def compute_masked_dense_similarity(query_inputs, candidate_inputs, start: int, end: int) -> np.ndarray:
    if query_inputs["clinical_values"].shape[1] == 0 or candidate_inputs["clinical_values"].shape[1] == 0:
        return np.zeros((end - start, candidate_inputs["meds"].shape[0]), dtype=np.float32)
    query_values = query_inputs["clinical_values"][start:end]
    query_masks = query_inputs["clinical_masks"][start:end]
    candidate_values = candidate_inputs["clinical_values"]
    candidate_masks = candidate_inputs["clinical_masks"]
    query_masked_values = query_values * query_masks
    candidate_masked_values = candidate_values * candidate_masks
    numerator = np.matmul(query_masked_values, candidate_masked_values.T)
    query_sq_common = np.matmul((query_values * query_values * query_masks), candidate_masks.T)
    candidate_sq_common = np.matmul(query_masks, (candidate_values * candidate_values * candidate_masks).T)
    denominator = np.sqrt(np.maximum(query_sq_common, 0.0)) * np.sqrt(np.maximum(candidate_sq_common, 0.0))
    cosine = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 1e-8,
    )
    common_count = np.matmul(query_masks, candidate_masks.T)
    coverage = common_count / max(query_masks.shape[1], 1)
    return np.clip(cosine, 0.0, 1.0) * coverage


def compute_history_similarity(
    query_inputs,
    candidate_inputs,
    start: int,
    end: int,
    candidate_component: str = "meds",
) -> np.ndarray:
    if candidate_component not in {"meds", "history_meds"}:
        raise ValueError(f"Unsupported treatment episode history component: {candidate_component}")
    target_history = query_inputs["history_meds"][start:end]
    candidate_actions = candidate_inputs[candidate_component]
    intersections = np.matmul(target_history, candidate_actions.T)
    target_counts = target_history.sum(axis=1, keepdims=True)
    candidate_counts = candidate_actions.sum(axis=1, keepdims=True).T
    unions = target_counts + candidate_counts - intersections
    return np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float32),
        where=unions > 0,
    )


def _candidate_outcome_scores(candidate_inputs, config) -> np.ndarray:
    min_mask = float(config.get("min_outcome_mask", 0.0))
    scores = candidate_inputs["outcome_score"].astype(np.float32)
    return np.where(candidate_inputs["outcome_mask_count"] >= min_mask, scores, 0.0).astype(np.float32)


def _candidate_safety_penalties(candidate_inputs) -> np.ndarray:
    return (
        candidate_inputs["ddi_risk"].astype(np.float32)
        + candidate_inputs["clinical_worsening_rate"].astype(np.float32)
    )


def compute_action_similarity(
    query_inputs,
    candidate_inputs,
    start: int,
    end: int,
    target_action_prior,
    mode: str = "transition",
) -> np.ndarray:
    action_prior = np.asarray(target_action_prior, dtype=np.float32)[start:end]
    if mode == "current":
        candidate_counts = np.maximum(candidate_inputs["meds"].sum(axis=1).reshape(1, -1), 1.0)
        return np.matmul(action_prior, candidate_inputs["meds"].T) / candidate_counts
    if mode not in {"transition", "transition_jaccard", "transition_cosine"}:
        raise ValueError(f"Unsupported treatment episode action similarity mode: {mode}")

    target_history = query_inputs["history_meds"][start:end].astype(np.float32)
    target_not_history = np.maximum(1.0 - target_history, 0.0)
    add_signal = action_prior * target_not_history
    continue_signal = action_prior * target_history
    remove_signal = (1.0 - action_prior) * target_history
    candidate_action_counts = np.maximum(
        (
            candidate_inputs["added_meds"]
            + candidate_inputs["continued_meds"]
            + candidate_inputs["removed_meds"]
        ).sum(axis=1).reshape(1, -1),
        1.0,
    )
    action_intersection = (
        np.matmul(add_signal, candidate_inputs["added_meds"].T)
        + np.matmul(continue_signal, candidate_inputs["continued_meds"].T)
        + np.matmul(remove_signal, candidate_inputs["removed_meds"].T)
    )
    if mode == "transition_jaccard":
        target_action_counts = (
            add_signal.sum(axis=1, keepdims=True)
            + continue_signal.sum(axis=1, keepdims=True)
            + remove_signal.sum(axis=1, keepdims=True)
        )
        unions = target_action_counts + candidate_action_counts - action_intersection
        action_score = np.divide(
            action_intersection,
            unions,
            out=np.zeros_like(action_intersection, dtype=np.float32),
            where=unions > 1e-8,
        )
        return action_score.astype(np.float32)
    if mode == "transition_cosine":
        target_action_counts = (
            add_signal.sum(axis=1, keepdims=True)
            + continue_signal.sum(axis=1, keepdims=True)
            + remove_signal.sum(axis=1, keepdims=True)
        )
        denom = np.sqrt(np.maximum(target_action_counts, 0.0) * candidate_action_counts)
        action_score = np.divide(
            action_intersection,
            denom,
            out=np.zeros_like(action_intersection, dtype=np.float32),
            where=denom > 1e-8,
        )
        return action_score.astype(np.float32)
    action_score = action_intersection / candidate_action_counts
    return action_score.astype(np.float32)


def compute_episode_transfer_scores(
    query_inputs,
    candidate_inputs,
    start: int,
    end: int,
    config=None,
    target_action_prior=None,
) -> np.ndarray:
    config = episode_config(**(config or {}))
    dxproc_weight = float(config.get("dxproc_weight", 1.0))
    clinical_weight = float(config.get("clinical_weight", 0.0))
    state_weight = float(config.get("state_weight", 1.0))
    history_weight = float(config.get("history_weight", 0.0))
    action_weight = float(config.get("action_weight", 0.0))
    outcome_weight = float(config.get("outcome_weight", 0.0))
    safety_weight = float(config.get("safety_weight", 0.0))
    history_similarity_mode = config.get("history_similarity_mode", "current")
    if history_similarity_mode not in {"current", "history"}:
        raise ValueError(f"Unsupported treatment episode history similarity mode: {history_similarity_mode}")
    action_similarity_mode = config.get("action_similarity_mode", "transition_jaccard")
    if action_similarity_mode not in {"current", "transition", "transition_jaccard", "transition_cosine"}:
        raise ValueError(f"Unsupported treatment episode action similarity mode: {action_similarity_mode}")

    score = np.zeros((end - start, candidate_inputs["meds"].shape[0]), dtype=np.float32)
    state_parts = []
    state_weights = []
    if dxproc_weight > 0:
        state_parts.append(compute_dxproc_similarity(query_inputs, candidate_inputs, start, end))
        state_weights.append(dxproc_weight)
    if clinical_weight > 0:
        state_parts.append(compute_masked_dense_similarity(query_inputs, candidate_inputs, start, end))
        state_weights.append(clinical_weight)
    if state_parts and state_weight > 0:
        state_score = sum(weight * part for weight, part in zip(state_weights, state_parts)) / max(sum(state_weights), 1e-8)
        score += state_weight * state_score.astype(np.float32)
    if history_weight > 0:
        history_component = "history_meds" if history_similarity_mode == "history" else "meds"
        score += history_weight * compute_history_similarity(
            query_inputs,
            candidate_inputs,
            start,
            end,
            candidate_component=history_component,
        )
    if action_weight > 0 and target_action_prior is not None:
        score += action_weight * compute_action_similarity(
            query_inputs,
            candidate_inputs,
            start,
            end,
            target_action_prior,
            mode=action_similarity_mode,
        )
    if outcome_weight > 0:
        score += outcome_weight * _candidate_outcome_scores(candidate_inputs, config).reshape(1, -1)
    if safety_weight > 0:
        score -= safety_weight * _candidate_safety_penalties(candidate_inputs).reshape(1, -1)
    return score.astype(np.float32)


def apply_candidate_scope_mask(
    scores: np.ndarray,
    target_inputs,
    candidate_inputs,
    row_offset: int = 0,
    exclude_same_patient_current_and_future: bool = False,
    allow_train_or_own_history_only: bool = False,
    train_patient_id_limit: Optional[int] = None,
) -> np.ndarray:
    if not exclude_same_patient_current_and_future and not allow_train_or_own_history_only:
        return scores
    if allow_train_or_own_history_only and train_patient_id_limit is None:
        raise ValueError("train_patient_id_limit is required for train_plus_own_history retrieval")
    for local_row in range(scores.shape[0]):
        target_row = row_offset + local_row
        if exclude_same_patient_current_and_future:
            same_patient_future = (
                (candidate_inputs["patient_ids"] == target_inputs["patient_ids"][target_row])
                & (candidate_inputs["visit_ids"] >= target_inputs["visit_ids"][target_row])
            )
            scores[local_row, same_patient_future] = -np.inf
        if allow_train_or_own_history_only:
            allowed = (
                (candidate_inputs["patient_ids"] < train_patient_id_limit)
                | (
                    (candidate_inputs["patient_ids"] == target_inputs["patient_ids"][target_row])
                    & (candidate_inputs["visit_ids"] < target_inputs["visit_ids"][target_row])
                )
            )
            scores[local_row, ~allowed] = -np.inf
    return scores


def _global_prior(candidate_inputs, allow_train_or_own_history_only=False, train_patient_id_limit=None):
    if candidate_inputs["meds"].shape[0] == 0:
        return np.zeros(candidate_inputs["meds"].shape[1], dtype=np.float32)
    if allow_train_or_own_history_only:
        train_mask = candidate_inputs["patient_ids"] < train_patient_id_limit
        if train_mask.any():
            return candidate_inputs["meds"][train_mask].mean(axis=0).astype(np.float32)
    return candidate_inputs["meds"].mean(axis=0).astype(np.float32)


def _tensor_rows(values: np.ndarray, device=None):
    return [torch.tensor(row, dtype=torch.float32, device=device) for row in values.astype(np.float32)]


def episode_experience_dim(m_voc_size: int) -> int:
    return 4 * int(m_voc_size) + EPISODE_EXPERIENCE_SCALAR_DIM


def build_episode_experience_matrix(inputs) -> np.ndarray:
    med_size = int(inputs["meds"].shape[1])
    if inputs["meds"].shape[0] == 0:
        return np.zeros((0, episode_experience_dim(med_size)), dtype=np.float32)
    denom = float(max(med_size, 1))
    outcome_dim = float(max(inputs["outcome_masks"].shape[1], 1))
    scalars = np.stack(
        (
            inputs["outcome_score"],
            inputs["outcome_mask_count"] / outcome_dim,
            inputs["ddi_risk"],
            inputs["clinical_worsening_rate"],
            inputs["meds"].sum(axis=1) / denom,
            inputs["history_meds"].sum(axis=1) / denom,
            inputs["added_meds"].sum(axis=1) / denom,
            inputs["continued_meds"].sum(axis=1) / denom,
        ),
        axis=1,
    ).astype(np.float32)
    return np.concatenate(
        (
            inputs["history_meds"].astype(np.float32),
            inputs["added_meds"].astype(np.float32),
            inputs["continued_meds"].astype(np.float32),
            inputs["removed_meds"].astype(np.float32),
            scalars,
        ),
        axis=1,
    ).astype(np.float32)


def _build_inputs_pair(
    target_data,
    candidate_data,
    voc_size,
    target_clinical_payload=None,
    candidate_clinical_payload=None,
    target_clinical_state_payload=None,
    candidate_clinical_state_payload=None,
    ddi_adj=None,
    target_patient_id_offsets=None,
    candidate_patient_id_offsets=None,
    target_start_visit: int = 0,
    candidate_start_visit: Optional[int] = None,
    history_mode: str = "cumulative",
):
    if candidate_start_visit is None:
        candidate_start_visit = target_start_visit
    target_inputs = build_episode_inputs(
        target_data,
        voc_size,
        clinical_payload=target_clinical_payload,
        clinical_state_payload=target_clinical_state_payload,
        ddi_adj=ddi_adj,
        patient_id_offsets=target_patient_id_offsets,
        start_visit=target_start_visit,
        history_mode=history_mode,
    )
    candidate_inputs = build_episode_inputs(
        candidate_data,
        voc_size,
        clinical_payload=candidate_clinical_payload,
        clinical_state_payload=candidate_clinical_state_payload,
        ddi_adj=ddi_adj,
        patient_id_offsets=candidate_patient_id_offsets,
        start_visit=candidate_start_visit,
        history_mode=history_mode,
    )
    return target_inputs, candidate_inputs


def build_episode_retrieval_priors(
    target_data,
    candidate_data,
    voc_size,
    device=None,
    topk: int = 3,
    config=None,
    prior_transform: str = "centered",
    target_clinical_payload=None,
    candidate_clinical_payload=None,
    target_clinical_state_payload=None,
    candidate_clinical_state_payload=None,
    ddi_adj=None,
    exclude_same_patient_current_and_future: bool = False,
    allow_train_or_own_history_only: bool = False,
    train_patient_id_limit: Optional[int] = None,
    target_patient_id_offsets=None,
    candidate_patient_id_offsets=None,
    target_start_visit: int = 0,
    candidate_start_visit: Optional[int] = None,
    target_action_prior=None,
):
    config = episode_config(**(config or {}))
    history_mode = config.get("history_mode", "cumulative")
    target_inputs, candidate_inputs = _build_inputs_pair(
        target_data,
        candidate_data,
        voc_size,
        target_clinical_payload=target_clinical_payload,
        candidate_clinical_payload=candidate_clinical_payload,
        target_clinical_state_payload=target_clinical_state_payload,
        candidate_clinical_state_payload=candidate_clinical_state_payload,
        ddi_adj=ddi_adj,
        target_patient_id_offsets=target_patient_id_offsets,
        candidate_patient_id_offsets=candidate_patient_id_offsets,
        target_start_visit=target_start_visit,
        candidate_start_visit=candidate_start_visit,
        history_mode=history_mode,
    )
    global_prior = _global_prior(candidate_inputs, allow_train_or_own_history_only, train_patient_id_limit)
    priors = np.zeros((target_inputs["meds"].shape[0], voc_size[2]), dtype=np.float32)
    chunk_size = int(config.get("chunk_size", 256))
    for start in range(0, priors.shape[0], chunk_size):
        end = min(start + chunk_size, priors.shape[0])
        scores = compute_episode_transfer_scores(
            target_inputs,
            candidate_inputs,
            start,
            end,
            config=config,
            target_action_prior=target_action_prior,
        )
        scores = apply_candidate_scope_mask(
            scores,
            target_inputs,
            candidate_inputs,
            row_offset=start,
            exclude_same_patient_current_and_future=exclude_same_patient_current_and_future,
            allow_train_or_own_history_only=allow_train_or_own_history_only,
            train_patient_id_limit=train_patient_id_limit,
        )
        top_indexes, top_scores = _select_topk(scores, topk)
        weights = _normalise_candidate_scores(top_scores, config.get("score_transform", "minmax"))
        score_sums = weights.sum(axis=1, keepdims=True)
        for row in range(end - start):
            if score_sums[row, 0] <= 0:
                priors[start + row] = global_prior
                continue
            row_weights = weights[row] / score_sums[row, 0]
            priors[start + row] = np.matmul(row_weights, candidate_inputs["meds"][top_indexes[row]])
    if prior_transform == "centered":
        priors = priors - global_prior.reshape(1, -1)
    elif prior_transform != "positive":
        raise ValueError(f"Unsupported retrieval_prior_transform: {prior_transform}")
    return _tensor_rows(priors, device=device)


def build_episode_retrieval_context(
    target_data,
    candidate_data,
    voc_size,
    device=None,
    topk: int = 3,
    config=None,
    target_clinical_payload=None,
    candidate_clinical_payload=None,
    target_clinical_state_payload=None,
    candidate_clinical_state_payload=None,
    ddi_adj=None,
    exclude_same_patient_current_and_future: bool = False,
    allow_train_or_own_history_only: bool = False,
    train_patient_id_limit: Optional[int] = None,
    target_patient_id_offsets=None,
    candidate_patient_id_offsets=None,
    target_start_visit: int = 0,
    candidate_start_visit: Optional[int] = None,
    target_action_prior=None,
    return_metadata: bool = False,
):
    config = episode_config(**(config or {}))
    history_mode = config.get("history_mode", "cumulative")
    target_inputs, candidate_inputs = _build_inputs_pair(
        target_data,
        candidate_data,
        voc_size,
        target_clinical_payload=target_clinical_payload,
        candidate_clinical_payload=candidate_clinical_payload,
        target_clinical_state_payload=target_clinical_state_payload,
        candidate_clinical_state_payload=candidate_clinical_state_payload,
        ddi_adj=ddi_adj,
        target_patient_id_offsets=target_patient_id_offsets,
        candidate_patient_id_offsets=candidate_patient_id_offsets,
        target_start_visit=target_start_visit,
        candidate_start_visit=candidate_start_visit,
        history_mode=history_mode,
    )
    retrieval_meds = []
    retrieval_scores = []
    retrieval_experiences = []
    metadata_indexes = []
    metadata_scores = []
    metadata_normalised = []
    candidate_experiences = build_episode_experience_matrix(candidate_inputs)
    chunk_size = int(config.get("chunk_size", 256))
    for start in range(0, target_inputs["meds"].shape[0], chunk_size):
        end = min(start + chunk_size, target_inputs["meds"].shape[0])
        scores = compute_episode_transfer_scores(
            target_inputs,
            candidate_inputs,
            start,
            end,
            config=config,
            target_action_prior=target_action_prior,
        )
        scores = apply_candidate_scope_mask(
            scores,
            target_inputs,
            candidate_inputs,
            row_offset=start,
            exclude_same_patient_current_and_future=exclude_same_patient_current_and_future,
            allow_train_or_own_history_only=allow_train_or_own_history_only,
            train_patient_id_limit=train_patient_id_limit,
        )
        top_indexes, top_scores = _select_topk(scores, topk)
        normalised_scores = _normalise_candidate_scores(top_scores, config.get("score_transform", "minmax"))
        meds = candidate_inputs["meds"][top_indexes].astype(np.float32)
        invalid = ~np.isfinite(top_scores)
        if invalid.any():
            meds[invalid] = 0.0
            normalised_scores[invalid] = 0.0
        experiences = candidate_experiences[top_indexes].astype(np.float32)
        if invalid.any():
            experiences[invalid] = 0.0
        retrieval_meds.extend(torch.tensor(row, dtype=torch.float32, device=device) for row in meds)
        retrieval_scores.extend(torch.tensor(row, dtype=torch.float32, device=device) for row in normalised_scores)
        retrieval_experiences.extend(torch.tensor(row, dtype=torch.float32, device=device) for row in experiences)
        if return_metadata:
            metadata_indexes.append(top_indexes)
            metadata_scores.append(top_scores)
            metadata_normalised.append(normalised_scores)
    result = (retrieval_meds, retrieval_scores, None, retrieval_experiences)
    if not return_metadata:
        return result
    metadata = {
        "top_indexes": np.concatenate(metadata_indexes, axis=0) if metadata_indexes else np.zeros((0, topk), dtype=np.int64),
        "top_scores": np.concatenate(metadata_scores, axis=0) if metadata_scores else np.zeros((0, topk), dtype=np.float32),
        "normalised_scores": np.concatenate(metadata_normalised, axis=0) if metadata_normalised else np.zeros((0, topk), dtype=np.float32),
        "target_inputs": target_inputs,
        "candidate_inputs": candidate_inputs,
        "candidate_experiences": candidate_experiences,
        "experience_dim": int(candidate_experiences.shape[1]),
    }
    return result, metadata
