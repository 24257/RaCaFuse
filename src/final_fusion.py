"""Small, auditable implementation of the frozen SEER final fusion protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AFFINE_SLOPE_GRID = tuple(float(v) for v in np.round(np.arange(0.0, 2.01, 0.05), 2))
AFFINE_INTERCEPT_OFFSETS = tuple(float(v) for v in np.arange(-1.0, 1.01, 0.25))


@dataclass(frozen=True)
class AffineCountCalibrator:
    slope: float
    intercept: float
    med_dim: int

    def transform(self, counts: np.ndarray) -> np.ndarray:
        values = np.rint(self.slope * np.asarray(counts, dtype=np.float64) + self.intercept)
        return np.clip(values, 0, self.med_dim).astype(np.int32)


def fit_affine_count_calibrator(raw_count: np.ndarray, true_count: np.ndarray, med_dim: int) -> AffineCountCalibrator:
    raw = np.asarray(raw_count, dtype=np.float64)
    truth = np.asarray(true_count, dtype=np.float64)
    if raw.shape != truth.shape or raw.ndim != 1:
        raise ValueError("count calibration expects two equally shaped 1-D arrays")
    pairs = np.column_stack([raw.astype(np.int32), truth.astype(np.int32)])
    unique, frequency = np.unique(pairs, axis=0, return_counts=True)
    pair_raw = unique[:, 0].astype(np.float64)
    pair_truth = unique[:, 1].astype(np.float64)
    frequency = frequency.astype(np.float64)
    total = float(frequency.sum())
    candidates: list[tuple[tuple[float, ...], float, float]] = []
    for slope in AFFINE_SLOPE_GRID:
        center = float(np.median(truth - slope * raw))
        intercepts = {center + offset for offset in AFFINE_INTERCEPT_OFFSETS}
        if math.isclose(slope, 1.0):
            intercepts.add(0.0)
        for intercept in sorted(intercepts):
            predicted = np.clip(np.rint(slope * pair_raw + intercept), 0, med_dim)
            error = predicted - pair_truth
            mae = float(np.sum(frequency * np.abs(error)) / total)
            rmse = float(np.sqrt(np.sum(frequency * error**2) / total))
            bias = float(np.sum(frequency * error) / total)
            exact = float(np.sum(frequency * (error == 0)) / total)
            key = (mae, rmse, abs(bias), -exact, abs(slope - 1.0), abs(intercept), slope, intercept)
            candidates.append((key, slope, float(intercept)))
    _, slope, intercept = min(candidates, key=lambda item: item[0])
    return AffineCountCalibrator(float(slope), float(intercept), int(med_dim))


def logit(probability: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(p / (1.0 - p))


def row_zscore(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    mean = score.mean(axis=1, keepdims=True)
    std = score.std(axis=1, keepdims=True)
    return (score - mean) / np.maximum(std, 1e-8)


def ddi_symmetry(ddi_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(ddi_matrix)
    sym = ((matrix > 0) | (matrix.T > 0)).astype(np.int8)
    np.fill_diagonal(sym, 0)
    return sym


def greedy_ddi_decode(scores: np.ndarray, counts: np.ndarray, ddi_sym: np.ndarray, penalty: float, top_window: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    counts = np.clip(np.asarray(counts, dtype=np.int64), 0, scores.shape[1])
    result = np.zeros(scores.shape, dtype=np.int8)
    ddi_float = np.asarray(ddi_sym, dtype=np.float64)
    for row, count in enumerate(counts):
        k = int(count)
        if k <= 0:
            continue
        row_scores = scores[row]
        candidate_count = max(k, min(scores.shape[1], int(top_window)))
        candidates = np.argsort(-row_scores)[:candidate_count]
        available = np.zeros(scores.shape[1], dtype=bool)
        available[candidates] = True
        risk = np.zeros(scores.shape[1], dtype=np.float64)
        for _ in range(k):
            adjusted = np.where(available, row_scores - float(penalty) * risk, -np.inf)
            med = int(np.argmax(adjusted))
            if not np.isfinite(adjusted[med]):
                raise RuntimeError(f"DDI decoder exhausted candidates at row {row}")
            result[row, med] = 1
            available[med] = False
            risk += ddi_float[med]
    return result


def row_average_precision(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    truth = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    output = np.zeros(truth.shape[0], dtype=np.float64)
    for row in range(truth.shape[0]):
        order = np.argsort(-scores[row], kind="mergesort")
        ordered_score = scores[row, order]
        ordered_truth = truth[row, order].astype(np.int64)
        cumulative_true = np.cumsum(ordered_truth)
        group_ends = np.r_[ordered_score[:-1] != ordered_score[1:], True]
        end_index = np.flatnonzero(group_ends)
        end_true = cumulative_true[end_index]
        positives = int(end_true[-1]) if end_true.size else 0
        if positives:
            positive_in_group = np.diff(np.r_[0, end_true])
            output[row] = np.sum((end_true / (end_index + 1.0)) * positive_in_group) / positives
    return output


def set_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    intersection = np.logical_and(truth, pred).sum(axis=1).astype(np.float64)
    union = np.logical_or(truth, pred).sum(axis=1).astype(np.float64)
    pred_count = pred.sum(axis=1).astype(np.float64)
    true_count = truth.sum(axis=1).astype(np.float64)
    jaccard = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    precision = np.divide(intersection, pred_count, out=np.zeros_like(intersection), where=pred_count > 0)
    recall = np.divide(intersection, true_count, out=np.zeros_like(intersection), where=true_count > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    return {
        "jaccard": float(jaccard.mean()),
        "f1": float(f1.mean()),
        "prauc": float(row_average_precision(truth, scores).mean()),
        "count_mae": float(np.abs(pred_count - true_count).mean()),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
    }


def ddi_rate(y_pred: np.ndarray, ddi_sym: np.ndarray) -> float:
    pred = np.asarray(y_pred, dtype=np.float64)
    count = pred.sum(axis=1)
    denominator = np.sum(count * (count - 1.0) / 2.0)
    if denominator <= 0:
        return 0.0
    numerator = np.sum(np.sum((pred @ np.asarray(ddi_sym, dtype=np.float64)) * pred, axis=1) / 2.0)
    return float(numerator / denominator)


def fuse_with_calibrators(
    long_eval: dict[str, np.ndarray],
    auxiliary_eval: dict[str, np.ndarray],
    ddi_matrix: np.ndarray,
    long_cal: AffineCountCalibrator,
    auxiliary_cal: AffineCountCalibrator,
    *,
    score_long_weight: float = 0.3,
    count_long_weight: float = 0.4,
    count_offset: int = 1,
    ddi_penalty: float = 0.04,
    ddi_window: int = 60,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    for key in ("patient_ids", "visit_ids", "y_true"):
        if key in long_eval and key in auxiliary_eval and not np.array_equal(long_eval[key], auxiliary_eval[key]):
            raise ValueError(f"longitudinal and SEERAuxiliary eval components are not aligned: {key}")
    y_true = np.asarray(long_eval["y_true"], dtype=np.int8)
    med_dim = y_true.shape[1]
    if long_cal.med_dim != med_dim or auxiliary_cal.med_dim != med_dim:
        raise ValueError("count calibrator medication dimension does not match evaluation components")
    count_long = long_cal.transform(np.asarray(long_eval["y_pred"]).sum(axis=1))
    count_auxiliary = auxiliary_cal.transform(np.asarray(auxiliary_eval["y_pred"]).sum(axis=1))
    predicted_count = np.clip(np.rint(count_long_weight * count_long + (1.0 - count_long_weight) * count_auxiliary) + count_offset, 0, med_dim).astype(np.int16)
    ranking_score = score_long_weight * row_zscore(np.asarray(long_eval["final_decoder_score"], dtype=np.float64)) + (1.0 - score_long_weight) * row_zscore(logit(np.asarray(auxiliary_eval["y_prob"], dtype=np.float64)))
    ddi = ddi_symmetry(ddi_matrix)
    y_pred = greedy_ddi_decode(ranking_score, predicted_count, ddi, ddi_penalty, ddi_window)
    if not np.array_equal(y_pred.sum(axis=1), predicted_count):
        raise RuntimeError("decoded medication count does not match predicted count")
    metrics = set_metrics(y_true, y_pred, ranking_score)
    metrics["ddi_rate"] = ddi_rate(y_pred, ddi)
    arrays = {
        "patient_ids": np.asarray(long_eval.get("patient_ids", np.arange(y_true.shape[0]))),
        "visit_ids": np.asarray(long_eval.get("visit_ids", np.arange(y_true.shape[0]))),
        "y_true": y_true,
        "ranking_score": ranking_score.astype(np.float32),
        "predicted_count": predicted_count,
        "y_pred": y_pred.astype(np.int8),
    }
    return arrays, metrics


def fuse_components(long_train: dict[str, np.ndarray], auxiliary_train: dict[str, np.ndarray], long_eval: dict[str, np.ndarray], auxiliary_eval: dict[str, np.ndarray], ddi_matrix: np.ndarray, *, score_long_weight: float = 0.3, count_long_weight: float = 0.4, count_offset: int = 1, ddi_penalty: float = 0.04, ddi_window: int = 60) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if not np.array_equal(long_train["y_true"], auxiliary_train["y_true"]):
        raise ValueError("longitudinal and SEERAuxiliary train targets are not aligned")
    y_train = np.asarray(long_train["y_true"], dtype=np.int8)
    med_dim = np.asarray(long_eval["y_true"]).shape[1]
    long_cal = fit_affine_count_calibrator(np.asarray(long_train["y_pred"]).sum(axis=1), y_train.sum(axis=1), med_dim)
    auxiliary_cal = fit_affine_count_calibrator(np.asarray(auxiliary_train["y_pred"]).sum(axis=1), y_train.sum(axis=1), med_dim)
    return fuse_with_calibrators(
        long_eval, auxiliary_eval, ddi_matrix, long_cal, auxiliary_cal,
        score_long_weight=score_long_weight,
        count_long_weight=count_long_weight,
        count_offset=count_offset,
        ddi_penalty=ddi_penalty,
        ddi_window=ddi_window,
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
