from __future__ import annotations

import torch
import torch.nn.functional as F

from action_transition import (
    action_transition_loss,
    build_action_transition_labels,
    masked_bce_loss,
)


def masked_multilabel_bce(logits, targets, mask):
    raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = mask.to(dtype=raw.dtype)
    denom = weights.sum()
    if denom.item() == 0:
        return raw.new_tensor(0.0)
    return (raw * weights).sum() / denom


def masked_transition_cross_entropy(logits, targets, mask):
    weights = mask.to(dtype=logits.dtype)
    if weights.sum().item() == 0:
        return logits.new_tensor(0.0)
    raw = F.cross_entropy(logits, targets, reduction="none")
    return (raw * weights).sum() / weights.sum()


previous_visit_action_labels = build_action_transition_labels


def previous_visit_count_labels(previous_meds, current_meds):
    labels = previous_visit_action_labels(previous_meds, current_meds)
    previous = labels["previous_mask"].to(dtype=current_meds.dtype)
    return {
        "total": current_meds.to(dtype=current_meds.dtype).sum(dim=1),
        "previous": previous.sum(dim=1),
        "continue": labels["continue"].sum(dim=1),
        "add": labels["add"].sum(dim=1),
        "remove": labels["remove"].sum(dim=1),
    }


masked_binary_bce = masked_bce_loss


def action_count_loss(
    total_count_pred,
    continue_count_pred,
    add_count_pred,
    remove_count_pred,
    previous_meds,
    current_meds,
    total_weight=1.0,
    continue_weight=1.0,
    add_weight=1.0,
    remove_weight=1.0,
):
    labels = previous_visit_count_labels(
        previous_meds.to(device=total_count_pred.device, dtype=total_count_pred.dtype),
        current_meds.to(device=total_count_pred.device, dtype=total_count_pred.dtype),
    )
    total = total_count_pred.new_tensor(0.0)
    weight_total = total_count_pred.new_tensor(0.0)
    components = (
        ("total", total_count_pred, total_weight),
        ("continue", continue_count_pred, continue_weight),
        ("add", add_count_pred, add_weight),
        ("remove", remove_count_pred, remove_weight),
    )
    for name, pred, weight in components:
        if weight <= 0:
            continue
        target = labels[name].to(device=pred.device, dtype=pred.dtype)
        total = total + float(weight) * F.smooth_l1_loss(pred.view_as(target), target)
        weight_total = weight_total + float(weight)
    if weight_total.item() == 0:
        return total_count_pred.sum() * 0.0
    return total / weight_total.clamp_min(1.0)


def action_ddi_add_loss(
    add_logits,
    previous_meds,
    ddi_adj,
    add_ddi_loss_weight=1.0,
    ddi_risk_threshold=0.0,
):
    previous = previous_meds.to(device=add_logits.device, dtype=add_logits.dtype)
    non_previous = 1.0 - previous.clamp(0.0, 1.0)
    weights = non_previous
    denom = weights.sum()
    if denom.item() == 0 or add_ddi_loss_weight <= 0:
        return add_logits.sum() * 0.0

    ddi_matrix = ddi_adj.to(device=add_logits.device, dtype=add_logits.dtype)
    ddi_matrix = torch.maximum(ddi_matrix, ddi_matrix.t())
    pair_risk = previous.unsqueeze(1) * ddi_matrix.unsqueeze(0)
    risk = pair_risk.max(dim=2).values.clamp_min(0.0)
    if ddi_risk_threshold > 0:
        risk = torch.where(
            risk > float(ddi_risk_threshold),
            risk,
            torch.zeros_like(risk),
        )
    return float(add_ddi_loss_weight) * (torch.sigmoid(add_logits) * risk * weights).sum() / denom.clamp_min(1.0)
