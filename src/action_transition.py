from __future__ import annotations

import torch
import torch.nn.functional as F


ACTION_NAMES = ("continue", "add", "remove")


def build_action_transition_labels(previous_meds, current_meds):
    previous = previous_meds > 0
    current = current_meds > 0
    return {
        "continue": (previous & current).to(dtype=current_meds.dtype),
        "add": ((~previous) & current).to(dtype=current_meds.dtype),
        "remove": (previous & (~current)).to(dtype=current_meds.dtype),
        "previous_mask": previous.to(dtype=current_meds.dtype),
        "non_previous_mask": (~previous).to(dtype=current_meds.dtype),
    }


def masked_bce_loss(logits, labels, mask, pos_weight=1.0, focal_gamma=0.0):
    weights = mask.to(device=logits.device, dtype=logits.dtype)
    denom = weights.sum()
    if denom.item() == 0:
        return logits.sum() * 0.0

    labels = labels.to(device=logits.device, dtype=logits.dtype)
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if pos_weight != 1.0:
        positive_weights = torch.where(
            labels > 0,
            logits.new_full((), float(pos_weight)),
            logits.new_tensor(1.0),
        )
        weights = weights * positive_weights
    if focal_gamma > 0.0:
        prob = torch.sigmoid(logits)
        pt = torch.where(labels > 0, prob, 1.0 - prob)
        weights = weights * (1.0 - pt).clamp_min(0.0).pow(float(focal_gamma))
    return (raw * weights).sum() / denom.clamp_min(1.0)


def action_transition_loss(
    continue_logits,
    add_logits,
    remove_logits,
    previous_meds,
    current_meds,
    continue_weight=1.0,
    add_weight=1.0,
    remove_weight=1.0,
    add_pos_weight=1.0,
    add_focal_gamma=0.0,
):
    labels = build_action_transition_labels(
        previous_meds.to(device=add_logits.device, dtype=add_logits.dtype),
        current_meds.to(device=add_logits.device, dtype=add_logits.dtype),
    )
    total = add_logits.new_tensor(0.0)
    weight_total = add_logits.new_tensor(0.0)
    if continue_weight > 0:
        total = total + float(continue_weight) * masked_bce_loss(
            continue_logits,
            labels["continue"],
            labels["previous_mask"],
        )
        weight_total = weight_total + float(continue_weight)
    if add_weight > 0:
        total = total + float(add_weight) * masked_bce_loss(
            add_logits,
            labels["add"],
            labels["non_previous_mask"],
            pos_weight=add_pos_weight,
            focal_gamma=add_focal_gamma,
        )
        weight_total = weight_total + float(add_weight)
    if remove_weight > 0:
        total = total + float(remove_weight) * masked_bce_loss(
            remove_logits,
            labels["remove"],
            labels["previous_mask"],
        )
        weight_total = weight_total + float(remove_weight)
    if weight_total.item() == 0:
        return add_logits.sum() * 0.0
    return total / weight_total.clamp_min(1.0)


previous_visit_action_labels = build_action_transition_labels
masked_binary_bce = masked_bce_loss
