from __future__ import annotations

import torch


def _require_action_logits(continue_logits, add_logits, remove_logits):
    if continue_logits is None or add_logits is None or remove_logits is None:
        raise ValueError("action-guided decoding requires continue/add/remove logits")


def score_fusion_logits(
    med_logits,
    continue_logits,
    add_logits,
    remove_logits,
    previous_meds,
    continue_weight=0.0,
    add_weight=0.0,
    remove_weight=0.0,
):
    _require_action_logits(continue_logits, add_logits, remove_logits)
    previous = previous_meds.to(device=med_logits.device, dtype=torch.bool)
    continue_logits = continue_logits.to(device=med_logits.device, dtype=med_logits.dtype)
    add_logits = add_logits.to(device=med_logits.device, dtype=med_logits.dtype)
    remove_logits = remove_logits.to(device=med_logits.device, dtype=med_logits.dtype)
    old_score = (
        med_logits
        + float(continue_weight) * continue_logits
        - float(remove_weight) * remove_logits
    )
    new_score = med_logits + float(add_weight) * add_logits
    return torch.where(previous, old_score, new_score)


def action_score_fusion_logits(
    med_logits,
    continue_logits,
    add_logits,
    remove_logits,
    previous_meds,
    continue_weight=0.0,
    add_weight=0.0,
    remove_weight=0.0,
):
    return score_fusion_logits(
        med_logits,
        continue_logits,
        add_logits,
        remove_logits,
        previous_meds,
        continue_weight=continue_weight,
        add_weight=add_weight,
        remove_weight=remove_weight,
    )


def direct_action_decode(
    continue_logits,
    add_logits,
    remove_logits,
    previous_meds,
    threshold=0.5,
):
    _require_action_logits(continue_logits, add_logits, remove_logits)
    previous = previous_meds.to(device=add_logits.device, dtype=torch.bool)
    continue_pred = torch.sigmoid(continue_logits) >= float(threshold)
    add_pred = torch.sigmoid(add_logits) >= float(threshold)
    remove_pred = torch.sigmoid(remove_logits) >= float(threshold)
    return (previous & continue_pred & (~remove_pred)) | ((~previous) & add_pred)


def direct_action_scores(
    continue_logits,
    add_logits,
    remove_logits,
    previous_meds,
):
    _require_action_logits(continue_logits, add_logits, remove_logits)
    previous = previous_meds.to(device=add_logits.device, dtype=torch.bool)
    continue_prob = torch.sigmoid(continue_logits)
    remove_prob = torch.sigmoid(remove_logits)
    add_prob = torch.sigmoid(add_logits)
    old_score = continue_prob * (1.0 - remove_prob)
    return torch.where(previous, old_score, add_prob)
