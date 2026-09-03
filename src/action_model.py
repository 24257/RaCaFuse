from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TrainableActionFusionLayer(nn.Module):
    def __init__(
        self,
        init_from_p4=False,
        freeze_weights=False,
    ):
        super().__init__()
        if init_from_p4:
            init = {
                "wc": 0.1,
                "wa": 0.2,
                "wr": 0.0,
                "sc": 1.0,
                "sa": 2.0,
                "sr": 0.5,
                "bc": 0.5,
                "ba": 0.0,
                "br": 0.5,
            }
        else:
            init = {
                "wc": 0.1,
                "wa": 0.2,
                "wr": 0.0,
                "sc": 1.0,
                "sa": 1.0,
                "sr": 1.0,
                "bc": 0.0,
                "ba": 0.0,
                "br": 0.0,
            }
        for name, value in init.items():
            parameter = nn.Parameter(torch.tensor(float(value)))
            if freeze_weights and name in {"wc", "wa", "wr"}:
                parameter.requires_grad = False
            setattr(self, name, parameter)

    def logging_dict(self):
        return {
            name: getattr(self, name)
            for name in ("wc", "wa", "wr", "sc", "sa", "sr", "bc", "ba", "br")
        }

    def forward(
        self,
        med_logits,
        continue_logits,
        add_logits,
        remove_logits,
        previous_meds,
    ):
        if continue_logits is None or add_logits is None or remove_logits is None:
            raise ValueError("TrainableActionFusionLayer requires continue/add/remove logits")
        if previous_meds is None:
            raise ValueError("TrainableActionFusionLayer requires previous_meds")
        previous = previous_meds.to(device=med_logits.device, dtype=torch.bool)
        continue_logits = continue_logits.to(device=med_logits.device, dtype=med_logits.dtype)
        add_logits = add_logits.to(device=med_logits.device, dtype=med_logits.dtype)
        remove_logits = remove_logits.to(device=med_logits.device, dtype=med_logits.dtype)
        params = {
            name: getattr(self, name).to(device=med_logits.device, dtype=med_logits.dtype)
            for name in ("wc", "wa", "wr", "sc", "sa", "sr", "bc", "ba", "br")
        }
        old_score = (
            med_logits
            + params["wc"] * (params["sc"] * continue_logits + params["bc"])
            - params["wr"] * (params["sr"] * remove_logits + params["br"])
        )
        new_score = med_logits + params["wa"] * (params["sa"] * add_logits + params["ba"])
        return torch.where(previous, old_score, new_score), self.logging_dict()


class RoutedActionFusionLayer(nn.Module):
    """Decoder-only Action-Conditioned Label Routing.

    This layer treats each drug-action pair as a masked multi-task label decision.
    It preserves the existing AMTN backbone and replaces only the final trainable
    action fusion decoder with shared and action-specific experts selected by a
    drug-conditioned router.
    """

    def __init__(
        self,
        m_voc_size,
        state_dim,
        hidden_dim=128,
        drug_emb_dim=32,
        dropout=0.1,
        entropy_weight=0.0,
    ):
        super().__init__()
        if m_voc_size <= 0:
            raise ValueError("m_voc_size must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if drug_emb_dim <= 0:
            raise ValueError("drug_emb_dim must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        if entropy_weight < 0:
            raise ValueError("entropy_weight must be non-negative")
        self.m_voc_size = int(m_voc_size)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.drug_emb_dim = int(drug_emb_dim)
        self.entropy_weight = float(entropy_weight)
        self.num_experts = 4
        self.num_actions = 3
        self.expert_names = ("shared", "continue", "add", "remove")
        self.action_names = ("continue", "add", "remove")

        self.drug_embedding = nn.Embedding(self.m_voc_size, self.drug_emb_dim)
        self.state_norm = nn.LayerNorm(self.state_dim)
        self.drug_norm = nn.LayerNorm(self.drug_emb_dim)
        self.signal_norm = nn.LayerNorm(5)
        expert_input_dim = self.state_dim + self.drug_emb_dim + 5
        router_input_dim = expert_input_dim + self.num_actions

        def make_expert():
            return nn.Sequential(
                nn.Linear(expert_input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, 1),
            )

        self.shared_expert = make_expert()
        self.continue_expert = make_expert()
        self.add_expert = make_expert()
        self.remove_expert = make_expert()
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_experts),
        )

        self.base_scale = nn.Parameter(torch.tensor(1.0))
        self.contribution_scale = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("drug_ids", torch.arange(self.m_voc_size, dtype=torch.long), persistent=False)

    def logging_dict(self):
        return {
            "base_scale": self.base_scale,
            "contribution_scale": self.contribution_scale,
        }

    def _drug_features(self, batch_size, device, dtype):
        drug_ids = self.drug_ids.to(device=device)
        drug_emb = self.drug_norm(self.drug_embedding(drug_ids).to(dtype=dtype))
        return drug_emb.unsqueeze(0).expand(batch_size, -1, -1)

    def _base_features(
        self,
        med_logits,
        continue_logits,
        add_logits,
        remove_logits,
        previous_meds,
        patient_state,
    ):
        batch_size, med_count = med_logits.shape
        if med_count != self.m_voc_size:
            raise ValueError(
                f"RoutedActionFusionLayer expected {self.m_voc_size} medication logits, got {med_count}"
            )
        if patient_state is None:
            patient_state = med_logits.new_zeros(batch_size, self.state_dim)
        if patient_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"RoutedActionFusionLayer expected patient_state dim {self.state_dim}, got {patient_state.shape[-1]}"
            )
        previous = previous_meds.to(device=med_logits.device, dtype=med_logits.dtype).clamp(0.0, 1.0)
        state = self.state_norm(patient_state.to(device=med_logits.device, dtype=med_logits.dtype))
        state = state.unsqueeze(1).expand(-1, med_count, -1)
        drug = self._drug_features(batch_size, med_logits.device, med_logits.dtype)
        signals = torch.stack(
            (
                med_logits,
                continue_logits,
                add_logits,
                remove_logits,
                previous,
            ),
            dim=-1,
        )
        signals = self.signal_norm(signals)
        return torch.cat((state, drug, signals), dim=-1), previous

    def _expert_values(self, expert_input):
        return torch.cat(
            (
                self.shared_expert(expert_input),
                self.continue_expert(expert_input),
                self.add_expert(expert_input),
                self.remove_expert(expert_input),
            ),
            dim=-1,
        )

    def _route(self, expert_input, action_index):
        batch_size, med_count, _ = expert_input.shape
        action = expert_input.new_zeros(batch_size, med_count, self.num_actions)
        action[..., action_index] = 1.0
        return F.softmax(self.router(torch.cat((expert_input, action), dim=-1)), dim=-1)

    def _logging(self, routing_weights, entropy):
        routing_mean = routing_weights.mean(dim=(0, 1, 2))
        action_routing_mean = routing_weights.mean(dim=(0, 1))
        max_route = routing_weights.max(dim=-1).values.mean()
        return {
            "routing_mean": routing_mean.detach(),
            "expert_routing_mean": routing_mean.detach(),
            "routing_entropy": entropy.detach(),
            "continue_routing_mean": action_routing_mean[0].detach(),
            "add_routing_mean": action_routing_mean[1].detach(),
            "remove_routing_mean": action_routing_mean[2].detach(),
            "router_max_weight_mean": max_route.detach(),
            "router_collapse": (max_route.detach() > 0.95),
        }

    def forward(
        self,
        med_logits,
        continue_logits,
        add_logits,
        remove_logits,
        previous_meds,
        patient_state=None,
        drug_embeddings=None,
    ):
        del drug_embeddings
        if continue_logits is None or add_logits is None or remove_logits is None:
            raise ValueError("RoutedActionFusionLayer requires continue/add/remove logits")
        if previous_meds is None:
            raise ValueError("RoutedActionFusionLayer requires previous_meds")
        med_logits = med_logits.to(dtype=torch.float32) if not med_logits.is_floating_point() else med_logits
        continue_logits = continue_logits.to(device=med_logits.device, dtype=med_logits.dtype)
        add_logits = add_logits.to(device=med_logits.device, dtype=med_logits.dtype)
        remove_logits = remove_logits.to(device=med_logits.device, dtype=med_logits.dtype)

        expert_input, previous = self._base_features(
            med_logits,
            continue_logits,
            add_logits,
            remove_logits,
            previous_meds,
            patient_state,
        )
        expert_values = self._expert_values(expert_input)
        continue_route = self._route(expert_input, 0)
        add_route = self._route(expert_input, 1)
        remove_route = self._route(expert_input, 2)
        routed_continue = (continue_route * expert_values).sum(dim=-1)
        routed_add = (add_route * expert_values).sum(dim=-1)
        routed_remove = (remove_route * expert_values).sum(dim=-1)

        scale = self.contribution_scale.to(device=med_logits.device, dtype=med_logits.dtype)
        base = self.base_scale.to(device=med_logits.device, dtype=med_logits.dtype) * med_logits
        old_score = base + scale * (routed_continue - routed_remove)
        new_score = base + scale * routed_add
        fused_logits = torch.where(previous.to(dtype=torch.bool), old_score, new_score)

        routing_weights = torch.stack((continue_route, add_route, remove_route), dim=2)
        entropy = -(routing_weights * routing_weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        logging = self._logging(routing_weights, entropy)
        if self.entropy_weight > 0:
            logging["routing_entropy_loss"] = -float(self.entropy_weight) * entropy
        return fused_logits, logging


ActionConditionedLabelRouter = RoutedActionFusionLayer


class RoutedResidualActionFusionLayer(nn.Module):
    """P4-initialized residual Action-Conditioned Label Routing.

    The p4 scalar fusion branch provides a conservative base score. The routed
    branch learns small drug-action conditioned residuals on top of that base,
    so ACLR can specialize without replacing the existing clean AMTN decoder.
    """

    def __init__(
        self,
        m_voc_size,
        state_dim,
        hidden_dim=128,
        drug_emb_dim=32,
        dropout=0.1,
        residual_scale=0.1,
        max_gate=1.0,
        gate_init_bias=-3.0,
        freeze_base=True,
        l2_weight=0.0,
        add_penalty_weight=0.0,
        entropy_weight=0.0,
        load_balance_weight=0.0,
    ):
        super().__init__()
        if m_voc_size <= 0:
            raise ValueError("m_voc_size must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if drug_emb_dim <= 0:
            raise ValueError("drug_emb_dim must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        if not 0.0 <= max_gate <= 1.0:
            raise ValueError("max_gate must be in [0, 1]")
        for name, value in (
            ("residual_scale", residual_scale),
            ("l2_weight", l2_weight),
            ("add_penalty_weight", add_penalty_weight),
            ("entropy_weight", entropy_weight),
            ("load_balance_weight", load_balance_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        self.m_voc_size = int(m_voc_size)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.drug_emb_dim = int(drug_emb_dim)
        self.freeze_base = bool(freeze_base)
        self.max_gate = float(max_gate)
        self.l2_weight = float(l2_weight)
        self.add_penalty_weight = float(add_penalty_weight)
        self.entropy_weight = float(entropy_weight)
        self.load_balance_weight = float(load_balance_weight)
        self.num_experts = 4
        self.num_actions = 3
        self.expert_names = ("shared", "continue", "add", "remove")
        self.action_names = ("continue", "add", "remove")

        self.base_fusion = TrainableActionFusionLayer(init_from_p4=True, freeze_weights=False)
        if self.freeze_base:
            for parameter in self.base_fusion.parameters():
                parameter.requires_grad = False

        self.drug_embedding = nn.Embedding(self.m_voc_size, self.drug_emb_dim)
        self.state_norm = nn.LayerNorm(self.state_dim)
        self.drug_norm = nn.LayerNorm(self.drug_emb_dim)
        self.signal_norm = nn.LayerNorm(5)
        expert_input_dim = self.state_dim + self.drug_emb_dim + 5
        router_input_dim = expert_input_dim + self.num_actions

        def make_expert():
            return nn.Sequential(
                nn.Linear(expert_input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, 1),
            )

        self.shared_expert = make_expert()
        self.continue_expert = make_expert()
        self.add_expert = make_expert()
        self.remove_expert = make_expert()
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_experts),
        )
        self.residual_gate = nn.Linear(router_input_dim, 1)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, float(gate_init_bias))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.register_buffer("drug_ids", torch.arange(self.m_voc_size, dtype=torch.long), persistent=False)

    def logging_dict(self):
        result = {
            "residual_scale": self.residual_scale,
            "residual_max_gate": torch.tensor(float(self.max_gate), device=self.residual_scale.device),
            "residual_gate_bias": self.residual_gate.bias,
            "freeze_base": torch.tensor(float(self.freeze_base), device=self.residual_scale.device),
        }
        for name, value in self.base_fusion.logging_dict().items():
            result["base_p4_" + name] = value
        return result

    def _drug_features(self, batch_size, device, dtype):
        drug_ids = self.drug_ids.to(device=device)
        drug_emb = self.drug_norm(self.drug_embedding(drug_ids).to(dtype=dtype))
        return drug_emb.unsqueeze(0).expand(batch_size, -1, -1)

    def _base_features(
        self,
        med_logits,
        continue_logits,
        add_logits,
        remove_logits,
        previous_meds,
        patient_state,
    ):
        batch_size, med_count = med_logits.shape
        if med_count != self.m_voc_size:
            raise ValueError(
                f"RoutedResidualActionFusionLayer expected {self.m_voc_size} medication logits, got {med_count}"
            )
        if patient_state is None:
            patient_state = med_logits.new_zeros(batch_size, self.state_dim)
        if patient_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"RoutedResidualActionFusionLayer expected patient_state dim {self.state_dim}, got {patient_state.shape[-1]}"
            )
        previous = previous_meds.to(device=med_logits.device, dtype=med_logits.dtype).clamp(0.0, 1.0)
        state = self.state_norm(patient_state.to(device=med_logits.device, dtype=med_logits.dtype))
        state = state.unsqueeze(1).expand(-1, med_count, -1)
        drug = self._drug_features(batch_size, med_logits.device, med_logits.dtype)
        signals = torch.stack(
            (
                med_logits,
                continue_logits,
                add_logits,
                remove_logits,
                previous,
            ),
            dim=-1,
        )
        signals = self.signal_norm(signals)
        return torch.cat((state, drug, signals), dim=-1), previous

    def _expert_values(self, expert_input):
        return torch.cat(
            (
                self.shared_expert(expert_input),
                self.continue_expert(expert_input),
                self.add_expert(expert_input),
                self.remove_expert(expert_input),
            ),
            dim=-1,
        )

    def _action_input(self, expert_input, action_index):
        batch_size, med_count, _ = expert_input.shape
        action = expert_input.new_zeros(batch_size, med_count, self.num_actions)
        action[..., action_index] = 1.0
        return torch.cat((expert_input, action), dim=-1)

    def _route_and_gate(self, expert_input, action_index):
        action_input = self._action_input(expert_input, action_index)
        route = F.softmax(self.router(action_input), dim=-1)
        gate = float(self.max_gate) * torch.sigmoid(self.residual_gate(action_input)).squeeze(-1)
        return route, gate

    @staticmethod
    def _masked_mean(value, mask):
        mask = mask.to(device=value.device, dtype=value.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (value * mask).sum() / denom

    def _logging(
        self,
        p4_score,
        final_score,
        residual_delta,
        residual_continue,
        residual_add,
        residual_remove,
        continue_gate,
        add_gate,
        remove_gate,
        previous,
        routing_weights,
        entropy,
    ):
        routing_mean = routing_weights.mean(dim=(0, 1, 2))
        action_routing_mean = routing_weights.mean(dim=(0, 1))
        max_route = routing_weights.max(dim=-1).values
        old_mask = previous
        new_mask = 1.0 - previous
        old_gate = torch.stack((continue_gate, remove_gate), dim=-1).mean(dim=-1)
        logging = {
            "residual_gate_mean": torch.stack((continue_gate, add_gate, remove_gate), dim=-1).mean().detach(),
            "residual_gate_old_mean": self._masked_mean(old_gate, old_mask).detach(),
            "residual_gate_new_mean": self._masked_mean(add_gate, new_mask).detach(),
            "residual_max_gate": p4_score.new_tensor(float(self.max_gate)).detach(),
            "residual_delta_mean": residual_delta.detach().mean(),
            "residual_delta_std": residual_delta.detach().std(unbiased=False),
            "residual_continue_mean": residual_continue.detach().mean(),
            "residual_add_mean": residual_add.detach().mean(),
            "residual_remove_mean": residual_remove.detach().mean(),
            "routing_entropy": entropy.detach(),
            "expert_load": routing_mean.detach(),
            "expert_routing_mean": routing_mean.detach(),
            "collapse_rate": (max_route.detach() > 0.95).to(dtype=p4_score.dtype).mean(),
            "router_collapse": (max_route.detach().mean() > 0.95),
            "router_max_weight_mean": max_route.detach().mean(),
            "route_continue_mean": action_routing_mean[0].detach(),
            "route_add_mean": action_routing_mean[1].detach(),
            "route_remove_mean": action_routing_mean[2].detach(),
            "continue_routing_mean": action_routing_mean[0].detach(),
            "add_routing_mean": action_routing_mean[1].detach(),
            "remove_routing_mean": action_routing_mean[2].detach(),
            "route_cos_continue_add": F.cosine_similarity(action_routing_mean[0], action_routing_mean[1], dim=0).detach(),
            "route_cos_continue_remove": F.cosine_similarity(action_routing_mean[0], action_routing_mean[2], dim=0).detach(),
            "route_cos_add_remove": F.cosine_similarity(action_routing_mean[1], action_routing_mean[2], dim=0).detach(),
            "base_p4_score_mean": p4_score.detach().mean(),
            "base_p4_score_std": p4_score.detach().std(unbiased=False),
            "final_score_mean": final_score.detach().mean(),
            "final_score_std": final_score.detach().std(unbiased=False),
        }
        if self.entropy_weight > 0:
            logging["routing_entropy_loss"] = -float(self.entropy_weight) * entropy
        if self.load_balance_weight > 0:
            target = torch.full_like(routing_mean, 1.0 / float(self.num_experts))
            logging["routed_residual_load_balance_loss"] = float(self.load_balance_weight) * (routing_mean - target).pow(2).mean()
        if self.l2_weight > 0:
            logging["routed_residual_l2_loss"] = float(self.l2_weight) * residual_delta.pow(2).mean()
        if self.add_penalty_weight > 0:
            logging["routed_residual_add_penalty_loss"] = (
                float(self.add_penalty_weight) * self._masked_mean(F.relu(residual_add).pow(2), new_mask)
            )
        return logging

    def forward(
        self,
        med_logits,
        continue_logits,
        add_logits,
        remove_logits,
        previous_meds,
        patient_state=None,
        drug_embeddings=None,
    ):
        del drug_embeddings
        if continue_logits is None or add_logits is None or remove_logits is None:
            raise ValueError("RoutedResidualActionFusionLayer requires continue/add/remove logits")
        if previous_meds is None:
            raise ValueError("RoutedResidualActionFusionLayer requires previous_meds")
        med_logits = med_logits.to(dtype=torch.float32) if not med_logits.is_floating_point() else med_logits
        continue_logits = continue_logits.to(device=med_logits.device, dtype=med_logits.dtype)
        add_logits = add_logits.to(device=med_logits.device, dtype=med_logits.dtype)
        remove_logits = remove_logits.to(device=med_logits.device, dtype=med_logits.dtype)
        p4_score, _ = self.base_fusion(
            med_logits,
            continue_logits,
            add_logits,
            remove_logits,
            previous_meds,
        )

        expert_input, previous = self._base_features(
            med_logits,
            continue_logits,
            add_logits,
            remove_logits,
            previous_meds,
            patient_state,
        )
        expert_values = self._expert_values(expert_input)
        continue_route, continue_gate = self._route_and_gate(expert_input, 0)
        add_route, add_gate = self._route_and_gate(expert_input, 1)
        remove_route, remove_gate = self._route_and_gate(expert_input, 2)
        routed_continue = (continue_route * expert_values).sum(dim=-1)
        routed_add = (add_route * expert_values).sum(dim=-1)
        routed_remove = (remove_route * expert_values).sum(dim=-1)

        scale = self.residual_scale.to(device=med_logits.device, dtype=med_logits.dtype)
        residual_continue = scale * continue_gate * routed_continue
        residual_add = scale * add_gate * routed_add
        residual_remove = scale * remove_gate * routed_remove
        residual_old = residual_continue - residual_remove
        residual_new = residual_add
        residual_delta = torch.where(previous.to(dtype=torch.bool), residual_old, residual_new)
        final_score = p4_score + residual_delta

        routing_weights = torch.stack((continue_route, add_route, remove_route), dim=2)
        entropy = -(routing_weights * routing_weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        logging = self._logging(
            p4_score,
            final_score,
            residual_delta,
            residual_continue,
            residual_add,
            residual_remove,
            continue_gate,
            add_gate,
            remove_gate,
            previous,
            routing_weights,
            entropy,
        )
        return final_score, logging


class SimpleFFNActionFusionLayer(nn.Module):
    """Parameter-matched decoder reference without action-conditioned routing.

    This layer consumes the same drug-conditioned features as ACLR, but a single
    FFN predicts continue/add/remove contributions directly. It is intended as
    the capacity control for `RoutedActionFusionLayer`.
    """

    def __init__(
        self,
        m_voc_size,
        state_dim,
        hidden_dim=128,
        drug_emb_dim=32,
        dropout=0.1,
        entropy_weight=0.0,
    ):
        super().__init__()
        if m_voc_size <= 0:
            raise ValueError("m_voc_size must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if drug_emb_dim <= 0:
            raise ValueError("drug_emb_dim must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        if entropy_weight < 0:
            raise ValueError("entropy_weight must be non-negative")
        self.m_voc_size = int(m_voc_size)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.drug_emb_dim = int(drug_emb_dim)
        self.entropy_weight = float(entropy_weight)

        self.drug_embedding = nn.Embedding(self.m_voc_size, self.drug_emb_dim)
        self.state_norm = nn.LayerNorm(self.state_dim)
        self.drug_norm = nn.LayerNorm(self.drug_emb_dim)
        self.signal_norm = nn.LayerNorm(5)
        self.input_dim = self.state_dim + self.drug_emb_dim + 5
        routed_like_params = 5 * self.input_dim * self.hidden_dim + 16 * self.hidden_dim + 8
        self.ffn_hidden_dim = max(
            self.hidden_dim,
            int(round(routed_like_params / float(self.input_dim + 4))),
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.input_dim, self.ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ffn_hidden_dim, 3),
        )

        self.base_scale = nn.Parameter(torch.tensor(1.0))
        self.contribution_scale = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("drug_ids", torch.arange(self.m_voc_size, dtype=torch.long), persistent=False)

    def logging_dict(self):
        return {
            "base_scale": self.base_scale,
            "contribution_scale": self.contribution_scale,
            "ffn_hidden_dim": torch.tensor(float(self.ffn_hidden_dim), device=self.base_scale.device),
        }

    def _drug_features(self, batch_size, device, dtype):
        drug_ids = self.drug_ids.to(device=device)
        drug_emb = self.drug_norm(self.drug_embedding(drug_ids).to(dtype=dtype))
        return drug_emb.unsqueeze(0).expand(batch_size, -1, -1)

    def _base_features(
        self,
        med_logits,
        continue_logits,
        add_logits,
        remove_logits,
        previous_meds,
        patient_state,
    ):
        batch_size, med_count = med_logits.shape
        if med_count != self.m_voc_size:
            raise ValueError(
                f"SimpleFFNActionFusionLayer expected {self.m_voc_size} medication logits, got {med_count}"
            )
        if patient_state is None:
            patient_state = med_logits.new_zeros(batch_size, self.state_dim)
        if patient_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"SimpleFFNActionFusionLayer expected patient_state dim {self.state_dim}, got {patient_state.shape[-1]}"
            )
        previous = previous_meds.to(device=med_logits.device, dtype=med_logits.dtype).clamp(0.0, 1.0)
        state = self.state_norm(patient_state.to(device=med_logits.device, dtype=med_logits.dtype))
        state = state.unsqueeze(1).expand(-1, med_count, -1)
        drug = self._drug_features(batch_size, med_logits.device, med_logits.dtype)
        signals = torch.stack(
            (
                med_logits,
                continue_logits,
                add_logits,
                remove_logits,
                previous,
            ),
            dim=-1,
        )
        signals = self.signal_norm(signals)
        return torch.cat((state, drug, signals), dim=-1), previous

    def forward(
        self,
        med_logits,
        continue_logits,
        add_logits,
        remove_logits,
        previous_meds,
        patient_state=None,
        drug_embeddings=None,
    ):
        del drug_embeddings
        if continue_logits is None or add_logits is None or remove_logits is None:
            raise ValueError("SimpleFFNActionFusionLayer requires continue/add/remove logits")
        if previous_meds is None:
            raise ValueError("SimpleFFNActionFusionLayer requires previous_meds")
        med_logits = med_logits.to(dtype=torch.float32) if not med_logits.is_floating_point() else med_logits
        continue_logits = continue_logits.to(device=med_logits.device, dtype=med_logits.dtype)
        add_logits = add_logits.to(device=med_logits.device, dtype=med_logits.dtype)
        remove_logits = remove_logits.to(device=med_logits.device, dtype=med_logits.dtype)

        features, previous = self._base_features(
            med_logits,
            continue_logits,
            add_logits,
            remove_logits,
            previous_meds,
            patient_state,
        )
        contributions = self.ffn(features)
        ffn_continue = contributions[..., 0]
        ffn_add = contributions[..., 1]
        ffn_remove = contributions[..., 2]

        scale = self.contribution_scale.to(device=med_logits.device, dtype=med_logits.dtype)
        base = self.base_scale.to(device=med_logits.device, dtype=med_logits.dtype) * med_logits
        old_score = base + scale * (ffn_continue - ffn_remove)
        new_score = base + scale * ffn_add
        fused_logits = torch.where(previous.to(dtype=torch.bool), old_score, new_score)
        logging = {
            "ffn_contribution_mean": contributions.detach().mean(dim=(0, 1)),
            "ffn_contribution_abs_mean": contributions.detach().abs().mean(),
            "ffn_hidden_dim": torch.tensor(float(self.ffn_hidden_dim), device=med_logits.device),
        }
        return fused_logits, logging


class ActionAwareMedicationTransitionBackbone(nn.Module):
    def __init__(
        self,
        state_dim,
        m_voc_size,
        hidden_dim=0,
        dropout=0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or state_dim
        self.m_voc_size = m_voc_size
        self.state_norm = nn.LayerNorm(state_dim)
        self.prev_norm = nn.LayerNorm(m_voc_size)
        self.feature = nn.Sequential(
            nn.Linear(state_dim + m_voc_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.continue_head = nn.Linear(hidden_dim, m_voc_size)
        self.add_head = nn.Linear(hidden_dim, m_voc_size)
        self.remove_head = nn.Linear(hidden_dim, m_voc_size)

        # Kept for compatibility with released checkpoints. The clean core
        # trains only continue/add/remove action logits.
        self.total_count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.continue_count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.add_count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.remove_count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def previous_meds(self, meds, reference):
        if meds is None or meds.size(1) <= 1:
            return reference.new_zeros(reference.size(0), self.m_voc_size)
        return (meds[:, 1, :] > 0).to(device=reference.device, dtype=reference.dtype)

    def forward(self, patient_state, meds):
        previous = self.previous_meds(meds, patient_state)
        feature_input = torch.cat((self.state_norm(patient_state), self.prev_norm(previous)), dim=-1)
        feature = self.feature(feature_input)
        continue_logits = self.continue_head(feature)
        add_logits = self.add_head(feature)
        remove_logits = self.remove_head(feature)
        return {
            "continue_logits": continue_logits,
            "add_logits": add_logits,
            "remove_logits": remove_logits,
            "total_count_pred": F.softplus(self.total_count_head(feature)).squeeze(-1),
            "continue_count_pred": F.softplus(self.continue_count_head(feature)).squeeze(-1),
            "add_count_pred": F.softplus(self.add_count_head(feature)).squeeze(-1),
            "remove_count_pred": F.softplus(self.remove_count_head(feature)).squeeze(-1),
            "previous_meds": previous,
        }
