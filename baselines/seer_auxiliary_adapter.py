#!/usr/bin/env python3
"""SEER auxiliary branch used by the released SEER fusion."""

from __future__ import annotations

import sys as _seer_sys
from pathlib import Path as _SeerPath
_seer_sys.path.insert(0, str(_SeerPath(__file__).resolve().parents[1] / "src"))
import seer_security

import argparse
import json
import math
import os
import sys
import time
import types
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import dill
import numpy as np
import torch
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    multilabel_margin_loss,
)
from torch.optim import Adam

from baseline_common import (
    checkpoint_path_for,
    configure_dataset_args,
    ddi_filename,
    existing_dataset_file,
    records_filename,
    save_command_metadata,
    save_prediction_artifact,
    split_patients_with_ids,
    target_start_visit_for,
    voc_filename,
)
from util import ddi_rate_score, get_n_params, multi_label_metric


MODEL_NAME = "SEERAuxiliary"
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]


def resolve_seer_auxiliary_source() -> Path:
    candidates = []
    if os.environ.get("SEER_AUXILIARY_SRC"):
        candidates.append(Path(os.environ["SEER_AUXILIARY_SRC"]))
    candidates.append(REPO_ROOT / "src" / "seer_auxiliary_branch")
    for candidate in candidates:
        if (candidate / "modules" / "__init__.py").exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "The bundled SEER SEERAuxiliary-derived runtime was not found. "
        "SEER_AUXILIARY_SRC may be used only as an explicit debugging override. "
        f"Searched: {searched}"
    )


SEER_AUXILIARY_SRC = resolve_seer_auxiliary_source()


def ensure_matplotlib_shim() -> None:
    """SEERAuxiliary imports pyplot for optional visualization; training does not use it."""
    if "matplotlib" in sys.modules:
        return
    try:
        __import__("matplotlib")
        return
    except ModuleNotFoundError:
        pass
    matplotlib_module = types.ModuleType("matplotlib")
    pyplot_module = types.ModuleType("matplotlib.pyplot")

    def _noop(*args, **kwargs):
        return None

    pyplot_module.figure = _noop
    pyplot_module.scatter = _noop
    pyplot_module.title = _noop
    pyplot_module.xlabel = _noop
    pyplot_module.ylabel = _noop
    pyplot_module.grid = _noop
    pyplot_module.legend = _noop
    pyplot_module.savefig = _noop
    pyplot_module.close = _noop
    pyplot_module.Line2D = lambda *args, **kwargs: None
    pyplot_module.gca = lambda *args, **kwargs: types.SimpleNamespace(
        spines={key: types.SimpleNamespace(set_linewidth=_noop) for key in ("left", "bottom", "right", "top")},
        tick_params=_noop,
        get_xticklabels=lambda: [],
        get_yticklabels=lambda: [],
    )
    matplotlib_module.pyplot = pyplot_module
    sys.modules["matplotlib"] = matplotlib_module
    sys.modules["matplotlib.pyplot"] = pyplot_module


def ensure_geometric_import_shim() -> None:
    """Allow importing SEERAuxiliary when optional molecular-GNN packages are absent."""
    try:
        __import__("torch_geometric")
        __import__("ogb")
        __import__("torch_scatter")
        return
    except ModuleNotFoundError:
        pass

    tg_module = types.ModuleType("torch_geometric")
    tg_nn = types.ModuleType("torch_geometric.nn")
    tg_utils = types.ModuleType("torch_geometric.utils")
    tg_data = types.ModuleType("torch_geometric.data")
    tg_inits = types.ModuleType("torch_geometric.nn.inits")
    ogb_module = types.ModuleType("ogb")
    ogb_graph = types.ModuleType("ogb.graphproppred")
    ogb_encoder = types.ModuleType("ogb.graphproppred.mol_encoder")
    ogb_utils = types.ModuleType("ogb.utils")
    scatter_module = types.ModuleType("torch_scatter")

    class _MessagePassing(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def propagate(self, *args, **kwargs):
            raise RuntimeError("torch_geometric is not available; this SEERAuxiliary adapter does not use the GNN branch")

    class _MissingEncoder(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            raise RuntimeError("ogb is not available; this SEERAuxiliary adapter does not use the GNN branch")

    def _missing_pool(*args, **kwargs):
        raise RuntimeError("torch_geometric is not available; this SEERAuxiliary adapter does not use the GNN branch")

    tg_nn.MessagePassing = _MessagePassing
    tg_nn.global_add_pool = _missing_pool
    tg_nn.global_mean_pool = _missing_pool
    tg_nn.global_max_pool = _missing_pool
    tg_nn.GlobalAttention = _MissingEncoder
    tg_nn.Set2Set = _MissingEncoder
    tg_utils.degree = lambda row, num_nodes, dtype=None: torch.zeros(num_nodes, dtype=dtype or torch.float32, device=row.device)
    tg_inits.uniform = lambda *args, **kwargs: None
    ogb_encoder.AtomEncoder = _MissingEncoder
    ogb_encoder.BondEncoder = _MissingEncoder
    ogb_utils.smiles2graph = _missing_pool
    scatter_module.scatter_mean = _missing_pool

    class _Data:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def to(self, device):
            for key, value in list(self.__dict__.items()):
                if hasattr(value, "to"):
                    setattr(self, key, value.to(device))
            return self

    tg_data.Data = _Data

    sys.modules.setdefault("torch_geometric", tg_module)
    sys.modules.setdefault("torch_geometric.nn", tg_nn)
    sys.modules.setdefault("torch_geometric.utils", tg_utils)
    sys.modules.setdefault("torch_geometric.data", tg_data)
    sys.modules.setdefault("torch_geometric.nn.inits", tg_inits)
    sys.modules.setdefault("ogb", ogb_module)
    sys.modules.setdefault("ogb.utils", ogb_utils)
    sys.modules.setdefault("ogb.graphproppred", ogb_graph)
    sys.modules.setdefault("ogb.graphproppred.mol_encoder", ogb_encoder)
    sys.modules.setdefault("torch_scatter", scatter_module)


ensure_matplotlib_shim()
ensure_geometric_import_shim()
if str(SEER_AUXILIARY_SRC) not in sys.path:
    sys.path.insert(0, str(SEER_AUXILIARY_SRC))

from modules import SEERAuxiliaryModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SEERAuxiliary under the local unified baseline protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--Test", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--dataset", choices=["mimic-iii", "mimic-iv", "eicu"], default="mimic-iii")
    parser.add_argument("--datadir", default="data/mimic-iii")
    parser.add_argument("--seed", type=int, default=1203)
    parser.add_argument("--cuda", type=int, default=-1)
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--early_stop", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--coef", type=float, default=2.5)
    parser.add_argument("--target_ddi", type=float, default=0.06)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dp", type=float, default=0.7)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--resume_path", default="")
    parser.add_argument(
        "--init_checkpoint",
        default="",
        help="training-only model-weight initialization; optimizer and RNG start fresh",
    )
    parser.add_argument(
        "--save_every_epoch",
        action="store_true",
        default=False,
        help="retain an immutable checkpoint after every completed training epoch",
    )
    parser.add_argument("--split_path", default="")
    parser.add_argument("--paper_protocol", action="store_true", default=False)
    parser.add_argument(
        "--validation_only_no_test",
        action="store_true",
        help="require an empty omitted test split and reject test evaluation",
    )
    parser.add_argument("--target_start_visit", type=int, default=-1)
    parser.add_argument("--eval_split", choices=["train", "valid", "eval", "test"], default="test")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--save_predictions", action="store_true", default=False)
    parser.add_argument("--representation_output", default="")
    parser.add_argument("--representation_alignment_predictions", default="")
    parser.add_argument("--representation_alignment_components", default="")
    parser.add_argument("--records_file", default="", help="explicit records filename under --datadir")
    parser.add_argument("--voc_file", default="", help="explicit vocabulary filename under --datadir")
    parser.add_argument("--ddi_file", default="", help="explicit DDI adjacency filename under --datadir")
    parser.add_argument("--ddi_mask_path", default="", help="explicit SEERAuxiliary medication mask input")
    parser.add_argument("--smoke_patients", type=int, default=0)
    parser.add_argument("--cuda_graphs", action="store_true", default=False)
    parser.add_argument("--cuda_graph_signatures", type=int, default=50)
    return configure_dataset_args(parser.parse_args(), "mimic-iii")


def load_array(path: Path) -> np.ndarray:
    return np.asarray(seer_security.load_pickle(open(path, "rb")))


def ddi_path_for(args: argparse.Namespace) -> str:
    return os.path.join(args.datadir, ddi_filename(args))


def build_model(args: argparse.Namespace, voc_size: tuple[int, int, int], ehr_adj: np.ndarray, ddi_adj: np.ndarray, device: torch.device) -> SEERAuxiliaryModel:
    return SEERAuxiliaryModel(
        emb_dim=args.dim,
        ehr_adj=ehr_adj,
        ddi_adj=torch.from_numpy(np.asarray(ddi_adj)).float().to(device),
        voc_size=voc_size,
        use_embedding=False,
        device=device,
        dropout=args.dp,
    ).to(device)


def next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def graph_signature(visit_prefix: list) -> tuple[int, int, int]:
    return (
        len(visit_prefix),
        next_power_of_two(max(len(visit[0]) for visit in visit_prefix)),
        next_power_of_two(max(len(visit[1]) for visit in visit_prefix)),
    )


def select_graph_signatures(data_train: list, limit: int) -> set[tuple[int, int, int]]:
    counts: Counter[tuple[int, int, int]] = Counter()
    for patient in data_train:
        for visit_idx in range(len(patient)):
            counts[graph_signature(patient[: visit_idx + 1])] += 1
    return {signature for signature, _ in counts.most_common(max(0, int(limit)))}


@dataclass
class GraphInputs:
    diag_ids: torch.Tensor
    diag_valid: torch.Tensor
    proc_ids: torch.Tensor
    proc_valid: torch.Tensor
    med_history: torch.Tensor
    target: torch.Tensor
    margin_target: torch.Tensor

    @classmethod
    def create(
        cls,
        signature: tuple[int, int, int],
        med_count: int,
        device: torch.device,
    ) -> "GraphInputs":
        visit_count, diag_bucket, proc_bucket = signature
        return cls(
            diag_ids=torch.zeros((visit_count, diag_bucket), dtype=torch.long, device=device),
            diag_valid=torch.zeros((visit_count, diag_bucket), dtype=torch.bool, device=device),
            proc_ids=torch.zeros((visit_count, proc_bucket), dtype=torch.long, device=device),
            proc_valid=torch.zeros((visit_count, proc_bucket), dtype=torch.bool, device=device),
            med_history=torch.zeros((max(0, visit_count - 1), med_count), device=device),
            target=torch.zeros((1, med_count), device=device),
            margin_target=-torch.ones((1, med_count), dtype=torch.long, device=device),
        )

    def load(self, visit_prefix: list) -> None:
        self.diag_ids.zero_()
        self.diag_valid.zero_()
        self.proc_ids.zero_()
        self.proc_valid.zero_()
        self.med_history.zero_()
        self.target.zero_()
        self.margin_target.fill_(-1)
        for visit_idx, visit in enumerate(visit_prefix):
            diag_count = len(visit[0])
            proc_count = len(visit[1])
            self.diag_ids[visit_idx, :diag_count] = torch.as_tensor(
                visit[0], dtype=torch.long, device=self.diag_ids.device
            )
            self.diag_valid[visit_idx, :diag_count] = True
            self.proc_ids[visit_idx, :proc_count] = torch.as_tensor(
                visit[1], dtype=torch.long, device=self.proc_ids.device
            )
            self.proc_valid[visit_idx, :proc_count] = True
            if visit_idx + 1 < len(visit_prefix):
                self.med_history[visit_idx, visit[2]] = 1.0
        labels = visit_prefix[-1][2]
        self.target[0, labels] = 1.0
        self.margin_target[0, : len(labels)] = torch.as_tensor(
            labels, dtype=torch.long, device=self.margin_target.device
        )


@dataclass
class CapturedTrainStep:
    inputs: GraphInputs
    graph: torch.cuda.CUDAGraph
    loss: torch.Tensor


@dataclass
class CapturedPredictStep:
    inputs: GraphInputs
    graph: torch.cuda.CUDAGraph
    probability: torch.Tensor


def tensorized_model_forward(
    model: SEERAuxiliaryModel,
    med_tensor: torch.Tensor,
    ddi_adj: torch.Tensor,
    inputs: GraphInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    med_embedding = model.med_embedding(med_tensor)
    ehr_embedding, ddi_embedding = model.gcn()
    med_repr = med_embedding + ehr_embedding - model.inter * ddi_embedding

    current = inputs.diag_ids.shape[0] - 1
    diag_repr = model.diag_encoder(
        model.diag_embedding(inputs.diag_ids[current : current + 1]),
        inputs.diag_valid[current : current + 1],
    )
    proc_repr = model.proc_encoder(
        model.proc_embedding(inputs.proc_ids[current : current + 1]),
        inputs.proc_valid[current : current + 1],
    )

    if inputs.diag_ids.shape[0] > 1:
        diag_history = []
        proc_history = []
        for visit_idx in range(inputs.diag_ids.shape[0]):
            diag_valid = inputs.diag_valid[visit_idx : visit_idx + 1]
            proc_valid = inputs.proc_valid[visit_idx : visit_idx + 1]
            diag_encoded = model.diag_encoder(
                model.diag_embedding(inputs.diag_ids[visit_idx : visit_idx + 1]),
                diag_valid,
            )
            proc_encoded = model.proc_encoder(
                model.proc_embedding(inputs.proc_ids[visit_idx : visit_idx + 1]),
                proc_valid,
            )
            diag_history.append(model.pma_d(diag_encoded, diag_valid).squeeze(0))
            proc_history.append(model.pma_p(proc_encoded, proc_valid).squeeze(0))

        diag_sequence = torch.stack(diag_history, dim=0)
        proc_sequence = torch.stack(proc_history, dim=0)
        diag_output, _ = model.seq_encoders[0](diag_sequence)
        proc_output, _ = model.seq_encoders[1](proc_sequence)
        visit_output = model.Out_visit(torch.cat([diag_output, proc_output], dim=-1)).squeeze(1)
        history_score = torch.softmax(
            model.aggregator(visit_output[-1], visit_output), dim=-1
        ).squeeze(0)
        med_history = (inputs.med_history * history_score[:-1, None]).sum(dim=0)
        med_repr = med_repr * (1.0 + med_history * model.garm).unsqueeze(1)

    decoder = model.decoder
    hidden = med_repr.unsqueeze(0)
    hidden = decoder.norm1(hidden + decoder.self_block(hidden, attn_mask=None))
    diag_attention = decoder.m2d_multihead_attn(
        hidden,
        diag_repr,
        diag_repr,
        key_padding_mask=~inputs.diag_valid[current : current + 1],
        need_weights=False,
    )[0]
    proc_attention = decoder.m2p_multihead_attn(
        hidden,
        proc_repr,
        proc_repr,
        key_padding_mask=~inputs.proc_valid[current : current + 1],
        need_weights=False,
    )[0]
    hidden = decoder.norm2(
        hidden + decoder.dropout2(diag_attention) + decoder.dropout2(proc_attention)
    )
    hidden = decoder.norm3(hidden + decoder._ff_block(hidden)).squeeze(0)
    score = model.score_extractor(hidden).t()
    probability = torch.sigmoid(score)
    loss_ddi = 0.0005 * torch.matmul(probability.t(), probability).mul(ddi_adj).sum()
    return score, loss_ddi


def tensorized_training_loss(
    model: SEERAuxiliaryModel,
    med_tensor: torch.Tensor,
    ddi_adj: torch.Tensor,
    ddi_pairs: torch.Tensor,
    inputs: GraphInputs,
    coef: float,
    target_ddi: float,
) -> torch.Tensor:
    result, loss_ddi = tensorized_model_forward(model, med_tensor, ddi_adj, inputs)
    sigmoid_result = torch.sigmoid(result)
    loss_bce = binary_cross_entropy_with_logits(result, inputs.target)
    loss_multi = multilabel_margin_loss(sigmoid_result, inputs.margin_target)
    pred_mask = sigmoid_result.detach()[0] >= 0.5
    selected_pairs = torch.triu(
        torch.logical_and(pred_mask[:, None], pred_mask[None, :]), diagonal=1
    )
    current_ddi_rate = torch.logical_and(selected_pairs, ddi_pairs).sum().float() / selected_pairs.sum().clamp_min(1)
    accuracy_loss = 0.95 * loss_bce + 0.05 * loss_multi
    beta = torch.exp(coef * (1 - current_ddi_rate / target_ddi)).clamp(max=1.0)
    return torch.where(
        current_ddi_rate <= target_ddi,
        accuracy_loss,
        beta * accuracy_loss + (1 - beta) * loss_ddi,
    )


class SEERAuxiliaryCudaGraphRuntime:
    def __init__(
        self,
        model: SEERAuxiliaryModel,
        med_tensor: torch.Tensor,
        ddi_adj: torch.Tensor,
        med_count: int,
        coef: float,
        target_ddi: float,
        optimizer: Adam | None = None,
        allowed_signatures: set[tuple[int, int, int]] | None = None,
    ) -> None:
        self.model = model
        self.med_tensor = med_tensor
        self.ddi_adj = ddi_adj
        self.ddi_pairs = torch.logical_or(ddi_adj > 0, ddi_adj.t() > 0)
        self.med_count = med_count
        self.coef = coef
        self.target_ddi = target_ddi
        self.optimizer = optimizer
        self.allowed_signatures = allowed_signatures
        self.train_graphs: dict[tuple[int, int, int], CapturedTrainStep] = {}
        self.predict_graphs: dict[tuple[int, int, int], CapturedPredictStep] = {}
        self.capture_stream = torch.cuda.Stream(device=med_tensor.device)

    def _warm_training_step(self, inputs: GraphInputs) -> None:
        if self.optimizer is None:
            raise RuntimeError("CUDA graph training requires an optimizer")
        parameter_snapshot = [parameter.detach().clone() for parameter in self.model.parameters()]
        state_snapshot = {
            parameter: {
                key: value.detach().clone() if torch.is_tensor(value) else value
                for key, value in state.items()
            }
            for parameter, state in self.optimizer.state.items()
        }
        rng_state = torch.cuda.get_rng_state(self.med_tensor.device)
        current_stream = torch.cuda.current_stream(self.med_tensor.device)
        self.capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.capture_stream):
            self.optimizer.zero_grad(set_to_none=True)
            loss = tensorized_training_loss(
                self.model,
                self.med_tensor,
                self.ddi_adj,
                self.ddi_pairs,
                inputs,
                self.coef,
                self.target_ddi,
            )
            loss.backward()
            self.optimizer.step()
        current_stream.wait_stream(self.capture_stream)
        current_stream.synchronize()
        with torch.no_grad():
            for parameter, snapshot in zip(self.model.parameters(), parameter_snapshot):
                parameter.copy_(snapshot)
        if state_snapshot:
            for parameter, snapshot in state_snapshot.items():
                for key, value in snapshot.items():
                    if torch.is_tensor(value):
                        self.optimizer.state[parameter][key].copy_(value)
                    else:
                        self.optimizer.state[parameter][key] = value
        else:
            for state in self.optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        value.zero_()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.set_rng_state(rng_state, self.med_tensor.device)

    def train_step(self, visit_prefix: list) -> torch.Tensor | None:
        if self.optimizer is None:
            raise RuntimeError("CUDA graph training requires an optimizer")
        signature = graph_signature(visit_prefix)
        if self.allowed_signatures is not None and signature not in self.allowed_signatures:
            return None
        captured = self.train_graphs.get(signature)
        if captured is None:
            inputs = GraphInputs.create(signature, self.med_count, self.med_tensor.device)
            inputs.load(visit_prefix)
            self._warm_training_step(inputs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph,
                stream=self.capture_stream,
            ):
                self.optimizer.zero_grad(set_to_none=True)
                loss = tensorized_training_loss(
                    self.model,
                    self.med_tensor,
                    self.ddi_adj,
                    self.ddi_pairs,
                    inputs,
                    self.coef,
                    self.target_ddi,
                )
                loss.backward()
                self.optimizer.step()
            captured = CapturedTrainStep(inputs=inputs, graph=graph, loss=loss)
            self.train_graphs[signature] = captured
        else:
            captured.inputs.load(visit_prefix)
            captured.graph.replay()
        return captured.loss.detach()

    def predict(self, visit_prefix: list) -> torch.Tensor | None:
        signature = graph_signature(visit_prefix)
        if self.allowed_signatures is not None and signature not in self.allowed_signatures:
            return None
        captured = self.predict_graphs.get(signature)
        if captured is None:
            inputs = GraphInputs.create(signature, self.med_count, self.med_tensor.device)
            inputs.load(visit_prefix)
            current_stream = torch.cuda.current_stream(self.med_tensor.device)
            self.capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.capture_stream), torch.no_grad():
                tensorized_model_forward(self.model, self.med_tensor, self.ddi_adj, inputs)
            current_stream.wait_stream(self.capture_stream)
            current_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph,
                stream=self.capture_stream,
            ), torch.no_grad():
                score, _ = tensorized_model_forward(
                    self.model, self.med_tensor, self.ddi_adj, inputs
                )
                probability = torch.sigmoid(score)
            captured = CapturedPredictStep(
                inputs=inputs,
                graph=graph,
                probability=probability,
            )
            self.predict_graphs[signature] = captured
        else:
            captured.inputs.load(visit_prefix)
            captured.graph.replay()
        return captured.probability[0]


def prediction_for_visit(
    model: SEERAuxiliaryModel,
    med_tensor: torch.Tensor,
    visit_prefix: list,
    ddi_mask_h: torch.Tensor,
    ddi_adj: torch.Tensor,
    voc_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    output, _ = model(med_tensor, visit_prefix, ddi_mask_h, ddi_adj)
    prob = torch.sigmoid(output).detach().cpu().numpy()[0].astype(np.float32)
    pred = (prob >= 0.5).astype(np.float32)
    label = visit_prefix[-1][2]
    y_true = np.zeros(voc_size[2], dtype=np.float32)
    y_true[label] = 1.0
    pred_label = np.where(pred == 1)[0].astype(int).tolist()
    return y_true, pred, prob, pred_label


def representations_for_visit(
    model: SEERAuxiliaryModel,
    med_tensor: torch.Tensor,
    visit_prefix: list,
    ddi_mask_h: torch.Tensor,
    ddi_adj: torch.Tensor,
    voc_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    captured: dict[str, torch.Tensor] = {}

    def capture_hidden(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["medication_hidden"] = inputs[0].detach()

    handle = model.score_extractor.register_forward_pre_hook(capture_hidden)
    try:
        output, _ = model(med_tensor, visit_prefix, ddi_mask_h, ddi_adj)
    finally:
        handle.remove()
    if "medication_hidden" not in captured:
        raise RuntimeError("SEERAuxiliary score input hook did not capture medication hidden states")
    diag = torch.as_tensor([visit_prefix[-1][0]], dtype=torch.long, device=med_tensor.device)
    proc = torch.as_tensor([visit_prefix[-1][1]], dtype=torch.long, device=med_tensor.device)
    diag_repr = model.diag_encoder(model.diag_embedding(diag))
    proc_repr = model.proc_encoder(model.proc_embedding(proc))
    diag_pool = model.pma_d(diag_repr).squeeze(0)
    proc_pool = model.pma_p(proc_repr).squeeze(0)
    patient_vector = model.W_visit(torch.cat((diag_pool, proc_pool), dim=-1))
    probability = torch.sigmoid(output).detach().cpu().numpy()[0].astype(np.float32)
    y_true = np.zeros(voc_size[2], dtype=np.float32)
    y_true[visit_prefix[-1][2]] = 1.0
    return (
        y_true,
        probability,
        patient_vector.detach().cpu().numpy()[0].astype(np.float16),
        captured["medication_hidden"].cpu().numpy().astype(np.float16),
    )


def export_hidden_representations(
    model: SEERAuxiliaryModel,
    data_eval: list,
    med_tensor: torch.Tensor,
    voc_size: tuple[int, int, int],
    ddi_mask_h: torch.Tensor,
    ddi_adj: torch.Tensor,
    args: argparse.Namespace,
    split_name: str,
    target_start_visit: int,
) -> None:
    if split_name != "train":
        raise ValueError("hidden representation export is restricted to the train split")
    if not args.representation_alignment_predictions:
        raise ValueError("--representation_alignment_predictions is required")
    if not args.representation_alignment_components:
        raise ValueError("--representation_alignment_components is required")
    model.eval()
    targets = []
    probabilities = []
    patient_vectors = []
    medication_hidden = []
    local_patient_ids = []
    visit_ids = []
    with torch.no_grad():
        for patient_index, patient in enumerate(data_eval):
            start = min(target_start_visit, len(patient))
            for visit_index in range(start, len(patient)):
                target, probability, patient_vector, hidden = representations_for_visit(
                    model,
                    med_tensor,
                    patient[: visit_index + 1],
                    ddi_mask_h,
                    ddi_adj,
                    voc_size,
                )
                targets.append(target)
                probabilities.append(probability)
                patient_vectors.append(patient_vector)
                medication_hidden.append(hidden)
                local_patient_ids.append(patient_index)
                visit_ids.append(visit_index)
            if (patient_index + 1) % 100 == 0 or patient_index + 1 == len(data_eval):
                print(
                    "\rrepresentation step: {} / {}".format(patient_index + 1, len(data_eval)),
                    end="",
                    flush=True,
                )
    print()
    y_true = np.asarray(targets, dtype=np.float32)
    y_prob = np.asarray(probabilities, dtype=np.float32)
    patient_vector = np.asarray(patient_vectors, dtype=np.float16)
    med_hidden = np.asarray(medication_hidden, dtype=np.float16)
    local_ids = np.asarray(local_patient_ids, dtype=np.int64)
    visits = np.asarray(visit_ids, dtype=np.int64)
    with np.load(args.representation_alignment_predictions, allow_pickle=False) as alignment:
        alignment_true = np.asarray(alignment["y_true"], dtype=np.float32)
        alignment_probability = np.asarray(alignment["y_prob"], dtype=np.float32)
    if not np.array_equal(y_true, alignment_true):
        raise RuntimeError("SEERAuxiliary representation targets do not match train predictions")
    probability_max_abs_diff = float(np.max(np.abs(y_prob - alignment_probability)))
    prediction_diff_entries = int(
        np.sum((y_prob >= 0.5) != (alignment_probability >= 0.5))
    )
    if probability_max_abs_diff > 5e-6 or prediction_diff_entries > 0:
        raise RuntimeError("SEERAuxiliary representation probability parity failed")
    with np.load(args.representation_alignment_components, allow_pickle=False) as components:
        component_true = np.asarray(components["y_true"], dtype=np.float32)
        component_local_ids = np.asarray(components["local_patient_ids"], dtype=np.int64)
        component_patient_ids = np.asarray(components["patient_ids"], dtype=np.int64)
        component_visit_ids = np.asarray(components["visit_ids"], dtype=np.int64)
    if not np.array_equal(y_true, component_true):
        raise RuntimeError("SEERAuxiliary representation targets do not match anchor components")
    if not np.array_equal(local_ids, component_local_ids) or not np.array_equal(
        visits,
        component_visit_ids,
    ):
        raise RuntimeError("SEERAuxiliary representation row identities do not match anchor components")
    output_path = Path(args.representation_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        y_true=y_true,
        y_prob=y_prob,
        patient_vector=patient_vector,
        medication_hidden=med_hidden,
        patient_ids=component_patient_ids,
        local_patient_ids=local_ids,
        visit_ids=visits,
        metadata=np.array(
            {
                "schema_version": "seer_auxiliary-hidden-representation-v1",
                "split": split_name,
                "seed": int(args.seed),
                "checkpoint": args.resume_path,
                "patient_source": "PMA(diag/proc) -> W_visit",
                "medication_source": "score_extractor forward input",
                "storage_dtype": "float16",
                "probability_max_abs_diff": probability_max_abs_diff,
                "probability_tolerance": 5e-6,
                "prediction_diff_entries": prediction_diff_entries,
                "eval_loaded": False,
                "test_loaded": False,
            },
            dtype=object,
        ),
    )
    print("representation_output", output_path)


def ddi_pair_rate(meds: list[int], ddi_adj: np.ndarray) -> float:
    all_cnt = 0
    ddi_cnt = 0
    for i, med_i in enumerate(meds):
        for med_j in meds[i + 1 :]:
            all_cnt += 1
            if ddi_adj[med_i, med_j] == 1 or ddi_adj[med_j, med_i] == 1:
                ddi_cnt += 1
    return 0.0 if all_cnt == 0 else ddi_cnt / all_cnt


def evaluate(
    model: SEERAuxiliaryModel,
    data_eval: list,
    med_tensor: torch.Tensor,
    voc_size: tuple[int, int, int],
    ddi_mask_h: torch.Tensor,
    ddi_adj: torch.Tensor,
    ddi_adj_np: np.ndarray,
    args: argparse.Namespace,
    split_name: str,
    target_start_visit: int,
    patient_ids: list[int] | None = None,
    epoch: int = 0,
    graph_runtime: SEERAuxiliaryCudaGraphRuntime | None = None,
) -> tuple[float, float, float, float, float, float, float]:
    model.eval()
    y_true_all: list[np.ndarray] = []
    row_patient_ids: list[int] = []
    row_local_patient_ids: list[int] = []
    row_visit_ids: list[int] = []
    patient_target_counts: list[int] = []
    if patient_ids is None:
        patient_ids = list(range(len(data_eval)))
    if len(patient_ids) != len(data_eval):
        raise ValueError("patient_ids must align with SEERAuxiliary evaluation patients")
    probability_gpu = None
    probability_row = 0
    if graph_runtime is not None:
        target_count = sum(max(0, len(patient) - min(target_start_visit, len(patient))) for patient in data_eval)
        probability_gpu = torch.empty((target_count, voc_size[2]), device=med_tensor.device)
    else:
        y_prob_all: list[np.ndarray] = []
    with torch.no_grad():
        for step, patient in enumerate(data_eval):
            start = min(target_start_visit, len(patient))
            patient_target_counts.append(len(patient) - start)
            for visit_idx in range(start, len(patient)):
                visit_prefix = patient[: visit_idx + 1]
                y_true = np.zeros(voc_size[2], dtype=np.float32)
                y_true[visit_prefix[-1][2]] = 1.0
                y_true_all.append(y_true)
                row_patient_ids.append(int(patient_ids[step]))
                row_local_patient_ids.append(int(step))
                row_visit_ids.append(int(visit_idx))
                if graph_runtime is not None:
                    graph_probability = graph_runtime.predict(visit_prefix)
                    if graph_probability is None:
                        _, _, y_prob, _ = prediction_for_visit(
                            model,
                            med_tensor,
                            visit_prefix,
                            ddi_mask_h,
                            ddi_adj,
                            voc_size,
                        )
                        probability_gpu[probability_row].copy_(
                            torch.from_numpy(y_prob).to(probability_gpu.device)
                        )
                    else:
                        probability_gpu[probability_row].copy_(graph_probability)
                    probability_row += 1
                else:
                    _, _, y_prob, _ = prediction_for_visit(
                        model,
                        med_tensor,
                        visit_prefix,
                        ddi_mask_h,
                        ddi_adj,
                        voc_size,
                    )
                    y_prob_all.append(y_prob)
            if (step + 1) % 100 == 0 or step + 1 == len(data_eval):
                print(f"\reval step: {step + 1} / {len(data_eval)}", end="", flush=True)
    print()
    if not y_true_all:
        raise ValueError("SEERAuxiliary evaluation produced no target visits")
    y_true_arr = np.asarray(y_true_all, dtype=np.float32)
    if probability_gpu is not None:
        y_prob_arr = probability_gpu.cpu().numpy().astype(np.float32, copy=False)
    else:
        y_prob_arr = np.asarray(y_prob_all, dtype=np.float32)
    if not np.isfinite(y_prob_arr).all():
        invalid = int(y_prob_arr.size - np.isfinite(y_prob_arr).sum())
        raise FloatingPointError(f"SEERAuxiliary evaluation produced {invalid} non-finite probabilities")
    y_pred_arr = (y_prob_arr >= 0.5).astype(np.float32)
    smm_record: list[list[list[int]]] = []
    row = 0
    for patient_count in patient_target_counts:
        patient_predictions = []
        for _ in range(patient_count):
            patient_predictions.append(np.where(y_pred_arr[row] == 1)[0].astype(int).tolist())
            row += 1
        if patient_predictions:
            smm_record.append(patient_predictions)
    med_cnt = int(y_pred_arr.sum())
    visit_cnt = int(y_pred_arr.shape[0])
    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y_true_arr, y_pred_arr, y_prob_arr)
    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_np)
    save_prediction_artifact(
        args,
        y_true_arr,
        y_pred_arr,
        y_prob_arr,
        {"split": split_name, "epoch": int(epoch), "decode": "threshold_0.5"},
        extra_arrays={
            "patient_ids": np.asarray(row_patient_ids, dtype=np.int64),
            "local_patient_ids": np.asarray(row_local_patient_ids, dtype=np.int64),
            "visit_ids": np.asarray(row_visit_ids, dtype=np.int64),
        },
    )
    return ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, med_cnt / max(1, visit_cnt)


def train_epoch(
    model: SEERAuxiliaryModel,
    data_train: list,
    med_tensor: torch.Tensor,
    voc_size: tuple[int, int, int],
    ddi_mask_h: torch.Tensor,
    ddi_adj: torch.Tensor,
    ddi_adj_np: np.ndarray,
    optimizer: Adam,
    coef: float,
    target_ddi: float,
    device: torch.device,
    graph_runtime: SEERAuxiliaryCudaGraphRuntime | None = None,
) -> float:
    model.train()
    loss_sum = torch.zeros((), device=device)
    loss_count = 0
    ddi_pairs = torch.logical_or(ddi_adj > 0, ddi_adj.t() > 0)
    for step, patient in enumerate(data_train):
        for visit_idx in range(len(patient)):
            if graph_runtime is not None:
                graph_loss = graph_runtime.train_step(patient[: visit_idx + 1])
                if graph_loss is not None:
                    loss = graph_loss
                    loss_sum.add_(loss)
                    loss_count += 1
                    continue
            target = torch.zeros((1, voc_size[2]), device=device)
            target[:, patient[visit_idx][2]] = 1.0
            margin_target = -torch.ones((1, voc_size[2]), dtype=torch.long)
            for idx, med in enumerate(patient[visit_idx][2]):
                if idx < voc_size[2]:
                    margin_target[0, idx] = int(med)
            margin_target = margin_target.to(device)

            result, loss_ddi = model(med_tensor, patient[: visit_idx + 1], ddi_mask_h, ddi_adj)
            sigmoid_result = torch.sigmoid(result)
            loss_bce = binary_cross_entropy_with_logits(result, target)
            loss_multi = multilabel_margin_loss(sigmoid_result, margin_target)

            result_np = sigmoid_result.detach().cpu().numpy()[0]
            result_np[result_np >= 0.5] = 1
            result_np[result_np < 0.5] = 0
            y_label = np.where(result_np == 1)[0]
            current_ddi_rate = ddi_rate_score([[y_label]], path=ddi_adj_np)
            accuracy_loss = 0.95 * loss_bce + 0.05 * loss_multi
            if current_ddi_rate <= target_ddi:
                loss = accuracy_loss
            else:
                beta = coef * (1 - current_ddi_rate / target_ddi)
                beta = min(math.exp(beta), 1)
                loss = beta * accuracy_loss + (1 - beta) * loss_ddi

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum.add_(loss.detach())
            loss_count += 1
        if (step + 1) % 100 == 0 or step + 1 == len(data_train):
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    f"SEERAuxiliary training produced a non-finite loss at patient {step + 1}"
                )
            print(
                f"\rtraining step: {step + 1} / {len(data_train)} "
                f"loss:{loss.item():.4f}",
                end="",
                flush=True,
            )
    print()
    return float((loss_sum / max(1, loss_count)).item())


def main() -> int:
    args = parse_args()
    if args.eval_only:
        args.Test = True
    if args.cuda_graphs:
        raise ValueError(
            "--cuda_graphs is disabled: the bucketed tensor forward is not "
            "logit-equivalent to SEERAuxiliary's variable-length reference forward"
        )
    if args.validation_only_no_test and args.eval_split == "test":
        raise ValueError("--validation_only_no_test cannot be combined with --eval_split test")
    if args.Test and args.init_checkpoint:
        raise ValueError("--init_checkpoint is training-only; use --resume_path for evaluation")
    if not args.Test and args.resume_path:
        raise ValueError("--resume_path is evaluation-only; use --init_checkpoint for warm-start training")
    save_command_metadata(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda > -1 and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if args.cuda > -1 and torch.cuda.is_available() else "cpu")

    records_name = args.records_file or existing_dataset_file(
        args.datadir, records_filename(args), "records_final.pkl"
    )
    data = seer_security.load_pickle(open(os.path.join(args.datadir, records_name), "rb"))
    if args.smoke_patients > 0:
        data = data[: args.smoke_patients]
    voc_name = args.voc_file or existing_dataset_file(
        args.datadir, voc_filename(args), "voc_final.pkl"
    )
    voc = seer_security.load_pickle(open(os.path.join(args.datadir, voc_name), "rb"))
    ehr_name = existing_dataset_file(args.datadir, "ehr_adj_final_4.pkl", "ehr_adj_final.pkl", "ehr.pkl")
    ehr_adj = load_array(Path(args.datadir) / ehr_name)
    ddi_name = args.ddi_file or existing_dataset_file(
        args.datadir, ddi_filename(args), "ddi_A_final.pkl"
    )
    ddi_adj_np = load_array(Path(args.datadir) / ddi_name)
    ddi_mask_h = torch.eye(len(voc["med_voc"].idx2word), device=device)
    external_mask = Path(args.ddi_mask_path) if args.ddi_mask_path else None
    if external_mask is not None and external_mask.exists():
        loaded_mask = seer_security.load_pickle(open(external_mask, "rb"))
        if np.asarray(loaded_mask).shape[0] == len(voc["med_voc"].idx2word):
            ddi_mask_h = torch.from_numpy(np.asarray(loaded_mask)).float().to(device)
    ddi_adj = torch.from_numpy(np.asarray(ddi_adj_np)).float().to(device)

    voc_size = (len(voc["diag_voc"].idx2word), len(voc["pro_voc"].idx2word), len(voc["med_voc"].idx2word))
    (data_train, data_eval, data_test), (train_ids, eval_ids, test_ids) = split_patients_with_ids(
        data,
        args.seed,
        args.split_path,
        args.paper_protocol,
    )
    if args.validation_only_no_test:
        if not args.split_path:
            raise ValueError("--validation_only_no_test requires --split_path")
        split_manifest = json.loads(Path(args.split_path).read_text(encoding="utf-8"))
        if test_ids or data_test:
            raise ValueError("--validation_only_no_test requires an empty test_patient_indices list")
        if not bool(split_manifest.get("test_split_omitted")):
            raise ValueError("--validation_only_no_test split manifest must set test_split_omitted=true")
    if args.eval_split == "train":
        final_eval_data = data_train
        final_eval_ids = train_ids
    elif args.eval_split == "test":
        final_eval_data = data_test
        final_eval_ids = test_ids
    else:
        final_eval_data = data_eval
        final_eval_ids = eval_ids
    target_start_visit = target_start_visit_for(args)
    med_tensor = torch.arange(voc_size[2], device=device, dtype=torch.long)
    allowed_graph_signatures = (
        select_graph_signatures(data_train, args.cuda_graph_signatures)
        if args.cuda_graphs and device.type == "cuda"
        else None
    )

    model = build_model(args, voc_size, ehr_adj, ddi_adj_np, device)
    if args.Test:
        if not args.resume_path:
            raise ValueError("--Test requires --resume_path")
        checkpoint = seer_security.load_checkpoint(args.resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint)
        if args.representation_output:
            export_hidden_representations(
                model,
                final_eval_data,
                med_tensor,
                voc_size,
                ddi_mask_h,
                ddi_adj,
                args,
                split_name=args.eval_split,
                target_start_visit=target_start_visit,
            )
            return 0
        graph_runtime = None
        if args.cuda_graphs and device.type == "cuda":
            graph_runtime = SEERAuxiliaryCudaGraphRuntime(
                model,
                med_tensor,
                ddi_adj,
                voc_size[2],
                args.coef,
                args.target_ddi,
                allowed_signatures=allowed_graph_signatures,
            )
        tic = time.time()
        metrics = evaluate(
            model,
            final_eval_data,
            med_tensor,
            voc_size,
            ddi_mask_h,
            ddi_adj,
            ddi_adj_np,
            args,
                split_name=args.eval_split,
                target_start_visit=target_start_visit,
                patient_ids=final_eval_ids,
                graph_runtime=graph_runtime,
        )
        print(
            "evaluation time: {:.2f}, DDI Rate: {:.4f}, Jaccard: {:.4f}, PRAUC: {:.4f}, "
            "AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}".format(
                time.time() - tic, *metrics
            )
        )
        return 0

    if args.init_checkpoint:
        initialization = seer_security.load_checkpoint(args.init_checkpoint, map_location=device)
        initialization_state = (
            initialization["model"]
            if isinstance(initialization, dict) and "model" in initialization
            else initialization
        )
        model.load_state_dict(initialization_state, strict=True)
        print(f"warm-start model weights: {args.init_checkpoint}")

    print("parameters", get_n_params(model))
    use_cuda_graphs = args.cuda_graphs and device.type == "cuda"
    if use_cuda_graphs:
        optimizer = Adam(
            model.parameters(),
            lr=args.lr,
            fused=True,
            capturable=True,
        )
        graph_runtime = SEERAuxiliaryCudaGraphRuntime(
            model,
            med_tensor,
            ddi_adj,
            voc_size[2],
            args.coef,
            args.target_ddi,
            optimizer,
            allowed_graph_signatures,
        )
    else:
        optimizer = Adam(model.parameters(), lr=args.lr)
        graph_runtime = None
    history: dict[str, list[float]] = defaultdict(list)
    best_epoch = -1
    best_ja = -1.0
    best_checkpoint = checkpoint_path_for(args)
    latest_checkpoint = checkpoint_path_for(args, "latest.model")
    history_path = Path(args.output_dir) / "history.json" if args.output_dir else Path("saved") / args.model_name / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epoch):
        tic = time.time()
        print(f"\nepoch {epoch + 1} --------------------------")
        loss = train_epoch(
            model,
            data_train,
            med_tensor,
            voc_size,
            ddi_mask_h,
            ddi_adj,
            ddi_adj_np,
            optimizer,
            args.coef,
            args.target_ddi,
            device,
            graph_runtime,
        )
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = evaluate(
            model,
            data_eval,
            med_tensor,
            voc_size,
            ddi_mask_h,
            ddi_adj,
            ddi_adj_np,
            args,
            split_name="valid",
            target_start_visit=target_start_visit,
            patient_ids=eval_ids,
            epoch=epoch,
            graph_runtime=graph_runtime,
        )
        history["loss"].append(loss)
        history["ja"].append(float(ja))
        history["ddi_rate"].append(float(ddi_rate))
        history["prauc"].append(float(prauc))
        history["avg_p"].append(float(avg_p))
        history["avg_r"].append(float(avg_r))
        history["avg_f1"].append(float(avg_f1))
        history["avg_med"].append(float(avg_med))
        print(
            "training time: {:.2f}, loss: {:.4f}, DDI Rate: {:.4f}, Jaccard: {:.4f}, "
            "PRAUC: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}".format(
                time.time() - tic, loss, ddi_rate, ja, prauc, avg_f1, avg_med
            )
        )
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "initialization": {
                "mode": "model_weights_only" if args.init_checkpoint else "random",
                "checkpoint": args.init_checkpoint,
                "optimizer_reused": False,
                "rng_reused": False,
            },
            "rng_state": {
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "numpy": (lambda state: (state[0], state[1].tolist(), state[2], state[3], state[4]))(np.random.get_state()),
            },
        }
        torch.save(payload, open(latest_checkpoint, "wb"))
        if args.save_every_epoch:
            epoch_checkpoint = checkpoint_path_for(args, f"epoch_{epoch + 1:02d}.model")
            torch.save(payload, open(epoch_checkpoint, "wb"))
        if ja > best_ja:
            best_epoch = epoch
            best_ja = float(ja)
            torch.save(payload, open(best_checkpoint, "wb"))
        history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"best_epoch: {best_epoch}")
        if args.early_stop >= 0 and epoch - best_epoch > args.early_stop:
            print("Early Stop...")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
