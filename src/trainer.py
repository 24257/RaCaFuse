
import sys as _seer_sys
from pathlib import Path as _SeerPath
_seer_sys.path.insert(0, str(_SeerPath(__file__).resolve().parents[1] / "src"))
import seer_security
#%%
import sys
from pathlib import Path
import torch.nn.functional as F
from torch.optim import Adam

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from seer_utils import *;
from seer_model import SEER
from cache_identity import content_digest, file_digest
from action_guided_decode import action_score_fusion_logits, direct_action_decode, direct_action_scores
from training_losses import action_transition_loss as action_heads_transition_loss
from training_losses import action_count_loss as action_heads_count_loss
from training_losses import action_ddi_add_loss as action_heads_ddi_add_loss
from training_losses import masked_multilabel_bce, masked_transition_cross_entropy
from treatment_episode import build_episode_retrieval_context, build_episode_retrieval_priors, episode_experience_dim
import argparse
import copy;
import json
import random
import zlib

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=3, help='3(mimic-iii), 4(mimic-iv), or 5(eicu)')
parser.add_argument('--cuda', type=int, default=0, help='visible CUDA device index; -1 selects CPU')
parser.add_argument('--data_root_override', type=str, default='', help='optional dataset directory containing records_final.pkl/voc_final.pkl/ddi_A_final.pkl/ehr.pkl')
parser.add_argument('--disable_procedure_branch', action='store_true', help='mask procedure inputs and bypass the PatientRep procedure encoder')
parser.add_argument('--dim', type=int, default=256, help='embedding dimension')
parser.add_argument('--batch', type=int, default=32, help='mini-batch size')
parser.add_argument('--visit', type=int, default=3, help='history visit length')
parser.add_argument('--seed', type=int, default=1203, help='seed')
parser.add_argument('--test', type=int, default=0, help='test mode')
parser.add_argument('--save_model', type=int, default=0, help='save model')
parser.add_argument("--save_model_path", type=str, default="weights/seer_base_1.pth", help="path for saving the best model when --save_model > 0")
parser.add_argument("--save_model_selection", type=str, default="best", choices=["best", "last"], help="whether --save_model saves the best validation checkpoint or the final epoch checkpoint")
parser.add_argument("--save_every_epoch_dir", type=str, default="", help="optional directory for retaining the current model state after every validated epoch")
parser.add_argument("--init_model_checkpoint", type=str, default="", help="checkpoint used to initialize the trainable model before optimization")
parser.add_argument("--init_model_strict", action="store_true", help="strictly match checkpoint keys when initializing the trainable model")
parser.add_argument("--init_model_ignore_size_mismatch", action="store_true", help="when initializing, skip checkpoint tensors whose names exist but shapes differ")
parser.add_argument("--eval_model_checkpoint", type=str, default="", help="checkpoint loaded for --eval_only validation/test runs")
parser.add_argument("--eval_only", action="store_true", help="skip training and evaluate --eval_model_checkpoint with the configured decoder/search settings")
parser.add_argument('--epoches', type=int, default=80, help='epoches')
parser.add_argument("--lr", type=float, default=0.0002, help="Adam learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-6, help="Adam weight decay")
parser.add_argument('--temporal_type', type=str, default='mamba', choices=['mamba', 'gru', 'transformer'], help='shared temporal backend')
parser.add_argument('--patient_temporal_type', type=str, default='', choices=['', 'mamba', 'gru', 'transformer'], help='patient branch temporal backend override')
parser.add_argument('--med_temporal_type', type=str, default='gru', choices=['', 'mamba', 'gru', 'transformer'], help='medication branch temporal backend override')
parser.add_argument('--med_history_definition', type=str, default='input_order', choices=['input_order', 'real_time'], help='medication old/new history semantics')
parser.add_argument('--output_mix_weight', type=float, default=1.0, help='final mix weight for the direct prediction branch')
parser.add_argument('--output_mix_mode', type=str, default='fixed', choices=['fixed', 'learnable_scalar', 'dynamic_gate'], help='final output fusion strategy')
parser.add_argument('--output_branch_calibration', type=str, default='none', choices=['none', 'o2_scale_bias'], help='optional calibration for the medication matching branch before final fusion')
parser.add_argument('--med_scorer_type', type=str, default='cosine', choices=['cosine', 'dot', 'bilinear', 'mlp'], help='medication candidate scoring strategy')
parser.add_argument('--med_candidate_encoder_type', type=str, default='shared', choices=['shared', 'split'], help='medication candidate encoder strategy')
parser.add_argument('--med_fusion_type', type=str, default='fixed_sum', choices=['fixed_sum', 'gated_sum'], help='medication sequence/graph fusion strategy')
parser.add_argument('--patient_fusion_type', type=str, default='sum', choices=['sum', 'gated_sum'], help='patient diag/proc/med fusion strategy')
parser.add_argument('--patient_query_type', type=str, default='flatten', choices=['flatten', 'mean_pool', 'attention_pool', 'learned_query'], help='patient query projection strategy')
parser.add_argument('--direct_state_source', type=str, default='raw', choices=['raw', 'query', 'raw_query_sum', 'raw_query_gated'], help='state used by the direct medication-logit branch')
parser.add_argument('--direct_query_gate_init', type=float, default=-3.0, help='initial logit for the gated temporal-query residual in the direct branch')
parser.add_argument("--backbone_innovation_mode", type=str, default="none", choices=["none", "action_retrieval_graph_count"], help="enable the staged action-aware backbone line without changing decoder, retrieval, graph, or count defaults")
parser.add_argument("--multitask_mode", type=str, default="none", choices=["none", "clinical_state"])
parser.add_argument("--state_aux_weight", type=float, default=0.05)
parser.add_argument("--transition_aux_weight", type=float, default=0.05)
parser.add_argument("--med_count_mode", type=str, default="none", choices=["none", "aux"])
parser.add_argument("--med_count_aux_weight", type=float, default=0.02)
parser.add_argument(
    "--med_count_target",
    type=str,
    default="label_count",
    choices=["label_count", "score_best_count", "graph_score_best_count"],
    help="target used by the auxiliary medication-count head",
)
parser.add_argument("--freeze_base_for_med_count", action="store_true", help="freeze the V2 backbone and train only the medication-count head")
parser.add_argument("--bce_weight", type=float, default=1.0, help="weight for the primary multi-label BCE loss")
parser.add_argument("--action_transition_weight", type=float, default=0.0, help="auxiliary loss weight for action-aware continue/add/remove heads")
parser.add_argument("--continue_aux_weight", type=float, default=1.0, help="relative auxiliary weight for continue action head")
parser.add_argument("--add_aux_weight", type=float, default=1.0, help="relative auxiliary weight for add action head")
parser.add_argument("--remove_aux_weight", type=float, default=1.0, help="relative auxiliary weight for remove action head")
parser.add_argument("--add_pos_weight", type=float, default=1.0, help="positive-label weight applied only inside the masked add action BCE")
parser.add_argument("--add_focal_gamma", type=float, default=0.0, help="optional focal gamma applied only inside the masked add action BCE")
parser.add_argument("--action_ddi_loss_weight", type=float, default=0.0, help="auxiliary DDI-risk regularization weight for add action probabilities")
parser.add_argument("--add_ddi_loss_weight", type=float, default=1.0, help="relative weight inside the add-head DDI-risk regularizer")
parser.add_argument("--ddi_risk_threshold", type=float, default=0.0, help="ignore previous-med DDI risk values at or below this threshold for add-head DDI loss")
parser.add_argument("--action_transition_mode", type=str, default="none", choices=["none", "heads"], help="trainable backbone heads for continue/add/remove transition diagnosis")
parser.add_argument("--action_transition_hidden_dim", type=int, default=0, help="hidden size for action transition heads; 0 uses patient-state dimension")
parser.add_argument("--action_transition_dropout", type=float, default=0.1)
parser.add_argument("--action_transition_threshold", type=float, default=0.5, help="threshold for action-head metrics")
parser.add_argument("--action_fusion_trainable", action="store_true", help="enable the trainable action-guided fusion layer")
parser.add_argument("--action_fusion_type", type=str, default="trainable_scalar", choices=["trainable_scalar", "routed", "ffn", "simple_ffn_routed", "routed_residual", "aclr_residual"], help="checkpointed trainable action fusion decoder")
parser.add_argument("--action_fusion_loss_weight", type=float, default=0.0, help="auxiliary BCE weight for trainable action-fused logits")
parser.add_argument("--action_fusion_init_from_p4", action="store_true", help="initialize trainable action fusion parameters from the p4 calibrated action-best neighborhood")
parser.add_argument("--action_fusion_freeze_weights", action="store_true", help="freeze wc/wa/wr while learning action fusion scale and bias")
parser.add_argument("--routed_fusion_hidden_dim", type=int, default=128, help="hidden size for routed action fusion")
parser.add_argument("--routed_fusion_drug_emb_dim", type=int, default=32, help="drug identity embedding size for routed action fusion")
parser.add_argument("--routed_fusion_dropout", type=float, default=0.1, help="dropout for routed action fusion")
parser.add_argument("--routed_fusion_entropy_weight", type=float, default=0.0, help="optional entropy regularizer weight for routed action fusion")
parser.add_argument("--routed_residual_freeze_base", action="store_true", help="freeze p4 scalar base inside routed residual fusion")
parser.add_argument("--routed_residual_train_base", action="store_true", help="allow p4 scalar base to train inside routed residual fusion")
parser.add_argument("--routed_residual_scale", type=float, default=0.1, help="initial residual scale for routed residual fusion")
parser.add_argument("--routed_residual_max_gate", type=float, default=1.0, help="hard upper bound for routed residual gate")
parser.add_argument("--routed_residual_gate_init_bias", type=float, default=-3.0, help="initial bias for routed residual gate")
parser.add_argument("--routed_residual_l2_weight", type=float, default=0.0, help="optional L2 penalty on routed residual deltas")
parser.add_argument("--routed_residual_add_penalty_weight", type=float, default=0.0, help="optional penalty on positive add residuals")
parser.add_argument("--routed_residual_entropy_weight", type=float, default=0.0, help="optional entropy regularizer for routed residual fusion")
parser.add_argument("--routed_residual_load_balance_weight", type=float, default=0.0, help="optional expert load-balance penalty for routed residual fusion")
parser.add_argument("--use_trainable_action_fusion", action="store_true", help="decode/evaluate medication predictions with checkpoint trainable action fusion logits")
parser.add_argument("--freeze_backbone_for_action_fusion", action="store_true", help="freeze all model parameters except the trainable action fusion layer")
parser.add_argument("--fusion_bce_weight", type=float, default=1.0, help="BCE term weight inside the trainable action fusion objective")
parser.add_argument("--fusion_soft_jaccard_weight", type=float, default=0.0, help="soft Jaccard term weight inside the trainable action fusion objective")
parser.add_argument("--fusion_count_anchor_weight", type=float, default=0.0, help="SmoothL1 predicted-count term weight inside the trainable action fusion objective")
parser.add_argument("--fusion_ddi_soft_weight", type=float, default=0.0, help="soft DDI penalty weight inside the trainable action fusion objective")
parser.add_argument("--action_decode_mode", type=str, default="none", choices=["none", "score_fusion", "direct_action"], help="optional action-guided medication set decoding")
parser.add_argument("--action_decode_continue_weight", type=float, default=0.0)
parser.add_argument("--action_decode_add_weight", type=float, default=0.0)
parser.add_argument("--action_decode_remove_weight", type=float, default=0.0)
parser.add_argument("--action_decode_threshold", type=float, default=0.5)
parser.add_argument("--count_aux_weight", type=float, default=0.0, help="auxiliary loss weight for action-aware count heads")
parser.add_argument("--total_count_weight", type=float, default=1.0)
parser.add_argument("--continue_count_weight", type=float, default=1.0)
parser.add_argument("--add_count_weight", type=float, default=1.0)
parser.add_argument("--remove_count_weight", type=float, default=1.0)
parser.add_argument("--count_decode_mode", type=str, default="none", choices=["none", "total_topk", "action_split_topk"], help="optional count-aware decode mode using action count heads")
parser.add_argument("--count_decode_anchor_ddi", type=float, default=-1.0, help="optional validation DDI anchor for count-aware selection; negative uses the current checkpoint action-score anchor")
parser.add_argument("--count_decode_anchor_avg_med", type=float, default=-1.0, help="optional validation AvgMed anchor for count-aware selection; negative uses the current checkpoint action-score anchor")
parser.add_argument("--action_transition_bce_weight", type=float, default=0.0, help="deprecated alias for --action_transition_weight")
parser.add_argument("--action_continue_loss_weight", type=float, default=1.0, help="deprecated alias for --continue_aux_weight")
parser.add_argument("--action_add_loss_weight", type=float, default=1.0, help="deprecated alias for --add_aux_weight")
parser.add_argument("--action_remove_loss_weight", type=float, default=1.0, help="deprecated alias for --remove_aux_weight")
parser.add_argument("--seq_decoder_mode", type=str, default="none", choices=["none", "aux", "hidr_memory", "hidr_transformer"])
parser.add_argument("--seq_decoder_weight", type=float, default=0.1)
parser.add_argument("--seq_decoder_max_len", type=int, default=30)
parser.add_argument("--seq_decoder_dropout", type=float, default=0.1)
parser.add_argument("--seq_transformer_hidden_dim", type=int, default=0, help="hidden size for HI-DR-style Transformer sequence decoder; 0 uses visit embedding dim")
parser.add_argument("--seq_transformer_nhead", type=int, default=4, help="attention heads for HI-DR-style Transformer sequence decoder")
parser.add_argument("--seq_transformer_layers", type=int, default=1, help="layers for HI-DR-style Transformer sequence decoder")
parser.add_argument("--seq_decoder_order", type=str, default="index", choices=["index", "rare_train", "rare_all"], help="teacher-forcing medication order for the sequence decoder")
parser.add_argument("--seq_copy_mode", type=str, default="none", choices=["none", "history_attention"], help="HI-DR-style history copy branch for the sequence decoder")
parser.add_argument("--seq_copy_gate_init", type=float, default=1.0, help="initial generate-gate logit for sequence copy mixing")
parser.add_argument("--seq_copy_recency_decay", type=float, default=0.7, help="recency decay used by sequence history copy attention")
parser.add_argument("--seq_retrieval_copy_mode", type=str, default="none", choices=["none", "retrieval_attention"], help="copy medications from Top-K similar visits inside the sequence decoder")
parser.add_argument("--seq_retrieval_copy_gate_init", type=float, default=1.0, help="initial generate-gate logit for sequence retrieval copy mixing")
parser.add_argument("--seq_hidr_gumbel_tau", type=float, default=0.6, help="gumbel-softmax temperature for HI-DR memory history selection")
parser.add_argument("--seq_hidr_att_tau", type=float, default=20.0, help="attention temperature for HI-DR memory visit weighting")
parser.add_argument("--freeze_base_for_seq_decoder", action="store_true", help="freeze the V2 backbone and train only the sequence decoder")
parser.add_argument("--seq_decoder_end_bias", type=float, default=0.0, help="fixed END-token logit bias for sequence decoding")
parser.add_argument("--seq_decoder_end_bias_search", action="store_true", help="search END-token logit bias for sequence decoders during validation")
parser.add_argument("--seq_decoder_end_bias_min", type=float, default=-2.0)
parser.add_argument("--seq_decoder_end_bias_max", type=float, default=2.0)
parser.add_argument("--seq_decoder_end_bias_step", type=float, default=0.5)
parser.add_argument("--seq_decoder_beam_size", type=int, default=4, help="beam size for HI-DR-style autoregressive sequence decoding")
parser.add_argument("--soft_jaccard_weight", type=float, default=0.0, help="auxiliary set-level soft Jaccard loss weight")
parser.add_argument("--soft_dice_weight", type=float, default=0.0, help="auxiliary set-level soft Dice loss weight")
parser.add_argument("--asymmetric_bce_weight", type=float, default=0.0, help="auxiliary asymmetric focal BCE weight for medication logits")
parser.add_argument("--asymmetric_bce_gamma_pos", type=float, default=0.0)
parser.add_argument("--asymmetric_bce_gamma_neg", type=float, default=2.0)
parser.add_argument("--asymmetric_bce_neg_clip", type=float, default=0.0)
parser.add_argument("--pairwise_rank_weight", type=float, default=0.0, help="auxiliary hard-negative pairwise ranking loss weight for main medication logits")
parser.add_argument("--pairwise_rank_mode", type=str, default="mean_positive", choices=["mean_positive", "all_positive"], help="positive anchor strategy for pairwise ranking loss")
parser.add_argument("--pairwise_rank_negatives", type=int, default=20, help="number of highest-scoring negative medications used per sample by pairwise ranking loss")
parser.add_argument("--pairwise_rank_margin", type=float, default=0.0, help="optional positive-vs-negative margin added inside pairwise ranking loss")
parser.add_argument("--listwise_rank_weight", type=float, default=0.0, help="auxiliary listwise positive-vs-hard-negative ranking loss weight")
parser.add_argument("--listwise_rank_negatives", type=int, default=40, help="number of highest-scoring negatives used by listwise ranking loss")
parser.add_argument("--listwise_rank_margin", type=float, default=0.0, help="optional margin for listwise ranking loss")
parser.add_argument("--train_graph_rank_weight", type=float, default=0.0, help="auxiliary hard-negative ranking loss on the final Graph+ decode score")
parser.add_argument("--train_graph_rank_mode", type=str, default="mean_positive", choices=["mean_positive", "all_positive"], help="positive anchor strategy for train-time Graph+ ranking loss")
parser.add_argument("--train_graph_rank_negatives", type=int, default=40, help="number of highest-scoring negative medications used by train-time Graph+ ranking loss")
parser.add_argument("--train_graph_rank_margin", type=float, default=0.0, help="optional positive-vs-negative margin added inside train-time Graph+ ranking loss")
parser.add_argument("--train_graph_boundary_rank_weight", type=float, default=0.0, help="auxiliary Graph+ boundary-swap loss for selected false positives vs missed true positives")
parser.add_argument("--train_graph_boundary_rank_mode", type=str, default="mean_missed", choices=["mean_missed", "all_missed"], help="missed-positive anchor strategy for train-time Graph+ boundary loss")
parser.add_argument("--train_graph_boundary_negatives", type=int, default=20, help="number of selected false positives used by train-time Graph+ boundary loss")
parser.add_argument("--train_graph_boundary_margin", type=float, default=0.0, help="optional margin for train-time Graph+ boundary loss")
parser.add_argument("--action_boundary_rank_weight", type=float, default=0.0, help="auxiliary action-aware boundary loss for continue/add/remove ranking swaps")
parser.add_argument("--action_boundary_continue_weight", type=float, default=1.0, help="relative weight for missed-continue vs selected-remove swaps")
parser.add_argument("--action_boundary_add_weight", type=float, default=1.0, help="relative weight for missed-add vs selected-new-false-positive swaps")
parser.add_argument("--action_boundary_global_weight", type=float, default=1.0, help="relative weight for missed-positive vs selected-false-positive swaps")
parser.add_argument("--action_boundary_negatives", type=int, default=20, help="number of selected hard negatives used by action-aware boundary loss")
parser.add_argument("--action_boundary_margin", type=float, default=0.0, help="optional margin for action-aware boundary loss")
parser.add_argument("--train_graph_count_weight", type=float, default=0.0, help="auxiliary loss aligning Graph+ count decode with a train-only count target")
parser.add_argument("--train_graph_count_mode", type=str, default="prob_sum", choices=["prob_sum", "expected_jaccard"], help="count estimate optimized by train-time Graph+ count loss")
parser.add_argument("--train_graph_count_target", type=str, default="label_count", choices=["label_count", "best_count"], help="target count used by train-time Graph+ count calibration")
parser.add_argument("--train_graph_score_distill_weight", type=float, default=0.0, help="KL loss distilling Graph+ ranking scores into the main medication logits")
parser.add_argument("--train_graph_score_distill_temperature", type=float, default=2.0)
parser.add_argument("--train_graph_score_distill_topk", type=int, default=40, help="top-K teacher candidates used by Graph+ score distillation; 0 uses the full vocabulary")
parser.add_argument("--freeze_base_for_graph_score_distill", action="store_true", help="freeze the V2 backbone and train only the direct medication-logit head")
parser.add_argument("--teacher_anchor_components", type=str, default="", help="train-only NPZ containing row-aligned locked-expert components")
parser.add_argument("--teacher_rank_distill_weight", type=float, default=0.0, help="weight for row-standardized locked-expert ranking distillation")
parser.add_argument("--teacher_count_distill_weight", type=float, default=0.0, help="weight for locked-expert predicted-count distillation")
parser.add_argument("--teacher_parameter_anchor_weight", type=float, default=0.0, help="weight for anchoring trainable parameters to their initialized values")
parser.add_argument("--posthoc_calibration_mode", type=str, default="none", choices=["none", "per_med_linear", "per_med_graph"], help="identity-initialized final logit calibration layer")
parser.add_argument("--freeze_base_for_posthoc_calibration", action="store_true", help="freeze the V2 backbone and train only the posthoc calibration layer")
parser.add_argument("--posthoc_graph_source", type=str, default="hybrid", choices=["model", "history", "retrieval", "hybrid"], help="prior source propagated through EHR/DDI graphs for per_med_graph calibration")
parser.add_argument("--soft_ddi_weight", type=float, default=0.0, help="auxiliary differentiable DDI-risk penalty on predicted medication probabilities")
parser.add_argument("--soft_ddi_power", type=float, default=1.0, help="power applied to medication probabilities before soft DDI penalty")
parser.add_argument("--prefix_aux_weight", type=float, default=0.0, help="BCE auxiliary loss on historical visits inside each target prefix")
parser.add_argument("--prefix_seq_decoder_weight", type=float, default=0.0, help="sequence-decoder auxiliary loss on historical/retrieved visits inside each target prefix")
parser.add_argument(
    "--prefix_metric_mode",
    type=str,
    default="target",
    choices=["target", "paper_prefix"],
    help="target evaluates only the target visit; paper_prefix also evaluates historical visits inside each HI-DR-style prefix",
)
parser.add_argument("--aux_head_dropout", type=float, default=0.1)
parser.add_argument("--clinical_feature_mode", type=str, default="none", choices=["none", "input"])
parser.add_argument("--clinical_fusion_type", type=str, default="clinical_gated_sum", choices=["clinical_gated_sum", "concat", "sum"])
parser.add_argument("--clinical_feature_dropout", type=float, default=0.1)
parser.add_argument("--clinical_temporal_fusion", type=str, default="add", choices=["add", "gated_residual"])
parser.add_argument("--clinical_temporal_gate_init", type=float, default=-3.0)
parser.add_argument("--retrieval_mode", type=str, default="none", choices=["none", "dxproc", "clinical", "dxproc_clinical", "history_med", "learned", "learned_clinical", "paper_m3", "official_topidx", "treatment_episode"])
parser.add_argument("--retrieval_topk", type=int, default=3)
parser.add_argument("--retrieval_alpha", type=float, default=0.5, help="non-clinical similarity weight for hybrid clinical retrieval")
parser.add_argument("--retrieval_weight", type=float, default=0.0, help="positive logit bias weight for the retrieved prescription prior")
parser.add_argument("--retrieval_prior_transform", type=str, default="centered", choices=["positive", "centered"])
parser.add_argument("--retrieval_copy_weight", type=float, default=0.0, help="probability-space copy weight for similar-visit medication priors")
parser.add_argument("--retrieval_copy_mode", type=str, default="none", choices=["none", "fixed_prob", "prob_mix"], help="fixed_prob is the old inference-time prior mix; prob_mix trains an HI-DR-style retrieved-medication copy gate")
parser.add_argument("--retrieval_copy_gate_init", type=float, default=2.0, help="initial generate-gate logit for learned retrieval copy mixing")
parser.add_argument("--retrieval_copy_dropout", type=float, default=0.1)
parser.add_argument("--retrieval_readout_mode", type=str, default="none", choices=["none", "logit_bias"], help="trainable HI-DR-style Top-K retrieved-medication readout fused as a final logit bias")
parser.add_argument("--retrieval_readout_weight", type=float, default=0.0, help="weight for the trainable retrieved-medication readout logit bias")
parser.add_argument("--retrieval_readout_gate_init", type=float, default=-2.0, help="initial gate logit for the retrieved-medication readout")
parser.add_argument("--retrieval_readout_dropout", type=float, default=0.1)
parser.add_argument("--retrieval_readout_prior_transform", type=str, default="logit", choices=["logit", "positive", "centered"], help="how the retrieved medication prior is converted before the readout logit bias")
parser.add_argument("--freeze_base_for_retrieval_readout", action="store_true", help="freeze the V2 backbone and train only the retrieval readout adapter")
parser.add_argument("--treatment_episode_adapter_mode", type=str, default="none", choices=["none", "logit_bias", "state", "both", "action"], help="model-side adapter that consumes treatment episode action/outcome/safety experience tensors")
parser.add_argument("--treatment_episode_adapter_weight", type=float, default=0.0, help="weight for the treatment episode experience adapter")
parser.add_argument("--treatment_episode_adapter_dropout", type=float, default=0.1)
parser.add_argument("--treatment_episode_adapter_gate_init", type=float, default=-3.0)
parser.add_argument("--freeze_base_for_treatment_episode_adapter", action="store_true", help="freeze the V2 backbone and train only the treatment episode adapter")
parser.add_argument("--treatment_episode_action_aux_weight", type=float, default=0.0, help="auxiliary add/continue/remove BCE on treatment episode adapter action heads")
parser.add_argument("--treatment_episode_action_continue_weight", type=float, default=1.0, help="relative treatment episode action-head weight for continued previous medications")
parser.add_argument("--treatment_episode_action_add_weight", type=float, default=1.0, help="relative treatment episode action-head weight for newly added medications")
parser.add_argument("--treatment_episode_action_remove_weight", type=float, default=1.0, help="relative treatment episode action-head weight for removed previous medications")
parser.add_argument("--treatment_episode_action_bias_weight", type=float, default=0.0, help="weight for fusing treatment episode add/continue/remove heads into final medication logits")
parser.add_argument("--treatment_episode_fused_action_loss_weight", type=float, default=0.0, help="auxiliary BCE on the fused treatment episode action residual used by final logits")
parser.add_argument("--treatment_episode_candidate_gain_weight", type=float, default=0.0, help="auxiliary candidate-level gain ranking loss for TET action heads")
parser.add_argument("--treatment_episode_candidate_gain_temperature", type=float, default=0.03)
parser.add_argument("--treatment_episode_candidate_gain_value_weight", type=float, default=0.25)
parser.add_argument("--treatment_episode_candidate_gain_pair_weight", type=float, default=0.25)
parser.add_argument("--treatment_episode_candidate_gain_pair_margin", type=float, default=0.01)
parser.add_argument("--treatment_episode_candidate_gain_min_pair_gap", type=float, default=0.001)
parser.add_argument("--treatment_episode_bias_scale_mode", type=str, default="none", choices=["none", "logit_abs_mean"], help="optional per-visit scaling for treatment episode final logit residuals")
parser.add_argument("--retrieval_query_rank_weight", type=float, default=0.0, help="auxiliary Top-K query ranking loss weight for static retrieved visits")
parser.add_argument("--retrieval_query_rank_negatives", type=int, default=5, help="number of hard lower-quality retrieved visits used by the query ranking loss")
parser.add_argument("--retrieval_query_rank_margin", type=float, default=0.05, help="margin for Top-K query ranking loss")
parser.add_argument("--retrieval_query_rank_min_delta", type=float, default=1e-6, help="minimum target-Jaccard gap between positive and negative retrieved visits")
parser.add_argument(
    "--retrieval_candidate_scope",
    type=str,
    default="train_only",
    choices=["train_only", "train_plus_own_history", "paper_like"],
    help=(
        "candidate pool for non-learned similar-visit retrieval; "
        "train_plus_own_history adds only the target split patients so their past visits can be used, "
        "while paper_like adds the whole current eval/test split like the paper protocol "
        "but still excludes same-patient current/future visits."
    ),
)
parser.add_argument(
    "--retrieval_exclude_self_future",
    action="store_true",
    help="exclude the same patient's current and future visits from retrieval candidates when split metadata is available",
)
parser.add_argument("--treatment_episode_state_weight", type=float, default=1.0, help="weight for current-state similarity inside treatment episode transfer")
parser.add_argument("--treatment_episode_dxproc_weight", type=float, default=1.0, help="diagnosis/procedure component weight for treatment episode state similarity")
parser.add_argument("--treatment_episode_clinical_weight", type=float, default=0.0, help="clinical-feature component weight for treatment episode state similarity")
parser.add_argument("--treatment_episode_history_weight", type=float, default=0.25, help="weight for overlap between current history medications and historical episode treatment")
parser.add_argument("--treatment_episode_action_weight", type=float, default=0.0, help="weight for optional target action-prior agreement when a probability prior is supplied")
parser.add_argument("--treatment_episode_outcome_weight", type=float, default=0.25, help="weight for favorable next-visit clinical-state change in historical episodes")
parser.add_argument("--treatment_episode_safety_weight", type=float, default=0.25, help="penalty weight for DDI and worsening-state burden in historical episodes")
parser.add_argument("--treatment_episode_score_transform", type=str, default="minmax", choices=["minmax", "sigmoid", "positive"], help="normalization applied to Top-K treatment episode transfer scores")
parser.add_argument("--treatment_episode_min_outcome_mask", type=float, default=1.0, help="minimum observed next-state labels required before using an episode outcome proxy")
parser.add_argument("--treatment_episode_history_mode", type=str, default="cumulative", choices=["cumulative", "immediate"], help="medication history scope used for treatment episode action labels")
parser.add_argument("--treatment_episode_history_similarity_mode", type=str, default="current", choices=["current", "history"], help="candidate component used for history-overlap retrieval scoring")
parser.add_argument(
    "--treatment_episode_action_similarity_mode",
    type=str,
    default="transition_jaccard",
    choices=["current", "transition", "transition_jaccard", "transition_cosine"],
    help="candidate component used for action-prior retrieval scoring",
)
parser.add_argument("--history_prior_weight", type=float, default=0.0, help="positive logit bias for medications seen in the patient's history")
parser.add_argument("--history_prior_mode", type=str, default="binary", choices=["binary", "recency"], help="how to score historical medications before adding the history prior")
parser.add_argument("--history_prior_decay", type=float, default=0.7, help="recency decay for historical medications; the immediate previous visit has weight 1.0")
parser.add_argument("--retrieval_refresh", type=str, default="epoch", choices=["static", "epoch"], help="refresh cadence for learned retrieval priors")
parser.add_argument("--retrieval_warmup_epochs", type=int, default=1, help="epochs without learned retrieval priors before refreshing learned Top-K")
parser.add_argument("--retrieval_context_mode", type=str, default="none", choices=["none", "attention"])
parser.add_argument("--retrieval_context_source", type=str, default="learned", choices=["learned", "static"], help="source of Top-K retrieval memory when retrieval_context_mode is enabled")
parser.add_argument("--retrieval_context_target", type=str, default="logit", choices=["logit", "state"])
parser.add_argument("--retrieval_context_weight", type=float, default=0.0)
parser.add_argument("--retrieval_context_dropout", type=float, default=0.1)
parser.add_argument("--retrieval_context_gate_init", type=float, default=-3.0)
parser.add_argument("--external_memory_mode", type=str, default="none", choices=["none", "state", "logit", "both"], help="AAAI-style learnable global memory branch")
parser.add_argument("--external_memory_num", type=int, default=32, help="number of learnable global memory slots")
parser.add_argument("--external_memory_hidden_size", type=int, default=0, help="hidden size for learnable global memory; 0 uses patient state size")
parser.add_argument("--external_memory_weight", type=float, default=0.0)
parser.add_argument("--external_memory_dropout", type=float, default=0.1)
parser.add_argument("--external_memory_gate_init", type=float, default=-3.0)
parser.add_argument("--local_memory_mode", type=str, default="none", choices=["none", "state"], help="EXCERF-style local MemNN over the current patient's previous visits")
parser.add_argument("--local_memory_weight", type=float, default=0.0)
parser.add_argument("--local_memory_dropout", type=float, default=0.1)
parser.add_argument("--local_memory_gate_init", type=float, default=-3.0)
parser.add_argument("--ehr_graph_mode", type=str, default="binary", choices=["binary", "directed_weighted"], help="binary uses the stored EHR graph; directed_weighted builds HI-DR-style EHR graph+ from training co-prescriptions")
parser.add_argument("--ehr_graph_cache_dir", type=str, default="../cache/ehr_graph", help="directory for cached directed weighted EHR graph+")
parser.add_argument("--patient_graph_mode", type=str, default="none", choices=["none", "node_gate", "edge_bias"], help="patient-conditioned EHR/DDI graph adaptation")
parser.add_argument("--patient_graph_ehr_scale", type=float, default=0.25)
parser.add_argument("--patient_graph_ddi_scale", type=float, default=0.25)
parser.add_argument("--patient_graph_dropout", type=float, default=0.1)
parser.add_argument("--graph_logit_bias_weight", type=float, default=0.0, help="residual weight for graph-conditioned medication similarity logits")
parser.add_argument("--graph_edge_bias_weight", type=float, default=0.0, help="residual weight for patient-conditioned EHR/DDI edge prior logit bias")
parser.add_argument("--graph_edge_ddi_weight", type=float, default=1.0, help="DDI penalty weight inside patient-conditioned edge bias")
parser.add_argument("--graph_edge_prior_source", type=str, default="history", choices=["model", "history", "retrieval", "hybrid"])
parser.add_argument("--graph_edge_detach_prior", action="store_true", default=True, help="detach edge-bias medication prior before graph propagation")
parser.add_argument("--graph_edge_no_detach_prior", action="store_false", dest="graph_edge_detach_prior", help="allow edge-bias gradients through the medication prior")
parser.add_argument("--freeze_base_for_edge_bias", action="store_true", help="freeze the V2 backbone and train only patient-conditioned edge-bias parameters")
parser.add_argument("--graph_med_encoder_mode", type=str, default="none", choices=["none", "static", "patient_gate"], help="graph-aware medication representation encoder")
parser.add_argument("--graph_med_ehr_weight", type=float, default=0.0, help="EHR message weight inside graph-aware medication encoder")
parser.add_argument("--graph_med_ddi_weight", type=float, default=0.0, help="DDI message penalty weight inside graph-aware medication encoder")
parser.add_argument("--graph_med_gate_hidden", type=int, default=128, help="hidden size for patient-gated graph-aware medication encoder")
parser.add_argument("--graph_med_dropout", type=float, default=0.1, help="dropout inside graph-aware medication encoder")
parser.add_argument("--patient_med_scorer_mode", type=str, default="none", choices=["none", "lowrank_bias"], help="patient-conditioned residual scorer for the graph medication branch")
parser.add_argument("--patient_med_scorer_rank", type=int, default=32, help="low-rank dimension for patient-conditioned medication scoring")
parser.add_argument("--patient_med_scorer_weight", type=float, default=0.0, help="residual weight applied to the patient-conditioned medication scorer")
parser.add_argument("--patient_med_scorer_dropout", type=float, default=0.1)
parser.add_argument("--freeze_base_for_patient_med_scorer", action="store_true", help="freeze the V2 backbone and train only patient-conditioned medication scorer parameters")
parser.add_argument("--main_med_scorer_mode", type=str, default="none", choices=["none", "feature_delta"], help="identity-initialized replacement scorer applied after the SEER output mixer")
parser.add_argument("--main_med_scorer_hidden_dim", type=int, default=128)
parser.add_argument("--main_med_scorer_rank", type=int, default=32)
parser.add_argument("--main_med_scorer_dropout", type=float, default=0.1)
parser.add_argument("--freeze_base_for_main_med_scorer", action="store_true", help="freeze the V2 backbone and train only the replacement medication scorer")
parser.add_argument("--main_med_scorer_delta_l2_weight", type=float, default=0.0, help="penalize replacement scorer logit deltas to keep the no-op reference neighborhood")
parser.add_argument("--main_med_scorer_count_kl_weight", type=float, default=0.0, help="match replacement scorer probability mass to its detached no-delta reference")
parser.add_argument("--current_med_attention_mode", type=str, default="none", choices=["none", "history_cross_attention"], help="current diagnosis/procedure attention over historical medications")
parser.add_argument("--current_med_attention_dropout", type=float, default=0.1)
parser.add_argument("--current_med_attention_gate_init", type=float, default=-3.0)
parser.add_argument("--freeze_base_for_current_med_attention", action="store_true", help="freeze the V2 backbone and train only current-code/history-med attention parameters")
parser.add_argument("--own_history_context_mode", type=str, default="none", choices=["none", "visit_attention"], help="current-visit attention over the patient's own historical visits")
parser.add_argument("--own_history_context_weight", type=float, default=0.0, help="residual weight for relevant own-history visit context")
parser.add_argument("--own_history_context_dropout", type=float, default=0.1)
parser.add_argument("--own_history_context_gate_init", type=float, default=-3.0)
parser.add_argument("--own_history_context_recency_decay", type=float, default=1.0, help="optional recency bias for own-history visit attention; 1.0 disables recency bias")
parser.add_argument("--freeze_base_for_own_history_context", action="store_true", help="freeze the V2 backbone and train only own-history context parameters")
parser.add_argument("--history_copy_mode", type=str, default="none", choices=["none", "gated", "prob_mix"], help="learned history-medication copy branch")
parser.add_argument("--history_copy_weight", type=float, default=0.0, help="weight for the learned history copy branch")
parser.add_argument("--history_copy_prior_mode", type=str, default="binary", choices=["binary", "recency"], help="historical medication prior used by the copy branch")
parser.add_argument("--history_copy_decay", type=float, default=0.7)
parser.add_argument("--history_copy_gate_init", type=float, default=-2.0)
parser.add_argument("--history_copy_dropout", type=float, default=0.1)
parser.add_argument("--insert_retrieved_visits", action="store_true", help="insert Top-K similar training visits before the current target visit")
parser.add_argument(
    "--insert_visit_mode",
    type=str,
    default="dxproc",
    choices=[
        "dxproc",
        "clinical",
        "dxproc_clinical",
        "learned",
        "learned_clinical",
        "dxproc_learned",
        "dxproc_clinical_learned",
        "history_med",
        "paper_m3",
    ],
)
parser.add_argument("--insert_visit_topk", type=int, default=3)
parser.add_argument("--insert_visit_alpha", type=float, default=0.5)
parser.add_argument(
    "--paper_m3_topk",
    type=int,
    default=0,
    help="Top-K similar visits for paper_m3 retrieval priors/context; 0 reuses --insert_visit_topk for backward compatibility",
)
parser.add_argument(
    "--paper_m3_checkpoint",
    type=str,
    default="",
    help="checkpoint used as the fixed Phi model for HI-DR-style M3 similar-visit insertion",
)
parser.add_argument("--paper_m3_cache_dir", type=str, default="../cache/paper_m3", help="directory for cached HI-DR-style M3 inserted visits")
parser.add_argument("--paper_m3_no_cache", action="store_true", help="disable reading/writing cached HI-DR-style M3 inserted visits")
parser.add_argument(
    "--paper_m3_query_similarity",
    type=str,
    default="cosine",
    choices=["cosine", "dot", "angular", "euclidean", "manhattan", "chebyshev", "pearson", "spearman", "rbf"],
    help="similarity metric used for fixed Phi patient-query vectors in paper_m3 retrieval",
)
parser.add_argument("--paper_m3_rbf_gamma", type=float, default=0.00390625, help="gamma for --paper_m3_query_similarity rbf; default is 1/256")
parser.add_argument(
    "--paper_m3_query_transform",
    type=str,
    default="none",
    choices=["none", "label_ridge", "label_ridge_sigmoid", "label_ridge_to_meds", "label_ridge_sigmoid_to_meds"],
    help="optional train-label supervised transform applied to Phi queries before paper_m3 similarity scoring",
)
parser.add_argument("--paper_m3_query_transform_ridge", type=float, default=1.0, help="ridge strength for --paper_m3_query_transform label_ridge*")
parser.add_argument("--paper_m3_label_prior_weight", type=float, default=0.0, help="weight for adding train-only label-ridge predicted medication prior into paper_m3 retrieval priors")
parser.add_argument("--paper_m3_clinical_weight", type=float, default=0.0, help="clinical-status similarity weight mixed into HI-DR-style M3 retrieval")
parser.add_argument("--paper_m3_clinical_alpha", type=float, default=0.8, help="query-representation weight when mixing paper M3 and clinical similarity")
parser.add_argument("--paper_m3_dxproc_weight", type=float, default=0.0, help="dx/proc Jaccard similarity weight added to HI-DR-style M3 retrieval")
parser.add_argument("--paper_m3_history_overlap_weight", type=float, default=0.0, help="target-history medication overlap weight mixed into paper_m3 retrieval")
parser.add_argument("--official_topidx_dir", type=str, default="", help="directory containing official HI-DR *_train/eval/test_top*.pkl Top-K index artifacts")
parser.add_argument("--paper_protocol", action="store_true", help="match HI-DR protocol: drop single-visit patients and predict only visits with history")
parser.add_argument("--split_path", type=str, default="", help="optional patient-level split manifest; --eval_split eval uses valid_patient_indices and --eval_split test uses test_patient_indices")
parser.add_argument(
    "--include_first_visit_in_paper_protocol",
    action="store_true",
    help="with --paper_protocol, also supervise/evaluate the first visit like HI-DR's sequence-level evaluator",
)
parser.add_argument("--eval_split", type=str, default="eval", choices=["train", "eval", "test"], help="which split the validation/eval-only path should use")
parser.add_argument("--train_include_test_split", action="store_true", help="train on the original train+test patient splits while keeping the final eval split held out")
parser.add_argument(
    "--validation_only_no_test",
    action="store_true",
    help="validation-only protocol: do not materialize the test split and reject every test-mode option",
)
parser.add_argument("--best_metric", type=str, default="prauc", choices=["prauc", "ja", "f1"], help="validation metric used to select the best checkpoint")
parser.add_argument("--metric_aggregation", type=str, default="patient", choices=["patient", "visit"], help="patient matches the V2 legacy evaluator; visit is closer to HI-DR's per-target-visit averaging")
parser.add_argument("--eval_threshold", type=float, default=0.3, help="fixed medication probability threshold used when threshold search is disabled")
parser.add_argument("--threshold_search", action="store_true", help="select the medication probability threshold on the validation set each epoch")
parser.add_argument("--threshold_min", type=float, default=0.15)
parser.add_argument("--threshold_max", type=float, default=0.5)
parser.add_argument("--threshold_step", type=float, default=0.01)
parser.add_argument("--threshold_metric", type=str, default="", choices=["", "ja", "f1"], help="metric used to choose threshold; defaults to best_metric when threshold-dependent, otherwise JA")
parser.add_argument(
    "--decode_mode",
    type=str,
    default="threshold",
    choices=[
        "threshold",
        "per_med_threshold",
        "history_threshold",
        "history_rank_fixed_topk",
        "history_rank_threshold_count",
        "history_rank_expected_count",
        "history_rank_history_count",
        "history_rank_predicted_count",
        "history_retrieval_rank_fixed_topk",
        "history_retrieval_rank_threshold_count",
        "history_retrieval_rank_expected_count",
        "history_retrieval_rank_history_count",
        "history_retrieval_rank_predicted_count",
        "history_retrieval_freq_rank_fixed_topk",
        "history_retrieval_freq_rank_threshold_count",
        "history_retrieval_freq_rank_expected_count",
        "history_retrieval_freq_rank_expected_jaccard_count",
        "history_retrieval_freq_rank_history_count",
        "history_retrieval_freq_rank_predicted_count",
        "history_retrieval_freq_graph_rank_expected_count",
        "history_retrieval_freq_graph_rank_expected_jaccard_count",
        "history_retrieval_freq_graph_rank_count_pair_mix",
        "history_retrieval_freq_graph_rank_predicted_count",
        "history_retrieval_freq_graph_rank_retrieval_count",
        "history_retrieval_freq_ddi_rank_expected_count",
        "history_retrieval_seq_rank_fixed_topk",
        "history_retrieval_seq_rank_threshold_count",
        "history_retrieval_seq_rank_expected_count",
        "history_retrieval_seq_rank_history_count",
        "history_retrieval_seq_rank_predicted_count",
        "retrieval_rank_fixed_topk",
        "retrieval_rank_threshold_count",
        "retrieval_rank_expected_count",
        "retrieval_rank_history_count",
        "retrieval_rank_predicted_count",
        "fixed_topk",
        "oracle_count",
        "expected_count",
        "history_count",
        "predicted_count",
        "seq_greedy",
        "seq_hidr",
        "seq_beam",
        "seq_prob_fixed_topk",
        "seq_prob_expected_count",
        "seq_prob_history_count",
        "seq_prob_predicted_count",
        "seq_rank_fixed_topk",
        "seq_rank_expected_count",
        "seq_rank_history_count",
        "seq_rank_predicted_count",
    ],
    help="medication-set decoder used when decode search is disabled",
)
parser.add_argument("--decode_search", action="store_true", help="select the medication-set decoder on the validation set each epoch")
parser.add_argument("--decode_search_modes", type=str, default="threshold,fixed_topk,expected_count,history_count,predicted_count")
parser.add_argument("--log_decode_search_details", action="store_true", help="log every validation decoder candidate when decode search is enabled")
parser.add_argument("--prediction_output", type=str, default="", help="write split-specific y_true/y_pred/y_prob arrays for unified external evaluation")
parser.add_argument("--component_output", type=str, default="", help="write split-specific runtime decoder component arrays for downstream fusion")
parser.add_argument(
    "--fixed_component_export_no_metric_selection",
    action="store_true",
    help="export components with decode_configs()[0] without selecting a decoder from evaluation labels",
)
parser.add_argument("--decode_config_json", type=str, default="", help="explicit decoder config JSON file used instead of generated decode search grids")
parser.add_argument("--per_med_threshold_min", type=float, default=0.05)
parser.add_argument("--per_med_threshold_max", type=float, default=0.70)
parser.add_argument("--per_med_threshold_step", type=float, default=0.01)
parser.add_argument("--history_threshold_delta_min", type=float, default=-0.12, help="minimum history-threshold offset from the base threshold")
parser.add_argument("--history_threshold_delta_max", type=float, default=0.04, help="maximum history-threshold offset from the base threshold")
parser.add_argument("--history_threshold_delta_step", type=float, default=0.02)
parser.add_argument("--history_rerank_weight", type=float, default=0.1, help="fixed history-copy rerank weight when decode search is disabled")
parser.add_argument("--history_rerank_weight_min", type=float, default=0.05)
parser.add_argument("--history_rerank_weight_max", type=float, default=0.30)
parser.add_argument("--history_rerank_weight_step", type=float, default=0.05)
parser.add_argument("--history_retrieval_history_weight", type=float, default=0.1, help="fixed history weight for combined history/retrieval rerank")
parser.add_argument("--history_retrieval_history_weight_min", type=float, default=0.10)
parser.add_argument("--history_retrieval_history_weight_max", type=float, default=0.25)
parser.add_argument("--history_retrieval_history_weight_step", type=float, default=0.05)
parser.add_argument("--history_retrieval_retrieval_weight", type=float, default=0.1, help="fixed retrieval weight for combined history/retrieval rerank")
parser.add_argument("--history_retrieval_retrieval_weight_min", type=float, default=0.05)
parser.add_argument("--history_retrieval_retrieval_weight_max", type=float, default=0.25)
parser.add_argument("--history_retrieval_retrieval_weight_step", type=float, default=0.05)
parser.add_argument("--history_retrieval_seq_weight", type=float, default=0.1, help="fixed HI-DR sequence-rank weight for combined history/retrieval/sequence rerank")
parser.add_argument("--history_retrieval_seq_weight_min", type=float, default=0.0)
parser.add_argument("--history_retrieval_seq_weight_max", type=float, default=0.2)
parser.add_argument("--history_retrieval_seq_weight_step", type=float, default=0.05)
parser.add_argument("--history_retrieval_freq_weight", type=float, default=0.0, help="fixed training-frequency prior weight for history/retrieval/frequency rerank")
parser.add_argument("--history_retrieval_freq_weight_min", type=float, default=-0.10)
parser.add_argument("--history_retrieval_freq_weight_max", type=float, default=0.10)
parser.add_argument("--history_retrieval_freq_weight_step", type=float, default=0.02)
parser.add_argument("--history_retrieval_ehr_graph_weight", type=float, default=0.0, help="EHR graph-propagation score weight for history/retrieval/frequency graph rerank")
parser.add_argument("--history_retrieval_ehr_graph_weight_min", type=float, default=0.0)
parser.add_argument("--history_retrieval_ehr_graph_weight_max", type=float, default=0.10)
parser.add_argument("--history_retrieval_ehr_graph_weight_step", type=float, default=0.02)
parser.add_argument("--history_retrieval_ddi_graph_weight", type=float, default=0.0, help="DDI graph-propagation risk penalty for history/retrieval/frequency graph rerank")
parser.add_argument("--history_retrieval_ddi_graph_weight_min", type=float, default=0.0)
parser.add_argument("--history_retrieval_ddi_graph_weight_max", type=float, default=0.05)
parser.add_argument("--history_retrieval_ddi_graph_weight_step", type=float, default=0.01)
parser.add_argument(
    "--history_retrieval_graph_source",
    type=str,
    default="model",
    choices=["model", "history", "retrieval", "hybrid"],
    help="medication prior propagated through EHR/DDI graphs for graph rerank",
)
parser.add_argument("--history_retrieval_ddi_penalty", type=float, default=0.0, help="greedy DDI-risk penalty for history/retrieval/frequency rerank")
parser.add_argument("--history_retrieval_ddi_penalty_min", type=float, default=0.0)
parser.add_argument("--history_retrieval_ddi_penalty_max", type=float, default=0.05)
parser.add_argument("--history_retrieval_ddi_penalty_step", type=float, default=0.01)
parser.add_argument("--retrieval_rerank_weight", type=float, default=0.1, help="fixed similar-visit rerank weight when decode search is disabled")
parser.add_argument("--retrieval_rerank_weight_min", type=float, default=0.05)
parser.add_argument("--retrieval_rerank_weight_max", type=float, default=0.30)
parser.add_argument("--retrieval_rerank_weight_step", type=float, default=0.05)
parser.add_argument("--decode_topk", type=int, default=20)
parser.add_argument("--decode_topk_min", type=int, default=10)
parser.add_argument("--decode_topk_max", type=int, default=30)
parser.add_argument("--decode_topk_step", type=int, default=1)
parser.add_argument("--decode_count_ratio", type=float, default=1.0)
parser.add_argument("--decode_count_bias", type=float, default=0.0)
parser.add_argument("--decode_count_ratio_min", type=float, default=0.7)
parser.add_argument("--decode_count_ratio_max", type=float, default=1.3)
parser.add_argument("--decode_count_ratio_step", type=float, default=0.05)
parser.add_argument("--decode_count_bias_min", type=float, default=-2.0)
parser.add_argument("--decode_count_bias_max", type=float, default=2.0)
parser.add_argument("--decode_count_bias_step", type=float, default=1.0)
parser.add_argument(
    "--retrieval_count_modes",
    type=str,
    default="weighted_mean",
    help="comma-separated topK retrieval count estimators for *_retrieval_count decoders: weighted_mean,mean,top1,union",
)
parser.add_argument("--decode_min_count", type=int, default=1)
parser.add_argument("--decode_max_count", type=int, default=0, help="0 means medication vocabulary size")
parser.add_argument('--early_stop_patience', type=int, default=4, help='number of consecutive PRAUC declines before early stop; <=0 disables early stop')
# args = parser.parse_args(args=[])
args = parser.parse_args()

if args.action_transition_bce_weight > 0 and args.action_transition_weight == 0.0:
    args.action_transition_weight = args.action_transition_bce_weight
if args.action_continue_loss_weight != 1.0 and args.continue_aux_weight == 1.0:
    args.continue_aux_weight = args.action_continue_loss_weight
if args.action_add_loss_weight != 1.0 and args.add_aux_weight == 1.0:
    args.add_aux_weight = args.action_add_loss_weight
if args.action_remove_loss_weight != 1.0 and args.remove_aux_weight == 1.0:
    args.remove_aux_weight = args.action_remove_loss_weight
if args.action_transition_weight > 0 and args.action_transition_mode == "none":
    args.action_transition_mode = "heads"
if args.action_ddi_loss_weight > 0 and args.action_transition_mode == "none":
    args.action_transition_mode = "heads"
fusion_v2_objective_enabled = (
    args.fusion_soft_jaccard_weight > 0
    or args.fusion_count_anchor_weight > 0
    or args.fusion_ddi_soft_weight > 0
)
fusion_objective_enabled = (
    args.action_fusion_loss_weight > 0
    or fusion_v2_objective_enabled
    or args.routed_fusion_entropy_weight > 0
    or args.routed_residual_entropy_weight > 0
    or args.routed_residual_l2_weight > 0
    or args.routed_residual_add_penalty_weight > 0
    or args.routed_residual_load_balance_weight > 0
)
if (args.action_fusion_trainable or fusion_objective_enabled or args.use_trainable_action_fusion) and args.action_transition_mode == "none":
    args.action_transition_mode = "heads"
if (fusion_objective_enabled or args.use_trainable_action_fusion) and not args.action_fusion_trainable:
    args.action_fusion_trainable = True

if args.lr <= 0:
    raise ValueError("--lr must be positive")
if args.weight_decay < 0:
    raise ValueError("--weight_decay must be non-negative")
if args.action_transition_weight < 0 or args.action_transition_bce_weight < 0:
    raise ValueError("--action_transition_weight must be non-negative")
if args.continue_aux_weight < 0 or args.add_aux_weight < 0 or args.remove_aux_weight < 0:
    raise ValueError("--continue_aux_weight, --add_aux_weight, and --remove_aux_weight must be non-negative")
if args.add_pos_weight <= 0:
    raise ValueError("--add_pos_weight must be positive")
if args.add_focal_gamma < 0:
    raise ValueError("--add_focal_gamma must be non-negative")
if args.action_ddi_loss_weight < 0:
    raise ValueError("--action_ddi_loss_weight must be non-negative")
if args.add_ddi_loss_weight < 0:
    raise ValueError("--add_ddi_loss_weight must be non-negative")
if args.ddi_risk_threshold < 0:
    raise ValueError("--ddi_risk_threshold must be non-negative")
if args.action_continue_loss_weight < 0 or args.action_add_loss_weight < 0 or args.action_remove_loss_weight < 0:
    raise ValueError("--action_*_loss_weight values must be non-negative")
if args.action_transition_hidden_dim < 0:
    raise ValueError("--action_transition_hidden_dim must be non-negative")
if not 0.0 <= args.action_transition_dropout <= 1.0:
    raise ValueError("--action_transition_dropout must be in [0, 1]")
if not 0.0 <= args.action_transition_threshold <= 1.0:
    raise ValueError("--action_transition_threshold must be in [0, 1]")
if args.routed_fusion_hidden_dim <= 0:
    raise ValueError("--routed_fusion_hidden_dim must be positive")
if args.routed_fusion_drug_emb_dim <= 0:
    raise ValueError("--routed_fusion_drug_emb_dim must be positive")
if not 0.0 <= args.routed_fusion_dropout <= 1.0:
    raise ValueError("--routed_fusion_dropout must be in [0, 1]")
if args.routed_fusion_entropy_weight < 0:
    raise ValueError("--routed_fusion_entropy_weight must be non-negative")
if args.routed_residual_scale < 0:
    raise ValueError("--routed_residual_scale must be non-negative")
if not 0.0 <= args.routed_residual_max_gate <= 1.0:
    raise ValueError("--routed_residual_max_gate must be in [0, 1]")
for name, value in (
    ("--routed_residual_l2_weight", args.routed_residual_l2_weight),
    ("--routed_residual_add_penalty_weight", args.routed_residual_add_penalty_weight),
    ("--routed_residual_entropy_weight", args.routed_residual_entropy_weight),
    ("--routed_residual_load_balance_weight", args.routed_residual_load_balance_weight),
):
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
if args.action_fusion_loss_weight < 0:
    raise ValueError("--action_fusion_loss_weight must be non-negative")
if (
    args.fusion_bce_weight < 0
    or args.fusion_soft_jaccard_weight < 0
    or args.fusion_count_anchor_weight < 0
    or args.fusion_ddi_soft_weight < 0
):
    raise ValueError("--fusion_*_weight values must be non-negative")
if args.use_trainable_action_fusion and args.action_decode_mode == "direct_action":
    raise ValueError("--use_trainable_action_fusion is incompatible with --action_decode_mode direct_action")
if not 0.0 <= args.action_decode_threshold <= 1.0:
    raise ValueError("--action_decode_threshold must be in [0, 1]")
if args.action_decode_continue_weight < 0 or args.action_decode_add_weight < 0 or args.action_decode_remove_weight < 0:
    raise ValueError("--action_decode_*_weight values must be non-negative")
if args.count_aux_weight < 0:
    raise ValueError("--count_aux_weight must be non-negative")
if (
    args.total_count_weight < 0
    or args.continue_count_weight < 0
    or args.add_count_weight < 0
    or args.remove_count_weight < 0
):
    raise ValueError("--*_count_weight values must be non-negative")
if args.treatment_episode_action_aux_weight < 0:
    raise ValueError("--treatment_episode_action_aux_weight must be non-negative")
if args.treatment_episode_action_bias_weight < 0:
    raise ValueError("--treatment_episode_action_bias_weight must be non-negative")
if args.treatment_episode_fused_action_loss_weight < 0:
    raise ValueError("--treatment_episode_fused_action_loss_weight must be non-negative")
if args.treatment_episode_candidate_gain_weight < 0:
    raise ValueError("--treatment_episode_candidate_gain_weight must be non-negative")
if args.treatment_episode_candidate_gain_temperature <= 0:
    raise ValueError("--treatment_episode_candidate_gain_temperature must be positive")
if args.treatment_episode_candidate_gain_value_weight < 0:
    raise ValueError("--treatment_episode_candidate_gain_value_weight must be non-negative")
if args.treatment_episode_candidate_gain_pair_weight < 0:
    raise ValueError("--treatment_episode_candidate_gain_pair_weight must be non-negative")
if args.treatment_episode_candidate_gain_pair_margin < 0:
    raise ValueError("--treatment_episode_candidate_gain_pair_margin must be non-negative")
if args.treatment_episode_candidate_gain_min_pair_gap < 0:
    raise ValueError("--treatment_episode_candidate_gain_min_pair_gap must be non-negative")
if (
    args.treatment_episode_action_continue_weight < 0
    or args.treatment_episode_action_add_weight < 0
    or args.treatment_episode_action_remove_weight < 0
):
    raise ValueError("--treatment_episode_action_*_weight values must be non-negative")
if not 0.0 <= args.eval_threshold <= 1.0:
    raise ValueError("--eval_threshold must be in [0, 1]")
if args.threshold_search or args.decode_search:
    if args.threshold_step <= 0:
        raise ValueError("--threshold_step must be positive")
    if not 0.0 <= args.threshold_min <= args.threshold_max <= 1.0:
        raise ValueError("--threshold_min and --threshold_max must satisfy 0 <= min <= max <= 1")
if args.per_med_threshold_step <= 0:
    raise ValueError("--per_med_threshold_step must be positive")
if not 0.0 <= args.per_med_threshold_min <= args.per_med_threshold_max <= 1.0:
    raise ValueError("--per_med_threshold_min and --per_med_threshold_max must satisfy 0 <= min <= max <= 1")
if args.history_threshold_delta_step <= 0:
    raise ValueError("--history_threshold_delta_step must be positive")
if args.history_rerank_weight < 0 or args.retrieval_rerank_weight < 0:
    raise ValueError("--history_rerank_weight and --retrieval_rerank_weight must be non-negative")
if args.history_rerank_weight_step <= 0 or args.retrieval_rerank_weight_step <= 0:
    raise ValueError("--history_rerank_weight_step and --retrieval_rerank_weight_step must be positive")
if args.history_rerank_weight_min > args.history_rerank_weight_max:
    raise ValueError("--history_rerank_weight_min must be <= --history_rerank_weight_max")
if args.retrieval_rerank_weight_min > args.retrieval_rerank_weight_max:
    raise ValueError("--retrieval_rerank_weight_min must be <= --retrieval_rerank_weight_max")
if args.decode_topk <= 0:
    raise ValueError("--decode_topk must be positive")
if args.decode_topk_step <= 0:
    raise ValueError("--decode_topk_step must be positive")
if args.decode_topk_min <= 0 or args.decode_topk_min > args.decode_topk_max:
    raise ValueError("--decode_topk_min and --decode_topk_max must satisfy 0 < min <= max")
if args.decode_count_ratio_step <= 0 or args.decode_count_bias_step <= 0:
    raise ValueError("--decode_count_ratio_step and --decode_count_bias_step must be positive")
if args.decode_count_ratio_min > args.decode_count_ratio_max:
    raise ValueError("--decode_count_ratio_min must be <= --decode_count_ratio_max")
if args.decode_count_bias_min > args.decode_count_bias_max:
    raise ValueError("--decode_count_bias_min must be <= --decode_count_bias_max")
RETRIEVAL_COUNT_MODE_CHOICES = {"weighted_mean", "mean", "top1", "union"}
requested_retrieval_count_modes = [
    value.strip()
    for value in args.retrieval_count_modes.split(",")
    if value.strip()
]
if not requested_retrieval_count_modes:
    raise ValueError("--retrieval_count_modes must include at least one mode")
unknown_retrieval_count_modes = sorted(set(requested_retrieval_count_modes) - RETRIEVAL_COUNT_MODE_CHOICES)
if unknown_retrieval_count_modes:
    raise ValueError(
        "--retrieval_count_modes contains unsupported values: {}".format(
            ",".join(unknown_retrieval_count_modes)
        )
    )
if args.decode_min_count < 0:
    raise ValueError("--decode_min_count must be non-negative")
if args.decode_max_count < 0:
    raise ValueError("--decode_max_count must be non-negative")
if args.seq_decoder_max_len <= 0:
    raise ValueError("--seq_decoder_max_len must be positive")
if args.seq_copy_mode != "none" and args.seq_decoder_mode not in {"aux", "hidr_memory", "hidr_transformer"}:
    raise ValueError("--seq_copy_mode requires --seq_decoder_mode aux, hidr_memory, or hidr_transformer")
if args.seq_retrieval_copy_mode != "none" and args.seq_decoder_mode not in {"aux", "hidr_memory", "hidr_transformer"}:
    raise ValueError("--seq_retrieval_copy_mode requires --seq_decoder_mode aux, hidr_memory, or hidr_transformer")
if args.seq_decoder_mode in {"hidr_memory", "hidr_transformer"}:
    if args.seq_retrieval_copy_mode != "none":
        raise ValueError("--seq_retrieval_copy_mode is only used by the aux sequence decoder; HI-DR sequence decoders have their own retrieved-visit copy")
    if args.seq_hidr_gumbel_tau <= 0:
        raise ValueError("--seq_hidr_gumbel_tau must be positive")
    if args.seq_hidr_att_tau <= 0:
        raise ValueError("--seq_hidr_att_tau must be positive")
if args.seq_decoder_mode == "hidr_transformer":
    if args.seq_transformer_hidden_dim < 0:
        raise ValueError("--seq_transformer_hidden_dim must be non-negative")
    if args.seq_transformer_nhead <= 0:
        raise ValueError("--seq_transformer_nhead must be positive")
    if args.seq_transformer_layers <= 0:
        raise ValueError("--seq_transformer_layers must be positive")
    effective_seq_hidden = args.seq_transformer_hidden_dim or args.dim
    if effective_seq_hidden % args.seq_transformer_nhead != 0:
        raise ValueError("--seq_transformer_hidden_dim or --dim must be divisible by --seq_transformer_nhead")
if args.seq_copy_recency_decay <= 0:
    raise ValueError("--seq_copy_recency_decay must be positive")
if args.seq_decoder_end_bias_step <= 0:
    raise ValueError("--seq_decoder_end_bias_step must be positive")
if args.seq_decoder_end_bias_min > args.seq_decoder_end_bias_max:
    raise ValueError("--seq_decoder_end_bias_min must be <= --seq_decoder_end_bias_max")
if args.seq_decoder_beam_size <= 0:
    raise ValueError("--seq_decoder_beam_size must be positive")
if args.history_retrieval_history_weight_step <= 0 or args.history_retrieval_retrieval_weight_step <= 0:
    raise ValueError("--history_retrieval_*_weight_step must be positive")
if args.history_retrieval_freq_weight_step <= 0:
    raise ValueError("--history_retrieval_freq_weight_step must be positive")
if args.history_retrieval_freq_weight_min > args.history_retrieval_freq_weight_max:
    raise ValueError("--history_retrieval_freq_weight_min must be <= --history_retrieval_freq_weight_max")
if args.history_retrieval_ehr_graph_weight < 0 or args.history_retrieval_ddi_graph_weight < 0:
    raise ValueError("--history_retrieval_*_graph_weight must be non-negative")
if args.history_retrieval_ehr_graph_weight_step <= 0 or args.history_retrieval_ddi_graph_weight_step <= 0:
    raise ValueError("--history_retrieval_*_graph_weight_step must be positive")
if args.history_retrieval_ehr_graph_weight_min > args.history_retrieval_ehr_graph_weight_max:
    raise ValueError("--history_retrieval_ehr_graph_weight_min must be <= --history_retrieval_ehr_graph_weight_max")
if args.history_retrieval_ddi_graph_weight_min > args.history_retrieval_ddi_graph_weight_max:
    raise ValueError("--history_retrieval_ddi_graph_weight_min must be <= --history_retrieval_ddi_graph_weight_max")
if args.history_retrieval_ddi_penalty < 0:
    raise ValueError("--history_retrieval_ddi_penalty must be non-negative")
if args.history_retrieval_ddi_penalty_step <= 0:
    raise ValueError("--history_retrieval_ddi_penalty_step must be positive")
if args.history_retrieval_ddi_penalty_min > args.history_retrieval_ddi_penalty_max:
    raise ValueError("--history_retrieval_ddi_penalty_min must be <= --history_retrieval_ddi_penalty_max")
if args.bce_weight < 0:
    raise ValueError("--bce_weight must be non-negative")
if args.soft_jaccard_weight < 0 or args.soft_dice_weight < 0:
    raise ValueError("--soft_jaccard_weight and --soft_dice_weight must be non-negative")
if args.asymmetric_bce_weight < 0:
    raise ValueError("--asymmetric_bce_weight must be non-negative")
if args.asymmetric_bce_gamma_pos < 0 or args.asymmetric_bce_gamma_neg < 0:
    raise ValueError("--asymmetric_bce_gamma_* must be non-negative")
if not 0.0 <= args.asymmetric_bce_neg_clip < 1.0:
    raise ValueError("--asymmetric_bce_neg_clip must be in [0, 1)")
if args.pairwise_rank_weight < 0:
    raise ValueError("--pairwise_rank_weight must be non-negative")
if args.pairwise_rank_negatives <= 0:
    raise ValueError("--pairwise_rank_negatives must be positive")
if args.listwise_rank_weight < 0:
    raise ValueError("--listwise_rank_weight must be non-negative")
if args.listwise_rank_negatives <= 0:
    raise ValueError("--listwise_rank_negatives must be positive")
if args.train_graph_rank_weight < 0:
    raise ValueError("--train_graph_rank_weight must be non-negative")
if args.train_graph_rank_negatives <= 0:
    raise ValueError("--train_graph_rank_negatives must be positive")
if args.train_graph_rank_margin < 0:
    raise ValueError("--train_graph_rank_margin must be non-negative")
if args.train_graph_boundary_rank_weight < 0:
    raise ValueError("--train_graph_boundary_rank_weight must be non-negative")
if args.train_graph_boundary_negatives <= 0:
    raise ValueError("--train_graph_boundary_negatives must be positive")
if args.train_graph_boundary_margin < 0:
    raise ValueError("--train_graph_boundary_margin must be non-negative")
if args.action_boundary_rank_weight < 0:
    raise ValueError("--action_boundary_rank_weight must be non-negative")
if args.action_boundary_continue_weight < 0 or args.action_boundary_add_weight < 0 or args.action_boundary_global_weight < 0:
    raise ValueError("--action_boundary_*_weight must be non-negative")
if args.action_boundary_negatives <= 0:
    raise ValueError("--action_boundary_negatives must be positive")
if args.action_boundary_margin < 0:
    raise ValueError("--action_boundary_margin must be non-negative")
if args.train_graph_count_weight < 0:
    raise ValueError("--train_graph_count_weight must be non-negative")
if args.retrieval_query_rank_weight < 0:
    raise ValueError("--retrieval_query_rank_weight must be non-negative")
if args.retrieval_query_rank_negatives <= 0:
    raise ValueError("--retrieval_query_rank_negatives must be positive")
if args.retrieval_query_rank_margin < 0:
    raise ValueError("--retrieval_query_rank_margin must be non-negative")
if args.retrieval_query_rank_min_delta < 0:
    raise ValueError("--retrieval_query_rank_min_delta must be non-negative")
if args.soft_ddi_weight < 0:
    raise ValueError("--soft_ddi_weight must be non-negative")
if args.soft_ddi_power <= 0:
    raise ValueError("--soft_ddi_power must be positive")
if args.patient_graph_ehr_scale < 0 or args.patient_graph_ddi_scale < 0:
    raise ValueError("--patient_graph_*_scale must be non-negative")
if args.patient_graph_dropout < 0:
    raise ValueError("--patient_graph_dropout must be non-negative")
if args.graph_logit_bias_weight < 0:
    raise ValueError("--graph_logit_bias_weight must be non-negative")
if args.graph_edge_bias_weight < 0 or args.graph_edge_ddi_weight < 0:
    raise ValueError("--graph_edge_*_weight must be non-negative")
if args.graph_edge_bias_weight > 0 and args.patient_graph_mode != "edge_bias":
    raise ValueError("--graph_edge_bias_weight > 0 requires --patient_graph_mode edge_bias")
if args.graph_med_ehr_weight < 0 or args.graph_med_ddi_weight < 0:
    raise ValueError("--graph_med_*_weight must be non-negative")
if args.graph_med_gate_hidden <= 0:
    raise ValueError("--graph_med_gate_hidden must be positive")
if not 0.0 <= args.graph_med_dropout <= 1.0:
    raise ValueError("--graph_med_dropout must be in [0, 1]")
if (
    args.bce_weight == 0
    and args.soft_jaccard_weight == 0
    and args.soft_dice_weight == 0
    and args.asymmetric_bce_weight == 0
    and args.pairwise_rank_weight == 0
    and args.listwise_rank_weight == 0
    and args.train_graph_rank_weight == 0
    and args.train_graph_boundary_rank_weight == 0
    and args.action_boundary_rank_weight == 0
    and args.train_graph_count_weight == 0
    and args.train_graph_score_distill_weight == 0
    and args.soft_ddi_weight == 0
    and not fusion_objective_enabled
    and not (args.med_count_mode == "aux" and args.med_count_aux_weight > 0)
    and not (args.action_transition_mode == "heads" and args.action_transition_weight > 0)
    and not (args.action_transition_mode == "heads" and args.action_ddi_loss_weight > 0)
    and args.seq_decoder_mode == "none"
):
    raise ValueError("at least one target prediction loss must be enabled")
if args.prefix_aux_weight < 0:
    raise ValueError("--prefix_aux_weight must be non-negative")
if args.prefix_seq_decoder_weight < 0:
    raise ValueError("--prefix_seq_decoder_weight must be non-negative")
if args.prefix_seq_decoder_weight > 0 and args.seq_decoder_mode == "none":
    raise ValueError("--prefix_seq_decoder_weight requires a sequence decoder")
if args.prefix_metric_mode == "paper_prefix" and not args.paper_protocol:
    raise ValueError("--prefix_metric_mode paper_prefix requires --paper_protocol")
if args.validation_only_no_test:
    if args.test > 0:
        raise ValueError("--validation_only_no_test cannot be combined with --test")
    if args.eval_split == "test":
        raise ValueError("--validation_only_no_test cannot be combined with --eval_split test")
    if args.train_include_test_split:
        raise ValueError("--validation_only_no_test cannot be combined with --train_include_test_split")
if not 0.0 < args.history_prior_decay <= 1.0:
    raise ValueError("--history_prior_decay must be in (0, 1]")
if not 0.0 <= args.retrieval_copy_weight <= 1.0:
    raise ValueError("--retrieval_copy_weight must be in [0, 1]")
if args.retrieval_copy_weight > 0 and args.retrieval_copy_mode == "none":
    raise ValueError("--retrieval_copy_weight > 0 requires --retrieval_copy_mode fixed_prob or prob_mix")
if args.retrieval_copy_mode != "none" and args.retrieval_copy_weight <= 0:
    raise ValueError("--retrieval_copy_mode requires --retrieval_copy_weight > 0")
if args.retrieval_readout_weight < 0:
    raise ValueError("--retrieval_readout_weight must be non-negative")
if args.retrieval_readout_weight > 0 and args.retrieval_readout_mode == "none":
    raise ValueError("--retrieval_readout_weight > 0 requires --retrieval_readout_mode logit_bias")
if args.retrieval_readout_mode != "none" and args.retrieval_readout_weight <= 0:
    raise ValueError("--retrieval_readout_mode requires --retrieval_readout_weight > 0")
if args.freeze_base_for_retrieval_readout and args.retrieval_readout_mode == "none":
    raise ValueError("--freeze_base_for_retrieval_readout requires --retrieval_readout_mode logit_bias")
if args.treatment_episode_adapter_mode in {"logit_bias", "state", "both"}:
    if args.treatment_episode_adapter_weight <= 0:
        raise ValueError("--treatment_episode_adapter_weight must be positive when treatment episode adapter is enabled")
if args.treatment_episode_adapter_mode in {"none", "action"} and args.treatment_episode_adapter_weight != 0:
    raise ValueError("--treatment_episode_adapter_weight requires --treatment_episode_adapter_mode")
if args.treatment_episode_adapter_dropout < 0:
    raise ValueError("--treatment_episode_adapter_dropout must be non-negative")
if args.freeze_base_for_treatment_episode_adapter and args.treatment_episode_adapter_mode == "none":
    raise ValueError("--freeze_base_for_treatment_episode_adapter requires --treatment_episode_adapter_mode")
if args.treatment_episode_action_aux_weight > 0 and args.treatment_episode_adapter_mode == "none":
    raise ValueError("--treatment_episode_action_aux_weight requires --treatment_episode_adapter_mode")
if args.treatment_episode_action_bias_weight > 0 and args.treatment_episode_adapter_mode == "none":
    raise ValueError("--treatment_episode_action_bias_weight requires --treatment_episode_adapter_mode")
if args.treatment_episode_fused_action_loss_weight > 0 and args.treatment_episode_adapter_mode == "none":
    raise ValueError("--treatment_episode_fused_action_loss_weight requires --treatment_episode_adapter_mode")
if args.treatment_episode_candidate_gain_weight > 0 and args.treatment_episode_adapter_mode == "none":
    raise ValueError("--treatment_episode_candidate_gain_weight requires --treatment_episode_adapter_mode")
if args.treatment_episode_bias_scale_mode != "none" and args.treatment_episode_adapter_mode == "none":
    raise ValueError("--treatment_episode_bias_scale_mode requires --treatment_episode_adapter_mode")
if args.freeze_base_for_edge_bias and args.patient_graph_mode != "edge_bias":
    raise ValueError("--freeze_base_for_edge_bias requires --patient_graph_mode edge_bias")
if args.freeze_base_for_edge_bias and args.graph_edge_bias_weight <= 0:
    raise ValueError("--freeze_base_for_edge_bias requires --graph_edge_bias_weight > 0")
if args.patient_med_scorer_rank <= 0:
    raise ValueError("--patient_med_scorer_rank must be positive")
if args.patient_med_scorer_weight < 0:
    raise ValueError("--patient_med_scorer_weight must be non-negative")
if not 0.0 <= args.patient_med_scorer_dropout <= 1.0:
    raise ValueError("--patient_med_scorer_dropout must be in [0, 1]")
if args.patient_med_scorer_weight > 0 and args.patient_med_scorer_mode == "none":
    raise ValueError("--patient_med_scorer_weight > 0 requires --patient_med_scorer_mode lowrank_bias")
if args.freeze_base_for_patient_med_scorer and args.patient_med_scorer_mode == "none":
    raise ValueError("--freeze_base_for_patient_med_scorer requires --patient_med_scorer_mode lowrank_bias")
if args.freeze_base_for_patient_med_scorer and args.patient_med_scorer_weight <= 0:
    raise ValueError("--freeze_base_for_patient_med_scorer requires --patient_med_scorer_weight > 0")
if args.main_med_scorer_hidden_dim <= 0:
    raise ValueError("--main_med_scorer_hidden_dim must be positive")
if args.main_med_scorer_rank <= 0:
    raise ValueError("--main_med_scorer_rank must be positive")
if not 0.0 <= args.main_med_scorer_dropout <= 1.0:
    raise ValueError("--main_med_scorer_dropout must be in [0, 1]")
if args.freeze_base_for_main_med_scorer and args.main_med_scorer_mode == "none":
    raise ValueError("--freeze_base_for_main_med_scorer requires --main_med_scorer_mode feature_delta")
if args.main_med_scorer_delta_l2_weight < 0:
    raise ValueError("--main_med_scorer_delta_l2_weight must be non-negative")
if args.main_med_scorer_count_kl_weight < 0:
    raise ValueError("--main_med_scorer_count_kl_weight must be non-negative")
if not 0.0 <= args.current_med_attention_dropout <= 1.0:
    raise ValueError("--current_med_attention_dropout must be in [0, 1]")
if args.freeze_base_for_current_med_attention and args.current_med_attention_mode == "none":
    raise ValueError("--freeze_base_for_current_med_attention requires --current_med_attention_mode history_cross_attention")
if args.own_history_context_weight < 0:
    raise ValueError("--own_history_context_weight must be non-negative")
if not 0.0 <= args.own_history_context_dropout <= 1.0:
    raise ValueError("--own_history_context_dropout must be in [0, 1]")
if not 0.0 < args.own_history_context_recency_decay <= 1.0:
    raise ValueError("--own_history_context_recency_decay must be in (0, 1]")
if args.own_history_context_weight > 0 and args.own_history_context_mode == "none":
    raise ValueError("--own_history_context_weight > 0 requires --own_history_context_mode visit_attention")
if args.own_history_context_mode != "none" and args.own_history_context_weight <= 0:
    raise ValueError("--own_history_context_mode requires --own_history_context_weight > 0")
if args.freeze_base_for_own_history_context and args.own_history_context_mode == "none":
    raise ValueError("--freeze_base_for_own_history_context requires --own_history_context_mode visit_attention")
if args.train_graph_score_distill_weight < 0:
    raise ValueError("--train_graph_score_distill_weight must be non-negative")
if args.train_graph_score_distill_temperature <= 0:
    raise ValueError("--train_graph_score_distill_temperature must be positive")
if args.train_graph_score_distill_topk < 0:
    raise ValueError("--train_graph_score_distill_topk must be non-negative")
if args.freeze_base_for_graph_score_distill and args.train_graph_score_distill_weight <= 0:
    raise ValueError("--freeze_base_for_graph_score_distill requires --train_graph_score_distill_weight > 0")
for teacher_weight_name in (
    "teacher_rank_distill_weight",
    "teacher_count_distill_weight",
    "teacher_parameter_anchor_weight",
):
    if getattr(args, teacher_weight_name) < 0:
        raise ValueError(f"--{teacher_weight_name} must be non-negative")
teacher_anchor_enabled = any(
    getattr(args, name) > 0
    for name in (
        "teacher_rank_distill_weight",
        "teacher_count_distill_weight",
        "teacher_parameter_anchor_weight",
    )
)
if teacher_anchor_enabled and not args.teacher_anchor_components:
    raise ValueError("teacher anchor weights require --teacher_anchor_components")
if args.teacher_anchor_components and "test" in {part.lower() for part in Path(args.teacher_anchor_components).parts}:
    raise ValueError("teacher anchor components must not reference a test path")
if args.freeze_base_for_posthoc_calibration and args.posthoc_calibration_mode == "none":
    raise ValueError("--freeze_base_for_posthoc_calibration requires --posthoc_calibration_mode per_med_linear or per_med_graph")
if args.freeze_base_for_med_count and args.med_count_mode != "aux":
    raise ValueError("--freeze_base_for_med_count requires --med_count_mode aux")
if args.history_copy_weight < 0:
    raise ValueError("--history_copy_weight must be non-negative")
if not 0.0 < args.history_copy_decay <= 1.0:
    raise ValueError("--history_copy_decay must be in (0, 1]")
if args.history_copy_weight > 0 and args.history_copy_mode == "none":
    raise ValueError("--history_copy_weight > 0 requires --history_copy_mode gated")
if args.decode_mode in {
    "seq_greedy",
    "seq_hidr",
    "seq_beam",
    "seq_prob_fixed_topk",
    "seq_prob_expected_count",
    "seq_prob_history_count",
    "seq_prob_predicted_count",
    "seq_rank_fixed_topk",
    "seq_rank_expected_count",
    "seq_rank_history_count",
    "seq_rank_predicted_count",
    "history_retrieval_seq_rank_fixed_topk",
    "history_retrieval_seq_rank_threshold_count",
    "history_retrieval_seq_rank_expected_count",
    "history_retrieval_seq_rank_history_count",
    "history_retrieval_seq_rank_predicted_count",
} and args.seq_decoder_mode == "none":
    raise ValueError(f"--decode_mode {args.decode_mode} requires a sequence decoder")
if (
    args.retrieval_context_source == "static"
    and (
        args.retrieval_context_mode != "none"
        or args.retrieval_copy_mode == "prob_mix"
        or args.retrieval_readout_mode != "none"
        or args.retrieval_query_rank_weight > 0
        or args.seq_retrieval_copy_mode != "none"
        or args.seq_decoder_mode in {"hidr_memory", "hidr_transformer"}
    )
):
    paper_m3_static_context_requested = (
        args.retrieval_mode == "paper_m3"
        or (
            args.insert_retrieved_visits
            and args.insert_visit_mode == "paper_m3"
        )
    )
    if (
        not paper_m3_static_context_requested
        and args.retrieval_mode not in {"dxproc", "clinical", "dxproc_clinical", "history_med", "treatment_episode"}
    ):
        raise ValueError("--retrieval_context_source static requires --retrieval_mode dxproc, clinical, dxproc_clinical, history_med, treatment_episode, or --insert_visit_mode paper_m3")
if args.external_memory_mode != "none":
    if args.external_memory_weight <= 0:
        raise ValueError("--external_memory_weight must be positive when external memory is enabled")
    if args.external_memory_num <= 0:
        raise ValueError("--external_memory_num must be positive")
    if args.external_memory_hidden_size < 0:
        raise ValueError("--external_memory_hidden_size must be non-negative")
else:
    if args.external_memory_weight != 0:
        raise ValueError("--external_memory_weight requires --external_memory_mode state, logit, or both")
if args.local_memory_mode != "none":
    if args.local_memory_weight <= 0:
        raise ValueError("--local_memory_weight must be positive when local memory is enabled")
else:
    if args.local_memory_weight != 0:
        raise ValueError("--local_memory_weight requires --local_memory_mode state")
if "learned" in args.insert_visit_mode and args.insert_visit_topk > args.retrieval_topk:
    raise ValueError("--insert_visit_topk must be <= --retrieval_topk for learned inserted-visit modes")
if args.paper_m3_topk < 0:
    raise ValueError("--paper_m3_topk must be non-negative")
if (args.insert_visit_mode == "paper_m3" or args.retrieval_mode == "paper_m3") and not args.paper_m3_checkpoint:
    raise ValueError("--insert_visit_mode paper_m3 and --retrieval_mode paper_m3 require --paper_m3_checkpoint")
if args.paper_m3_rbf_gamma <= 0:
    raise ValueError("--paper_m3_rbf_gamma must be positive")
if args.paper_m3_query_transform_ridge < 0:
    raise ValueError("--paper_m3_query_transform_ridge must be non-negative")
if args.paper_m3_label_prior_weight < 0:
    raise ValueError("--paper_m3_label_prior_weight must be non-negative")
if args.paper_m3_label_prior_weight > 0 and args.paper_m3_query_transform == "none":
    raise ValueError("--paper_m3_label_prior_weight requires --paper_m3_query_transform")
if args.paper_m3_clinical_weight < 0:
    raise ValueError("--paper_m3_clinical_weight must be non-negative")
if not 0.0 <= args.paper_m3_clinical_alpha <= 1.0:
    raise ValueError("--paper_m3_clinical_alpha must be in [0, 1]")
if args.paper_m3_dxproc_weight < 0:
    raise ValueError("--paper_m3_dxproc_weight must be non-negative")
if args.paper_m3_history_overlap_weight < 0:
    raise ValueError("--paper_m3_history_overlap_weight must be non-negative")
for treatment_episode_weight_name in (
    "treatment_episode_state_weight",
    "treatment_episode_dxproc_weight",
    "treatment_episode_clinical_weight",
    "treatment_episode_history_weight",
    "treatment_episode_action_weight",
    "treatment_episode_outcome_weight",
    "treatment_episode_safety_weight",
):
    if getattr(args, treatment_episode_weight_name) < 0:
        raise ValueError(f"--{treatment_episode_weight_name} must be non-negative")
if args.treatment_episode_min_outcome_mask < 0:
    raise ValueError("--treatment_episode_min_outcome_mask must be non-negative")
if args.retrieval_mode == "official_topidx" and not args.official_topidx_dir:
    raise ValueError("--retrieval_mode official_topidx requires --official_topidx_dir")
if (args.prefix_aux_weight > 0 or args.prefix_seq_decoder_weight > 0) and (
    args.retrieval_context_mode != "none"
    or args.retrieval_copy_mode == "prob_mix"
    or args.retrieval_readout_mode != "none"
):
    raise ValueError("--prefix auxiliary losses currently support learned retrieval-free paths")
if args.prefix_metric_mode == "paper_prefix" and (
    args.retrieval_context_mode != "none"
    or args.retrieval_weight > 0
    or args.retrieval_copy_mode != "none"
    or args.retrieval_readout_mode != "none"
):
    raise ValueError("--prefix_metric_mode paper_prefix currently supports retrieval-free paths")


def runtime_decode_config_from_json(config):
    if not isinstance(config, dict):
        raise ValueError("decode config must be an object")
    runtime_config = dict(config)
    if isinstance(runtime_config.get("thresholds"), list):
        runtime_config["thresholds"] = torch.tensor(runtime_config["thresholds"], dtype=torch.float32)
    if isinstance(runtime_config.get("base_decoder"), dict):
        runtime_config["base_decoder"] = runtime_decode_config_from_json(runtime_config["base_decoder"])
    if isinstance(runtime_config.get("base_config"), dict):
        runtime_config["base_config"] = runtime_decode_config_from_json(runtime_config["base_config"])
    return runtime_config


def runtime_decode_config_for_current_modes(decode_config):
    runtime_config = decode_config
    if args.action_decode_mode == "direct_action" and runtime_config["mode"] != "action_direct":
        runtime_config = {
            "mode": "action_direct",
            "threshold": float(args.action_decode_threshold),
            "base_config": decode_config_for_json(runtime_config),
        }
    if args.count_decode_mode != "none" and runtime_config["mode"] not in {"count_total_topk", "count_action_split_topk"}:
        runtime_config = {
            "mode": "count_total_topk" if args.count_decode_mode == "total_topk" else "count_action_split_topk",
            "base_decoder": decode_config_for_json(runtime_config),
        }
    return runtime_config


def load_decode_config_json(path):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        if "configs" in payload:
            payload = payload["configs"]
        elif "decode_configs" in payload:
            payload = payload["decode_configs"]
    if not isinstance(payload, list):
        raise ValueError("--decode_config_json must contain a list or an object with a configs list")

    configs = []
    for index, config in enumerate(payload):
        if not isinstance(config, dict):
            raise ValueError(f"decode config #{index} must be an object")
        configs.append(runtime_decode_config_from_json(config))
    if not configs:
        raise ValueError("--decode_config_json did not contain any decoder configs")
    return configs


def mode_is_retrieval_rank(mode):
    return (
        mode.startswith("retrieval_rank_")
        or mode.startswith("history_retrieval_rank_")
        or mode.startswith("history_retrieval_freq_rank_")
        or mode.startswith("history_retrieval_freq_graph_rank_")
        or mode.startswith("history_retrieval_freq_ddi_rank_")
        or mode.startswith("history_retrieval_seq_rank_")
    )


def mode_uses_retrieval_count(mode):
    if mode == "retrieval_count":
        return True
    if mode.startswith("history_retrieval_freq_graph_rank_"):
        return mode.removeprefix("history_retrieval_freq_graph_rank_") == "retrieval_count"
    return False


EPISODE_ACTION_COUNT_SOURCE_NAMES = {
    "episode_add_count",
    "episode_continue_count",
    "episode_remove_count",
    "episode_action_count",
    "episode_previous_add_count",
    "episode_net_count",
    "episode_mean_action_count",
    "episode_mean_previous_add_count",
    "episode_mean_net_count",
    "episode_top1_action_count",
    "episode_top1_previous_add_count",
    "episode_top1_net_count",
}


def config_uses_episode_action_count_source(config):
    if str(config.get("mode", "")) != "history_retrieval_freq_graph_rank_count_pair_mix":
        return False
    return any(str(config.get(key, "")) in EPISODE_ACTION_COUNT_SOURCE_NAMES for key in ("left", "right"))


def requested_episode_action_count_sources():
    if args.decode_config_json and any(config_uses_episode_action_count_source(config) for config in load_decode_config_json(args.decode_config_json)):
        return True
    return False


def seq_decoder_enabled():
    return args.seq_decoder_mode != "none"


def decode_uses_retrieval_rank():
    if args.decode_config_json:
        return any(mode_is_retrieval_rank(str(config.get("mode", ""))) for config in load_decode_config_json(args.decode_config_json))
    if args.decode_search:
        return any(mode_is_retrieval_rank(mode.strip()) for mode in args.decode_search_modes.split(","))
    return mode_is_retrieval_rank(args.decode_mode)


def decode_uses_retrieval_count():
    if args.decode_config_json:
        return any(mode_uses_retrieval_count(str(config.get("mode", ""))) for config in load_decode_config_json(args.decode_config_json))
    if args.decode_search:
        return any(mode_uses_retrieval_count(mode.strip()) for mode in args.decode_search_modes.split(","))
    return mode_uses_retrieval_count(args.decode_mode)


if (
    decode_uses_retrieval_rank()
    and args.retrieval_mode == "none"
    and not (args.insert_retrieved_visits and args.insert_visit_mode == "paper_m3")
):
    raise ValueError("retrieval_rank decoders require --retrieval_mode or --insert_visit_mode paper_m3 to select similar visits")
if (
    decode_uses_retrieval_count()
    and args.retrieval_mode not in {"paper_m3", "treatment_episode"}
    and not (args.insert_retrieved_visits and args.insert_visit_mode == "paper_m3")
):
    raise ValueError("--decode_mode *_retrieval_count currently requires --retrieval_mode paper_m3/treatment_episode or --insert_visit_mode paper_m3")
if (
    args.retrieval_copy_mode == "prob_mix"
    and args.retrieval_mode == "none"
    and not (args.insert_retrieved_visits and args.insert_visit_mode == "paper_m3")
):
    raise ValueError("--retrieval_copy_mode prob_mix requires --retrieval_mode or --insert_visit_mode paper_m3")
if (
    args.retrieval_readout_mode != "none"
    and args.retrieval_mode == "none"
    and not (args.insert_retrieved_visits and args.insert_visit_mode == "paper_m3")
):
    raise ValueError("--retrieval_readout_mode requires --retrieval_mode or --insert_visit_mode paper_m3")
if (
    args.retrieval_query_rank_weight > 0
    and args.retrieval_mode == "none"
    and not (args.insert_retrieved_visits and args.insert_visit_mode == "paper_m3")
):
    raise ValueError("--retrieval_query_rank_weight requires --retrieval_mode or --insert_visit_mode paper_m3")
if (
    args.train_graph_rank_weight > 0
    and args.retrieval_mode == "none"
    and not (args.insert_retrieved_visits and args.insert_visit_mode == "paper_m3")
):
    raise ValueError("--train_graph_rank_weight requires --retrieval_mode or --insert_visit_mode paper_m3")
if (
    (args.train_graph_boundary_rank_weight > 0 or args.action_boundary_rank_weight > 0 or args.train_graph_count_weight > 0)
    and args.retrieval_mode == "none"
    and not (args.insert_retrieved_visits and args.insert_visit_mode == "paper_m3")
):
    raise ValueError("--train_graph_boundary_rank_weight/--action_boundary_rank_weight/--train_graph_count_weight require --retrieval_mode or --insert_visit_mode paper_m3")
if (
    args.seq_retrieval_copy_mode != "none"
    and args.retrieval_mode == "none"
    and not (args.insert_retrieved_visits and args.insert_visit_mode == "paper_m3")
):
    raise ValueError("--seq_retrieval_copy_mode requires --retrieval_mode or --insert_visit_mode paper_m3")
if (
    args.seq_decoder_mode in {"hidr_memory", "hidr_transformer"}
    and args.retrieval_mode == "none"
    and not (args.insert_retrieved_visits and args.insert_visit_mode == "paper_m3")
):
    raise ValueError("--seq_decoder_mode hidr_memory/hidr_transformer requires --retrieval_mode or --insert_visit_mode paper_m3")
if args.retrieval_copy_mode == "prob_mix" and args.retrieval_context_source != "static" and args.retrieval_mode == "paper_m3":
    raise ValueError("--retrieval_copy_mode prob_mix with --retrieval_mode paper_m3 requires --retrieval_context_source static")
if args.retrieval_readout_mode != "none" and args.retrieval_context_source != "static" and args.retrieval_mode == "paper_m3":
    raise ValueError("--retrieval_readout_mode with --retrieval_mode paper_m3 requires --retrieval_context_source static")
if args.retrieval_query_rank_weight > 0 and args.retrieval_context_source != "static" and not args.retrieval_mode.startswith("learned"):
    raise ValueError("--retrieval_query_rank_weight with non-static context requires --retrieval_mode learned or learned_clinical")
if args.seq_retrieval_copy_mode != "none" and args.retrieval_context_source != "static" and args.retrieval_mode == "paper_m3":
    raise ValueError("--seq_retrieval_copy_mode with --retrieval_mode paper_m3 requires --retrieval_context_source static")
if args.seq_decoder_mode in {"hidr_memory", "hidr_transformer"} and args.retrieval_context_source != "static" and args.retrieval_mode == "paper_m3":
    raise ValueError("--seq_decoder_mode hidr_memory/hidr_transformer with --retrieval_mode paper_m3 requires --retrieval_context_source static")
if args.retrieval_copy_mode == "fixed_prob" and args.retrieval_mode == "none":
    raise ValueError("--retrieval_copy_mode fixed_prob requires --retrieval_mode")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
#%%
print('loading data...')
# load data
dataset=''
if args.dataset == 3:
    dataset = 'mimic-iii'
elif args.dataset == 4:
    dataset = 'mimic-iv'
    #args.epoches=args.epoches//args.dataset
elif args.dataset == 5:
    dataset = 'eicu'
else:
    raise ValueError(f"unsupported dataset id: {args.dataset}")
dataset_path = args.data_root_override if args.data_root_override else 'data/{}/'.format(dataset)
raw_data,voc,ddi_A,ehr_A=get_data(dataset_path)
dataset_cache_fingerprint = content_digest({
    "records": raw_data,
    "vocabularies": {name: vocabulary.idx2word for name, vocabulary in voc.items()},
    "input_files": {path.name: file_digest(path) for path in sorted(Path(dataset_path).glob("*.pkl"))},
})
patient_keep_mask = [True] * len(raw_data)
data = raw_data
if args.paper_protocol:
    patient_keep_mask = [len(patient) >= 2 for patient in raw_data]
    data = [patient for patient, keep in zip(raw_data, patient_keep_mask) if keep]
size_diag_voc, size_pro_voc, size_med_voc = len(voc['diag_voc'].idx2word), len(voc['pro_voc'].idx2word), len(voc['med_voc'].idx2word)
voc_size=(size_diag_voc, size_pro_voc, size_med_voc)
target_start_visit = 0 if args.include_first_visit_in_paper_protocol else (1 if args.paper_protocol else 0)
seq_decoder_med_rank = None


def count_prediction_visits(records):
    return sum(max(0, len(patient) - target_start_visit) for patient in records)


def count_visits_from(records, start_visit):
    return sum(max(0, len(patient) - start_visit) for patient in records)


def build_prefix_samples_from_batches(batches, batch_id, data_subset, include_target=True):
    diag_parts = []
    proc_parts = []
    med_parts = []
    label_parts = []
    clinical_feature_parts = []
    clinical_feature_mask_parts = []
    clinical_feature_abnormal_parts = []
    clinical_feature_delta_parts = []
    owner_rows = []
    has_clinical_features = batches.get("clinical_feature_batches") is not None
    for row_idx, seq_len in enumerate(batches["batch_seq_len"][batch_id]):
        first_offset = 0 if include_target else 1
        for seq_offset in range(first_offset, seq_len):
            label = (
                batches["label_batches"][batch_id][row_idx].clone()
                if seq_offset == 0
                else batches["med_batches"][batch_id][row_idx, seq_offset].clone()
            )
            if label.sum().item() <= 0:
                continue
            diag_parts.append(batches["diag_batches"][batch_id][row_idx, seq_offset:seq_len].clone())
            proc_parts.append(batches["proc_batches"][batch_id][row_idx, seq_offset:seq_len].clone())
            med_seq = batches["med_batches"][batch_id][row_idx, seq_offset:seq_len].clone()
            med_seq[0] = 0
            med_parts.append(med_seq)
            label_parts.append(label)
            if has_clinical_features:
                clinical_feature_parts.append(
                    batches["clinical_feature_batches"][batch_id][row_idx, seq_offset:seq_len].clone()
                )
                clinical_feature_mask_parts.append(
                    batches["clinical_feature_mask_batches"][batch_id][row_idx, seq_offset:seq_len].clone()
                )
                clinical_feature_abnormal_parts.append(
                    batches["clinical_feature_abnormal_batches"][batch_id][row_idx, seq_offset:seq_len].clone()
                )
                clinical_feature_delta_parts.append(
                    batches["clinical_feature_delta_batches"][batch_id][row_idx, seq_offset:seq_len].clone()
                )
            owner_rows.append(row_idx)
    clinical_parts = None
    if has_clinical_features:
        clinical_parts = (
            clinical_feature_parts,
            clinical_feature_mask_parts,
            clinical_feature_abnormal_parts,
            clinical_feature_delta_parts,
        )
    return diag_parts, proc_parts, med_parts, label_parts, owner_rows, clinical_parts


def build_med_order_rank(records):
    med_counts = np.zeros(size_med_voc, dtype=np.int64)
    for patient in records:
        for visit in patient:
            for med in visit[2]:
                med_counts[int(med)] += 1
    ordered_meds = sorted(range(size_med_voc), key=lambda med: (med_counts[med], med))
    rank = torch.empty(size_med_voc, dtype=torch.long)
    rank[torch.tensor(ordered_meds, dtype=torch.long)] = torch.arange(
        size_med_voc,
        dtype=torch.long,
    )
    return rank

split_manifest = None
if args.split_path:
    with open(args.split_path, "r", encoding="utf-8") as file:
        split_manifest = json.load(file)
    split_index_space = split_manifest.get("patient_index_space", "raw")
    if args.paper_protocol and split_index_space != "paper_filtered":
        raise ValueError(
            "--paper_protocol with --split_path requires "
            "patient_index_space='paper_filtered'"
        )
    if not args.paper_protocol and split_index_space != "raw":
        raise ValueError(
            "--split_path patient_index_space must be 'raw' when paper protocol is disabled"
        )
    train_patient_id_offsets = list(split_manifest["train_patient_indices"])
    valid_patient_id_offsets = list(split_manifest["valid_patient_indices"])
    test_patient_id_offsets = list(split_manifest["test_patient_indices"])
    if args.validation_only_no_test:
        if test_patient_id_offsets:
            raise ValueError("--validation_only_no_test requires an empty test_patient_indices list")
        if not bool(split_manifest.get("test_split_omitted")):
            raise ValueError("--validation_only_no_test split manifest must set test_split_omitted=true")
    if not train_patient_id_offsets or not valid_patient_id_offsets:
        raise ValueError("split manifest requires non-empty train and valid patient indices")
    all_split_indexes = train_patient_id_offsets + valid_patient_id_offsets + test_patient_id_offsets
    if any(index < 0 for index in all_split_indexes):
        raise ValueError("split manifest patient indices must be non-negative")
    max_split_index = max(all_split_indexes)
    if max_split_index >= len(data):
        raise ValueError(f"split manifest index {max_split_index} exceeds loaded patient count {len(data)}")
    data_train = [data[index] for index in train_patient_id_offsets]
    data_test = [] if args.validation_only_no_test else [data[index] for index in test_patient_id_offsets]
    eval_patient_id_offsets = test_patient_id_offsets if args.eval_split == "test" else valid_patient_id_offsets
    data_eval = [data[index] for index in eval_patient_id_offsets]
    split_point = len(data_train)
    eval_len = len(data_test)
    eval_slice_start = 0
    eval_slice_end = None
    train_slice_end = len(data_train)
else:
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    if args.validation_only_no_test:
        data_test = []
    else:
        data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]
    train_patient_id_offsets = list(range(len(data_train)))
    test_patient_id_offsets = (
        []
        if args.validation_only_no_test
        else list(range(split_point, split_point + len(data_test)))
    )
    eval_patient_id_offsets = list(range(split_point + eval_len, split_point + eval_len + len(data_eval)))
    eval_slice_start = split_point + eval_len
    eval_slice_end = None
    train_slice_end = split_point
    if args.validation_only_no_test:
        print(
            "validation_only_no_test enabled: train_patients={} eval_patients={} omitted_middle_patients={}".format(
                len(data_train),
                len(data_eval),
                eval_len,
            )
        )
    if args.train_include_test_split:
        if args.eval_split == "test":
            raise ValueError("--train_include_test_split cannot be combined with --eval_split test")
        data_train = data_train + data_test
        train_patient_id_offsets = train_patient_id_offsets + test_patient_id_offsets
        train_slice_end = split_point + eval_len
    if args.eval_split == "test":
        data_eval = data_test
        eval_patient_id_offsets = test_patient_id_offsets
        eval_slice_start = split_point
        eval_slice_end = split_point + eval_len

if args.split_path and args.train_include_test_split:
    if args.eval_split == "test":
        raise ValueError("--train_include_test_split cannot be combined with --eval_split test")
    data_train = data_train + data_test
    train_patient_id_offsets = train_patient_id_offsets + test_patient_id_offsets
    train_slice_end = len(data_train)

if args.seq_decoder_order == "rare_train":
    seq_decoder_med_rank = build_med_order_rank(data_train)
elif args.seq_decoder_order == "rare_all":
    seq_decoder_med_rank = build_med_order_rank(data)


def split_visit_level_artifacts_by_patient(full_data, labels, masks):
    labels_by_patient = []
    masks_by_patient = []
    cursor = 0
    for patient in full_data:
        visit_count = len(patient)
        labels_by_patient.append(labels[cursor:cursor + visit_count])
        masks_by_patient.append(masks[cursor:cursor + visit_count])
        cursor += visit_count
    if cursor != len(labels):
        raise ValueError("clinical state labels are not aligned with visit sample count")
    return labels_by_patient, masks_by_patient


def flatten_visit_level_artifacts(patient_level_artifacts):
    flattened = []
    for patient_visits in patient_level_artifacts:
        flattened.extend(patient_visits[target_start_visit:])
    return flattened


def split_visit_level_artifact_by_patient(full_data, artifact, artifact_name):
    by_patient = []
    cursor = 0
    for patient in full_data:
        visit_count = len(patient)
        by_patient.append(artifact[cursor:cursor + visit_count])
        cursor += visit_count
    if cursor != len(artifact):
        raise ValueError(f"{artifact_name} is not aligned with visit sample count")
    return by_patient


def filter_patients_by_protocol(patient_level_artifacts):
    if not args.paper_protocol:
        return patient_level_artifacts
    return [artifact for artifact, keep in zip(patient_level_artifacts, patient_keep_mask) if keep]


def split_patient_artifacts(patient_level_artifacts):
    if args.split_path:
        return (
            [patient_level_artifacts[index] for index in train_patient_id_offsets],
            [patient_level_artifacts[index] for index in test_patient_id_offsets],
            [patient_level_artifacts[index] for index in eval_patient_id_offsets],
        )
    test_artifacts = (
        []
        if args.validation_only_no_test
        else patient_level_artifacts[split_point:split_point + eval_len]
    )
    return (
        patient_level_artifacts[:train_slice_end],
        test_artifacts,
        patient_level_artifacts[eval_slice_start:eval_slice_end],
    )


def add_clinical_feature_kwargs(kwargs, feature_payload):
    if feature_payload is None:
        return kwargs
    features, masks, abnormal, delta_hours = feature_payload
    kwargs.update(
        clinical_features=features,
        clinical_feature_masks=masks,
        clinical_feature_abnormal=abnormal,
        clinical_feature_delta_hours=delta_hours,
    )
    return kwargs


def make_treatment_episode_config():
    return {
        "state_weight": args.treatment_episode_state_weight,
        "dxproc_weight": args.treatment_episode_dxproc_weight,
        "clinical_weight": args.treatment_episode_clinical_weight,
        "history_weight": args.treatment_episode_history_weight,
        "action_weight": args.treatment_episode_action_weight,
        "outcome_weight": args.treatment_episode_outcome_weight,
        "safety_weight": args.treatment_episode_safety_weight,
        "score_transform": args.treatment_episode_score_transform,
        "min_outcome_mask": args.treatment_episode_min_outcome_mask,
        "history_mode": args.treatment_episode_history_mode,
        "history_similarity_mode": args.treatment_episode_history_similarity_mode,
        "action_similarity_mode": args.treatment_episode_action_similarity_mode,
    }


def resolve_patient_id_offsets(data_subset, patient_id_offsets=None):
    if patient_id_offsets is None:
        return list(range(len(data_subset)))
    if len(patient_id_offsets) != len(data_subset):
        raise ValueError("patient_id_offsets must align with data_subset")
    return patient_id_offsets


def build_visit_retrieval_inputs(data_subset, voc_size, clinical_payload=None, patient_id_offsets=None, start_visit=None):
    from scipy.sparse import csr_matrix

    if start_visit is None:
        start_visit = target_start_visit
    d_voc_size, p_voc_size, m_voc_size = voc_size
    patient_id_offsets = resolve_patient_id_offsets(data_subset, patient_id_offsets)
    dxproc_rows = []
    dxproc_cols = []
    med_vectors = []
    patient_ids = []
    visit_ids = []
    clinical_values = []
    clinical_masks = []
    history_med_vectors = []
    row_idx = 0

    for patient_idx, patient in enumerate(data_subset):
        global_patient_idx = patient_id_offsets[patient_idx]
        history_med_vector = np.zeros(m_voc_size, dtype=np.float32)
        for previous_visit in patient[:start_visit]:
            if previous_visit[2]:
                history_med_vector[np.asarray(previous_visit[2], dtype=np.int64)] = 1.0
        for visit_idx in range(start_visit, len(patient)):
            visit = patient[visit_idx]
            diag_codes, proc_codes, med_codes = visit
            dxproc_rows.extend([row_idx] * (len(diag_codes) + len(proc_codes)))
            dxproc_cols.extend(list(diag_codes) + [d_voc_size + code for code in proc_codes])

            med_vector = np.zeros(m_voc_size, dtype=np.float32)
            med_vector[np.asarray(med_codes, dtype=np.int64)] = 1.0
            med_vectors.append(med_vector)
            history_med_vectors.append(history_med_vector.copy())
            patient_ids.append(global_patient_idx)
            visit_ids.append(visit_idx)

            if clinical_payload is not None:
                clinical_values.append(np.asarray(clinical_payload[0][patient_idx][visit_idx], dtype=np.float32))
                clinical_masks.append(np.asarray(clinical_payload[1][patient_idx][visit_idx], dtype=np.float32))
            if med_codes:
                history_med_vector[np.asarray(med_codes, dtype=np.int64)] = 1.0
            row_idx += 1

    dxproc_data = np.ones(len(dxproc_rows), dtype=np.float32)
    dxproc = csr_matrix(
        (dxproc_data, (dxproc_rows, dxproc_cols)),
        shape=(row_idx, d_voc_size + p_voc_size),
        dtype=np.float32,
    )
    payload = {
        "dxproc": dxproc,
        "dxproc_count": np.asarray(dxproc.sum(axis=1)).reshape(-1).astype(np.float32),
        "meds": np.stack(med_vectors, axis=0).astype(np.float32),
        "history_meds": np.stack(history_med_vectors, axis=0).astype(np.float32),
        "patient_ids": np.asarray(patient_ids, dtype=np.int64),
        "visit_ids": np.asarray(visit_ids, dtype=np.int64),
    }
    if clinical_payload is not None:
        payload["clinical_values"] = np.stack(clinical_values, axis=0).astype(np.float32)
        payload["clinical_masks"] = np.stack(clinical_masks, axis=0).astype(np.float32)
    return payload


def build_visit_metadata(data_subset, voc_size, patient_id_offsets=None, start_visit=None):
    if start_visit is None:
        start_visit = target_start_visit
    m_voc_size = voc_size[2]
    patient_id_offsets = resolve_patient_id_offsets(data_subset, patient_id_offsets)
    med_vectors = []
    history_med_vectors = []
    patient_ids = []
    visit_ids = []
    for patient_idx, patient in enumerate(data_subset):
        global_patient_idx = patient_id_offsets[patient_idx]
        for visit_idx in range(start_visit, len(patient)):
            visit = patient[visit_idx]
            med_vector = np.zeros(m_voc_size, dtype=np.float32)
            med_vector[np.asarray(visit[2], dtype=np.int64)] = 1.0
            history_med_vector = np.zeros(m_voc_size, dtype=np.float32)
            if visit_idx > 0:
                history_meds = [
                    med
                    for previous_visit in patient[:visit_idx]
                    for med in previous_visit[2]
                ]
                if history_meds:
                    history_med_vector[np.asarray(history_meds, dtype=np.int64)] = 1.0
            med_vectors.append(med_vector)
            history_med_vectors.append(history_med_vector)
            patient_ids.append(global_patient_idx)
            visit_ids.append(visit_idx)
    return {
        "meds": np.stack(med_vectors, axis=0).astype(np.float32),
        "history_meds": np.stack(history_med_vectors, axis=0).astype(np.float32),
        "patient_ids": np.asarray(patient_ids, dtype=np.int64),
        "visit_ids": np.asarray(visit_ids, dtype=np.int64),
    }


def flatten_target_visits(data_subset, start_visit=None):
    if start_visit is None:
        start_visit = target_start_visit
    return [
        patient[visit_idx]
        for patient in data_subset
        for visit_idx in range(start_visit, len(patient))
    ]


def flatten_target_clinical_artifacts(clinical_payload, start_visit=None):
    if clinical_payload is None:
        return None
    if start_visit is None:
        start_visit = target_start_visit
    features, masks, abnormal, delta_hours = clinical_payload
    flattened = {
        "clinical_features": [],
        "clinical_feature_masks": [],
        "clinical_feature_abnormal": [],
        "clinical_feature_delta_hours": [],
    }
    for patient_idx, patient_features in enumerate(features):
        for visit_idx in range(start_visit, len(patient_features)):
            flattened["clinical_features"].append(patient_features[visit_idx])
            flattened["clinical_feature_masks"].append(masks[patient_idx][visit_idx])
            flattened["clinical_feature_abnormal"].append(abnormal[patient_idx][visit_idx])
            flattened["clinical_feature_delta_hours"].append(delta_hours[patient_idx][visit_idx])
    return flattened


def compute_dxproc_similarity(query_inputs, candidate_inputs, start, end):
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


def compute_clinical_similarity(query_inputs, candidate_inputs, start, end):
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


def build_retrieval_priors(
    target_data,
    candidate_data,
    voc_size,
    device,
    mode,
    topk,
    alpha=0.5,
    prior_transform="centered",
    target_clinical_payload=None,
    candidate_clinical_payload=None,
    exclude_same_patient_current_and_future=False,
    allow_train_or_own_history_only=False,
    train_patient_id_limit=None,
    target_patient_id_offsets=None,
    candidate_patient_id_offsets=None,
    candidate_start_visit=None,
):
    if mode == "none":
        return None
    if topk <= 0:
        raise ValueError("retrieval_topk must be positive when retrieval is enabled")
    if "clinical" in mode and (target_clinical_payload is None or candidate_clinical_payload is None):
        raise ValueError("clinical retrieval requires clinical feature sidecars")

    target_inputs = build_visit_retrieval_inputs(
        target_data,
        voc_size,
        target_clinical_payload,
        patient_id_offsets=target_patient_id_offsets,
    )
    candidate_inputs = build_visit_retrieval_inputs(
        candidate_data,
        voc_size,
        candidate_clinical_payload,
        patient_id_offsets=candidate_patient_id_offsets,
        start_visit=candidate_start_visit,
    )
    candidate_count = candidate_inputs["meds"].shape[0]
    topk = min(topk, candidate_count)
    priors = np.zeros((target_inputs["meds"].shape[0], voc_size[2]), dtype=np.float32)
    if allow_train_or_own_history_only:
        if train_patient_id_limit is None:
            raise ValueError("train_patient_id_limit is required for train_plus_own_history retrieval")
        train_candidate_mask = candidate_inputs["patient_ids"] < train_patient_id_limit
        global_prior = candidate_inputs["meds"][train_candidate_mask].mean(axis=0).astype(np.float32)
    else:
        global_prior = candidate_inputs["meds"].mean(axis=0).astype(np.float32)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    chunk_size = 256
    for start in range(0, priors.shape[0], chunk_size):
        end = min(start + chunk_size, priors.shape[0])
        if mode == "dxproc":
            sim = compute_dxproc_similarity(target_inputs, candidate_inputs, start, end)
        elif mode == "clinical":
            sim = compute_clinical_similarity(target_inputs, candidate_inputs, start, end)
        elif mode == "history_med":
            sim = compute_history_med_overlap_similarity(target_inputs, candidate_inputs, start, end)
            if sim is None:
                raise ValueError("history_med retrieval requires target history medications and candidate medications")
        else:
            dxproc_sim = compute_dxproc_similarity(target_inputs, candidate_inputs, start, end)
            clinical_sim = compute_clinical_similarity(target_inputs, candidate_inputs, start, end)
            sim = alpha * dxproc_sim + (1.0 - alpha) * clinical_sim

        if exclude_same_patient_current_and_future or allow_train_or_own_history_only:
            for local_row, query_row in enumerate(range(start, end)):
                if exclude_same_patient_current_and_future:
                    same_patient_future = (
                        (candidate_inputs["patient_ids"] == target_inputs["patient_ids"][query_row])
                        & (candidate_inputs["visit_ids"] >= target_inputs["visit_ids"][query_row])
                    )
                    sim[local_row, same_patient_future] = -np.inf
                if allow_train_or_own_history_only:
                    allowed = (
                        (candidate_inputs["patient_ids"] < train_patient_id_limit)
                        | (
                            (candidate_inputs["patient_ids"] == target_inputs["patient_ids"][query_row])
                            & (candidate_inputs["visit_ids"] < target_inputs["visit_ids"][query_row])
                        )
                    )
                    sim[local_row, ~allowed] = -np.inf

        top_indexes = np.argpartition(sim, -topk, axis=1)[:, -topk:]
        top_scores = np.take_along_axis(sim, top_indexes, axis=1)
        top_scores = np.maximum(top_scores, 0.0)
        score_sums = top_scores.sum(axis=1, keepdims=True)

        for local_row in range(end - start):
            if score_sums[local_row, 0] <= 0:
                priors[start + local_row] = global_prior
                continue
            weights = top_scores[local_row] / score_sums[local_row, 0]
            priors[start + local_row] = np.matmul(weights, candidate_inputs["meds"][top_indexes[local_row]])

    if prior_transform == "centered":
        priors = priors - global_prior.reshape(1, -1)
    elif prior_transform != "positive":
        raise ValueError(f"Unsupported retrieval_prior_transform: {prior_transform}")
    return [torch.tensor(row, dtype=torch.float32, device=device) for row in priors]


def build_inserted_retrieved_visits(
    target_data,
    candidate_data,
    voc_size,
    mode,
    topk,
    alpha=0.5,
    target_clinical_payload=None,
    candidate_clinical_payload=None,
    exclude_same_patient_current_and_future=False,
    allow_train_or_own_history_only=False,
    train_patient_id_limit=None,
    target_patient_id_offsets=None,
    candidate_patient_id_offsets=None,
    candidate_start_visit=None,
):
    if topk <= 0:
        return [[] for _ in range(count_prediction_visits(target_data))]
    if "clinical" in mode and (target_clinical_payload is None or candidate_clinical_payload is None):
        raise ValueError("clinical inserted-visit retrieval requires clinical feature sidecars")

    target_inputs = build_visit_retrieval_inputs(
        target_data,
        voc_size,
        target_clinical_payload,
        patient_id_offsets=target_patient_id_offsets,
    )
    candidate_inputs = build_visit_retrieval_inputs(
        candidate_data,
        voc_size,
        candidate_clinical_payload,
        patient_id_offsets=candidate_patient_id_offsets,
        start_visit=candidate_start_visit,
    )
    candidate_visits = flatten_target_visits(candidate_data, start_visit=candidate_start_visit)
    candidate_clinical = flatten_target_clinical_artifacts(candidate_clinical_payload, start_visit=candidate_start_visit)
    candidate_count = len(candidate_visits)
    topk = min(topk, candidate_count)
    inserted = []
    alpha = float(np.clip(alpha, 0.0, 1.0))

    chunk_size = 256
    for start in range(0, target_inputs["meds"].shape[0], chunk_size):
        end = min(start + chunk_size, target_inputs["meds"].shape[0])
        if mode == "dxproc":
            sim = compute_dxproc_similarity(target_inputs, candidate_inputs, start, end)
        elif mode == "clinical":
            sim = compute_clinical_similarity(target_inputs, candidate_inputs, start, end)
        elif mode == "history_med":
            sim = compute_history_med_overlap_similarity(target_inputs, candidate_inputs, start, end)
            if sim is None:
                raise ValueError("history_med inserted-visit retrieval requires target history medications and candidate medications")
        else:
            dxproc_sim = compute_dxproc_similarity(target_inputs, candidate_inputs, start, end)
            clinical_sim = compute_clinical_similarity(target_inputs, candidate_inputs, start, end)
            sim = alpha * dxproc_sim + (1.0 - alpha) * clinical_sim

        if exclude_same_patient_current_and_future or allow_train_or_own_history_only:
            for local_row, query_row in enumerate(range(start, end)):
                if exclude_same_patient_current_and_future:
                    same_patient_future = (
                        (candidate_inputs["patient_ids"] == target_inputs["patient_ids"][query_row])
                        & (candidate_inputs["visit_ids"] >= target_inputs["visit_ids"][query_row])
                    )
                    sim[local_row, same_patient_future] = -np.inf
                if allow_train_or_own_history_only:
                    if train_patient_id_limit is None:
                        raise ValueError("train_patient_id_limit is required for train_plus_own_history retrieval")
                    allowed = (
                        (candidate_inputs["patient_ids"] < train_patient_id_limit)
                        | (
                            (candidate_inputs["patient_ids"] == target_inputs["patient_ids"][query_row])
                            & (candidate_inputs["visit_ids"] < target_inputs["visit_ids"][query_row])
                        )
                    )
                    sim[local_row, ~allowed] = -np.inf

        top_indexes = np.argpartition(sim, -topk, axis=1)[:, -topk:]
        top_scores = np.take_along_axis(sim, top_indexes, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_indexes = np.take_along_axis(top_indexes, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        for row, row_scores in zip(top_indexes, top_scores):
            row_inserted = []
            for candidate_idx, score in zip(row, row_scores):
                if not np.isfinite(score):
                    continue
                candidate_idx = int(candidate_idx)
                visit = candidate_visits[candidate_idx]
                if candidate_clinical is None:
                    row_inserted.append(visit)
                else:
                    row_inserted.append({
                        "visit": visit,
                        "clinical_features": candidate_clinical["clinical_features"][candidate_idx],
                        "clinical_feature_masks": candidate_clinical["clinical_feature_masks"][candidate_idx],
                        "clinical_feature_abnormal": candidate_clinical["clinical_feature_abnormal"][candidate_idx],
                        "clinical_feature_delta_hours": candidate_clinical["clinical_feature_delta_hours"][candidate_idx],
                    })
            inserted.append(row_inserted)
    return inserted


def build_static_retrieval_context(
    target_data,
    candidate_data,
    voc_size,
    mode,
    topk,
    alpha=0.5,
    target_clinical_payload=None,
    candidate_clinical_payload=None,
    exclude_same_patient_current_and_future=False,
    allow_train_or_own_history_only=False,
    train_patient_id_limit=None,
    target_patient_id_offsets=None,
    candidate_patient_id_offsets=None,
    candidate_start_visit=None,
):
    if topk <= 0:
        raise ValueError("retrieval_topk must be positive when static retrieval context is enabled")
    if "clinical" in mode and (target_clinical_payload is None or candidate_clinical_payload is None):
        raise ValueError("clinical static retrieval context requires clinical feature sidecars")

    target_inputs = build_visit_retrieval_inputs(
        target_data,
        voc_size,
        target_clinical_payload,
        patient_id_offsets=target_patient_id_offsets,
    )
    candidate_inputs = build_visit_retrieval_inputs(
        candidate_data,
        voc_size,
        candidate_clinical_payload,
        patient_id_offsets=candidate_patient_id_offsets,
        start_visit=candidate_start_visit,
    )
    candidate_count = candidate_inputs["meds"].shape[0]
    topk = min(topk, candidate_count)
    retrieval_meds = []
    retrieval_scores = []
    alpha = float(np.clip(alpha, 0.0, 1.0))

    chunk_size = 256
    for start in range(0, target_inputs["meds"].shape[0], chunk_size):
        end = min(start + chunk_size, target_inputs["meds"].shape[0])
        if mode == "dxproc":
            sim = compute_dxproc_similarity(target_inputs, candidate_inputs, start, end)
        elif mode == "clinical":
            sim = compute_clinical_similarity(target_inputs, candidate_inputs, start, end)
        elif mode == "history_med":
            sim = compute_history_med_overlap_similarity(target_inputs, candidate_inputs, start, end)
            if sim is None:
                raise ValueError("history_med static retrieval context requires target history medications and candidate medications")
        else:
            dxproc_sim = compute_dxproc_similarity(target_inputs, candidate_inputs, start, end)
            clinical_sim = compute_clinical_similarity(target_inputs, candidate_inputs, start, end)
            sim = alpha * dxproc_sim + (1.0 - alpha) * clinical_sim

        if exclude_same_patient_current_and_future or allow_train_or_own_history_only:
            for local_row, query_row in enumerate(range(start, end)):
                if exclude_same_patient_current_and_future:
                    same_patient_future = (
                        (candidate_inputs["patient_ids"] == target_inputs["patient_ids"][query_row])
                        & (candidate_inputs["visit_ids"] >= target_inputs["visit_ids"][query_row])
                    )
                    sim[local_row, same_patient_future] = -np.inf
                if allow_train_or_own_history_only:
                    if train_patient_id_limit is None:
                        raise ValueError("train_patient_id_limit is required for train_plus_own_history retrieval")
                    allowed = (
                        (candidate_inputs["patient_ids"] < train_patient_id_limit)
                        | (
                            (candidate_inputs["patient_ids"] == target_inputs["patient_ids"][query_row])
                            & (candidate_inputs["visit_ids"] < target_inputs["visit_ids"][query_row])
                        )
                    )
                    sim[local_row, ~allowed] = -np.inf

        top_indexes = np.argpartition(sim, -topk, axis=1)[:, -topk:]
        top_scores = np.take_along_axis(sim, top_indexes, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_indexes = np.take_along_axis(top_indexes, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)

        finite_scores = np.where(np.isfinite(top_scores), top_scores, 0.0).astype(np.float32)
        score_min = finite_scores.min(axis=1, keepdims=True)
        score_max = finite_scores.max(axis=1, keepdims=True)
        score_range = np.maximum(score_max - score_min, 1e-6)
        normalized_scores = (finite_scores - score_min) / score_range

        meds = candidate_inputs["meds"][top_indexes].astype(np.float32)
        invalid = ~np.isfinite(top_scores)
        if invalid.any():
            meds[invalid] = 0.0
            normalized_scores[invalid] = 0.0
        retrieval_meds.extend(torch.tensor(row, dtype=torch.float32, device=device) for row in meds)
        retrieval_scores.extend(torch.tensor(row, dtype=torch.float32, device=device) for row in normalized_scores.astype(np.float32))
    return retrieval_meds, retrieval_scores, None


def build_priors_from_similarity(
    sim,
    target_metadata,
    candidate_metadata,
    row_offset=0,
    exclude_same_patient_current_and_future=False,
    allow_train_or_own_history_only=False,
    train_patient_id_limit=None,
    topk=None,
):
    if topk is None:
        topk = args.retrieval_topk
    topk = min(topk, candidate_metadata["meds"].shape[0])
    priors = np.zeros((sim.shape[0], candidate_metadata["meds"].shape[1]), dtype=np.float32)
    if allow_train_or_own_history_only:
        if train_patient_id_limit is None:
            raise ValueError("train_patient_id_limit is required for train_plus_own_history retrieval")
        train_candidate_mask = candidate_metadata["patient_ids"] < train_patient_id_limit
        global_prior = candidate_metadata["meds"][train_candidate_mask].mean(axis=0).astype(np.float32)
    else:
        global_prior = candidate_metadata["meds"].mean(axis=0).astype(np.float32)

    if exclude_same_patient_current_and_future or allow_train_or_own_history_only:
        for query_row in range(sim.shape[0]):
            target_row = row_offset + query_row
            if exclude_same_patient_current_and_future:
                same_patient_future = (
                    (candidate_metadata["patient_ids"] == target_metadata["patient_ids"][target_row])
                    & (candidate_metadata["visit_ids"] >= target_metadata["visit_ids"][target_row])
                )
                sim[query_row, same_patient_future] = -np.inf
            if allow_train_or_own_history_only:
                allowed = (
                    (candidate_metadata["patient_ids"] < train_patient_id_limit)
                    | (
                        (candidate_metadata["patient_ids"] == target_metadata["patient_ids"][target_row])
                        & (candidate_metadata["visit_ids"] < target_metadata["visit_ids"][target_row])
                    )
                )
                sim[query_row, ~allowed] = -np.inf

    top_indexes = np.argpartition(sim, -topk, axis=1)[:, -topk:]
    top_scores = np.take_along_axis(sim, top_indexes, axis=1)
    top_scores = np.maximum(top_scores, 0.0)
    score_sums = top_scores.sum(axis=1, keepdims=True)
    for row in range(sim.shape[0]):
        if score_sums[row, 0] <= 0:
            priors[row] = global_prior
            continue
        weights = top_scores[row] / score_sums[row, 0]
        priors[row] = np.matmul(weights, candidate_metadata["meds"][top_indexes[row]])

    if args.retrieval_prior_transform == "centered":
        priors = priors - global_prior.reshape(1, -1)
    elif args.retrieval_prior_transform != "positive":
        raise ValueError(f"Unsupported retrieval_prior_transform: {args.retrieval_prior_transform}")
    return [torch.tensor(row, dtype=torch.float32, device=device) for row in priors]


def visit_meds_to_multihot(visit, med_voc_size):
    med_vector = np.zeros(med_voc_size, dtype=np.float32)
    med_vector[np.asarray(visit[2], dtype=np.int64)] = 1.0
    return med_vector


def load_official_topidx(split_name, expected_rows=None):
    topidx_dir = Path(args.official_topidx_dir)
    if not topidx_dir.exists():
        raise FileNotFoundError(f"--official_topidx_dir not found: {topidx_dir}")
    matches = sorted(topidx_dir.glob(f"*_{split_name}_top*.pkl"))
    if not matches:
        raise FileNotFoundError(f"no official Top-K artifact matching *_{split_name}_top*.pkl under {topidx_dir}")
    if len(matches) > 1:
        raise ValueError(f"multiple official Top-K artifacts for split {split_name}: {matches}")
    topidx = np.asarray(seer_security.load_pickle(open(matches[0], "rb")), dtype=np.int64)
    if topidx.ndim != 2:
        raise ValueError(f"official Top-K artifact must be 2-D: {matches[0]}")
    if expected_rows is not None and topidx.shape[0] != expected_rows:
        raise ValueError(f"{matches[0]} rows {topidx.shape[0]} do not match expected rows {expected_rows}")
    return topidx


def build_official_candidate_meds(split_name, train_data, split_data, med_voc_size):
    train_visits = flatten_target_visits(train_data, start_visit=target_start_visit)
    if split_name == "train":
        return np.stack([visit_meds_to_multihot(visit, med_voc_size) for visit in train_visits], axis=0)
    split_visits = flatten_target_visits(split_data, start_visit=target_start_visit)
    candidate_visits = train_visits + split_visits
    return np.stack([visit_meds_to_multihot(visit, med_voc_size) for visit in candidate_visits], axis=0)


def build_official_topidx_priors(split_name, target_data, split_data):
    if not args.paper_protocol or target_start_visit != 1:
        raise ValueError("official_topidx artifacts are aligned to --paper_protocol with target_start_visit=1")
    topidx = load_official_topidx(split_name, expected_rows=count_prediction_visits(target_data))
    if args.retrieval_topk <= 0:
        raise ValueError("--retrieval_topk must be positive for official_topidx")
    topk = min(args.retrieval_topk, topidx.shape[1])
    candidate_meds = build_official_candidate_meds(split_name, data_train, split_data, size_med_voc)
    if topidx.size and (topidx.min() < 0 or topidx.max() >= candidate_meds.shape[0]):
        raise ValueError(f"official_topidx {split_name} contains index outside candidate range")
    global_prior = candidate_meds[:count_prediction_visits(data_train)].mean(axis=0).astype(np.float32)
    priors = candidate_meds[topidx[:, :topk]].mean(axis=1).astype(np.float32)
    if args.retrieval_prior_transform == "centered":
        priors = priors - global_prior.reshape(1, -1)
    elif args.retrieval_prior_transform != "positive":
        raise ValueError(f"Unsupported retrieval_prior_transform: {args.retrieval_prior_transform}")
    return [torch.tensor(row, dtype=torch.float32, device=device) for row in priors]


def build_context_from_similarity(
    sim,
    target_metadata,
    candidate_metadata,
    candidate_queries,
    row_offset=0,
    exclude_same_patient_current_and_future=False,
    allow_train_or_own_history_only=False,
    train_patient_id_limit=None,
):
    topk = min(args.retrieval_topk, candidate_metadata["meds"].shape[0])
    if exclude_same_patient_current_and_future or allow_train_or_own_history_only:
        for query_row in range(sim.shape[0]):
            target_row = row_offset + query_row
            if exclude_same_patient_current_and_future:
                same_patient_future = (
                    (candidate_metadata["patient_ids"] == target_metadata["patient_ids"][target_row])
                    & (candidate_metadata["visit_ids"] >= target_metadata["visit_ids"][target_row])
                )
                sim[query_row, same_patient_future] = -np.inf
            if allow_train_or_own_history_only:
                if train_patient_id_limit is None:
                    raise ValueError("train_patient_id_limit is required for train_plus_own_history retrieval")
                allowed = (
                    (candidate_metadata["patient_ids"] < train_patient_id_limit)
                    | (
                        (candidate_metadata["patient_ids"] == target_metadata["patient_ids"][target_row])
                        & (candidate_metadata["visit_ids"] < target_metadata["visit_ids"][target_row])
                    )
                )
                sim[query_row, ~allowed] = -np.inf

    top_indexes = np.argpartition(sim, -topk, axis=1)[:, -topk:]
    top_scores = np.take_along_axis(sim, top_indexes, axis=1)
    order = np.argsort(-top_scores, axis=1)
    top_indexes = np.take_along_axis(top_indexes, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)

    finite_scores = np.where(np.isfinite(top_scores), top_scores, 0.0).astype(np.float32)
    score_min = finite_scores.min(axis=1, keepdims=True)
    score_max = finite_scores.max(axis=1, keepdims=True)
    score_range = np.maximum(score_max - score_min, 1e-6)
    normalized_scores = (finite_scores - score_min) / score_range

    meds = candidate_metadata["meds"][top_indexes].astype(np.float32)
    queries = candidate_queries[top_indexes].astype(np.float32)
    invalid = ~np.isfinite(top_scores)
    if invalid.any():
        meds[invalid] = 0.0
        queries[invalid] = 0.0
        normalized_scores[invalid] = 0.0

    return (
        [torch.tensor(row, dtype=torch.float32, device=device) for row in meds],
        [torch.tensor(row, dtype=torch.float32, device=device) for row in normalized_scores.astype(np.float32)],
        [torch.tensor(row, dtype=torch.float32, device=device) for row in queries],
    )


def build_med_context_from_similarity(
    sim,
    target_metadata,
    candidate_metadata,
    candidate_queries=None,
    row_offset=0,
    exclude_same_patient_current_and_future=False,
    allow_train_or_own_history_only=False,
    train_patient_id_limit=None,
    topk=None,
):
    if topk is None:
        topk = args.retrieval_topk
    topk = min(topk, candidate_metadata["meds"].shape[0])
    if exclude_same_patient_current_and_future or allow_train_or_own_history_only:
        for query_row in range(sim.shape[0]):
            target_row = row_offset + query_row
            if exclude_same_patient_current_and_future:
                same_patient_future = (
                    (candidate_metadata["patient_ids"] == target_metadata["patient_ids"][target_row])
                    & (candidate_metadata["visit_ids"] >= target_metadata["visit_ids"][target_row])
                )
                sim[query_row, same_patient_future] = -np.inf
            if allow_train_or_own_history_only:
                if train_patient_id_limit is None:
                    raise ValueError("train_patient_id_limit is required for train_plus_own_history retrieval")
                allowed = (
                    (candidate_metadata["patient_ids"] < train_patient_id_limit)
                    | (
                        (candidate_metadata["patient_ids"] == target_metadata["patient_ids"][target_row])
                        & (candidate_metadata["visit_ids"] < target_metadata["visit_ids"][target_row])
                    )
                )
                sim[query_row, ~allowed] = -np.inf

    top_indexes = np.argpartition(sim, -topk, axis=1)[:, -topk:]
    top_scores = np.take_along_axis(sim, top_indexes, axis=1)
    order = np.argsort(-top_scores, axis=1)
    top_indexes = np.take_along_axis(top_indexes, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)

    finite_scores = np.where(np.isfinite(top_scores), top_scores, 0.0).astype(np.float32)
    score_min = finite_scores.min(axis=1, keepdims=True)
    score_max = finite_scores.max(axis=1, keepdims=True)
    score_range = np.maximum(score_max - score_min, 1e-6)
    normalized_scores = (finite_scores - score_min) / score_range

    meds = candidate_metadata["meds"][top_indexes].astype(np.float32)
    queries = candidate_queries[top_indexes].astype(np.float32) if candidate_queries is not None else None
    invalid = ~np.isfinite(top_scores)
    if invalid.any():
        meds[invalid] = 0.0
        if queries is not None:
            queries[invalid] = 0.0
        normalized_scores[invalid] = 0.0
    return (
        [torch.tensor(row, dtype=torch.float32, device=device) for row in meds],
        [torch.tensor(row, dtype=torch.float32, device=device) for row in normalized_scores.astype(np.float32)],
        [torch.tensor(row, dtype=torch.float32, device=device) for row in queries] if queries is not None else None,
    )


def build_inserted_from_similarity(
    sim,
    target_metadata,
    candidate_metadata,
    candidate_visits,
    candidate_clinical=None,
    row_offset=0,
    exclude_same_patient_current_and_future=False,
    allow_train_or_own_history_only=False,
    train_patient_id_limit=None,
):
    topk = min(args.insert_visit_topk, len(candidate_visits))
    if topk <= 0:
        return [[] for _ in range(sim.shape[0])]
    if exclude_same_patient_current_and_future or allow_train_or_own_history_only:
        for query_row in range(sim.shape[0]):
            target_row = row_offset + query_row
            if exclude_same_patient_current_and_future:
                same_patient_future = (
                    (candidate_metadata["patient_ids"] == target_metadata["patient_ids"][target_row])
                    & (candidate_metadata["visit_ids"] >= target_metadata["visit_ids"][target_row])
                )
                sim[query_row, same_patient_future] = -np.inf
            if allow_train_or_own_history_only:
                if train_patient_id_limit is None:
                    raise ValueError("train_patient_id_limit is required for train_plus_own_history retrieval")
                allowed = (
                    (candidate_metadata["patient_ids"] < train_patient_id_limit)
                    | (
                        (candidate_metadata["patient_ids"] == target_metadata["patient_ids"][target_row])
                        & (candidate_metadata["visit_ids"] < target_metadata["visit_ids"][target_row])
                    )
                )
                sim[query_row, ~allowed] = -np.inf

    top_indexes = np.argpartition(sim, -topk, axis=1)[:, -topk:]
    top_scores = np.take_along_axis(sim, top_indexes, axis=1)
    order = np.argsort(-top_scores, axis=1)
    top_indexes = np.take_along_axis(top_indexes, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)

    inserted = []
    for row, row_scores in zip(top_indexes, top_scores):
        row_inserted = []
        for candidate_idx, score in zip(row, row_scores):
            if not np.isfinite(score):
                continue
            candidate_idx = int(candidate_idx)
            visit = candidate_visits[candidate_idx]
            if candidate_clinical is None:
                row_inserted.append(visit)
            else:
                row_inserted.append({
                    "visit": visit,
                    "clinical_features": candidate_clinical["clinical_features"][candidate_idx],
                    "clinical_feature_masks": candidate_clinical["clinical_feature_masks"][candidate_idx],
                    "clinical_feature_abnormal": candidate_clinical["clinical_feature_abnormal"][candidate_idx],
                    "clinical_feature_delta_hours": candidate_clinical["clinical_feature_delta_hours"][candidate_idx],
                })
        inserted.append(row_inserted)
    return inserted


def forward_model(model, batches, batch_id):
    seq_decoder_inputs = None
    if seq_decoder_enabled():
        seq_decoder_inputs, _ = build_seq_decoder_targets(batches['label_batches'][batch_id])
    return model(
        batches['diag_batches'][batch_id],
        batches['proc_batches'][batch_id],
        batches['med_batches'][batch_id],
        clinical_features=batches["clinical_feature_batches"][batch_id] if batches.get("clinical_feature_batches") is not None else None,
        clinical_feature_masks=batches["clinical_feature_mask_batches"][batch_id] if batches.get("clinical_feature_mask_batches") is not None else None,
        clinical_feature_abnormal=batches["clinical_feature_abnormal_batches"][batch_id] if batches.get("clinical_feature_abnormal_batches") is not None else None,
        clinical_feature_delta_hours=batches["clinical_feature_delta_batches"][batch_id] if batches.get("clinical_feature_delta_batches") is not None else None,
        retrieval_meds=batches["retrieval_med_batches"][batch_id] if batches.get("retrieval_med_batches") is not None else None,
        retrieval_scores=batches["retrieval_score_batches"][batch_id] if batches.get("retrieval_score_batches") is not None else None,
        retrieval_queries=batches["retrieval_query_batches"][batch_id] if batches.get("retrieval_query_batches") is not None else None,
        treatment_episode_experiences=batches["treatment_episode_experience_batches"][batch_id] if batches.get("treatment_episode_experience_batches") is not None else None,
        treatment_episode_scores=batches["treatment_episode_score_batches"][batch_id] if batches.get("treatment_episode_score_batches") is not None else None,
        retrieval_priors=batches["retrieval_prior_batches"][batch_id] if batches.get("retrieval_prior_batches") is not None else None,
        own_diags=batches["own_diag_batches"][batch_id] if batches.get("own_diag_batches") is not None else None,
        own_procs=batches["own_proc_batches"][batch_id] if batches.get("own_proc_batches") is not None else None,
        own_meds=batches["own_med_batches"][batch_id] if batches.get("own_med_batches") is not None else None,
        seq_decoder_inputs=seq_decoder_inputs,
    )


def forward_plain_prefix_batch(model, diag_parts, proc_parts, med_parts, label_parts, batch_size, clinical_parts=None):
    logits = []
    labels = []
    clinical_feature_parts = clinical_feature_mask_parts = None
    clinical_feature_abnormal_parts = clinical_feature_delta_parts = None
    if clinical_parts is not None:
        (
            clinical_feature_parts,
            clinical_feature_mask_parts,
            clinical_feature_abnormal_parts,
            clinical_feature_delta_parts,
        ) = clinical_parts
    for start in range(0, len(diag_parts), batch_size):
        end = min(start + batch_size, len(diag_parts))
        diag_batch, _, _ = tensors_to_dynamic_batches(diag_parts[start:end], end - start, sort=False)
        proc_batch, _, _ = tensors_to_dynamic_batches(proc_parts[start:end], end - start, sort=False)
        med_batch, _, _ = tensors_to_dynamic_batches(med_parts[start:end], end - start, sort=False)
        model_kwargs = {}
        if clinical_parts is not None:
            clinical_feature_batch, _, _ = tensors_to_dynamic_batches(
                clinical_feature_parts[start:end],
                end - start,
                sort=False,
            )
            clinical_feature_mask_batch, _, _ = tensors_to_dynamic_batches(
                clinical_feature_mask_parts[start:end],
                end - start,
                sort=False,
            )
            clinical_feature_abnormal_batch, _, _ = tensors_to_dynamic_batches(
                clinical_feature_abnormal_parts[start:end],
                end - start,
                sort=False,
            )
            clinical_feature_delta_batch, _, _ = tensors_to_dynamic_batches(
                clinical_feature_delta_parts[start:end],
                end - start,
                sort=False,
            )
            model_kwargs.update({
                "clinical_features": clinical_feature_batch[0],
                "clinical_feature_masks": clinical_feature_mask_batch[0],
                "clinical_feature_abnormal": clinical_feature_abnormal_batch[0],
                "clinical_feature_delta_hours": clinical_feature_delta_batch[0],
            })
        output = model(diag_batch[0], proc_batch[0], med_batch[0], **model_kwargs)
        logits.append(apply_history_prior(output["med_logits"], med_batch[0]))
        labels.extend(label for label in label_parts[start:end])
    return torch.cat(logits, dim=0), torch.stack(labels, dim=0)


def encode_patient_queries(model, batches, total_samples):
    return encode_patient_queries_with_normalization(model, batches, total_samples, normalize=True)


def encode_patient_queries_with_normalization(model, batches, total_samples, normalize=True):
    previous_training = model.training
    model.eval()
    encoded = [None] * total_samples
    with torch.no_grad():
        for batch_id in range(batches["n_batches"]):
            _, patient_query = model.encode_patient_query(
                batches['diag_batches'][batch_id],
                batches['proc_batches'][batch_id],
                batches['med_batches'][batch_id],
                clinical_features=batches["clinical_feature_batches"][batch_id] if batches.get("clinical_feature_batches") is not None else None,
                clinical_feature_masks=batches["clinical_feature_mask_batches"][batch_id] if batches.get("clinical_feature_mask_batches") is not None else None,
                clinical_feature_abnormal=batches["clinical_feature_abnormal_batches"][batch_id] if batches.get("clinical_feature_abnormal_batches") is not None else None,
                clinical_feature_delta_hours=batches["clinical_feature_delta_batches"][batch_id] if batches.get("clinical_feature_delta_batches") is not None else None,
            )
            if normalize:
                patient_query = F.normalize(patient_query, p=2, dim=1)
            patient_query = patient_query.detach().cpu().numpy().astype(np.float32)
            for local_row, sample_idx in enumerate(batches["sample_index_batches"][batch_id]):
                encoded[sample_idx] = patient_query[local_row]
    if previous_training:
        model.train()
    return np.stack(encoded, axis=0)


def _normalize_rows_numpy(matrix, eps=1e-8):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def _center_rows_numpy(matrix):
    return matrix - matrix.mean(axis=1, keepdims=True)


def _rank_rows_numpy(matrix):
    order = np.argsort(matrix, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    row_ids = np.arange(matrix.shape[0])[:, None]
    ranks[row_ids, order] = np.arange(matrix.shape[1], dtype=np.float32)
    return ranks


def _pairwise_similarity_blockwise(query_matrix, candidate_matrix, reducer, block_size=512):
    output = np.empty((query_matrix.shape[0], candidate_matrix.shape[0]), dtype=np.float32)
    for start in range(0, candidate_matrix.shape[0], block_size):
        end = min(start + block_size, candidate_matrix.shape[0])
        output[:, start:end] = reducer(query_matrix, candidate_matrix[start:end])
    return output


def compute_query_similarity(query_matrix, candidate_matrix, mode, rbf_gamma=None):
    mode = mode.lower()
    if mode == "cosine":
        return np.matmul(_normalize_rows_numpy(query_matrix), _normalize_rows_numpy(candidate_matrix).T)
    if mode == "dot":
        return np.matmul(query_matrix, candidate_matrix.T)
    if mode == "angular":
        cosine = compute_query_similarity(query_matrix, candidate_matrix, "cosine")
        return 1.0 - np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi
    if mode == "euclidean":
        query_sq = np.sum(query_matrix * query_matrix, axis=1, keepdims=True)
        candidate_sq = np.sum(candidate_matrix * candidate_matrix, axis=1, keepdims=True).T
        dist_sq = np.maximum(query_sq + candidate_sq - 2.0 * np.matmul(query_matrix, candidate_matrix.T), 0.0)
        return 1.0 / (1.0 + np.sqrt(dist_sq))
    if mode == "manhattan":
        return _pairwise_similarity_blockwise(
            query_matrix,
            candidate_matrix,
            lambda q, c: 1.0 / (1.0 + np.sum(np.abs(q[:, None, :] - c[None, :, :]), axis=-1)),
        )
    if mode == "chebyshev":
        return _pairwise_similarity_blockwise(
            query_matrix,
            candidate_matrix,
            lambda q, c: 1.0 / (1.0 + np.max(np.abs(q[:, None, :] - c[None, :, :]), axis=-1)),
        )
    if mode == "pearson":
        centered_query = _center_rows_numpy(query_matrix)
        centered_candidate = _center_rows_numpy(candidate_matrix)
        return np.matmul(_normalize_rows_numpy(centered_query), _normalize_rows_numpy(centered_candidate).T)
    if mode == "spearman":
        ranked_query = _rank_rows_numpy(query_matrix)
        ranked_candidate = _rank_rows_numpy(candidate_matrix)
        return np.matmul(
            _normalize_rows_numpy(_center_rows_numpy(ranked_query)),
            _normalize_rows_numpy(_center_rows_numpy(ranked_candidate)).T,
        )
    if mode == "rbf":
        gamma = 0.00390625 if rbf_gamma is None else float(rbf_gamma)
        query_sq = np.sum(query_matrix * query_matrix, axis=1, keepdims=True)
        candidate_sq = np.sum(candidate_matrix * candidate_matrix, axis=1, keepdims=True).T
        dist_sq = np.maximum(query_sq + candidate_sq - 2.0 * np.matmul(query_matrix, candidate_matrix.T), 0.0)
        return np.exp(-gamma * dist_sq)
    raise ValueError(f"Unsupported paper_m3 query similarity: {mode}")


def fit_label_ridge_query_transform(query_matrix, label_matrix, ridge):
    query = np.asarray(query_matrix, dtype=np.float32)
    label = np.asarray(label_matrix, dtype=np.float32)
    augmented_query = np.concatenate((query, np.ones((query.shape[0], 1), dtype=np.float32)), axis=1)
    gram = np.matmul(augmented_query.T, augmented_query)
    if ridge > 0:
        regularizer = np.eye(gram.shape[0], dtype=np.float32) * float(ridge)
        regularizer[-1, -1] = 0.0
        gram = gram + regularizer
    rhs = np.matmul(augmented_query.T, label)
    try:
        weights = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gram).dot(rhs)
    return weights.astype(np.float32)


def apply_label_ridge_query_transform(query_matrix, weights, mode):
    query = np.asarray(query_matrix, dtype=np.float32)
    augmented_query = np.concatenate((query, np.ones((query.shape[0], 1), dtype=np.float32)), axis=1)
    transformed = np.matmul(augmented_query, weights).astype(np.float32)
    if mode in {"label_ridge_sigmoid", "label_ridge_sigmoid_to_meds"}:
        transformed = 1.0 / (1.0 + np.exp(-np.clip(transformed, -30.0, 30.0)))
    return transformed.astype(np.float32)


def transform_candidate_queries_for_similarity(candidate_queries, candidate_metadata, query_transform_weights):
    if query_transform_weights is None:
        return candidate_queries
    if args.paper_m3_query_transform in {"label_ridge_to_meds", "label_ridge_sigmoid_to_meds"}:
        return candidate_metadata["meds"].astype(np.float32)
    return apply_label_ridge_query_transform(
        candidate_queries,
        query_transform_weights,
        args.paper_m3_query_transform,
    )


def candidate_global_med_prior(candidate_metadata, allow_train_or_own_history_only=False, train_patient_id_limit=None):
    if allow_train_or_own_history_only:
        if train_patient_id_limit is None:
            raise ValueError("train_patient_id_limit is required for train_plus_own_history retrieval")
        train_candidate_mask = candidate_metadata["patient_ids"] < train_patient_id_limit
        return candidate_metadata["meds"][train_candidate_mask].mean(axis=0).astype(np.float32)
    return candidate_metadata["meds"].mean(axis=0).astype(np.float32)


def apply_label_prior_to_retrieval_priors(priors, label_prior, global_prior):
    if args.paper_m3_label_prior_weight <= 0:
        return priors
    label_prior = np.asarray(label_prior, dtype=np.float32)
    if args.retrieval_prior_transform == "centered":
        label_prior = label_prior - global_prior.reshape(1, -1)
    elif args.retrieval_prior_transform != "positive":
        raise ValueError(f"Unsupported retrieval_prior_transform: {args.retrieval_prior_transform}")
    weight = float(args.paper_m3_label_prior_weight)
    return [
        prior + weight * torch.tensor(label_prior[row_idx], dtype=torch.float32, device=device)
        for row_idx, prior in enumerate(priors)
    ]


def build_learned_retrieval_priors(
    model,
    target_batches,
    candidate_batches,
    target_metadata,
    candidate_metadata,
    target_total,
    candidate_total,
    exclude_same_patient_current_and_future=False,
    allow_train_or_own_history_only=False,
    train_patient_id_limit=None,
):
    if args.retrieval_topk <= 0:
        raise ValueError("retrieval_topk must be positive when retrieval is enabled")
    target_queries = encode_patient_queries(model, target_batches, target_total)
    candidate_queries = encode_patient_queries(model, candidate_batches, candidate_total)
    priors = []
    chunk_size = 256
    for start in range(0, target_total, chunk_size):
        end = min(start + chunk_size, target_total)
        sim = np.matmul(target_queries[start:end], candidate_queries.T)
        if args.retrieval_mode == "learned_clinical":
            if not retrieval_requires_clinical:
                raise ValueError("learned_clinical retrieval requires clinical feature artifacts")
            clinical_sim = compute_clinical_similarity(
                retrieval_learned_target_inputs[target_batches["split_name"]],
                retrieval_learned_candidate_inputs,
                start,
                end,
            )
            sim = args.retrieval_alpha * sim + (1.0 - args.retrieval_alpha) * clinical_sim
        priors.extend(
            build_priors_from_similarity(
                sim=sim,
                target_metadata=target_metadata,
                candidate_metadata=candidate_metadata,
                row_offset=start,
                exclude_same_patient_current_and_future=exclude_same_patient_current_and_future,
                allow_train_or_own_history_only=allow_train_or_own_history_only,
                train_patient_id_limit=train_patient_id_limit,
            )
        )
    return priors


def build_learned_retrieval_context(
    model,
    target_batches,
    candidate_batches,
    target_metadata,
    candidate_metadata,
    target_total,
    candidate_total,
    exclude_same_patient_current_and_future=False,
    allow_train_or_own_history_only=False,
    train_patient_id_limit=None,
):
    if args.retrieval_topk <= 0:
        raise ValueError("retrieval_topk must be positive when retrieval context is enabled")
    target_queries = encode_patient_queries(model, target_batches, target_total)
    candidate_queries = encode_patient_queries(model, candidate_batches, candidate_total)
    retrieval_meds = []
    retrieval_scores = []
    retrieval_queries = []
    chunk_size = 256
    for start in range(0, target_total, chunk_size):
        end = min(start + chunk_size, target_total)
        sim = np.matmul(target_queries[start:end], candidate_queries.T)
        if args.retrieval_mode == "learned_clinical":
            if not retrieval_requires_clinical:
                raise ValueError("learned_clinical retrieval requires clinical feature artifacts")
            clinical_sim = compute_clinical_similarity(
                retrieval_learned_target_inputs[target_batches["split_name"]],
                retrieval_learned_candidate_inputs,
                start,
                end,
            )
            sim = args.retrieval_alpha * sim + (1.0 - args.retrieval_alpha) * clinical_sim
        chunk_meds, chunk_scores, chunk_queries = build_context_from_similarity(
            sim=sim,
            target_metadata=target_metadata,
            candidate_metadata=candidate_metadata,
            candidate_queries=candidate_queries,
            row_offset=start,
            exclude_same_patient_current_and_future=exclude_same_patient_current_and_future,
            allow_train_or_own_history_only=allow_train_or_own_history_only,
            train_patient_id_limit=train_patient_id_limit,
        )
        retrieval_meds.extend(chunk_meds)
        retrieval_scores.extend(chunk_scores)
        retrieval_queries.extend(chunk_queries)
    return retrieval_meds, retrieval_scores, retrieval_queries


def build_learned_inserted_visits(
    model,
    target_batches,
    candidate_batches,
    target_metadata,
    candidate_metadata,
    target_total,
    candidate_total,
    candidate_visits,
    candidate_clinical=None,
    exclude_same_patient_current_and_future=False,
    allow_train_or_own_history_only=False,
    train_patient_id_limit=None,
):
    if args.insert_visit_topk <= 0:
        return [[] for _ in range(target_total)]
    target_queries = encode_patient_queries(model, target_batches, target_total)
    candidate_queries = encode_patient_queries(model, candidate_batches, candidate_total)
    inserted = []
    chunk_size = 256
    for start in range(0, target_total, chunk_size):
        end = min(start + chunk_size, target_total)
        sim = np.matmul(target_queries[start:end], candidate_queries.T)
        if args.insert_visit_mode in ("learned_clinical", "dxproc_clinical_learned"):
            if not insert_requires_clinical:
                raise ValueError("learned_clinical inserted-visit retrieval requires clinical feature artifacts")
            clinical_sim = compute_clinical_similarity(
                insert_learned_target_inputs[target_batches["split_name"]],
                insert_learned_candidate_inputs,
                start,
                end,
            )
            sim = args.insert_visit_alpha * sim + (1.0 - args.insert_visit_alpha) * clinical_sim
        if args.insert_visit_mode.startswith("dxproc_"):
            candidate_filter_sim = compute_dxproc_similarity(
                insert_learned_target_inputs[target_batches["split_name"]],
                insert_learned_candidate_inputs,
                start,
                end,
            )
            if args.insert_visit_mode == "dxproc_clinical_learned":
                clinical_sim = compute_clinical_similarity(
                    insert_learned_target_inputs[target_batches["split_name"]],
                    insert_learned_candidate_inputs,
                    start,
                    end,
                )
                candidate_filter_sim = args.insert_visit_alpha * candidate_filter_sim + (1.0 - args.insert_visit_alpha) * clinical_sim
            candidate_pool_k = min(args.retrieval_topk, candidate_filter_sim.shape[1])
            candidate_pool_indexes = np.argpartition(candidate_filter_sim, -candidate_pool_k, axis=1)[:, -candidate_pool_k:]
            keep = np.zeros_like(sim, dtype=bool)
            row_indexes = np.arange(sim.shape[0])[:, None]
            keep[row_indexes, candidate_pool_indexes] = True
            sim = np.where(keep, sim, -np.inf)
        inserted.extend(
            build_inserted_from_similarity(
                sim=sim,
                target_metadata=target_metadata,
                candidate_metadata=candidate_metadata,
                candidate_visits=candidate_visits,
                candidate_clinical=candidate_clinical,
                row_offset=start,
                exclude_same_patient_current_and_future=exclude_same_patient_current_and_future,
                allow_train_or_own_history_only=allow_train_or_own_history_only,
                train_patient_id_limit=train_patient_id_limit,
            )
        )
    return inserted


def load_state_dict_file(path, device):
    checkpoint = seer_security.load_checkpoint(path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint at {path} does not contain a state dict")
    return checkpoint


def action_decode_requested():
    return (
        args.action_decode_mode != "none"
        or bool(args.use_trainable_action_fusion)
        or args.count_decode_mode != "none"
    )


def checkpoint_has_action_transition_heads(state_dict):
    required = (
        "action_transition_backbone.continue_head.weight",
        "action_transition_backbone.add_head.weight",
        "action_transition_backbone.remove_head.weight",
    )
    keys = set(state_dict.keys())
    for required_key in required:
        if required_key not in keys and ("module." + required_key) not in keys:
            return False
    return True


def checkpoint_has_action_fusion(state_dict, fusion_type="trainable_scalar"):
    keys = set(state_dict.keys())
    if fusion_type == "trainable_scalar":
        required = (
            "action_fusion.wc",
            "action_fusion.wa",
            "action_fusion.wr",
            "action_fusion.sc",
            "action_fusion.sa",
            "action_fusion.sr",
            "action_fusion.bc",
            "action_fusion.ba",
            "action_fusion.br",
        )
    elif fusion_type == "routed":
        required = (
            "action_fusion.drug_embedding.weight",
            "action_fusion.shared_expert.0.weight",
            "action_fusion.continue_expert.0.weight",
            "action_fusion.add_expert.0.weight",
            "action_fusion.remove_expert.0.weight",
            "action_fusion.router.0.weight",
        )
    elif fusion_type in ("ffn", "simple_ffn_routed"):
        required = (
            "action_fusion.drug_embedding.weight",
            "action_fusion.ffn.0.weight",
            "action_fusion.ffn.3.weight",
        )
    elif fusion_type in ("routed_residual", "aclr_residual"):
        required = (
            "action_fusion.base_fusion.wc",
            "action_fusion.drug_embedding.weight",
            "action_fusion.shared_expert.0.weight",
            "action_fusion.continue_expert.0.weight",
            "action_fusion.add_expert.0.weight",
            "action_fusion.remove_expert.0.weight",
            "action_fusion.router.0.weight",
            "action_fusion.residual_gate.weight",
        )
    else:
        required = ("action_fusion.",)
    for key in required:
        if key.endswith("."):
            if not any(existing.startswith(key) or existing.startswith("module." + key) for existing in keys):
                return False
            continue
        if key not in keys and ("module." + key) not in keys:
            return False
    return True


def require_action_head_checkpoint(state_dict, path):
    if action_decode_requested() and not checkpoint_has_action_transition_heads(state_dict):
        raise ValueError(
            "action-guided decode requires a checkpoint trained with stage-1 "
            f"action_transition_backbone heads; missing action head keys in {path}"
        )
    if args.use_trainable_action_fusion and not checkpoint_has_action_fusion(state_dict, args.action_fusion_type):
        raise ValueError(
            f"{args.action_fusion_type} action fusion eval requires a checkpoint with action_fusion parameters; "
            f"missing action_fusion keys in {path}"
        )
    if (
        args.count_decode_mode != "none"
        or args.count_aux_weight > 0
    ):
        keys = set(state_dict.keys())
        required = (
            "action_transition_backbone.total_count_head.0.weight",
            "action_transition_backbone.continue_count_head.0.weight",
            "action_transition_backbone.add_count_head.0.weight",
            "action_transition_backbone.remove_count_head.0.weight",
        )
        missing = [
            key
            for key in required
            if key not in keys and ("module." + key) not in keys
        ]
        if missing:
            raise ValueError(
                "count-aware decode requires a checkpoint trained with action count heads; "
                f"missing count head keys in {path}: {missing[:2]}"
            )


def filter_state_dict_by_shape(checkpoint_state, model_state):
    filtered = {}
    skipped = {}
    for key, value in checkpoint_state.items():
        target = model_state.get(key)
        if target is None:
            filtered[key] = value
            continue
        if tuple(value.shape) == tuple(target.shape):
            filtered[key] = value
        else:
            skipped[key] = {
                "checkpoint_shape": tuple(value.shape),
                "model_shape": tuple(target.shape),
            }
    return filtered, skipped


def stable_digest(payload):
    return content_digest(payload)


def checkpoint_fingerprint(path):
    checkpoint_path = Path(path)
    stat = checkpoint_path.stat()
    return {
        "path": str(checkpoint_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_digest(checkpoint_path),
    }


def effective_paper_m3_topk():
    return args.paper_m3_topk if args.paper_m3_topk > 0 else args.insert_visit_topk


def paper_m3_cache_path(split_name, target_patient_offsets, candidate_patient_offsets, candidate_start_visit):
    if args.paper_m3_no_cache:
        return None
    payload = {
        "input_fingerprint": dataset_cache_fingerprint,
        "dataset": dataset,
        "split": split_name,
        "checkpoint": checkpoint_fingerprint(args.paper_m3_checkpoint),
        "topk": effective_paper_m3_topk(),
        "insert_visit_topk": args.insert_visit_topk,
        "paper_m3_topk": args.paper_m3_topk,
        "retrieval_candidate_scope": args.retrieval_candidate_scope,
        "target_start_visit": target_start_visit,
        "candidate_start_visit": candidate_start_visit,
        "target_patient_offsets": list(target_patient_offsets),
        "candidate_patient_offsets": list(candidate_patient_offsets),
        "clinical_feature_mode": args.clinical_feature_mode,
        "paper_m3_clinical_weight": args.paper_m3_clinical_weight,
        "paper_m3_clinical_alpha": args.paper_m3_clinical_alpha,
        "paper_m3_dxproc_weight": args.paper_m3_dxproc_weight,
        "paper_m3_history_overlap_weight": args.paper_m3_history_overlap_weight,
        "paper_m3_query_similarity": args.paper_m3_query_similarity,
        "paper_m3_rbf_gamma": args.paper_m3_rbf_gamma,
        "paper_m3_query_transform": args.paper_m3_query_transform,
        "paper_m3_query_transform_ridge": args.paper_m3_query_transform_ridge,
        "paper_m3_label_prior_weight": args.paper_m3_label_prior_weight,
        "retrieval_prior_transform": args.retrieval_prior_transform,
        "artifact_version": "paper_m3_artifacts_with_query_context_dxproc_v4",
    }
    cache_dir = Path(args.paper_m3_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{split_name}-{stable_digest(payload)}.pkl"


def cached_paper_m3_artifacts(cache_path, builder):
    if cache_path is not None and cache_path.exists():
        cached = seer_security.load_pickle(open(cache_path, "rb"))
        retrieval_context = cached.get("retrieval_context") if isinstance(cached, dict) else None
        if (
            isinstance(cached, dict)
            and "inserted_visits" in cached
            and "retrieval_priors" in cached
            and isinstance(retrieval_context, (tuple, list))
            and len(retrieval_context) == 3
            and retrieval_context[2] is not None
        ):
            return cached
    artifacts = builder()
    if cache_path is not None:
        dill.dump(artifacts, open(cache_path, "wb"))
    return artifacts


def flatten_clinical_similarity_inputs(clinical_payload, start_visit=None):
    flattened = flatten_target_clinical_artifacts(clinical_payload, start_visit=start_visit)
    if flattened is None:
        return None
    values = np.asarray(flattened["clinical_features"], dtype=np.float32)
    masks = np.asarray(flattened["clinical_feature_masks"], dtype=np.float32)
    return values, masks


def compute_clinical_similarity_from_flattened(target_clinical_inputs, candidate_clinical_inputs, start, end):
    if target_clinical_inputs is None or candidate_clinical_inputs is None:
        return None
    target_values, target_masks = target_clinical_inputs
    candidate_values, candidate_masks = candidate_clinical_inputs
    query_values = target_values[start:end]
    query_masks = target_masks[start:end]
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


def compute_history_med_overlap_similarity(target_metadata, candidate_metadata, start, end):
    target_history = target_metadata.get("history_meds")
    candidate_meds = candidate_metadata.get("meds")
    if target_history is None or candidate_meds is None:
        return None
    target_history = target_history[start:end]
    intersections = np.matmul(target_history, candidate_meds.T)
    target_counts = target_history.sum(axis=1, keepdims=True)
    candidate_counts = candidate_meds.sum(axis=1, keepdims=True).T
    unions = target_counts + candidate_counts - intersections
    return np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float32),
        where=unions > 0,
    )


def cached_directed_weighted_ehr_adj(records, med_voc_size, patient_offsets):
    cache_dir = Path(args.ehr_graph_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_fingerprint": dataset_cache_fingerprint,
        "records": records,
        "dataset": dataset,
        "med_voc_size": med_voc_size,
        "patient_offsets": list(patient_offsets),
        "version": "directed_weighted_content_v2",
    }
    cache_path = cache_dir / f"ehr_graph_plus-{stable_digest(payload)}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return seer_security.load_pickle(handle)
    graph = create_directed_weighted_ehr_adj(records, med_voc_size)
    with cache_path.open("wb") as handle:
        dill.dump(graph, handle)
    return graph


def clone_phi_model_from(model, checkpoint_path, device):
    phi_model = copy.deepcopy(model).to(device)
    phi_model.load_state_dict(load_state_dict_file(checkpoint_path, device), strict=False)
    phi_model.eval()
    for parameter in phi_model.parameters():
        parameter.requires_grad_(False)
    return phi_model


def build_paper_m3_artifacts(
    phi_model,
    target_batches,
    candidate_batches,
    target_metadata,
    candidate_metadata,
    target_total,
    candidate_total,
    candidate_visits,
    target_retrieval_inputs=None,
    candidate_retrieval_inputs=None,
    candidate_clinical=None,
    target_clinical_inputs=None,
    candidate_clinical_inputs=None,
    exclude_same_patient_current_and_future=True,
    allow_train_or_own_history_only=False,
    train_patient_id_limit=None,
    query_transform_weights=None,
):
    paper_m3_topk = effective_paper_m3_topk()
    target_queries = encode_patient_queries_with_normalization(phi_model, target_batches, target_total, normalize=False)
    candidate_queries = encode_patient_queries_with_normalization(phi_model, candidate_batches, candidate_total, normalize=False)
    target_similarity_queries = target_queries
    candidate_similarity_queries = candidate_queries
    if query_transform_weights is not None:
        target_similarity_queries = apply_label_ridge_query_transform(
            target_queries,
            query_transform_weights,
            args.paper_m3_query_transform,
        )
        candidate_similarity_queries = transform_candidate_queries_for_similarity(
            candidate_queries,
            candidate_metadata,
            query_transform_weights,
        )
    inserted = []
    priors = []
    retrieval_meds = []
    retrieval_scores = []
    retrieval_queries = []
    label_global_prior = None
    if args.paper_m3_label_prior_weight > 0:
        label_global_prior = candidate_global_med_prior(
            candidate_metadata,
            allow_train_or_own_history_only=allow_train_or_own_history_only,
            train_patient_id_limit=train_patient_id_limit,
        )
    chunk_size = 256
    for start in range(0, target_total, chunk_size):
        end = min(start + chunk_size, target_total)
        sim = compute_query_similarity(
            target_similarity_queries[start:end],
            candidate_similarity_queries,
            args.paper_m3_query_similarity,
            rbf_gamma=args.paper_m3_rbf_gamma,
        )
        if args.paper_m3_dxproc_weight > 0:
            if target_retrieval_inputs is None or candidate_retrieval_inputs is None:
                raise ValueError("--paper_m3_dxproc_weight requires target/candidate dxproc retrieval inputs")
            dxproc_sim = compute_dxproc_similarity(
                target_retrieval_inputs,
                candidate_retrieval_inputs,
                start,
                end,
            )
            sim = sim + args.paper_m3_dxproc_weight * dxproc_sim
        if args.paper_m3_clinical_weight > 0:
            clinical_sim = compute_clinical_similarity_from_flattened(
                target_clinical_inputs,
                candidate_clinical_inputs,
                start,
                end,
            )
            if clinical_sim is None:
                raise ValueError("--paper_m3_clinical_weight requires clinical feature artifacts")
            alpha = args.paper_m3_clinical_alpha
            sim = alpha * sim + (1.0 - alpha) * args.paper_m3_clinical_weight * clinical_sim
        if args.paper_m3_history_overlap_weight > 0:
            history_overlap_sim = compute_history_med_overlap_similarity(
                target_metadata,
                candidate_metadata,
                start,
                end,
            )
            if history_overlap_sim is None:
                raise ValueError("--paper_m3_history_overlap_weight requires target/candidate medication metadata")
            sim = sim + args.paper_m3_history_overlap_weight * history_overlap_sim
        inserted.extend(
            build_inserted_from_similarity(
                sim=sim.copy(),
                target_metadata=target_metadata,
                candidate_metadata=candidate_metadata,
                candidate_visits=candidate_visits,
                candidate_clinical=candidate_clinical,
                row_offset=start,
                exclude_same_patient_current_and_future=exclude_same_patient_current_and_future,
                allow_train_or_own_history_only=allow_train_or_own_history_only,
                train_patient_id_limit=train_patient_id_limit,
            )
        )
        chunk_priors = build_priors_from_similarity(
            sim=sim,
            target_metadata=target_metadata,
            candidate_metadata=candidate_metadata,
            row_offset=start,
            exclude_same_patient_current_and_future=exclude_same_patient_current_and_future,
            allow_train_or_own_history_only=allow_train_or_own_history_only,
            train_patient_id_limit=train_patient_id_limit,
            topk=paper_m3_topk,
        )
        if args.paper_m3_label_prior_weight > 0:
            chunk_priors = apply_label_prior_to_retrieval_priors(
                chunk_priors,
                target_similarity_queries[start:end],
                label_global_prior,
            )
        priors.extend(chunk_priors)
        chunk_meds, chunk_scores, chunk_queries = build_med_context_from_similarity(
            sim=sim.copy(),
            target_metadata=target_metadata,
            candidate_metadata=candidate_metadata,
            candidate_queries=candidate_queries,
            row_offset=start,
            exclude_same_patient_current_and_future=exclude_same_patient_current_and_future,
            allow_train_or_own_history_only=allow_train_or_own_history_only,
            train_patient_id_limit=train_patient_id_limit,
            topk=paper_m3_topk,
        )
        retrieval_meds.extend(chunk_meds)
        retrieval_scores.extend(chunk_scores)
        retrieval_queries.extend(chunk_queries)
    return {
        "inserted_visits": inserted,
        "retrieval_priors": priors,
        "retrieval_context": (retrieval_meds, retrieval_scores, retrieval_queries),
    }


def clear_retrieval_context_batches(batches):
    batches["retrieval_med_batches"] = None
    batches["retrieval_score_batches"] = None
    batches["retrieval_query_batches"] = None


def set_retrieval_context_batches(batches, retrieval_context, batch_size):
    retrieval_meds, retrieval_scores, retrieval_queries = retrieval_context[:3]
    batches["retrieval_med_batches"] = tensors_to_fixed_batches(
        [retrieval_meds[i] for batch in batches["sample_index_batches"] for i in batch],
        batch_size,
    )
    batches["retrieval_score_batches"] = tensors_to_fixed_batches(
        [retrieval_scores[i] for batch in batches["sample_index_batches"] for i in batch],
        batch_size,
    )
    if retrieval_queries is None:
        batches["retrieval_query_batches"] = None
    else:
        batches["retrieval_query_batches"] = tensors_to_fixed_batches(
            [retrieval_queries[i] for batch in batches["sample_index_batches"] for i in batch],
            batch_size,
        )
    if len(retrieval_context) > 3:
        set_treatment_episode_context_batches(batches, retrieval_context, batch_size)


def set_retrieval_prior_batches(batches, retrieval_priors, batch_size):
    if retrieval_priors is None:
        batches["retrieval_prior_batches"] = None
        return
    batches["retrieval_prior_batches"] = tensors_to_fixed_batches(
        [retrieval_priors[i] for batch in batches["sample_index_batches"] for i in batch],
        batch_size,
    )


def clear_treatment_episode_context_batches(batches):
    batches["treatment_episode_score_batches"] = None
    batches["treatment_episode_experience_batches"] = None


def set_treatment_episode_context_batches(batches, retrieval_context, batch_size):
    _, retrieval_scores, _, treatment_episode_experiences = retrieval_context
    batches["treatment_episode_score_batches"] = tensors_to_fixed_batches(
        [retrieval_scores[i] for batch in batches["sample_index_batches"] for i in batch],
        batch_size,
    )
    if treatment_episode_experiences is None:
        batches["treatment_episode_experience_batches"] = None
    else:
        batches["treatment_episode_experience_batches"] = tensors_to_fixed_batches(
            [treatment_episode_experiences[i] for batch in batches["sample_index_batches"] for i in batch],
            batch_size,
        )


def build_seq_decoder_targets(label_batch):
    batch_size, med_voc_size = label_batch.shape
    end_token = med_voc_size + 1
    max_len = int(label_batch.sum(dim=1).max().item()) + 1
    decoder_inputs = torch.full((batch_size, max_len), end_token, dtype=torch.long, device=label_batch.device)
    decoder_targets = torch.full((batch_size, max_len), -100, dtype=torch.long, device=label_batch.device)
    decoder_inputs[:, 0] = med_voc_size
    for row_idx in range(batch_size):
        med_indexes = torch.nonzero(label_batch[row_idx] > 0, as_tuple=False).flatten().long()
        if seq_decoder_med_rank is not None and med_indexes.numel() > 1:
            med_rank = seq_decoder_med_rank.to(label_batch.device)
            med_indexes = med_indexes[torch.argsort(med_rank[med_indexes])]
        med_count = int(med_indexes.numel())
        if med_count > 0:
            decoder_targets[row_idx, :med_count] = med_indexes
            if med_count < max_len:
                decoder_inputs[row_idx, 1:med_count + 1] = med_indexes
        decoder_targets[row_idx, med_count] = end_token
    return decoder_inputs, decoder_targets


def apply_history_prior(med_logits, med_history):
    if args.history_prior_weight > 0:
        med_history = med_history.to(device=med_logits.device, dtype=med_logits.dtype)
        med_history = med_history[:, 1:, :]
        if med_history.shape[1] == 0:
            history_prior = torch.zeros_like(med_logits)
        elif args.history_prior_mode == "binary":
            history_prior = (med_history.sum(dim=1) > 0).to(dtype=med_logits.dtype)
        elif args.history_prior_mode == "recency":
            recency_steps = torch.arange(med_history.shape[1], device=med_logits.device, dtype=med_logits.dtype)
            recency_weights = torch.pow(
                torch.full_like(recency_steps, args.history_prior_decay),
                recency_steps,
            ).view(1, -1, 1)
            history_prior = ((med_history > 0).to(dtype=med_logits.dtype) * recency_weights).max(dim=1).values
        else:
            raise ValueError(f"Unsupported history_prior_mode: {args.history_prior_mode}")
        med_logits = med_logits + args.history_prior_weight * history_prior
    return med_logits


def apply_retrieval_prior(med_logits, batches, batch_id):
    retrieval_batches = batches.get("retrieval_prior_batches")
    if retrieval_batches is not None and args.retrieval_weight > 0:
        med_logits = med_logits + args.retrieval_weight * retrieval_batches[batch_id].to(med_logits.device)
    history_meds = batches.get("own_med_batches") or batches["med_batches"]
    return apply_history_prior(med_logits, history_meds[batch_id])


def action_fused_logits_for_batch(output, med_logits, batches, batch_id):
    fused_logits = output.get("fused_logits")
    if fused_logits is None:
        if args.use_trainable_action_fusion or fusion_objective_enabled:
            raise ValueError("trainable action fusion requires fused_logits from SEER")
        return None
    return fused_logits + (med_logits - output["med_logits"])


def action_fusion_aux_loss(output, y_gt, med_logits, batches, batch_id):
    if not fusion_objective_enabled:
        return med_logits.new_tensor(0.0)
    fused_logits = action_fused_logits_for_batch(output, med_logits, batches, batch_id)
    action_fusion_logging = output.get("action_fusion_logging") or {}
    routing_loss = fused_logits.new_tensor(0.0)
    for key in (
        "routing_entropy_loss",
        "routed_residual_load_balance_loss",
        "routed_residual_l2_loss",
        "routed_residual_add_penalty_loss",
    ):
        value = action_fusion_logging.get(key)
        if value is not None:
            routing_loss = routing_loss + value.to(device=fused_logits.device, dtype=fused_logits.dtype)
    if fusion_v2_objective_enabled:
        fused_prob = torch.sigmoid(fused_logits)
        loss = routing_loss
        if args.fusion_bce_weight > 0:
            loss = loss + args.fusion_bce_weight * F.binary_cross_entropy_with_logits(fused_logits, y_gt)
        if args.fusion_soft_jaccard_weight > 0:
            loss = loss + args.fusion_soft_jaccard_weight * soft_jaccard_loss_from_prob(fused_prob, y_gt)
        if args.fusion_count_anchor_weight > 0:
            loss = loss + args.fusion_count_anchor_weight * count_anchor_loss_from_prob(fused_prob, y_gt)
        if args.fusion_ddi_soft_weight > 0:
            loss = loss + args.fusion_ddi_soft_weight * soft_ddi_penalty_from_prob(
                fused_prob,
                args.fusion_ddi_soft_weight,
            )
        return loss
    return args.action_fusion_loss_weight * F.binary_cross_entropy_with_logits(fused_logits, y_gt) + routing_loss


def main_med_scorer_regularization(output, med_logits):
    delta = output.get("main_med_scorer_delta")
    if delta is None:
        return med_logits.new_tensor(0.0)
    reg = med_logits.new_tensor(0.0)
    if args.main_med_scorer_delta_l2_weight > 0:
        reg = reg + args.main_med_scorer_delta_l2_weight * delta.pow(2).mean()
    if args.main_med_scorer_count_kl_weight > 0:
        base_logits = (med_logits - delta).detach()
        base_count = torch.sigmoid(base_logits).sum(dim=-1)
        current_count = torch.sigmoid(med_logits).sum(dim=-1)
        reg = reg + args.main_med_scorer_count_kl_weight * F.smooth_l1_loss(current_count, base_count)
    return reg


def apply_retrieval_copy_prob(y_pred_prob, batches, batch_id):
    retrieval_batches = batches.get("retrieval_prior_batches")
    if retrieval_batches is None or args.retrieval_copy_mode != "fixed_prob" or args.retrieval_copy_weight <= 0:
        return y_pred_prob
    retrieval_prior = retrieval_batches[batch_id].to(device=y_pred_prob.device, dtype=y_pred_prob.dtype)
    if args.retrieval_prior_transform == "centered":
        retrieval_copy_prob = torch.sigmoid(retrieval_prior)
    else:
        retrieval_copy_prob = retrieval_prior.clamp(0.0, 1.0)
    return (
        (1.0 - args.retrieval_copy_weight) * y_pred_prob
        + args.retrieval_copy_weight * retrieval_copy_prob
    )


def action_logits_from_output(output):
    return output.get("continue_logits"), output.get("add_logits"), output.get("remove_logits")


def action_score_fusion_for_batch(output, med_logits, batches, batch_id):
    continue_logits, add_logits, remove_logits = action_logits_from_output(output)
    previous = previous_meds_for_batch(batches, batch_id, med_logits)
    return action_score_fusion_logits(
        med_logits,
        continue_logits,
        add_logits,
        remove_logits,
        previous,
        continue_weight=args.action_decode_continue_weight,
        add_weight=args.action_decode_add_weight,
        remove_weight=args.action_decode_remove_weight,
    )


def direct_action_prediction_for_batch(output, med_logits, batches, batch_id):
    continue_logits, add_logits, remove_logits = action_logits_from_output(output)
    previous = previous_meds_for_batch(batches, batch_id, med_logits)
    return direct_action_decode(
        continue_logits,
        add_logits,
        remove_logits,
        previous,
        threshold=args.action_decode_threshold,
    )


def direct_action_scores_for_batch(output, med_logits, batches, batch_id):
    continue_logits, add_logits, remove_logits = action_logits_from_output(output)
    previous = previous_meds_for_batch(batches, batch_id, med_logits)
    return direct_action_scores(
        continue_logits,
        add_logits,
        remove_logits,
        previous,
    )


def action_count_predictions_from_output(output):
    return {
        "total": output.get("total_count_pred"),
        "continue": output.get("continue_count_pred"),
        "add": output.get("add_count_pred"),
        "remove": output.get("remove_count_pred"),
    }


def metric_value(metric):
    metric_indexes = {"ja": 1, "prauc": 2, "f1": 5}
    return metric[metric_indexes[args.best_metric]]


def metric_value_without_epoch(metric, metric_name):
    metric_indexes = {"ja": 0, "prauc": 1, "f1": 4}
    return metric[metric_indexes[metric_name]]


def threshold_selection_metric():
    if args.threshold_metric:
        return args.threshold_metric
    if args.best_metric in {"ja", "f1"}:
        return args.best_metric
    return "ja"


def threshold_grid():
    if not (args.threshold_search or args.decode_search):
        return [args.eval_threshold]
    grid = np.arange(args.threshold_min, args.threshold_max + args.threshold_step / 2, args.threshold_step)
    return [round(float(threshold), 6) for threshold in grid]


def decode_count_limits(voc_size):
    min_count = max(0, args.decode_min_count)
    max_count = args.decode_max_count if args.decode_max_count > 0 else voc_size
    max_count = max(min_count, min(max_count, voc_size))
    return min_count, max_count


def float_grid(start, stop, step):
    grid = np.arange(start, stop + step / 2, step)
    return [round(float(value), 6) for value in grid]


def retrieval_count_mode_grid():
    return list(dict.fromkeys(requested_retrieval_count_modes))


def seq_end_bias_grid():
    if args.seq_decoder_end_bias_search:
        return float_grid(
            args.seq_decoder_end_bias_min,
            args.seq_decoder_end_bias_max,
            args.seq_decoder_end_bias_step,
        )
    return [args.seq_decoder_end_bias]


def decode_configs():
    if args.decode_config_json:
        return load_decode_config_json(args.decode_config_json)
    if args.threshold_search and not args.decode_search:
        return [{"mode": "threshold", "threshold": threshold} for threshold in threshold_grid()]
    if not args.decode_search:
        return [{
            "mode": args.decode_mode,
            "threshold": args.eval_threshold,
            "thresholds": None,
            "topk": args.decode_topk,
            "ratio": args.decode_count_ratio,
            "bias": args.decode_count_bias,
            "weight": (
                args.retrieval_rerank_weight
                if args.decode_mode.startswith("retrieval_rank_")
                else args.history_rerank_weight
            ),
            "history_weight": args.history_retrieval_history_weight,
            "retrieval_weight": args.history_retrieval_retrieval_weight,
            "seq_weight": args.history_retrieval_seq_weight,
            "freq_weight": args.history_retrieval_freq_weight,
            "ehr_graph_weight": args.history_retrieval_ehr_graph_weight,
            "ddi_graph_weight": args.history_retrieval_ddi_graph_weight,
            "graph_source": args.history_retrieval_graph_source,
            "retrieval_count_mode": retrieval_count_mode_grid()[0],
        }]

    allowed_modes = set(args.decode_search_modes.split(","))
    configs = []
    if "threshold" in allowed_modes:
        configs.extend({"mode": "threshold", "threshold": threshold} for threshold in threshold_grid())
    if "per_med_threshold" in allowed_modes:
        configs.append({"mode": "per_med_threshold", "thresholds": None})
    if "history_threshold" in allowed_modes:
        for threshold in threshold_grid():
            for delta in float_grid(
                args.history_threshold_delta_min,
                args.history_threshold_delta_max,
                args.history_threshold_delta_step,
            ):
                configs.append({
                    "mode": "history_threshold",
                    "threshold": threshold,
                    "history_threshold": min(max(threshold + delta, 0.0), 1.0),
                })
    history_rank_modes = {
        "history_rank_fixed_topk",
        "history_rank_threshold_count",
        "history_rank_expected_count",
        "history_rank_history_count",
        "history_rank_predicted_count",
    }
    retrieval_rank_modes = {
        "retrieval_rank_fixed_topk",
        "retrieval_rank_threshold_count",
        "retrieval_rank_expected_count",
        "retrieval_rank_history_count",
        "retrieval_rank_predicted_count",
        "history_retrieval_freq_rank_expected_count",
    }
    history_retrieval_rank_modes = {
        "history_retrieval_rank_fixed_topk",
        "history_retrieval_rank_threshold_count",
        "history_retrieval_rank_expected_count",
        "history_retrieval_rank_history_count",
        "history_retrieval_rank_predicted_count",
    }
    history_retrieval_seq_rank_modes = {
        "history_retrieval_seq_rank_fixed_topk",
        "history_retrieval_seq_rank_threshold_count",
        "history_retrieval_seq_rank_expected_count",
        "history_retrieval_seq_rank_history_count",
        "history_retrieval_seq_rank_predicted_count",
    }
    history_retrieval_freq_rank_modes = {
        "history_retrieval_freq_rank_fixed_topk",
        "history_retrieval_freq_rank_threshold_count",
        "history_retrieval_freq_rank_expected_count",
        "history_retrieval_freq_rank_expected_jaccard_count",
        "history_retrieval_freq_rank_history_count",
        "history_retrieval_freq_rank_predicted_count",
    }
    history_retrieval_freq_ddi_rank_modes = {
        "history_retrieval_freq_ddi_rank_expected_count",
    }
    history_retrieval_freq_graph_rank_modes = {
        "history_retrieval_freq_graph_rank_expected_count",
        "history_retrieval_freq_graph_rank_expected_jaccard_count",
        "history_retrieval_freq_graph_rank_count_pair_mix",
        "history_retrieval_freq_graph_rank_predicted_count",
        "history_retrieval_freq_graph_rank_retrieval_count",
    }
    history_rank_weights = float_grid(
        args.history_rerank_weight_min,
        args.history_rerank_weight_max,
        args.history_rerank_weight_step,
    )
    retrieval_rank_weights = float_grid(
        args.retrieval_rerank_weight_min,
        args.retrieval_rerank_weight_max,
        args.retrieval_rerank_weight_step,
    )
    history_retrieval_history_weights = float_grid(
        args.history_retrieval_history_weight_min,
        args.history_retrieval_history_weight_max,
        args.history_retrieval_history_weight_step,
    )
    history_retrieval_retrieval_weights = float_grid(
        args.history_retrieval_retrieval_weight_min,
        args.history_retrieval_retrieval_weight_max,
        args.history_retrieval_retrieval_weight_step,
    )
    history_retrieval_seq_weights = float_grid(
        args.history_retrieval_seq_weight_min,
        args.history_retrieval_seq_weight_max,
        args.history_retrieval_seq_weight_step,
    )
    history_retrieval_freq_weights = float_grid(
        args.history_retrieval_freq_weight_min,
        args.history_retrieval_freq_weight_max,
        args.history_retrieval_freq_weight_step,
    )
    history_retrieval_ddi_penalties = float_grid(
        args.history_retrieval_ddi_penalty_min,
        args.history_retrieval_ddi_penalty_max,
        args.history_retrieval_ddi_penalty_step,
    )
    history_retrieval_ehr_graph_weights = float_grid(
        args.history_retrieval_ehr_graph_weight_min,
        args.history_retrieval_ehr_graph_weight_max,
        args.history_retrieval_ehr_graph_weight_step,
    )
    history_retrieval_ddi_graph_weights = float_grid(
        args.history_retrieval_ddi_graph_weight_min,
        args.history_retrieval_ddi_graph_weight_max,
        args.history_retrieval_ddi_graph_weight_step,
    )
    rank_count_modes = {
        "fixed_topk",
        "expected_count",
        "history_count",
        "predicted_count",
    }
    for mode in sorted(history_rank_modes & allowed_modes):
        count_mode = mode.removeprefix("history_rank_")
        if count_mode == "predicted_count" and args.med_count_mode != "aux":
            continue
        for weight in history_rank_weights:
            if count_mode == "fixed_topk":
                for topk in range(args.decode_topk_min, args.decode_topk_max + 1, args.decode_topk_step):
                    configs.append({"mode": mode, "weight": weight, "topk": topk})
            elif count_mode == "threshold_count":
                for threshold in threshold_grid():
                    for ratio in float_grid(args.decode_count_ratio_min, args.decode_count_ratio_max, args.decode_count_ratio_step):
                        for bias in float_grid(args.decode_count_bias_min, args.decode_count_bias_max, args.decode_count_bias_step):
                            configs.append({
                                "mode": mode,
                                "weight": weight,
                                "threshold": threshold,
                                "ratio": ratio,
                                "bias": bias,
                            })
            elif count_mode in rank_count_modes:
                for ratio in float_grid(args.decode_count_ratio_min, args.decode_count_ratio_max, args.decode_count_ratio_step):
                    for bias in float_grid(args.decode_count_bias_min, args.decode_count_bias_max, args.decode_count_bias_step):
                        configs.append({"mode": mode, "weight": weight, "ratio": ratio, "bias": bias})
    for mode in sorted(retrieval_rank_modes & allowed_modes):
        count_mode = mode.removeprefix("retrieval_rank_")
        if count_mode == "predicted_count" and args.med_count_mode != "aux":
            continue
        for weight in retrieval_rank_weights:
            if count_mode == "fixed_topk":
                for topk in range(args.decode_topk_min, args.decode_topk_max + 1, args.decode_topk_step):
                    configs.append({"mode": mode, "weight": weight, "topk": topk})
            elif count_mode == "threshold_count":
                for threshold in threshold_grid():
                    for ratio in float_grid(args.decode_count_ratio_min, args.decode_count_ratio_max, args.decode_count_ratio_step):
                        for bias in float_grid(args.decode_count_bias_min, args.decode_count_bias_max, args.decode_count_bias_step):
                            configs.append({
                                "mode": mode,
                                "weight": weight,
                                "threshold": threshold,
                                "ratio": ratio,
                                "bias": bias,
                            })
            elif count_mode in rank_count_modes:
                for ratio in float_grid(args.decode_count_ratio_min, args.decode_count_ratio_max, args.decode_count_ratio_step):
                    for bias in float_grid(args.decode_count_bias_min, args.decode_count_bias_max, args.decode_count_bias_step):
                        configs.append({"mode": mode, "weight": weight, "ratio": ratio, "bias": bias})
    for mode in sorted(history_retrieval_rank_modes & allowed_modes):
        count_mode = mode.removeprefix("history_retrieval_rank_")
        if count_mode == "predicted_count" and args.med_count_mode != "aux":
            continue
        for history_weight in history_retrieval_history_weights:
            for retrieval_weight in history_retrieval_retrieval_weights:
                if count_mode == "fixed_topk":
                    for topk in range(args.decode_topk_min, args.decode_topk_max + 1, args.decode_topk_step):
                        configs.append({
                            "mode": mode,
                            "history_weight": history_weight,
                            "retrieval_weight": retrieval_weight,
                            "topk": topk,
                        })
                elif count_mode == "threshold_count":
                    for threshold in threshold_grid():
                        for ratio in float_grid(args.decode_count_ratio_min, args.decode_count_ratio_max, args.decode_count_ratio_step):
                            for bias in float_grid(args.decode_count_bias_min, args.decode_count_bias_max, args.decode_count_bias_step):
                                configs.append({
                                    "mode": mode,
                                    "history_weight": history_weight,
                                    "retrieval_weight": retrieval_weight,
                                    "threshold": threshold,
                                    "ratio": ratio,
                                    "bias": bias,
                                })
                elif count_mode in rank_count_modes:
                    for ratio in float_grid(args.decode_count_ratio_min, args.decode_count_ratio_max, args.decode_count_ratio_step):
                        for bias in float_grid(args.decode_count_bias_min, args.decode_count_bias_max, args.decode_count_bias_step):
                            configs.append({
                                "mode": mode,
                                "history_weight": history_weight,
                                "retrieval_weight": retrieval_weight,
                                "ratio": ratio,
                                "bias": bias,
                            })
    for mode in sorted(history_retrieval_seq_rank_modes & allowed_modes):
        if not seq_decoder_enabled():
            continue
        count_mode = mode.removeprefix("history_retrieval_seq_rank_")
        if count_mode == "predicted_count" and args.med_count_mode != "aux":
            continue
        for history_weight in history_retrieval_history_weights:
            for retrieval_weight in history_retrieval_retrieval_weights:
                for seq_weight in history_retrieval_seq_weights:
                    if count_mode == "fixed_topk":
                        for topk in range(args.decode_topk_min, args.decode_topk_max + 1, args.decode_topk_step):
                            configs.append({
                                "mode": mode,
                                "history_weight": history_weight,
                                "retrieval_weight": retrieval_weight,
                                "seq_weight": seq_weight,
                                "topk": topk,
                            })
                    elif count_mode == "threshold_count":
                        for threshold in threshold_grid():
                            for ratio in float_grid(args.decode_count_ratio_min, args.decode_count_ratio_max, args.decode_count_ratio_step):
                                for bias in float_grid(args.decode_count_bias_min, args.decode_count_bias_max, args.decode_count_bias_step):
                                    configs.append({
                                        "mode": mode,
                                        "history_weight": history_weight,
                                        "retrieval_weight": retrieval_weight,
                                        "seq_weight": seq_weight,
                                        "threshold": threshold,
                                        "ratio": ratio,
                                        "bias": bias,
                                    })
                    elif count_mode == "expected_jaccard_count":
                        configs.append({
                            "mode": mode,
                            "history_weight": history_weight,
                            "retrieval_weight": retrieval_weight,
                            "freq_weight": freq_weight,
                        })
                    elif count_mode in rank_count_modes:
                        for ratio in float_grid(args.decode_count_ratio_min, args.decode_count_ratio_max, args.decode_count_ratio_step):
                            for bias in float_grid(args.decode_count_bias_min, args.decode_count_bias_max, args.decode_count_bias_step):
                                configs.append({
                                    "mode": mode,
                                    "history_weight": history_weight,
                                    "retrieval_weight": retrieval_weight,
                                    "seq_weight": seq_weight,
                                    "ratio": ratio,
                                    "bias": bias,
                                })
    for mode in sorted(history_retrieval_freq_rank_modes & allowed_modes):
        count_mode = mode.removeprefix("history_retrieval_freq_rank_")
        if count_mode == "predicted_count" and args.med_count_mode != "aux":
            continue
        for history_weight in history_retrieval_history_weights:
            for retrieval_weight in history_retrieval_retrieval_weights:
                for freq_weight in history_retrieval_freq_weights:
                    if count_mode == "fixed_topk":
                        for topk in range(args.decode_topk_min, args.decode_topk_max + 1, args.decode_topk_step):
                            configs.append({
                                "mode": mode,
                                "history_weight": history_weight,
                                "retrieval_weight": retrieval_weight,
                                "freq_weight": freq_weight,
                                "topk": topk,
                            })
                    elif count_mode == "threshold_count":
                        for threshold in threshold_grid():
                            for ratio in float_grid(args.decode_count_ratio_min, args.decode_count_ratio_max, args.decode_count_ratio_step):
                                for bias in float_grid(args.decode_count_bias_min, args.decode_count_bias_max, args.decode_count_bias_step):
                                    configs.append({
                                        "mode": mode,
                                        "history_weight": history_weight,
                                        "retrieval_weight": retrieval_weight,
                                        "freq_weight": freq_weight,
                                        "threshold": threshold,
                                        "ratio": ratio,
                                        "bias": bias,
                                    })
                    elif count_mode in rank_count_modes:
                        for ratio in float_grid(args.decode_count_ratio_min, args.decode_count_ratio_max, args.decode_count_ratio_step):
                            for bias in float_grid(args.decode_count_bias_min, args.decode_count_bias_max, args.decode_count_bias_step):
                                configs.append({
                                    "mode": mode,
                                    "history_weight": history_weight,
                                    "retrieval_weight": retrieval_weight,
                                    "freq_weight": freq_weight,
                                    "ratio": ratio,
                                    "bias": bias,
                                })
    for mode in sorted(history_retrieval_freq_ddi_rank_modes & allowed_modes):
        count_mode = mode.removeprefix("history_retrieval_freq_ddi_rank_")
        for history_weight in history_retrieval_history_weights:
            for retrieval_weight in history_retrieval_retrieval_weights:
                for freq_weight in history_retrieval_freq_weights:
                    for ddi_penalty in history_retrieval_ddi_penalties:
                        if count_mode == "expected_count":
                            for ratio in float_grid(args.decode_count_ratio_min, args.decode_count_ratio_max, args.decode_count_ratio_step):
                                for bias in float_grid(args.decode_count_bias_min, args.decode_count_bias_max, args.decode_count_bias_step):
                                    configs.append({
                                        "mode": mode,
                                        "history_weight": history_weight,
                                        "retrieval_weight": retrieval_weight,
                                        "freq_weight": freq_weight,
                                        "ddi_penalty": ddi_penalty,
                                        "ratio": ratio,
                                        "bias": bias,
                                    })
    for mode in sorted(history_retrieval_freq_graph_rank_modes & allowed_modes):
        count_mode = mode.removeprefix("history_retrieval_freq_graph_rank_")
        if count_mode == "predicted_count" and args.med_count_mode != "aux":
            continue
        for history_weight in history_retrieval_history_weights:
            for retrieval_weight in history_retrieval_retrieval_weights:
                for freq_weight in history_retrieval_freq_weights:
                    for ehr_graph_weight in history_retrieval_ehr_graph_weights:
                        for ddi_graph_weight in history_retrieval_ddi_graph_weights:
                            if count_mode == "expected_jaccard_count":
                                configs.append({
                                    "mode": mode,
                                    "history_weight": history_weight,
                                    "retrieval_weight": retrieval_weight,
                                    "freq_weight": freq_weight,
                                    "ehr_graph_weight": ehr_graph_weight,
                                    "ddi_graph_weight": ddi_graph_weight,
                                    "graph_source": args.history_retrieval_graph_source,
                                })
                            elif count_mode == "count_pair_mix":
                                continue
                            elif count_mode == "retrieval_count":
                                for retrieval_count_mode in retrieval_count_mode_grid():
                                    for ratio in float_grid(args.decode_count_ratio_min, args.decode_count_ratio_max, args.decode_count_ratio_step):
                                        for bias in float_grid(args.decode_count_bias_min, args.decode_count_bias_max, args.decode_count_bias_step):
                                            configs.append({
                                                "mode": mode,
                                                "history_weight": history_weight,
                                                "retrieval_weight": retrieval_weight,
                                                "freq_weight": freq_weight,
                                                "ehr_graph_weight": ehr_graph_weight,
                                                "ddi_graph_weight": ddi_graph_weight,
                                                "graph_source": args.history_retrieval_graph_source,
                                                "retrieval_count_mode": retrieval_count_mode,
                                                "ratio": ratio,
                                                "bias": bias,
                                            })
                            elif count_mode in {"expected_count", "predicted_count"}:
                                for ratio in float_grid(args.decode_count_ratio_min, args.decode_count_ratio_max, args.decode_count_ratio_step):
                                    for bias in float_grid(args.decode_count_bias_min, args.decode_count_bias_max, args.decode_count_bias_step):
                                        configs.append({
                                            "mode": mode,
                                            "history_weight": history_weight,
                                            "retrieval_weight": retrieval_weight,
                                            "freq_weight": freq_weight,
                                            "ehr_graph_weight": ehr_graph_weight,
                                            "ddi_graph_weight": ddi_graph_weight,
                                            "graph_source": args.history_retrieval_graph_source,
                                            "ratio": ratio,
                                            "bias": bias,
                                        })
    rank_modes = {
        "seq_rank_fixed_topk",
        "seq_rank_expected_count",
        "seq_rank_history_count",
        "seq_rank_predicted_count",
    }
    seq_prob_modes = {
        "seq_prob_fixed_topk",
        "seq_prob_expected_count",
        "seq_prob_history_count",
        "seq_prob_predicted_count",
    }
    if "oracle_count" in allowed_modes:
        configs.append({"mode": "oracle_count"})
    if "fixed_topk" in allowed_modes or "seq_rank_fixed_topk" in allowed_modes or "seq_prob_fixed_topk" in allowed_modes:
        for topk in range(args.decode_topk_min, args.decode_topk_max + 1, args.decode_topk_step):
            if "fixed_topk" in allowed_modes:
                configs.append({"mode": "fixed_topk", "topk": topk})
            if "seq_rank_fixed_topk" in allowed_modes and seq_decoder_enabled():
                configs.append({"mode": "seq_rank_fixed_topk", "topk": topk})
            if "seq_prob_fixed_topk" in allowed_modes and seq_decoder_enabled():
                configs.append({"mode": "seq_prob_fixed_topk", "topk": topk})
    ratio_values = float_grid(args.decode_count_ratio_min, args.decode_count_ratio_max, args.decode_count_ratio_step)
    bias_values = float_grid(args.decode_count_bias_min, args.decode_count_bias_max, args.decode_count_bias_step)
    if "expected_count" in allowed_modes or "seq_rank_expected_count" in allowed_modes or "seq_prob_expected_count" in allowed_modes:
        for ratio in ratio_values:
            for bias in bias_values:
                if "expected_count" in allowed_modes:
                    configs.append({"mode": "expected_count", "ratio": ratio, "bias": bias})
                if "seq_rank_expected_count" in allowed_modes and seq_decoder_enabled():
                    configs.append({"mode": "seq_rank_expected_count", "ratio": ratio, "bias": bias})
                if "seq_prob_expected_count" in allowed_modes and seq_decoder_enabled():
                    configs.append({"mode": "seq_prob_expected_count", "ratio": ratio, "bias": bias})
    if "history_count" in allowed_modes or "seq_rank_history_count" in allowed_modes or "seq_prob_history_count" in allowed_modes:
        for ratio in ratio_values:
            for bias in bias_values:
                if "history_count" in allowed_modes:
                    configs.append({"mode": "history_count", "ratio": ratio, "bias": bias})
                if "seq_rank_history_count" in allowed_modes and seq_decoder_enabled():
                    configs.append({"mode": "seq_rank_history_count", "ratio": ratio, "bias": bias})
                if "seq_prob_history_count" in allowed_modes and seq_decoder_enabled():
                    configs.append({"mode": "seq_prob_history_count", "ratio": ratio, "bias": bias})
    if (
        "predicted_count" in allowed_modes
        or "seq_rank_predicted_count" in allowed_modes
        or "seq_prob_predicted_count" in allowed_modes
    ) and args.med_count_mode == "aux":
        for ratio in ratio_values:
            for bias in bias_values:
                if "predicted_count" in allowed_modes:
                    configs.append({"mode": "predicted_count", "ratio": ratio, "bias": bias})
                if "seq_rank_predicted_count" in allowed_modes and seq_decoder_enabled():
                    configs.append({"mode": "seq_rank_predicted_count", "ratio": ratio, "bias": bias})
                if "seq_prob_predicted_count" in allowed_modes and seq_decoder_enabled():
                    configs.append({"mode": "seq_prob_predicted_count", "ratio": ratio, "bias": bias})
    if "seq_greedy" in allowed_modes and seq_decoder_enabled():
        for end_bias in seq_end_bias_grid():
            configs.append({"mode": "seq_greedy", "end_bias": end_bias})
    if "seq_hidr" in allowed_modes and seq_decoder_enabled():
        for end_bias in seq_end_bias_grid():
            configs.append({"mode": "seq_hidr", "end_bias": end_bias})
    if "seq_beam" in allowed_modes and seq_decoder_enabled():
        for end_bias in seq_end_bias_grid():
            configs.append({"mode": "seq_beam", "end_bias": end_bias})
    if not configs:
        raise ValueError("--decode_search_modes did not enable any supported decoder")
    return configs


def format_decode_config(config):
    mode = config["mode"]
    if mode == "action_direct":
        return "action_direct@{:.2f}".format(float(config.get("threshold", args.action_decode_threshold)))
    if mode == "threshold":
        return "threshold@{:.2f}".format(config["threshold"])
    if mode == "per_med_threshold":
        return "per_med_threshold"
    if mode == "history_threshold":
        return "histthr@{:.2f}/{:.2f}".format(config["history_threshold"], config["threshold"])
    if mode == "history_rank_fixed_topk":
        return "hist_rank_top{}@{:.2f}".format(config["topk"], config["weight"])
    if mode == "history_rank_threshold_count":
        return "hist_rank_thrcount@{:.2f}/{:.2f}{:+.1f}/{:.2f}".format(
            config["threshold"],
            config["ratio"],
            config["bias"],
            config["weight"],
        )
    if mode == "history_rank_expected_count":
        return "hist_rank_expected@{:.2f}{:+.1f}/{:.2f}".format(config["ratio"], config["bias"], config["weight"])
    if mode == "history_rank_history_count":
        return "hist_rank_history@{:.2f}{:+.1f}/{:.2f}".format(config["ratio"], config["bias"], config["weight"])
    if mode == "history_rank_predicted_count":
        return "hist_rank_predcount@{:.2f}{:+.1f}/{:.2f}".format(config["ratio"], config["bias"], config["weight"])
    if mode == "history_retrieval_rank_fixed_topk":
        return "hist_retr_rank_top{}@{:.2f}/{:.2f}".format(
            config["topk"],
            config["history_weight"],
            config["retrieval_weight"],
        )
    if mode == "history_retrieval_rank_threshold_count":
        return "hist_retr_rank_thrcount@{:.2f}/{:.2f}{:+.1f}/{:.2f}/{:.2f}".format(
            config["threshold"],
            config["ratio"],
            config["bias"],
            config["history_weight"],
            config["retrieval_weight"],
        )
    if mode == "history_retrieval_rank_expected_count":
        return "hist_retr_rank_expected@{:.2f}{:+.1f}/{:.2f}/{:.2f}".format(
            config["ratio"],
            config["bias"],
            config["history_weight"],
            config["retrieval_weight"],
        )
    if mode == "history_retrieval_rank_history_count":
        return "hist_retr_rank_history@{:.2f}{:+.1f}/{:.2f}/{:.2f}".format(
            config["ratio"],
            config["bias"],
            config["history_weight"],
            config["retrieval_weight"],
        )
    if mode == "history_retrieval_rank_predicted_count":
        return "hist_retr_rank_predcount@{:.2f}{:+.1f}/{:.2f}/{:.2f}".format(
            config["ratio"],
            config["bias"],
            config["history_weight"],
            config["retrieval_weight"],
        )
    if mode == "history_retrieval_seq_rank_fixed_topk":
        return "hist_retr_seq_rank_top{}@{:.2f}/{:.2f}/{:.2f}".format(
            config["topk"],
            config["history_weight"],
            config["retrieval_weight"],
            config["seq_weight"],
        )
    if mode == "history_retrieval_seq_rank_threshold_count":
        return "hist_retr_seq_rank_thrcount@{:.2f}/{:.2f}{:+.1f}/{:.2f}/{:.2f}/{:.2f}".format(
            config["threshold"],
            config["ratio"],
            config["bias"],
            config["history_weight"],
            config["retrieval_weight"],
            config["seq_weight"],
        )
    if mode == "history_retrieval_seq_rank_expected_count":
        return "hist_retr_seq_rank_expected@{:.2f}{:+.1f}/{:.2f}/{:.2f}/{:.2f}".format(
            config["ratio"],
            config["bias"],
            config["history_weight"],
            config["retrieval_weight"],
            config["seq_weight"],
        )
    if mode == "history_retrieval_seq_rank_history_count":
        return "hist_retr_seq_rank_history@{:.2f}{:+.1f}/{:.2f}/{:.2f}/{:.2f}".format(
            config["ratio"],
            config["bias"],
            config["history_weight"],
            config["retrieval_weight"],
            config["seq_weight"],
        )
    if mode == "history_retrieval_seq_rank_predicted_count":
        return "hist_retr_seq_rank_predcount@{:.2f}{:+.1f}/{:.2f}/{:.2f}/{:.2f}".format(
            config["ratio"],
            config["bias"],
            config["history_weight"],
            config["retrieval_weight"],
            config["seq_weight"],
        )
    if mode == "history_retrieval_freq_rank_fixed_topk":
        return "hist_retr_freq_rank_top{}@{:.2f}/{:.3f}/{:.3f}".format(
            config["topk"],
            config["history_weight"],
            config["retrieval_weight"],
            config["freq_weight"],
        )
    if mode == "history_retrieval_freq_rank_threshold_count":
        return "hist_retr_freq_rank_thrcount@{:.2f}/{:.2f}{:+.2f}/{:.2f}/{:.3f}/{:.3f}".format(
            config["threshold"],
            config["ratio"],
            config["bias"],
            config["history_weight"],
            config["retrieval_weight"],
            config["freq_weight"],
        )
    if mode == "history_retrieval_freq_rank_expected_count":
        return "hist_retr_freq_rank_expected@{:.2f}{:+.2f}/{:.2f}/{:.3f}/{:.3f}".format(
            config["ratio"],
            config["bias"],
            config["history_weight"],
            config["retrieval_weight"],
            config["freq_weight"],
        )
    if mode == "history_retrieval_freq_rank_expected_jaccard_count":
        return "hist_retr_freq_rank_expected_jaccard/{:.2f}/{:.3f}/{:.3f}".format(
            config["history_weight"],
            config["retrieval_weight"],
            config["freq_weight"],
        )
    if mode == "history_retrieval_freq_rank_history_count":
        return "hist_retr_freq_rank_history@{:.2f}{:+.2f}/{:.2f}/{:.3f}/{:.3f}".format(
            config["ratio"],
            config["bias"],
            config["history_weight"],
            config["retrieval_weight"],
            config["freq_weight"],
        )
    if mode == "history_retrieval_freq_rank_predicted_count":
        return "hist_retr_freq_rank_predcount@{:.2f}{:+.2f}/{:.2f}/{:.3f}/{:.3f}".format(
            config["ratio"],
            config["bias"],
            config["history_weight"],
            config["retrieval_weight"],
            config["freq_weight"],
        )
    if mode == "history_retrieval_freq_ddi_rank_expected_count":
        return "hist_retr_freq_ddi_rank_expected@{:.2f}{:+.2f}/{:.2f}/{:.3f}/{:.3f}/ddi{:.3f}".format(
            config["ratio"],
            config["bias"],
            config["history_weight"],
            config["retrieval_weight"],
            config["freq_weight"],
            config["ddi_penalty"],
        )
    if mode == "history_retrieval_freq_graph_rank_expected_count":
        return "hist_retr_freq_graph_rank_expected@{:.2f}{:+.2f}/{:.2f}/{:.3f}/{:.3f}/ehr{:.3f}/ddi{:.3f}/{}".format(
            config["ratio"],
            config["bias"],
            config["history_weight"],
            config["retrieval_weight"],
            config["freq_weight"],
            config["ehr_graph_weight"],
            config["ddi_graph_weight"],
            config.get("graph_source", "model"),
        )
    if mode == "history_retrieval_freq_graph_rank_expected_jaccard_count":
        return "hist_retr_freq_graph_rank_expected_jaccard/{:.2f}/{:.3f}/{:.3f}/ehr{:.3f}/ddi{:.3f}/{}".format(
            config["history_weight"],
            config["retrieval_weight"],
            config["freq_weight"],
            config["ehr_graph_weight"],
            config["ddi_graph_weight"],
            config.get("graph_source", "model"),
        )
    if mode == "history_retrieval_freq_graph_rank_count_pair_mix":
        return "hist_retr_freq_graph_rank_count_pair_mix/{:.2f}/{:.3f}/{:.3f}/ehr{:.3f}/ddi{:.3f}/{}/{}/{}:{}/{:.2f}{:+.2f}".format(
            config["history_weight"],
            config["retrieval_weight"],
            config["freq_weight"],
            config["ehr_graph_weight"],
            config["ddi_graph_weight"],
            config.get("graph_source", "model"),
            config.get("base_count_mode", "expected_jaccard_count"),
            config.get("left", "configured"),
            config.get("right", "previous_count_add_prob"),
            config.get("alpha", 0.94),
            config.get("ratio", 1.0),
            config.get("bias", 0.0),
        )
    if mode == "history_retrieval_freq_graph_rank_predicted_count":
        return "hist_retr_freq_graph_rank_predicted@{:.2f}{:+.2f}/{:.2f}/{:.3f}/{:.3f}/ehr{:.3f}/ddi{:.3f}/{}".format(
            config["ratio"],
            config["bias"],
            config["history_weight"],
            config["retrieval_weight"],
            config["freq_weight"],
            config["ehr_graph_weight"],
            config["ddi_graph_weight"],
            config.get("graph_source", "model"),
        )
    if mode == "history_retrieval_freq_graph_rank_retrieval_count":
        return "hist_retr_freq_graph_rank_retrieval_{}@{:.2f}{:+.2f}/{:.2f}/{:.3f}/{:.3f}/ehr{:.3f}/ddi{:.3f}/{}".format(
            config.get("retrieval_count_mode", "weighted_mean"),
            config["ratio"],
            config["bias"],
            config["history_weight"],
            config["retrieval_weight"],
            config["freq_weight"],
            config["ehr_graph_weight"],
            config["ddi_graph_weight"],
            config.get("graph_source", "model"),
        )
    if mode == "retrieval_rank_fixed_topk":
        return "retr_rank_top{}@{:.2f}".format(config["topk"], config["weight"])
    if mode == "retrieval_rank_threshold_count":
        return "retr_rank_thrcount@{:.2f}/{:.2f}{:+.1f}/{:.2f}".format(
            config["threshold"],
            config["ratio"],
            config["bias"],
            config["weight"],
        )
    if mode == "retrieval_rank_expected_count":
        return "retr_rank_expected@{:.2f}{:+.1f}/{:.2f}".format(config["ratio"], config["bias"], config["weight"])
    if mode == "retrieval_rank_history_count":
        return "retr_rank_history@{:.2f}{:+.1f}/{:.2f}".format(config["ratio"], config["bias"], config["weight"])
    if mode == "retrieval_rank_predicted_count":
        return "retr_rank_predcount@{:.2f}{:+.1f}/{:.2f}".format(config["ratio"], config["bias"], config["weight"])
    if mode == "fixed_topk":
        return "top{}".format(config["topk"])
    if mode == "oracle_count":
        return "oracle_count"
    if mode == "expected_count":
        return "expected@{:.2f}{:+.1f}".format(config["ratio"], config["bias"])
    if mode == "history_count":
        return "history@{:.2f}{:+.1f}".format(config["ratio"], config["bias"])
    if mode == "predicted_count":
        return "predcount@{:.2f}{:+.1f}".format(config["ratio"], config["bias"])
    if mode == "seq_greedy":
        return "seq_greedy{:+.1f}".format(config.get("end_bias", args.seq_decoder_end_bias))
    if mode == "seq_hidr":
        return "seq_hidr{:+.1f}".format(config.get("end_bias", args.seq_decoder_end_bias))
    if mode == "seq_beam":
        return "seq_beam{}_{:+.1f}".format(args.seq_decoder_beam_size, config.get("end_bias", args.seq_decoder_end_bias))
    if mode == "seq_prob_fixed_topk":
        return "seq_prob_top{}".format(config["topk"])
    if mode == "seq_prob_expected_count":
        return "seq_prob_expected@{:.2f}{:+.1f}".format(config["ratio"], config["bias"])
    if mode == "seq_prob_history_count":
        return "seq_prob_history@{:.2f}{:+.1f}".format(config["ratio"], config["bias"])
    if mode == "seq_prob_predicted_count":
        return "seq_prob_predcount@{:.2f}{:+.1f}".format(config["ratio"], config["bias"])
    if mode == "seq_rank_fixed_topk":
        return "seq_rank_top{}".format(config["topk"])
    if mode == "seq_rank_expected_count":
        return "seq_rank_expected@{:.2f}{:+.1f}".format(config["ratio"], config["bias"])
    if mode == "seq_rank_history_count":
        return "seq_rank_history@{:.2f}{:+.1f}".format(config["ratio"], config["bias"])
    if mode == "seq_rank_predicted_count":
        return "seq_rank_predcount@{:.2f}{:+.1f}".format(config["ratio"], config["bias"])
    return mode


def fast_multi_label_metric(y_gt, y_pred, y_prob, prauc_override=None):
    y_gt_bool = y_gt.astype(bool)
    y_pred_bool = y_pred.astype(bool)
    inter = np.logical_and(y_gt_bool, y_pred_bool).sum(axis=1).astype(np.float64)
    union = np.logical_or(y_gt_bool, y_pred_bool).sum(axis=1).astype(np.float64)
    pred_count = y_pred_bool.sum(axis=1).astype(np.float64)
    target_count = y_gt_bool.sum(axis=1).astype(np.float64)

    ja = np.divide(inter, union, out=np.zeros_like(inter), where=union != 0).mean()
    avg_precision = np.divide(inter, pred_count, out=np.zeros_like(inter), where=pred_count != 0)
    avg_recall = np.divide(inter, target_count, out=np.zeros_like(inter), where=target_count != 0)
    denom = avg_precision + avg_recall
    avg_f1 = np.divide(2 * avg_precision * avg_recall, denom, out=np.zeros_like(denom), where=denom != 0)

    if prauc_override is None:
        prauc = np.mean([average_precision_score(y_gt[i], y_prob[i], average='macro') for i in range(len(y_gt))])
    else:
        prauc = prauc_override
    return ja, prauc, avg_precision.mean(), avg_recall.mean(), avg_f1.mean()


def fit_per_med_thresholds(y_gt, y_pred_prob):
    y_gt_tensor = y_gt.to(device=y_pred_prob.device, dtype=torch.bool)
    thresholds = torch.arange(
        args.per_med_threshold_min,
        args.per_med_threshold_max + args.per_med_threshold_step / 2,
        args.per_med_threshold_step,
        device=y_pred_prob.device,
        dtype=y_pred_prob.dtype,
    )
    if thresholds.numel() == 0:
        thresholds = torch.tensor([args.eval_threshold], device=y_pred_prob.device, dtype=y_pred_prob.dtype)
    pred = y_pred_prob.unsqueeze(0) >= thresholds.view(-1, 1, 1)
    truth = y_gt_tensor.unsqueeze(0)
    tp = (pred & truth).sum(dim=1).to(dtype=y_pred_prob.dtype)
    fp = (pred & ~truth).sum(dim=1).to(dtype=y_pred_prob.dtype)
    fn = (~pred & truth).sum(dim=1).to(dtype=y_pred_prob.dtype)
    denom = 2 * tp + fp + fn
    f1 = torch.divide(2 * tp, denom.clamp_min(1e-8))
    # Avoid selecting pathological thresholds for medications unseen in validation.
    positive_counts = y_gt_tensor.sum(dim=0)
    best_index = torch.argmax(f1, dim=0)
    fitted = thresholds[best_index]
    fitted = torch.where(
        positive_counts > 0,
        fitted,
        torch.full_like(fitted, args.per_med_threshold_max),
    )
    return fitted.detach().cpu()


clinical_state_labels = None
clinical_state_masks = None
clinical_state_meta = None
med_transition_labels = None
med_transition_masks = None
train_clinical_state_labels = None
train_clinical_state_masks = None
train_med_transition_labels = None
train_med_transition_masks = None
test_clinical_state_labels = None
test_clinical_state_masks = None
test_med_transition_labels = None
test_med_transition_masks = None
eval_clinical_state_labels = None
eval_clinical_state_masks = None
eval_med_transition_labels = None
eval_med_transition_masks = None
clinical_features = None
clinical_feature_masks = None
clinical_feature_abnormal = None
clinical_feature_delta_hours = None
clinical_feature_meta = None
clinical_feature_dim = 0
train_clinical_feature_payload = None
test_clinical_feature_payload = None
eval_clinical_feature_payload = None
treatment_episode_uses_clinical = (
    (args.retrieval_mode == "treatment_episode" or args.treatment_episode_adapter_mode != "none")
    and args.treatment_episode_clinical_weight > 0
)
retrieval_requires_clinical = "clinical" in args.retrieval_mode
insert_requires_clinical = args.insert_retrieved_visits and "clinical" in args.insert_visit_mode
clinical_feature_sidecar_required = (
    args.clinical_feature_mode == "input"
    or retrieval_requires_clinical
    or treatment_episode_uses_clinical
    or insert_requires_clinical
    or args.paper_m3_clinical_weight > 0
)
model_uses_clinical_features = args.clinical_feature_mode == "input"
clinical_state_sidecar_required = (
    args.multitask_mode == "clinical_state"
    or args.treatment_episode_adapter_mode != "none"
    or (
        args.retrieval_mode == "treatment_episode"
        and (
            args.treatment_episode_outcome_weight > 0
            or args.treatment_episode_safety_weight > 0
        )
    )
)
if clinical_state_sidecar_required:
    clinical_state_labels, clinical_state_masks, clinical_state_meta = load_clinical_state_artifacts(f"data/{dataset}/")
    if clinical_state_labels is None or clinical_state_masks is None:
        raise FileNotFoundError(f"missing clinical state artifacts under data/{dataset}/")
    med_transition_labels, med_transition_masks = build_med_transition_artifacts(data)
    raw_total_visits = sum(len(patient) for patient in raw_data)
    if len(clinical_state_labels) != raw_total_visits:
        raise ValueError("clinical state labels are not aligned with visit sample count")
    if args.multitask_mode == "clinical_state" and len(med_transition_labels) != sum(len(patient) for patient in data):
        raise ValueError("med transition labels are not aligned with visit sample count")
    clinical_state_labels_by_patient, clinical_state_masks_by_patient = split_visit_level_artifacts_by_patient(
        raw_data,
        clinical_state_labels,
        clinical_state_masks,
    )
    clinical_state_labels_by_patient = filter_patients_by_protocol(clinical_state_labels_by_patient)
    clinical_state_masks_by_patient = filter_patients_by_protocol(clinical_state_masks_by_patient)
    if args.multitask_mode == "clinical_state":
        med_transition_labels_by_patient, med_transition_masks_by_patient = split_visit_level_artifacts_by_patient(
            data,
            med_transition_labels,
            med_transition_masks,
        )
    else:
        med_transition_labels_by_patient = med_transition_masks_by_patient = None
    (
        train_clinical_state_labels,
        test_clinical_state_labels,
        eval_clinical_state_labels,
    ) = split_patient_artifacts(clinical_state_labels_by_patient)
    (
        train_clinical_state_masks,
        test_clinical_state_masks,
        eval_clinical_state_masks,
    ) = split_patient_artifacts(clinical_state_masks_by_patient)
    if med_transition_labels_by_patient is not None:
        (
            train_med_transition_labels,
            test_med_transition_labels,
            eval_med_transition_labels,
        ) = split_patient_artifacts(med_transition_labels_by_patient)
        (
            train_med_transition_masks,
            test_med_transition_masks,
            eval_med_transition_masks,
        ) = split_patient_artifacts(med_transition_masks_by_patient)

if clinical_feature_sidecar_required:
    (
        clinical_features,
        clinical_feature_masks,
        clinical_feature_abnormal,
        clinical_feature_delta_hours,
        clinical_feature_meta,
    ) = load_clinical_feature_artifacts(f"data/{dataset}/")
    if clinical_features is None:
        raise FileNotFoundError(f"missing clinical feature artifacts under data/{dataset}/")
    total_visits = sum(len(patient) for patient in raw_data)
    if len(clinical_features) != total_visits:
        raise ValueError("clinical features are not aligned with visit sample count")
    clinical_feature_dim = len(clinical_feature_meta["feature_names"])
    clinical_features_by_patient = filter_patients_by_protocol(
        split_visit_level_artifact_by_patient(raw_data, clinical_features, "clinical_features")
    )
    clinical_feature_masks_by_patient = filter_patients_by_protocol(
        split_visit_level_artifact_by_patient(raw_data, clinical_feature_masks, "clinical_feature_masks")
    )
    clinical_feature_abnormal_by_patient = filter_patients_by_protocol(
        split_visit_level_artifact_by_patient(raw_data, clinical_feature_abnormal, "clinical_feature_abnormal")
    )
    clinical_feature_delta_by_patient = filter_patients_by_protocol(
        split_visit_level_artifact_by_patient(raw_data, clinical_feature_delta_hours, "clinical_feature_delta_hours")
    )
    split_clinical_features = split_patient_artifacts(clinical_features_by_patient)
    split_clinical_feature_masks = split_patient_artifacts(clinical_feature_masks_by_patient)
    split_clinical_feature_abnormal = split_patient_artifacts(clinical_feature_abnormal_by_patient)
    split_clinical_feature_delta = split_patient_artifacts(clinical_feature_delta_by_patient)
    train_clinical_feature_payload = (
        split_clinical_features[0],
        split_clinical_feature_masks[0],
        split_clinical_feature_abnormal[0],
        split_clinical_feature_delta[0],
    )
    test_clinical_feature_payload = (
        split_clinical_features[1],
        split_clinical_feature_masks[1],
        split_clinical_feature_abnormal[1],
        split_clinical_feature_delta[1],
    )
    eval_clinical_feature_payload = (
        split_clinical_features[2],
        split_clinical_feature_masks[2],
        split_clinical_feature_abnormal[2],
        split_clinical_feature_delta[2],
    )

device=torch.device(f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu")

retrieval_copy_uses_context = args.retrieval_copy_mode == "prob_mix" and args.retrieval_copy_weight > 0
retrieval_readout_uses_context = args.retrieval_readout_mode != "none" and args.retrieval_readout_weight > 0
treatment_episode_adapter_enabled = (
    args.treatment_episode_adapter_mode != "none"
    and (
        args.treatment_episode_adapter_weight > 0
        or args.treatment_episode_action_aux_weight > 0
        or args.treatment_episode_action_bias_weight > 0
        or args.treatment_episode_fused_action_loss_weight > 0
    )
)
retrieval_query_rank_uses_context = args.retrieval_query_rank_weight > 0
seq_retrieval_copy_uses_context = args.seq_retrieval_copy_mode != "none"
seq_hidr_memory_uses_context = args.seq_decoder_mode in {"hidr_memory", "hidr_transformer"}
retrieval_uses_context = (
    (args.retrieval_context_mode != "none" and args.retrieval_context_weight > 0)
    or retrieval_copy_uses_context
    or retrieval_readout_uses_context
    or retrieval_query_rank_uses_context
    or seq_retrieval_copy_uses_context
    or seq_hidr_memory_uses_context
)
retrieval_count_decoder_enabled = decode_uses_retrieval_count()
retrieval_context_is_static = retrieval_uses_context and args.retrieval_context_source == "static"
retrieval_rank_decoder_enabled = decode_uses_retrieval_rank()
episode_action_count_sources_enabled = requested_episode_action_count_sources()
insert_is_learned = args.insert_retrieved_visits and "learned" in args.insert_visit_mode
insert_is_paper_m3 = args.insert_retrieved_visits and args.insert_visit_mode == "paper_m3"
retrieval_is_treatment_episode = args.retrieval_mode == "treatment_episode"
retrieval_is_paper_m3 = args.retrieval_mode == "paper_m3"
retrieval_is_official_topidx = args.retrieval_mode == "official_topidx"
paper_m3_artifacts_required = insert_is_paper_m3 or retrieval_is_paper_m3
retrieval_is_learned = args.retrieval_mode in ("learned", "learned_clinical") or (
    retrieval_uses_context and not retrieval_context_is_static
    or insert_is_learned
)
retrieval_train_clinical_payload = (
    train_clinical_feature_payload if (retrieval_requires_clinical or treatment_episode_uses_clinical) else None
)
retrieval_eval_clinical_payload = (
    eval_clinical_feature_payload if (retrieval_requires_clinical or treatment_episode_uses_clinical) else None
)
retrieval_test_clinical_payload = (
    test_clinical_feature_payload if (retrieval_requires_clinical or treatment_episode_uses_clinical) else None
)
retrieval_train_clinical_state_payload = (
    (train_clinical_state_labels, train_clinical_state_masks)
    if (retrieval_is_treatment_episode or treatment_episode_adapter_enabled) and train_clinical_state_labels is not None
    else None
)
retrieval_eval_clinical_state_payload = (
    (eval_clinical_state_labels, eval_clinical_state_masks)
    if (retrieval_is_treatment_episode or treatment_episode_adapter_enabled) and eval_clinical_state_labels is not None
    else None
)
retrieval_test_clinical_state_payload = (
    (test_clinical_state_labels, test_clinical_state_masks)
    if (retrieval_is_treatment_episode or treatment_episode_adapter_enabled) and test_clinical_state_labels is not None
    else None
)
if retrieval_requires_clinical and clinical_features is None:
    raise ValueError("clinical retrieval requires clinical feature artifacts; enable --clinical_feature_mode input")
if args.insert_retrieved_visits and "clinical" in args.insert_visit_mode and clinical_features is None:
    raise ValueError("clinical inserted-visit retrieval requires clinical feature artifacts; enable --clinical_feature_mode input")


def concat_patient_payloads(*payloads):
    if any(payload is None for payload in payloads):
        return None
    return tuple(
        [item for payload in payloads for item in payload[part_idx]]
        for part_idx in range(len(payloads[0]))
    )


def uses_train_plus_own_history_scope():
    return args.retrieval_candidate_scope == "train_plus_own_history"


def uses_split_augmented_candidate_scope():
    return args.retrieval_candidate_scope in {"train_plus_own_history", "paper_like"}


def effective_exclude_same_patient_current_and_future():
    return (
        args.retrieval_exclude_self_future
        or uses_train_plus_own_history_scope()
        or args.retrieval_candidate_scope == "paper_like"
    )


def retrieval_candidate_start_visit():
    return 0 if uses_train_plus_own_history_scope() else target_start_visit


def nonlearned_eval_candidate_data():
    if uses_split_augmented_candidate_scope():
        return data_train + data_eval
    return data_train


def nonlearned_eval_candidate_patient_id_offsets():
    if uses_split_augmented_candidate_scope():
        return train_patient_id_offsets + eval_patient_id_offsets
    return train_patient_id_offsets


def nonlearned_eval_candidate_clinical_payload():
    if not retrieval_requires_clinical and not treatment_episode_uses_clinical and not (
            paper_m3_artifacts_required
            or args.insert_retrieved_visits and ("clinical" in args.insert_visit_mode or model_uses_clinical_features)
    ):
        return None
    if uses_split_augmented_candidate_scope():
        return concat_patient_payloads(train_clinical_feature_payload, eval_clinical_feature_payload)
    return train_clinical_feature_payload


def nonlearned_eval_candidate_clinical_state_payload():
    if not (retrieval_is_treatment_episode or treatment_episode_adapter_enabled):
        return None
    if uses_split_augmented_candidate_scope():
        return concat_patient_payloads(
            retrieval_train_clinical_state_payload,
            retrieval_eval_clinical_state_payload,
        )
    return retrieval_train_clinical_state_payload


def nonlearned_test_candidate_data(sampled_test_data):
    if uses_split_augmented_candidate_scope():
        return data_train + sampled_test_data
    return data_train


def nonlearned_test_candidate_patient_id_offsets(sampled_payload):
    if uses_split_augmented_candidate_scope():
        sampled_offsets = [item["global_patient_idx"] for item in sampled_payload]
        return train_patient_id_offsets + sampled_offsets
    return train_patient_id_offsets


def nonlearned_test_candidate_clinical_payload(sampled_payload):
    if not retrieval_requires_clinical and not treatment_episode_uses_clinical and not (
            paper_m3_artifacts_required
            or args.insert_retrieved_visits and ("clinical" in args.insert_visit_mode or model_uses_clinical_features)
    ):
        return None
    if uses_split_augmented_candidate_scope():
        sampled_payload_features = (
            [item["clinical_features"] for item in sampled_payload],
            [item["clinical_feature_masks"] for item in sampled_payload],
            [item["clinical_feature_abnormal"] for item in sampled_payload],
            [item["clinical_feature_delta_hours"] for item in sampled_payload],
        )
        return concat_patient_payloads(train_clinical_feature_payload, sampled_payload_features)
    return train_clinical_feature_payload


def nonlearned_test_candidate_clinical_state_payload(sampled_payload):
    if not (retrieval_is_treatment_episode or treatment_episode_adapter_enabled):
        return None
    if uses_split_augmented_candidate_scope():
        sampled_payload_states = (
            [item["clinical_state_labels"] for item in sampled_payload],
            [item["clinical_state_masks"] for item in sampled_payload],
        )
        return concat_patient_payloads(retrieval_train_clinical_state_payload, sampled_payload_states)
    return retrieval_train_clinical_state_payload


eval_retrieval_candidate_data = nonlearned_eval_candidate_data()
eval_retrieval_candidate_patient_id_offsets = nonlearned_eval_candidate_patient_id_offsets()
eval_retrieval_candidate_clinical_payload = nonlearned_eval_candidate_clinical_payload()
eval_retrieval_candidate_clinical_state_payload = nonlearned_eval_candidate_clinical_state_payload()
learned_eval_candidate_data = eval_retrieval_candidate_data if uses_train_plus_own_history_scope() else data_train
learned_eval_candidate_patient_id_offsets = (
    eval_retrieval_candidate_patient_id_offsets if uses_train_plus_own_history_scope() else train_patient_id_offsets
)
learned_eval_candidate_clinical_payload = (
    eval_retrieval_candidate_clinical_payload if uses_train_plus_own_history_scope() else retrieval_train_clinical_payload
)
learned_eval_insert_candidate_clinical_payload = (
    eval_retrieval_candidate_clinical_payload if uses_train_plus_own_history_scope() else train_clinical_feature_payload
)

train_retrieval_priors = None
eval_retrieval_priors = None
if (
    args.retrieval_mode != "none"
    and not retrieval_is_learned
    and not retrieval_is_treatment_episode
    and not retrieval_is_paper_m3
    and not retrieval_is_official_topidx
    and (args.retrieval_weight > 0 or args.retrieval_copy_weight > 0 or retrieval_rank_decoder_enabled)
):
    train_retrieval_priors = build_retrieval_priors(
        target_data=data_train,
        candidate_data=data_train,
        voc_size=voc_size,
        device=device,
        mode=args.retrieval_mode,
        topk=args.retrieval_topk,
        alpha=args.retrieval_alpha,
        prior_transform=args.retrieval_prior_transform,
        target_clinical_payload=retrieval_train_clinical_payload,
        candidate_clinical_payload=retrieval_train_clinical_payload,
        exclude_same_patient_current_and_future=True,
        target_patient_id_offsets=train_patient_id_offsets,
        candidate_patient_id_offsets=train_patient_id_offsets,
        candidate_start_visit=retrieval_candidate_start_visit(),
    )
    eval_retrieval_priors = build_retrieval_priors(
        target_data=data_eval,
        candidate_data=eval_retrieval_candidate_data,
        voc_size=voc_size,
        device=device,
        mode=args.retrieval_mode,
        topk=args.retrieval_topk,
        alpha=args.retrieval_alpha,
        prior_transform=args.retrieval_prior_transform,
        target_clinical_payload=retrieval_eval_clinical_payload if retrieval_requires_clinical else None,
        candidate_clinical_payload=eval_retrieval_candidate_clinical_payload if retrieval_requires_clinical else None,
        exclude_same_patient_current_and_future=effective_exclude_same_patient_current_and_future(),
        allow_train_or_own_history_only=uses_train_plus_own_history_scope(),
        train_patient_id_limit=len(data_train),
        target_patient_id_offsets=eval_patient_id_offsets,
        candidate_patient_id_offsets=eval_retrieval_candidate_patient_id_offsets,
        candidate_start_visit=retrieval_candidate_start_visit(),
    )
if retrieval_is_treatment_episode and (
    args.retrieval_weight > 0 or args.retrieval_copy_weight > 0 or retrieval_rank_decoder_enabled
):
    treatment_episode_config = make_treatment_episode_config()
    train_retrieval_priors = build_episode_retrieval_priors(
        target_data=data_train,
        candidate_data=data_train,
        voc_size=voc_size,
        device=device,
        topk=args.retrieval_topk,
        config=treatment_episode_config,
        prior_transform=args.retrieval_prior_transform,
        target_clinical_payload=retrieval_train_clinical_payload if treatment_episode_uses_clinical else None,
        candidate_clinical_payload=retrieval_train_clinical_payload if treatment_episode_uses_clinical else None,
        target_clinical_state_payload=retrieval_train_clinical_state_payload,
        candidate_clinical_state_payload=retrieval_train_clinical_state_payload,
        ddi_adj=ddi_A,
        exclude_same_patient_current_and_future=True,
        target_patient_id_offsets=train_patient_id_offsets,
        candidate_patient_id_offsets=train_patient_id_offsets,
        target_start_visit=target_start_visit,
        candidate_start_visit=retrieval_candidate_start_visit(),
    )
    eval_retrieval_priors = build_episode_retrieval_priors(
        target_data=data_eval,
        candidate_data=eval_retrieval_candidate_data,
        voc_size=voc_size,
        device=device,
        topk=args.retrieval_topk,
        config=treatment_episode_config,
        prior_transform=args.retrieval_prior_transform,
        target_clinical_payload=retrieval_eval_clinical_payload if treatment_episode_uses_clinical else None,
        candidate_clinical_payload=eval_retrieval_candidate_clinical_payload if treatment_episode_uses_clinical else None,
        target_clinical_state_payload=retrieval_eval_clinical_state_payload,
        candidate_clinical_state_payload=eval_retrieval_candidate_clinical_state_payload,
        ddi_adj=ddi_A,
        exclude_same_patient_current_and_future=effective_exclude_same_patient_current_and_future(),
        allow_train_or_own_history_only=uses_train_plus_own_history_scope(),
        train_patient_id_limit=len(data_train),
        target_patient_id_offsets=eval_patient_id_offsets,
        candidate_patient_id_offsets=eval_retrieval_candidate_patient_id_offsets,
        target_start_visit=target_start_visit,
        candidate_start_visit=retrieval_candidate_start_visit(),
    )
if retrieval_is_official_topidx and (
    args.retrieval_weight > 0 or args.retrieval_copy_weight > 0 or retrieval_rank_decoder_enabled
):
    train_retrieval_priors = build_official_topidx_priors("train", data_train, data_train)
    eval_retrieval_priors = build_official_topidx_priors("eval", data_eval, data_eval)

train_static_retrieval_context = None
eval_static_retrieval_context = None
train_treatment_episode_context = None
eval_treatment_episode_context = None
if treatment_episode_adapter_enabled:
    treatment_episode_config = make_treatment_episode_config()
    train_treatment_episode_context = build_episode_retrieval_context(
        target_data=data_train,
        candidate_data=data_train,
        voc_size=voc_size,
        device=device,
        topk=args.retrieval_topk,
        config=treatment_episode_config,
        target_clinical_payload=retrieval_train_clinical_payload if treatment_episode_uses_clinical else None,
        candidate_clinical_payload=retrieval_train_clinical_payload if treatment_episode_uses_clinical else None,
        target_clinical_state_payload=retrieval_train_clinical_state_payload,
        candidate_clinical_state_payload=retrieval_train_clinical_state_payload,
        ddi_adj=ddi_A,
        exclude_same_patient_current_and_future=True,
        target_patient_id_offsets=train_patient_id_offsets,
        candidate_patient_id_offsets=train_patient_id_offsets,
        target_start_visit=target_start_visit,
        candidate_start_visit=retrieval_candidate_start_visit(),
    )
    eval_treatment_episode_context = build_episode_retrieval_context(
        target_data=data_eval,
        candidate_data=eval_retrieval_candidate_data,
        voc_size=voc_size,
        device=device,
        topk=args.retrieval_topk,
        config=treatment_episode_config,
        target_clinical_payload=retrieval_eval_clinical_payload if treatment_episode_uses_clinical else None,
        candidate_clinical_payload=eval_retrieval_candidate_clinical_payload if treatment_episode_uses_clinical else None,
        target_clinical_state_payload=retrieval_eval_clinical_state_payload,
        candidate_clinical_state_payload=eval_retrieval_candidate_clinical_state_payload,
        ddi_adj=ddi_A,
        exclude_same_patient_current_and_future=effective_exclude_same_patient_current_and_future(),
        allow_train_or_own_history_only=uses_train_plus_own_history_scope(),
        train_patient_id_limit=len(data_train),
        target_patient_id_offsets=eval_patient_id_offsets,
        candidate_patient_id_offsets=eval_retrieval_candidate_patient_id_offsets,
        target_start_visit=target_start_visit,
        candidate_start_visit=retrieval_candidate_start_visit(),
    )
if retrieval_context_is_static or (retrieval_count_decoder_enabled and retrieval_is_treatment_episode) or (
    episode_action_count_sources_enabled and retrieval_is_treatment_episode
):
    if retrieval_is_treatment_episode:
        train_static_retrieval_context = train_treatment_episode_context or build_episode_retrieval_context(
            target_data=data_train,
            candidate_data=data_train,
            voc_size=voc_size,
            device=device,
            topk=args.retrieval_topk,
            config=treatment_episode_config,
            target_clinical_payload=retrieval_train_clinical_payload if treatment_episode_uses_clinical else None,
            candidate_clinical_payload=retrieval_train_clinical_payload if treatment_episode_uses_clinical else None,
            target_clinical_state_payload=retrieval_train_clinical_state_payload,
            candidate_clinical_state_payload=retrieval_train_clinical_state_payload,
            ddi_adj=ddi_A,
            exclude_same_patient_current_and_future=True,
            target_patient_id_offsets=train_patient_id_offsets,
            candidate_patient_id_offsets=train_patient_id_offsets,
            target_start_visit=target_start_visit,
            candidate_start_visit=retrieval_candidate_start_visit(),
        )
        eval_static_retrieval_context = eval_treatment_episode_context or build_episode_retrieval_context(
            target_data=data_eval,
            candidate_data=eval_retrieval_candidate_data,
            voc_size=voc_size,
            device=device,
            topk=args.retrieval_topk,
            config=treatment_episode_config,
            target_clinical_payload=retrieval_eval_clinical_payload if treatment_episode_uses_clinical else None,
            candidate_clinical_payload=eval_retrieval_candidate_clinical_payload if treatment_episode_uses_clinical else None,
            target_clinical_state_payload=retrieval_eval_clinical_state_payload,
            candidate_clinical_state_payload=eval_retrieval_candidate_clinical_state_payload,
            ddi_adj=ddi_A,
            exclude_same_patient_current_and_future=effective_exclude_same_patient_current_and_future(),
            allow_train_or_own_history_only=uses_train_plus_own_history_scope(),
            train_patient_id_limit=len(data_train),
            target_patient_id_offsets=eval_patient_id_offsets,
            candidate_patient_id_offsets=eval_retrieval_candidate_patient_id_offsets,
            target_start_visit=target_start_visit,
            candidate_start_visit=retrieval_candidate_start_visit(),
        )
    elif paper_m3_artifacts_required:
        train_static_retrieval_context = None
        eval_static_retrieval_context = None
    else:
        train_static_retrieval_context = build_static_retrieval_context(
            target_data=data_train,
            candidate_data=data_train,
            voc_size=voc_size,
            mode=args.retrieval_mode,
            topk=args.retrieval_topk,
            alpha=args.retrieval_alpha,
            target_clinical_payload=retrieval_train_clinical_payload,
            candidate_clinical_payload=retrieval_train_clinical_payload,
            exclude_same_patient_current_and_future=True,
            target_patient_id_offsets=train_patient_id_offsets,
            candidate_patient_id_offsets=train_patient_id_offsets,
            candidate_start_visit=retrieval_candidate_start_visit(),
        )
        eval_static_retrieval_context = build_static_retrieval_context(
            target_data=data_eval,
            candidate_data=eval_retrieval_candidate_data,
            voc_size=voc_size,
            mode=args.retrieval_mode,
            topk=args.retrieval_topk,
            alpha=args.retrieval_alpha,
            target_clinical_payload=retrieval_eval_clinical_payload,
            candidate_clinical_payload=eval_retrieval_candidate_clinical_payload if retrieval_requires_clinical else None,
            exclude_same_patient_current_and_future=effective_exclude_same_patient_current_and_future(),
            allow_train_or_own_history_only=uses_train_plus_own_history_scope(),
            train_patient_id_limit=len(data_train),
            target_patient_id_offsets=eval_patient_id_offsets,
            candidate_patient_id_offsets=eval_retrieval_candidate_patient_id_offsets,
            candidate_start_visit=retrieval_candidate_start_visit(),
        )

train_inserted_visits = None
eval_inserted_visits = None
if args.insert_retrieved_visits:
    insert_needs_clinical_payload = "clinical" in args.insert_visit_mode or model_uses_clinical_features
    insert_train_clinical_payload = train_clinical_feature_payload if insert_needs_clinical_payload else None
    insert_eval_clinical_payload = eval_clinical_feature_payload if insert_needs_clinical_payload else None
    if not insert_is_learned and not insert_is_paper_m3:
        train_inserted_visits = build_inserted_retrieved_visits(
            target_data=data_train,
            candidate_data=data_train,
            voc_size=voc_size,
            mode=args.insert_visit_mode,
            topk=args.insert_visit_topk,
            alpha=args.insert_visit_alpha,
            target_clinical_payload=insert_train_clinical_payload,
            candidate_clinical_payload=insert_train_clinical_payload,
            exclude_same_patient_current_and_future=True,
            target_patient_id_offsets=train_patient_id_offsets,
            candidate_patient_id_offsets=train_patient_id_offsets,
            candidate_start_visit=retrieval_candidate_start_visit(),
        )
        eval_inserted_visits = build_inserted_retrieved_visits(
            target_data=data_eval,
            candidate_data=eval_retrieval_candidate_data,
            voc_size=voc_size,
            mode=args.insert_visit_mode,
            topk=args.insert_visit_topk,
            alpha=args.insert_visit_alpha,
            target_clinical_payload=insert_eval_clinical_payload,
            candidate_clinical_payload=eval_retrieval_candidate_clinical_payload if insert_needs_clinical_payload else None,
            exclude_same_patient_current_and_future=effective_exclude_same_patient_current_and_future(),
            allow_train_or_own_history_only=uses_train_plus_own_history_scope(),
            train_patient_id_limit=len(data_train),
            target_patient_id_offsets=eval_patient_id_offsets,
            candidate_patient_id_offsets=eval_retrieval_candidate_patient_id_offsets,
            candidate_start_visit=retrieval_candidate_start_visit(),
        )

retrieval_train_metadata = build_visit_metadata(data_train, voc_size, train_patient_id_offsets) if retrieval_is_learned else None
retrieval_eval_metadata = build_visit_metadata(data_eval, voc_size, eval_patient_id_offsets) if retrieval_is_learned else None
retrieval_learned_candidate_inputs = (
    build_visit_retrieval_inputs(data_train, voc_size, retrieval_train_clinical_payload)
    if args.retrieval_mode == "learned_clinical"
    else None
)
retrieval_learned_target_inputs = {}
if args.retrieval_mode == "learned_clinical":
    retrieval_learned_target_inputs["train"] = retrieval_learned_candidate_inputs
    retrieval_learned_target_inputs["eval"] = build_visit_retrieval_inputs(data_eval, voc_size, retrieval_eval_clinical_payload)
insert_candidate_visits = flatten_target_visits(data_train) if insert_is_learned else None
insert_candidate_clinical = (
    flatten_target_clinical_artifacts(insert_train_clinical_payload)
    if insert_is_learned and insert_needs_clinical_payload
    else None
)
insert_learned_candidate_inputs = (
    build_visit_retrieval_inputs(data_train, voc_size, insert_train_clinical_payload)
    if insert_is_learned and (args.insert_visit_mode.startswith("dxproc_") or "clinical" in args.insert_visit_mode)
    else None
)
insert_learned_target_inputs = {}
if insert_is_learned and (args.insert_visit_mode.startswith("dxproc_") or "clinical" in args.insert_visit_mode):
    insert_learned_target_inputs["train"] = insert_learned_candidate_inputs
    insert_learned_target_inputs["eval"] = build_visit_retrieval_inputs(data_eval, voc_size, insert_eval_clinical_payload)

med_freq=get_med_freq(data,size_med_voc).to(device)
if args.ehr_graph_mode == "directed_weighted":
    ehr_A_gpu=cached_directed_weighted_ehr_adj(data_train, size_med_voc, train_patient_id_offsets).to(device)
else:
    ehr_A_gpu=torch.tensor(ehr_A).to(device)
ddi_A_gpu=torch.tensor(ddi_A).to(device)

patient_temporal_type = args.patient_temporal_type or None
med_temporal_type = args.med_temporal_type or None
model = SEER(
    emb_dim=args.dim,
    k=args.visit,
    voc_size=voc_size,
    ehr_adj=ehr_A_gpu,
    ddi_adj=ddi_A_gpu,
    temporal_type=args.temporal_type,
    patient_temporal_type=patient_temporal_type,
    med_temporal_type=med_temporal_type,
    med_history_definition=args.med_history_definition,
    output_mix_weight=args.output_mix_weight,
    output_mix_mode=args.output_mix_mode,
    output_branch_calibration=args.output_branch_calibration,
    med_scorer_type=args.med_scorer_type,
    med_candidate_encoder_type=args.med_candidate_encoder_type,
    med_fusion_type=args.med_fusion_type,
    multitask_mode=args.multitask_mode,
    aux_label_dim=4 if args.multitask_mode == "clinical_state" else 0,
    aux_head_dropout=args.aux_head_dropout,
    patient_fusion_type=args.patient_fusion_type,
    patient_query_type=args.patient_query_type,
    direct_state_source=args.direct_state_source,
    direct_query_gate_init=args.direct_query_gate_init,
    clinical_feature_dim=clinical_feature_dim,
    clinical_feature_mode=args.clinical_feature_mode,
    clinical_fusion_type=args.clinical_fusion_type,
    clinical_feature_dropout=args.clinical_feature_dropout,
    clinical_temporal_fusion=args.clinical_temporal_fusion,
    clinical_temporal_gate_init=args.clinical_temporal_gate_init,
    retrieval_context_mode=args.retrieval_context_mode,
    retrieval_context_target=args.retrieval_context_target,
    retrieval_context_weight=args.retrieval_context_weight,
    retrieval_context_dropout=args.retrieval_context_dropout,
    retrieval_context_gate_init=args.retrieval_context_gate_init,
    external_memory_mode=args.external_memory_mode,
    external_memory_num=args.external_memory_num,
    external_memory_hidden_size=args.external_memory_hidden_size,
    external_memory_weight=args.external_memory_weight,
    external_memory_dropout=args.external_memory_dropout,
    external_memory_gate_init=args.external_memory_gate_init,
    local_memory_mode=args.local_memory_mode,
    local_memory_weight=args.local_memory_weight,
    local_memory_dropout=args.local_memory_dropout,
    local_memory_gate_init=args.local_memory_gate_init,
    patient_graph_mode=args.patient_graph_mode,
    patient_graph_ehr_scale=args.patient_graph_ehr_scale,
    patient_graph_ddi_scale=args.patient_graph_ddi_scale,
    patient_graph_dropout=args.patient_graph_dropout,
    graph_logit_bias_weight=args.graph_logit_bias_weight,
    graph_edge_bias_weight=args.graph_edge_bias_weight,
    graph_edge_ddi_weight=args.graph_edge_ddi_weight,
    graph_edge_prior_source=args.graph_edge_prior_source,
    graph_edge_detach_prior=args.graph_edge_detach_prior,
    graph_med_encoder_mode=args.graph_med_encoder_mode,
    graph_med_ehr_weight=args.graph_med_ehr_weight,
    graph_med_ddi_weight=args.graph_med_ddi_weight,
    graph_med_gate_hidden=args.graph_med_gate_hidden,
    graph_med_dropout=args.graph_med_dropout,
    patient_med_scorer_mode=args.patient_med_scorer_mode,
    patient_med_scorer_rank=args.patient_med_scorer_rank,
    patient_med_scorer_weight=args.patient_med_scorer_weight,
    patient_med_scorer_dropout=args.patient_med_scorer_dropout,
    main_med_scorer_mode=args.main_med_scorer_mode,
    main_med_scorer_hidden_dim=args.main_med_scorer_hidden_dim,
    main_med_scorer_rank=args.main_med_scorer_rank,
    main_med_scorer_dropout=args.main_med_scorer_dropout,
    current_med_attention_mode=args.current_med_attention_mode,
    current_med_attention_dropout=args.current_med_attention_dropout,
    current_med_attention_gate_init=args.current_med_attention_gate_init,
    disable_procedure_branch=args.disable_procedure_branch,
    own_history_context_mode=args.own_history_context_mode,
    own_history_context_weight=args.own_history_context_weight,
    own_history_context_dropout=args.own_history_context_dropout,
    own_history_context_gate_init=args.own_history_context_gate_init,
    own_history_context_recency_decay=args.own_history_context_recency_decay,
    posthoc_calibration_mode=args.posthoc_calibration_mode,
    posthoc_graph_source=args.posthoc_graph_source,
    retrieval_copy_mode="prob_mix" if retrieval_copy_uses_context else "none",
    retrieval_copy_weight=args.retrieval_copy_weight if retrieval_copy_uses_context else 0.0,
    retrieval_copy_gate_init=args.retrieval_copy_gate_init,
    retrieval_copy_dropout=args.retrieval_copy_dropout,
        retrieval_readout_mode=args.retrieval_readout_mode if retrieval_readout_uses_context else "none",
        retrieval_readout_weight=args.retrieval_readout_weight if retrieval_readout_uses_context else 0.0,
        retrieval_readout_gate_init=args.retrieval_readout_gate_init,
        retrieval_readout_dropout=args.retrieval_readout_dropout,
        retrieval_readout_prior_transform=args.retrieval_readout_prior_transform,
        treatment_episode_adapter_mode=args.treatment_episode_adapter_mode if treatment_episode_adapter_enabled else "none",
        treatment_episode_adapter_weight=args.treatment_episode_adapter_weight if treatment_episode_adapter_enabled else 0.0,
        treatment_episode_experience_dim=episode_experience_dim(size_med_voc),
        treatment_episode_adapter_dropout=args.treatment_episode_adapter_dropout,
        treatment_episode_adapter_gate_init=args.treatment_episode_adapter_gate_init,
        treatment_episode_action_bias_weight=args.treatment_episode_action_bias_weight if treatment_episode_adapter_enabled else 0.0,
        treatment_episode_bias_scale_mode=args.treatment_episode_bias_scale_mode if treatment_episode_adapter_enabled else "none",
    history_copy_mode=args.history_copy_mode,
    history_copy_weight=args.history_copy_weight,
    history_copy_prior_mode=args.history_copy_prior_mode,
    history_copy_decay=args.history_copy_decay,
    history_copy_gate_init=args.history_copy_gate_init,
    history_copy_dropout=args.history_copy_dropout,
    med_count_mode=args.med_count_mode,
    seq_decoder_mode=args.seq_decoder_mode,
    seq_decoder_dropout=args.seq_decoder_dropout,
    seq_transformer_hidden_dim=args.seq_transformer_hidden_dim,
    seq_transformer_nhead=args.seq_transformer_nhead,
    seq_transformer_layers=args.seq_transformer_layers,
    seq_copy_mode=args.seq_copy_mode,
    seq_copy_gate_init=args.seq_copy_gate_init,
    seq_copy_recency_decay=args.seq_copy_recency_decay,
    seq_retrieval_copy_mode=args.seq_retrieval_copy_mode,
    seq_retrieval_copy_gate_init=args.seq_retrieval_copy_gate_init,
    seq_hidr_gumbel_tau=args.seq_hidr_gumbel_tau,
    seq_hidr_att_tau=args.seq_hidr_att_tau,
    backbone_innovation_mode=args.backbone_innovation_mode,
    action_transition_mode=args.action_transition_mode,
    action_transition_hidden_dim=args.action_transition_hidden_dim,
    action_transition_dropout=args.action_transition_dropout,
    action_fusion_trainable=args.action_fusion_trainable,
    action_fusion_type=args.action_fusion_type,
    action_fusion_init_from_p4=args.action_fusion_init_from_p4,
    action_fusion_freeze_weights=args.action_fusion_freeze_weights,
    routed_fusion_hidden_dim=args.routed_fusion_hidden_dim,
    routed_fusion_drug_emb_dim=args.routed_fusion_drug_emb_dim,
    routed_fusion_dropout=args.routed_fusion_dropout,
    routed_fusion_entropy_weight=args.routed_fusion_entropy_weight,
    routed_residual_freeze_base=(args.routed_residual_freeze_base or not args.routed_residual_train_base),
    routed_residual_scale=args.routed_residual_scale,
    routed_residual_max_gate=args.routed_residual_max_gate,
    routed_residual_gate_init_bias=args.routed_residual_gate_init_bias,
    routed_residual_l2_weight=args.routed_residual_l2_weight,
    routed_residual_add_penalty_weight=args.routed_residual_add_penalty_weight,
    routed_residual_entropy_weight=args.routed_residual_entropy_weight,
    routed_residual_load_balance_weight=args.routed_residual_load_balance_weight,
    ).to(device)
if args.init_model_checkpoint:
    init_checkpoint_path = Path(args.init_model_checkpoint)
    if not init_checkpoint_path.exists():
        raise FileNotFoundError(f"--init_model_checkpoint not found: {args.init_model_checkpoint}")
    init_state_dict = load_state_dict_file(init_checkpoint_path, device)
    skipped_shape_keys = {}
    if args.init_model_ignore_size_mismatch:
        init_state_dict, skipped_shape_keys = filter_state_dict_by_shape(init_state_dict, model.state_dict())
    incompatible = model.load_state_dict(
        init_state_dict,
        strict=args.init_model_strict,
    )
    missing_keys = getattr(incompatible, "missing_keys", [])
    unexpected_keys = getattr(incompatible, "unexpected_keys", [])
    print(
        "initialized trainable model from {} (strict={}, ignore_size_mismatch={}, loaded_keys={}, missing_keys={}, unexpected_keys={}, skipped_shape_keys={})".format(
            init_checkpoint_path,
            args.init_model_strict,
            args.init_model_ignore_size_mismatch,
            len(init_state_dict),
            len(missing_keys),
            len(unexpected_keys),
            len(skipped_shape_keys),
        )
    )
    if skipped_shape_keys:
        preview = list(skipped_shape_keys.items())[:10]
        print("skipped size-mismatched init keys:", preview)
if args.freeze_base_for_seq_decoder:
    if not seq_decoder_enabled():
        raise ValueError("--freeze_base_for_seq_decoder requires a sequence decoder")
freeze_prefixes = []
if args.freeze_base_for_seq_decoder:
    freeze_prefixes.append("seq_decoder.")
if args.freeze_base_for_retrieval_readout:
    freeze_prefixes.append("retrieval_readout.")
if args.freeze_base_for_treatment_episode_adapter:
    freeze_prefixes.append("treatment_episode_adapter.")
if args.freeze_base_for_edge_bias:
    freeze_prefixes.append("patient_graph_edge_bias.")
if args.freeze_base_for_patient_med_scorer:
    freeze_prefixes.append("patient_med_scorer.")
if args.freeze_base_for_main_med_scorer:
    freeze_prefixes.append("main_med_scorer.")
if args.freeze_base_for_current_med_attention:
    freeze_prefixes.append("patrep.current_med_attention.")
if args.freeze_base_for_own_history_context:
    freeze_prefixes.append("own_history_context.")
if args.freeze_base_for_graph_score_distill:
    freeze_prefixes.append("lin1.")
if args.freeze_base_for_posthoc_calibration:
    freeze_prefixes.append("posthoc_calibrator.")
if args.freeze_base_for_med_count:
    freeze_prefixes.append("med_count_head.")
if args.freeze_backbone_for_action_fusion:
    if not args.action_fusion_trainable:
        raise ValueError("--freeze_backbone_for_action_fusion requires --action_fusion_trainable")
    freeze_prefixes.append("action_fusion.")
if freeze_prefixes:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = parameter.requires_grad and any(name.startswith(prefix) for prefix in freeze_prefixes)
teacher_parameter_reference = {
    name: parameter.detach().clone()
    for name, parameter in model.named_parameters()
    if parameter.requires_grad
} if args.teacher_parameter_anchor_weight > 0 else {}
optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=args.weight_decay)

# (diag_batch,proc_batch,med_batch,label_batch,ids_batch,batch_seq_len)
infer_batch_size=256
train_batch_kwargs = {}
eval_batch_kwargs = {}
if args.multitask_mode == "clinical_state":
    train_batch_kwargs.update(
        clinical_state_targets=flatten_visit_level_artifacts(train_clinical_state_labels) if train_clinical_state_labels is not None else None,
        clinical_state_masks=flatten_visit_level_artifacts(train_clinical_state_masks) if train_clinical_state_masks is not None else None,
        med_transition_targets=flatten_visit_level_artifacts(train_med_transition_labels) if train_med_transition_labels is not None else None,
        med_transition_masks=flatten_visit_level_artifacts(train_med_transition_masks) if train_med_transition_masks is not None else None,
    )
    eval_batch_kwargs.update(
        clinical_state_targets=flatten_visit_level_artifacts(eval_clinical_state_labels) if eval_clinical_state_labels is not None else None,
        clinical_state_masks=flatten_visit_level_artifacts(eval_clinical_state_masks) if eval_clinical_state_masks is not None else None,
        med_transition_targets=flatten_visit_level_artifacts(eval_med_transition_labels) if eval_med_transition_labels is not None else None,
        med_transition_masks=flatten_visit_level_artifacts(eval_med_transition_masks) if eval_med_transition_masks is not None else None,
    )


def build_train_batches(inserted_visits=None):
    batches = create_batches_dict(
        data_train,
        args.batch,
        voc_size,
        device,
        retrieval_priors=train_retrieval_priors,
        inserted_visits=inserted_visits,
        start_visit=target_start_visit,
        **add_clinical_feature_kwargs(
            dict(train_batch_kwargs),
            train_clinical_feature_payload if model_uses_clinical_features else None,
        ),
    )
    batches["split_name"] = "train"
    if train_static_retrieval_context is not None:
        set_retrieval_context_batches(batches, train_static_retrieval_context, args.batch)
    if train_treatment_episode_context is not None:
        set_treatment_episode_context_batches(batches, train_treatment_episode_context, args.batch)
    return batches


def build_eval_batches(inserted_visits=None):
    batches = create_batches_dict(
        data_eval,
        infer_batch_size,
        voc_size,
        device,
        retrieval_priors=eval_retrieval_priors,
        inserted_visits=inserted_visits,
        start_visit=target_start_visit,
        **add_clinical_feature_kwargs(
            dict(eval_batch_kwargs),
            eval_clinical_feature_payload if model_uses_clinical_features else None,
        ),
    )
    batches["split_name"] = "eval"
    if eval_static_retrieval_context is not None:
        set_retrieval_context_batches(batches, eval_static_retrieval_context, infer_batch_size)
    if eval_treatment_episode_context is not None:
        set_treatment_episode_context_batches(batches, eval_treatment_episode_context, infer_batch_size)
    return batches


train_batches=create_batches_dict(
    data_train,
    args.batch,
    voc_size,
    device,
    retrieval_priors=train_retrieval_priors,
    inserted_visits=train_inserted_visits,
    start_visit=target_start_visit,
    **add_clinical_feature_kwargs(
        train_batch_kwargs,
        train_clinical_feature_payload if model_uses_clinical_features else None,
    ),
)
eval_batches=create_batches_dict(
    data_eval,
    infer_batch_size,
    voc_size,
    device,
    retrieval_priors=eval_retrieval_priors,
    inserted_visits=eval_inserted_visits,
    start_visit=target_start_visit,
    **add_clinical_feature_kwargs(
        eval_batch_kwargs,
        eval_clinical_feature_payload if model_uses_clinical_features else None,
    ),
)
train_batches["split_name"] = "train"
eval_batches["split_name"] = "eval"


def attach_teacher_anchor_batches():
    if not args.teacher_anchor_components:
        return
    component_path = Path(args.teacher_anchor_components)
    if not component_path.is_file():
        raise FileNotFoundError(f"teacher anchor component not found: {component_path}")
    with np.load(component_path, allow_pickle=False) as payload:
        required = {"y_true", "y_pred", "final_decoder_score", "patient_ids", "visit_ids"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"teacher anchor component missing fields: {missing}")
        y_true = np.asarray(payload["y_true"], dtype=np.float32)
        teacher_scores = np.asarray(payload["final_decoder_score"], dtype=np.float32)
        teacher_counts = np.asarray(payload["y_pred"], dtype=np.float32).sum(axis=1)
        patient_ids = np.asarray(payload["patient_ids"])
        visit_ids = np.asarray(payload["visit_ids"])
    if y_true.shape != teacher_scores.shape or y_true.shape[0] != teacher_counts.shape[0]:
        raise ValueError("teacher anchor arrays have incompatible shapes")
    if not np.isfinite(teacher_scores).all() or not np.isfinite(teacher_counts).all():
        raise ValueError("teacher anchor arrays contain non-finite values")
    seen_indexes = []
    score_batches = []
    count_batches = []

    def scalar_batch_to_numpy(values):
        flattened = []
        for value in values:
            if torch.is_tensor(value):
                flattened.extend(value.detach().cpu().numpy().reshape(-1).tolist())
            else:
                flattened.extend(np.asarray(value).reshape(-1).tolist())
        return np.asarray(flattened)

    for batch_id, indexes in enumerate(train_batches["sample_index_batches"]):
        indexes = np.asarray(indexes, dtype=np.int64)
        seen_indexes.extend(indexes.tolist())
        label = train_batches["label_batches"][batch_id].detach().cpu().numpy()
        batch_patient_ids = scalar_batch_to_numpy(train_batches["ids_batches"][batch_id])
        batch_visit_ids = scalar_batch_to_numpy(train_batches["visit_ids_batches"][batch_id])
        if not np.array_equal(y_true[indexes], label):
            raise ValueError(f"teacher anchor y_true misalignment in train batch {batch_id}")
        if not np.array_equal(patient_ids[indexes], batch_patient_ids):
            raise ValueError(f"teacher anchor patient ID misalignment in train batch {batch_id}")
        if not np.array_equal(visit_ids[indexes], batch_visit_ids):
            raise ValueError(f"teacher anchor visit ID misalignment in train batch {batch_id}")
        score_batches.append(torch.as_tensor(teacher_scores[indexes], dtype=torch.float32, device=device))
        count_batches.append(torch.as_tensor(teacher_counts[indexes], dtype=torch.float32, device=device))
    if sorted(seen_indexes) != list(range(y_true.shape[0])):
        raise ValueError("teacher anchor rows are not a one-to-one cover of the training split")
    train_batches["teacher_score_batches"] = score_batches
    train_batches["teacher_count_batches"] = count_batches
    print(
        "attached train-only teacher anchors from {} (rows={}, rank_weight={}, count_weight={}, parameter_weight={})".format(
            component_path,
            y_true.shape[0],
            args.teacher_rank_distill_weight,
            args.teacher_count_distill_weight,
            args.teacher_parameter_anchor_weight,
        )
    )


attach_teacher_anchor_batches()
if train_static_retrieval_context is not None:
    set_retrieval_context_batches(train_batches, train_static_retrieval_context, args.batch)
if eval_static_retrieval_context is not None:
    set_retrieval_context_batches(eval_batches, eval_static_retrieval_context, infer_batch_size)
if train_treatment_episode_context is not None:
    set_treatment_episode_context_batches(train_batches, train_treatment_episode_context, args.batch)
if eval_treatment_episode_context is not None:
    set_treatment_episode_context_batches(eval_batches, eval_treatment_episode_context, infer_batch_size)

insert_base_train_batches = build_train_batches(inserted_visits=None) if insert_is_learned else train_batches
insert_base_eval_batches = build_eval_batches(inserted_visits=None) if insert_is_learned else eval_batches

paper_m3_phi_model = None
paper_m3_candidate_batches = None
paper_m3_candidate_metadata = None
paper_m3_candidate_visits = None
paper_m3_candidate_clinical = None
paper_m3_query_transform_weights = None
if paper_m3_artifacts_required:
    paper_m3_phi_model = clone_phi_model_from(model, args.paper_m3_checkpoint, device)
    paper_m3_train_candidate_data = data_train
    paper_m3_train_candidate_patient_id_offsets = train_patient_id_offsets
    paper_m3_train_candidate_clinical_payload = (
        train_clinical_feature_payload
        if model_uses_clinical_features or args.paper_m3_clinical_weight > 0
        else None
    )
    paper_m3_eval_candidate_data = eval_retrieval_candidate_data
    paper_m3_eval_candidate_patient_id_offsets = eval_retrieval_candidate_patient_id_offsets
    paper_m3_eval_candidate_clinical_payload = (
        eval_retrieval_candidate_clinical_payload
        if model_uses_clinical_features or args.paper_m3_clinical_weight > 0
        else None
    )
    paper_m3_train_target_clinical_inputs = flatten_clinical_similarity_inputs(
        train_clinical_feature_payload,
        start_visit=target_start_visit,
    ) if args.paper_m3_clinical_weight > 0 else None
    paper_m3_eval_target_clinical_inputs = flatten_clinical_similarity_inputs(
        eval_clinical_feature_payload,
        start_visit=target_start_visit,
    ) if args.paper_m3_clinical_weight > 0 else None
    paper_m3_train_candidate_clinical_inputs = flatten_clinical_similarity_inputs(
        paper_m3_train_candidate_clinical_payload,
        start_visit=retrieval_candidate_start_visit(),
    ) if args.paper_m3_clinical_weight > 0 else None
    paper_m3_eval_candidate_clinical_inputs = flatten_clinical_similarity_inputs(
        paper_m3_eval_candidate_clinical_payload,
        start_visit=retrieval_candidate_start_visit(),
    ) if args.paper_m3_clinical_weight > 0 else None
    paper_m3_train_candidate_batches = create_batches_dict(
        paper_m3_train_candidate_data,
        infer_batch_size,
        voc_size,
        device,
        start_visit=retrieval_candidate_start_visit(),
        **add_clinical_feature_kwargs(
            {},
            paper_m3_train_candidate_clinical_payload if model_uses_clinical_features else None,
        ),
    )
    paper_m3_train_candidate_batches["split_name"] = "paper_m3_train_candidate"
    paper_m3_train_candidate_metadata = build_visit_metadata(
        paper_m3_train_candidate_data,
        voc_size,
        paper_m3_train_candidate_patient_id_offsets,
        start_visit=retrieval_candidate_start_visit(),
    )
    paper_m3_train_target_retrieval_inputs = build_visit_retrieval_inputs(
        data_train,
        voc_size,
        patient_id_offsets=train_patient_id_offsets,
    ) if args.paper_m3_dxproc_weight > 0 else None
    paper_m3_train_candidate_retrieval_inputs = build_visit_retrieval_inputs(
        paper_m3_train_candidate_data,
        voc_size,
        patient_id_offsets=paper_m3_train_candidate_patient_id_offsets,
        start_visit=retrieval_candidate_start_visit(),
    ) if args.paper_m3_dxproc_weight > 0 else None
    paper_m3_train_candidate_visits = flatten_target_visits(
        paper_m3_train_candidate_data,
        start_visit=retrieval_candidate_start_visit(),
    )
    paper_m3_train_candidate_clinical = (
        flatten_target_clinical_artifacts(
            paper_m3_train_candidate_clinical_payload,
            start_visit=retrieval_candidate_start_visit(),
        )
        if paper_m3_train_candidate_clinical_payload is not None
        else None
    )
    if args.paper_m3_query_transform != "none":
        transform_target_batches = build_train_batches(inserted_visits=None)
        transform_train_queries = encode_patient_queries_with_normalization(
            paper_m3_phi_model,
            transform_target_batches,
            count_prediction_visits(data_train),
            normalize=False,
        )
        transform_train_metadata = build_visit_metadata(data_train, voc_size, train_patient_id_offsets)
        paper_m3_query_transform_weights = fit_label_ridge_query_transform(
            transform_train_queries,
            transform_train_metadata["meds"],
            args.paper_m3_query_transform_ridge,
        )
    paper_m3_eval_candidate_batches = create_batches_dict(
        paper_m3_eval_candidate_data,
        infer_batch_size,
        voc_size,
        device,
        start_visit=retrieval_candidate_start_visit(),
        **add_clinical_feature_kwargs(
            {},
            paper_m3_eval_candidate_clinical_payload if model_uses_clinical_features else None,
        ),
    )
    paper_m3_eval_candidate_batches["split_name"] = "paper_m3_eval_candidate"
    paper_m3_eval_candidate_metadata = build_visit_metadata(
        paper_m3_eval_candidate_data,
        voc_size,
        paper_m3_eval_candidate_patient_id_offsets,
        start_visit=retrieval_candidate_start_visit(),
    )
    paper_m3_eval_target_retrieval_inputs = build_visit_retrieval_inputs(
        data_eval,
        voc_size,
        patient_id_offsets=eval_patient_id_offsets,
    ) if args.paper_m3_dxproc_weight > 0 else None
    paper_m3_eval_candidate_retrieval_inputs = build_visit_retrieval_inputs(
        paper_m3_eval_candidate_data,
        voc_size,
        patient_id_offsets=paper_m3_eval_candidate_patient_id_offsets,
        start_visit=retrieval_candidate_start_visit(),
    ) if args.paper_m3_dxproc_weight > 0 else None
    paper_m3_eval_candidate_visits = flatten_target_visits(
        paper_m3_eval_candidate_data,
        start_visit=retrieval_candidate_start_visit(),
    )
    paper_m3_eval_candidate_clinical = (
        flatten_target_clinical_artifacts(
            paper_m3_eval_candidate_clinical_payload,
            start_visit=retrieval_candidate_start_visit(),
        )
        if paper_m3_eval_candidate_clinical_payload is not None
        else None
    )
    train_paper_m3_artifacts = cached_paper_m3_artifacts(
        paper_m3_cache_path(
            "train",
            train_patient_id_offsets,
            paper_m3_train_candidate_patient_id_offsets,
            retrieval_candidate_start_visit(),
        ),
        lambda: build_paper_m3_artifacts(
            phi_model=paper_m3_phi_model,
            target_batches=build_train_batches(inserted_visits=None),
            candidate_batches=paper_m3_train_candidate_batches,
            target_metadata=build_visit_metadata(data_train, voc_size, train_patient_id_offsets),
            candidate_metadata=paper_m3_train_candidate_metadata,
            target_total=count_prediction_visits(data_train),
            candidate_total=count_visits_from(paper_m3_train_candidate_data, retrieval_candidate_start_visit()),
            candidate_visits=paper_m3_train_candidate_visits,
            target_retrieval_inputs=paper_m3_train_target_retrieval_inputs,
            candidate_retrieval_inputs=paper_m3_train_candidate_retrieval_inputs,
            candidate_clinical=paper_m3_train_candidate_clinical,
            target_clinical_inputs=paper_m3_train_target_clinical_inputs,
            candidate_clinical_inputs=paper_m3_train_candidate_clinical_inputs,
            exclude_same_patient_current_and_future=True,
            query_transform_weights=paper_m3_query_transform_weights,
        ),
    )
    eval_paper_m3_artifacts = cached_paper_m3_artifacts(
        paper_m3_cache_path(
            "eval",
            eval_patient_id_offsets,
            paper_m3_eval_candidate_patient_id_offsets,
            retrieval_candidate_start_visit(),
        ),
        lambda: build_paper_m3_artifacts(
            phi_model=paper_m3_phi_model,
            target_batches=build_eval_batches(inserted_visits=None),
            candidate_batches=paper_m3_eval_candidate_batches,
            target_metadata=build_visit_metadata(data_eval, voc_size, eval_patient_id_offsets),
            candidate_metadata=paper_m3_eval_candidate_metadata,
            target_total=count_prediction_visits(data_eval),
            candidate_total=count_visits_from(paper_m3_eval_candidate_data, retrieval_candidate_start_visit()),
            candidate_visits=paper_m3_eval_candidate_visits,
            target_retrieval_inputs=paper_m3_eval_target_retrieval_inputs,
            candidate_retrieval_inputs=paper_m3_eval_candidate_retrieval_inputs,
            candidate_clinical=paper_m3_eval_candidate_clinical,
            target_clinical_inputs=paper_m3_eval_target_clinical_inputs,
            candidate_clinical_inputs=paper_m3_eval_candidate_clinical_inputs,
            exclude_same_patient_current_and_future=effective_exclude_same_patient_current_and_future(),
            allow_train_or_own_history_only=uses_train_plus_own_history_scope(),
            train_patient_id_limit=len(data_train),
            query_transform_weights=paper_m3_query_transform_weights,
        ),
    )
    if insert_is_paper_m3:
        train_inserted_visits = train_paper_m3_artifacts["inserted_visits"]
        eval_inserted_visits = eval_paper_m3_artifacts["inserted_visits"]
    train_retrieval_priors = train_paper_m3_artifacts["retrieval_priors"]
    eval_retrieval_priors = eval_paper_m3_artifacts["retrieval_priors"]
    if retrieval_context_is_static or retrieval_count_decoder_enabled:
        train_static_retrieval_context = train_paper_m3_artifacts["retrieval_context"]
        eval_static_retrieval_context = eval_paper_m3_artifacts["retrieval_context"]
    train_batches = build_train_batches(inserted_visits=train_inserted_visits if insert_is_paper_m3 else None)
    eval_batches = build_eval_batches(inserted_visits=eval_inserted_visits if insert_is_paper_m3 else None)
    if train_treatment_episode_context is not None:
        set_treatment_episode_context_batches(train_batches, train_treatment_episode_context, args.batch)
    if eval_treatment_episode_context is not None:
        set_treatment_episode_context_batches(eval_batches, eval_treatment_episode_context, infer_batch_size)


def refresh_learned_retrieval_batches(epoch_idx=None, include_train=True, include_eval=True):
    global train_batches, eval_batches
    if not retrieval_is_learned:
        return
    if epoch_idx is not None and epoch_idx < args.retrieval_warmup_epochs:
        if insert_is_learned:
            if include_train:
                train_batches = insert_base_train_batches
            if include_eval:
                eval_batches = insert_base_eval_batches
        if include_train:
            train_batches["retrieval_prior_batches"] = None
            clear_retrieval_context_batches(train_batches)
            if train_treatment_episode_context is not None:
                set_treatment_episode_context_batches(train_batches, train_treatment_episode_context, args.batch)
        if include_eval:
            eval_batches["retrieval_prior_batches"] = None
            clear_retrieval_context_batches(eval_batches)
            if eval_treatment_episode_context is not None:
                set_treatment_episode_context_batches(eval_batches, eval_treatment_episode_context, infer_batch_size)
        return
    if insert_is_learned:
        if include_train:
            train_inserted = build_learned_inserted_visits(
                model=model,
                target_batches=insert_base_train_batches,
                candidate_batches=insert_base_train_batches,
                target_metadata=retrieval_train_metadata,
                candidate_metadata=retrieval_train_metadata,
                target_total=count_prediction_visits(data_train),
                candidate_total=count_prediction_visits(data_train),
                candidate_visits=insert_candidate_visits,
                candidate_clinical=insert_candidate_clinical,
                exclude_same_patient_current_and_future=True,
            )
            train_batches = build_train_batches(inserted_visits=train_inserted)
        if include_eval:
            eval_inserted = build_learned_inserted_visits(
                model=model,
                target_batches=insert_base_eval_batches,
                candidate_batches=insert_base_train_batches,
                target_metadata=retrieval_eval_metadata,
                candidate_metadata=retrieval_train_metadata,
                target_total=count_prediction_visits(data_eval),
                candidate_total=count_prediction_visits(data_train),
                candidate_visits=insert_candidate_visits,
                candidate_clinical=insert_candidate_clinical,
            )
            eval_batches = build_eval_batches(inserted_visits=eval_inserted)
    if include_train:
        if args.retrieval_weight > 0 or args.retrieval_copy_mode == "fixed_prob":
            train_priors = build_learned_retrieval_priors(
                model=model,
                target_batches=train_batches,
                candidate_batches=train_batches,
                target_metadata=retrieval_train_metadata,
                candidate_metadata=retrieval_train_metadata,
                target_total=count_prediction_visits(data_train),
                candidate_total=count_prediction_visits(data_train),
                exclude_same_patient_current_and_future=True,
            )
            train_batches["retrieval_prior_batches"] = tensors_to_fixed_batches(
                [train_priors[i] for batch in train_batches["sample_index_batches"] for i in batch],
                args.batch,
            )
        if retrieval_uses_context:
            train_context = build_learned_retrieval_context(
                model=model,
                target_batches=train_batches,
                candidate_batches=train_batches,
                target_metadata=retrieval_train_metadata,
                candidate_metadata=retrieval_train_metadata,
                target_total=count_prediction_visits(data_train),
                candidate_total=count_prediction_visits(data_train),
                exclude_same_patient_current_and_future=True,
            )
            set_retrieval_context_batches(train_batches, train_context, args.batch)
    if include_eval:
        if args.retrieval_weight > 0 or args.retrieval_copy_mode == "fixed_prob":
            eval_priors = build_learned_retrieval_priors(
                model=model,
                target_batches=eval_batches,
                candidate_batches=train_batches,
                target_metadata=retrieval_eval_metadata,
                candidate_metadata=retrieval_train_metadata,
                target_total=count_prediction_visits(data_eval),
                candidate_total=count_prediction_visits(data_train),
            )
            eval_batches["retrieval_prior_batches"] = tensors_to_fixed_batches(
                [eval_priors[i] for batch in eval_batches["sample_index_batches"] for i in batch],
                infer_batch_size,
            )
        if retrieval_uses_context:
            eval_context = build_learned_retrieval_context(
                model=model,
                target_batches=eval_batches,
                candidate_batches=train_batches,
                target_metadata=retrieval_eval_metadata,
                candidate_metadata=retrieval_train_metadata,
                target_total=count_prediction_visits(data_eval),
                candidate_total=count_prediction_visits(data_train),
            )
            set_retrieval_context_batches(eval_batches, eval_context, infer_batch_size)


if retrieval_is_learned and args.retrieval_refresh == "static" and args.retrieval_warmup_epochs <= 0:
    refresh_learned_retrieval_batches(include_train=True, include_eval=True)

#%%
import torch
import torch.nn.functional as F
#%%

#%%
# shape: [batch, size_med_voc]
def loss_func(output, y_gt):
    # focal loss?
    return F.binary_cross_entropy_with_logits(output, y_gt)
    # bce_loss=F.binary_cross_entropy_with_logits(output, y_gt, reduction='none')
    # weights=1.0-med_freq
    # return (weights.view(1,-1).repeat(output.shape[0],1)*bce_loss).sum() * 10


def previous_meds_for_batch(batches, batch_id, reference):
    med_batches = batches.get("own_med_batches") or batches["med_batches"]
    med_batch = med_batches[batch_id]
    if med_batch.shape[1] > 1:
        return (med_batch[:, 1, :] > 0).to(device=reference.device, dtype=reference.dtype)
    return torch.zeros_like(reference, dtype=reference.dtype)


def action_transition_aux_loss(output, y_gt, batches, batch_id):
    if args.action_transition_weight <= 0:
        return output["med_logits"].new_tensor(0.0)
    continue_logits = output.get("continue_logits")
    add_logits = output.get("add_logits")
    remove_logits = output.get("remove_logits")
    if continue_logits is None or add_logits is None or remove_logits is None:
        raise ValueError("--action_transition_weight requires --action_transition_mode heads")
    previous = previous_meds_for_batch(batches, batch_id, add_logits)
    return action_heads_transition_loss(
        continue_logits,
        add_logits,
        remove_logits,
        previous,
        y_gt,
        continue_weight=args.continue_aux_weight,
        add_weight=args.add_aux_weight,
        remove_weight=args.remove_aux_weight,
        add_pos_weight=args.add_pos_weight,
        add_focal_gamma=args.add_focal_gamma,
    )


def action_ddi_add_aux_loss(output, batches, batch_id):
    if args.action_ddi_loss_weight <= 0:
        return output["med_logits"].new_tensor(0.0)
    add_logits = output.get("add_logits")
    if add_logits is None:
        raise ValueError("--action_ddi_loss_weight requires --action_transition_mode heads")
    previous = previous_meds_for_batch(batches, batch_id, add_logits)
    return action_heads_ddi_add_loss(
        add_logits,
        previous,
        ddi_A_gpu,
        add_ddi_loss_weight=args.add_ddi_loss_weight,
        ddi_risk_threshold=args.ddi_risk_threshold,
    )


def action_count_aux_loss(output, y_gt, batches, batch_id):
    if args.count_aux_weight <= 0:
        return output["med_logits"].new_tensor(0.0)
    required = (
        "total_count_pred",
        "continue_count_pred",
        "add_count_pred",
        "remove_count_pred",
    )
    if any(output.get(name) is None for name in required):
        raise ValueError("--count_aux_weight requires --action_transition_mode heads with count outputs")
    previous = previous_meds_for_batch(batches, batch_id, output["total_count_pred"])
    return action_heads_count_loss(
        output["total_count_pred"],
        output["continue_count_pred"],
        output["add_count_pred"],
        output["remove_count_pred"],
        previous,
        y_gt,
        total_weight=args.total_count_weight,
        continue_weight=args.continue_count_weight,
        add_weight=args.add_count_weight,
        remove_weight=args.remove_count_weight,
    )


def action_head_batch_metrics(output, y_gt, batches, batch_id):
    continue_logits = output.get("continue_logits")
    add_logits = output.get("add_logits")
    remove_logits = output.get("remove_logits")
    if continue_logits is None or add_logits is None or remove_logits is None:
        return None
    previous = previous_meds_for_batch(batches, batch_id, add_logits).to(dtype=torch.bool)
    target = y_gt.to(device=add_logits.device, dtype=torch.bool)
    threshold = float(args.action_transition_threshold)
    predictions = {
        "continue": torch.sigmoid(continue_logits) >= threshold,
        "add": torch.sigmoid(add_logits) >= threshold,
        "remove": torch.sigmoid(remove_logits) >= threshold,
    }
    targets = {
        "continue": previous & target,
        "add": (~previous) & target,
        "remove": previous & (~target),
    }
    masks = {
        "continue": previous,
        "add": ~previous,
        "remove": previous,
    }
    metrics = {}
    for name in ("continue", "add", "remove"):
        pred = predictions[name] & masks[name]
        truth = targets[name] & masks[name]
        tp = (pred & truth).sum().to(dtype=torch.float64)
        fp = (pred & (~truth) & masks[name]).sum().to(dtype=torch.float64)
        fn = ((~pred) & truth).sum().to(dtype=torch.float64)
        metrics[f"{name}_tp"] = float(tp.item())
        metrics[f"{name}_fp"] = float(fp.item())
        metrics[f"{name}_fn"] = float(fn.item())
    return metrics


def empty_action_head_metric_counts():
    return {
        f"{name}_{kind}": 0.0
        for name in ("continue", "add", "remove")
        for kind in ("tp", "fp", "fn")
    }


def update_action_head_metric_counts(counts, batch_metrics):
    if batch_metrics is None:
        return
    for key, value in batch_metrics.items():
        counts[key] += value


def finalize_action_head_metrics(counts):
    result = {}
    for name in ("continue", "add", "remove"):
        tp = counts[f"{name}_tp"]
        fp = counts[f"{name}_fp"]
        fn = counts[f"{name}_fn"]
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
        result[f"{name}_f1"] = f1
    result["transition_macro_f1"] = float(np.mean([
        result["continue_f1"],
        result["add_f1"],
        result["remove_f1"],
    ]))
    return result


def action_head_metrics_for_batches(model, batches):
    if args.action_transition_mode == "none":
        return finalize_action_head_metrics(empty_action_head_metric_counts())
    model.eval()
    counts = empty_action_head_metric_counts()
    with torch.no_grad():
        for batch_id in range(batches["n_batches"]):
            output = forward_model(model, batches, batch_id)
            update_action_head_metric_counts(
                counts,
                action_head_batch_metrics(output, batches["label_batches"][batch_id], batches, batch_id),
            )
    return finalize_action_head_metrics(counts)


def treatment_episode_action_aux_loss(output, y_gt, batches, batch_id):
    if (
        args.treatment_episode_action_aux_weight <= 0
        and args.treatment_episode_fused_action_loss_weight <= 0
        and args.treatment_episode_candidate_gain_weight <= 0
    ):
        med_logits = output["med_logits"]
        return med_logits.new_tensor(0.0)
    add_logits = output.get("treatment_episode_add_logits")
    continue_logits = output.get("treatment_episode_continue_logits")
    remove_logits = output.get("treatment_episode_remove_logits")
    if add_logits is None or continue_logits is None or remove_logits is None:
        raise ValueError("--treatment_episode_action_aux_weight requires treatment episode adapter action logits")
    target = y_gt.to(device=add_logits.device, dtype=add_logits.dtype)
    previous = previous_meds_for_batch(batches, batch_id, add_logits)
    has_previous = previous.sum(dim=1, keepdim=True) > 0
    has_experience = output.get("treatment_episode_has_experience")
    if has_experience is not None:
        has_experience = has_experience.to(device=add_logits.device, dtype=torch.bool)
        has_previous = has_previous & has_experience
    if not has_previous.any():
        return add_logits.sum() * 0.0
    previous_bool = previous > 0
    target_bool = target > 0
    add_mask = (~previous_bool) & has_previous
    previous_mask = previous_bool & has_previous
    add_target = ((~previous_bool) & target_bool).to(dtype=add_logits.dtype)
    continue_target = (previous_bool & target_bool).to(dtype=continue_logits.dtype)
    remove_target = (previous_bool & (~target_bool)).to(dtype=remove_logits.dtype)

    def masked_bce(logits, targets, mask):
        mask = mask.to(device=logits.device, dtype=logits.dtype)
        if mask.sum() <= 0:
            return logits.sum() * 0.0
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        positive = mask * (targets > 0).to(dtype=logits.dtype)
        negative = mask * (targets <= 0).to(dtype=logits.dtype)
        positive_count = positive.sum()
        negative_count = negative.sum()
        balanced_loss = logits.new_tensor(0.0)
        component_count = logits.new_tensor(0.0)
        if positive_count > 0:
            balanced_loss = balanced_loss + (loss * positive).sum() / positive_count.clamp_min(1.0)
            component_count = component_count + 1.0
        if negative_count > 0:
            balanced_loss = balanced_loss + (loss * negative).sum() / negative_count.clamp_min(1.0)
            component_count = component_count + 1.0
        if component_count <= 0:
            return logits.sum() * 0.0
        return balanced_loss / component_count.clamp_min(1.0)

    weighted_loss = add_logits.new_tensor(0.0)
    weight_total = add_logits.new_tensor(0.0)
    if args.treatment_episode_action_add_weight > 0:
        weighted_loss = weighted_loss + float(args.treatment_episode_action_add_weight) * masked_bce(
            add_logits,
            add_target,
            add_mask,
        )
        weight_total = weight_total + float(args.treatment_episode_action_add_weight)
    if args.treatment_episode_action_continue_weight > 0:
        weighted_loss = weighted_loss + float(args.treatment_episode_action_continue_weight) * masked_bce(
            continue_logits,
            continue_target,
            previous_mask,
        )
        weight_total = weight_total + float(args.treatment_episode_action_continue_weight)
    if args.treatment_episode_action_remove_weight > 0:
        weighted_loss = weighted_loss + float(args.treatment_episode_action_remove_weight) * masked_bce(
            remove_logits,
            remove_target,
            previous_mask,
        )
        weight_total = weight_total + float(args.treatment_episode_action_remove_weight)
    if weight_total <= 0:
        component_loss = add_logits.new_tensor(0.0)
    else:
        component_loss = weighted_loss / weight_total.clamp_min(1.0)

    loss = add_logits.new_tensor(0.0)
    if args.treatment_episode_action_aux_weight > 0:
        loss = loss + float(args.treatment_episode_action_aux_weight) * component_loss
    action_bias = output.get("treatment_episode_action_bias")
    if args.treatment_episode_fused_action_loss_weight > 0:
        if action_bias is None:
            raise ValueError("--treatment_episode_fused_action_loss_weight requires treatment_episode_action_bias")
        action_bias = action_bias.to(device=add_logits.device, dtype=add_logits.dtype)
        fused_loss = masked_bce(action_bias, target, has_previous.expand_as(action_bias))
        loss = loss + float(args.treatment_episode_fused_action_loss_weight) * fused_loss
    if args.treatment_episode_candidate_gain_weight > 0:
        gain_loss = treatment_episode_candidate_gain_loss(
            output,
            y_gt,
            batches,
            batch_id,
            has_previous,
        )
        loss = loss + float(args.treatment_episode_candidate_gain_weight) * gain_loss
    return loss


def batch_top_count_predictions(scores, counts):
    med_voc_size = int(scores.shape[1])
    counts = counts.to(device=scores.device, dtype=torch.long).clamp(min=1, max=med_voc_size)
    sorted_indexes = torch.argsort(scores, dim=1, descending=True)
    ranks = torch.empty_like(sorted_indexes)
    rank_values = torch.arange(med_voc_size, device=scores.device, dtype=sorted_indexes.dtype).view(1, -1)
    ranks.scatter_(1, sorted_indexes, rank_values.expand(scores.shape[0], -1))
    return ranks < counts.view(-1, 1)


def candidate_jaccards_for_training(candidates, target):
    target = target.to(device=candidates.device, dtype=torch.bool).unsqueeze(1)
    intersection = (candidates & target).sum(dim=2).to(dtype=torch.float32)
    union = (candidates | target).sum(dim=2).to(dtype=torch.float32)
    return torch.where(union > 0, intersection / union.clamp_min(1.0), torch.zeros_like(intersection))


def treatment_episode_candidate_gain_loss(output, y_gt, batches, batch_id, eligible_mask):
    add_logits = output.get("treatment_episode_add_logits")
    continue_logits = output.get("treatment_episode_continue_logits")
    remove_logits = output.get("treatment_episode_remove_logits")
    med_logits = output["med_logits"]
    if add_logits is None or continue_logits is None or remove_logits is None:
        raise ValueError("--treatment_episode_candidate_gain_weight requires treatment episode adapter action logits")
    target = y_gt.to(device=med_logits.device, dtype=torch.bool)
    previous = previous_meds_for_batch(batches, batch_id, med_logits).to(device=med_logits.device, dtype=torch.bool)
    eligible = eligible_mask.to(device=med_logits.device, dtype=torch.bool).view(-1)
    if not eligible.any():
        return med_logits.sum() * 0.0
    base_counts = target.sum(dim=1).clamp(min=1, max=med_logits.shape[1])
    base_pred = batch_top_count_predictions(med_logits.detach(), base_counts).to(dtype=torch.bool)
    add_signal = add_logits + 0.10 * med_logits.detach()
    drop_signal = remove_logits - 0.10 * med_logits.detach() - 0.05 * continue_logits
    add_indexes, add_valid = masked_topk_indexes(add_signal, ~base_pred, 2)
    drop_indexes, drop_valid = masked_topk_indexes(drop_signal, base_pred, 2)
    rows = torch.arange(base_pred.shape[0], device=base_pred.device)
    candidates = [base_pred]
    candidate_scores = [add_logits.new_zeros(add_logits.shape[0])]

    def append_candidate(add_pos=None, drop_pos=None):
        pred = base_pred.clone()
        score = add_logits.new_zeros(add_logits.shape[0])
        if drop_pos is not None:
            valid = drop_valid[:, int(drop_pos)]
            if valid.any():
                drop_idx = drop_indexes[:, int(drop_pos)]
                pred[rows[valid], drop_idx[valid]] = False
                score = score + torch.where(
                    valid,
                    remove_logits[rows, drop_idx] - continue_logits[rows, drop_idx],
                    torch.zeros_like(score),
                )
        if add_pos is not None:
            valid = add_valid[:, int(add_pos)]
            if valid.any():
                add_idx = add_indexes[:, int(add_pos)]
                pred[rows[valid], add_idx[valid]] = True
                score = score + torch.where(
                    valid,
                    add_logits[rows, add_idx],
                    torch.zeros_like(score),
                )
        candidates.append(pred)
        candidate_scores.append(score)

    append_candidate(add_pos=0)
    append_candidate(add_pos=1)
    append_candidate(drop_pos=0)
    append_candidate(drop_pos=1)
    append_candidate(add_pos=0, drop_pos=0)
    append_candidate(add_pos=1, drop_pos=1)
    candidate_tensor = torch.stack(candidates, dim=1)
    candidate_scores = torch.stack(candidate_scores, dim=1)
    candidate_jaccards = candidate_jaccards_for_training(candidate_tensor, target)
    candidate_gains = (candidate_jaccards - candidate_jaccards[:, 0:1]).detach()
    candidate_scores = candidate_scores[eligible]
    candidate_gains = candidate_gains[eligible]
    if candidate_scores.numel() <= 0:
        return med_logits.sum() * 0.0
    temperature = float(args.treatment_episode_candidate_gain_temperature)
    list_targets = torch.softmax(candidate_gains / temperature, dim=1)
    list_loss = F.kl_div(
        F.log_softmax(candidate_scores, dim=1),
        list_targets.to(device=candidate_scores.device, dtype=candidate_scores.dtype),
        reduction="batchmean",
    )
    centered_scores = candidate_scores - candidate_scores[:, 0:1]
    value_loss = F.smooth_l1_loss(centered_scores, candidate_gains.to(device=candidate_scores.device, dtype=candidate_scores.dtype).clamp_min(0.0))
    gain_gap = candidate_gains.unsqueeze(2) - candidate_gains.unsqueeze(1)
    pair_mask = gain_gap > float(args.treatment_episode_candidate_gain_min_pair_gap)
    if pair_mask.any():
        score_gap = candidate_scores.unsqueeze(2) - candidate_scores.unsqueeze(1)
        pair_loss = F.softplus(float(args.treatment_episode_candidate_gain_pair_margin) - score_gap)[pair_mask].mean()
    else:
        pair_loss = candidate_scores.new_tensor(0.0)
    return (
        list_loss
        + float(args.treatment_episode_candidate_gain_value_weight) * value_loss
        + float(args.treatment_episode_candidate_gain_pair_weight) * pair_loss
    )


def set_level_loss(output, y_gt):
    if args.soft_jaccard_weight <= 0 and args.soft_dice_weight <= 0:
        return output.new_tensor(0.0)
    pred = torch.sigmoid(output)
    target = y_gt.to(dtype=pred.dtype)
    intersection = (pred * target).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)
    loss = output.new_tensor(0.0)
    if args.soft_jaccard_weight > 0:
        union = pred_sum + target_sum - intersection
        soft_jaccard = (intersection + 1e-8) / (union + 1e-8)
        loss = loss + args.soft_jaccard_weight * (1.0 - soft_jaccard).mean()
    if args.soft_dice_weight > 0:
        soft_dice = (2.0 * intersection + 1e-8) / (pred_sum + target_sum + 1e-8)
        loss = loss + args.soft_dice_weight * (1.0 - soft_dice).mean()
    return loss


def soft_jaccard_loss_from_prob(pred, y_gt):
    target = y_gt.to(device=pred.device, dtype=pred.dtype)
    intersection = (pred * target).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)
    union = pred_sum + target_sum - intersection
    soft_jaccard = (intersection + 1e-8) / (union + 1e-8)
    return (1.0 - soft_jaccard).mean()


def count_anchor_loss_from_prob(pred, y_gt):
    target_count = y_gt.to(device=pred.device, dtype=pred.dtype).sum(dim=1)
    pred_count = pred.sum(dim=1)
    return F.smooth_l1_loss(pred_count, target_count)


def teacher_anchor_losses(med_logits, batches, batch_id):
    zero = med_logits.sum() * 0.0
    rank_loss = zero
    count_loss = zero
    parameter_loss = zero
    teacher_score_batches = batches.get("teacher_score_batches")
    if teacher_score_batches is not None and args.teacher_rank_distill_weight > 0:
        teacher = teacher_score_batches[batch_id].to(device=med_logits.device, dtype=med_logits.dtype)
        student = med_logits - med_logits.mean(dim=1, keepdim=True)
        teacher = teacher - teacher.mean(dim=1, keepdim=True)
        student = student / student.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        teacher = teacher / teacher.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        rank_loss = F.mse_loss(student, teacher)
    teacher_count_batches = batches.get("teacher_count_batches")
    if teacher_count_batches is not None and args.teacher_count_distill_weight > 0:
        teacher_count = teacher_count_batches[batch_id].to(device=med_logits.device, dtype=med_logits.dtype)
        count_loss = F.smooth_l1_loss(torch.sigmoid(med_logits).sum(dim=1), teacher_count)
    if teacher_parameter_reference and args.teacher_parameter_anchor_weight > 0:
        numerator = zero
        denominator = 0
        for name, parameter in model.named_parameters():
            reference = teacher_parameter_reference.get(name)
            if reference is None or not parameter.requires_grad:
                continue
            numerator = numerator + (parameter - reference).square().sum()
            denominator += parameter.numel()
        parameter_loss = numerator / max(denominator, 1)
    return rank_loss, count_loss, parameter_loss


def soft_ddi_penalty_from_prob(pred, weight, power=1.0):
    if weight <= 0:
        return pred.new_tensor(0.0)
    prob = pred.clamp(0.0, 1.0)
    if power != 1.0:
        prob = prob.pow(power)
    ddi_matrix = ddi_A_gpu.to(device=prob.device, dtype=prob.dtype)
    ddi_matrix = torch.triu(ddi_matrix, diagonal=1)
    pair_risk = torch.matmul(prob, ddi_matrix) * prob
    pair_count = prob.sum(dim=1).clamp_min(1.0)
    normalizer = (pair_count * (pair_count - 1.0) / 2.0).clamp_min(1.0)
    return pair_risk.sum(dim=1).div(normalizer).mean()


def asymmetric_bce_loss(output, y_gt):
    if args.asymmetric_bce_weight <= 0:
        return output.new_tensor(0.0)
    target = y_gt.to(dtype=output.dtype)
    prob = torch.sigmoid(output)
    pos_prob = prob
    neg_prob = 1.0 - prob
    if args.asymmetric_bce_neg_clip > 0:
        neg_prob = (neg_prob + args.asymmetric_bce_neg_clip).clamp(max=1.0)
    pos_loss = target * torch.log(pos_prob.clamp_min(1e-8))
    neg_loss = (1.0 - target) * torch.log(neg_prob.clamp_min(1e-8))
    if args.asymmetric_bce_gamma_pos > 0:
        pos_loss = pos_loss * torch.pow(1.0 - pos_prob, args.asymmetric_bce_gamma_pos)
    if args.asymmetric_bce_gamma_neg > 0:
        neg_loss = neg_loss * torch.pow(prob, args.asymmetric_bce_gamma_neg)
    return -(pos_loss + neg_loss).mean()


def pairwise_rank_loss(output, y_gt):
    if args.pairwise_rank_weight <= 0:
        return output.new_tensor(0.0)
    target = y_gt.to(dtype=torch.bool)
    valid_rows = target.any(dim=1) & (~target).any(dim=1)
    if not valid_rows.any():
        return output.new_tensor(0.0)
    logits = output[valid_rows]
    target = target[valid_rows]
    neg_logits = logits.masked_fill(target, -torch.inf)
    pos_counts = target.sum(dim=1)
    neg_k = min(args.pairwise_rank_negatives, logits.shape[1])
    neg_values = torch.topk(neg_logits, k=neg_k, dim=1).values
    finite_neg_mask = torch.isfinite(neg_values)
    if not finite_neg_mask.any():
        return output.new_tensor(0.0)
    row_weights = target.to(dtype=logits.dtype) / pos_counts.clamp_min(1).unsqueeze(1).to(dtype=logits.dtype)
    if args.pairwise_rank_mode == "mean_positive":
        positive_anchor = (logits * row_weights).sum(dim=1, keepdim=True)
        losses = F.softplus(neg_values - positive_anchor + args.pairwise_rank_margin)
    elif args.pairwise_rank_mode == "all_positive":
        pos_values = logits.masked_fill(~target, -torch.inf)
        losses = F.softplus(neg_values.unsqueeze(1) - pos_values.unsqueeze(2) + args.pairwise_rank_margin)
        finite_pos_mask = torch.isfinite(pos_values)
        finite_neg_mask = finite_neg_mask.unsqueeze(1) & finite_pos_mask.unsqueeze(2)
    else:
        raise ValueError(f"unsupported pairwise rank mode: {args.pairwise_rank_mode}")
    return losses.masked_select(finite_neg_mask).mean()


def listwise_rank_loss(output, y_gt):
    if args.listwise_rank_weight <= 0:
        return output.new_tensor(0.0)
    target = y_gt.to(dtype=torch.bool)
    valid_rows = target.any(dim=1) & (~target).any(dim=1)
    if not valid_rows.any():
        return output.new_tensor(0.0)
    logits = output[valid_rows]
    target = target[valid_rows]
    pos_logits = logits.masked_fill(~target, -torch.inf)
    neg_logits = logits.masked_fill(target, -torch.inf)
    neg_k = min(args.listwise_rank_negatives, logits.shape[1])
    neg_values = torch.topk(neg_logits, k=neg_k, dim=1).values
    finite_neg_mask = torch.isfinite(neg_values)
    if not finite_neg_mask.any():
        return output.new_tensor(0.0)
    positive_score = torch.logsumexp(pos_logits, dim=1)
    masked_neg_values = neg_values.masked_fill(~finite_neg_mask, -torch.inf)
    negative_score = torch.logsumexp(masked_neg_values, dim=1)
    losses = F.softplus(negative_score - positive_score + args.listwise_rank_margin)
    return losses.mean()


def retrieval_query_rank_loss(patient_query, retrieval_queries, retrieval_meds, y_gt):
    if args.retrieval_query_rank_weight <= 0:
        return patient_query.new_tensor(0.0)
    if retrieval_queries is None or retrieval_meds is None:
        return patient_query.new_tensor(0.0)
    if retrieval_queries.ndim != 3 or retrieval_meds.ndim != 3:
        raise ValueError("retrieval query ranking expects [batch, topk, dim] queries and [batch, topk, meds] meds")

    target = y_gt.to(device=patient_query.device, dtype=patient_query.dtype)
    retrieved_meds = retrieval_meds.to(device=patient_query.device, dtype=patient_query.dtype)
    retrieved_queries = retrieval_queries.to(device=patient_query.device, dtype=patient_query.dtype)

    query = F.normalize(patient_query, p=2, dim=-1, eps=1e-6)
    candidate_query = F.normalize(retrieved_queries, p=2, dim=-1, eps=1e-6)
    scores = torch.bmm(candidate_query, query.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        target_expanded = target.unsqueeze(1)
        intersection = (retrieved_meds * target_expanded).sum(dim=-1)
        union = (retrieved_meds + target_expanded).clamp(max=1.0).sum(dim=-1)
        quality = intersection / union.clamp_min(1e-6)
        valid_candidate = retrieved_meds.sum(dim=-1) > 0
        quality = quality.masked_fill(~valid_candidate, -torch.inf)
        best_quality, pos_idx = quality.max(dim=1)
        valid_rows = torch.isfinite(best_quality)

    if not valid_rows.any():
        return patient_query.new_tensor(0.0)

    row_index = torch.arange(scores.shape[0], device=scores.device)
    pos_scores = scores[row_index, pos_idx]
    quality_gap = (best_quality.unsqueeze(1) - quality).masked_fill(~valid_candidate, -torch.inf)
    neg_mask = quality_gap > args.retrieval_query_rank_min_delta
    neg_scores = scores.masked_fill(~neg_mask, -torch.inf)
    neg_count = min(args.retrieval_query_rank_negatives, neg_scores.shape[1])
    hard_neg_scores = torch.topk(neg_scores, k=neg_count, dim=1).values
    finite_neg = torch.isfinite(hard_neg_scores)
    valid_pairs = valid_rows.unsqueeze(1) & finite_neg
    if not valid_pairs.any():
        return patient_query.new_tensor(0.0)

    losses = F.softplus(hard_neg_scores - pos_scores.unsqueeze(1) + args.retrieval_query_rank_margin)
    return losses.masked_select(valid_pairs).mean()


def retrieval_query_rank_loss_for_batch(output, batches, batch_id):
    return retrieval_query_rank_loss(
        output["patient_query"],
        batches["retrieval_query_batches"][batch_id] if batches.get("retrieval_query_batches") is not None else None,
        batches["retrieval_med_batches"][batch_id] if batches.get("retrieval_med_batches") is not None else None,
        batches["label_batches"][batch_id],
    )


def score_best_count_targets_from_scores(scores, target):
    with torch.no_grad():
        batch_size, med_voc_size = scores.shape
        min_count, max_count = decode_count_limits(med_voc_size)
        sorted_indexes = torch.argsort(scores.detach(), dim=1, descending=True)
        sorted_target = target.to(device=scores.device, dtype=torch.bool).gather(1, sorted_indexes)
        cumulative_hits = sorted_target.to(dtype=torch.float64).cumsum(dim=1)
        target_counts = target.to(device=scores.device, dtype=torch.bool).sum(dim=1).to(dtype=torch.float64)
        counts_grid = torch.arange(1, med_voc_size + 1, device=scores.device, dtype=torch.float64).view(1, -1)
        unions = target_counts.view(-1, 1) + counts_grid - cumulative_hits
        jaccards = torch.where(unions > 0, cumulative_hits / unions.clamp_min(1.0), torch.zeros_like(unions))
        if min_count == 0:
            zero_jaccard = torch.zeros(batch_size, 1, device=scores.device, dtype=torch.float64)
            max_count = min(max_count, med_voc_size)
            jaccards = torch.cat((zero_jaccard, jaccards[:, :max_count]), dim=1)
            return torch.argmax(jaccards, dim=1).to(device=scores.device, dtype=scores.dtype)
        allowed = torch.zeros(med_voc_size, device=scores.device, dtype=torch.bool)
        allowed[min_count - 1:max_count] = True
        masked = torch.where(allowed.view(1, -1), jaccards, torch.full_like(jaccards, -1.0))
        return (torch.argmax(masked, dim=1) + 1).to(device=scores.device, dtype=scores.dtype)


def graph_rank_config_from_args():
    return {
        "history_weight": args.history_retrieval_history_weight,
        "retrieval_weight": args.history_retrieval_retrieval_weight,
        "freq_weight": args.history_retrieval_freq_weight,
        "ehr_graph_weight": args.history_retrieval_ehr_graph_weight,
        "ddi_graph_weight": args.history_retrieval_ddi_graph_weight,
        "graph_source": args.history_retrieval_graph_source,
    }


def graph_decode_prob_for_batch(med_logits, batches, batch_id):
    y_pred_prob = torch.sigmoid(med_logits)
    return apply_retrieval_copy_prob(y_pred_prob, batches, batch_id)


def graph_decode_scores_for_batch(med_logits, batches, batch_id):
    y_pred_prob = graph_decode_prob_for_batch(med_logits, batches, batch_id)
    history_meds = batches.get("own_med_batches") or batches["med_batches"]
    batch_kwargs = {
        "med_history_binary": (history_meds[batch_id][:, 1:, :].sum(dim=1) > 0),
        "retrieval_priors": batches["retrieval_prior_batches"][batch_id] if batches.get("retrieval_prior_batches") is not None else None,
        "med_frequency_prior": med_freq.to(device=y_pred_prob.device, dtype=y_pred_prob.dtype),
    }
    return history_retrieval_freq_graph_scores(y_pred_prob, batch_kwargs, graph_rank_config_from_args())


def graph_score_best_count_targets_for_batch(med_logits, y_gt, batches, batch_id):
    scores = graph_decode_scores_for_batch(med_logits, batches, batch_id)
    return score_best_count_targets_from_scores(scores, y_gt)


def graph_expected_counts_from_prob(y_pred_prob):
    return y_pred_prob.sum(dim=1) * float(args.decode_count_ratio) + float(args.decode_count_bias)


def differentiable_expected_jaccard_counts(scores, prob):
    batch_size, med_voc_size = scores.shape
    min_count, max_count = decode_count_limits(med_voc_size)
    if max_count <= 0:
        return torch.zeros(batch_size, device=scores.device, dtype=scores.dtype)
    sorted_indexes = torch.argsort(scores, dim=1, descending=True)
    prob = prob.to(device=scores.device, dtype=scores.dtype).clamp(0.0, 1.0)
    sorted_prob = prob.gather(1, sorted_indexes)
    cumulative_expected_hits = sorted_prob.cumsum(dim=1)
    expected_target_count = prob.sum(dim=1, keepdim=True)
    counts_grid = torch.arange(1, med_voc_size + 1, device=scores.device, dtype=scores.dtype).view(1, -1)
    expected_union = expected_target_count + counts_grid - cumulative_expected_hits
    expected_jaccard = cumulative_expected_hits / expected_union.clamp_min(1e-6)
    allowed = torch.zeros(med_voc_size, device=scores.device, dtype=torch.bool)
    allowed[min_count - 1 if min_count > 0 else 0:max_count] = True
    masked = torch.where(allowed.view(1, -1), expected_jaccard, torch.full_like(expected_jaccard, -1e9))
    weights = torch.softmax(masked * 50.0, dim=1)
    return (weights * counts_grid).sum(dim=1)


def graph_decode_counts(med_logits, batches, batch_id):
    y_pred_prob = graph_decode_prob_for_batch(med_logits, batches, batch_id)
    med_voc_size = med_logits.shape[1]
    min_count, max_count = decode_count_limits(med_voc_size)
    if args.train_graph_count_mode == "expected_jaccard":
        scores = graph_decode_scores_for_batch(med_logits, batches, batch_id)
        count_estimate = differentiable_expected_jaccard_counts(scores, y_pred_prob)
    else:
        count_estimate = graph_expected_counts_from_prob(y_pred_prob)
    counts = torch.round(count_estimate).to(dtype=torch.long)
    return counts.clamp(min=min_count, max=max_count)


def train_graph_rank_loss(med_logits, y_gt, batches, batch_id):
    if args.train_graph_rank_weight <= 0:
        return med_logits.new_tensor(0.0)
    scores = graph_decode_scores_for_batch(med_logits, batches, batch_id)
    target = y_gt.to(device=scores.device, dtype=torch.bool)
    valid_rows = target.any(dim=1) & (~target).any(dim=1)
    if not valid_rows.any():
        return scores.new_tensor(0.0)
    scores = scores[valid_rows]
    target = target[valid_rows]
    neg_scores = scores.masked_fill(target, -torch.inf)
    pos_counts = target.sum(dim=1)
    neg_k = min(args.train_graph_rank_negatives, scores.shape[1])
    neg_values = torch.topk(neg_scores, k=neg_k, dim=1).values
    finite_neg_mask = torch.isfinite(neg_values)
    if not finite_neg_mask.any():
        return scores.new_tensor(0.0)
    row_weights = target.to(dtype=scores.dtype) / pos_counts.clamp_min(1).unsqueeze(1).to(dtype=scores.dtype)
    if args.train_graph_rank_mode == "mean_positive":
        positive_anchor = (scores * row_weights).sum(dim=1, keepdim=True)
        losses = F.softplus(neg_values - positive_anchor + args.train_graph_rank_margin)
    elif args.train_graph_rank_mode == "all_positive":
        pos_values = scores.masked_fill(~target, -torch.inf)
        losses = F.softplus(neg_values.unsqueeze(1) - pos_values.unsqueeze(2) + args.train_graph_rank_margin)
        finite_pos_mask = torch.isfinite(pos_values)
        finite_neg_mask = finite_neg_mask.unsqueeze(1) & finite_pos_mask.unsqueeze(2)
    else:
        raise ValueError(f"unsupported train graph rank mode: {args.train_graph_rank_mode}")
    return losses.masked_select(finite_neg_mask).mean()


def train_graph_boundary_rank_loss(med_logits, y_gt, batches, batch_id):
    if args.train_graph_boundary_rank_weight <= 0:
        return med_logits.new_tensor(0.0)
    scores = graph_decode_scores_for_batch(med_logits, batches, batch_id)
    target = y_gt.to(device=scores.device, dtype=torch.bool)
    counts = graph_decode_counts(med_logits, batches, batch_id)
    selected = top_count_predictions(scores.detach(), counts)
    selected_false_positive = selected & (~target)
    missed_positive = target & (~selected)
    valid_rows = selected_false_positive.any(dim=1) & missed_positive.any(dim=1)
    if not valid_rows.any():
        return scores.new_tensor(0.0)
    scores = scores[valid_rows]
    selected_false_positive = selected_false_positive[valid_rows]
    missed_positive = missed_positive[valid_rows]
    false_positive_scores = scores.masked_fill(~selected_false_positive, -torch.inf)
    fp_k = min(args.train_graph_boundary_negatives, scores.shape[1])
    fp_values = torch.topk(false_positive_scores, k=fp_k, dim=1).values
    finite_fp_mask = torch.isfinite(fp_values)
    if not finite_fp_mask.any():
        return scores.new_tensor(0.0)
    missed_counts = missed_positive.sum(dim=1)
    row_weights = missed_positive.to(dtype=scores.dtype) / missed_counts.clamp_min(1).unsqueeze(1).to(dtype=scores.dtype)
    if args.train_graph_boundary_rank_mode == "mean_missed":
        missed_anchor = (scores * row_weights).sum(dim=1, keepdim=True)
        losses = F.softplus(fp_values - missed_anchor + args.train_graph_boundary_margin)
    elif args.train_graph_boundary_rank_mode == "all_missed":
        missed_values = scores.masked_fill(~missed_positive, -torch.inf)
        losses = F.softplus(fp_values.unsqueeze(1) - missed_values.unsqueeze(2) + args.train_graph_boundary_margin)
        finite_missed_mask = torch.isfinite(missed_values)
        finite_fp_mask = finite_fp_mask.unsqueeze(1) & finite_missed_mask.unsqueeze(2)
    else:
        raise ValueError(f"unsupported train graph boundary rank mode: {args.train_graph_boundary_rank_mode}")
    return losses.masked_select(finite_fp_mask).mean()


def action_boundary_pair_loss(scores, positive_mask, negative_mask, negative_k, margin):
    valid_rows = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not valid_rows.any():
        return scores.new_tensor(0.0), scores.new_tensor(0.0)
    row_scores = scores[valid_rows]
    row_positive = positive_mask[valid_rows]
    row_negative = negative_mask[valid_rows]
    positive_counts = row_positive.sum(dim=1)
    positive_weights = row_positive.to(dtype=row_scores.dtype) / positive_counts.clamp_min(1).unsqueeze(1).to(dtype=row_scores.dtype)
    positive_anchor = (row_scores * positive_weights).sum(dim=1, keepdim=True)
    negative_scores = row_scores.masked_fill(~row_negative, -torch.inf)
    k = min(int(negative_k), row_scores.shape[1])
    negative_values = torch.topk(negative_scores, k=k, dim=1).values
    finite_mask = torch.isfinite(negative_values)
    if not finite_mask.any():
        return scores.new_tensor(0.0), scores.new_tensor(0.0)
    losses = F.softplus(negative_values - positive_anchor + margin)
    return losses.masked_select(finite_mask).mean(), finite_mask.to(dtype=scores.dtype).sum()


def action_boundary_rank_loss(med_logits, y_gt, batches, batch_id):
    if args.action_boundary_rank_weight <= 0:
        return med_logits.new_tensor(0.0)
    scores = graph_decode_scores_for_batch(med_logits, batches, batch_id)
    target = y_gt.to(device=scores.device, dtype=torch.bool)
    previous = previous_meds_for_batch(batches, batch_id, scores).to(dtype=torch.bool)
    counts = graph_decode_counts(med_logits, batches, batch_id)
    selected = top_count_predictions(scores.detach(), counts)
    continue_pos = previous & target
    add_pos = (~previous) & target
    remove_neg = previous & (~target)
    new_neg = (~previous) & (~target)
    missed_continue = continue_pos & (~selected)
    missed_add = add_pos & (~selected)
    missed_positive = target & (~selected)
    selected_remove = selected & remove_neg
    selected_new_negative = selected & new_neg
    selected_false_positive = selected & (~target)
    weighted_loss = scores.new_tensor(0.0)
    total_weight = scores.new_tensor(0.0)

    def add_term(term_weight, positive_mask, negative_mask):
        nonlocal weighted_loss, total_weight
        if term_weight <= 0:
            return
        term_loss, term_support = action_boundary_pair_loss(
            scores,
            positive_mask,
            negative_mask,
            args.action_boundary_negatives,
            args.action_boundary_margin,
        )
        if term_support.item() <= 0:
            return
        weight = scores.new_tensor(float(term_weight))
        weighted_loss = weighted_loss + weight * term_loss
        total_weight = total_weight + weight

    add_term(args.action_boundary_continue_weight, missed_continue, selected_remove)
    add_term(args.action_boundary_add_weight, missed_add, selected_new_negative)
    add_term(args.action_boundary_global_weight, missed_positive, selected_false_positive)
    if total_weight.item() <= 0:
        return scores.new_tensor(0.0)
    return weighted_loss / total_weight


def train_graph_count_loss(med_logits, y_gt, batches=None, batch_id=None):
    if args.train_graph_count_weight <= 0:
        return med_logits.new_tensor(0.0)
    if batches is None or batch_id is None:
        raise ValueError("--train_graph_count_weight requires batch context")
    y_pred_prob = graph_decode_prob_for_batch(med_logits, batches, batch_id)
    if args.train_graph_count_mode == "expected_jaccard":
        scores = graph_decode_scores_for_batch(med_logits, batches, batch_id)
        expected_counts = differentiable_expected_jaccard_counts(scores, y_pred_prob)
    else:
        expected_counts = graph_expected_counts_from_prob(y_pred_prob)
    if args.train_graph_count_target == "label_count":
        target_counts = y_gt.sum(dim=1).to(device=med_logits.device, dtype=med_logits.dtype)
    elif args.train_graph_count_target == "best_count":
        if batches is None or batch_id is None:
            raise ValueError("--train_graph_count_target best_count requires batch context")
        scores = graph_decode_scores_for_batch(med_logits, batches, batch_id)
        target_counts = score_best_count_targets_from_scores(scores, y_gt).to(device=med_logits.device, dtype=med_logits.dtype)
    else:
        raise ValueError(f"unsupported train graph count target: {args.train_graph_count_target}")
    return F.smooth_l1_loss(expected_counts, target_counts)


def train_graph_score_distill_loss(med_logits, batches, batch_id):
    if args.train_graph_score_distill_weight <= 0:
        return med_logits.new_tensor(0.0)
    with torch.no_grad():
        teacher_prob = graph_decode_prob_for_batch(med_logits.detach(), batches, batch_id)
        teacher_scores = graph_decode_scores_for_batch(med_logits.detach(), batches, batch_id)
    student_logits = med_logits
    topk = args.train_graph_score_distill_topk
    if topk > 0 and topk < teacher_scores.shape[1]:
        candidate_idx = torch.topk(teacher_scores, k=topk, dim=1).indices
        teacher_scores = teacher_scores.gather(1, candidate_idx)
        student_logits = student_logits.gather(1, candidate_idx)
    temperature = float(args.train_graph_score_distill_temperature)
    teacher = torch.softmax(teacher_scores / temperature, dim=1)
    student_log_prob = torch.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(student_log_prob, teacher, reduction="batchmean") * (temperature ** 2)


def med_count_targets_for_logits(med_logits, y_gt, batches=None, batch_id=None):
    if args.med_count_target == "label_count":
        return y_gt.sum(dim=1)
    if args.med_count_target == "score_best_count":
        return score_best_count_targets_from_scores(med_logits, y_gt)
    if args.med_count_target == "graph_score_best_count":
        if batches is None or batch_id is None:
            raise ValueError("--med_count_target graph_score_best_count requires batch context")
        return graph_score_best_count_targets_for_batch(med_logits, y_gt, batches, batch_id)
    raise ValueError(f"unsupported med_count_target: {args.med_count_target}")


def soft_ddi_loss(output):
    if args.soft_ddi_weight <= 0:
        return output.new_tensor(0.0)
    prob = torch.sigmoid(output).clamp(0.0, 1.0)
    return soft_ddi_penalty_from_prob(prob, args.soft_ddi_weight, power=args.soft_ddi_power)


def seq_decoder_loss(seq_logits, label_batch):
    if seq_logits is None:
        return label_batch.new_tensor(0.0)
    _, decoder_targets = build_seq_decoder_targets(label_batch)
    return F.cross_entropy(
        seq_logits.reshape(-1, seq_logits.shape[-1]),
        decoder_targets.reshape(-1),
        ignore_index=-100,
    )


def seq_decoder_loss_from_parts(model, diag_parts, proc_parts, med_parts, label_parts, batch_size, clinical_parts=None):
    if not label_parts:
        raise ValueError("prefix sequence decoder loss requires at least one label")
    loss_sum = None
    sample_count = 0
    clinical_feature_parts = clinical_feature_mask_parts = None
    clinical_feature_abnormal_parts = clinical_feature_delta_parts = None
    if clinical_parts is not None:
        (
            clinical_feature_parts,
            clinical_feature_mask_parts,
            clinical_feature_abnormal_parts,
            clinical_feature_delta_parts,
        ) = clinical_parts
    for start in range(0, len(label_parts), batch_size):
        end = min(start + batch_size, len(label_parts))
        current_batch_size = end - start
        diag_batch, _, _ = tensors_to_dynamic_batches(diag_parts[start:end], current_batch_size, sort=False)
        proc_batch, _, _ = tensors_to_dynamic_batches(proc_parts[start:end], current_batch_size, sort=False)
        med_batch, _, _ = tensors_to_dynamic_batches(med_parts[start:end], current_batch_size, sort=False)
        label_batch = torch.stack(label_parts[start:end], dim=0)
        decoder_inputs, _ = build_seq_decoder_targets(label_batch)
        model_kwargs = {}
        if clinical_parts is not None:
            clinical_feature_batch, _, _ = tensors_to_dynamic_batches(
                clinical_feature_parts[start:end],
                current_batch_size,
                sort=False,
            )
            clinical_feature_mask_batch, _, _ = tensors_to_dynamic_batches(
                clinical_feature_mask_parts[start:end],
                current_batch_size,
                sort=False,
            )
            clinical_feature_abnormal_batch, _, _ = tensors_to_dynamic_batches(
                clinical_feature_abnormal_parts[start:end],
                current_batch_size,
                sort=False,
            )
            clinical_feature_delta_batch, _, _ = tensors_to_dynamic_batches(
                clinical_feature_delta_parts[start:end],
                current_batch_size,
                sort=False,
            )
            model_kwargs.update({
                "clinical_features": clinical_feature_batch[0],
                "clinical_feature_masks": clinical_feature_mask_batch[0],
                "clinical_feature_abnormal": clinical_feature_abnormal_batch[0],
                "clinical_feature_delta_hours": clinical_feature_delta_batch[0],
            })
        output = model(
            diag_batch[0],
            proc_batch[0],
            med_batch[0],
            seq_decoder_inputs=decoder_inputs,
            **model_kwargs,
        )
        batch_loss = seq_decoder_loss(output["seq_logits"], label_batch)
        if loss_sum is None:
            loss_sum = batch_loss * current_batch_size
        else:
            loss_sum = loss_sum + batch_loss * current_batch_size
        sample_count += current_batch_size
    return loss_sum / max(sample_count, 1)

def infer_prob(model, batches):
    model.eval()
    with torch.no_grad():
        y_gt_full=[]
        y_pred_prob_full=[]
        med_count_pred_full=[]
        action_count_pred_full = {
            "total": [],
            "continue": [],
            "add": [],
            "remove": [],
        }
        action_direct_prediction_full=[]
        if args.action_decode_mode != "direct_action" or args.use_trainable_action_fusion:
            batches.pop("action_direct_predictions", None)
        loss_full=0
        n_samples=0
        for batch_id in range(batches['n_batches']):
            output=forward_model(model, batches, batch_id)
            med_logits = apply_retrieval_prior(output["med_logits"], batches, batch_id)
            state_logits = output["clinical_state_logits"]
            transition_logits = output["med_transition_logits"]
            med_count_pred = output["med_count_pred"]
            action_count_pred = {
                "total": output.get("total_count_pred"),
                "continue": output.get("continue_count_pred"),
                "add": output.get("add_count_pred"),
                "remove": output.get("remove_count_pred"),
            }
            seq_logits = output["seq_logits"]
            # y_pred_prob: [batch, size_med_voc], the probability of each med label
            decode_logits = med_logits
            if args.use_trainable_action_fusion:
                decode_logits = action_fused_logits_for_batch(output, med_logits, batches, batch_id)
            elif args.action_decode_mode == "score_fusion":
                decode_logits = action_score_fusion_for_batch(output, med_logits, batches, batch_id)
            if args.action_decode_mode == "direct_action" and not args.use_trainable_action_fusion:
                y_pred_prob = direct_action_scores_for_batch(output, med_logits, batches, batch_id)
                action_direct_prediction_full.append(
                    direct_action_prediction_for_batch(output, med_logits, batches, batch_id)
                )
            else:
                y_pred_prob=F.sigmoid(decode_logits)
                y_pred_prob=apply_retrieval_copy_prob(y_pred_prob, batches, batch_id)
            y_gt=batches['label_batches'][batch_id]
            state_aux_loss = med_logits.new_tensor(0.0)
            transition_aux_loss = med_logits.new_tensor(0.0)
            med_count_loss = med_logits.new_tensor(0.0)
            if args.multitask_mode == "clinical_state":
                state_targets = batches["clinical_state_batches"][batch_id]
                state_masks = batches["clinical_state_mask_batches"][batch_id]
                transition_targets = batches["med_transition_batches"][batch_id]
                transition_masks = batches["med_transition_mask_batches"][batch_id]
                state_aux_loss = masked_multilabel_bce(state_logits, state_targets, state_masks)
                transition_aux_loss = masked_transition_cross_entropy(transition_logits, transition_targets, transition_masks)
            if args.med_count_mode == "aux":
                med_count_targets = med_count_targets_for_logits(med_logits, y_gt, batches, batch_id)
                med_count_loss = F.smooth_l1_loss(med_count_pred, med_count_targets)
            seq_loss = seq_decoder_loss(seq_logits, y_gt) if seq_decoder_enabled() else med_logits.new_tensor(0.0)
            retrieval_query_loss = retrieval_query_rank_loss_for_batch(output, batches, batch_id)
            graph_rank_loss = train_graph_rank_loss(med_logits, y_gt, batches, batch_id)
            graph_boundary_rank_loss = train_graph_boundary_rank_loss(med_logits, y_gt, batches, batch_id)
            action_boundary_rank_loss_value = action_boundary_rank_loss(med_logits, y_gt, batches, batch_id)
            graph_count_loss = train_graph_count_loss(med_logits, y_gt, batches, batch_id)
            action_transition_loss = action_transition_aux_loss(output, y_gt, batches, batch_id)
            action_ddi_add_loss = action_ddi_add_aux_loss(output, batches, batch_id)
            count_aux_loss = action_count_aux_loss(output, y_gt, batches, batch_id)
            action_fusion_loss = action_fusion_aux_loss(output, y_gt, med_logits, batches, batch_id)
            treatment_episode_action_loss = treatment_episode_action_aux_loss(output, y_gt, batches, batch_id)
            main_med_scorer_reg_loss = main_med_scorer_regularization(output, med_logits)
            loss = (
                args.bce_weight * loss_func(med_logits, y_gt)
                + args.action_transition_weight * action_transition_loss
                + args.action_ddi_loss_weight * action_ddi_add_loss
                + args.count_aux_weight * count_aux_loss
                + action_fusion_loss
                + treatment_episode_action_loss
                + set_level_loss(med_logits, y_gt)
                + args.asymmetric_bce_weight * asymmetric_bce_loss(med_logits, y_gt)
                + args.pairwise_rank_weight * pairwise_rank_loss(med_logits, y_gt)
                + args.listwise_rank_weight * listwise_rank_loss(med_logits, y_gt)
                + args.train_graph_rank_weight * graph_rank_loss
                + args.train_graph_boundary_rank_weight * graph_boundary_rank_loss
                + args.action_boundary_rank_weight * action_boundary_rank_loss_value
                + args.train_graph_count_weight * graph_count_loss
                + args.retrieval_query_rank_weight * retrieval_query_loss
                + args.soft_ddi_weight * soft_ddi_loss(med_logits)
                + main_med_scorer_reg_loss
                + args.state_aux_weight * state_aux_loss
                + args.transition_aux_weight * transition_aux_loss
                + args.med_count_aux_weight * med_count_loss
                + args.seq_decoder_weight * seq_loss
            )
            y_gt_full.append(y_gt)
            y_pred_prob_full.append(y_pred_prob)
            if med_count_pred is not None:
                med_count_pred_full.append(med_count_pred)
            for name, pred in action_count_pred.items():
                if pred is not None:
                    action_count_pred_full[name].append(pred)
            loss_full+=loss.item() * y_gt.shape[0]
            n_samples+=y_gt.shape[0]
        med_count_pred = torch.cat(med_count_pred_full) if med_count_pred_full else None
        action_count_pred = {
            name: torch.cat(parts) if parts else None
            for name, parts in action_count_pred_full.items()
        }
        if any(value is not None for value in action_count_pred.values()):
            batches["action_count_predictions"] = action_count_pred
        else:
            batches.pop("action_count_predictions", None)
        if args.action_decode_mode == "direct_action":
            batches["action_direct_predictions"] = torch.cat(action_direct_prediction_full, dim=0)
        return torch.cat(y_gt_full).cpu(), torch.cat(y_pred_prob_full), loss_full/n_samples, med_count_pred
#%%
# result=infer(model, eval_batches,threshold=0.3)
#%%

def decode_predictions(y_pred_prob, batches, config):
    mode = config["mode"]
    if mode == "action_direct":
        prediction = batches.get("action_direct_predictions")
        if prediction is None:
            raise ValueError("action_direct decoder requires action_direct_predictions")
        return prediction.to(device=y_pred_prob.device, dtype=torch.bool)
    if mode in {"count_total_topk", "count_action_split_topk"}:
        base_config = config.get("base_decoder")
        if isinstance(base_config, dict):
            base_config = runtime_decode_config_from_json(base_config)
        scores = y_pred_prob
        if mode == "count_total_topk":
            return count_total_topk_predictions(scores, batches)
        previous = batches.get("previous_med_binary")
        if previous is None:
            raise ValueError("count_action_split_topk decoder requires previous_med_binary")
        return action_split_topk_predictions(
            scores,
            scores,
            previous.to(device=scores.device),
            batches,
        )
    if mode == "threshold":
        return y_pred_prob >= config["threshold"]
    if mode == "per_med_threshold":
        thresholds = config.get("thresholds")
        if thresholds is None:
            raise ValueError("per_med_threshold decoder requires fitted thresholds")
        thresholds = thresholds.to(device=y_pred_prob.device, dtype=y_pred_prob.dtype)
        return y_pred_prob >= thresholds.view(1, -1)
    if mode == "history_threshold":
        if "med_history_binary" not in batches or batches["med_history_binary"] is None:
            raise ValueError("history_threshold decoder requires med_history_binary")
        history_mask = batches["med_history_binary"].to(device=y_pred_prob.device, dtype=torch.bool)
        base_thresholds = torch.full_like(y_pred_prob, float(config["threshold"]))
        history_thresholds = torch.full_like(y_pred_prob, float(config["history_threshold"]))
        thresholds = torch.where(history_mask, history_thresholds, base_thresholds)
        return y_pred_prob >= thresholds
    if mode == "oracle_count":
        if "label_counts" not in batches or batches["label_counts"] is None:
            raise ValueError("oracle_count decoder requires label_counts")
        scores = y_pred_prob
        batch_size, med_voc_size = scores.shape
        min_count, max_count = decode_count_limits(med_voc_size)
        counts = batches["label_counts"].to(device=scores.device, dtype=torch.long).clamp(
            min=min_count,
            max=max_count,
        )
        y_pred = torch.zeros_like(scores, dtype=torch.bool)
        sorted_indexes = torch.argsort(scores, dim=1, descending=True)
        for row_idx in range(batch_size):
            count = int(counts[row_idx].item())
            if count > 0:
                y_pred[row_idx, sorted_indexes[row_idx, :count]] = True
        return y_pred
    if mode.startswith("history_retrieval_seq_rank_"):
        if "med_history_binary" not in batches or batches["med_history_binary"] is None:
            raise ValueError(f"{mode} decoder requires med_history_binary")
        if "retrieval_priors" not in batches or batches["retrieval_priors"] is None:
            raise ValueError(f"{mode} decoder requires retrieval_priors")
        if "seq_rankings" not in batches or batches["seq_rankings"] is None:
            raise ValueError(f"{mode} decoder requires seq_rankings")
        history_signal = batches["med_history_binary"].to(device=y_pred_prob.device, dtype=y_pred_prob.dtype)
        retrieval_signal = batches["retrieval_priors"].to(device=y_pred_prob.device, dtype=y_pred_prob.dtype)
        rankings = batches["seq_rankings"].to(device=y_pred_prob.device)
        batch_size, med_voc_size = y_pred_prob.shape
        seq_signal = torch.zeros_like(y_pred_prob)
        rank_width = max(int(rankings.shape[1]), 1)
        rank_values = torch.linspace(1.0, 0.0, steps=rank_width, device=y_pred_prob.device, dtype=y_pred_prob.dtype)
        for row_idx in range(batch_size):
            med_indexes = rankings[row_idx]
            valid = (med_indexes >= 0) & (med_indexes < med_voc_size)
            if valid.any():
                seq_signal[row_idx, med_indexes[valid]] = rank_values[valid]
        scores = (
            y_pred_prob
            + float(config["history_weight"]) * history_signal
            + float(config["retrieval_weight"]) * retrieval_signal
            + float(config["seq_weight"]) * seq_signal
        )
        count_mode = mode.removeprefix("history_retrieval_seq_rank_")
        min_count, max_count = decode_count_limits(med_voc_size)
        if count_mode == "fixed_topk":
            counts = torch.full((batch_size,), int(config["topk"]), dtype=torch.long, device=scores.device)
        elif count_mode == "threshold_count":
            threshold_counts = (y_pred_prob >= float(config["threshold"])).sum(dim=1).to(dtype=scores.dtype)
            threshold_counts = threshold_counts * float(config["ratio"]) + float(config["bias"])
            counts = torch.round(threshold_counts).long()
        elif count_mode == "expected_count":
            expected_counts = y_pred_prob.sum(dim=1) * float(config["ratio"]) + float(config["bias"])
            counts = torch.round(expected_counts).long()
        elif count_mode == "expected_jaccard_count":
            counts = expected_jaccard_counts(scores, y_pred_prob)
        elif count_mode == "history_count":
            if "med_history_counts" in batches and batches["med_history_counts"] is not None:
                history_base_counts = batches["med_history_counts"].to(device=scores.device, dtype=scores.dtype)
            else:
                med_history = (batches["med_batches"].sum(dim=1) > 0).float()
                history_base_counts = med_history.sum(dim=1)
            history_counts = history_base_counts * float(config["ratio"]) + float(config["bias"])
            counts = torch.round(history_counts).long()
        elif count_mode == "predicted_count":
            if "med_count_pred" not in batches or batches["med_count_pred"] is None:
                raise ValueError(f"{mode} decoder requires med_count_pred")
            predicted_counts = batches["med_count_pred"].to(device=scores.device, dtype=scores.dtype)
            counts = torch.round(predicted_counts * float(config["ratio"]) + float(config["bias"])).long()
        elif count_mode == "retrieval_count":
            counts = retrieval_context_counts(batches, scores, config)
        else:
            raise ValueError(f"unsupported decode mode: {mode}")
        counts = counts.clamp(min=min_count, max=max_count)
        y_pred = torch.zeros_like(scores, dtype=torch.bool)
        if max_count <= 0:
            return y_pred
        sorted_indexes = torch.argsort(scores, dim=1, descending=True)
        for row_idx in range(batch_size):
            count = int(counts[row_idx].item())
            if count > 0:
                y_pred[row_idx, sorted_indexes[row_idx, :count]] = True
        return y_pred
    if mode.startswith("history_retrieval_freq_rank_"):
        if "med_history_binary" not in batches or batches["med_history_binary"] is None:
            raise ValueError(f"{mode} decoder requires med_history_binary")
        if "retrieval_priors" not in batches or batches["retrieval_priors"] is None:
            raise ValueError(f"{mode} decoder requires retrieval_priors")
        if "med_frequency_prior" not in batches or batches["med_frequency_prior"] is None:
            raise ValueError(f"{mode} decoder requires med_frequency_prior")
        history_signal = batches["med_history_binary"].to(device=y_pred_prob.device, dtype=y_pred_prob.dtype)
        retrieval_signal = batches["retrieval_priors"].to(device=y_pred_prob.device, dtype=y_pred_prob.dtype)
        freq_signal = batches["med_frequency_prior"].to(device=y_pred_prob.device, dtype=y_pred_prob.dtype)
        scores = (
            y_pred_prob
            + float(config["history_weight"]) * history_signal
            + float(config["retrieval_weight"]) * retrieval_signal
            + float(config["freq_weight"]) * freq_signal.view(1, -1)
        )
        count_mode = mode.removeprefix("history_retrieval_freq_rank_")
        batch_size, med_voc_size = scores.shape
        min_count, max_count = decode_count_limits(med_voc_size)
        if count_mode == "fixed_topk":
            counts = torch.full((batch_size,), int(config["topk"]), dtype=torch.long, device=scores.device)
        elif count_mode == "threshold_count":
            threshold_counts = (y_pred_prob >= float(config["threshold"])).sum(dim=1).to(dtype=scores.dtype)
            counts = torch.round(threshold_counts * float(config["ratio"]) + float(config["bias"])).long()
        elif count_mode == "expected_count":
            expected_counts = y_pred_prob.sum(dim=1) * float(config["ratio"]) + float(config["bias"])
            counts = torch.round(expected_counts).long()
        elif count_mode == "history_count":
            if "med_history_counts" in batches and batches["med_history_counts"] is not None:
                history_base_counts = batches["med_history_counts"].to(device=scores.device, dtype=scores.dtype)
            else:
                med_history = (batches["med_batches"].sum(dim=1) > 0).float()
                history_base_counts = med_history.sum(dim=1)
            counts = torch.round(history_base_counts * float(config["ratio"]) + float(config["bias"])).long()
        elif count_mode == "predicted_count":
            if "med_count_pred" not in batches or batches["med_count_pred"] is None:
                raise ValueError(f"{mode} decoder requires med_count_pred")
            predicted_counts = batches["med_count_pred"].to(device=scores.device, dtype=scores.dtype)
            counts = torch.round(predicted_counts * float(config["ratio"]) + float(config["bias"])).long()
        elif count_mode == "retrieval_count":
            counts = retrieval_context_counts(batches, scores, config)
        else:
            raise ValueError(f"unsupported decode mode: {mode}")
        counts = counts.clamp(min=min_count, max=max_count)
        y_pred = torch.zeros_like(scores, dtype=torch.bool)
        if max_count <= 0:
            return y_pred
        sorted_indexes = torch.argsort(scores, dim=1, descending=True)
        for row_idx in range(batch_size):
            count = int(counts[row_idx].item())
            if count > 0:
                y_pred[row_idx, sorted_indexes[row_idx, :count]] = True
        return y_pred
    if mode.startswith("history_retrieval_freq_ddi_rank_"):
        if "med_history_binary" not in batches or batches["med_history_binary"] is None:
            raise ValueError(f"{mode} decoder requires med_history_binary")
        if "retrieval_priors" not in batches or batches["retrieval_priors"] is None:
            raise ValueError(f"{mode} decoder requires retrieval_priors")
        if "med_frequency_prior" not in batches or batches["med_frequency_prior"] is None:
            raise ValueError(f"{mode} decoder requires med_frequency_prior")
        history_signal = batches["med_history_binary"].to(device=y_pred_prob.device, dtype=y_pred_prob.dtype)
        retrieval_signal = batches["retrieval_priors"].to(device=y_pred_prob.device, dtype=y_pred_prob.dtype)
        freq_signal = batches["med_frequency_prior"].to(device=y_pred_prob.device, dtype=y_pred_prob.dtype)
        scores = (
            y_pred_prob
            + float(config["history_weight"]) * history_signal
            + float(config["retrieval_weight"]) * retrieval_signal
            + float(config["freq_weight"]) * freq_signal.view(1, -1)
        )
        count_mode = mode.removeprefix("history_retrieval_freq_ddi_rank_")
        batch_size, med_voc_size = scores.shape
        min_count, max_count = decode_count_limits(med_voc_size)
        if count_mode == "expected_count":
            expected_counts = y_pred_prob.sum(dim=1) * float(config["ratio"]) + float(config["bias"])
            counts = torch.round(expected_counts).long()
        else:
            raise ValueError(f"unsupported decode mode: {mode}")
        counts = counts.clamp(min=min_count, max=max_count)
        return ddi_aware_top_count_predictions(scores, counts, float(config["ddi_penalty"]))
    if mode.startswith("history_retrieval_freq_graph_rank_"):
        scores = history_retrieval_freq_graph_scores(y_pred_prob, batches, config)
        count_mode = mode.removeprefix("history_retrieval_freq_graph_rank_")
        batch_size, med_voc_size = scores.shape
        min_count, max_count = decode_count_limits(med_voc_size)
        if count_mode == "expected_count":
            expected_counts = y_pred_prob.sum(dim=1) * float(config["ratio"]) + float(config["bias"])
            counts = torch.round(expected_counts).long()
        elif count_mode == "expected_jaccard_count":
            counts = expected_jaccard_counts(scores, y_pred_prob)
        elif count_mode == "count_pair_mix":
            counts = count_pair_mix_counts(y_pred_prob, scores, batches, config)
        elif count_mode == "predicted_count":
            if "med_count_pred" not in batches or batches["med_count_pred"] is None:
                raise ValueError(f"{mode} decoder requires med_count_pred")
            predicted_counts = batches["med_count_pred"].to(device=scores.device, dtype=scores.dtype)
            counts = torch.round(predicted_counts * float(config["ratio"]) + float(config["bias"])).long()
        elif count_mode == "retrieval_count":
            counts = retrieval_context_counts(batches, scores, config)
        else:
            raise ValueError(f"unsupported decode mode: {mode}")
        counts = counts.clamp(min=min_count, max=max_count)
        if max_count <= 0:
            return torch.zeros_like(scores, dtype=torch.bool)
        return top_count_predictions(scores, counts)
    if mode.startswith("history_rank_") or mode.startswith("retrieval_rank_") or mode.startswith("history_retrieval_rank_"):
        if mode.startswith("history_retrieval_rank_"):
            if "med_history_binary" not in batches or batches["med_history_binary"] is None:
                raise ValueError(f"{mode} decoder requires med_history_binary")
            if "retrieval_priors" not in batches or batches["retrieval_priors"] is None:
                raise ValueError(f"{mode} decoder requires retrieval_priors")
            history_signal = batches["med_history_binary"].to(device=y_pred_prob.device, dtype=y_pred_prob.dtype)
            retrieval_signal = batches["retrieval_priors"].to(device=y_pred_prob.device, dtype=y_pred_prob.dtype)
            scores = (
                y_pred_prob
                + float(config["history_weight"]) * history_signal
                + float(config["retrieval_weight"]) * retrieval_signal
            )
            count_mode = mode.removeprefix("history_retrieval_rank_")
        elif mode.startswith("history_rank_"):
            if "med_history_binary" not in batches or batches["med_history_binary"] is None:
                raise ValueError(f"{mode} decoder requires med_history_binary")
            rerank_signal = batches["med_history_binary"].to(device=y_pred_prob.device, dtype=y_pred_prob.dtype)
            count_mode = mode.removeprefix("history_rank_")
            scores = y_pred_prob + float(config["weight"]) * rerank_signal
        else:
            if "retrieval_priors" not in batches or batches["retrieval_priors"] is None:
                raise ValueError(f"{mode} decoder requires retrieval_priors")
            rerank_signal = batches["retrieval_priors"].to(device=y_pred_prob.device, dtype=y_pred_prob.dtype)
            count_mode = mode.removeprefix("retrieval_rank_")
            scores = y_pred_prob + float(config["weight"]) * rerank_signal

        batch_size, med_voc_size = scores.shape
        min_count, max_count = decode_count_limits(med_voc_size)
        if count_mode == "fixed_topk":
            counts = torch.full((batch_size,), int(config["topk"]), dtype=torch.long, device=scores.device)
        elif count_mode == "threshold_count":
            threshold_counts = (y_pred_prob >= float(config["threshold"])).sum(dim=1).to(dtype=scores.dtype)
            threshold_counts = threshold_counts * float(config["ratio"]) + float(config["bias"])
            counts = torch.round(threshold_counts).long()
        elif count_mode == "expected_count":
            expected_counts = y_pred_prob.sum(dim=1) * float(config["ratio"]) + float(config["bias"])
            counts = torch.round(expected_counts).long()
        elif count_mode == "history_count":
            if "med_history_counts" in batches and batches["med_history_counts"] is not None:
                history_base_counts = batches["med_history_counts"].to(device=scores.device, dtype=scores.dtype)
            else:
                med_history = (batches["med_batches"].sum(dim=1) > 0).float()
                history_base_counts = med_history.sum(dim=1)
            history_counts = history_base_counts * float(config["ratio"]) + float(config["bias"])
            counts = torch.round(history_counts).long()
        elif count_mode == "predicted_count":
            if "med_count_pred" not in batches or batches["med_count_pred"] is None:
                raise ValueError(f"{mode} decoder requires med_count_pred")
            predicted_counts = batches["med_count_pred"].to(device=scores.device, dtype=scores.dtype)
            counts = torch.round(predicted_counts * float(config["ratio"]) + float(config["bias"])).long()
        else:
            raise ValueError(f"unsupported decode mode: {mode}")

        counts = counts.clamp(min=min_count, max=max_count)
        y_pred = torch.zeros_like(scores, dtype=torch.bool)
        if max_count <= 0:
            return y_pred
        sorted_indexes = torch.argsort(scores, dim=1, descending=True)
        for row_idx in range(batch_size):
            count = int(counts[row_idx].item())
            if count > 0:
                y_pred[row_idx, sorted_indexes[row_idx, :count]] = True
        return y_pred
    if mode == "seq_greedy":
        if "seq_predictions" not in batches or batches["seq_predictions"] is None:
            raise ValueError("seq_greedy decoder requires seq_predictions")
        return batches["seq_predictions"]
    if mode == "seq_hidr":
        if "seq_hidr_predictions" not in batches or batches["seq_hidr_predictions"] is None:
            raise ValueError("seq_hidr decoder requires seq_hidr_predictions")
        return batches["seq_hidr_predictions"]
    if mode == "seq_beam":
        if "seq_beam_predictions" not in batches or batches["seq_beam_predictions"] is None:
            raise ValueError("seq_beam decoder requires seq_beam_predictions")
        return batches["seq_beam_predictions"]
    if mode.startswith("seq_prob_"):
        if "seq_prob_scores" not in batches or batches["seq_prob_scores"] is None:
            raise ValueError(f"{mode} decoder requires seq_prob_scores")
        scores = batches["seq_prob_scores"].to(device=y_pred_prob.device)
        batch_size, med_voc_size = scores.shape
        min_count, max_count = decode_count_limits(med_voc_size)
        if mode == "seq_prob_fixed_topk":
            counts = torch.full((batch_size,), int(config["topk"]), dtype=torch.long, device=scores.device)
        elif mode == "seq_prob_expected_count":
            expected_counts = y_pred_prob.sum(dim=1) * float(config["ratio"]) + float(config["bias"])
            counts = torch.round(expected_counts).long()
        elif mode == "seq_prob_history_count":
            if "med_history_counts" in batches and batches["med_history_counts"] is not None:
                history_base_counts = batches["med_history_counts"].to(device=scores.device, dtype=scores.dtype)
            else:
                med_history = (batches["med_batches"].sum(dim=1) > 0).float()
                history_base_counts = med_history.sum(dim=1)
            history_counts = history_base_counts * float(config["ratio"]) + float(config["bias"])
            counts = torch.round(history_counts).long()
        elif mode == "seq_prob_predicted_count":
            if "med_count_pred" not in batches or batches["med_count_pred"] is None:
                raise ValueError("seq_prob_predicted_count decoder requires med_count_pred")
            predicted_counts = batches["med_count_pred"].to(device=scores.device, dtype=scores.dtype)
            counts = torch.round(predicted_counts * float(config["ratio"]) + float(config["bias"])).long()
        else:
            raise ValueError(f"unsupported decode mode: {mode}")
        counts = counts.clamp(min=min_count, max=max_count)
        y_pred = torch.zeros_like(scores, dtype=torch.bool)
        if max_count <= 0:
            return y_pred
        sorted_indexes = torch.argsort(scores, dim=1, descending=True)
        for row_idx in range(batch_size):
            count = int(counts[row_idx].item())
            if count > 0:
                y_pred[row_idx, sorted_indexes[row_idx, :count]] = True
        return y_pred
    if mode.startswith("seq_rank_"):
        if "seq_rankings" not in batches or batches["seq_rankings"] is None:
            raise ValueError(f"{mode} decoder requires seq_rankings")
        rankings = batches["seq_rankings"].to(device=y_pred_prob.device)
        scores = y_pred_prob
        batch_size, med_voc_size = scores.shape
        min_count, max_count = decode_count_limits(med_voc_size)
        if mode == "seq_rank_fixed_topk":
            counts = torch.full((batch_size,), int(config["topk"]), dtype=torch.long, device=scores.device)
        elif mode == "seq_rank_expected_count":
            expected_counts = scores.sum(dim=1) * float(config["ratio"]) + float(config["bias"])
            counts = torch.round(expected_counts).long()
        elif mode == "seq_rank_history_count":
            if "med_history_counts" in batches and batches["med_history_counts"] is not None:
                history_base_counts = batches["med_history_counts"].to(device=scores.device, dtype=scores.dtype)
            else:
                med_history = (batches["med_batches"].sum(dim=1) > 0).float()
                history_base_counts = med_history.sum(dim=1)
            history_counts = history_base_counts * float(config["ratio"]) + float(config["bias"])
            counts = torch.round(history_counts).long()
        elif mode == "seq_rank_predicted_count":
            if "med_count_pred" not in batches or batches["med_count_pred"] is None:
                raise ValueError("seq_rank_predicted_count decoder requires med_count_pred")
            predicted_counts = batches["med_count_pred"].to(device=scores.device, dtype=scores.dtype)
            counts = torch.round(predicted_counts * float(config["ratio"]) + float(config["bias"])).long()
        else:
            raise ValueError(f"unsupported decode mode: {mode}")
        counts = counts.clamp(min=min_count, max=max_count)
        y_pred = torch.zeros_like(scores, dtype=torch.bool)
        for row_idx in range(batch_size):
            count = min(int(counts[row_idx].item()), rankings.shape[1])
            if count > 0:
                med_indexes = rankings[row_idx, :count]
                med_indexes = med_indexes[(med_indexes >= 0) & (med_indexes < med_voc_size)]
                if med_indexes.numel() > 0:
                    y_pred[row_idx, med_indexes] = True
        return y_pred

    scores = y_pred_prob
    batch_size, med_voc_size = scores.shape
    min_count, max_count = decode_count_limits(med_voc_size)
    if mode == "fixed_topk":
        counts = torch.full((batch_size,), int(config["topk"]), dtype=torch.long, device=scores.device)
    elif mode == "expected_count":
        expected_counts = scores.sum(dim=1) * float(config["ratio"]) + float(config["bias"])
        counts = torch.round(expected_counts).long()
    elif mode == "history_count":
        if "med_history_counts" in batches and batches["med_history_counts"] is not None:
            history_base_counts = batches["med_history_counts"].to(device=scores.device, dtype=scores.dtype)
        else:
            med_history = (batches["med_batches"].sum(dim=1) > 0).float()
            history_base_counts = med_history.sum(dim=1)
        history_counts = history_base_counts * float(config["ratio"]) + float(config["bias"])
        counts = torch.round(history_counts).long()
    elif mode == "predicted_count":
        if "med_count_pred" not in batches or batches["med_count_pred"] is None:
            raise ValueError("predicted_count decoder requires med_count_pred")
        predicted_counts = batches["med_count_pred"].to(device=scores.device, dtype=scores.dtype)
        counts = torch.round(predicted_counts * float(config["ratio"]) + float(config["bias"])).long()
    else:
        raise ValueError(f"unsupported decode mode: {mode}")

    counts = counts.clamp(min=min_count, max=max_count)
    y_pred = torch.zeros_like(scores, dtype=torch.bool)
    if max_count <= 0:
        return y_pred
    sorted_indexes = torch.argsort(scores, dim=1, descending=True)
    for row_idx in range(batch_size):
        count = int(counts[row_idx].item())
        if count > 0:
            y_pred[row_idx, sorted_indexes[row_idx, :count]] = True
    return y_pred


def concatenate_med_history_counts(batches):
    med_batches = batches.get("own_med_batches") or batches["med_batches"]
    return torch.cat(
        [
            (med_batch.sum(dim=1) > 0).float().sum(dim=1)
            for med_batch in med_batches
        ],
        dim=0,
    )


def concatenate_med_history_binary(batches):
    med_batches = batches.get("own_med_batches") or batches["med_batches"]
    return torch.cat(
        [
            (med_batch[:, 1:, :].sum(dim=1) > 0)
            for med_batch in med_batches
        ],
        dim=0,
    )


def concatenate_previous_med_binary(batches):
    med_batches = batches.get("own_med_batches") or batches["med_batches"]
    previous_batches = []
    for med_batch in med_batches:
        if med_batch.shape[1] > 1:
            previous_batches.append(med_batch[:, 1, :] > 0)
        else:
            previous_batches.append(torch.zeros_like(med_batch[:, 0, :], dtype=torch.bool))
    return torch.cat(previous_batches, dim=0)


def concatenate_retrieval_priors(batches):
    retrieval_batches = batches.get("retrieval_prior_batches")
    if retrieval_batches is None:
        return None
    return torch.cat(retrieval_batches, dim=0)


def concatenate_retrieval_meds(batches):
    retrieval_batches = batches.get("retrieval_med_batches")
    if retrieval_batches is None:
        return None
    return torch.cat(retrieval_batches, dim=0)


def concatenate_retrieval_scores(batches):
    retrieval_batches = batches.get("retrieval_score_batches")
    if retrieval_batches is None:
        return None
    return torch.cat(retrieval_batches, dim=0)


def concatenate_treatment_episode_experiences(batches):
    experience_batches = batches.get("treatment_episode_experience_batches")
    if experience_batches is None:
        return None
    return torch.cat(experience_batches, dim=0)


def concatenate_treatment_episode_scores(batches):
    score_batches = batches.get("treatment_episode_score_batches")
    if score_batches is None:
        return None
    return torch.cat(score_batches, dim=0)


def concatenate_visit_context_features(batches):
    feature_batches = []
    for batch_id in range(batches["n_batches"]):
        diag_batch = batches["diag_batches"][batch_id].float()
        proc_batch = batches["proc_batches"][batch_id].float()
        seq_len = torch.tensor(
            batches["batch_seq_len"][batch_id],
            device=diag_batch.device,
            dtype=torch.float32,
        )
        visit_ids = torch.tensor(
            batches["visit_ids_batches"][batch_id],
            device=diag_batch.device,
            dtype=torch.float32,
        )
        current_diag = diag_batch[:, 0, :]
        current_proc = proc_batch[:, 0, :]
        if diag_batch.shape[1] > 1:
            history_diag = (diag_batch[:, 1:, :].sum(dim=1) > 0).float()
            history_proc = (proc_batch[:, 1:, :].sum(dim=1) > 0).float()
        else:
            history_diag = torch.zeros_like(current_diag)
            history_proc = torch.zeros_like(current_proc)
        all_diag = (diag_batch.sum(dim=1) > 0).float()
        all_proc = (proc_batch.sum(dim=1) > 0).float()
        diag_overlap = (current_diag * history_diag).sum(dim=1)
        proc_overlap = (current_proc * history_proc).sum(dim=1)
        current_diag_count = current_diag.sum(dim=1)
        current_proc_count = current_proc.sum(dim=1)
        history_diag_count = history_diag.sum(dim=1)
        history_proc_count = history_proc.sum(dim=1)
        diag_union_count = all_diag.sum(dim=1)
        proc_union_count = all_proc.sum(dim=1)
        diag_jaccard = diag_overlap / (diag_union_count.clamp_min(1.0))
        proc_jaccard = proc_overlap / (proc_union_count.clamp_min(1.0))
        current_diag_only = (current_diag * (1.0 - history_diag).clamp_min(0.0)).sum(dim=1)
        current_proc_only = (current_proc * (1.0 - history_proc).clamp_min(0.0)).sum(dim=1)
        history_diag_only = (history_diag * (1.0 - current_diag).clamp_min(0.0)).sum(dim=1)
        history_proc_only = (history_proc * (1.0 - current_proc).clamp_min(0.0)).sum(dim=1)
        scalar_features = torch.stack([
            seq_len,
            visit_ids,
            current_diag_count,
            current_proc_count,
            history_diag_count,
            history_proc_count,
            diag_union_count,
            proc_union_count,
            current_diag_count - history_diag_count,
            current_proc_count - history_proc_count,
            diag_overlap,
            proc_overlap,
            diag_jaccard,
            proc_jaccard,
            current_diag_only,
            current_proc_only,
            history_diag_only,
            history_proc_only,
        ], dim=1)
        feature_batches.append(scalar_features)
    return torch.cat(feature_batches, dim=0)


def greedy_sequence_predictions(model, batches, end_bias=None):
    if not seq_decoder_enabled():
        return None
    if end_bias is None:
        end_bias = args.seq_decoder_end_bias
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_id in range(batches["n_batches"]):
            pred = model.greedy_sequence_decode(
                batches['diag_batches'][batch_id],
                batches['proc_batches'][batch_id],
                batches['med_batches'][batch_id],
                max_len=args.seq_decoder_max_len,
                end_bias=end_bias,
                clinical_features=batches["clinical_feature_batches"][batch_id] if batches.get("clinical_feature_batches") is not None else None,
                clinical_feature_masks=batches["clinical_feature_mask_batches"][batch_id] if batches.get("clinical_feature_mask_batches") is not None else None,
                clinical_feature_abnormal=batches["clinical_feature_abnormal_batches"][batch_id] if batches.get("clinical_feature_abnormal_batches") is not None else None,
                clinical_feature_delta_hours=batches["clinical_feature_delta_batches"][batch_id] if batches.get("clinical_feature_delta_batches") is not None else None,
                retrieval_meds=batches["retrieval_med_batches"][batch_id] if batches.get("retrieval_med_batches") is not None else None,
                retrieval_scores=batches["retrieval_score_batches"][batch_id] if batches.get("retrieval_score_batches") is not None else None,
                retrieval_queries=batches["retrieval_query_batches"][batch_id] if batches.get("retrieval_query_batches") is not None else None,
                treatment_episode_experiences=batches["treatment_episode_experience_batches"][batch_id] if batches.get("treatment_episode_experience_batches") is not None else None,
                treatment_episode_scores=batches["treatment_episode_score_batches"][batch_id] if batches.get("treatment_episode_score_batches") is not None else None,
            )
            predictions.append(pred)
    return torch.cat(predictions, dim=0)


def hidr_sequence_predictions(model, batches, end_bias=None):
    if not seq_decoder_enabled():
        return None
    if end_bias is None:
        end_bias = args.seq_decoder_end_bias
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_id in range(batches["n_batches"]):
            pred = model.hidr_sequence_decode(
                batches['diag_batches'][batch_id],
                batches['proc_batches'][batch_id],
                batches['med_batches'][batch_id],
                max_len=args.seq_decoder_max_len,
                end_bias=end_bias,
                clinical_features=batches["clinical_feature_batches"][batch_id] if batches.get("clinical_feature_batches") is not None else None,
                clinical_feature_masks=batches["clinical_feature_mask_batches"][batch_id] if batches.get("clinical_feature_mask_batches") is not None else None,
                clinical_feature_abnormal=batches["clinical_feature_abnormal_batches"][batch_id] if batches.get("clinical_feature_abnormal_batches") is not None else None,
                clinical_feature_delta_hours=batches["clinical_feature_delta_batches"][batch_id] if batches.get("clinical_feature_delta_batches") is not None else None,
                retrieval_meds=batches["retrieval_med_batches"][batch_id] if batches.get("retrieval_med_batches") is not None else None,
                retrieval_scores=batches["retrieval_score_batches"][batch_id] if batches.get("retrieval_score_batches") is not None else None,
                retrieval_queries=batches["retrieval_query_batches"][batch_id] if batches.get("retrieval_query_batches") is not None else None,
                treatment_episode_experiences=batches["treatment_episode_experience_batches"][batch_id] if batches.get("treatment_episode_experience_batches") is not None else None,
                treatment_episode_scores=batches["treatment_episode_score_batches"][batch_id] if batches.get("treatment_episode_score_batches") is not None else None,
            )
            predictions.append(pred)
    return torch.cat(predictions, dim=0)


def beam_sequence_predictions(model, batches, end_bias=None):
    if not seq_decoder_enabled():
        return None
    if end_bias is None:
        end_bias = args.seq_decoder_end_bias
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_id in range(batches["n_batches"]):
            pred = model.beam_sequence_decode(
                batches['diag_batches'][batch_id],
                batches['proc_batches'][batch_id],
                batches['med_batches'][batch_id],
                max_len=args.seq_decoder_max_len,
                beam_size=args.seq_decoder_beam_size,
                end_bias=end_bias,
                clinical_features=batches["clinical_feature_batches"][batch_id] if batches.get("clinical_feature_batches") is not None else None,
                clinical_feature_masks=batches["clinical_feature_mask_batches"][batch_id] if batches.get("clinical_feature_mask_batches") is not None else None,
                clinical_feature_abnormal=batches["clinical_feature_abnormal_batches"][batch_id] if batches.get("clinical_feature_abnormal_batches") is not None else None,
                clinical_feature_delta_hours=batches["clinical_feature_delta_batches"][batch_id] if batches.get("clinical_feature_delta_batches") is not None else None,
                retrieval_meds=batches["retrieval_med_batches"][batch_id] if batches.get("retrieval_med_batches") is not None else None,
                retrieval_scores=batches["retrieval_score_batches"][batch_id] if batches.get("retrieval_score_batches") is not None else None,
                retrieval_queries=batches["retrieval_query_batches"][batch_id] if batches.get("retrieval_query_batches") is not None else None,
                treatment_episode_experiences=batches["treatment_episode_experience_batches"][batch_id] if batches.get("treatment_episode_experience_batches") is not None else None,
                treatment_episode_scores=batches["treatment_episode_score_batches"][batch_id] if batches.get("treatment_episode_score_batches") is not None else None,
            )
            predictions.append(pred)
    return torch.cat(predictions, dim=0)


def sequence_rankings(model, batches):
    if not seq_decoder_enabled():
        return None
    model.eval()
    rankings = []
    with torch.no_grad():
        for batch_id in range(batches["n_batches"]):
            ranking = model.sequence_rank_decode(
                batches['diag_batches'][batch_id],
                batches['proc_batches'][batch_id],
                batches['med_batches'][batch_id],
                max_len=args.seq_decoder_max_len,
                clinical_features=batches["clinical_feature_batches"][batch_id] if batches.get("clinical_feature_batches") is not None else None,
                clinical_feature_masks=batches["clinical_feature_mask_batches"][batch_id] if batches.get("clinical_feature_mask_batches") is not None else None,
                clinical_feature_abnormal=batches["clinical_feature_abnormal_batches"][batch_id] if batches.get("clinical_feature_abnormal_batches") is not None else None,
                clinical_feature_delta_hours=batches["clinical_feature_delta_batches"][batch_id] if batches.get("clinical_feature_delta_batches") is not None else None,
                retrieval_meds=batches["retrieval_med_batches"][batch_id] if batches.get("retrieval_med_batches") is not None else None,
                retrieval_scores=batches["retrieval_score_batches"][batch_id] if batches.get("retrieval_score_batches") is not None else None,
                retrieval_queries=batches["retrieval_query_batches"][batch_id] if batches.get("retrieval_query_batches") is not None else None,
                treatment_episode_experiences=batches["treatment_episode_experience_batches"][batch_id] if batches.get("treatment_episode_experience_batches") is not None else None,
                treatment_episode_scores=batches["treatment_episode_score_batches"][batch_id] if batches.get("treatment_episode_score_batches") is not None else None,
            )
            rankings.append(ranking)
    return torch.cat(rankings, dim=0)


def sequence_prob_scores(model, batches, end_bias=None):
    if not seq_decoder_enabled():
        return None
    if end_bias is None:
        end_bias = args.seq_decoder_end_bias
    model.eval()
    scores = []
    with torch.no_grad():
        for batch_id in range(batches["n_batches"]):
            score = model.sequence_prob_decode(
                batches['diag_batches'][batch_id],
                batches['proc_batches'][batch_id],
                batches['med_batches'][batch_id],
                max_len=args.seq_decoder_max_len,
                end_bias=end_bias,
                clinical_features=batches["clinical_feature_batches"][batch_id] if batches.get("clinical_feature_batches") is not None else None,
                clinical_feature_masks=batches["clinical_feature_mask_batches"][batch_id] if batches.get("clinical_feature_mask_batches") is not None else None,
                clinical_feature_abnormal=batches["clinical_feature_abnormal_batches"][batch_id] if batches.get("clinical_feature_abnormal_batches") is not None else None,
                clinical_feature_delta_hours=batches["clinical_feature_delta_batches"][batch_id] if batches.get("clinical_feature_delta_batches") is not None else None,
                retrieval_meds=batches["retrieval_med_batches"][batch_id] if batches.get("retrieval_med_batches") is not None else None,
                retrieval_scores=batches["retrieval_score_batches"][batch_id] if batches.get("retrieval_score_batches") is not None else None,
                retrieval_queries=batches["retrieval_query_batches"][batch_id] if batches.get("retrieval_query_batches") is not None else None,
                treatment_episode_experiences=batches["treatment_episode_experience_batches"][batch_id] if batches.get("treatment_episode_experience_batches") is not None else None,
                treatment_episode_scores=batches["treatment_episode_score_batches"][batch_id] if batches.get("treatment_episode_score_batches") is not None else None,
            )
            scores.append(score)
    return torch.cat(scores, dim=0)


def sequence_prediction_key(end_bias):
    return "{:.6f}".format(float(end_bias))


def get_sequence_predictions_from_batches(batches, name, config):
    predictions = batches.get(name)
    if isinstance(predictions, dict):
        key = sequence_prediction_key(config.get("end_bias", args.seq_decoder_end_bias))
        return predictions.get(key)
    return predictions


def required_sequence_end_biases(configs):
    return sorted({
        float(config.get("end_bias", args.seq_decoder_end_bias))
        for config in configs
        if config["mode"] in {"seq_greedy", "seq_hidr", "seq_beam"}
    })


def set_sequence_prediction_batches(model, batches, configs):
    if not seq_decoder_enabled():
        return
    end_biases = required_sequence_end_biases(configs)
    needs_greedy = any(config["mode"] == "seq_greedy" for config in configs)
    needs_hidr = any(config["mode"] == "seq_hidr" for config in configs)
    needs_beam = any(config["mode"] == "seq_beam" for config in configs)
    needs_rankings = any(
        config["mode"].startswith("seq_rank_")
        or config["mode"].startswith("history_retrieval_seq_rank_")
        for config in configs
    )
    needs_prob_scores = any(config["mode"].startswith("seq_prob_") for config in configs)
    batches["seq_rankings"] = sequence_rankings(model, batches) if needs_rankings else None
    batches["seq_prob_scores"] = sequence_prob_scores(model, batches) if needs_prob_scores else None
    if not end_biases:
        batches["seq_predictions"] = None
        batches["seq_hidr_predictions"] = None
        batches["seq_beam_predictions"] = None
        return
    if len(end_biases) == 1:
        end_bias = end_biases[0]
        batches["seq_predictions"] = greedy_sequence_predictions(model, batches, end_bias=end_bias) if needs_greedy else None
        batches["seq_hidr_predictions"] = hidr_sequence_predictions(model, batches, end_bias=end_bias) if needs_hidr else None
        batches["seq_beam_predictions"] = beam_sequence_predictions(model, batches, end_bias=end_bias) if needs_beam else None
        return
    batches["seq_predictions"] = {} if needs_greedy else None
    batches["seq_hidr_predictions"] = {} if needs_hidr else None
    batches["seq_beam_predictions"] = {} if needs_beam else None
    for end_bias in end_biases:
        key = sequence_prediction_key(end_bias)
        if needs_greedy:
            batches["seq_predictions"][key] = greedy_sequence_predictions(model, batches, end_bias=end_bias)
        if needs_hidr:
            batches["seq_hidr_predictions"][key] = hidr_sequence_predictions(model, batches, end_bias=end_bias)
        if needs_beam:
            batches["seq_beam_predictions"][key] = beam_sequence_predictions(model, batches, end_bias=end_bias)


def get_metrics_from_prob(batches, y_gt, y_pred_prob, loss, decode_config, fast=False, prauc_override=None, med_count_pred=None):
    # restore the original order of samples to aggregate patient-level labels and predictions
    indexes=np.concatenate([np.array(a) for a in batches['ids_batches']])
    sorted_indexes=sorted([(i,indexes[i]) for i in range(len(indexes))], key=lambda x: x[1])
    o_indexes=[t[0] for t in sorted_indexes]
    p_len=[]
    counter=0
    for i in range(1,len(sorted_indexes)):
        if sorted_indexes[i][1] != sorted_indexes[i-1][1]:
            p_len.append(i-counter)
            counter=i
    p_len.append(len(sorted_indexes)-counter)

    decode_batches = {
        "med_history_counts": concatenate_med_history_counts(batches),
        "med_history_binary": concatenate_med_history_binary(batches),
        "previous_med_binary": concatenate_previous_med_binary(batches),
        "retrieval_priors": concatenate_retrieval_priors(batches),
        "retrieval_meds": concatenate_retrieval_meds(batches),
        "retrieval_scores": concatenate_retrieval_scores(batches),
        "treatment_episode_experiences": concatenate_treatment_episode_experiences(batches),
        "med_frequency_prior": med_freq.to(device=y_pred_prob.device, dtype=y_pred_prob.dtype),
        "med_count_pred": med_count_pred,
        "label_counts": y_gt.sum(dim=1),
        "seq_predictions": get_sequence_predictions_from_batches(batches, "seq_predictions", decode_config),
        "seq_hidr_predictions": get_sequence_predictions_from_batches(batches, "seq_hidr_predictions", decode_config),
        "seq_beam_predictions": get_sequence_predictions_from_batches(batches, "seq_beam_predictions", decode_config),
        "seq_rankings": batches.get("seq_rankings"),
        "seq_prob_scores": batches.get("seq_prob_scores"),
        "action_direct_predictions": batches.get("action_direct_predictions"),
        "action_count_predictions": batches.get("action_count_predictions"),
    }
    runtime_decode_config = decode_config
    runtime_decode_config = runtime_decode_config_for_current_modes(decode_config)
    y_pred = decode_predictions(y_pred_prob, decode_batches, runtime_decode_config)
    y_gt=torch.stack([y_gt[i] for i in o_indexes],dim=0)
    y_pred=torch.stack([y_pred[i] for i in o_indexes],dim=0).cpu().numpy().astype(np.int32)
    y_pred_prob=torch.stack([y_pred_prob[i] for i in o_indexes],dim=0).cpu().numpy().astype(np.float64)

    if args.metric_aggregation == "visit":
        if fast:
            ja,prauc,avg_precision,avg_recall,avg_f1 = fast_multi_label_metric(
                np.array(y_gt),
                y_pred,
                np.array(y_pred_prob),
                prauc_override=prauc_override,
            )
        else:
            ja,prauc,avg_precision,avg_recall,avg_f1 = multi_label_metric(
                np.array(y_gt),
                y_pred,
                np.array(y_pred_prob),
            )
        if fast:
            ddi = 0.0
        else:
            pred_meds_list=[np.where(y_pred[i] == 1)[0].tolist() for i in range(y_pred.shape[0])]
            ddi=ddi_rate_score(pred_meds_list, ddi_A)
        mean_med=y_pred.sum(axis=1).mean()
        return ja,prauc,avg_precision,avg_recall,avg_f1,ddi,mean_med,loss

    # calculate scores in patient-level
    idx=0
    patient_scores=[]
    for len_visits in p_len:
        pred_meds=np.array(y_pred[idx:idx+len_visits,:])
        pred_meds_list=[np.where(pred_meds[i] == 1)[0].tolist() for i in range(pred_meds.shape[0])]
        if fast:
            result:list=list(fast_multi_label_metric(
                np.array(y_gt[idx:idx+len_visits,:]),
                pred_meds,
                np.array(y_pred_prob[idx:idx+len_visits,:]),
                prauc_override=prauc_override,
            ))
            result.append(0.0)
        else:
            result:list=list(multi_label_metric(
                np.array(y_gt[idx:idx+len_visits,:]),
                pred_meds,
                np.array(y_pred_prob[idx:idx+len_visits,:])
            ))
            result.append(ddi_rate_score(pred_meds_list, ddi_A))
        patient_scores.append(result)
        idx += len_visits
    patient_scores=np.array(patient_scores)
    ja,prauc,avg_precision,avg_recall,avg_f1,ddi=patient_scores.mean(axis=0)
    # average med num
    mean_med=y_pred.sum(axis=1).mean()
    return ja,prauc,avg_precision,avg_recall,avg_f1,ddi,mean_med,loss


def prediction_order_from_batches(batches):
    indexes = np.concatenate([np.array(a) for a in batches["ids_batches"]])
    sorted_indexes = sorted([(i, indexes[i]) for i in range(len(indexes))], key=lambda item: item[1])
    return [item[0] for item in sorted_indexes]


def ordered_numpy(tensor, ordered_indexes, dtype=None):
    ordered = torch.stack([tensor[i] for i in ordered_indexes], dim=0).cpu().numpy()
    return ordered.astype(dtype) if dtype is not None else ordered


def prediction_tensor_outputs(model, batches):
    model.eval()
    storage = {
        "med_logits": [],
        "fused_logits": [],
        "continue_logits": [],
        "add_logits": [],
        "remove_logits": [],
        "total_count_pred": [],
        "continue_count_pred": [],
        "add_count_pred": [],
        "remove_count_pred": [],
    }

    def append_tensor(name, value):
        if storage[name] is None:
            return
        if value is None:
            storage[name] = None
            return
        storage[name].append(value.detach().cpu())

    with torch.no_grad():
        for batch_id in range(batches["n_batches"]):
            output = forward_model(model, batches, batch_id)
            med_logits = apply_retrieval_prior(output["med_logits"], batches, batch_id)
            decode_logits = med_logits
            if args.use_trainable_action_fusion:
                decode_logits = action_fused_logits_for_batch(output, med_logits, batches, batch_id)
            elif args.action_decode_mode == "score_fusion":
                decode_logits = action_score_fusion_for_batch(output, med_logits, batches, batch_id)
            append_tensor("med_logits", med_logits)
            append_tensor("fused_logits", decode_logits)
            append_tensor("continue_logits", output.get("continue_logits"))
            append_tensor("add_logits", output.get("add_logits"))
            append_tensor("remove_logits", output.get("remove_logits"))
            for key, value in action_count_predictions_from_output(output).items():
                append_tensor(f"{key}_count_pred", value)
    return {
        key: (torch.cat(parts, dim=0) if parts else None)
        for key, parts in storage.items()
        if parts is not None
    }


def prediction_arrays_from_prob(batches, y_gt, y_pred_prob, decode_config, med_count_pred=None):
    ordered_indexes = prediction_order_from_batches(batches)
    decode_batches = {
        "med_history_counts": concatenate_med_history_counts(batches),
        "med_history_binary": concatenate_med_history_binary(batches),
        "previous_med_binary": concatenate_previous_med_binary(batches),
        "retrieval_priors": concatenate_retrieval_priors(batches),
        "retrieval_meds": concatenate_retrieval_meds(batches),
        "retrieval_scores": concatenate_retrieval_scores(batches),
        "treatment_episode_experiences": concatenate_treatment_episode_experiences(batches),
        "med_frequency_prior": med_freq.to(device=y_pred_prob.device, dtype=y_pred_prob.dtype),
        "med_count_pred": med_count_pred,
        "label_counts": y_gt.sum(dim=1),
        "seq_predictions": get_sequence_predictions_from_batches(batches, "seq_predictions", decode_config),
        "seq_hidr_predictions": get_sequence_predictions_from_batches(batches, "seq_hidr_predictions", decode_config),
        "seq_beam_predictions": get_sequence_predictions_from_batches(batches, "seq_beam_predictions", decode_config),
        "seq_rankings": batches.get("seq_rankings"),
        "seq_prob_scores": batches.get("seq_prob_scores"),
        "action_direct_predictions": batches.get("action_direct_predictions"),
        "action_count_predictions": batches.get("action_count_predictions"),
    }
    runtime_decode_config = decode_config
    runtime_decode_config = runtime_decode_config_for_current_modes(decode_config)
    y_pred = decode_predictions(y_pred_prob, decode_batches, runtime_decode_config)
    return (
        ordered_numpy(y_gt, ordered_indexes, np.float64),
        ordered_numpy(y_pred, ordered_indexes, np.int32),
        ordered_numpy(y_pred_prob, ordered_indexes, np.float64),
    )


def prediction_split_batches():
    if args.eval_split == "train":
        return train_batches
    return eval_batches


def prediction_split_patient_offsets():
    if args.eval_split == "train":
        return train_patient_id_offsets
    return eval_patient_id_offsets


def write_prediction_output(model, decode_config, path):
    output_batches = prediction_split_batches()
    y_gt, y_pred_prob, loss, med_count_pred = infer_prob(model, output_batches)
    tensor_outputs = prediction_tensor_outputs(model, output_batches)
    set_sequence_prediction_batches(model, output_batches, [decode_config])
    y_true_array, y_pred_array, y_prob_array = prediction_arrays_from_prob(
        output_batches,
        y_gt,
        y_pred_prob,
        decode_config,
        med_count_pred=med_count_pred,
    )
    ordered_indexes = prediction_order_from_batches(output_batches)
    previous_med_binary = concatenate_previous_med_binary(output_batches)
    history_med_binary = concatenate_med_history_binary(output_batches)
    previous_array = ordered_numpy(previous_med_binary, ordered_indexes, np.int32)
    history_array = ordered_numpy(history_med_binary, ordered_indexes, np.int32)
    gold_total_count = y_true_array.sum(axis=1).astype(np.int32)
    previous_count = previous_array.sum(axis=1).astype(np.int32)
    gold_previous_current_count = ((y_true_array > 0) & (previous_array > 0)).sum(axis=1).astype(np.int32)
    gold_add_count = ((y_true_array > 0) & (previous_array <= 0)).sum(axis=1).astype(np.int32)
    gold_continue_count = gold_previous_current_count.astype(np.int32)
    gold_remove_count = (previous_count - gold_previous_current_count).astype(np.int32)
    ordered_ids = np.asarray([np.asarray(item).reshape(-1)[0] for item in np.concatenate([np.array(a) for a in output_batches["ids_batches"]])])
    ordered_ids = ordered_ids[np.asarray(ordered_indexes)].astype(np.int64)
    visit_ids = np.concatenate([np.asarray(a) for a in output_batches.get("visit_ids_batches", [])])
    ordered_visit_ids = visit_ids[np.asarray(ordered_indexes)].astype(np.int64) if len(visit_ids) else np.arange(len(ordered_indexes), dtype=np.int64)
    patient_offsets = np.asarray(prediction_split_patient_offsets(), dtype=np.int64)
    ordered_patient_ids = patient_offsets[ordered_ids] if len(patient_offsets) else ordered_ids
    real_tensor_arrays = {}
    for key, tensor in tensor_outputs.items():
        real_tensor_arrays[key] = ordered_numpy(tensor, ordered_indexes, np.float64)
    decoder_arrays = {}
    decoder_export_notes = []
    try:
        runtime_decode_config = runtime_decode_config_for_current_modes(decode_config)
        metric_cache = build_decode_metric_cache(output_batches, y_gt, y_pred_prob, loss, med_count_pred)
        decoder_arrays["decoder_scores"] = scores_from_config(metric_cache, runtime_decode_config).detach().cpu().numpy().astype(np.float64)
        decoder_arrays["decoder_counts"] = counts_from_config(metric_cache, runtime_decode_config).detach().cpu().numpy().astype(np.int32)
    except Exception as exc:
        decoder_export_notes.append(f"decoder_scores_unavailable: {type(exc).__name__}: {exc}")
    clipped_prob = np.clip(y_prob_array, 1e-7, 1.0 - 1e-7)
    prob_logit_array = np.log(clipped_prob / (1.0 - clipped_prob)).astype(np.float64)
    y_score_array = real_tensor_arrays.get("fused_logits", prob_logit_array).astype(np.float64)
    action_logit_fields = ["continue_logits", "add_logits", "remove_logits"]
    missing_action_logit_fields = [key for key in action_logit_fields if key not in real_tensor_arrays]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "seer-prediction-v1",
        "model": "SEER",
        "dataset": dataset,
        "seed": args.seed,
        "split": args.eval_split,
        "actual_split": args.eval_split,
        "split_path": args.split_path,
        "run_id": output_path.parent.name,
        "checkpoint": args.eval_model_checkpoint,
        "decode_config_json": args.decode_config_json,
        "metric_aggregation": args.metric_aggregation,
        "sample_count": int(y_true_array.shape[0]),
        "patient_count": int(len(prediction_split_patient_offsets())),
        "tensor_fields": sorted(real_tensor_arrays),
        "decoder_tensor_fields": sorted(decoder_arrays),
        "decoder_export_notes": decoder_export_notes,
        "missing_action_logit_fields": missing_action_logit_fields,
    }
    arrays = dict(
        y_true=y_true_array,
        y_pred=y_pred_array,
        y_prob=y_prob_array,
        fused_prob=y_prob_array,
        y_score=y_score_array,
        previous_meds=previous_array,
        history_meds=history_array,
        patient_ids=ordered_patient_ids,
        local_patient_ids=ordered_ids,
        visit_ids=ordered_visit_ids,
        gold_total_count=gold_total_count,
        previous_count=previous_count,
        gold_previous_current_count=gold_previous_current_count,
        gold_add_count=gold_add_count,
        gold_continue_count=gold_continue_count,
        gold_remove_count=gold_remove_count,
        metadata=np.array(metadata, dtype=object),
        dataset=np.array(dataset, dtype=object),
    )
    arrays.update(real_tensor_arrays)
    arrays.update(decoder_arrays)
    np.savez_compressed(output_path, **arrays)


def write_component_output(model, decode_config, path):
    """Export the frozen decoder signals required by the final SEER fusion."""
    output_batches = prediction_split_batches()
    y_gt, y_pred_prob, loss, med_count_pred = infer_prob(model, output_batches)
    runtime_decode_config = runtime_decode_config_for_current_modes(decode_config)
    cache = build_decode_metric_cache(output_batches, y_gt, y_pred_prob, loss, med_count_pred)
    prob = cache["prob"]
    y_pred = decode_predictions(prob, cache["decode_batches"], runtime_decode_config)
    if runtime_decode_config.get("mode") == "threshold":
        scores = prob
        counts = y_pred.sum(dim=1).to(device=prob.device, dtype=torch.long)
    else:
        scores = scores_from_config(cache, runtime_decode_config)
        counts = counts_from_config(cache, runtime_decode_config)

    ordered_indexes = prediction_order_from_batches(output_batches)
    ordered_ids = np.asarray([
        np.asarray(item).reshape(-1)[0]
        for item in np.concatenate([np.array(a) for a in output_batches["ids_batches"]])
    ])
    ordered_ids = ordered_ids[np.asarray(ordered_indexes)].astype(np.int64)
    visit_ids = np.concatenate([np.asarray(a) for a in output_batches.get("visit_ids_batches", [])])
    ordered_visit_ids = (
        visit_ids[np.asarray(ordered_indexes)].astype(np.int64)
        if len(visit_ids)
        else np.arange(len(ordered_indexes), dtype=np.int64)
    )
    patient_offsets = np.asarray(prediction_split_patient_offsets(), dtype=np.int64)
    ordered_patient_ids = patient_offsets[ordered_ids] if len(patient_offsets) else ordered_ids

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        y_true=cache["target"].detach().cpu().numpy().astype(np.float64),
        y_pred=y_pred.detach().cpu().numpy().astype(np.int8),
        y_prob=prob.detach().cpu().numpy().astype(np.float64),
        final_decoder_score=scores.detach().cpu().numpy().astype(np.float64),
        decoder_count=counts.detach().cpu().numpy().astype(np.int32),
        patient_ids=ordered_patient_ids,
        local_patient_ids=ordered_ids,
        visit_ids=ordered_visit_ids,
        dataset=np.array(dataset, dtype=object),
    )


def build_decode_metric_cache(batches, y_gt, y_pred_prob, loss, med_count_pred):
    indexes = np.concatenate([np.array(a) for a in batches['ids_batches']])
    sorted_indexes = sorted([(i, indexes[i]) for i in range(len(indexes))], key=lambda x: x[1])
    cache_device = y_pred_prob.device
    order = torch.tensor([item[0] for item in sorted_indexes], device=cache_device, dtype=torch.long)
    patient_group_ids = []
    current_group = -1
    previous_patient = None
    for _, patient_id in sorted_indexes:
        if patient_id != previous_patient:
            current_group += 1
            previous_patient = patient_id
        patient_group_ids.append(current_group)
    group_ids = torch.tensor(patient_group_ids, device=cache_device, dtype=torch.long)

    def order_tensor(value):
        if value is None:
            return None
        return value.to(device=cache_device)[order]

    def order_action_count_predictions(value):
        if not value:
            return None
        ordered = {}
        for name in ("total", "continue", "add", "remove"):
            pred = value.get(name)
            ordered[name] = order_tensor(pred)
        if all(item is None for item in ordered.values()):
            return None
        return ordered

    decode_batches = {
        "med_history_counts": order_tensor(concatenate_med_history_counts(batches)),
        "med_history_binary": order_tensor(concatenate_med_history_binary(batches)),
        "previous_med_binary": order_tensor(concatenate_previous_med_binary(batches)),
        "retrieval_priors": order_tensor(concatenate_retrieval_priors(batches)),
        "retrieval_meds": order_tensor(concatenate_retrieval_meds(batches)),
        "retrieval_scores": order_tensor(concatenate_retrieval_scores(batches)),
        "treatment_episode_scores": order_tensor(concatenate_treatment_episode_scores(batches)),
        "treatment_episode_experiences": order_tensor(concatenate_treatment_episode_experiences(batches)),
        "visit_context_features": order_tensor(concatenate_visit_context_features(batches)),
        "med_frequency_prior": med_freq.to(device=cache_device, dtype=y_pred_prob.dtype),
        "med_count_pred": order_tensor(med_count_pred),
        "label_counts": order_tensor(y_gt.sum(dim=1)),
        "seq_predictions": order_tensor(batches.get("seq_predictions")),
        "seq_hidr_predictions": order_tensor(batches.get("seq_hidr_predictions")),
        "seq_beam_predictions": order_tensor(batches.get("seq_beam_predictions")),
        "seq_rankings": order_tensor(batches.get("seq_rankings")),
        "seq_prob_scores": order_tensor(batches.get("seq_prob_scores")),
        "action_direct_predictions": order_tensor(batches.get("action_direct_predictions")),
        "action_count_predictions": order_action_count_predictions(batches.get("action_count_predictions")),
    }
    return {
        "target": y_gt.to(device=cache_device)[order],
        "prob": y_pred_prob[order],
        "loss": loss,
        "group_ids": group_ids,
        "num_groups": current_group + 1,
        "decode_batches": decode_batches,
    }


def top_count_predictions(scores, counts):
    batch_size, med_voc_size = scores.shape
    min_count, max_count = decode_count_limits(med_voc_size)
    counts = counts.long().clamp(min=min_count, max=max_count)
    if max_count <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    sorted_indexes = torch.argsort(scores, dim=1, descending=True)
    ranks = torch.empty_like(sorted_indexes)
    rank_values = torch.arange(med_voc_size, device=scores.device, dtype=sorted_indexes.dtype).view(1, -1)
    ranks.scatter_(1, sorted_indexes, rank_values.expand(batch_size, -1))
    return ranks < counts.view(-1, 1)


def require_action_count_predictions(decode_batches):
    count_pred = decode_batches.get("action_count_predictions")
    if not count_pred:
        raise ValueError("count-aware decode requires action_count_predictions")
    required = ("total", "continue", "add", "remove")
    if any(count_pred.get(name) is None for name in required):
        raise ValueError("count-aware decode requires total/continue/add/remove count predictions")
    return count_pred


def rounded_count_prediction(count_pred, name, scores):
    pred = count_pred[name].to(device=scores.device, dtype=scores.dtype)
    min_count, max_count = decode_count_limits(scores.shape[1])
    return torch.round(pred).to(dtype=torch.long).clamp(min=min_count, max=max_count)


def count_total_topk_predictions(scores, decode_batches):
    count_pred = require_action_count_predictions(decode_batches)
    counts = rounded_count_prediction(count_pred, "total", scores)
    return top_count_predictions(scores, counts)


def action_split_topk_predictions(old_scores, new_scores, previous, decode_batches):
    count_pred = require_action_count_predictions(decode_batches)
    batch_size, med_voc_size = old_scores.shape
    previous = previous.to(device=old_scores.device, dtype=torch.bool)
    min_count, max_count = decode_count_limits(med_voc_size)
    continue_counts = torch.round(
        count_pred["continue"].to(device=old_scores.device, dtype=old_scores.dtype)
    ).to(dtype=torch.long)
    add_counts = torch.round(
        count_pred["add"].to(device=old_scores.device, dtype=old_scores.dtype)
    ).to(dtype=torch.long)
    previous_counts = previous.sum(dim=1).to(dtype=torch.long)
    non_previous_counts = (~previous).sum(dim=1).to(dtype=torch.long)
    continue_counts = continue_counts.clamp(min=0).minimum(previous_counts)
    add_counts = add_counts.clamp(min=0).minimum(non_previous_counts)
    if max_count >= min_count:
        total_counts = (continue_counts + add_counts).clamp(min=min_count, max=max_count)
        overflow = (continue_counts + add_counts - total_counts).clamp_min(0)
        if overflow.any():
            add_counts = (add_counts - overflow).clamp_min(0)
    y_pred = torch.zeros_like(old_scores, dtype=torch.bool)
    row_indexes = torch.arange(batch_size, device=old_scores.device)
    masked_old_scores = old_scores.masked_fill(~previous, -torch.inf)
    masked_new_scores = new_scores.masked_fill(previous, -torch.inf)
    old_order = torch.argsort(masked_old_scores, dim=1, descending=True)
    new_order = torch.argsort(masked_new_scores, dim=1, descending=True)
    old_rank = torch.empty_like(old_order)
    new_rank = torch.empty_like(new_order)
    rank_values = torch.arange(med_voc_size, device=old_scores.device, dtype=old_order.dtype).view(1, -1)
    old_rank.scatter_(1, old_order, rank_values.expand(batch_size, -1))
    new_rank.scatter_(1, new_order, rank_values.expand(batch_size, -1))
    y_pred = y_pred | (previous & (old_rank < continue_counts.view(-1, 1)))
    y_pred = y_pred | ((~previous) & (new_rank < add_counts.view(-1, 1)))
    if max_count >= min_count and min_count > 0:
        total_counts = y_pred.sum(dim=1).to(dtype=torch.long)
        need_fill = total_counts < min_count
        if need_fill.any():
            combined_scores = torch.where(previous, old_scores, new_scores)
            combined_scores = combined_scores.masked_fill(y_pred, -torch.inf)
            fill_order = torch.argsort(combined_scores, dim=1, descending=True)
            for row_idx in torch.where(need_fill)[0]:
                needed = int(min_count - total_counts[row_idx].item())
                if needed > 0:
                    y_pred[row_idx, fill_order[row_idx, :needed]] = True
    return y_pred


def expected_jaccard_counts(scores, prob):
    batch_size, med_voc_size = scores.shape
    min_count, max_count = decode_count_limits(med_voc_size)
    if max_count <= 0:
        return torch.zeros(batch_size, device=scores.device, dtype=torch.long)
    sorted_indexes = torch.argsort(scores, dim=1, descending=True)
    prob = prob.to(device=scores.device, dtype=scores.dtype).clamp(0.0, 1.0)
    sorted_prob = prob.gather(1, sorted_indexes)
    cumulative_expected_hits = sorted_prob.cumsum(dim=1)
    expected_target_count = prob.sum(dim=1, keepdim=True)
    counts_grid = torch.arange(1, med_voc_size + 1, device=scores.device, dtype=scores.dtype).view(1, -1)
    expected_union = expected_target_count + counts_grid - cumulative_expected_hits
    expected_jaccard = cumulative_expected_hits / expected_union.clamp_min(1e-6)
    if min_count == 0:
        zero_score = torch.zeros(batch_size, 1, device=scores.device, dtype=scores.dtype)
        expected_jaccard = torch.cat((zero_score, expected_jaccard[:, :max_count]), dim=1)
        return torch.argmax(expected_jaccard, dim=1).to(dtype=torch.long)
    allowed = torch.zeros(med_voc_size, device=scores.device, dtype=torch.bool)
    allowed[min_count - 1:max_count] = True
    masked = torch.where(allowed.view(1, -1), expected_jaccard, torch.full_like(expected_jaccard, -1.0))
    return torch.argmax(masked, dim=1).to(dtype=torch.long) + 1


def base_count_config_value(config, key, fallback_key, default):
    if key in config:
        return config[key]
    if fallback_key in config:
        return config[fallback_key]
    return default


def graph_rank_base_counts(prob, scores, decode_batches, config):
    count_mode = config.get("base_count_mode", "expected_jaccard_count")
    min_count, max_count = decode_count_limits(scores.shape[1])
    if count_mode == "fixed_topk":
        counts = torch.full((scores.shape[0],), int(config["topk"]), device=scores.device, dtype=torch.long)
    elif count_mode == "threshold_count":
        threshold = float(base_count_config_value(config, "base_threshold", "threshold", args.eval_threshold))
        ratio = float(base_count_config_value(config, "base_ratio", "ratio", 1.0))
        bias = float(base_count_config_value(config, "base_bias", "bias", 0.0))
        threshold_counts = (prob >= threshold).sum(dim=1).to(device=scores.device, dtype=scores.dtype)
        counts = torch.round(threshold_counts * ratio + bias)
    elif count_mode == "expected_count":
        ratio = float(base_count_config_value(config, "base_ratio", "ratio", 1.0))
        bias = float(base_count_config_value(config, "base_bias", "bias", 0.0))
        counts = torch.round(prob.to(device=scores.device, dtype=scores.dtype).sum(dim=1) * ratio + bias)
    elif count_mode == "expected_jaccard_count":
        counts = expected_jaccard_counts(scores, prob)
    elif count_mode == "history_count":
        history_counts = decode_batches["med_history_counts"]
        if history_counts is None:
            raise ValueError("{} decoder requires med_history_counts".format(config["mode"]))
        ratio = float(base_count_config_value(config, "base_ratio", "ratio", 1.0))
        bias = float(base_count_config_value(config, "base_bias", "bias", 0.0))
        counts = torch.round(history_counts.to(device=scores.device, dtype=scores.dtype) * ratio + bias)
    elif count_mode == "predicted_count":
        med_count_pred = decode_batches["med_count_pred"]
        if med_count_pred is None:
            raise ValueError("{} decoder requires med_count_pred".format(config["mode"]))
        ratio = float(base_count_config_value(config, "base_ratio", "ratio", 1.0))
        bias = float(base_count_config_value(config, "base_bias", "bias", 0.0))
        counts = torch.round(med_count_pred.to(device=scores.device, dtype=scores.dtype) * ratio + bias)
    elif count_mode == "retrieval_count":
        base_config = dict(config)
        base_config["ratio"] = float(base_count_config_value(config, "base_ratio", "ratio", 1.0))
        base_config["bias"] = float(base_count_config_value(config, "base_bias", "bias", 0.0))
        counts = retrieval_context_counts(decode_batches, scores, base_config)
    else:
        raise ValueError("unsupported base_count_mode for {}: {}".format(config["mode"], count_mode))
    return counts.to(device=scores.device, dtype=torch.long).clamp(min=min_count, max=max_count)


def treatment_episode_component_priors_from_batches(decode_batches, scores):
    experiences = decode_batches.get("treatment_episode_experiences")
    if experiences is None:
        return None
    med_voc_size = int(scores.shape[1])
    experiences = experiences.to(device=scores.device, dtype=torch.float32)
    if experiences.dim() != 3 or experiences.shape[-1] < 4 * med_voc_size:
        raise ValueError("treatment_episode_experiences must have shape [batch, topk, 4 * med_voc + scalars]")
    valid = experiences[:, :, :4 * med_voc_size].abs().sum(dim=2) > 0
    episode_scores = decode_batches.get("treatment_episode_scores")
    if episode_scores is not None and episode_scores.shape[:2] == experiences.shape[:2]:
        weights = episode_scores.to(device=scores.device, dtype=torch.float32).clamp_min(0.0)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
    else:
        weights = valid.to(dtype=torch.float32)
    weight_sums = weights.sum(dim=1, keepdim=True)

    def weighted_component(start, end):
        component = experiences[:, :, start:end]
        prior = (component * weights.unsqueeze(-1)).sum(dim=1) / weight_sums.clamp_min(1e-6)
        return torch.where(weight_sums > 0, prior, torch.zeros_like(prior))

    return {
        "history": weighted_component(0, med_voc_size),
        "added": weighted_component(med_voc_size, 2 * med_voc_size),
        "continued": weighted_component(2 * med_voc_size, 3 * med_voc_size),
        "removed": weighted_component(3 * med_voc_size, 4 * med_voc_size),
    }


def treatment_episode_action_count_sources(decode_batches, scores, prob, previous=None):
    component_priors = treatment_episode_component_priors_from_batches(decode_batches, scores)
    if component_priors is None:
        return {}
    experiences = decode_batches.get("treatment_episode_experiences").to(device=scores.device, dtype=torch.float32)
    med_voc_size = int(scores.shape[1])
    episode_added = experiences[:, :, med_voc_size:2 * med_voc_size]
    episode_continued = experiences[:, :, 2 * med_voc_size:3 * med_voc_size]
    episode_removed = experiences[:, :, 3 * med_voc_size:4 * med_voc_size]
    valid = experiences[:, :, :4 * med_voc_size].abs().sum(dim=2) > 0
    valid_float = valid.to(dtype=torch.float32)
    valid_counts = valid_float.sum(dim=1).clamp_min(1.0)
    if previous is None:
        previous = decode_batches.get("previous_med_binary")
    if previous is None:
        return {}
    previous = previous.to(device=scores.device, dtype=torch.float32)
    not_previous = (1.0 - previous).clamp_min(0.0)
    added = component_priors["added"]
    continued = component_priors["continued"]
    removed = component_priors["removed"]
    previous_count = previous.sum(dim=1)
    episode_add_count = (not_previous * added).sum(dim=1)
    episode_continue_count = (previous * continued).sum(dim=1)
    episode_remove_count = (previous * removed).sum(dim=1)
    mean_add_count = (episode_added.sum(dim=2) * valid_float).sum(dim=1) / valid_counts
    mean_continue_count = (episode_continued.sum(dim=2) * valid_float).sum(dim=1) / valid_counts
    mean_remove_count = (episode_removed.sum(dim=2) * valid_float).sum(dim=1) / valid_counts
    top1_add_count = torch.where(valid[:, 0], episode_added[:, 0].sum(dim=1), mean_add_count)
    top1_continue_count = torch.where(valid[:, 0], episode_continued[:, 0].sum(dim=1), mean_continue_count)
    top1_remove_count = torch.where(valid[:, 0], episode_removed[:, 0].sum(dim=1), mean_remove_count)
    return {
        "episode_add_count": episode_add_count,
        "episode_continue_count": episode_continue_count,
        "episode_remove_count": episode_remove_count,
        "episode_action_count": episode_continue_count + episode_add_count,
        "episode_previous_add_count": previous_count + episode_add_count,
        "episode_net_count": (previous_count - episode_remove_count + episode_add_count).clamp_min(0.0),
        "episode_mean_action_count": mean_continue_count + mean_add_count,
        "episode_mean_previous_add_count": previous_count + mean_add_count,
        "episode_mean_net_count": (previous_count - mean_remove_count + mean_add_count).clamp_min(0.0),
        "episode_top1_action_count": top1_continue_count + top1_add_count,
        "episode_top1_previous_add_count": previous_count + top1_add_count,
        "episode_top1_net_count": (previous_count - top1_remove_count + top1_add_count).clamp_min(0.0),
    }


def count_source_from_batches(source_name, prob, scores, decode_batches, config):
    prob = prob.to(device=scores.device, dtype=scores.dtype)
    if source_name == "configured":
        return graph_rank_base_counts(prob, scores, decode_batches, config).to(dtype=torch.float32)
    if source_name == "expected_jaccard":
        return expected_jaccard_counts(scores, prob).to(device=scores.device, dtype=torch.float32)
    if source_name == "prob_sum":
        return prob.sum(dim=1).to(dtype=torch.float32)
    if source_name == "history":
        history_counts = decode_batches["med_history_counts"]
        if history_counts is None:
            raise ValueError("{} count source requires med_history_counts".format(source_name))
        return history_counts.to(device=scores.device, dtype=torch.float32)
    retrieval = decode_batches.get("retrieval_meds")
    if source_name in {"retrieval_mean", "retrieval_top1"}:
        if retrieval is None:
            return graph_rank_base_counts(prob, scores, decode_batches, config).to(dtype=torch.float32)
        retrieval = retrieval.to(device=scores.device, dtype=torch.float32)
        retrieval_counts = retrieval.sum(dim=2)
        if source_name == "retrieval_mean":
            return retrieval_counts.mean(dim=1)
        return retrieval_counts[:, 0]
    previous = decode_batches.get("previous_med_binary")
    if source_name in {
        "previous_count",
        "previous_prob_sum",
        "add_prob_sum",
        "continue_add_prob_sum",
        "previous_count_add_prob",
        "previous_prob_add_prob",
    }:
        if previous is None:
            raise ValueError("{} count source requires previous_med_binary".format(source_name))
        previous = previous.to(device=scores.device, dtype=torch.float32)
        prob_float = prob.to(dtype=torch.float32)
        not_previous = (1.0 - previous).clamp_min(0.0)
        previous_count = previous.sum(dim=1)
        previous_prob_sum = (prob_float * previous).sum(dim=1)
        add_prob_sum = (prob_float * not_previous).sum(dim=1)
        if source_name == "previous_count":
            return previous_count
        if source_name == "previous_prob_sum":
            return previous_prob_sum
        if source_name == "add_prob_sum":
            return add_prob_sum
        if source_name == "continue_add_prob_sum":
            return previous_prob_sum + add_prob_sum
        if source_name == "previous_count_add_prob":
            return previous_count + add_prob_sum
        return previous_prob_sum + add_prob_sum
    episode_sources = treatment_episode_action_count_sources(decode_batches, scores, prob)
    if source_name in episode_sources:
        return episode_sources[source_name].to(device=scores.device, dtype=torch.float32)
    raise ValueError("unknown count source for {}: {}".format(config["mode"], source_name))


def count_pair_mix_counts(prob, scores, decode_batches, config):
    left_name = config.get("left", "configured")
    right_name = config.get("right", "previous_count_add_prob")
    left_counts = count_source_from_batches(left_name, prob, scores, decode_batches, config)
    right_counts = count_source_from_batches(right_name, prob, scores, decode_batches, config)
    alpha = float(config.get("alpha", 0.94))
    mixed = alpha * left_counts + (1.0 - alpha) * right_counts
    ratio = float(config.get("ratio", 1.0))
    bias = float(config.get("bias", 0.0))
    return clamp_round_counts(mixed * ratio + bias, scores.shape[1]).to(device=scores.device)


def retrieval_context_counts(decode_batches, scores, config):
    retrieval_meds = decode_batches.get("retrieval_meds")
    if retrieval_meds is None:
        raise ValueError("{} decoder requires retrieval_meds".format(config["mode"]))
    retrieval_meds = retrieval_meds.to(device=scores.device, dtype=scores.dtype)
    if retrieval_meds.dim() != 3:
        raise ValueError("retrieval_meds must have shape [batch, topk, med_voc]")
    visit_counts = retrieval_meds.sum(dim=2)
    valid = visit_counts > 0
    count_mode = config.get("retrieval_count_mode", "weighted_mean")
    if count_mode == "top1":
        base_counts = visit_counts[:, 0]
        fallback_counts = visit_counts.sum(dim=1) / valid.sum(dim=1).clamp_min(1).to(dtype=scores.dtype)
        base_counts = torch.where(base_counts > 0, base_counts, fallback_counts)
    elif count_mode == "mean":
        base_counts = visit_counts.sum(dim=1) / valid.sum(dim=1).clamp_min(1).to(dtype=scores.dtype)
    elif count_mode == "union":
        base_counts = (retrieval_meds.sum(dim=1) > 0).sum(dim=1).to(dtype=scores.dtype)
    elif count_mode == "weighted_mean":
        retrieval_scores = decode_batches.get("retrieval_scores")
        if retrieval_scores is None:
            base_counts = visit_counts.sum(dim=1) / valid.sum(dim=1).clamp_min(1).to(dtype=scores.dtype)
        else:
            weights = retrieval_scores.to(device=scores.device, dtype=scores.dtype).clamp_min(0.0)
            weights = torch.where(valid, weights, torch.zeros_like(weights))
            weight_sums = weights.sum(dim=1)
            weighted_counts = (visit_counts * weights).sum(dim=1) / weight_sums.clamp_min(1e-6)
            mean_counts = visit_counts.sum(dim=1) / valid.sum(dim=1).clamp_min(1).to(dtype=scores.dtype)
            base_counts = torch.where(weight_sums > 0, weighted_counts, mean_counts)
    else:
        raise ValueError(f"unsupported retrieval_count_mode: {count_mode}")
    counts = torch.round(base_counts * float(config["ratio"]) + float(config["bias"]))
    min_count, max_count = decode_count_limits(scores.shape[1])
    return counts.to(device=scores.device, dtype=torch.long).clamp(min=min_count, max=max_count)


_graph_prior_cache = {}


def row_normalized_dense_graph(adj, device, dtype):
    cache_key = (id(adj), str(device), str(dtype))
    cached = _graph_prior_cache.get(cache_key)
    if cached is not None:
        return cached
    graph = adj.to(device=device, dtype=dtype)
    if graph.is_sparse:
        graph = graph.to_dense()
    graph = graph.clone()
    graph.fill_diagonal_(0.0)
    graph = graph / graph.sum(dim=1, keepdim=True).clamp_min(1e-6)
    _graph_prior_cache[cache_key] = graph
    return graph


def graph_propagation_signal(prior):
    ehr_graph = row_normalized_dense_graph(ehr_A_gpu, prior.device, prior.dtype)
    ddi_graph = row_normalized_dense_graph(ddi_A_gpu, prior.device, prior.dtype)
    return {
        "ehr": torch.matmul(prior, ehr_graph),
        "ddi": torch.matmul(prior, ddi_graph),
    }


def history_retrieval_freq_graph_scores(prob, decode_batches, config):
    history_signal = decode_batches["med_history_binary"]
    retrieval_signal = decode_batches["retrieval_priors"]
    freq_signal = decode_batches["med_frequency_prior"]
    if history_signal is None or retrieval_signal is None or freq_signal is None:
        raise ValueError("history_retrieval_freq_graph_rank decoder requires med_history_binary, retrieval_priors, and med_frequency_prior")
    history_signal = history_signal.to(device=prob.device, dtype=prob.dtype)
    retrieval_signal = retrieval_signal.to(device=prob.device, dtype=prob.dtype)
    graph_source = config.get("graph_source", "model")
    if graph_source == "model":
        graph_prior = prob
    elif graph_source == "history":
        graph_prior = history_signal
    elif graph_source == "retrieval":
        graph_prior = retrieval_signal
    elif graph_source == "hybrid":
        graph_prior = torch.stack((prob, history_signal, retrieval_signal), dim=0).mean(dim=0)
    else:
        raise ValueError(f"unsupported graph_source: {graph_source}")
    graph_signal = graph_propagation_signal(graph_prior)
    return (
        prob
        + float(config["history_weight"]) * history_signal
        + float(config["retrieval_weight"]) * retrieval_signal
        + float(config["freq_weight"]) * freq_signal.to(device=prob.device, dtype=prob.dtype).view(1, -1)
        + float(config["ehr_graph_weight"]) * graph_signal["ehr"]
        - float(config["ddi_graph_weight"]) * graph_signal["ddi"]
    )


def ddi_aware_top_count_predictions(scores, counts, ddi_penalty):
    if ddi_penalty <= 0:
        return top_count_predictions(scores, counts)
    batch_size, med_voc_size = scores.shape
    min_count, max_count = decode_count_limits(med_voc_size)
    counts = counts.long().clamp(min=min_count, max=max_count)
    if max_count <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    ddi_matrix = ddi_A_gpu.to(device=scores.device, dtype=scores.dtype)
    y_pred = torch.zeros_like(scores, dtype=torch.bool)
    for row_idx in range(batch_size):
        count = int(counts[row_idx].item())
        if count <= 0:
            continue
        selected = []
        row_scores = scores[row_idx].clone()
        risk_scores = torch.zeros(med_voc_size, device=scores.device, dtype=scores.dtype)
        for _ in range(count):
            adjusted_scores = row_scores - float(ddi_penalty) * risk_scores
            med_idx = int(torch.argmax(adjusted_scores).item())
            selected.append(med_idx)
            row_scores[med_idx] = -float("inf")
            risk_scores = risk_scores + ddi_matrix[med_idx]
        y_pred[row_idx, torch.tensor(selected, device=scores.device, dtype=torch.long)] = True
    return y_pred


def collect_paper_prefix_outputs(model, batches, data_subset):
    model.eval()
    prob_parts = []
    target_parts = []
    history_count_parts = []
    history_binary_parts = []
    owner_parts = []
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch_id in range(batches["n_batches"]):
            prefix_diags, prefix_procs, prefix_meds, prefix_labels, owner_rows, clinical_parts = build_prefix_samples_from_batches(
                batches,
                batch_id,
                data_subset,
                include_target=True,
            )
            if not prefix_labels:
                continue
            prefix_logits, prefix_targets = forward_plain_prefix_batch(
                model,
                prefix_diags,
                prefix_procs,
                prefix_meds,
                prefix_labels,
                infer_batch_size,
                clinical_parts=clinical_parts,
            )
            prefix_prob = torch.sigmoid(prefix_logits)
            history_counts = torch.tensor(
                [(med_seq.sum(dim=0) > 0).float().sum().item() for med_seq in prefix_meds],
                dtype=prefix_prob.dtype,
                device=prefix_prob.device,
            )
            history_binary = torch.stack(
                [
                    (med_seq[1:].sum(dim=0) > 0)
                    if med_seq.shape[0] > 1
                    else torch.zeros(size_med_voc, dtype=torch.bool, device=prefix_prob.device)
                    for med_seq in prefix_meds
                ],
                dim=0,
            ).to(device=prefix_prob.device)
            batch_loss = loss_func(prefix_logits, prefix_targets)
            loss_sum += batch_loss.item() * prefix_targets.shape[0]
            sample_count += prefix_targets.shape[0]
            prob_parts.append(prefix_prob)
            target_parts.append(prefix_targets)
            history_count_parts.append(history_counts)
            history_binary_parts.append(history_binary)
            owner_parts.extend((batch_id, owner_row) for owner_row in owner_rows)
    if not prob_parts:
        return None
    return {
        "prob": torch.cat(prob_parts, dim=0),
        "target": torch.cat(target_parts, dim=0),
        "history_counts": torch.cat(history_count_parts, dim=0),
        "history_binary": torch.cat(history_binary_parts, dim=0),
        "owner_rows": owner_parts,
        "loss": loss_sum / max(sample_count, 1),
    }


def get_paper_prefix_metrics_from_cache(cache, decode_config, fast=False, prauc_override=None):
    y_pred = decode_predictions(
        cache["prob"],
        {
            "med_history_counts": cache["history_counts"],
            "med_history_binary": cache["history_binary"],
        },
        decode_config,
    ).cpu().numpy().astype(np.int32)
    y_gt = cache["target"].cpu().numpy().astype(np.int32)
    y_prob = cache["prob"].cpu().numpy().astype(np.float64)
    prefix_scores = []
    prefix_ddi_scores = []
    prefix_avg_meds = []
    owner_rows = cache["owner_rows"]
    start = 0
    while start < len(owner_rows):
        end = start + 1
        while end < len(owner_rows) and owner_rows[end] == owner_rows[start]:
            end += 1
        labels_for_prefix = [np.where(y_pred[row] == 1)[0].tolist() for row in range(start, end)]
        if fast:
            ja, prauc, avg_precision, avg_recall, avg_f1 = fast_multi_label_metric(
                y_gt[start:end],
                y_pred[start:end],
                y_prob[start:end],
                prauc_override=prauc_override,
            )
        else:
            ja, prauc, avg_precision, avg_recall, avg_f1 = multi_label_metric(
                y_gt[start:end],
                y_pred[start:end],
                y_prob[start:end],
            )
        prefix_scores.append([ja, prauc, avg_precision, avg_recall, avg_f1])
        prefix_ddi_scores.append(0.0 if fast else ddi_rate_score(labels_for_prefix, ddi_A))
        prefix_avg_meds.append(y_pred[start:end].sum(axis=1).mean())
        start = end
    scores = np.asarray(prefix_scores, dtype=np.float64)
    ja, prauc, avg_precision, avg_recall, avg_f1 = scores.mean(axis=0)
    ddi = float(np.mean(prefix_ddi_scores)) if prefix_ddi_scores else 0.0
    mean_med = float(np.mean(prefix_avg_meds)) if prefix_avg_meds else 0.0
    return ja, prauc, avg_precision, avg_recall, avg_f1, ddi, mean_med, cache["loss"]


def get_paper_prefix_metrics(model, batches, data_subset, decode_config, fast=False, prauc_override=None):
    cache = collect_paper_prefix_outputs(model, batches, data_subset)
    if cache is None:
        return get_metrics(model, batches)
    return get_paper_prefix_metrics_from_cache(cache, decode_config, fast=fast, prauc_override=prauc_override)


def select_paper_prefix_validation_metrics(model, log_file=None, epoch=None):
    configs = paper_prefix_decode_configs()
    if len(configs) == 1:
        config = configs[0]
        return get_paper_prefix_metrics(model, eval_batches, data_eval, config), config

    cache = collect_paper_prefix_outputs(model, eval_batches, data_eval)
    if cache is None:
        config = configs[0]
        return get_metrics(model, eval_batches), config
    selected_metric = threshold_selection_metric()
    best_config = configs[0]
    best_metrics = None
    best_score = -float("inf")
    best_tie_breaker = -float("inf")
    for config in configs:
        metrics = get_paper_prefix_metrics_from_cache(
            cache,
            config,
            fast=True,
            prauc_override=0.0,
        )
        score = metric_value_without_epoch(metrics, selected_metric)
        tie_breaker = metrics[1] - 1e-4 * metrics[6]
        if args.log_decode_search_details and log_file is not None:
            log(
                log_file,
                "decode_search epoch={} candidate={} ja={:.4f} prauc={:.4f} f1={:.4f} ddi={:.4f} avg_med={:.2f}".format(
                    epoch + 1 if epoch is not None else "?",
                    format_decode_config(config),
                    metrics[0],
                    metrics[1],
                    metrics[4],
                    metrics[5],
                    metrics[6],
                ),
            )
        if score > best_score or (score == best_score and tie_breaker > best_tie_breaker):
            best_score = score
            best_tie_breaker = tie_breaker
            best_metrics = metrics
            best_config = config
    best_metrics = get_paper_prefix_metrics_from_cache(cache, best_config)
    return best_metrics, best_config


def get_metrics(model, batches, threshold=None):
    if threshold is None:
        threshold = args.eval_threshold
    y_gt, y_pred_prob, loss, med_count_pred = infer_prob(model, batches)
    return get_metrics_from_prob(batches, y_gt, y_pred_prob, loss, {"mode": "threshold", "threshold": threshold}, med_count_pred=med_count_pred)
#%%
# result=get_metrics(model, eval_batches, threshold=0.3)
#%%

def select_validation_metrics(model, log_file=None, epoch=None):
    if args.prefix_metric_mode == "paper_prefix":
        return select_paper_prefix_validation_metrics(model, log_file=log_file, epoch=epoch)
    if retrieval_is_treatment_episode and args.treatment_episode_action_weight > 0:
        refresh_treatment_episode_batches(model, include_train=False, include_eval=True)
    y_gt, y_pred_prob, loss, med_count_pred = infer_prob(model, eval_batches)
    configs = decode_configs()
    set_sequence_prediction_batches(model, eval_batches, configs)
    if len(configs) == 1:
        config = configs[0]
        if config["mode"] == "per_med_threshold" and config.get("thresholds") is None:
            config = dict(config)
            config["thresholds"] = fit_per_med_thresholds(y_gt, y_pred_prob)
        return get_metrics_from_prob(eval_batches, y_gt, y_pred_prob, loss, config, med_count_pred=med_count_pred), config

    selected_metric = threshold_selection_metric()
    best_config = configs[0]
    best_metrics = None
    best_score = -float("inf")
    best_tie_breaker = -float("inf")
    use_fast_search = len(configs) > 1
    for config in configs:
        if config["mode"] == "per_med_threshold" and config.get("thresholds") is None:
            config = dict(config)
            config["thresholds"] = fit_per_med_thresholds(y_gt, y_pred_prob)
        metrics = get_metrics_from_prob(
            eval_batches,
            y_gt,
            y_pred_prob,
            loss,
            config,
            fast=use_fast_search,
            prauc_override=0.0 if use_fast_search else None,
            med_count_pred=med_count_pred,
        )
        score = metric_value_without_epoch(metrics, selected_metric)
        if args.log_decode_search_details and log_file is not None:
            log(
                log_file,
                "decode_search epoch={} candidate={} ja={:.4f} prauc={:.4f} f1={:.4f} ddi={:.4f} avg_med={:.2f}".format(
                    epoch + 1 if epoch is not None else "?",
                    format_decode_config(config),
                    metrics[0],
                    metrics[1],
                    metrics[4],
                    metrics[5],
                    metrics[6],
                ),
            )
        # Tie-break toward higher PRAUC and then fewer predicted medications.
        tie_breaker = metrics[1] - 1e-4 * metrics[6]
        if score > best_score or (score == best_score and tie_breaker > best_tie_breaker):
            best_score = score
            best_tie_breaker = tie_breaker
            best_metrics = metrics
            best_config = config
    if use_fast_search:
        best_metrics = get_metrics_from_prob(eval_batches, y_gt, y_pred_prob, loss, best_config, med_count_pred=med_count_pred)
    return best_metrics, best_config


METRIC_NAMES = ("ja", "prauc", "avg_precision", "avg_recall", "avg_f1", "ddi", "avg_med", "loss")


def jsonable_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def decode_config_for_json(config):
    return {key: jsonable_value(value) for key, value in config.items()}


ACTION_METRIC_NAMES = (
    "eligible_visit_count",
    "first_visit_count",
    "continue_precision",
    "continue_recall",
    "continue_f1",
    "add_precision",
    "add_recall",
    "add_f1",
    "remove_precision",
    "remove_recall",
    "remove_f1",
    "new_drug_precision",
    "transition_macro_f1",
    "true_continue_mean",
    "pred_continue_mean",
    "true_add_mean",
    "pred_add_mean",
    "true_remove_mean",
    "pred_remove_mean",
)


GRADIENT_LOSS_NAMES = ("L_med", "L_continue", "L_add", "L_remove")
GRADIENT_LOSS_PAIRS = (
    ("L_med", "L_continue"),
    ("L_med", "L_add"),
    ("L_med", "L_remove"),
    ("L_continue", "L_add"),
    ("L_continue", "L_remove"),
    ("L_add", "L_remove"),
)


ACTION_NAMES = ("continue", "add", "remove")


COUNT_PREDICTOR_TARGETS = ("total", "continue", "add", "remove")


def unsupported_paper_prefix_modes():
    return {
        "seq_greedy",
        "seq_hidr",
        "seq_rank_fixed_topk",
        "seq_rank_expected_count",
        "seq_rank_history_count",
        "seq_rank_predicted_count",
        "predicted_count",
        "oracle_count",
        "per_med_threshold",
        "retrieval_rank_fixed_topk",
        "retrieval_rank_threshold_count",
        "retrieval_rank_expected_count",
        "retrieval_rank_history_count",
        "retrieval_rank_predicted_count",
        "history_retrieval_freq_rank_expected_count",
        "history_retrieval_freq_ddi_rank_expected_count",
        "history_retrieval_freq_graph_rank_expected_count",
        "history_retrieval_freq_graph_rank_retrieval_count",
        "history_retrieval_freq_graph_rank_count_pair_mix",
    }


def paper_prefix_decode_configs():
    configs = [
        config for config in decode_configs()
        if config["mode"] not in unsupported_paper_prefix_modes()
    ]
    if not configs:
        raise ValueError("--prefix_metric_mode paper_prefix does not currently support the selected decoder modes")
    return configs


def counts_from_config(cache, config):
    prob = cache["prob"]
    mode = config["mode"]
    count_mode = None
    if mode.startswith("history_retrieval_freq_graph_rank_"):
        count_mode = mode.removeprefix("history_retrieval_freq_graph_rank_")
    elif mode.startswith("history_retrieval_freq_ddi_rank_"):
        count_mode = mode.removeprefix("history_retrieval_freq_ddi_rank_")
    elif mode.startswith("history_retrieval_freq_rank_"):
        count_mode = mode.removeprefix("history_retrieval_freq_rank_")
    elif mode.startswith("history_retrieval_rank_"):
        count_mode = mode.removeprefix("history_retrieval_rank_")
    elif mode.startswith("history_rank_"):
        count_mode = mode.removeprefix("history_rank_")
    elif mode.startswith("retrieval_rank_"):
        count_mode = mode.removeprefix("retrieval_rank_")
    elif mode in {"fixed_topk", "expected_count", "expected_jaccard_count", "history_count", "predicted_count", "oracle_count"}:
        count_mode = mode

    if count_mode is None:
        raise ValueError(f"count decoding requires a count-based decoder, got {mode}")

    decode_batches = cache["decode_batches"]
    if count_mode == "fixed_topk":
        counts = torch.full((prob.shape[0],), int(config["topk"]), device=prob.device, dtype=torch.long)
    elif count_mode == "threshold_count":
        threshold_counts = (prob >= float(config["threshold"])).sum(dim=1).to(dtype=prob.dtype)
        counts = torch.round(threshold_counts * float(config["ratio"]) + float(config["bias"]))
    elif count_mode == "expected_count":
        counts = torch.round(prob.sum(dim=1) * float(config["ratio"]) + float(config["bias"]))
    elif count_mode == "expected_jaccard_count":
        counts = expected_jaccard_counts(scores_from_config(cache, config), prob)
    elif count_mode == "count_pair_mix":
        counts = count_pair_mix_counts(prob, scores_from_config(cache, config), decode_batches, config)
    elif count_mode == "history_count":
        history_counts = decode_batches["med_history_counts"]
        if history_counts is None:
            raise ValueError(f"{mode} decoder requires med_history_counts")
        counts = torch.round(history_counts.to(device=prob.device, dtype=prob.dtype) * float(config["ratio"]) + float(config["bias"]))
    elif count_mode == "predicted_count":
        med_count_pred = decode_batches["med_count_pred"]
        if med_count_pred is None:
            raise ValueError(f"{mode} decoder requires med_count_pred")
        counts = torch.round(med_count_pred.to(device=prob.device, dtype=prob.dtype) * float(config["ratio"]) + float(config["bias"]))
    elif count_mode == "retrieval_count":
        counts = retrieval_context_counts(decode_batches, scores_from_config(cache, config), config)
    elif count_mode == "oracle_count":
        counts = decode_batches["label_counts"]
        if counts is None:
            raise ValueError("oracle_count decoder requires label_counts")
    else:
        raise ValueError(f"unsupported count mode: {count_mode}")
    min_count, max_count = decode_count_limits(prob.shape[1])
    return counts.to(device=prob.device, dtype=torch.long).clamp(min=min_count, max=max_count)


def scores_from_config(cache, config):
    prob = cache["prob"]
    mode = config["mode"]
    decode_batches = cache["decode_batches"]
    if mode.startswith("history_retrieval_freq_graph_rank_"):
        return history_retrieval_freq_graph_scores(prob, decode_batches, config)
    if mode.startswith("history_retrieval_freq_ddi_rank_") or mode.startswith("history_retrieval_freq_rank_"):
        history_signal = decode_batches["med_history_binary"]
        retrieval_signal = decode_batches["retrieval_priors"]
        freq_signal = decode_batches["med_frequency_prior"]
        if history_signal is None or retrieval_signal is None or freq_signal is None:
            raise ValueError(f"{mode} decoder requires med_history_binary, retrieval_priors, and med_frequency_prior")
        return (
            prob
            + float(config["history_weight"]) * history_signal.to(device=prob.device, dtype=prob.dtype)
            + float(config["retrieval_weight"]) * retrieval_signal.to(device=prob.device, dtype=prob.dtype)
            + float(config["freq_weight"]) * freq_signal.to(device=prob.device, dtype=prob.dtype).view(1, -1)
        )
    if mode.startswith("history_retrieval_rank_"):
        history_signal = decode_batches["med_history_binary"]
        retrieval_signal = decode_batches["retrieval_priors"]
        if history_signal is None or retrieval_signal is None:
            raise ValueError(f"{mode} decoder requires med_history_binary and retrieval_priors")
        return (
            prob
            + float(config["history_weight"]) * history_signal.to(device=prob.device, dtype=prob.dtype)
            + float(config["retrieval_weight"]) * retrieval_signal.to(device=prob.device, dtype=prob.dtype)
        )
    if mode.startswith("history_rank_"):
        history_signal = decode_batches["med_history_binary"]
        if history_signal is None:
            raise ValueError(f"{mode} decoder requires med_history_binary")
        return prob + float(config["weight"]) * history_signal.to(device=prob.device, dtype=prob.dtype)
    if mode.startswith("retrieval_rank_"):
        retrieval_signal = decode_batches["retrieval_priors"]
        if retrieval_signal is None:
            raise ValueError(f"{mode} decoder requires retrieval_priors")
        return prob + float(config["weight"]) * retrieval_signal.to(device=prob.device, dtype=prob.dtype)
    if mode in {"fixed_topk", "expected_count", "expected_jaccard_count", "history_count", "predicted_count", "oracle_count"}:
        return prob
    raise ValueError(f"rank/count decoding requires a rank/count decoder, got {mode}")


def treatment_episode_action_prior_from_prob(prob):
    if prob is None:
        return None
    return prob.detach().cpu().to(dtype=torch.float64).clamp(0.0, 1.0)


def build_treatment_episode_static_context(
    target_data,
    candidate_data,
    target_clinical_payload,
    candidate_clinical_payload,
    target_clinical_state_payload,
    candidate_clinical_state_payload,
    target_patient_id_offsets,
    candidate_patient_id_offsets,
    target_start_visit,
    candidate_start_visit,
    target_action_prior=None,
    exclude_same_patient_current_and_future=None,
    allow_train_or_own_history_only=None,
):
    if exclude_same_patient_current_and_future is None:
        exclude_same_patient_current_and_future = effective_exclude_same_patient_current_and_future()
    if allow_train_or_own_history_only is None:
        allow_train_or_own_history_only = uses_train_plus_own_history_scope()
    return build_episode_retrieval_context(
        target_data=target_data,
        candidate_data=candidate_data,
        voc_size=voc_size,
        device=device,
        topk=args.retrieval_topk,
        config=make_treatment_episode_config(),
        target_clinical_payload=target_clinical_payload,
        candidate_clinical_payload=candidate_clinical_payload,
        target_clinical_state_payload=target_clinical_state_payload,
        candidate_clinical_state_payload=candidate_clinical_state_payload,
        ddi_adj=ddi_A,
        exclude_same_patient_current_and_future=exclude_same_patient_current_and_future,
        allow_train_or_own_history_only=allow_train_or_own_history_only,
        train_patient_id_limit=len(data_train),
        target_patient_id_offsets=target_patient_id_offsets,
        candidate_patient_id_offsets=candidate_patient_id_offsets,
        target_start_visit=target_start_visit,
        candidate_start_visit=candidate_start_visit,
        target_action_prior=target_action_prior,
    )


def refresh_treatment_episode_static_context(
    model,
    batches,
    target_data,
    candidate_data,
    target_clinical_payload,
    candidate_clinical_payload,
    target_clinical_state_payload,
    candidate_clinical_state_payload,
    target_patient_id_offsets,
    candidate_patient_id_offsets,
    batch_size,
    target_start_visit,
    candidate_start_visit,
    exclude_same_patient_current_and_future=None,
    allow_train_or_own_history_only=None,
):
    if not (retrieval_is_treatment_episode and args.treatment_episode_action_weight > 0):
        return None
    was_training = model.training
    clear_retrieval_context_batches(batches)
    clear_treatment_episode_context_batches(batches)
    batches["retrieval_prior_batches"] = None
    try:
        _, y_pred_prob, _, _ = infer_prob(model, batches)
    finally:
        model.train(was_training)
    target_action_prior = treatment_episode_action_prior_from_prob(y_pred_prob)
    if exclude_same_patient_current_and_future is None:
        exclude_same_patient_current_and_future = effective_exclude_same_patient_current_and_future()
    if allow_train_or_own_history_only is None:
        allow_train_or_own_history_only = uses_train_plus_own_history_scope()
    if args.retrieval_weight > 0 or args.retrieval_copy_weight > 0 or retrieval_rank_decoder_enabled:
        retrieval_priors = build_episode_retrieval_priors(
            target_data=target_data,
            candidate_data=candidate_data,
            voc_size=voc_size,
            device=device,
            topk=args.retrieval_topk,
            config=make_treatment_episode_config(),
            prior_transform=args.retrieval_prior_transform,
            target_clinical_payload=target_clinical_payload,
            candidate_clinical_payload=candidate_clinical_payload,
            target_clinical_state_payload=target_clinical_state_payload,
            candidate_clinical_state_payload=candidate_clinical_state_payload,
            ddi_adj=ddi_A,
            exclude_same_patient_current_and_future=exclude_same_patient_current_and_future,
            allow_train_or_own_history_only=allow_train_or_own_history_only,
            train_patient_id_limit=len(data_train),
            target_patient_id_offsets=target_patient_id_offsets,
            candidate_patient_id_offsets=candidate_patient_id_offsets,
            target_start_visit=target_start_visit,
            candidate_start_visit=candidate_start_visit,
            target_action_prior=target_action_prior,
        )
        set_retrieval_prior_batches(batches, retrieval_priors, batch_size)
    context = build_treatment_episode_static_context(
        target_data=target_data,
        candidate_data=candidate_data,
        target_clinical_payload=target_clinical_payload,
        candidate_clinical_payload=candidate_clinical_payload,
        target_clinical_state_payload=target_clinical_state_payload,
        candidate_clinical_state_payload=candidate_clinical_state_payload,
        target_patient_id_offsets=target_patient_id_offsets,
        candidate_patient_id_offsets=candidate_patient_id_offsets,
        target_start_visit=target_start_visit,
        candidate_start_visit=candidate_start_visit,
        target_action_prior=target_action_prior,
        exclude_same_patient_current_and_future=exclude_same_patient_current_and_future,
        allow_train_or_own_history_only=allow_train_or_own_history_only,
    )
    set_retrieval_context_batches(batches, context, batch_size)
    return context


def infer_prob_without_treatment_episode_context(model, batches):
    was_training = model.training
    previous_retrieval_prior_batches = batches.get("retrieval_prior_batches")
    previous_retrieval_med_batches = batches.get("retrieval_med_batches")
    previous_retrieval_score_batches = batches.get("retrieval_score_batches")
    previous_retrieval_query_batches = batches.get("retrieval_query_batches")
    previous_treatment_episode_score_batches = batches.get("treatment_episode_score_batches")
    previous_treatment_episode_experience_batches = batches.get("treatment_episode_experience_batches")
    clear_retrieval_context_batches(batches)
    clear_treatment_episode_context_batches(batches)
    batches["retrieval_prior_batches"] = None
    try:
        return infer_prob(model, batches)
    finally:
        batches["retrieval_prior_batches"] = previous_retrieval_prior_batches
        batches["retrieval_med_batches"] = previous_retrieval_med_batches
        batches["retrieval_score_batches"] = previous_retrieval_score_batches
        batches["retrieval_query_batches"] = previous_retrieval_query_batches
        batches["treatment_episode_score_batches"] = previous_treatment_episode_score_batches
        batches["treatment_episode_experience_batches"] = previous_treatment_episode_experience_batches
        model.train(was_training)


def refresh_treatment_episode_batches(model, include_train=True, include_eval=True):
    global train_retrieval_priors, eval_retrieval_priors
    global train_treatment_episode_context, eval_treatment_episode_context
    global train_static_retrieval_context, eval_static_retrieval_context
    if not retrieval_is_treatment_episode or args.treatment_episode_action_weight <= 0:
        return
    was_training = model.training
    try:
        train_action_prior = None
        eval_action_prior = None
        if include_train:
            _, train_prob, _, _ = infer_prob_without_treatment_episode_context(model, train_batches)
            train_action_prior = treatment_episode_action_prior_from_prob(train_prob)
        if include_eval:
            _, eval_prob, _, _ = infer_prob_without_treatment_episode_context(model, eval_batches)
            eval_action_prior = treatment_episode_action_prior_from_prob(eval_prob)

        if include_train and (args.retrieval_weight > 0 or args.retrieval_copy_weight > 0 or retrieval_rank_decoder_enabled):
            train_retrieval_priors = build_episode_retrieval_priors(
                target_data=data_train,
                candidate_data=data_train,
                voc_size=voc_size,
                device=device,
                topk=args.retrieval_topk,
                config=make_treatment_episode_config(),
                prior_transform=args.retrieval_prior_transform,
                target_clinical_payload=retrieval_train_clinical_payload if treatment_episode_uses_clinical else None,
                candidate_clinical_payload=retrieval_train_clinical_payload if treatment_episode_uses_clinical else None,
                target_clinical_state_payload=retrieval_train_clinical_state_payload,
                candidate_clinical_state_payload=retrieval_train_clinical_state_payload,
                ddi_adj=ddi_A,
                exclude_same_patient_current_and_future=True,
                target_patient_id_offsets=train_patient_id_offsets,
                candidate_patient_id_offsets=train_patient_id_offsets,
                target_start_visit=target_start_visit,
                candidate_start_visit=retrieval_candidate_start_visit(),
                target_action_prior=train_action_prior,
            )
            set_retrieval_prior_batches(train_batches, train_retrieval_priors, args.batch)
        if include_eval and (args.retrieval_weight > 0 or args.retrieval_copy_weight > 0 or retrieval_rank_decoder_enabled):
            eval_retrieval_priors = build_episode_retrieval_priors(
                target_data=data_eval,
                candidate_data=eval_retrieval_candidate_data,
                voc_size=voc_size,
                device=device,
                topk=args.retrieval_topk,
                config=make_treatment_episode_config(),
                prior_transform=args.retrieval_prior_transform,
                target_clinical_payload=retrieval_eval_clinical_payload if treatment_episode_uses_clinical else None,
                candidate_clinical_payload=eval_retrieval_candidate_clinical_payload if treatment_episode_uses_clinical else None,
                target_clinical_state_payload=retrieval_eval_clinical_state_payload,
                candidate_clinical_state_payload=eval_retrieval_candidate_clinical_state_payload,
                ddi_adj=ddi_A,
                exclude_same_patient_current_and_future=effective_exclude_same_patient_current_and_future(),
                allow_train_or_own_history_only=uses_train_plus_own_history_scope(),
                train_patient_id_limit=len(data_train),
                target_patient_id_offsets=eval_patient_id_offsets,
                candidate_patient_id_offsets=eval_retrieval_candidate_patient_id_offsets,
                target_start_visit=target_start_visit,
                candidate_start_visit=retrieval_candidate_start_visit(),
                target_action_prior=eval_action_prior,
            )
            set_retrieval_prior_batches(eval_batches, eval_retrieval_priors, infer_batch_size)

        if include_train and (
            train_treatment_episode_context is not None
            or retrieval_context_is_static
            or (retrieval_count_decoder_enabled and retrieval_is_treatment_episode)
            or (episode_action_count_sources_enabled and retrieval_is_treatment_episode)
        ):
            train_treatment_episode_context = build_treatment_episode_static_context(
                target_data=data_train,
                candidate_data=data_train,
                target_clinical_payload=retrieval_train_clinical_payload if treatment_episode_uses_clinical else None,
                candidate_clinical_payload=retrieval_train_clinical_payload if treatment_episode_uses_clinical else None,
                target_clinical_state_payload=retrieval_train_clinical_state_payload,
                candidate_clinical_state_payload=retrieval_train_clinical_state_payload,
                target_patient_id_offsets=train_patient_id_offsets,
                candidate_patient_id_offsets=train_patient_id_offsets,
                target_start_visit=target_start_visit,
                candidate_start_visit=retrieval_candidate_start_visit(),
                target_action_prior=train_action_prior,
                exclude_same_patient_current_and_future=True,
                allow_train_or_own_history_only=False,
            )
            if retrieval_context_is_static or (retrieval_count_decoder_enabled and retrieval_is_treatment_episode) or (
                episode_action_count_sources_enabled and retrieval_is_treatment_episode
            ):
                train_static_retrieval_context = train_treatment_episode_context
            set_retrieval_context_batches(train_batches, train_treatment_episode_context, args.batch)
            if train_treatment_episode_context is not None:
                set_treatment_episode_context_batches(train_batches, train_treatment_episode_context, args.batch)
        if include_eval and (
            eval_treatment_episode_context is not None
            or retrieval_context_is_static
            or (retrieval_count_decoder_enabled and retrieval_is_treatment_episode)
            or (episode_action_count_sources_enabled and retrieval_is_treatment_episode)
        ):
            eval_treatment_episode_context = build_treatment_episode_static_context(
                target_data=data_eval,
                candidate_data=eval_retrieval_candidate_data,
                target_clinical_payload=retrieval_eval_clinical_payload if treatment_episode_uses_clinical else None,
                candidate_clinical_payload=eval_retrieval_candidate_clinical_payload if treatment_episode_uses_clinical else None,
                target_clinical_state_payload=retrieval_eval_clinical_state_payload,
                candidate_clinical_state_payload=eval_retrieval_candidate_clinical_state_payload,
                target_patient_id_offsets=eval_patient_id_offsets,
                candidate_patient_id_offsets=eval_retrieval_candidate_patient_id_offsets,
                target_start_visit=target_start_visit,
                candidate_start_visit=retrieval_candidate_start_visit(),
                target_action_prior=eval_action_prior,
            )
            if retrieval_context_is_static or (retrieval_count_decoder_enabled and retrieval_is_treatment_episode) or (
                episode_action_count_sources_enabled and retrieval_is_treatment_episode
            ):
                eval_static_retrieval_context = eval_treatment_episode_context
            set_retrieval_context_batches(eval_batches, eval_treatment_episode_context, infer_batch_size)
            if eval_treatment_episode_context is not None:
                set_treatment_episode_context_batches(eval_batches, eval_treatment_episode_context, infer_batch_size)
    finally:
        model.train(was_training)


def clamp_round_counts(counts, voc_size):
    min_count, max_count = decode_count_limits(voc_size)
    return torch.round(counts).to(dtype=torch.long).clamp(min=min_count, max=max_count)


def masked_topk_indexes(values, mask, k):
    if k <= 0:
        raise ValueError("masked_topk_indexes requires positive k")
    masked = torch.where(mask, values, torch.full_like(values, -1e9))
    top_values, top_indexes = torch.topk(masked, k=min(int(k), values.shape[1]), dim=1)
    valid = top_values > -1e8
    if top_values.shape[1] < k:
        pad_count = int(k) - int(top_values.shape[1])
        top_indexes = torch.cat(
            (
                top_indexes,
                torch.zeros((values.shape[0], pad_count), device=values.device, dtype=top_indexes.dtype),
            ),
            dim=1,
        )
        valid = torch.cat(
            (
                valid,
                torch.zeros((values.shape[0], pad_count), device=values.device, dtype=torch.bool),
            ),
            dim=1,
        )
    return top_indexes, valid

def get_metrics_on_test_data(model, decode_config=None):
    if decode_config is None:
        decode_config = decode_configs()[0]
    results=[]
    test_patient_payload = []
    insert_needs_clinical_payload = (
        args.insert_retrieved_visits
        and ("clinical" in args.insert_visit_mode or model_uses_clinical_features)
    )
    insert_train_clinical_payload = train_clinical_feature_payload if insert_needs_clinical_payload else None
    test_global_patient_ids = test_patient_id_offsets
    for patient_idx, patient in enumerate(data_test):
        local_patient_start = count_prediction_visits(data_test[:patient_idx])
        item = {
            "data": patient,
            "global_patient_idx": test_global_patient_ids[patient_idx],
            "local_patient_start": local_patient_start,
            "local_patient_end": local_patient_start + max(0, len(patient) - target_start_visit),
        }
        if test_clinical_state_labels is not None and test_clinical_state_masks is not None:
            item.update(
                clinical_state_labels=test_clinical_state_labels[patient_idx],
                clinical_state_masks=test_clinical_state_masks[patient_idx],
            )
            if test_med_transition_labels is not None and test_med_transition_masks is not None:
                item.update(
                    med_transition_labels=test_med_transition_labels[patient_idx],
                    med_transition_masks=test_med_transition_masks[patient_idx],
                )
        if test_clinical_feature_payload is not None:
            item.update(
                clinical_features=test_clinical_feature_payload[0][patient_idx],
                clinical_feature_masks=test_clinical_feature_payload[1][patient_idx],
                clinical_feature_abnormal=test_clinical_feature_payload[2][patient_idx],
                clinical_feature_delta_hours=test_clinical_feature_payload[3][patient_idx],
            )
        test_patient_payload.append(item)
    for i in range(10):
        random.shuffle(test_patient_payload)
        sampled_payload = test_patient_payload[:int(len(test_patient_payload)*0.8)]
        sampled_test_data = [item["data"] for item in sampled_payload]
        test_batch_kwargs = {}
        if args.multitask_mode == "clinical_state" and test_clinical_state_labels is not None and test_clinical_state_masks is not None:
            test_batch_kwargs.update(
                clinical_state_targets=flatten_visit_level_artifacts([item["clinical_state_labels"] for item in sampled_payload]),
                clinical_state_masks=flatten_visit_level_artifacts([item["clinical_state_masks"] for item in sampled_payload]),
                med_transition_targets=flatten_visit_level_artifacts([item["med_transition_labels"] for item in sampled_payload]),
                med_transition_masks=flatten_visit_level_artifacts([item["med_transition_masks"] for item in sampled_payload]),
            )
        if model_uses_clinical_features and test_clinical_feature_payload is not None:
            test_batch_kwargs.update(
                clinical_features=[item["clinical_features"] for item in sampled_payload],
                clinical_feature_masks=[item["clinical_feature_masks"] for item in sampled_payload],
                clinical_feature_abnormal=[item["clinical_feature_abnormal"] for item in sampled_payload],
                clinical_feature_delta_hours=[item["clinical_feature_delta_hours"] for item in sampled_payload],
            )
        test_static_retrieval_context = None
        if (
            args.retrieval_mode != "none"
            and not retrieval_is_learned
            and not retrieval_is_treatment_episode
            and not retrieval_is_paper_m3
            and not retrieval_is_official_topidx
            and (args.retrieval_weight > 0 or args.retrieval_copy_mode == "fixed_prob" or retrieval_rank_decoder_enabled)
        ):
            target_clinical_payload = None
            if retrieval_requires_clinical:
                target_clinical_payload = (
                    [item["clinical_features"] for item in sampled_payload],
                    [item["clinical_feature_masks"] for item in sampled_payload],
                    [item["clinical_feature_abnormal"] for item in sampled_payload],
                    [item["clinical_feature_delta_hours"] for item in sampled_payload],
                )
            test_candidate_data = nonlearned_test_candidate_data(sampled_test_data)
            test_candidate_patient_id_offsets = nonlearned_test_candidate_patient_id_offsets(sampled_payload)
            test_candidate_clinical_payload = (
                nonlearned_test_candidate_clinical_payload(sampled_payload)
                if retrieval_requires_clinical
                else None
            )
            test_batch_kwargs["retrieval_priors"] = build_retrieval_priors(
                target_data=sampled_test_data,
                candidate_data=test_candidate_data,
                voc_size=voc_size,
                device=device,
                mode=args.retrieval_mode,
                topk=args.retrieval_topk,
                alpha=args.retrieval_alpha,
                prior_transform=args.retrieval_prior_transform,
                target_clinical_payload=target_clinical_payload,
                candidate_clinical_payload=test_candidate_clinical_payload,
                exclude_same_patient_current_and_future=effective_exclude_same_patient_current_and_future(),
                allow_train_or_own_history_only=uses_train_plus_own_history_scope(),
                train_patient_id_limit=len(data_train),
                target_patient_id_offsets=[item["global_patient_idx"] for item in sampled_payload],
                candidate_patient_id_offsets=test_candidate_patient_id_offsets,
                candidate_start_visit=retrieval_candidate_start_visit(),
            )
        if retrieval_is_treatment_episode and (
            args.retrieval_weight > 0 or args.retrieval_copy_mode == "fixed_prob" or retrieval_rank_decoder_enabled
        ):
            target_clinical_payload = None
            if treatment_episode_uses_clinical:
                target_clinical_payload = (
                    [item["clinical_features"] for item in sampled_payload],
                    [item["clinical_feature_masks"] for item in sampled_payload],
                    [item["clinical_feature_abnormal"] for item in sampled_payload],
                    [item["clinical_feature_delta_hours"] for item in sampled_payload],
                )
            target_clinical_state_payload = (
                (
                    [item["clinical_state_labels"] for item in sampled_payload],
                    [item["clinical_state_masks"] for item in sampled_payload],
                )
                if retrieval_test_clinical_state_payload is not None
                else None
            )
            test_candidate_data = nonlearned_test_candidate_data(sampled_test_data)
            test_candidate_patient_id_offsets = nonlearned_test_candidate_patient_id_offsets(sampled_payload)
            test_candidate_clinical_payload = (
                nonlearned_test_candidate_clinical_payload(sampled_payload)
                if treatment_episode_uses_clinical
                else None
            )
            test_candidate_clinical_state_payload = nonlearned_test_candidate_clinical_state_payload(sampled_payload)
            test_batch_kwargs["retrieval_priors"] = build_episode_retrieval_priors(
                target_data=sampled_test_data,
                candidate_data=test_candidate_data,
                voc_size=voc_size,
                device=device,
                topk=args.retrieval_topk,
                config=make_treatment_episode_config(),
                prior_transform=args.retrieval_prior_transform,
                target_clinical_payload=target_clinical_payload,
                candidate_clinical_payload=test_candidate_clinical_payload,
                target_clinical_state_payload=target_clinical_state_payload,
                candidate_clinical_state_payload=test_candidate_clinical_state_payload,
                ddi_adj=ddi_A,
                exclude_same_patient_current_and_future=effective_exclude_same_patient_current_and_future(),
                allow_train_or_own_history_only=uses_train_plus_own_history_scope(),
                train_patient_id_limit=len(data_train),
                target_patient_id_offsets=[item["global_patient_idx"] for item in sampled_payload],
                candidate_patient_id_offsets=test_candidate_patient_id_offsets,
                target_start_visit=target_start_visit,
                candidate_start_visit=retrieval_candidate_start_visit(),
            )
        if retrieval_is_official_topidx and (
            args.retrieval_weight > 0 or args.retrieval_copy_mode == "fixed_prob" or retrieval_rank_decoder_enabled
        ):
            full_test_priors = build_official_topidx_priors("test", data_test, data_test)
            sampled_test_priors = [
                prior
                for item in sampled_payload
                for prior in full_test_priors[item["local_patient_start"]:item["local_patient_end"]]
            ]
            test_batch_kwargs["retrieval_priors"] = sampled_test_priors
        if retrieval_context_is_static and not paper_m3_artifacts_required and not retrieval_is_treatment_episode:
            target_clinical_payload = None
            if retrieval_requires_clinical:
                target_clinical_payload = (
                    [item["clinical_features"] for item in sampled_payload],
                    [item["clinical_feature_masks"] for item in sampled_payload],
                    [item["clinical_feature_abnormal"] for item in sampled_payload],
                    [item["clinical_feature_delta_hours"] for item in sampled_payload],
                )
            test_candidate_data = nonlearned_test_candidate_data(sampled_test_data)
            test_candidate_patient_id_offsets = nonlearned_test_candidate_patient_id_offsets(sampled_payload)
            test_candidate_clinical_payload = (
                nonlearned_test_candidate_clinical_payload(sampled_payload)
                if retrieval_requires_clinical
                else None
            )
            test_static_retrieval_context = build_static_retrieval_context(
                target_data=sampled_test_data,
                candidate_data=test_candidate_data,
                voc_size=voc_size,
                mode=args.retrieval_mode,
                topk=args.retrieval_topk,
                alpha=args.retrieval_alpha,
                target_clinical_payload=target_clinical_payload,
                candidate_clinical_payload=test_candidate_clinical_payload,
                exclude_same_patient_current_and_future=effective_exclude_same_patient_current_and_future(),
                allow_train_or_own_history_only=uses_train_plus_own_history_scope(),
                train_patient_id_limit=len(data_train),
                target_patient_id_offsets=[item["global_patient_idx"] for item in sampled_payload],
                candidate_patient_id_offsets=test_candidate_patient_id_offsets,
                candidate_start_visit=retrieval_candidate_start_visit(),
            )
        if (
            retrieval_context_is_static
            or retrieval_count_decoder_enabled
            or episode_action_count_sources_enabled
        ) and retrieval_is_treatment_episode:
            target_clinical_payload = None
            if treatment_episode_uses_clinical:
                target_clinical_payload = (
                    [item["clinical_features"] for item in sampled_payload],
                    [item["clinical_feature_masks"] for item in sampled_payload],
                    [item["clinical_feature_abnormal"] for item in sampled_payload],
                    [item["clinical_feature_delta_hours"] for item in sampled_payload],
                )
            target_clinical_state_payload = (
                (
                    [item["clinical_state_labels"] for item in sampled_payload],
                    [item["clinical_state_masks"] for item in sampled_payload],
                )
                if retrieval_test_clinical_state_payload is not None
                else None
            )
            test_candidate_data = nonlearned_test_candidate_data(sampled_test_data)
            test_candidate_patient_id_offsets = nonlearned_test_candidate_patient_id_offsets(sampled_payload)
            test_candidate_clinical_payload = (
                nonlearned_test_candidate_clinical_payload(sampled_payload)
                if treatment_episode_uses_clinical
                else None
            )
            test_candidate_clinical_state_payload = nonlearned_test_candidate_clinical_state_payload(sampled_payload)
            test_static_retrieval_context = build_episode_retrieval_context(
                target_data=sampled_test_data,
                candidate_data=test_candidate_data,
                voc_size=voc_size,
                device=device,
                topk=args.retrieval_topk,
                config=make_treatment_episode_config(),
                target_clinical_payload=target_clinical_payload,
                candidate_clinical_payload=test_candidate_clinical_payload,
                target_clinical_state_payload=target_clinical_state_payload,
                candidate_clinical_state_payload=test_candidate_clinical_state_payload,
                ddi_adj=ddi_A,
                exclude_same_patient_current_and_future=effective_exclude_same_patient_current_and_future(),
                allow_train_or_own_history_only=uses_train_plus_own_history_scope(),
                train_patient_id_limit=len(data_train),
                target_patient_id_offsets=[item["global_patient_idx"] for item in sampled_payload],
                candidate_patient_id_offsets=test_candidate_patient_id_offsets,
                target_start_visit=target_start_visit,
                candidate_start_visit=retrieval_candidate_start_visit(),
            )
        if args.insert_retrieved_visits and not insert_is_learned and not insert_is_paper_m3:
            target_clinical_payload = None
            if insert_needs_clinical_payload:
                target_clinical_payload = (
                    [item["clinical_features"] for item in sampled_payload],
                    [item["clinical_feature_masks"] for item in sampled_payload],
                    [item["clinical_feature_abnormal"] for item in sampled_payload],
                    [item["clinical_feature_delta_hours"] for item in sampled_payload],
                )
            test_candidate_data = nonlearned_test_candidate_data(sampled_test_data)
            test_candidate_patient_id_offsets = nonlearned_test_candidate_patient_id_offsets(sampled_payload)
            test_candidate_clinical_payload = (
                nonlearned_test_candidate_clinical_payload(sampled_payload)
                if insert_needs_clinical_payload
                else None
            )
            test_batch_kwargs["inserted_visits"] = build_inserted_retrieved_visits(
                target_data=sampled_test_data,
                candidate_data=test_candidate_data,
                voc_size=voc_size,
                mode=args.insert_visit_mode,
                topk=args.insert_visit_topk,
                alpha=args.insert_visit_alpha,
                target_clinical_payload=target_clinical_payload,
                candidate_clinical_payload=test_candidate_clinical_payload,
                exclude_same_patient_current_and_future=effective_exclude_same_patient_current_and_future(),
                allow_train_or_own_history_only=uses_train_plus_own_history_scope(),
                train_patient_id_limit=len(data_train),
                target_patient_id_offsets=[item["global_patient_idx"] for item in sampled_payload],
                candidate_patient_id_offsets=test_candidate_patient_id_offsets,
                candidate_start_visit=retrieval_candidate_start_visit(),
            )
        if paper_m3_artifacts_required:
            test_candidate_data = nonlearned_test_candidate_data(sampled_test_data)
            test_candidate_patient_id_offsets = nonlearned_test_candidate_patient_id_offsets(sampled_payload)
            test_candidate_clinical_payload = None
            if model_uses_clinical_features or args.paper_m3_clinical_weight > 0:
                test_candidate_clinical_payload = nonlearned_test_candidate_clinical_payload(sampled_payload)
            test_candidate_batches = create_batches_dict(
                test_candidate_data,
                infer_batch_size,
                voc_size,
                device,
                start_visit=retrieval_candidate_start_visit(),
                **add_clinical_feature_kwargs(
                    {},
                    test_candidate_clinical_payload if model_uses_clinical_features else None,
                ),
            )
            test_candidate_batches["split_name"] = "paper_m3_test_candidate"
            test_candidate_metadata = build_visit_metadata(
                test_candidate_data,
                voc_size,
                test_candidate_patient_id_offsets,
                start_visit=retrieval_candidate_start_visit(),
            )
            test_target_retrieval_inputs = build_visit_retrieval_inputs(
                sampled_test_data,
                voc_size,
                patient_id_offsets=[item["global_patient_idx"] for item in sampled_payload],
            ) if args.paper_m3_dxproc_weight > 0 else None
            test_candidate_retrieval_inputs = build_visit_retrieval_inputs(
                test_candidate_data,
                voc_size,
                patient_id_offsets=test_candidate_patient_id_offsets,
                start_visit=retrieval_candidate_start_visit(),
            ) if args.paper_m3_dxproc_weight > 0 else None
            test_candidate_visits = flatten_target_visits(
                test_candidate_data,
                start_visit=retrieval_candidate_start_visit(),
            )
            test_candidate_clinical = (
                flatten_target_clinical_artifacts(
                    test_candidate_clinical_payload,
                    start_visit=retrieval_candidate_start_visit(),
                )
                if test_candidate_clinical_payload is not None
                else None
            )
            target_clinical_payload = None
            if model_uses_clinical_features or args.paper_m3_clinical_weight > 0:
                target_clinical_payload = (
                    [item["clinical_features"] for item in sampled_payload],
                    [item["clinical_feature_masks"] for item in sampled_payload],
                    [item["clinical_feature_abnormal"] for item in sampled_payload],
                    [item["clinical_feature_delta_hours"] for item in sampled_payload],
                )
            test_target_clinical_inputs = flatten_clinical_similarity_inputs(
                target_clinical_payload,
                start_visit=target_start_visit,
            ) if args.paper_m3_clinical_weight > 0 else None
            test_candidate_clinical_inputs = flatten_clinical_similarity_inputs(
                test_candidate_clinical_payload,
                start_visit=retrieval_candidate_start_visit(),
            ) if args.paper_m3_clinical_weight > 0 else None
            target_batches_for_m3 = create_batches_dict(
                sampled_test_data,
                infer_batch_size,
                voc_size,
                device,
                start_visit=target_start_visit,
                **add_clinical_feature_kwargs(
                    {},
                    target_clinical_payload if model_uses_clinical_features else None,
                ),
            )
            target_batches_for_m3["split_name"] = "paper_m3_test_target"
            sampled_offsets = [item["global_patient_idx"] for item in sampled_payload]
            test_paper_m3_artifacts = cached_paper_m3_artifacts(
                paper_m3_cache_path(
                    "test",
                    sampled_offsets,
                    test_candidate_patient_id_offsets,
                    retrieval_candidate_start_visit(),
                ),
                lambda: build_paper_m3_artifacts(
                    phi_model=paper_m3_phi_model,
                    target_batches=target_batches_for_m3,
                    candidate_batches=test_candidate_batches,
                    target_metadata=build_visit_metadata(
                        sampled_test_data,
                        voc_size,
                        sampled_offsets,
                    ),
                    candidate_metadata=test_candidate_metadata,
                    target_total=count_prediction_visits(sampled_test_data),
                    candidate_total=count_visits_from(test_candidate_data, retrieval_candidate_start_visit()),
                    candidate_visits=test_candidate_visits,
                    target_retrieval_inputs=test_target_retrieval_inputs,
                    candidate_retrieval_inputs=test_candidate_retrieval_inputs,
                    candidate_clinical=test_candidate_clinical,
                    target_clinical_inputs=test_target_clinical_inputs,
                    candidate_clinical_inputs=test_candidate_clinical_inputs,
                    exclude_same_patient_current_and_future=effective_exclude_same_patient_current_and_future(),
                    allow_train_or_own_history_only=uses_train_plus_own_history_scope(),
                    train_patient_id_limit=len(data_train),
                    query_transform_weights=paper_m3_query_transform_weights,
                ),
            )
            if insert_is_paper_m3:
                test_batch_kwargs["inserted_visits"] = test_paper_m3_artifacts["inserted_visits"]
            test_batch_kwargs["retrieval_priors"] = test_paper_m3_artifacts["retrieval_priors"]
            if retrieval_context_is_static or retrieval_count_decoder_enabled:
                test_static_retrieval_context = test_paper_m3_artifacts["retrieval_context"]
        test_batches=create_batches_dict(
            sampled_test_data,
            infer_batch_size,
            voc_size,
            device,
            start_visit=target_start_visit,
            **test_batch_kwargs,
        )
        test_batches["split_name"] = "test"
        if test_static_retrieval_context is not None:
            set_retrieval_context_batches(test_batches, test_static_retrieval_context, infer_batch_size)
        if insert_is_learned:
            target_clinical_payload = None
            if args.insert_visit_mode in ("learned_clinical", "dxproc_clinical_learned"):
                target_clinical_payload = (
                    [item["clinical_features"] for item in sampled_payload],
                    [item["clinical_feature_masks"] for item in sampled_payload],
                    [item["clinical_feature_abnormal"] for item in sampled_payload],
                    [item["clinical_feature_delta_hours"] for item in sampled_payload],
                )
                insert_learned_target_inputs["test"] = build_visit_retrieval_inputs(
                    sampled_test_data,
                    voc_size,
                    target_clinical_payload,
                )
            elif args.insert_visit_mode.startswith("dxproc_"):
                insert_learned_target_inputs["test"] = build_visit_retrieval_inputs(
                    sampled_test_data,
                    voc_size,
                    None,
                )
            test_metadata = build_visit_metadata(
                sampled_test_data,
                voc_size,
                [item["global_patient_idx"] for item in sampled_payload],
            )
            test_inserted = build_learned_inserted_visits(
                model=model,
                target_batches=test_batches,
                candidate_batches=insert_base_train_batches,
                target_metadata=test_metadata,
                candidate_metadata=retrieval_train_metadata,
                target_total=count_prediction_visits(sampled_test_data),
                candidate_total=count_prediction_visits(data_train),
                candidate_visits=insert_candidate_visits,
                candidate_clinical=insert_candidate_clinical,
            )
            test_batch_kwargs["inserted_visits"] = test_inserted
            test_batches=create_batches_dict(
                sampled_test_data,
                infer_batch_size,
                voc_size,
                device,
                start_visit=target_start_visit,
                **test_batch_kwargs,
            )
            test_batches["split_name"] = "test"
            if test_static_retrieval_context is not None:
                set_retrieval_context_batches(test_batches, test_static_retrieval_context, infer_batch_size)
        if retrieval_is_learned:
            target_clinical_payload = None
            if args.retrieval_mode == "learned_clinical":
                target_clinical_payload = (
                    [item["clinical_features"] for item in sampled_payload],
                    [item["clinical_feature_masks"] for item in sampled_payload],
                    [item["clinical_feature_abnormal"] for item in sampled_payload],
                    [item["clinical_feature_delta_hours"] for item in sampled_payload],
                )
                retrieval_learned_target_inputs["test"] = build_visit_retrieval_inputs(
                    sampled_test_data,
                    voc_size,
                    target_clinical_payload,
                )
            test_metadata = build_visit_metadata(
                sampled_test_data,
                voc_size,
                [item["global_patient_idx"] for item in sampled_payload],
            )
            if args.retrieval_weight > 0 or args.retrieval_copy_mode == "fixed_prob":
                test_priors = build_learned_retrieval_priors(
                    model=model,
                    target_batches=test_batches,
                    candidate_batches=train_batches,
                    target_metadata=test_metadata,
                    candidate_metadata=retrieval_train_metadata,
                    target_total=count_prediction_visits(sampled_test_data),
                    candidate_total=count_prediction_visits(data_train),
                )
                test_batches["retrieval_prior_batches"] = tensors_to_fixed_batches(
                    [test_priors[i] for batch in test_batches["sample_index_batches"] for i in batch],
                    infer_batch_size,
                )
            if retrieval_uses_context:
                test_context = build_learned_retrieval_context(
                    model=model,
                    target_batches=test_batches,
                    candidate_batches=train_batches,
                    target_metadata=test_metadata,
                    candidate_metadata=retrieval_train_metadata,
                    target_total=count_prediction_visits(sampled_test_data),
                    candidate_total=count_prediction_visits(data_train),
                )
                set_retrieval_context_batches(test_batches, test_context, infer_batch_size)
        if retrieval_is_treatment_episode and args.treatment_episode_action_weight > 0:
            target_clinical_payload = None
            if treatment_episode_uses_clinical:
                target_clinical_payload = (
                    [item["clinical_features"] for item in sampled_payload],
                    [item["clinical_feature_masks"] for item in sampled_payload],
                    [item["clinical_feature_abnormal"] for item in sampled_payload],
                    [item["clinical_feature_delta_hours"] for item in sampled_payload],
                )
            target_clinical_state_payload = (
                (
                    [item["clinical_state_labels"] for item in sampled_payload],
                    [item["clinical_state_masks"] for item in sampled_payload],
                )
                if retrieval_test_clinical_state_payload is not None
                else None
            )
            test_candidate_data = nonlearned_test_candidate_data(sampled_test_data)
            test_candidate_patient_id_offsets = nonlearned_test_candidate_patient_id_offsets(sampled_payload)
            test_candidate_clinical_payload = (
                nonlearned_test_candidate_clinical_payload(sampled_payload)
                if treatment_episode_uses_clinical
                else None
            )
            test_candidate_clinical_state_payload = nonlearned_test_candidate_clinical_state_payload(sampled_payload)
            refresh_treatment_episode_static_context(
                model=model,
                batches=test_batches,
                target_data=sampled_test_data,
                candidate_data=test_candidate_data,
                target_clinical_payload=target_clinical_payload,
                candidate_clinical_payload=test_candidate_clinical_payload,
                target_clinical_state_payload=target_clinical_state_payload,
                candidate_clinical_state_payload=test_candidate_clinical_state_payload,
                target_patient_id_offsets=[item["global_patient_idx"] for item in sampled_payload],
                candidate_patient_id_offsets=test_candidate_patient_id_offsets,
                batch_size=infer_batch_size,
                target_start_visit=target_start_visit,
                candidate_start_visit=retrieval_candidate_start_visit(),
            )
        set_sequence_prediction_batches(model, test_batches, [decode_config])
        y_gt, y_pred_prob, loss, med_count_pred = infer_prob(model, test_batches)
        result = list(get_metrics_from_prob(test_batches, y_gt, y_pred_prob, loss, decode_config, med_count_pred=med_count_pred))
        print('iteration %d / 10' % (i+1))
        results.append(result)
    results=np.array(results)
    return results.mean(axis=0), results.std(axis=0)
#%%
# result=get_metrics_on_test_data(model)
#%%

def train(model, optimizer):
    log_file=create_log_file('train_seer')
    log(log_file, 'config: ' + str(args) + '\n')
    log(log_file, "traing...")
    log(log_file, time.strftime("Date: %Y%m%d-%H%M%S"))
    log(log_file, "File: {}".format(__file__) + '\n')
    log(log_file, 'params: %d' % count_parameters(model))
    log(log_file, 'epoch\tja\tprauc\tavg_p\tavg_r\tavg_f1\tddi\tavg_med\tdecoder\tt_loss\tv_loss\tstate_aux_loss\ttransition_aux_loss\tcount_aux_loss\tretrieval_query_loss\tgraph_rank_loss\tgraph_boundary_rank_loss\taction_boundary_rank_loss\tgraph_count_loss\tgraph_score_distill_loss\taction_transition_loss\taction_ddi_add_loss\ttreatment_episode_action_loss\tcontinue_f1\tadd_f1\tremove_f1\ttransition_macro_f1\tmix_w_mean\tmix_w_min\tmix_w_max\to2_scale\to2_bias\taction_fusion_loss')

    start_time=time.time()
    continuous_decline=0
    prev_selection_metric = 0.0
    best=[0]*100
    best_decode_config=decode_configs()[0]
    best_model_params=model.state_dict()
    last_model_params=copy.deepcopy(model.state_dict())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    for epoch in range(args.epoches):
        if retrieval_is_learned and args.retrieval_refresh == "epoch":
            refresh_learned_retrieval_batches(epoch, include_train=True, include_eval=False)
        elif retrieval_is_learned and args.retrieval_refresh == "static" and epoch == args.retrieval_warmup_epochs:
            refresh_learned_retrieval_batches(epoch, include_train=True, include_eval=False)
        if retrieval_is_treatment_episode and args.treatment_episode_action_weight > 0:
            refresh_treatment_episode_batches(model, include_train=True, include_eval=False)
        train_loss_sum=0
        train_state_aux_loss_sum=0
        train_transition_aux_loss_sum=0
        train_count_aux_loss_sum=0
        train_retrieval_query_loss_sum=0
        train_graph_rank_loss_sum=0
        train_graph_boundary_rank_loss_sum=0
        train_action_boundary_rank_loss_sum=0
        train_graph_count_loss_sum=0
        train_graph_score_distill_loss_sum=0
        train_action_transition_loss_sum=0
        train_action_ddi_add_loss_sum=0
        train_action_fusion_loss_sum=0
        train_teacher_rank_loss_sum=0
        train_teacher_count_loss_sum=0
        train_teacher_parameter_loss_sum=0
        train_treatment_episode_action_loss_sum=0
        n_samples=0
        model.train()
        for i in range(train_batches['n_batches']):
            blen=train_batches['diag_batches'][i].shape[0]
            n_samples+=blen
            output=forward_model(model, train_batches, i)
            med_logits = apply_retrieval_prior(output["med_logits"], train_batches, i)
            state_logits = output["clinical_state_logits"]
            transition_logits = output["med_transition_logits"]
            med_count_pred = output["med_count_pred"]
            seq_logits = output["seq_logits"]
            state_aux_loss = med_logits.new_tensor(0.0)
            transition_aux_loss = med_logits.new_tensor(0.0)
            med_count_loss = med_logits.new_tensor(0.0)
            seq_loss = med_logits.new_tensor(0.0)
            if args.multitask_mode == "clinical_state":
                state_targets = train_batches["clinical_state_batches"][i]
                state_masks = train_batches["clinical_state_mask_batches"][i]
                transition_targets = train_batches["med_transition_batches"][i]
                transition_masks = train_batches["med_transition_mask_batches"][i]
                state_aux_loss = masked_multilabel_bce(state_logits, state_targets, state_masks)
                transition_aux_loss = masked_transition_cross_entropy(transition_logits, transition_targets, transition_masks)
            if args.med_count_mode == "aux":
                med_count_targets = med_count_targets_for_logits(med_logits, train_batches['label_batches'][i], train_batches, i)
                med_count_loss = F.smooth_l1_loss(med_count_pred, med_count_targets)
            if seq_decoder_enabled():
                seq_loss = seq_decoder_loss(seq_logits, train_batches['label_batches'][i])
            retrieval_query_loss = retrieval_query_rank_loss_for_batch(output, train_batches, i)
            graph_rank_loss = train_graph_rank_loss(med_logits, train_batches['label_batches'][i], train_batches, i)
            graph_boundary_rank_loss = train_graph_boundary_rank_loss(med_logits, train_batches['label_batches'][i], train_batches, i)
            action_boundary_rank_loss_value = action_boundary_rank_loss(med_logits, train_batches['label_batches'][i], train_batches, i)
            graph_count_loss = train_graph_count_loss(med_logits, train_batches['label_batches'][i], train_batches, i)
            graph_score_distill_loss = train_graph_score_distill_loss(med_logits, train_batches, i)
            action_transition_loss = action_transition_aux_loss(output, train_batches['label_batches'][i], train_batches, i)
            action_ddi_add_loss = action_ddi_add_aux_loss(output, train_batches, i)
            count_aux_loss = action_count_aux_loss(output, train_batches['label_batches'][i], train_batches, i)
            action_fusion_loss = action_fusion_aux_loss(output, train_batches['label_batches'][i], med_logits, train_batches, i)
            treatment_episode_action_loss = treatment_episode_action_aux_loss(output, train_batches['label_batches'][i], train_batches, i)
            main_med_scorer_reg_loss = main_med_scorer_regularization(output, med_logits)
            teacher_rank_loss, teacher_count_loss, teacher_parameter_loss = teacher_anchor_losses(
                med_logits,
                train_batches,
                i,
            )
            prefix_aux_loss = med_logits.new_tensor(0.0)
            prefix_seq_loss = med_logits.new_tensor(0.0)
            if args.prefix_aux_weight > 0 or args.prefix_seq_decoder_weight > 0:
                prefix_diags, prefix_procs, prefix_meds, prefix_labels, _, prefix_clinical_parts = build_prefix_samples_from_batches(
                    train_batches,
                    i,
                    data_train,
                    include_target=False,
                )
                if prefix_labels and args.prefix_aux_weight > 0:
                    prefix_logits, prefix_targets = forward_plain_prefix_batch(
                        model,
                        prefix_diags,
                        prefix_procs,
                        prefix_meds,
                        prefix_labels,
                        args.batch,
                        clinical_parts=prefix_clinical_parts,
                    )
                    prefix_aux_loss = loss_func(prefix_logits, prefix_targets)
                if prefix_labels and args.prefix_seq_decoder_weight > 0:
                    prefix_seq_loss = seq_decoder_loss_from_parts(
                        model,
                        prefix_diags,
                        prefix_procs,
                        prefix_meds,
                        prefix_labels,
                        args.batch,
                        clinical_parts=prefix_clinical_parts,
                    )
            loss = (
                args.bce_weight * loss_func(med_logits, train_batches['label_batches'][i])
                + args.action_transition_weight * action_transition_loss
                + args.action_ddi_loss_weight * action_ddi_add_loss
                + args.count_aux_weight * count_aux_loss
                + action_fusion_loss
                + treatment_episode_action_loss
                + set_level_loss(med_logits, train_batches['label_batches'][i])
                + args.asymmetric_bce_weight * asymmetric_bce_loss(med_logits, train_batches['label_batches'][i])
                + args.pairwise_rank_weight * pairwise_rank_loss(med_logits, train_batches['label_batches'][i])
                + args.listwise_rank_weight * listwise_rank_loss(med_logits, train_batches['label_batches'][i])
                + args.train_graph_rank_weight * graph_rank_loss
                + args.train_graph_boundary_rank_weight * graph_boundary_rank_loss
                + args.action_boundary_rank_weight * action_boundary_rank_loss_value
                + args.train_graph_count_weight * graph_count_loss
                + args.train_graph_score_distill_weight * graph_score_distill_loss
                + args.retrieval_query_rank_weight * retrieval_query_loss
                + args.soft_ddi_weight * soft_ddi_loss(med_logits)
                + main_med_scorer_reg_loss
                + args.state_aux_weight * state_aux_loss
                + args.transition_aux_weight * transition_aux_loss
                + args.med_count_aux_weight * med_count_loss
                + args.seq_decoder_weight * seq_loss
                + args.prefix_aux_weight * prefix_aux_loss
                + args.prefix_seq_decoder_weight * prefix_seq_loss
                + args.teacher_rank_distill_weight * teacher_rank_loss
                + args.teacher_count_distill_weight * teacher_count_loss
                + args.teacher_parameter_anchor_weight * teacher_parameter_loss
            )
            train_loss_sum += loss.detach().item() * blen
            train_state_aux_loss_sum += state_aux_loss.detach().item() * blen
            train_transition_aux_loss_sum += transition_aux_loss.detach().item() * blen
            train_count_aux_loss_sum += (med_count_loss.detach().item() + count_aux_loss.detach().item()) * blen
            train_retrieval_query_loss_sum += retrieval_query_loss.detach().item() * blen
            train_graph_rank_loss_sum += graph_rank_loss.detach().item() * blen
            train_graph_boundary_rank_loss_sum += graph_boundary_rank_loss.detach().item() * blen
            train_action_boundary_rank_loss_sum += action_boundary_rank_loss_value.detach().item() * blen
            train_graph_count_loss_sum += graph_count_loss.detach().item() * blen
            train_graph_score_distill_loss_sum += graph_score_distill_loss.detach().item() * blen
            train_action_transition_loss_sum += action_transition_loss.detach().item() * blen
            train_action_ddi_add_loss_sum += action_ddi_add_loss.detach().item() * blen
            train_action_fusion_loss_sum += action_fusion_loss.detach().item() * blen
            train_treatment_episode_action_loss_sum += treatment_episode_action_loss.detach().item() * blen
            train_teacher_rank_loss_sum += teacher_rank_loss.detach().item() * blen
            train_teacher_count_loss_sum += teacher_count_loss.detach().item() * blen
            train_teacher_parameter_loss_sum += teacher_parameter_loss.detach().item() * blen
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('train epoch progress: {}/{}\r'.format(i+1, train_batches['n_batches']), end='', flush=True)
        last_model_params=copy.deepcopy(model.state_dict())
        print('validating...' + ' ' * 100 + '\r',end='', flush=True)
        if retrieval_is_learned and args.retrieval_refresh == "epoch":
            refresh_learned_retrieval_batches(epoch, include_train=False, include_eval=True)
        elif retrieval_is_learned and args.retrieval_refresh == "static" and epoch == args.retrieval_warmup_epochs:
            refresh_learned_retrieval_batches(epoch, include_train=False, include_eval=True)
        # check performance
        validation_metrics, selected_decode_config = select_validation_metrics(model, log_file=log_file, epoch=epoch)
        validation_action_metrics = action_head_metrics_for_batches(model, eval_batches)
        ja,prauc,avg_precision,avg_recall,avg_f1,ddi,mean_med,v_loss=validation_metrics
        t_loss=train_loss_sum/n_samples
        state_aux_loss_mean=train_state_aux_loss_sum/n_samples
        transition_aux_loss_mean=train_transition_aux_loss_sum/n_samples
        count_aux_loss_mean=train_count_aux_loss_sum/n_samples
        retrieval_query_loss_mean=train_retrieval_query_loss_sum/n_samples
        graph_rank_loss_mean=train_graph_rank_loss_sum/n_samples
        graph_boundary_rank_loss_mean=train_graph_boundary_rank_loss_sum/n_samples
        action_boundary_rank_loss_mean=train_action_boundary_rank_loss_sum/n_samples
        graph_count_loss_mean=train_graph_count_loss_sum/n_samples
        graph_score_distill_loss_mean=train_graph_score_distill_loss_sum/n_samples
        action_transition_loss_mean=train_action_transition_loss_sum/n_samples
        action_ddi_add_loss_mean=train_action_ddi_add_loss_sum/n_samples
        action_fusion_loss_mean=train_action_fusion_loss_sum/n_samples
        treatment_episode_action_loss_mean=train_treatment_episode_action_loss_sum/n_samples
        teacher_rank_loss_mean=train_teacher_rank_loss_sum/n_samples
        teacher_count_loss_mean=train_teacher_count_loss_sum/n_samples
        teacher_parameter_loss_mean=train_teacher_parameter_loss_sum/n_samples
        mix_stats = model.get_output_mix_logging_stats()
        metric=(
            epoch+1,
            ja,
            prauc,
            avg_precision,
            avg_recall,
            avg_f1,
            ddi,
            mean_med,
            format_decode_config(selected_decode_config),
            t_loss,
            v_loss,
            state_aux_loss_mean,
            transition_aux_loss_mean,
            count_aux_loss_mean,
            retrieval_query_loss_mean,
            graph_rank_loss_mean,
            graph_boundary_rank_loss_mean,
            action_boundary_rank_loss_mean,
            graph_count_loss_mean,
            graph_score_distill_loss_mean,
            action_transition_loss_mean,
            action_ddi_add_loss_mean,
            treatment_episode_action_loss_mean,
            validation_action_metrics["continue_f1"],
            validation_action_metrics["add_f1"],
            validation_action_metrics["remove_f1"],
            validation_action_metrics["transition_macro_f1"],
            mix_stats['mean'],
            mix_stats['min'],
            mix_stats['max'],
            mix_stats['o2_scale'],
            mix_stats['o2_bias'],
            action_fusion_loss_mean,
        )
        print(' ' * 100 + '\r',end='', flush=True)
        log(
            log_file,
            '{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(*metric)
        )
        log(
            log_file,
            'teacher_anchor\tepoch={}\trank_loss={:.8f}\tcount_loss={:.8f}\tparameter_loss={:.10f}'.format(
                epoch + 1,
                teacher_rank_loss_mean,
                teacher_count_loss_mean,
                teacher_parameter_loss_mean,
            ),
        )
        if args.save_every_epoch_dir:
            epoch_checkpoint_dir = Path(args.save_every_epoch_dir)
            epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            epoch_checkpoint_path = epoch_checkpoint_dir / f"epoch_{epoch + 1:02d}.pth"
            temporary_checkpoint_path = epoch_checkpoint_path.with_suffix(".pth.tmp")
            torch.save(model.state_dict(), temporary_checkpoint_path)
            temporary_checkpoint_path.replace(epoch_checkpoint_path)
        # save the best model
        current_selection_metric = metric_value(metric)
        if current_selection_metric > metric_value(best):
            best = metric
            best_decode_config = selected_decode_config
            best_model_params = copy.deepcopy(model.state_dict())
        # early stop
        if args.early_stop_patience > 0 and current_selection_metric < prev_selection_metric:
            continuous_decline += 1
            if continuous_decline >= args.early_stop_patience:
                log(log_file, ' [ early stop ] ')
                break
        else:
            continuous_decline = 0
        prev_selection_metric = current_selection_metric
    log(log_file, 'best:')
    log(
        log_file,
        '{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(*best)
    )
    log(log_file, "training stopped.")
    log(log_file, "Time used: %.2f" % (time.time()-start_time))
    log_file.close()
    # end training
    return best_model_params, best_decode_config, last_model_params


# %%
if args.eval_only:
    if not args.eval_model_checkpoint:
        raise ValueError("--eval_only requires --eval_model_checkpoint")
    if args.fixed_component_export_no_metric_selection and not args.component_output:
        raise ValueError("--fixed_component_export_no_metric_selection requires --component_output")

    eval_state_dict = load_state_dict_file(args.eval_model_checkpoint, device)
    require_action_head_checkpoint(eval_state_dict, args.eval_model_checkpoint)
    model.load_state_dict(eval_state_dict, strict=False)

    if args.fixed_component_export_no_metric_selection:
        validation_metrics = None
        best_decode_config = decode_configs()[0]
        print("validation metric selection skipped for fixed component export")
    else:
        validation_metrics, best_decode_config = select_validation_metrics(model)
        print("validation", validation_metrics, format_decode_config(best_decode_config))

    if args.prediction_output:
        write_prediction_output(model, best_decode_config, args.prediction_output)
        print("prediction_output", args.prediction_output)

    if args.component_output:
        write_component_output(model, best_decode_config, args.component_output)
        print("component_output", args.component_output)

    if args.test > 0:
        result = get_metrics_on_test_data(model, decode_config=best_decode_config)
        print(result)
else:
    best_model_params, best_decode_config, last_model_params = train(model, optimizer)
    if args.save_model > 0:
        save_model_path = Path(args.save_model_path)
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        params_to_save = last_model_params if args.save_model_selection == "last" else best_model_params
        torch.save(params_to_save, save_model_path)
    if args.test > 0:
        model.load_state_dict(best_model_params)
        result = get_metrics_on_test_data(model, decode_config=best_decode_config)
        print(result)
