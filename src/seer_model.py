#%%
import torch
from torch import nn
from seer_base import SimpleNetV
import math
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union
from action_model import ActionAwareMedicationTransitionBackbone, RoutedActionFusionLayer, RoutedResidualActionFusionLayer, SimpleFFNActionFusionLayer, TrainableActionFusionLayer

#%%
@dataclass
class MambaModelArgs:
    d_model: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class MambaBlock(nn.Module):
    def __init__(self, args: MambaModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

#%%
# test
if __name__ == '__main__':
    b,l,d = 2,3,4
    x = torch.randn(b,l,d)
    m= MambaBlock(MambaModelArgs(d_model=d))
    y = m(x)
#%%

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        device = ehr_adj.device
        ehr_adj = ehr_adj.to(device=device, dtype=torch.float32)
        ddi_adj = ddi_adj.to(device=device, dtype=torch.float32)

        self.ehr_adj = self.normalize(ehr_adj + torch.eye(ehr_adj.shape[0], device=device, dtype=ehr_adj.dtype))
        self.ddi_adj = self.normalize(ddi_adj + torch.eye(ddi_adj.shape[0], device=device, dtype=ddi_adj.dtype))
        self.x = torch.eye(voc_size, device=device, dtype=torch.float32)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ehr_node_embedding = F.relu(ehr_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)

        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    # def normalize(self, mx):
    #     """Row-normalize sparse matrix"""
    #     rowsum = np.array(mx.sum(1))
    #     r_inv = np.power(rowsum, -1).flatten()
    #     r_inv[np.isinf(r_inv)] = 0.
    #     r_mat_inv = np.diagflat(r_inv)
    #     mx = r_mat_inv.dot(mx)
    #     return mx

    # gpu version
    def normalize(self,mx):
        """Row-normalize sparse matrix"""
        mx = mx.to_dense()
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx
    
#%%
# test
if __name__ == '__main__':
    ehr_adj=(torch.rand((131,131))>0.7).float()
    ddi_adj=(torch.rand((131,131))>0.7).float()
    gcn=GCN(131, 64, ehr_adj, ddi_adj)
    ehr_node_embedding, ddi_node_embedding=gcn()

# %%
class SequenceSegmenter(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, seq):
        batch, seq_len, emb_dim = seq.shape
        if seq_len < self.k:
            pad_len = self.k - seq_len
            padding = torch.zeros(batch, pad_len, emb_dim, device=seq.device, dtype=seq.dtype)
            seq = torch.cat((seq, padding), dim=1)
        near_seq = seq[:, :self.k, :]
        far_seq = seq[:, self.k:, :]
        return near_seq, far_seq


class NearTemporalProjector(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, near_seq, norm, lin):
        batch, _, emb_dim = near_seq.shape
        projected = lin(norm(near_seq.view(batch, -1))).view(batch, self.k, emb_dim)
        return projected + near_seq


class FarTemporalEncoder(nn.Module):
    def forward(self, far_seq, sequence_model):
        if far_seq.size(1) == 0:
            return far_seq
        return sequence_model(torch.flip(far_seq, [1]))


class NearFarAggregator(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, near_h, far_seq):
        far_h = near_h.new_zeros(near_h.size(0), self.k, near_h.size(2))
        if far_seq.size(1) == 0:
            return far_h
        scores = torch.bmm(near_h, far_seq.transpose(-2, -1))
        attn_scores = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(attn_scores, far_seq)


class GRUSequenceModel(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True)

    def forward(self, seq):
        output, _ = self.gru(seq)
        return output


class TransformerSequenceModel(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        nhead = self._infer_nhead(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=emb_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def _infer_nhead(self, emb_dim):
        for candidate in (8, 4, 2):
            if emb_dim % candidate == 0:
                return candidate
        return 1

    def _positional_encoding(self, seq):
        seq_len = seq.size(1)
        position = torch.arange(seq_len, device=seq.device, dtype=seq.dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.emb_dim, 2, device=seq.device, dtype=seq.dtype)
            * (-math.log(10000.0) / self.emb_dim)
        )
        pe = seq.new_zeros(seq_len, self.emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)

    def forward(self, seq):
        return self.encoder(seq + self._positional_encoding(seq))


def build_temporal_backbone(temporal_type, emb_dim):
    if temporal_type == 'mamba':
        return MambaBlock(MambaModelArgs(d_model=emb_dim))
    if temporal_type == 'gru':
        return GRUSequenceModel(emb_dim)
    if temporal_type == 'transformer':
        return TransformerSequenceModel(emb_dim)
    raise ValueError(f'Unsupported temporal_type: {temporal_type}')


class TemporalEncoder(nn.Module):
    def __init__(self, emb_dim, k, temporal_type='mamba'):
        super().__init__()
        self.k = k
        self.temporal_type = temporal_type
        self.segmenter = SequenceSegmenter(k)
        self.near_projector = NearTemporalProjector(k)
        self.far_encoder = FarTemporalEncoder()
        self.aggregator = NearFarAggregator(k)
        self.sequence_model = build_temporal_backbone(temporal_type, emb_dim)
        self.lin = torch.nn.Linear(k * emb_dim, k * emb_dim)
        self.norm = torch.nn.LayerNorm(k * emb_dim)
        if temporal_type == 'mamba':
            self.mamba = self.sequence_model

    # seq: [batch, seq_len, emb_dim]
    def forward(self, seq):
        near_seq, far_seq = self.segmenter(seq)
        near_h = self.near_projector(near_seq, self.norm, self.lin)
        encoded_far = self.far_encoder(far_seq, self.sequence_model)
        far_h = self.aggregator(near_h, encoded_far)
        return {
            'near': near_h,
            'far': far_h,
            'combined': torch.concat((near_h, far_h), dim=1),
        }


class PiecewiseTSL(TemporalEncoder):
    # Kept as a compatibility wrapper while callers migrate to the structured output.
    def forward(self, seq):
        return super().forward(seq)['combined']

        # result=self.gru(seq)[0]
        # return torch.concat((result,result.clone().detach()),dim=1)
#%%
# q=torch.randn((32,3,128))
# kv=torch.randn((32,5,128))
# attn_scores = torch.bmm(q, kv.transpose(-2, -1))
# far_h=torch.bmm(attn_scores, kv)

# test
if __name__ == '__main__':
    model=PiecewiseTSL(128,3).to('cuda:0')
    seq=torch.randn(32, 2, 128, device='cuda:0')
    y=model(seq)
    seq=torch.randn(32, 5, 128, device='cuda:0')
    y=model(seq)

#%%
class PatientSignalFusion(nn.Module):
    def __init__(self, emb_dim, fusion_type='sum'):
        super().__init__()
        self.emb_dim = emb_dim
        self.fusion_type = fusion_type
        if fusion_type == 'gated_sum':
            self.gate_proj = nn.Linear(emb_dim * 3, emb_dim * 3)
        elif fusion_type != 'sum':
            raise ValueError(f'Unsupported patient fusion_type: {fusion_type}')

    def compute_gates(self, e_d, e_p, e_m):
        if self.fusion_type != 'gated_sum':
            raise ValueError('compute_gates is only available for gated_sum fusion.')
        gate_inputs = torch.cat((e_d, e_p, e_m), dim=-1)
        gate_values = torch.sigmoid(self.gate_proj(gate_inputs))
        return gate_values.chunk(3, dim=-1)

    def forward(self, e_d, e_p, e_m):
        if self.fusion_type == 'sum':
            return e_d + e_p + e_m
        gate_d, gate_p, gate_m = self.compute_gates(e_d, e_p, e_m)
        return gate_d * e_d + gate_p * e_p + gate_m * e_m


class ClinicalFeatureEncoder(nn.Module):
    def __init__(self, feature_dim, emb_dim, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.input_dim = feature_dim * 4
        self.norm = nn.LayerNorm(self.input_dim)
        self.fc1 = nn.Linear(self.input_dim, emb_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, features, masks, abnormal, delta_hours):
        bounded_delta = torch.log1p(torch.clamp(delta_hours, min=0.0, max=168.0)) / math.log1p(168.0)
        x = torch.cat((features, masks, abnormal, bounded_delta), dim=-1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


class ClinicalPatientSignalFusion(nn.Module):
    def __init__(self, emb_dim, fusion_type='clinical_gated_sum'):
        super().__init__()
        self.emb_dim = emb_dim
        self.fusion_type = fusion_type
        if fusion_type == 'clinical_gated_sum':
            self.gate_proj = nn.Linear(emb_dim * 4 + 1, emb_dim * 4)
        elif fusion_type == 'concat':
            self.out = nn.Linear(emb_dim * 4, emb_dim)
        elif fusion_type != 'sum':
            raise ValueError(f'Unsupported clinical_fusion_type: {fusion_type}')

    def forward(self, e_d, e_p, e_m, e_c, clinical_masks):
        if self.fusion_type == 'sum':
            return e_d + e_p + e_m + e_c
        if self.fusion_type == 'concat':
            return self.out(torch.cat((e_d, e_p, e_m, e_c), dim=-1))
        clinical_missing_rate = 1.0 - clinical_masks.mean(dim=-1, keepdim=True)
        gate_inputs = torch.cat((e_d, e_p, e_m, e_c, clinical_missing_rate), dim=-1)
        gate_d, gate_p, gate_m, gate_c = torch.sigmoid(self.gate_proj(gate_inputs)).chunk(4, dim=-1)
        return gate_d * e_d + gate_p * e_p + gate_m * e_m + gate_c * e_c


class PatientQueryProjector(nn.Module):
    def __init__(self, emb_dim, k, query_type='flatten'):
        super().__init__()
        self.emb_dim = emb_dim
        self.k = k
        self.query_type = query_type
        self.output_dim = 2 * k * emb_dim
        if query_type in ('mean_pool', 'attention_pool', 'learned_query'):
            self.expand = nn.Linear(emb_dim, self.output_dim)
        if query_type == 'attention_pool':
            self.attn_score = nn.Linear(emb_dim, 1)
        elif query_type == 'learned_query':
            self.query_token = nn.Parameter(torch.empty(emb_dim))
            nn.init.normal_(self.query_token, mean=0.0, std=0.02)
        elif query_type not in ('flatten', 'mean_pool'):
            raise ValueError(f'Unsupported patient query_type: {query_type}')

    def compute_pooling_weights(self, patient_query):
        batch, seq_len, _ = patient_query.shape
        if self.query_type == 'mean_pool':
            return patient_query.new_full((batch, seq_len), 1.0 / seq_len)
        if self.query_type == 'attention_pool':
            logits = self.attn_score(patient_query).squeeze(-1)
            return torch.softmax(logits, dim=1)
        if self.query_type == 'learned_query':
            logits = torch.matmul(patient_query, self.query_token) / math.sqrt(self.emb_dim)
            return torch.softmax(logits, dim=1)
        raise ValueError('compute_pooling_weights is only available for pooled patient query strategies.')

    def forward(self, patient_query):
        if self.query_type == 'flatten':
            return patient_query.reshape(patient_query.shape[0], -1)
        weights = self.compute_pooling_weights(patient_query)
        pooled = torch.bmm(weights.unsqueeze(1), patient_query).squeeze(1)
        return self.expand(pooled)


class ClinicalStateHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


class PatientStateProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.fc(self.norm(x)))


class MedicationTransitionHead(nn.Module):
    def __init__(self, input_dim, output_dim=3, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


class MedicationCountHead(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return F.softplus(self.fc2(x)).squeeze(-1)


class CurrentCodeHistoryMedAttention(nn.Module):
    def __init__(self, emb_dim, dropout=0.1, gate_init=-3.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.query_norm = nn.LayerNorm(emb_dim)
        self.history_norm = nn.LayerNorm(emb_dim)
        self.query_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.out_norm = nn.LayerNorm(emb_dim)
        self.out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        self.gate = nn.Linear(emb_dim * 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init)

    def forward(self, e_d, e_p, e_m, meds, e_c=None, clinical_masks=None):
        batch = e_d.size(0)
        current_code = e_d[:, 0, :] + e_p[:, 0, :]
        if e_c is not None:
            current_clinical = e_c[:, 0, :]
            if clinical_masks is not None:
                current_clinical_mask = (clinical_masks[:, 0, :] > 0).any(dim=-1, keepdim=True)
                current_clinical = current_clinical * current_clinical_mask.to(dtype=current_clinical.dtype)
            current_code = current_code + current_clinical

        history_meds = e_m[:, 1:, :]
        history_mask = meds[:, 1:, :].sum(dim=-1) > 0
        if history_meds.size(1) == 0:
            zero = e_d.new_zeros(batch, self.emb_dim)
            return {
                'delta': zero,
                'gate': zero[:, :1],
                'attention': e_d.new_zeros(batch, 0),
                'valid_history': history_mask,
            }

        query = self.query_proj(self.query_norm(current_code)).unsqueeze(1)
        keys = self.key_proj(self.history_norm(history_meds))
        values = self.value_proj(history_meds)
        scores = (query * keys).sum(dim=-1) / math.sqrt(self.emb_dim)
        scores = scores.masked_fill(~history_mask, -1e4)
        attention = torch.softmax(scores, dim=1)
        attention = attention * history_mask.to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
        context = torch.bmm(attention.unsqueeze(1), values).squeeze(1)
        delta = self.out(self.out_norm(context))
        gate = torch.sigmoid(self.gate(torch.cat((current_code, context), dim=-1)))
        has_history = history_mask.any(dim=1, keepdim=True).to(dtype=delta.dtype)
        return {
            'delta': delta * has_history,
            'gate': gate * has_history,
            'attention': attention,
            'valid_history': history_mask,
        }


class PatientRepLearn(nn.Module):
    def __init__(
        self,
        emb_dim,
        k,
        d_voc_size,
        p_voc_size,
        m_voc_size,
        temporal_type='mamba',
        fusion_type='sum',
        clinical_feature_dim=0,
        clinical_feature_mode='none',
        clinical_fusion_type='clinical_gated_sum',
        clinical_feature_dropout=0.1,
        clinical_temporal_fusion='add',
        clinical_temporal_gate_init=-3.0,
        current_med_attention_mode='none',
        current_med_attention_dropout=0.1,
        current_med_attention_gate_init=-3.0,
        disable_procedure_branch=False,
    ):
        super().__init__()
        self.emb_dim=emb_dim
        self.clinical_feature_mode = clinical_feature_mode
        self.clinical_temporal_fusion = clinical_temporal_fusion
        self.current_med_attention_mode = current_med_attention_mode
        self.disable_procedure_branch = bool(disable_procedure_branch)
        self.tsl_d=TemporalEncoder(emb_dim,k,temporal_type=temporal_type)
        self.tsl_p=TemporalEncoder(emb_dim,k,temporal_type=temporal_type)
        self.tsl_c = None
        self.clinical_temporal_gate = None
        self.d_lin=nn.Linear(d_voc_size,emb_dim)
        self.p_lin=nn.Linear(p_voc_size,emb_dim)
        self.m_lin=nn.Linear(m_voc_size,emb_dim)
        self.signal_fusion=PatientSignalFusion(emb_dim,fusion_type=fusion_type)
        self.clinical_encoder = None
        self.clinical_signal_fusion = None
        if clinical_feature_mode == 'input':
            if clinical_temporal_fusion not in ('add', 'gated_residual'):
                raise ValueError(f'Unsupported clinical_temporal_fusion: {clinical_temporal_fusion}')
            if clinical_feature_dim <= 0:
                raise ValueError("clinical_feature_dim must be positive when clinical_feature_mode='input'")
            self.clinical_encoder = ClinicalFeatureEncoder(
                feature_dim=clinical_feature_dim,
                emb_dim=emb_dim,
                dropout=clinical_feature_dropout,
            )
            self.clinical_signal_fusion = ClinicalPatientSignalFusion(
                emb_dim=emb_dim,
                fusion_type=clinical_fusion_type,
            )
            self.tsl_c = TemporalEncoder(emb_dim,k,temporal_type=temporal_type)
            if clinical_temporal_fusion == 'gated_residual':
                self.clinical_temporal_gate = nn.Linear(emb_dim * 2, 1)
                nn.init.zeros_(self.clinical_temporal_gate.weight)
                nn.init.constant_(self.clinical_temporal_gate.bias, clinical_temporal_gate_init)
        elif clinical_feature_mode != 'none':
            raise ValueError(f'Unsupported clinical_feature_mode: {clinical_feature_mode}')
        self.current_med_attention = None
        if current_med_attention_mode == 'history_cross_attention':
            self.current_med_attention = CurrentCodeHistoryMedAttention(
                emb_dim=emb_dim,
                dropout=current_med_attention_dropout,
                gate_init=current_med_attention_gate_init,
            )
        elif current_med_attention_mode != 'none':
            raise ValueError(f'Unsupported current_med_attention_mode: {current_med_attention_mode}')
    # diags: [batch, seq_len, d_voc_size]
    # procs: [batch, seq_len, p_voc_size]
    def _align_visit_context(self, visit_context, target_len):
        if visit_context.size(1) < target_len:
            pad_len=target_len-visit_context.size(1)
            padding=torch.zeros(
                visit_context.shape[0],
                pad_len,
                visit_context.shape[2],
                device=visit_context.device,
                dtype=visit_context.dtype,
            )
            return torch.cat((visit_context,padding),dim=1)
        return visit_context[:,:target_len,:]

    def _current_code_context(self, diags, procs, clinical_context=None, clinical_masks=None):
        batch = diags.size(0)
        device = diags.device
        dtype = diags.dtype
        diag_embeddings = self.d_lin.weight.transpose(0, 1).to(device=device, dtype=dtype)
        proc_embeddings = self.p_lin.weight.transpose(0, 1).to(device=device, dtype=dtype)
        rows = []
        lengths = []
        current_diags = diags[:, 0, :]
        current_procs = procs[:, 0, :]
        for row_idx in range(batch):
            parts = []
            diag_idx = torch.nonzero(current_diags[row_idx] > 0, as_tuple=False).flatten()
            if diag_idx.numel() > 0:
                parts.append(diag_embeddings.index_select(0, diag_idx))
            proc_idx = torch.nonzero(current_procs[row_idx] > 0, as_tuple=False).flatten()
            if proc_idx.numel() > 0:
                parts.append(proc_embeddings.index_select(0, proc_idx))
            if clinical_context is not None:
                has_clinical = True
                if clinical_masks is not None:
                    has_clinical = bool((clinical_masks[row_idx, 0] > 0).any().item())
                if has_clinical:
                    parts.append(clinical_context[row_idx, 0:1].to(device=device, dtype=dtype))
            if parts:
                row = torch.cat(parts, dim=0)
            else:
                row = torch.zeros(0, self.emb_dim, device=device, dtype=dtype)
            rows.append(row)
            lengths.append(row.size(0))
        max_len = max(max(lengths), 1)
        code_context = torch.zeros(batch, max_len, self.emb_dim, device=device, dtype=dtype)
        code_mask = torch.ones(batch, max_len, device=device, dtype=torch.bool)
        for row_idx, row in enumerate(rows):
            if row.size(0) == 0:
                continue
            code_context[row_idx, :row.size(0)] = row
            code_mask[row_idx, :row.size(0)] = False
        return code_context, code_mask

    def forward(
        self,
        diags,
        procs,
        meds,
        clinical_features=None,
        clinical_feature_masks=None,
        clinical_feature_abnormal=None,
        clinical_feature_delta_hours=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
        own_diags=None,
        own_procs=None,
        own_meds=None,
        seq_decoder_inputs=None,
    ):
        # [batch, seq_len, emb_dim]
        e_d=self.d_lin(diags)
        e_p=(
            torch.zeros(
                procs.shape[0],
                procs.shape[1],
                self.emb_dim,
                device=procs.device,
                dtype=diags.dtype,
            )
            if self.disable_procedure_branch
            else self.p_lin(procs)
        )
        e_m=self.m_lin(meds)
        if self.clinical_encoder is not None and clinical_features is not None:
            e_c = self.clinical_encoder(
                clinical_features,
                clinical_feature_masks,
                clinical_feature_abnormal,
                clinical_feature_delta_hours,
            )
            e_h = self.clinical_signal_fusion(e_d, e_p, e_m, e_c, clinical_feature_masks)
        else:
            e_c = None
            e_h=self.signal_fusion(e_d,e_p,e_m)
        # [batch, 2k, emb_dim]
        h_d=self.tsl_d(e_d)['combined']
        h_p=(
            torch.zeros_like(h_d)
            if self.disable_procedure_branch
            else self.tsl_p(e_p)['combined']
        )
        if self.clinical_encoder is not None and clinical_features is not None:
            h_c=self.tsl_c(e_c)['combined']
            if self.clinical_temporal_gate is None:
                h_p=h_p+h_c
            else:
                temporal_gate = torch.sigmoid(self.clinical_temporal_gate(torch.cat((h_p, h_c), dim=-1)))
                h_p=h_p+temporal_gate*h_c
        visit_context=self._align_visit_context(e_h,h_d.size(1))
        patient_query=h_d+h_p
        current_med_attention_output = None
        if self.current_med_attention is not None:
            current_med_attention_output = self.current_med_attention(
                e_d,
                e_p,
                e_m,
                meds,
                e_c=e_c,
                clinical_masks=clinical_feature_masks,
            )
            current_med_delta = (
                current_med_attention_output['gate']
                * current_med_attention_output['delta']
            )
            visit_context = visit_context.clone()
            patient_query = patient_query.clone()
            visit_context[:, 0, :] = visit_context[:, 0, :] + current_med_delta
            patient_query[:, 0, :] = patient_query[:, 0, :] + current_med_delta
        code_context, code_context_mask = self._current_code_context(
            diags,
            procs,
            clinical_context=e_c,
            clinical_masks=clinical_feature_masks,
        )
        return {
            'visit_context': visit_context,
            'patient_query': patient_query,
            'code_context': code_context,
            'code_context_mask': code_context_mask,
            'current_med_attention_gate': current_med_attention_output['gate'] if current_med_attention_output is not None else None,
            'current_med_attention': current_med_attention_output['attention'] if current_med_attention_output is not None else None,
        }

#%%
# test
if __name__=='__main__':
    diags=torch.randn(32, 5, 10, device='cuda:0')
    procs=torch.randn(32, 5, 20, device='cuda:0')
    meds=torch.randn(32, 5, 24, device='cuda:0')
    model=PatientRepLearn(128,3,10,20,24).to('cuda:0')
    y=model(diags,procs,meds)
    print(y['visit_context'].shape, y['patient_query'].shape) # (32, 6, 128) (32, 6, 128)
#%%
class MedicationHistorySplitter(nn.Module):
    def __init__(self, m_voc_size, history_definition='input_order'):
        super().__init__()
        self.m_voc_size = m_voc_size
        self.history_definition = history_definition
        if history_definition not in ('input_order', 'real_time'):
            raise ValueError(f'Unsupported medication history_definition: {history_definition}')

    def _build_prefix_history_mask(self, meds):
        batch = meds.shape[0]
        return torch.concat(
            (
                torch.zeros(batch, 1, self.m_voc_size, device=meds.device, dtype=meds.dtype),
                (torch.cumsum(meds, dim=1) > 0).to(meds.dtype)[:, :-1, :],
            ),
            dim=1,
        )

    def forward(self, meds):
        if self.history_definition == 'input_order':
            # The training pipeline flips visit sequences to near-to-far order in seer_utils.py.
            # This splitter therefore preserves the current behavior: "old" means medications
            # seen in earlier positions of the model input, not strictly earlier in real time.
            history_mask = self._build_prefix_history_mask(meds)
        else:
            # The model input is near-to-far, so real chronology is recovered by flipping
            # the sequence to far-to-near, building prefix history there, then flipping back.
            history_mask = torch.flip(
                self._build_prefix_history_mask(torch.flip(meds, dims=[1])),
                dims=[1],
            )
        old_meds = meds * history_mask
        new_meds = meds - old_meds
        return {
            'history_mask': history_mask,
            'old_meds': old_meds,
            'new_meds': new_meds,
        }


class MedicationTemporalEncoder(nn.Module):
    def __init__(self, emb_dim, k, temporal_type='mamba'):
        super().__init__()
        self.tsl_old = TemporalEncoder(emb_dim, k, temporal_type=temporal_type)
        self.tsl_new = TemporalEncoder(emb_dim, k, temporal_type=temporal_type)

    def forward(self, old_meds, new_meds, med_embeddings):
        batch = old_meds.shape[0]
        old_sequence_embeddings = torch.matmul(old_meds, med_embeddings)
        new_sequence_embeddings = torch.matmul(new_meds, med_embeddings)
        old_state = self.tsl_old(old_sequence_embeddings)['combined'].reshape(batch, -1)
        new_state = self.tsl_new(new_sequence_embeddings)['combined'].reshape(batch, -1)
        return {
            'old_sequence_embeddings': old_sequence_embeddings,
            'new_sequence_embeddings': new_sequence_embeddings,
            'old_state': old_state,
            'new_state': new_state,
        }


class CandidateMedicationEncoder(nn.Module):
    def __init__(self, emb_dim, encoder_type='shared'):
        super().__init__()
        self.emb_dim = emb_dim
        self.encoder_type = encoder_type
        if encoder_type == 'split':
            self.old_encoder = nn.Linear(emb_dim, emb_dim)
            self.new_encoder = nn.Linear(emb_dim, emb_dim)
        elif encoder_type != 'shared':
            raise ValueError(f'Unsupported medication candidate encoder_type: {encoder_type}')

    def forward(self, meds, med_embeddings):
        batch = meds.shape[0]
        base_embeddings = med_embeddings.unsqueeze(0).repeat(batch, 1, 1)
        if self.encoder_type == 'shared':
            encoded_old_candidates = base_embeddings
            encoded_new_candidates = base_embeddings
        else:
            encoded_old_candidates = self.old_encoder(base_embeddings)
            encoded_new_candidates = self.new_encoder(base_embeddings)
        # This matches the current implementation: "seen" means the medication appears
        # anywhere in the current input window, regardless of real chronological direction.
        seen_mask = (meds.sum(dim=1) > 0).to(meds.dtype).unsqueeze(2).repeat(1, 1, self.emb_dim)
        old_candidate_embeddings = encoded_old_candidates * seen_mask
        new_candidate_embeddings = encoded_new_candidates * (1 - seen_mask)
        return {
            'base_candidate_embeddings': base_embeddings,
            'encoded_old_candidates': encoded_old_candidates,
            'encoded_new_candidates': encoded_new_candidates,
            'seen_mask': seen_mask,
            'old_candidate_embeddings': old_candidate_embeddings,
            'new_candidate_embeddings': new_candidate_embeddings,
        }


class MedicationScoreHead(nn.Module):
    def __init__(self, state_dim, scorer_type='cosine'):
        super().__init__()
        self.state_dim = state_dim
        self.scorer_type = scorer_type
        if scorer_type == 'bilinear':
            self.bilinear_weight = nn.Parameter(torch.empty(state_dim, state_dim))
            self.bilinear_bias = nn.Parameter(torch.zeros(1))
            nn.init.xavier_uniform_(self.bilinear_weight)
        elif scorer_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(state_dim * 2, state_dim),
                nn.GELU(),
                nn.Linear(state_dim, 1),
            )
        elif scorer_type not in ('cosine', 'dot'):
            raise ValueError(f'Unsupported medication scorer_type: {scorer_type}')

    def forward(self, state, candidate_states):
        if self.scorer_type == 'cosine':
            return torch.cosine_similarity(state.unsqueeze(1), candidate_states, dim=2)
        if self.scorer_type == 'dot':
            return torch.sum(state.unsqueeze(1) * candidate_states, dim=2)

        batch, num_candidates, _ = candidate_states.shape
        expanded_state = state.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, self.state_dim)
        flat_candidates = candidate_states.reshape(-1, self.state_dim)
        if self.scorer_type == 'bilinear':
            transformed_state = torch.matmul(expanded_state, self.bilinear_weight)
            scores = (transformed_state * flat_candidates).sum(dim=1, keepdim=True) + self.bilinear_bias
        else:
            scores = self.mlp(torch.cat((expanded_state, flat_candidates), dim=1))
        return scores.view(batch, num_candidates)


class MedicationScorer(nn.Module):
    def __init__(self, emb_dim, k, scorer_type='cosine'):
        super().__init__()
        self.state_dim = 2 * k * emb_dim
        self.scorer_type = scorer_type
        self.lin_expand = nn.Linear(emb_dim, self.state_dim)
        self.old_head = MedicationScoreHead(self.state_dim, scorer_type=scorer_type)
        self.new_head = MedicationScoreHead(self.state_dim, scorer_type=scorer_type)

    def forward(self, old_state, new_state, old_candidate_embeddings, new_candidate_embeddings, base_candidate_embeddings):
        projected_old_candidates = self.lin_expand(old_candidate_embeddings)
        projected_new_candidates = self.lin_expand(new_candidate_embeddings)
        old_scores = self.old_head(old_state, projected_old_candidates)
        new_scores = self.new_head(new_state, projected_new_candidates)
        total_scores = old_scores + new_scores
        candidate_embeddings = base_candidate_embeddings * total_scores.unsqueeze(2)
        return {
            'candidate_embeddings': candidate_embeddings,
            'final_embeddings': candidate_embeddings,
            'projected_old_candidates': projected_old_candidates,
            'projected_new_candidates': projected_new_candidates,
            'old_scores': old_scores,
            'new_scores': new_scores,
            'total_scores': total_scores,
        }


class MedRepLearn(nn.Module):
    def __init__(self, emb_dim, k, m_voc_size, temporal_type='mamba', history_definition='input_order', scorer_type='cosine', candidate_encoder_type='shared'):
        super().__init__()
        self.emb_dim=emb_dim
        self.m_voc_size=m_voc_size
        self.history_definition = history_definition
        self.scorer_type = scorer_type
        self.candidate_encoder_type = candidate_encoder_type
        self.history_splitter=MedicationHistorySplitter(m_voc_size,history_definition=history_definition)
        self.temporal_encoder=MedicationTemporalEncoder(emb_dim,k,temporal_type=temporal_type)
        self.candidate_encoder=CandidateMedicationEncoder(emb_dim,encoder_type=candidate_encoder_type)
        self.scorer=MedicationScorer(emb_dim,k,scorer_type=scorer_type)
        self.m_embs=nn.Embedding(m_voc_size,emb_dim)
        # Keep legacy attribute names available while callers migrate to the structured output.
        self.tsl_old=self.temporal_encoder.tsl_old
        self.tsl_new=self.temporal_encoder.tsl_new
        self.lin_expand=self.scorer.lin_expand
    # meds: [batch, seq_len, m_voc_size]
    def forward(self,meds):
        split_repr=self.history_splitter(meds)
        temporal_repr=self.temporal_encoder(
            split_repr['old_meds'],
            split_repr['new_meds'],
            self.m_embs.weight,
        )
        candidate_repr=self.candidate_encoder(meds,self.m_embs.weight)
        score_repr=self.scorer(
            temporal_repr['old_state'],
            temporal_repr['new_state'],
            candidate_repr['old_candidate_embeddings'],
            candidate_repr['new_candidate_embeddings'],
            candidate_repr['base_candidate_embeddings'],
        )
        return {
            **split_repr,
            **temporal_repr,
            **candidate_repr,
            **score_repr,
        }
#%%
# test
if __name__ == '__main__':
    emb_dim=32
    k=3
    m_voc_size=24
    batch=16
    seq_len=5
    model=MedRepLearn(emb_dim,k,m_voc_size).to('cuda:0')
    meds=(torch.randn((batch,seq_len,m_voc_size),device='cuda:0')>0.9).float()
    y=model(meds)
    print(y['final_embeddings'].shape) # (16, 24, 32)
#%%

class MedicationRepresentationFusion(nn.Module):
    def __init__(self, emb_dim, fusion_type='fixed_sum'):
        super().__init__()
        self.emb_dim = emb_dim
        self.fusion_type = fusion_type
        if fusion_type == 'gated_sum':
            self.gate_proj = nn.Linear(emb_dim * 3, emb_dim * 3)
        elif fusion_type != 'fixed_sum':
            raise ValueError(f'Unsupported medication fusion_type: {fusion_type}')

    def compute_gates(self, seq_embeddings, ehr_embeddings, ddi_embeddings):
        if self.fusion_type != 'gated_sum':
            raise ValueError('compute_gates is only available for gated_sum fusion.')
        gate_inputs = torch.cat((seq_embeddings, ehr_embeddings, ddi_embeddings), dim=-1)
        gate_values = torch.sigmoid(self.gate_proj(gate_inputs))
        return gate_values.chunk(3, dim=-1)

    def forward(self, seq_embeddings, ehr_embeddings, ddi_embeddings):
        batch = seq_embeddings.shape[0]
        if ehr_embeddings.dim() == 2:
            ehr_embeddings = ehr_embeddings.unsqueeze(0).repeat(batch, 1, 1)
        if ddi_embeddings.dim() == 2:
            ddi_embeddings = ddi_embeddings.unsqueeze(0).repeat(batch, 1, 1)
        if self.fusion_type == 'fixed_sum':
            return seq_embeddings + ehr_embeddings * 1 + ddi_embeddings * 0.5
        gate_seq, gate_ehr, gate_ddi = self.compute_gates(seq_embeddings, ehr_embeddings, ddi_embeddings)
        return gate_seq * seq_embeddings + gate_ehr * ehr_embeddings + gate_ddi * ddi_embeddings


class GraphAwareMedicationEncoder(nn.Module):
    def __init__(
        self,
        emb_dim,
        state_dim,
        ehr_adj,
        ddi_adj,
        mode='none',
        ehr_weight=0.0,
        ddi_weight=0.0,
        gate_hidden=128,
        dropout=0.1,
    ):
        super().__init__()
        if mode not in ('none', 'static', 'patient_gate'):
            raise ValueError(f'Unsupported graph_med_encoder_mode: {mode}')
        if gate_hidden <= 0:
            raise ValueError('graph_med_gate_hidden must be positive')
        self.mode = mode
        self.ehr_weight = float(ehr_weight)
        self.ddi_weight = float(ddi_weight)
        self.emb_norm = nn.LayerNorm(emb_dim)
        self.state_norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)
        self.ehr_msg_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.ddi_msg_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        nn.init.eye_(self.ehr_msg_proj.weight)
        nn.init.eye_(self.ddi_msg_proj.weight)
        self.gate_mlp = None
        if mode == 'patient_gate':
            self.gate_mlp = nn.Sequential(
                nn.Linear(state_dim + emb_dim, gate_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.zeros_(self.gate_mlp[-1].bias)
        self.register_buffer('ehr_adj', self._normalize_adj(ehr_adj))
        self.register_buffer('ddi_adj', self._normalize_adj(ddi_adj))

    @staticmethod
    def _normalize_adj(adj):
        adj = adj.to(dtype=torch.float32)
        if adj.is_sparse:
            adj = adj.to_dense()
        adj = adj.clone()
        adj.fill_diagonal_(0.0)
        rowsum = adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return adj / rowsum

    def _graph_delta(self, base_embeddings):
        base_for_message = self.emb_norm(base_embeddings)
        ehr_adj = self.ehr_adj.to(device=base_for_message.device, dtype=base_for_message.dtype)
        ddi_adj = self.ddi_adj.to(device=base_for_message.device, dtype=base_for_message.dtype)
        if base_for_message.dim() == 3:
            ehr_adj = ehr_adj.unsqueeze(0).expand(base_for_message.size(0), -1, -1)
            ddi_adj = ddi_adj.unsqueeze(0).expand(base_for_message.size(0), -1, -1)
        ehr_msg = torch.matmul(
            ehr_adj,
            base_for_message,
        )
        ddi_msg = torch.matmul(
            ddi_adj,
            base_for_message,
        )
        delta = (
            self.ehr_weight * self.ehr_msg_proj(ehr_msg)
            - self.ddi_weight * self.ddi_msg_proj(ddi_msg)
        )
        return delta

    def forward(self, base_embeddings, patient_state=None):
        if self.mode == 'none':
            return {
                'embeddings': base_embeddings,
                'delta': None,
                'gate': None,
            }
        delta = self._graph_delta(base_embeddings)
        if self.mode == 'static':
            gate = None
            final_embeddings = base_embeddings + self.dropout(delta)
        else:
            if patient_state is None:
                raise ValueError('patient_gate graph medication encoder requires patient_state')
            batch, med_count, _ = base_embeddings.shape
            state = self.state_norm(patient_state).unsqueeze(1).expand(-1, med_count, -1)
            gate_input = torch.cat((state, self.emb_norm(base_embeddings)), dim=-1)
            gate = torch.sigmoid(self.gate_mlp(gate_input))
            final_embeddings = base_embeddings + gate * self.dropout(delta)
        return {
            'embeddings': final_embeddings,
            'delta': delta,
            'gate': gate,
        }


class PatientConditionedGraphGate(nn.Module):
    def __init__(self, state_dim, m_voc_size, ehr_scale=0.25, ddi_scale=0.25, dropout=0.1):
        super().__init__()
        self.ehr_scale = ehr_scale
        self.ddi_scale = ddi_scale
        self.norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)
        self.ehr_proj = nn.Linear(state_dim, m_voc_size)
        self.ddi_proj = nn.Linear(state_dim, m_voc_size)
        nn.init.zeros_(self.ehr_proj.weight)
        nn.init.zeros_(self.ehr_proj.bias)
        nn.init.zeros_(self.ddi_proj.weight)
        nn.init.zeros_(self.ddi_proj.bias)

    def forward(self, patient_state, ehr_embeddings, ddi_embeddings):
        context = self.dropout(self.norm(patient_state))
        ehr_node_scale = 1.0 + self.ehr_scale * torch.tanh(self.ehr_proj(context))
        ddi_node_scale = 1.0 + self.ddi_scale * torch.tanh(self.ddi_proj(context))
        return {
            'ehr_embeddings': ehr_embeddings.unsqueeze(0) * ehr_node_scale.unsqueeze(-1),
            'ddi_embeddings': ddi_embeddings.unsqueeze(0) * ddi_node_scale.unsqueeze(-1),
            'ehr_scale': ehr_node_scale,
            'ddi_scale': ddi_node_scale,
        }


class PatientConditionedEdgeBias(nn.Module):
    def __init__(
        self,
        state_dim,
        ehr_adj,
        ddi_adj,
        ehr_scale=0.25,
        ddi_scale=0.25,
        ddi_weight=1.0,
        dropout=0.1,
        detach_prior=True,
    ):
        super().__init__()
        self.ehr_scale = ehr_scale
        self.ddi_scale = ddi_scale
        self.ddi_weight = ddi_weight
        self.detach_prior = detach_prior
        self.norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)
        self.ehr_src_proj = nn.Linear(state_dim, ehr_adj.shape[0])
        self.ehr_dst_proj = nn.Linear(state_dim, ehr_adj.shape[0])
        self.ddi_src_proj = nn.Linear(state_dim, ddi_adj.shape[0])
        self.ddi_dst_proj = nn.Linear(state_dim, ddi_adj.shape[0])
        for layer in (self.ehr_src_proj, self.ehr_dst_proj, self.ddi_src_proj, self.ddi_dst_proj):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.register_buffer("ehr_adj", self._normalize_adj(ehr_adj))
        self.register_buffer("ddi_adj", self._normalize_adj(ddi_adj))

    @staticmethod
    def _normalize_adj(adj):
        adj = adj.to(dtype=torch.float32)
        if adj.is_sparse:
            adj = adj.to_dense()
        rowsum = adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return adj / rowsum

    @staticmethod
    def _history_prior(meds):
        if meds is None or meds.shape[1] <= 1:
            return None
        return (meds[:, 1:, :].sum(dim=1) > 0).to(dtype=meds.dtype)

    @staticmethod
    def _retrieval_prior(retrieval_meds, retrieval_scores=None):
        if retrieval_meds is None:
            return None
        retrieval_meds = retrieval_meds.float()
        valid_visits = retrieval_meds.sum(dim=-1) > 0
        if retrieval_scores is None:
            visit_logits = retrieval_meds.new_zeros(valid_visits.shape)
        else:
            visit_logits = torch.nan_to_num(
                retrieval_scores.to(device=retrieval_meds.device, dtype=retrieval_meds.dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        visit_logits = visit_logits.masked_fill(~valid_visits, -1e4)
        visit_attn = torch.softmax(visit_logits, dim=-1)
        visit_attn = torch.where(valid_visits, visit_attn, torch.zeros_like(visit_attn))
        prior = torch.bmm(visit_attn.unsqueeze(1), retrieval_meds).squeeze(1)
        return prior.clamp(0.0, 1.0)

    def _select_prior(self, source, med_logits, meds, retrieval_meds=None, retrieval_scores=None):
        priors = []
        if source in ("model", "hybrid"):
            priors.append(torch.sigmoid(med_logits))
        if source in ("history", "hybrid"):
            history_prior = self._history_prior(meds)
            if history_prior is not None:
                priors.append(history_prior.to(device=med_logits.device, dtype=med_logits.dtype))
        if source in ("retrieval", "hybrid"):
            retrieval_prior = self._retrieval_prior(retrieval_meds, retrieval_scores=retrieval_scores)
            if retrieval_prior is not None:
                priors.append(retrieval_prior.to(device=med_logits.device, dtype=med_logits.dtype))
        if not priors:
            return torch.sigmoid(med_logits)
        prior = torch.stack(priors, dim=0).mean(dim=0).clamp(0.0, 1.0)
        return prior.detach() if self.detach_prior else prior

    def forward(self, patient_state, med_logits, meds, prior_source="history", retrieval_meds=None, retrieval_scores=None):
        prior = self._select_prior(
            prior_source,
            med_logits,
            meds,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
        )
        context = self.dropout(self.norm(patient_state))
        ehr_src = 1.0 + self.ehr_scale * torch.tanh(self.ehr_src_proj(context))
        ehr_dst = 1.0 + self.ehr_scale * torch.tanh(self.ehr_dst_proj(context))
        ddi_src = 1.0 + self.ddi_scale * torch.tanh(self.ddi_src_proj(context))
        ddi_dst = 1.0 + self.ddi_scale * torch.tanh(self.ddi_dst_proj(context))

        ehr_prior = prior * ehr_src
        ddi_prior = prior * ddi_src
        ehr_bias = torch.matmul(ehr_prior, self.ehr_adj.transpose(0, 1)) * ehr_dst
        ddi_bias = torch.matmul(ddi_prior, self.ddi_adj.transpose(0, 1)) * ddi_dst
        graph_bias = ehr_bias - self.ddi_weight * ddi_bias
        graph_bias = graph_bias - graph_bias.mean(dim=-1, keepdim=True)
        return {
            "bias": graph_bias,
            "prior": prior,
            "ehr_gate": ehr_src,
            "ddi_gate": ddi_src,
        }


class PatientConditionedMedicationScorer(nn.Module):
    def __init__(self, state_dim, rank=32, dropout=0.1):
        super().__init__()
        if rank <= 0:
            raise ValueError('patient-conditioned medication scorer rank must be positive')
        self.rank = rank
        self.patient_norm = nn.LayerNorm(state_dim)
        self.med_norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)
        self.patient_proj = nn.Linear(state_dim, rank)
        self.med_proj = nn.Linear(state_dim, rank, bias=False)
        nn.init.zeros_(self.patient_proj.weight)
        nn.init.zeros_(self.patient_proj.bias)
        nn.init.xavier_uniform_(self.med_proj.weight)

    def forward(self, patient_state, medication_states):
        patient_factor = self.patient_proj(self.dropout(self.patient_norm(patient_state)))
        med_factor = self.med_proj(self.med_norm(medication_states))
        bias = (patient_factor.unsqueeze(1) * med_factor).sum(dim=-1) / math.sqrt(self.rank)
        bias = bias - bias.mean(dim=-1, keepdim=True)
        return {
            'bias': bias,
            'patient_factor': patient_factor,
        }


class MainMedicationFeatureScorer(nn.Module):
    def __init__(self, state_dim, m_voc_size, hidden_dim=128, rank=32, dropout=0.1):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError('main medication scorer hidden_dim must be positive')
        if rank <= 0:
            raise ValueError('main medication scorer rank must be positive')
        self.state_norm = nn.LayerNorm(state_dim)
        self.query_norm = nn.LayerNorm(state_dim)
        self.med_norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)
        self.state_proj = nn.Linear(state_dim, rank)
        self.query_proj = nn.Linear(state_dim, rank)
        self.med_proj = nn.Linear(state_dim, rank, bias=False)
        self.med_bias = nn.Parameter(torch.zeros(m_voc_size))
        self.feature_mlp = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.state_proj.weight)
        nn.init.zeros_(self.state_proj.bias)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.xavier_uniform_(self.med_proj.weight)
        nn.init.zeros_(self.feature_mlp[-1].weight)
        nn.init.zeros_(self.feature_mlp[-1].bias)

    def forward(self, base_logits, direct_logits, graph_logits, direct_state, patient_query, medication_states):
        med_factor = self.med_proj(self.med_norm(medication_states))
        state_factor = self.state_proj(self.dropout(self.state_norm(direct_state)))
        query_factor = self.query_proj(self.dropout(self.query_norm(patient_query)))
        state_med_score = (state_factor.unsqueeze(1) * med_factor).sum(dim=-1) / math.sqrt(med_factor.shape[-1])
        query_med_score = (query_factor.unsqueeze(1) * med_factor).sum(dim=-1) / math.sqrt(med_factor.shape[-1])
        med_bias = self.med_bias.unsqueeze(0).expand_as(base_logits)
        features = torch.stack(
            (
                base_logits,
                direct_logits,
                graph_logits,
                state_med_score,
                query_med_score,
                med_bias,
            ),
            dim=-1,
        )
        delta = self.feature_mlp(features).squeeze(-1)
        delta = delta - delta.mean(dim=-1, keepdim=True)
        return {
            'logits': base_logits + delta,
            'delta': delta,
            'state_med_score': state_med_score,
            'query_med_score': query_med_score,
        }


class PosthocLogitCalibrator(nn.Module):
    def __init__(
        self,
        m_voc_size,
        use_history=True,
        use_retrieval=True,
        graph_mode='none',
        ehr_adj=None,
        ddi_adj=None,
        graph_source='hybrid',
    ):
        super().__init__()
        self.use_history = use_history
        self.use_retrieval = use_retrieval
        self.graph_mode = graph_mode
        self.graph_source = graph_source
        self.logit_scale_delta = nn.Parameter(torch.zeros(m_voc_size))
        self.logit_bias = nn.Parameter(torch.zeros(m_voc_size))
        self.history_weight = nn.Parameter(torch.zeros(m_voc_size))
        self.retrieval_weight = nn.Parameter(torch.zeros(m_voc_size))
        if graph_mode not in ('none', 'per_med'):
            raise ValueError(f'Unsupported posthoc graph_mode: {graph_mode}')
        if graph_source not in ('model', 'history', 'retrieval', 'hybrid'):
            raise ValueError(f'Unsupported posthoc graph_source: {graph_source}')
        self.graph_ehr_weight = None
        self.graph_ddi_weight = None
        if graph_mode == 'per_med':
            if ehr_adj is None or ddi_adj is None:
                raise ValueError('posthoc graph calibration requires EHR and DDI adjacency matrices')
            self.register_buffer('ehr_adj', self._normalize_adj(ehr_adj))
            self.register_buffer('ddi_adj', self._normalize_adj(ddi_adj))
            self.graph_ehr_weight = nn.Parameter(torch.zeros(m_voc_size))
            self.graph_ddi_weight = nn.Parameter(torch.zeros(m_voc_size))
        else:
            self.register_buffer('ehr_adj', torch.empty(0))
            self.register_buffer('ddi_adj', torch.empty(0))

    @staticmethod
    def _history_prior(meds, dtype):
        if meds is None or meds.shape[1] <= 1:
            return None
        return (meds[:, 1:, :].sum(dim=1) > 0).to(dtype=dtype)

    @staticmethod
    def _normalize_adj(adj):
        adj = adj.to(dtype=torch.float32)
        if adj.is_sparse:
            adj = adj.to_dense()
        adj = adj.clone()
        adj.fill_diagonal_(0.0)
        return adj / adj.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _graph_prior(self, med_prob, history_prior, retrieval_prior):
        priors = []
        if self.graph_source in ('model', 'hybrid'):
            priors.append(med_prob)
        if self.graph_source in ('history', 'hybrid') and history_prior is not None:
            priors.append(history_prior)
        if self.graph_source in ('retrieval', 'hybrid') and retrieval_prior is not None:
            priors.append(retrieval_prior)
        if not priors:
            return med_prob
        return torch.stack(priors, dim=0).mean(dim=0).clamp(0.0, 1.0)

    def forward(self, med_logits, meds=None, retrieval_priors=None):
        scale = 1.0 + self.logit_scale_delta
        calibrated = med_logits * scale + self.logit_bias
        history_prior = None
        retrieval_prior = None
        if self.use_history:
            history_prior = self._history_prior(meds, med_logits.dtype)
            if history_prior is not None:
                history_prior = history_prior.to(device=med_logits.device, dtype=med_logits.dtype)
                calibrated = calibrated + history_prior * self.history_weight
        if self.use_retrieval and retrieval_priors is not None:
            retrieval_prior = retrieval_priors.to(device=med_logits.device, dtype=med_logits.dtype)
            calibrated = calibrated + retrieval_prior * self.retrieval_weight
        graph_ehr_prior = None
        graph_ddi_prior = None
        if self.graph_mode == 'per_med':
            med_prob = torch.sigmoid(med_logits)
            graph_prior = self._graph_prior(med_prob, history_prior, retrieval_prior)
            ehr_adj = self.ehr_adj.to(device=med_logits.device, dtype=med_logits.dtype)
            ddi_adj = self.ddi_adj.to(device=med_logits.device, dtype=med_logits.dtype)
            graph_ehr_prior = torch.matmul(graph_prior, ehr_adj)
            graph_ddi_prior = torch.matmul(graph_prior, ddi_adj)
            calibrated = (
                calibrated
                + graph_ehr_prior * self.graph_ehr_weight
                - graph_ddi_prior * self.graph_ddi_weight
            )
        return {
            "logits": calibrated,
            "scale": scale,
            "bias": self.logit_bias,
            "history_prior": history_prior,
            "retrieval_prior": retrieval_prior,
            "graph_ehr_prior": graph_ehr_prior,
            "graph_ddi_prior": graph_ddi_prior,
        }


class OutputMixFusion(nn.Module):
    def __init__(self, gate_input_dim, mix_mode='fixed', init_weight=0.7, branch_calibration='none'):
        super().__init__()
        self.mix_mode = mix_mode
        self.branch_calibration = branch_calibration
        self.last_mix_weights = None

        if branch_calibration == 'o2_scale_bias':
            self.o2_scale = nn.Parameter(torch.tensor(1.0))
            self.o2_bias = nn.Parameter(torch.tensor(0.0))
        elif branch_calibration != 'none':
            raise ValueError(f'Unsupported output branch calibration: {branch_calibration}')

        if mix_mode == 'fixed':
            self.register_buffer('fixed_mix_weight', torch.tensor(float(init_weight), dtype=torch.float32))
        elif mix_mode == 'learnable_scalar':
            init_mix_weight = self._clamp_mix_weight(init_weight)
            self.w_logits = nn.Parameter(torch.logit(torch.tensor(init_mix_weight, dtype=torch.float32)))
        elif mix_mode == 'dynamic_gate':
            init_mix_weight = self._clamp_mix_weight(init_weight)
            self.gate_proj = nn.Linear(gate_input_dim, 1)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, torch.logit(torch.tensor(init_mix_weight)).item())
        else:
            raise ValueError(f'Unsupported output mix mode: {mix_mode}')

    @staticmethod
    def _clamp_mix_weight(weight, eps=1e-4):
        return min(max(float(weight), eps), 1.0 - eps)

    @property
    def current_weight(self):
        if self.mix_mode == 'fixed':
            return self.fixed_mix_weight
        if self.mix_mode == 'learnable_scalar':
            return torch.sigmoid(self.w_logits)
        if self.last_mix_weights is not None:
            return self.last_mix_weights.mean()
        return torch.sigmoid(self.gate_proj.bias).squeeze(0)

    def calibrate_o2(self, o2):
        if self.branch_calibration == 'none':
            return o2
        return o2 * self.o2_scale.to(o2.dtype) + self.o2_bias.to(o2.dtype)

    def compute_mix_weights(self, gate_inputs, ref_tensor):
        batch = ref_tensor.shape[0]
        if self.mix_mode == 'fixed':
            return torch.full(
                (batch, 1),
                self.fixed_mix_weight.item(),
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )
        if self.mix_mode == 'learnable_scalar':
            scalar = torch.sigmoid(self.w_logits).to(device=ref_tensor.device, dtype=ref_tensor.dtype)
            return scalar.view(1, 1).expand(batch, 1)
        return torch.sigmoid(self.gate_proj(gate_inputs))

    def get_logging_stats(self):
        if self.last_mix_weights is None:
            current_weight = float(self.current_weight.detach().item())
            stats = {
                'mean': current_weight,
                'min': current_weight,
                'max': current_weight,
            }
        else:
            stats = {
                'mean': float(self.last_mix_weights.mean().item()),
                'min': float(self.last_mix_weights.min().item()),
                'max': float(self.last_mix_weights.max().item()),
            }
        stats['o2_scale'] = float(self.o2_scale.detach().item()) if self.branch_calibration == 'o2_scale_bias' else 1.0
        stats['o2_bias'] = float(self.o2_bias.detach().item()) if self.branch_calibration == 'o2_scale_bias' else 0.0
        return stats

    def forward(self, o_1, o_2, gate_inputs):
        mix_weights = self.compute_mix_weights(gate_inputs, o_1)
        calibrated_o_2 = self.calibrate_o2(o_2)
        self.last_mix_weights = mix_weights.detach()
        return o_1 * mix_weights + calibrated_o_2 * (1 - mix_weights)


class RetrievedMedicationAttention(nn.Module):
    def __init__(self, emb_dim, state_dim, m_voc_size, dropout=0.1, gate_init=-3.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.state_dim = state_dim
        self.m_voc_size = m_voc_size
        self.query_proj = nn.Linear(state_dim, emb_dim)
        self.visit_query_proj = nn.Linear(state_dim, emb_dim)
        self.candidate_proj = nn.Linear(emb_dim, emb_dim)
        self.health_status_query_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.context_norm = nn.LayerNorm(emb_dim)
        self.out = nn.Linear(state_dim + emb_dim, m_voc_size)
        self.state_delta = nn.Linear(emb_dim, state_dim)
        self.gate = nn.Linear(state_dim + emb_dim, 1)
        nn.init.zeros_(self.state_delta.weight)
        nn.init.zeros_(self.state_delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init)
        self.drop = nn.Dropout(dropout)
        self.last_gate = None

    def forward(self, patient_query, med_embeddings, retrieval_meds, retrieval_scores=None, retrieval_queries=None):
        if retrieval_meds is None:
            return None

        retrieval_meds = retrieval_meds.to(device=patient_query.device, dtype=patient_query.dtype)
        batch, topk, _ = retrieval_meds.shape
        med_embeddings = med_embeddings.to(device=patient_query.device, dtype=patient_query.dtype)

        med_query = self.query_proj(patient_query)
        candidate_embeddings = self.candidate_proj(med_embeddings)
        med_logits = torch.matmul(med_query, candidate_embeddings.transpose(0, 1)) / math.sqrt(self.emb_dim)
        med_logits = med_logits.unsqueeze(1).expand(batch, topk, self.m_voc_size)
        med_logits = med_logits.masked_fill(retrieval_meds <= 0, -1e4)
        empty_visits = retrieval_meds.sum(dim=-1, keepdim=True) <= 0
        med_attn = torch.softmax(med_logits, dim=-1)
        med_attn = torch.where(empty_visits, torch.zeros_like(med_attn), med_attn)
        prescription_context = torch.matmul(med_attn, med_embeddings)

        visit_logits = torch.matmul(
            prescription_context,
            self.visit_query_proj(patient_query).unsqueeze(-1),
        ).squeeze(-1) / math.sqrt(self.emb_dim)
        if retrieval_scores is not None:
            retrieval_scores = retrieval_scores.to(device=patient_query.device, dtype=patient_query.dtype)
            visit_logits = visit_logits + retrieval_scores
        if retrieval_queries is not None:
            retrieval_queries = retrieval_queries.to(device=patient_query.device, dtype=patient_query.dtype)
            health_status_query = self.health_status_query_proj(patient_query)
            health_logits = torch.matmul(
                retrieval_queries,
                health_status_query.unsqueeze(-1),
            ).squeeze(-1) / math.sqrt(self.state_dim)
            visit_logits = visit_logits + health_logits
        visit_logits = visit_logits.masked_fill(empty_visits.squeeze(-1), -1e4)
        visit_attn = torch.softmax(visit_logits, dim=-1)
        visit_attn = torch.where(empty_visits.squeeze(-1), torch.zeros_like(visit_attn), visit_attn)
        retrieved_context = torch.bmm(visit_attn.unsqueeze(1), prescription_context).squeeze(1)
        retrieved_context = self.context_norm(self.drop(retrieved_context))
        gate_inputs = torch.cat((patient_query, retrieved_context), dim=-1)
        gate = torch.sigmoid(self.gate(gate_inputs))
        self.last_gate = gate.detach()
        return {
            'logits': self.out(gate_inputs),
            'state_delta': self.state_delta(retrieved_context),
            'gate': gate,
        }


class RetrievedMedicationProbMix(nn.Module):
    def __init__(
        self,
        state_dim,
        m_voc_size,
        gate_init=2.0,
        dropout=0.1,
    ):
        super().__init__()
        self.m_voc_size = m_voc_size
        self.context_norm = nn.LayerNorm(state_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.copy_score = nn.Linear(state_dim * 2, m_voc_size)
        self.health_status_query = nn.Linear(state_dim, state_dim, bias=False)
        self.generate_gate = nn.Linear(state_dim * 2, 1)
        nn.init.zeros_(self.copy_score.weight)
        nn.init.zeros_(self.copy_score.bias)
        nn.init.zeros_(self.generate_gate.weight)
        nn.init.constant_(self.generate_gate.bias, gate_init)
        self.last_generate_gate = None
        self.last_visit_attn = None

    def _retrieval_prior(self, patient_query, retrieval_meds, retrieval_scores=None, retrieval_queries=None):
        retrieval_meds = retrieval_meds.to(device=patient_query.device, dtype=patient_query.dtype)
        valid_visits = retrieval_meds.sum(dim=-1) > 0
        if retrieval_scores is None:
            visit_logits = retrieval_meds.new_zeros(valid_visits.shape)
        else:
            visit_logits = retrieval_scores.to(device=patient_query.device, dtype=patient_query.dtype)
        if retrieval_queries is not None:
            retrieval_queries = retrieval_queries.to(device=patient_query.device, dtype=patient_query.dtype)
            health_query = self.health_status_query(patient_query)
            health_logits = torch.matmul(
                retrieval_queries,
                health_query.unsqueeze(-1),
            ).squeeze(-1) / math.sqrt(patient_query.shape[-1])
            visit_logits = visit_logits + health_logits
        visit_logits = visit_logits.masked_fill(~valid_visits, -1e4)
        visit_attn = torch.softmax(visit_logits, dim=-1)
        visit_attn = torch.where(valid_visits, visit_attn, torch.zeros_like(visit_attn))
        has_retrieval = valid_visits.any(dim=-1, keepdim=True)
        copy_prior = torch.bmm(visit_attn.unsqueeze(1), retrieval_meds).squeeze(1)
        copy_prior = torch.where(has_retrieval, copy_prior, torch.zeros_like(copy_prior))
        self.last_visit_attn = visit_attn.detach()
        return copy_prior.clamp(0.0, 1.0), has_retrieval

    def forward(self, base_logits, patient_query, direct_state, retrieval_meds, retrieval_scores=None, retrieval_queries=None):
        if retrieval_meds is None:
            return None
        copy_prior, has_retrieval = self._retrieval_prior(
            patient_query,
            retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
        )
        context = self.context_norm(torch.cat((patient_query, direct_state), dim=-1))
        context = self.drop(context)
        generate_gate = torch.sigmoid(self.generate_gate(context))
        generate_gate = torch.where(has_retrieval, generate_gate, torch.ones_like(generate_gate))
        copy_logits = torch.logit(copy_prior.clamp(1e-6, 1.0 - 1e-6)) + self.copy_score(context)
        copy_prob = torch.sigmoid(copy_logits)
        base_prob = torch.sigmoid(base_logits)
        mixed_prob = generate_gate * base_prob + (1.0 - generate_gate) * copy_prob
        mixed_logits = torch.logit(mixed_prob.clamp(1e-6, 1.0 - 1e-6))
        self.last_generate_gate = generate_gate.detach()
        return {
            'logits': mixed_logits,
            'copy_prob': copy_prob,
            'generate_gate': generate_gate,
            'prior': copy_prior,
        }


class RetrievedMedicationReadout(nn.Module):
    def __init__(
        self,
        state_dim,
        m_voc_size,
        gate_init=-2.0,
        dropout=0.1,
        prior_transform='logit',
    ):
        super().__init__()
        self.m_voc_size = m_voc_size
        self.prior_transform = prior_transform
        if prior_transform not in ('logit', 'positive', 'centered'):
            raise ValueError(f'Unsupported retrieval readout prior transform: {prior_transform}')
        self.context_norm = nn.LayerNorm(state_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.copy_score = nn.Linear(state_dim * 2, m_voc_size)
        self.health_status_query = nn.Linear(state_dim, state_dim, bias=False)
        self.gate = nn.Linear(state_dim * 2, 1)
        nn.init.zeros_(self.copy_score.weight)
        nn.init.zeros_(self.copy_score.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init)
        self.last_gate = None
        self.last_visit_attn = None

    def _retrieval_prior(self, patient_query, retrieval_meds, retrieval_scores=None, retrieval_queries=None):
        retrieval_meds = retrieval_meds.to(device=patient_query.device, dtype=patient_query.dtype)
        valid_visits = retrieval_meds.sum(dim=-1) > 0
        visit_logits = retrieval_meds.new_zeros(valid_visits.shape)
        if retrieval_scores is not None:
            retrieval_scores = retrieval_scores.to(device=patient_query.device, dtype=patient_query.dtype)
            visit_logits = visit_logits + torch.nan_to_num(retrieval_scores, nan=0.0, posinf=0.0, neginf=0.0)
        if retrieval_queries is not None:
            retrieval_queries = retrieval_queries.to(device=patient_query.device, dtype=patient_query.dtype)
            health_query = F.normalize(self.health_status_query(patient_query), dim=-1, eps=1e-6)
            health_keys = F.normalize(retrieval_queries, dim=-1, eps=1e-6)
            health_logits = torch.matmul(
                health_keys,
                health_query.unsqueeze(-1),
            ).squeeze(-1)
            visit_logits = visit_logits + health_logits
        visit_logits = visit_logits.masked_fill(~valid_visits, -1e4)
        visit_attn = torch.softmax(visit_logits, dim=-1)
        visit_attn = torch.where(valid_visits, visit_attn, torch.zeros_like(visit_attn))
        has_retrieval = valid_visits.any(dim=-1, keepdim=True)
        prior = torch.bmm(visit_attn.unsqueeze(1), retrieval_meds).squeeze(1)
        prior = torch.where(has_retrieval, prior, torch.zeros_like(prior))
        self.last_visit_attn = visit_attn.detach()
        return prior.clamp(0.0, 1.0), has_retrieval

    def forward(self, patient_query, direct_state, retrieval_meds, retrieval_scores=None, retrieval_queries=None):
        if retrieval_meds is None:
            return None
        readout_prior, has_retrieval = self._retrieval_prior(
            patient_query,
            retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
        )
        context = self.context_norm(torch.cat((patient_query, direct_state), dim=-1))
        context = self.drop(context)
        gate = torch.sigmoid(self.gate(context))
        gate = torch.where(has_retrieval, gate, torch.zeros_like(gate))
        if self.prior_transform == 'logit':
            prior_signal = torch.logit(readout_prior.clamp(1e-4, 1.0 - 1e-4))
        elif self.prior_transform == 'positive':
            prior_signal = readout_prior
        else:
            prior_signal = readout_prior - readout_prior.mean(dim=-1, keepdim=True)
        readout_logits = torch.clamp(prior_signal + self.copy_score(context), min=-8.0, max=8.0)
        logit_bias = gate * readout_logits
        self.last_gate = gate.detach()
        return {
            'logit_bias': logit_bias,
            'gate': gate,
            'prior': readout_prior,
            'visit_attn': self.last_visit_attn,
        }


class TreatmentEpisodeExperienceAdapter(nn.Module):
    def __init__(
        self,
        state_dim,
        m_voc_size,
        experience_dim,
        mode='logit_bias',
        gate_init=-3.0,
        dropout=0.1,
    ):
        super().__init__()
        if mode not in ('logit_bias', 'state', 'both', 'action'):
            raise ValueError(f'Unsupported treatment episode adapter mode: {mode}')
        if experience_dim <= 0:
            raise ValueError('treatment episode adapter requires positive experience_dim')
        self.mode = mode
        self.state_dim = state_dim
        self.m_voc_size = m_voc_size
        self.experience_dim = experience_dim
        self.query_proj = nn.Linear(state_dim, experience_dim, bias=False)
        self.experience_norm = nn.LayerNorm(experience_dim)
        self.context_norm = nn.LayerNorm(state_dim + experience_dim)
        self.drop = nn.Dropout(dropout)
        self.logit_head = nn.Linear(state_dim + experience_dim, m_voc_size)
        self.action_add_head = nn.Linear(state_dim + experience_dim, m_voc_size)
        self.action_continue_head = nn.Linear(state_dim + experience_dim, m_voc_size)
        self.action_remove_head = nn.Linear(state_dim + experience_dim, m_voc_size)
        self.state_delta = nn.Linear(experience_dim, state_dim)
        self.gate = nn.Linear(state_dim + experience_dim, 1)
        nn.init.zeros_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)
        nn.init.zeros_(self.state_delta.weight)
        nn.init.zeros_(self.state_delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init)
        self.last_gate = None
        self.last_visit_attn = None

    def forward(self, patient_query, treatment_episode_experiences, retrieval_scores=None):
        if treatment_episode_experiences is None:
            return None
        experiences = treatment_episode_experiences.to(device=patient_query.device, dtype=patient_query.dtype)
        if experiences.dim() != 3:
            raise ValueError('treatment_episode_experiences must have shape [batch, topk, experience_dim]')
        if experiences.size(-1) != self.experience_dim:
            raise ValueError(
                f'treatment episode experience dim {experiences.size(-1)} does not match adapter dim {self.experience_dim}'
            )
        valid_visits = experiences.abs().sum(dim=-1) > 0
        query = self.query_proj(patient_query)
        norm_experiences = self.experience_norm(experiences)
        visit_logits = torch.matmul(norm_experiences, query.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.experience_dim)
        if retrieval_scores is not None:
            retrieval_scores = retrieval_scores.to(device=patient_query.device, dtype=patient_query.dtype)
            visit_logits = visit_logits + torch.nan_to_num(retrieval_scores, nan=0.0, posinf=0.0, neginf=0.0)
        visit_logits = visit_logits.masked_fill(~valid_visits, -1e4)
        visit_attn = torch.softmax(visit_logits, dim=-1)
        visit_attn = torch.where(valid_visits, visit_attn, torch.zeros_like(visit_attn))
        has_experience = valid_visits.any(dim=-1, keepdim=True)
        experience_context = torch.bmm(visit_attn.unsqueeze(1), norm_experiences).squeeze(1)
        adapter_context = self.context_norm(torch.cat((patient_query, experience_context), dim=-1))
        adapter_context = self.drop(adapter_context)
        gate = torch.sigmoid(self.gate(adapter_context))
        gate = torch.where(has_experience, gate, torch.zeros_like(gate))
        self.last_gate = gate.detach()
        self.last_visit_attn = visit_attn.detach()
        logit_bias = self.logit_head(adapter_context)
        action_add_logits = self.action_add_head(adapter_context)
        action_continue_logits = self.action_continue_head(adapter_context)
        action_remove_logits = self.action_remove_head(adapter_context)
        state_delta = self.state_delta(experience_context)
        logit_bias = torch.where(has_experience, logit_bias, torch.zeros_like(logit_bias))
        action_add_logits = torch.where(has_experience, action_add_logits, torch.zeros_like(action_add_logits))
        action_continue_logits = torch.where(has_experience, action_continue_logits, torch.zeros_like(action_continue_logits))
        action_remove_logits = torch.where(has_experience, action_remove_logits, torch.zeros_like(action_remove_logits))
        state_delta = torch.where(has_experience, state_delta, torch.zeros_like(state_delta))
        return {
            'logit_bias': logit_bias,
            'action_add_logits': action_add_logits,
            'action_continue_logits': action_continue_logits,
            'action_remove_logits': action_remove_logits,
            'state_delta': state_delta,
            'gate': gate,
            'has_experience': has_experience,
            'visit_attn': self.last_visit_attn,
            'experience_context': experience_context,
        }


class ExternalMemoryNetwork(nn.Module):
    def __init__(self, input_dim, state_dim, hidden_size, memory_num, m_voc_size, dropout=0.1, gate_init=-3.0):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.memory_num = memory_num
        self.query_proj = nn.Linear(input_dim, hidden_size)
        self.linear_memory_key = nn.Linear(hidden_size, memory_num, bias=False)
        self.linear_memory_value = nn.Linear(memory_num, hidden_size, bias=False)
        self.value_norm = nn.LayerNorm(hidden_size)
        self.state_delta = nn.Linear(hidden_size, state_dim)
        self.logit_head = nn.Linear(state_dim + hidden_size, m_voc_size)
        self.gate = nn.Linear(state_dim + hidden_size, 1)
        nn.init.zeros_(self.state_delta.weight)
        nn.init.zeros_(self.state_delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init)
        self.drop = nn.Dropout(dropout)
        self.last_attn = None
        self.last_gate = None

    def forward(self, query_input, patient_state):
        query = torch.tanh(self.query_proj(query_input))
        memory_logits = self.linear_memory_key(query)
        memory_attn = torch.softmax(memory_logits, dim=-1)
        memory_value = self.linear_memory_value(memory_attn)
        memory_value = self.value_norm(self.drop(memory_value))
        fusion_input = torch.cat((patient_state, memory_value), dim=-1)
        gate = torch.sigmoid(self.gate(fusion_input))
        self.last_attn = memory_attn.detach()
        self.last_gate = gate.detach()
        return {
            'state_delta': self.state_delta(memory_value),
            'logits': self.logit_head(fusion_input),
            'gate': gate,
            'attn': memory_attn,
        }


class LocalMemoryNetwork(nn.Module):
    def __init__(self, emb_dim, state_dim, dropout=0.1, gate_init=-3.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.state_dim = state_dim
        self.query_proj = nn.Linear(emb_dim, emb_dim)
        self.key_proj = nn.Linear(emb_dim, emb_dim)
        self.value_norm = nn.LayerNorm(emb_dim)
        self.state_delta = nn.Linear(emb_dim, state_dim)
        self.logit_head = nn.Linear(state_dim + emb_dim, state_dim)
        self.gate = nn.Linear(state_dim + emb_dim, 1)
        nn.init.zeros_(self.state_delta.weight)
        nn.init.zeros_(self.state_delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init)
        self.drop = nn.Dropout(dropout)
        self.last_gate = None
        self.last_visit_attn = None

    def forward(self, visit_embeddings, medication_sequence, drug_memory, patient_state):
        if visit_embeddings.size(1) <= 1:
            empty_value = patient_state.new_zeros(patient_state.size(0), self.emb_dim)
            gate = patient_state.new_zeros(patient_state.size(0), 1)
            self.last_gate = gate.detach()
            self.last_visit_attn = None
            return {
                'state_delta': patient_state.new_zeros(patient_state.shape),
                'value': empty_value,
                'gate': gate,
                'visit_attn': None,
            }

        history_len = min(visit_embeddings.size(1) - 1, medication_sequence.size(1) - 1)
        if history_len <= 0:
            empty_value = patient_state.new_zeros(patient_state.size(0), self.emb_dim)
            gate = patient_state.new_zeros(patient_state.size(0), 1)
            self.last_gate = gate.detach()
            self.last_visit_attn = None
            return {
                'state_delta': patient_state.new_zeros(patient_state.shape),
                'value': empty_value,
                'gate': gate,
                'visit_attn': None,
            }

        query = self.query_proj(visit_embeddings[:, :1, :])
        memory_keys = self.key_proj(visit_embeddings[:, 1:1 + history_len, :])
        memory_values = medication_sequence[:, 1:1 + history_len, :].to(device=patient_state.device, dtype=patient_state.dtype)
        valid_history = memory_values.sum(dim=-1) > 0
        visit_logits = torch.bmm(query, memory_keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.emb_dim)
        visit_logits = visit_logits.masked_fill(~valid_history, -1e4)
        visit_attn = torch.softmax(visit_logits, dim=-1)
        visit_attn = torch.where(valid_history, visit_attn, torch.zeros_like(visit_attn))
        has_history = valid_history.any(dim=-1, keepdim=True)
        weighted_meds = torch.bmm(visit_attn.unsqueeze(1), memory_values).squeeze(1)
        local_value = torch.matmul(weighted_meds, drug_memory.to(device=patient_state.device, dtype=patient_state.dtype))
        local_value = self.value_norm(self.drop(local_value))
        fusion_input = torch.cat((patient_state, local_value), dim=-1)
        gate = torch.sigmoid(self.gate(fusion_input))
        gate = torch.where(has_history, gate, torch.zeros_like(gate))
        local_value = torch.where(has_history, local_value, torch.zeros_like(local_value))
        state_delta = self.state_delta(local_value)
        state_delta = torch.where(has_history, state_delta, torch.zeros_like(state_delta))
        self.last_gate = gate.detach()
        self.last_visit_attn = visit_attn.detach()
        return {
            'state_delta': state_delta,
            'value': local_value,
            'gate': gate,
            'visit_attn': visit_attn,
        }


class OwnHistoryVisitAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        state_dim,
        m_voc_size,
        dropout=0.1,
        gate_init=-3.0,
        recency_decay=1.0,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.state_dim = state_dim
        self.m_voc_size = m_voc_size
        self.recency_decay = recency_decay
        self.query_norm = nn.LayerNorm(state_dim)
        self.history_norm = nn.LayerNorm(emb_dim)
        self.query_proj = nn.Linear(state_dim, emb_dim, bias=False)
        self.key_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.context_norm = nn.LayerNorm(emb_dim)
        self.fusion_norm = nn.LayerNorm(state_dim + emb_dim)
        self.state_delta = nn.Linear(emb_dim, state_dim)
        self.logit_bias = nn.Linear(state_dim + emb_dim, m_voc_size)
        self.gate = nn.Linear(state_dim + emb_dim, 1)
        nn.init.zeros_(self.state_delta.weight)
        nn.init.zeros_(self.state_delta.bias)
        nn.init.zeros_(self.logit_bias.weight)
        nn.init.zeros_(self.logit_bias.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init)
        self.drop = nn.Dropout(dropout)
        self.last_gate = None
        self.last_visit_attn = None

    def forward(self, patient_query, patient_state, visit_context, meds):
        batch = patient_query.size(0)
        if visit_context.size(1) <= 1 or meds.size(1) <= 1:
            empty_attention = patient_query.new_zeros(batch, 0)
            empty_gate = patient_query.new_zeros(batch, 1)
            self.last_gate = empty_gate.detach()
            self.last_visit_attn = empty_attention.detach()
            return {
                'state_delta': patient_state.new_zeros(patient_state.shape),
                'logit_bias': patient_query.new_zeros(batch, self.m_voc_size),
                'prior': patient_query.new_zeros(batch, self.m_voc_size),
                'gate': empty_gate,
                'visit_attn': empty_attention,
                'valid_history': empty_attention.to(dtype=torch.bool),
            }

        history_len = min(visit_context.size(1) - 1, meds.size(1) - 1)
        if history_len <= 0:
            empty_attention = patient_query.new_zeros(batch, 0)
            empty_gate = patient_query.new_zeros(batch, 1)
            self.last_gate = empty_gate.detach()
            self.last_visit_attn = empty_attention.detach()
            return {
                'state_delta': patient_state.new_zeros(patient_state.shape),
                'logit_bias': patient_query.new_zeros(batch, self.m_voc_size),
                'prior': patient_query.new_zeros(batch, self.m_voc_size),
                'gate': empty_gate,
                'visit_attn': empty_attention,
                'valid_history': empty_attention.to(dtype=torch.bool),
            }

        history_context = visit_context[:, 1:1 + history_len, :]
        history_meds = meds[:, 1:1 + history_len, :].to(device=patient_query.device, dtype=patient_query.dtype)
        valid_history = history_meds.sum(dim=-1) > 0

        query = self.query_proj(self.query_norm(patient_query)).unsqueeze(1)
        keys = self.key_proj(self.history_norm(history_context))
        values = self.value_proj(history_context)
        scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.emb_dim)
        if self.recency_decay < 1.0:
            steps = torch.arange(history_len, device=scores.device, dtype=scores.dtype)
            scores = scores + steps.view(1, -1) * math.log(max(self.recency_decay, 1e-6))
        scores = scores.masked_fill(~valid_history, -1e4)
        visit_attn = torch.softmax(scores, dim=-1)
        visit_attn = torch.where(valid_history, visit_attn, torch.zeros_like(visit_attn))
        visit_attn = visit_attn / visit_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        has_history = valid_history.any(dim=-1, keepdim=True)

        history_value = torch.bmm(visit_attn.unsqueeze(1), values).squeeze(1)
        history_value = self.context_norm(self.drop(history_value))
        history_value = torch.where(has_history, history_value, torch.zeros_like(history_value))
        prior = torch.bmm(visit_attn.unsqueeze(1), history_meds).squeeze(1).clamp(0.0, 1.0)
        prior = torch.where(has_history, prior, torch.zeros_like(prior))
        fusion_input = self.fusion_norm(torch.cat((patient_state, history_value), dim=-1))
        gate = torch.sigmoid(self.gate(fusion_input))
        gate = torch.where(has_history, gate, torch.zeros_like(gate))
        state_delta = self.state_delta(history_value)
        state_delta = torch.where(has_history, state_delta, torch.zeros_like(state_delta))
        logit_bias = self.logit_bias(fusion_input) * prior

        self.last_gate = gate.detach()
        self.last_visit_attn = visit_attn.detach()
        return {
            'state_delta': state_delta,
            'logit_bias': logit_bias,
            'prior': prior,
            'gate': gate,
            'visit_attn': visit_attn,
            'valid_history': valid_history,
        }


class HistoricalMedicationCopy(nn.Module):
    def __init__(
        self,
        state_dim,
        m_voc_size,
        prior_mode='binary',
        decay=0.7,
        gate_init=-2.0,
        dropout=0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.m_voc_size = m_voc_size
        self.prior_mode = prior_mode
        self.decay = decay
        if prior_mode not in ('binary', 'recency'):
            raise ValueError(f'Unsupported history copy prior mode: {prior_mode}')
        self.context_norm = nn.LayerNorm(state_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.copy_score = nn.Linear(state_dim * 2, m_voc_size)
        self.gate = nn.Linear(state_dim * 2, 1)
        nn.init.zeros_(self.copy_score.weight)
        nn.init.zeros_(self.copy_score.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init)
        self.last_gate = None

    def _history_prior(self, meds):
        history = meds[:, 1:, :]
        if history.shape[1] == 0:
            return meds.new_zeros(meds.shape[0], self.m_voc_size)
        history = (history > 0).to(dtype=meds.dtype)
        if self.prior_mode == 'binary':
            return (history.sum(dim=1) > 0).to(dtype=meds.dtype)
        recency_steps = torch.arange(history.shape[1], device=meds.device, dtype=meds.dtype)
        recency_weights = torch.pow(
            torch.full_like(recency_steps, self.decay),
            recency_steps,
        ).view(1, -1, 1)
        return (history * recency_weights).max(dim=1).values

    def forward(self, patient_query, direct_state, meds):
        history_prior = self._history_prior(meds)
        context = self.context_norm(torch.cat((patient_query, direct_state), dim=-1))
        context = self.drop(context)
        gate = torch.sigmoid(self.gate(context))
        copy_scores = self.copy_score(context)
        self.last_gate = gate.detach()
        return {
            'logits': gate * copy_scores * history_prior,
            'gate': gate,
            'prior': history_prior,
        }


class HistoricalMedicationProbMix(nn.Module):
    def __init__(
        self,
        state_dim,
        m_voc_size,
        prior_mode='binary',
        decay=0.7,
        gate_init=2.0,
        dropout=0.1,
    ):
        super().__init__()
        self.m_voc_size = m_voc_size
        self.prior_mode = prior_mode
        self.decay = decay
        if prior_mode not in ('binary', 'recency'):
            raise ValueError(f'Unsupported history copy prior mode: {prior_mode}')
        self.context_norm = nn.LayerNorm(state_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.copy_score = nn.Linear(state_dim * 2, m_voc_size)
        self.generate_gate = nn.Linear(state_dim * 2, 1)
        nn.init.zeros_(self.copy_score.weight)
        nn.init.zeros_(self.copy_score.bias)
        nn.init.zeros_(self.generate_gate.weight)
        nn.init.constant_(self.generate_gate.bias, gate_init)
        self.last_generate_gate = None

    def _history_prior(self, meds):
        history = meds[:, 1:, :]
        if history.shape[1] == 0:
            return meds.new_zeros(meds.shape[0], self.m_voc_size)
        history = (history > 0).to(dtype=meds.dtype)
        if self.prior_mode == 'binary':
            return (history.sum(dim=1) > 0).to(dtype=meds.dtype)
        recency_steps = torch.arange(history.shape[1], device=meds.device, dtype=meds.dtype)
        recency_weights = torch.pow(
            torch.full_like(recency_steps, self.decay),
            recency_steps,
        ).view(1, -1, 1)
        return (history * recency_weights).max(dim=1).values

    def forward(self, base_logits, patient_query, direct_state, meds):
        history_prior = self._history_prior(meds)
        context = self.context_norm(torch.cat((patient_query, direct_state), dim=-1))
        context = self.drop(context)
        generate_gate = torch.sigmoid(self.generate_gate(context))
        copy_prob = torch.sigmoid(self.copy_score(context)) * history_prior
        base_prob = torch.sigmoid(base_logits)
        mixed_prob = generate_gate * base_prob + (1.0 - generate_gate) * copy_prob
        mixed_logits = torch.logit(mixed_prob.clamp(1e-6, 1.0 - 1e-6))
        self.last_generate_gate = generate_gate.detach()
        return {
            'logits': mixed_logits,
            'copy_prob': copy_prob,
            'generate_gate': generate_gate,
            'prior': history_prior,
        }


class MedicationSequenceDecoder(nn.Module):
    def __init__(
        self,
        state_dim,
        m_voc_size,
        dropout=0.1,
        copy_mode='none',
        copy_gate_init=1.0,
        copy_recency_decay=0.7,
        retrieval_copy_mode='none',
        retrieval_copy_gate_init=1.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.m_voc_size = m_voc_size
        self.copy_mode = copy_mode
        self.copy_recency_decay = copy_recency_decay
        self.retrieval_copy_mode = retrieval_copy_mode
        self.sos_token = m_voc_size
        self.end_token = m_voc_size + 1
        self.vocab_size = m_voc_size + 2
        self.token_embedding = nn.Embedding(self.vocab_size, state_dim)
        self.context_norm = nn.LayerNorm(state_dim)
        self.input_norm = nn.LayerNorm(state_dim)
        self.init_hidden = nn.Linear(state_dim, state_dim)
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(state_dim, self.vocab_size)
        self.base_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.copy_query = None
        self.copy_gate = None
        if copy_mode == 'history_attention':
            self.copy_query = nn.Linear(state_dim, state_dim)
            self.copy_gate = nn.Linear(state_dim, 1)
            nn.init.zeros_(self.copy_gate.weight)
            nn.init.constant_(self.copy_gate.bias, copy_gate_init)
        elif copy_mode != 'none':
            raise ValueError(f'Unsupported seq copy mode: {copy_mode}')
        self.retrieval_copy_query = None
        self.retrieval_health_query = None
        self.retrieval_copy_gate = None
        if retrieval_copy_mode == 'retrieval_attention':
            self.retrieval_copy_query = nn.Linear(state_dim, state_dim)
            self.retrieval_health_query = nn.Linear(state_dim, state_dim, bias=False)
            self.retrieval_copy_gate = nn.Linear(state_dim, 1)
            nn.init.zeros_(self.retrieval_copy_gate.weight)
            nn.init.constant_(self.retrieval_copy_gate.bias, retrieval_copy_gate_init)
        elif retrieval_copy_mode != 'none':
            raise ValueError(f'Unsupported seq retrieval copy mode: {retrieval_copy_mode}')

    def _initial_hidden(self, patient_query):
        return torch.tanh(self.init_hidden(self.context_norm(patient_query))).unsqueeze(0)

    def _add_base_logits(self, seq_logits, med_logits):
        med_part = seq_logits[..., :self.m_voc_size] + self.base_logit_scale * med_logits.unsqueeze(1)
        special_part = seq_logits[..., self.m_voc_size:]
        return torch.cat((med_part, special_part), dim=-1)

    def _history_copy_logits(self, decoder_output, base_logits, meds):
        if self.copy_mode == 'none' or meds is None:
            return base_logits
        history = meds[:, 1:, :].to(dtype=base_logits.dtype)
        if history.shape[1] == 0:
            return base_logits

        history = (history > 0).to(dtype=base_logits.dtype)
        recency_steps = torch.arange(history.shape[1], device=history.device, dtype=history.dtype)
        recency_weights = torch.pow(
            torch.full_like(recency_steps, self.copy_recency_decay),
            recency_steps,
        ).view(1, -1, 1)
        history_weight = (history * recency_weights).sum(dim=1)
        empty_history = history_weight.sum(dim=1) <= 0

        med_embeddings = self.token_embedding.weight[:self.m_voc_size]
        copy_query = self.copy_query(decoder_output)
        copy_scores = torch.matmul(copy_query, med_embeddings.transpose(0, 1)) / math.sqrt(self.state_dim)
        copy_scores = copy_scores + torch.log(history_weight.clamp_min(1e-8)).unsqueeze(1)
        copy_scores = copy_scores.masked_fill(history_weight.unsqueeze(1) <= 0, -1e4)
        copy_prob = torch.softmax(copy_scores, dim=-1)
        copy_prob = torch.where(
            empty_history.view(-1, 1, 1),
            torch.zeros_like(copy_prob),
            copy_prob,
        )

        copy_full_prob = torch.zeros_like(base_logits)
        copy_full_prob[..., :self.m_voc_size] = copy_prob
        generate_gate = torch.sigmoid(self.copy_gate(decoder_output))
        generate_gate = torch.where(
            empty_history.view(-1, 1, 1),
            torch.ones_like(generate_gate),
            generate_gate,
        )
        generate_prob = torch.softmax(base_logits, dim=-1)
        mixed_prob = generate_gate * generate_prob + (1.0 - generate_gate) * copy_full_prob
        return torch.log(mixed_prob.clamp_min(1e-12))

    def _retrieval_copy_logits(
        self,
        decoder_output,
        base_logits,
        patient_query,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
    ):
        if self.retrieval_copy_mode == 'none' or retrieval_meds is None:
            return base_logits
        retrieval_meds = retrieval_meds.to(device=base_logits.device, dtype=base_logits.dtype)
        valid_visits = retrieval_meds.sum(dim=-1) > 0
        has_retrieval = valid_visits.any(dim=-1)
        if not bool(has_retrieval.any()):
            return base_logits

        if retrieval_scores is None:
            visit_logits = retrieval_meds.new_zeros(valid_visits.shape)
        else:
            visit_logits = retrieval_scores.to(device=base_logits.device, dtype=base_logits.dtype)
        if retrieval_queries is not None:
            retrieval_queries = retrieval_queries.to(device=base_logits.device, dtype=base_logits.dtype)
            health_query = self.retrieval_health_query(patient_query)
            health_logits = torch.matmul(
                retrieval_queries,
                health_query.unsqueeze(-1),
            ).squeeze(-1) / math.sqrt(self.state_dim)
            visit_logits = visit_logits + health_logits
        visit_logits = visit_logits.masked_fill(~valid_visits, -1e4)
        visit_attn = torch.softmax(visit_logits, dim=-1)
        visit_attn = torch.where(valid_visits, visit_attn, torch.zeros_like(visit_attn))
        visit_med_prior = visit_attn.unsqueeze(-1) * retrieval_meds

        med_embeddings = self.token_embedding.weight[:self.m_voc_size]
        copy_query = self.retrieval_copy_query(decoder_output)
        med_scores = torch.matmul(copy_query, med_embeddings.transpose(0, 1)) / math.sqrt(self.state_dim)
        copy_scores = med_scores.unsqueeze(2) + torch.log(visit_med_prior.clamp_min(1e-8)).unsqueeze(1)
        copy_scores = copy_scores.masked_fill(visit_med_prior.unsqueeze(1) <= 0, -1e4)
        batch, steps, _, med_count = copy_scores.shape
        copy_prob_flat = torch.softmax(copy_scores.view(batch, steps, -1), dim=-1)
        copy_prob = copy_prob_flat.view(batch, steps, -1, med_count).sum(dim=2)
        copy_prob = torch.where(has_retrieval.view(-1, 1, 1), copy_prob, torch.zeros_like(copy_prob))

        copy_full_prob = torch.zeros_like(base_logits)
        copy_full_prob[..., :self.m_voc_size] = copy_prob
        generate_gate = torch.sigmoid(self.retrieval_copy_gate(decoder_output))
        generate_gate = torch.where(has_retrieval.view(-1, 1, 1), generate_gate, torch.ones_like(generate_gate))
        generate_prob = torch.softmax(base_logits, dim=-1)
        mixed_prob = generate_gate * generate_prob + (1.0 - generate_gate) * copy_full_prob
        return torch.log(mixed_prob.clamp_min(1e-12))

    def _decode_logits(
        self,
        seq_logits,
        decoder_output,
        med_logits,
        patient_query,
        meds=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
    ):
        base_logits = self._add_base_logits(seq_logits, med_logits)
        base_logits = self._history_copy_logits(decoder_output, base_logits, meds)
        return self._retrieval_copy_logits(
            decoder_output,
            base_logits,
            patient_query,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
        )

    def forward(
        self,
        patient_query,
        med_logits,
        decoder_inputs,
        meds=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
    ):
        context = self.context_norm(patient_query).unsqueeze(1)
        token_context = self.token_embedding(decoder_inputs) + context
        decoder_input = self.input_norm(token_context)
        hidden = self._initial_hidden(patient_query)
        decoder_output, _ = self.gru(decoder_input, hidden)
        seq_logits = self.out(self.drop(decoder_output))
        return self._decode_logits(
            seq_logits,
            decoder_output,
            med_logits,
            patient_query,
            meds=meds,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
        )

    def greedy_decode(
        self,
        patient_query,
        med_logits,
        max_len,
        end_bias=0.0,
        meds=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
    ):
        batch = patient_query.shape[0]
        device = patient_query.device
        selected = torch.zeros(batch, self.m_voc_size, dtype=torch.bool, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        hidden = self._initial_hidden(patient_query)
        context = self.context_norm(patient_query).unsqueeze(1)
        current_token = torch.full((batch, 1), self.sos_token, dtype=torch.long, device=device)
        for _ in range(max_len):
            decoder_input = self.input_norm(self.token_embedding(current_token) + context)
            decoder_output, hidden = self.gru(decoder_input, hidden)
            seq_logits = self.out(decoder_output)
            step_logits = self._decode_logits(
                seq_logits,
                decoder_output,
                med_logits,
                patient_query,
                meds=meds,
                retrieval_meds=retrieval_meds,
                retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
            ).squeeze(1)
            step_logits[:, self.end_token] = step_logits[:, self.end_token] + end_bias
            step_logits[:, self.sos_token] = -1e4
            step_logits[:, :self.m_voc_size] = step_logits[:, :self.m_voc_size].masked_fill(selected, -1e4)
            if finished.any():
                step_logits[finished, :] = -1e4
                step_logits[finished, self.end_token] = 0.0
            next_token = torch.argmax(step_logits, dim=-1)
            is_med = next_token < self.m_voc_size
            if is_med.any():
                selected[is_med, next_token[is_med]] = True
            finished = finished | (next_token == self.end_token)
            current_token = next_token.unsqueeze(1)
            if bool(finished.all()):
                break
        return selected

    def hidr_decode(
        self,
        patient_query,
        med_logits,
        max_len,
        end_bias=0.0,
        meds=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
    ):
        batch = patient_query.shape[0]
        device = patient_query.device
        hidden = self._initial_hidden(patient_query)
        context = self.context_norm(patient_query).unsqueeze(1)
        current_token = torch.full((batch, 1), self.sos_token, dtype=torch.long, device=device)
        step_logits = []
        for _ in range(max_len):
            decoder_input = self.input_norm(self.token_embedding(current_token) + context)
            decoder_output, hidden = self.gru(decoder_input, hidden)
            seq_logits = self.out(decoder_output)
            logits = self._decode_logits(
                seq_logits,
                decoder_output,
                med_logits,
                patient_query,
                meds=meds,
                retrieval_meds=retrieval_meds,
                retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
            ).squeeze(1)
            logits[:, self.end_token] = logits[:, self.end_token] + end_bias
            step_logits.append(logits)
            current_token = torch.argmax(logits, dim=-1).unsqueeze(1)

        logits = torch.stack(step_logits, dim=1)
        sorted_indexes = torch.argsort(logits, dim=-1, descending=True)
        selected = torch.zeros(batch, self.m_voc_size, dtype=torch.bool, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for step in range(max_len):
            for row_idx in range(batch):
                if bool(finished[row_idx]):
                    continue
                for token in sorted_indexes[row_idx, step]:
                    token_idx = int(token.item())
                    if token_idx == self.sos_token or token_idx == self.end_token:
                        finished[row_idx] = True
                        break
                    if token_idx < self.m_voc_size and not bool(selected[row_idx, token_idx]):
                        selected[row_idx, token_idx] = True
                        break
        return selected

    def rank_decode(
        self,
        patient_query,
        med_logits,
        max_len,
        meds=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
    ):
        batch = patient_query.shape[0]
        device = patient_query.device
        selected = torch.zeros(batch, self.m_voc_size, dtype=torch.bool, device=device)
        rankings = torch.full((batch, max_len), -1, dtype=torch.long, device=device)
        hidden = self._initial_hidden(patient_query)
        context = self.context_norm(patient_query).unsqueeze(1)
        current_token = torch.full((batch, 1), self.sos_token, dtype=torch.long, device=device)
        for step in range(max_len):
            decoder_input = self.input_norm(self.token_embedding(current_token) + context)
            decoder_output, hidden = self.gru(decoder_input, hidden)
            seq_logits = self.out(decoder_output)
            step_logits = self._decode_logits(
                seq_logits,
                decoder_output,
                med_logits,
                patient_query,
                meds=meds,
                retrieval_meds=retrieval_meds,
                retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
            ).squeeze(1)
            med_step_logits = step_logits[:, :self.m_voc_size].masked_fill(selected, -1e4)
            next_token = torch.argmax(med_step_logits, dim=-1)
            rankings[:, step] = next_token
            selected[torch.arange(batch, device=device), next_token] = True
            current_token = next_token.unsqueeze(1)
        return rankings

    def prob_decode(
        self,
        patient_query,
        med_logits,
        max_len,
        end_bias=0.0,
        aggregate='max',
        meds=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
    ):
        batch = patient_query.shape[0]
        device = patient_query.device
        selected = torch.zeros(batch, self.m_voc_size, dtype=torch.bool, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        hidden = self._initial_hidden(patient_query)
        context = self.context_norm(patient_query).unsqueeze(1)
        current_token = torch.full((batch, 1), self.sos_token, dtype=torch.long, device=device)
        step_probs = []
        for _ in range(max_len):
            decoder_input = self.input_norm(self.token_embedding(current_token) + context)
            decoder_output, hidden = self.gru(decoder_input, hidden)
            seq_logits = self.out(decoder_output)
            step_logits = self._decode_logits(
                seq_logits,
                decoder_output,
                med_logits,
                patient_query,
                meds=meds,
                retrieval_meds=retrieval_meds,
                retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
            ).squeeze(1)
            step_logits[:, self.end_token] = step_logits[:, self.end_token] + end_bias
            step_probs.append(torch.softmax(step_logits, dim=-1)[:, :self.m_voc_size])
            decode_logits = step_logits.clone()
            decode_logits[:, self.sos_token] = -1e4
            decode_logits[:, :self.m_voc_size] = decode_logits[:, :self.m_voc_size].masked_fill(selected, -1e4)
            if finished.any():
                decode_logits[finished, :] = -1e4
                decode_logits[finished, self.end_token] = 0.0
            next_token = torch.argmax(decode_logits, dim=-1)
            is_med = next_token < self.m_voc_size
            if is_med.any():
                selected[is_med, next_token[is_med]] = True
            finished = finished | (next_token == self.end_token)
            current_token = next_token.unsqueeze(1)
            if bool(finished.all()):
                break
        probs = torch.stack(step_probs, dim=1)
        if aggregate == 'mean':
            return probs.mean(dim=1)
        if aggregate != 'max':
            raise ValueError(f'Unsupported sequence probability aggregate: {aggregate}')
        return probs.max(dim=1).values


class HidrMemorySequenceDecoder(nn.Module):
    def __init__(
        self,
        state_dim,
        visit_dim,
        m_voc_size,
        dropout=0.1,
        gumbel_tau=0.6,
        att_tau=20.0,
        gate_init=1.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.visit_dim = visit_dim
        self.m_voc_size = m_voc_size
        self.gumbel_tau = gumbel_tau
        self.att_tau = att_tau
        self.sos_token = m_voc_size
        self.end_token = m_voc_size + 1
        self.vocab_size = m_voc_size + 2
        self.token_embedding = nn.Embedding(self.vocab_size, state_dim)
        self.context_norm = nn.LayerNorm(state_dim)
        self.input_norm = nn.LayerNorm(state_dim)
        self.init_hidden = nn.Linear(state_dim, state_dim)
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(state_dim, self.vocab_size)
        self.base_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.visit_proj = nn.Linear(visit_dim, state_dim)
        self.retrieval_query_proj = nn.Linear(state_dim, state_dim)
        self.memory_query_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.memory_key_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.gumbel_selector = nn.Linear(state_dim * 2, 1)
        self.copy_query = nn.Linear(state_dim, state_dim)
        self.copy_gate = nn.Linear(state_dim, 1)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.constant_(self.copy_gate.bias, gate_init)

    def _initial_hidden(self, patient_query):
        return torch.tanh(self.init_hidden(self.context_norm(patient_query))).unsqueeze(0)

    def _add_base_logits(self, seq_logits, med_logits):
        med_part = seq_logits[..., :self.m_voc_size] + self.base_logit_scale * med_logits.unsqueeze(1)
        special_part = seq_logits[..., self.m_voc_size:]
        return torch.cat((med_part, special_part), dim=-1)

    def _memory_tensors(self, patient_query, visit_context, meds, retrieval_meds=None, retrieval_scores=None, retrieval_queries=None):
        device = patient_query.device
        dtype = patient_query.dtype
        visit_states = self.visit_proj(visit_context.to(device=device, dtype=dtype))
        current_state = self.context_norm(patient_query)
        if meds is None:
            history_meds = visit_states.new_zeros(visit_states.size(0), 0, self.m_voc_size)
        else:
            history_meds = meds[:, 1:, :].to(device=device, dtype=dtype)
        history_len = min(max(visit_states.size(1) - 1, 0), history_meds.size(1))
        history_states = visit_states[:, 1:1 + history_len, :]
        history_meds = history_meds[:, :history_len, :]
        valid_history = history_meds.sum(dim=-1) > 0
        if history_states.size(1) > 0:
            current_for_history = current_state.unsqueeze(1).expand(-1, history_states.size(1), -1)
            selector_input = torch.cat((history_states, current_for_history), dim=-1)
            keep_logit = self.gumbel_selector(selector_input).squeeze(-1)
            gumbel_input = torch.stack((keep_logit, -keep_logit), dim=-1)
            if self.training:
                keep = F.gumbel_softmax(gumbel_input, tau=self.gumbel_tau, hard=True, dim=-1)[..., 0]
            else:
                keep = (keep_logit >= 0).to(dtype=dtype)
            keep = keep * valid_history.to(dtype=dtype)
        else:
            keep = visit_states.new_zeros(visit_states.size(0), 0)
        if retrieval_meds is not None:
            retrieval_meds = retrieval_meds.to(device=device, dtype=dtype)
            retrieval_valid = retrieval_meds.sum(dim=-1) > 0
            if retrieval_queries is not None:
                retrieval_states = retrieval_queries.to(device=device, dtype=dtype)
                if retrieval_states.size(-1) != self.state_dim:
                    raise ValueError("retrieval query dimension must match HI-DR memory state_dim")
                retrieval_states = self.retrieval_query_proj(retrieval_states)
            else:
                retrieval_states = visit_states.new_zeros(retrieval_meds.size(0), retrieval_meds.size(1), self.state_dim)
            retrieval_bias = (
                visit_states.new_zeros(retrieval_meds.size(0), retrieval_meds.size(1))
                if retrieval_scores is None
                else retrieval_scores.to(device=device, dtype=dtype)
            )
        else:
            batch = visit_states.size(0)
            retrieval_meds = visit_states.new_zeros(batch, 0, self.m_voc_size)
            retrieval_states = visit_states.new_zeros(batch, 0, self.state_dim)
            retrieval_valid = torch.zeros(batch, 0, dtype=torch.bool, device=device)
            retrieval_bias = visit_states.new_zeros(batch, 0)
        memory_states = torch.cat((history_states, retrieval_states), dim=1)
        memory_meds = torch.cat((history_meds, retrieval_meds), dim=1)
        memory_valid = torch.cat((valid_history & (keep > 0), retrieval_valid), dim=1)
        memory_bias = torch.cat((torch.zeros_like(keep), retrieval_bias), dim=1)
        if memory_states.size(1) == 0:
            return memory_meds, memory_valid, memory_bias
        query = self.memory_query_proj(current_state).unsqueeze(1)
        keys = self.memory_key_proj(memory_states)
        visit_logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.state_dim)
        visit_logits = visit_logits + memory_bias
        visit_logits = visit_logits.masked_fill(~memory_valid, -1e4)
        visit_attn = torch.softmax(visit_logits / self.att_tau, dim=-1)
        visit_attn = torch.where(memory_valid, visit_attn, torch.zeros_like(visit_attn))
        return memory_meds, memory_valid, visit_attn

    def _decode_logits(self, seq_logits, decoder_output, med_logits, memory_meds, memory_valid, visit_attn):
        base_logits = self._add_base_logits(seq_logits, med_logits)
        if memory_meds.size(1) == 0:
            return base_logits
        med_prior = self._memory_med_prior(memory_meds, visit_attn)
        has_memory = memory_valid.any(dim=1)
        if not bool(has_memory.any()):
            return base_logits
        med_embeddings = self.token_embedding.weight[:self.m_voc_size]
        copy_query = self.copy_query(decoder_output)
        copy_scores = torch.matmul(copy_query, med_embeddings.transpose(0, 1)) / math.sqrt(self.state_dim)
        copy_scores = copy_scores + torch.log(med_prior.clamp_min(1e-8)).unsqueeze(1)
        copy_scores = copy_scores.masked_fill(med_prior.unsqueeze(1) <= 0, -1e4)
        copy_prob = torch.softmax(copy_scores, dim=-1)
        copy_prob = torch.where(has_memory.view(-1, 1, 1), copy_prob, torch.zeros_like(copy_prob))
        copy_full_prob = torch.zeros_like(base_logits)
        copy_full_prob[..., :self.m_voc_size] = copy_prob
        generate_gate = torch.sigmoid(self.copy_gate(decoder_output))
        generate_gate = torch.where(has_memory.view(-1, 1, 1), generate_gate, torch.ones_like(generate_gate))
        generate_prob = torch.softmax(base_logits, dim=-1)
        mixed_prob = generate_gate * generate_prob + (1.0 - generate_gate) * copy_full_prob
        return torch.log(mixed_prob.clamp_min(1e-12))

    def _memory_med_prior(self, memory_meds, visit_attn):
        if memory_meds.size(1) == 0:
            return memory_meds.new_zeros(memory_meds.size(0), self.m_voc_size)
        return (visit_attn.unsqueeze(-1) * memory_meds).sum(dim=1).clamp(0.0, 1.0)

    def forward(self, patient_query, visit_context, med_logits, decoder_inputs, meds=None, retrieval_meds=None, retrieval_scores=None, retrieval_queries=None):
        memory_meds, memory_valid, visit_attn = self._memory_tensors(
            patient_query, visit_context, meds, retrieval_meds=retrieval_meds, retrieval_scores=retrieval_scores, retrieval_queries=retrieval_queries
        )
        context = self.context_norm(patient_query).unsqueeze(1)
        decoder_input = self.input_norm(self.token_embedding(decoder_inputs) + context)
        hidden = self._initial_hidden(patient_query)
        decoder_output, _ = self.gru(decoder_input, hidden)
        seq_logits = self.out(self.drop(decoder_output))
        return self._decode_logits(seq_logits, decoder_output, med_logits, memory_meds, memory_valid, visit_attn)

    def _step_logits(self, current_token, hidden, context, med_logits, memory_meds, memory_valid, visit_attn):
        decoder_input = self.input_norm(self.token_embedding(current_token) + context)
        decoder_output, hidden = self.gru(decoder_input, hidden)
        seq_logits = self.out(decoder_output)
        logits = self._decode_logits(seq_logits, decoder_output, med_logits, memory_meds, memory_valid, visit_attn)
        return logits.squeeze(1), hidden

    def greedy_decode(self, patient_query, visit_context, med_logits, max_len, end_bias=0.0, meds=None, retrieval_meds=None, retrieval_scores=None, retrieval_queries=None):
        memory_meds, memory_valid, visit_attn = self._memory_tensors(
            patient_query, visit_context, meds, retrieval_meds=retrieval_meds, retrieval_scores=retrieval_scores, retrieval_queries=retrieval_queries
        )
        batch = patient_query.shape[0]
        device = patient_query.device
        selected = torch.zeros(batch, self.m_voc_size, dtype=torch.bool, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        hidden = self._initial_hidden(patient_query)
        context = self.context_norm(patient_query).unsqueeze(1)
        current_token = torch.full((batch, 1), self.sos_token, dtype=torch.long, device=device)
        for _ in range(max_len):
            step_logits, hidden = self._step_logits(current_token, hidden, context, med_logits, memory_meds, memory_valid, visit_attn)
            step_logits[:, self.end_token] = step_logits[:, self.end_token] + end_bias
            step_logits[:, self.sos_token] = -1e4
            step_logits[:, :self.m_voc_size] = step_logits[:, :self.m_voc_size].masked_fill(selected, -1e4)
            if finished.any():
                step_logits[finished, :] = -1e4
                step_logits[finished, self.end_token] = 0.0
            next_token = torch.argmax(step_logits, dim=-1)
            is_med = next_token < self.m_voc_size
            if is_med.any():
                selected[is_med, next_token[is_med]] = True
            finished = finished | (next_token == self.end_token)
            current_token = next_token.unsqueeze(1)
            if bool(finished.all()):
                break
        return selected

    def hidr_decode(self, patient_query, visit_context, med_logits, max_len, end_bias=0.0, meds=None, retrieval_meds=None, retrieval_scores=None, retrieval_queries=None):
        memory_meds, memory_valid, visit_attn = self._memory_tensors(
            patient_query, visit_context, meds, retrieval_meds=retrieval_meds, retrieval_scores=retrieval_scores, retrieval_queries=retrieval_queries
        )
        batch = patient_query.shape[0]
        device = patient_query.device
        hidden = self._initial_hidden(patient_query)
        context = self.context_norm(patient_query).unsqueeze(1)
        current_token = torch.full((batch, 1), self.sos_token, dtype=torch.long, device=device)
        step_logits = []
        for _ in range(max_len):
            logits, hidden = self._step_logits(current_token, hidden, context, med_logits, memory_meds, memory_valid, visit_attn)
            logits[:, self.end_token] = logits[:, self.end_token] + end_bias
            step_logits.append(logits)
            current_token = torch.argmax(logits, dim=-1).unsqueeze(1)
        logits = torch.stack(step_logits, dim=1)
        sorted_indexes = torch.argsort(logits, dim=-1, descending=True)
        selected = torch.zeros(batch, self.m_voc_size, dtype=torch.bool, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for step in range(max_len):
            for row_idx in range(batch):
                if bool(finished[row_idx]):
                    continue
                for token in sorted_indexes[row_idx, step]:
                    token_idx = int(token.item())
                    if token_idx == self.sos_token or token_idx == self.end_token:
                        finished[row_idx] = True
                        break
                    if token_idx < self.m_voc_size and not bool(selected[row_idx, token_idx]):
                        selected[row_idx, token_idx] = True
                        break
        return selected

    def rank_decode(self, patient_query, visit_context, med_logits, max_len, meds=None, retrieval_meds=None, retrieval_scores=None, retrieval_queries=None):
        memory_meds, memory_valid, visit_attn = self._memory_tensors(
            patient_query, visit_context, meds, retrieval_meds=retrieval_meds, retrieval_scores=retrieval_scores, retrieval_queries=retrieval_queries
        )
        batch = patient_query.shape[0]
        device = patient_query.device
        selected = torch.zeros(batch, self.m_voc_size, dtype=torch.bool, device=device)
        rankings = torch.full((batch, max_len), -1, dtype=torch.long, device=device)
        hidden = self._initial_hidden(patient_query)
        context = self.context_norm(patient_query).unsqueeze(1)
        current_token = torch.full((batch, 1), self.sos_token, dtype=torch.long, device=device)
        for step in range(max_len):
            step_logits, hidden = self._step_logits(current_token, hidden, context, med_logits, memory_meds, memory_valid, visit_attn)
            med_step_logits = step_logits[:, :self.m_voc_size].masked_fill(selected, -1e4)
            next_token = torch.argmax(med_step_logits, dim=-1)
            rankings[:, step] = next_token
            selected[torch.arange(batch, device=device), next_token] = True
            current_token = next_token.unsqueeze(1)
        return rankings

    def prob_decode(self, patient_query, visit_context, med_logits, max_len, end_bias=0.0, aggregate='max', meds=None, retrieval_meds=None, retrieval_scores=None, retrieval_queries=None):
        memory_meds, memory_valid, visit_attn = self._memory_tensors(
            patient_query, visit_context, meds, retrieval_meds=retrieval_meds, retrieval_scores=retrieval_scores, retrieval_queries=retrieval_queries
        )
        batch = patient_query.shape[0]
        device = patient_query.device
        selected = torch.zeros(batch, self.m_voc_size, dtype=torch.bool, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        hidden = self._initial_hidden(patient_query)
        context = self.context_norm(patient_query).unsqueeze(1)
        current_token = torch.full((batch, 1), self.sos_token, dtype=torch.long, device=device)
        step_probs = []
        for _ in range(max_len):
            step_logits, hidden = self._step_logits(current_token, hidden, context, med_logits, memory_meds, memory_valid, visit_attn)
            step_logits[:, self.end_token] = step_logits[:, self.end_token] + end_bias
            step_probs.append(torch.softmax(step_logits, dim=-1)[:, :self.m_voc_size])
            decode_logits = step_logits.clone()
            decode_logits[:, self.sos_token] = -1e4
            decode_logits[:, :self.m_voc_size] = decode_logits[:, :self.m_voc_size].masked_fill(selected, -1e4)
            if finished.any():
                decode_logits[finished, :] = -1e4
                decode_logits[finished, self.end_token] = 0.0
            next_token = torch.argmax(decode_logits, dim=-1)
            is_med = next_token < self.m_voc_size
            if is_med.any():
                selected[is_med, next_token[is_med]] = True
            finished = finished | (next_token == self.end_token)
            current_token = next_token.unsqueeze(1)
            if bool(finished.all()):
                break
        probs = torch.stack(step_probs, dim=1)
        if aggregate == 'mean':
            return probs.mean(dim=1)
        if aggregate != 'max':
            raise ValueError(f'Unsupported sequence probability aggregate: {aggregate}')
        return probs.max(dim=1).values


class HidrTransformerSequenceDecoder(nn.Module):
    def __init__(
        self,
        state_dim,
        visit_dim,
        m_voc_size,
        hidden_dim=0,
        nhead=4,
        num_layers=1,
        dropout=0.1,
        gumbel_tau=0.6,
        att_tau=20.0,
        gate_init=1.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.visit_dim = visit_dim
        self.m_voc_size = m_voc_size
        self.hidden_dim = hidden_dim or visit_dim
        self.gumbel_tau = gumbel_tau
        self.att_tau = att_tau
        self.sos_token = m_voc_size
        self.end_token = m_voc_size + 1
        self.vocab_size = m_voc_size + 2
        if self.hidden_dim % nhead != 0:
            raise ValueError("seq transformer hidden_dim must be divisible by nhead")

        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.context_proj = nn.Linear(state_dim, self.hidden_dim)
        self.visit_proj = nn.Linear(visit_dim, self.hidden_dim)
        self.drug_memory_proj = nn.Linear(visit_dim, self.hidden_dim)
        self.retrieval_query_proj = nn.Linear(state_dim, self.hidden_dim)
        self.context_norm = nn.LayerNorm(self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.memory_norm = nn.LayerNorm(self.hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=nhead,
            dim_feedforward=self.hidden_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(self.hidden_dim, self.vocab_size)
        self.base_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.memory_query_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.memory_key_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.gumbel_selector = nn.Linear(self.hidden_dim * 2, 1)
        self.copy_query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.copy_gate = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.constant_(self.copy_gate.bias, gate_init)

    def _add_base_logits(self, seq_logits, med_logits):
        med_part = seq_logits[..., :self.m_voc_size] + self.base_logit_scale * med_logits.unsqueeze(1)
        special_part = seq_logits[..., self.m_voc_size:]
        return torch.cat((med_part, special_part), dim=-1)

    def _memory_tensors(self, patient_query, visit_context, meds, retrieval_meds=None, retrieval_scores=None, retrieval_queries=None):
        device = patient_query.device
        dtype = patient_query.dtype
        visit_states = self.memory_norm(self.visit_proj(visit_context.to(device=device, dtype=dtype)))
        current_state = self.context_norm(self.context_proj(patient_query))
        if meds is None:
            history_meds = visit_states.new_zeros(visit_states.size(0), 0, self.m_voc_size)
        else:
            history_meds = meds[:, 1:, :].to(device=device, dtype=dtype)
        history_len = min(max(visit_states.size(1) - 1, 0), history_meds.size(1))
        history_states = visit_states[:, 1:1 + history_len, :]
        history_meds = history_meds[:, :history_len, :]
        valid_history = history_meds.sum(dim=-1) > 0
        if history_states.size(1) > 0:
            current_for_history = current_state.unsqueeze(1).expand(-1, history_states.size(1), -1)
            selector_input = torch.cat((history_states, current_for_history), dim=-1)
            keep_logit = self.gumbel_selector(selector_input).squeeze(-1)
            gumbel_input = torch.stack((keep_logit, -keep_logit), dim=-1)
            if self.training:
                keep = F.gumbel_softmax(gumbel_input, tau=self.gumbel_tau, hard=True, dim=-1)[..., 0]
            else:
                keep = (keep_logit >= 0).to(dtype=dtype)
            keep = keep * valid_history.to(dtype=dtype)
        else:
            keep = visit_states.new_zeros(visit_states.size(0), 0)

        if retrieval_meds is not None:
            retrieval_meds = retrieval_meds.to(device=device, dtype=dtype)
            retrieval_valid = retrieval_meds.sum(dim=-1) > 0
            if retrieval_queries is not None:
                retrieval_states = retrieval_queries.to(device=device, dtype=dtype)
                if retrieval_states.size(-1) != self.state_dim:
                    raise ValueError("retrieval query dimension must match HI-DR transformer state_dim")
                retrieval_states = self.retrieval_query_proj(retrieval_states)
            else:
                retrieval_states = visit_states.new_zeros(retrieval_meds.size(0), retrieval_meds.size(1), self.hidden_dim)
            retrieval_bias = (
                visit_states.new_zeros(retrieval_meds.size(0), retrieval_meds.size(1))
                if retrieval_scores is None
                else retrieval_scores.to(device=device, dtype=dtype)
            )
        else:
            batch = visit_states.size(0)
            retrieval_meds = visit_states.new_zeros(batch, 0, self.m_voc_size)
            retrieval_states = visit_states.new_zeros(batch, 0, self.hidden_dim)
            retrieval_valid = torch.zeros(batch, 0, dtype=torch.bool, device=device)
            retrieval_bias = visit_states.new_zeros(batch, 0)

        memory_states = torch.cat((history_states, retrieval_states), dim=1)
        memory_meds = torch.cat((history_meds, retrieval_meds), dim=1)
        memory_valid = torch.cat((valid_history & (keep > 0), retrieval_valid), dim=1)
        memory_bias = torch.cat((torch.zeros_like(keep), retrieval_bias), dim=1)
        if memory_states.size(1) == 0:
            return memory_meds, memory_valid, memory_bias

        query = self.memory_query_proj(current_state).unsqueeze(1)
        keys = self.memory_key_proj(memory_states)
        visit_logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.hidden_dim)
        visit_logits = visit_logits + memory_bias
        visit_logits = visit_logits.masked_fill(~memory_valid, -1e4)
        visit_attn = torch.softmax(visit_logits / self.att_tau, dim=-1)
        visit_attn = torch.where(memory_valid, visit_attn, torch.zeros_like(visit_attn))
        return memory_meds, memory_valid, visit_attn

    def _memory_med_prior(self, memory_meds, visit_attn):
        if memory_meds.size(1) == 0:
            return memory_meds.new_zeros(memory_meds.size(0), self.m_voc_size)
        return (visit_attn.unsqueeze(-1) * memory_meds).sum(dim=1).clamp(0.0, 1.0)

    def _causal_mask(self, length, device):
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def _token_embeddings(self, decoder_inputs, drug_memory=None):
        token_embeddings = self.token_embedding(decoder_inputs)
        if drug_memory is None:
            return token_embeddings
        drug_memory = drug_memory.to(device=decoder_inputs.device, dtype=token_embeddings.dtype)
        if drug_memory.dim() == 2:
            drug_memory = self.drug_memory_proj(drug_memory)
            special_memory = drug_memory.new_zeros(2, self.hidden_dim)
            drug_memory = torch.cat((drug_memory, special_memory), dim=0)
            return token_embeddings + drug_memory[decoder_inputs]
        if drug_memory.dim() == 3:
            batch, _, _ = drug_memory.shape
            drug_memory = self.drug_memory_proj(drug_memory)
            special_memory = drug_memory.new_zeros(batch, 2, self.hidden_dim)
            drug_memory = torch.cat((drug_memory, special_memory), dim=1)
            return token_embeddings + drug_memory.gather(
                1,
                decoder_inputs.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
            )
        raise ValueError("drug_memory must have shape [med, dim] or [batch, med, dim]")

    def _decode_hidden(
        self,
        patient_query,
        visit_context,
        decoder_inputs,
        drug_memory=None,
        code_context=None,
        code_context_mask=None,
    ):
        current_state = self.context_norm(self.context_proj(patient_query)).unsqueeze(1)
        decoder_input = self.input_norm(self._token_embeddings(decoder_inputs, drug_memory) + current_state)
        visit_context = visit_context.to(device=patient_query.device, dtype=patient_query.dtype)
        memory = self.memory_norm(self.visit_proj(visit_context))
        visit_padding_mask = visit_context.abs().sum(dim=-1) <= 1e-8
        memory_padding_mask = visit_padding_mask
        if code_context is not None:
            code_context = code_context.to(device=patient_query.device, dtype=patient_query.dtype)
            code_memory = self.memory_norm(self.visit_proj(code_context))
            memory = torch.cat((memory, code_memory), dim=1)
            if code_context_mask is None:
                code_padding_mask = code_context.abs().sum(dim=-1) <= 1e-8
            else:
                code_padding_mask = code_context_mask.to(device=patient_query.device, dtype=torch.bool)
                empty_rows = code_padding_mask.all(dim=1)
                if bool(empty_rows.any()):
                    code_padding_mask = code_padding_mask.clone()
                    code_padding_mask[empty_rows, 0] = False
            memory_padding_mask = torch.cat((visit_padding_mask, code_padding_mask), dim=1)
        empty_memory_rows = memory_padding_mask.all(dim=1)
        if bool(empty_memory_rows.any()):
            memory_padding_mask = memory_padding_mask.clone()
            memory_padding_mask[empty_memory_rows, 0] = False
        tgt_mask = self._causal_mask(decoder_inputs.size(1), decoder_inputs.device)
        return self.decoder(
            decoder_input,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

    def _decode_logits(self, decoder_hidden, med_logits, memory_meds, memory_valid, visit_attn):
        seq_logits = self.out(self.drop(decoder_hidden))
        base_logits = self._add_base_logits(seq_logits, med_logits)
        if memory_meds.size(1) == 0:
            return base_logits
        has_memory = memory_valid.any(dim=1)
        if not bool(has_memory.any()):
            return base_logits

        med_embeddings = self.token_embedding.weight[:self.m_voc_size]
        copy_query = self.copy_query(decoder_hidden)
        token_scores = torch.matmul(copy_query, med_embeddings.transpose(0, 1)) / math.sqrt(self.hidden_dim)
        visit_log_weight = torch.log(visit_attn.clamp_min(1e-8))
        pair_logits = token_scores.unsqueeze(2) + visit_log_weight.unsqueeze(1).unsqueeze(-1)
        pair_mask = memory_valid.unsqueeze(1).unsqueeze(-1) & (memory_meds.unsqueeze(1) > 0)
        pair_logits = pair_logits.masked_fill(~pair_mask, -1e4)
        copy_pair_prob = torch.softmax(pair_logits.reshape(pair_logits.size(0), pair_logits.size(1), -1), dim=-1)
        copy_pair_prob = torch.where(
            has_memory.view(-1, 1, 1),
            copy_pair_prob,
            torch.zeros_like(copy_pair_prob),
        )
        copy_prob = copy_pair_prob.view(
            pair_logits.size(0),
            pair_logits.size(1),
            pair_logits.size(2),
            pair_logits.size(3),
        ).sum(dim=2)

        copy_full_prob = torch.zeros_like(base_logits)
        copy_full_prob[..., :self.m_voc_size] = copy_prob
        generate_gate = torch.sigmoid(self.copy_gate(decoder_hidden))
        generate_gate = torch.where(has_memory.view(-1, 1, 1), generate_gate, torch.ones_like(generate_gate))
        generate_prob = torch.softmax(base_logits, dim=-1)
        mixed_prob = generate_gate * generate_prob + (1.0 - generate_gate) * copy_full_prob
        return torch.log(mixed_prob.clamp_min(1e-12))

    def forward(
        self,
        patient_query,
        visit_context,
        med_logits,
        decoder_inputs,
        meds=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
        drug_memory=None,
        code_context=None,
        code_context_mask=None,
    ):
        memory_meds, memory_valid, visit_attn = self._memory_tensors(
            patient_query,
            visit_context,
            meds,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
        )
        decoder_hidden = self._decode_hidden(
            patient_query,
            visit_context,
            decoder_inputs,
            drug_memory=drug_memory,
            code_context=code_context,
            code_context_mask=code_context_mask,
        )
        return self._decode_logits(decoder_hidden, med_logits, memory_meds, memory_valid, visit_attn)

    def _step_logits(
        self,
        tokens,
        patient_query,
        visit_context,
        med_logits,
        memory_meds,
        memory_valid,
        visit_attn,
        drug_memory=None,
        code_context=None,
        code_context_mask=None,
    ):
        decoder_hidden = self._decode_hidden(
            patient_query,
            visit_context,
            tokens,
            drug_memory=drug_memory,
            code_context=code_context,
            code_context_mask=code_context_mask,
        )
        logits = self._decode_logits(decoder_hidden, med_logits, memory_meds, memory_valid, visit_attn)
        return logits[:, -1, :]

    def greedy_decode(
        self,
        patient_query,
        visit_context,
        med_logits,
        max_len,
        end_bias=0.0,
        meds=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
        drug_memory=None,
        code_context=None,
        code_context_mask=None,
    ):
        memory_meds, memory_valid, visit_attn = self._memory_tensors(
            patient_query,
            visit_context,
            meds,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
        )
        batch = patient_query.shape[0]
        device = patient_query.device
        selected = torch.zeros(batch, self.m_voc_size, dtype=torch.bool, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        tokens = torch.full((batch, 1), self.sos_token, dtype=torch.long, device=device)
        for _ in range(max_len):
            step_logits = self._step_logits(
                tokens,
                patient_query,
                visit_context,
                med_logits,
                memory_meds,
                memory_valid,
                visit_attn,
                drug_memory=drug_memory,
                code_context=code_context,
                code_context_mask=code_context_mask,
            )
            step_logits[:, self.end_token] = step_logits[:, self.end_token] + end_bias
            step_logits[:, self.sos_token] = -1e4
            step_logits[:, :self.m_voc_size] = step_logits[:, :self.m_voc_size].masked_fill(selected, -1e4)
            if finished.any():
                step_logits[finished, :] = -1e4
                step_logits[finished, self.end_token] = 0.0
            next_token = torch.argmax(step_logits, dim=-1)
            is_med = next_token < self.m_voc_size
            if is_med.any():
                selected[is_med, next_token[is_med]] = True
            finished = finished | (next_token == self.end_token)
            tokens = torch.cat((tokens, next_token.unsqueeze(1)), dim=1)
            if bool(finished.all()):
                break
        return selected

    def hidr_decode(
        self,
        patient_query,
        visit_context,
        med_logits,
        max_len,
        end_bias=0.0,
        meds=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
        drug_memory=None,
        code_context=None,
        code_context_mask=None,
    ):
        memory_meds, memory_valid, visit_attn = self._memory_tensors(
            patient_query,
            visit_context,
            meds,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
        )
        batch = patient_query.shape[0]
        device = patient_query.device
        tokens = torch.full((batch, 1), self.sos_token, dtype=torch.long, device=device)
        step_logits = []
        for _ in range(max_len):
            logits = self._step_logits(
                tokens,
                patient_query,
                visit_context,
                med_logits,
                memory_meds,
                memory_valid,
                visit_attn,
                drug_memory=drug_memory,
                code_context=code_context,
                code_context_mask=code_context_mask,
            )
            logits[:, self.end_token] = logits[:, self.end_token] + end_bias
            step_logits.append(logits)
            tokens = torch.cat((tokens, torch.argmax(logits, dim=-1).unsqueeze(1)), dim=1)

        logits = torch.stack(step_logits, dim=1)
        sorted_indexes = torch.argsort(logits, dim=-1, descending=True)
        selected = torch.zeros(batch, self.m_voc_size, dtype=torch.bool, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for step in range(max_len):
            for row_idx in range(batch):
                if bool(finished[row_idx]):
                    continue
                for token in sorted_indexes[row_idx, step]:
                    token_idx = int(token.item())
                    if token_idx == self.sos_token or token_idx == self.end_token:
                        finished[row_idx] = True
                        break
                    if token_idx < self.m_voc_size and not bool(selected[row_idx, token_idx]):
                        selected[row_idx, token_idx] = True
                        break
        return selected

    def rank_decode(
        self,
        patient_query,
        visit_context,
        med_logits,
        max_len,
        meds=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
        drug_memory=None,
        code_context=None,
        code_context_mask=None,
    ):
        memory_meds, memory_valid, visit_attn = self._memory_tensors(
            patient_query,
            visit_context,
            meds,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
            )
        batch = patient_query.shape[0]
        device = patient_query.device
        selected = torch.zeros(batch, self.m_voc_size, dtype=torch.bool, device=device)
        rankings = torch.full((batch, max_len), -1, dtype=torch.long, device=device)
        tokens = torch.full((batch, 1), self.sos_token, dtype=torch.long, device=device)
        for step in range(max_len):
            step_logits = self._step_logits(
                tokens,
                patient_query,
                visit_context,
                med_logits,
                memory_meds,
                memory_valid,
                visit_attn,
                drug_memory=drug_memory,
                code_context=code_context,
                code_context_mask=code_context_mask,
            )
            med_step_logits = step_logits[:, :self.m_voc_size].masked_fill(selected, -1e4)
            next_token = torch.argmax(med_step_logits, dim=-1)
            rankings[:, step] = next_token
            selected[torch.arange(batch, device=device), next_token] = True
            tokens = torch.cat((tokens, next_token.unsqueeze(1)), dim=1)
        return rankings

    def prob_decode(
        self,
        patient_query,
        visit_context,
        med_logits,
        max_len,
        end_bias=0.0,
        aggregate='max',
        meds=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
        drug_memory=None,
        code_context=None,
        code_context_mask=None,
    ):
        memory_meds, memory_valid, visit_attn = self._memory_tensors(
            patient_query,
            visit_context,
            meds,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
        )
        batch = patient_query.shape[0]
        device = patient_query.device
        selected = torch.zeros(batch, self.m_voc_size, dtype=torch.bool, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        tokens = torch.full((batch, 1), self.sos_token, dtype=torch.long, device=device)
        step_probs = []
        for _ in range(max_len):
            step_logits = self._step_logits(
                tokens,
                patient_query,
                visit_context,
                med_logits,
                memory_meds,
                memory_valid,
                visit_attn,
                drug_memory=drug_memory,
                code_context=code_context,
                code_context_mask=code_context_mask,
            )
            step_logits[:, self.end_token] = step_logits[:, self.end_token] + end_bias
            step_probs.append(torch.softmax(step_logits, dim=-1)[:, :self.m_voc_size])
            decode_logits = step_logits.clone()
            decode_logits[:, self.sos_token] = -1e4
            decode_logits[:, :self.m_voc_size] = decode_logits[:, :self.m_voc_size].masked_fill(selected, -1e4)
            if finished.any():
                decode_logits[finished, :] = -1e4
                decode_logits[finished, self.end_token] = 0.0
            next_token = torch.argmax(decode_logits, dim=-1)
            is_med = next_token < self.m_voc_size
            if is_med.any():
                selected[is_med, next_token[is_med]] = True
            finished = finished | (next_token == self.end_token)
            tokens = torch.cat((tokens, next_token.unsqueeze(1)), dim=1)
            if bool(finished.all()):
                break
        probs = torch.stack(step_probs, dim=1)
        if aggregate == 'mean':
            return probs.mean(dim=1)
        if aggregate != 'max':
            raise ValueError(f'Unsupported sequence probability aggregate: {aggregate}')
        return probs.max(dim=1).values

    def beam_decode(
        self,
        patient_query,
        visit_context,
        med_logits,
        max_len,
        beam_size=4,
        end_bias=0.0,
        meds=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
        drug_memory=None,
        code_context=None,
        code_context_mask=None,
    ):
        memory_meds, memory_valid, visit_attn = self._memory_tensors(
            patient_query,
            visit_context,
            meds,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
        )
        batch = patient_query.shape[0]
        device = patient_query.device
        selected = torch.zeros(batch, self.m_voc_size, dtype=torch.bool, device=device)
        vocab_size = self.vocab_size
        beam_size = max(int(beam_size), 1)

        tokens = torch.full((batch, beam_size, 1), self.sos_token, dtype=torch.long, device=device)
        scores = patient_query.new_zeros(batch, beam_size)
        finished = torch.zeros(batch, beam_size, dtype=torch.bool, device=device)
        batch_offsets = torch.arange(batch, device=device).unsqueeze(1)

        patient_expanded = patient_query.unsqueeze(1).expand(-1, beam_size, -1).reshape(batch * beam_size, -1)
        visit_expanded = visit_context.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(
            batch * beam_size,
            visit_context.size(1),
            visit_context.size(2),
        )
        med_logits_expanded = med_logits.unsqueeze(1).expand(-1, beam_size, -1).reshape(batch * beam_size, -1)
        memory_meds_expanded = memory_meds.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(
            batch * beam_size,
            memory_meds.size(1),
            memory_meds.size(2),
        )
        memory_valid_expanded = memory_valid.unsqueeze(1).expand(-1, beam_size, -1).reshape(batch * beam_size, -1)
        visit_attn_expanded = visit_attn.unsqueeze(1).expand(-1, beam_size, -1).reshape(batch * beam_size, -1)
        if drug_memory is None or drug_memory.dim() == 2:
            drug_memory_expanded = drug_memory
        else:
            drug_memory_expanded = drug_memory.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(
                batch * beam_size,
                drug_memory.size(1),
                drug_memory.size(2),
            )
        if code_context is None:
            code_context_expanded = None
            code_context_mask_expanded = None
        else:
            code_context_expanded = code_context.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(
                batch * beam_size,
                code_context.size(1),
                code_context.size(2),
            )
            if code_context_mask is None:
                code_context_mask_expanded = None
            else:
                code_context_mask_expanded = code_context_mask.unsqueeze(1).expand(-1, beam_size, -1).reshape(
                    batch * beam_size,
                    code_context_mask.size(1),
                )

        for step in range(max_len):
            flat_tokens = tokens.reshape(batch * beam_size, -1)
            step_logits = self._step_logits(
                flat_tokens,
                patient_expanded,
                visit_expanded,
                med_logits_expanded,
                memory_meds_expanded,
                memory_valid_expanded,
                visit_attn_expanded,
                drug_memory=drug_memory_expanded,
                code_context=code_context_expanded,
                code_context_mask=code_context_mask_expanded,
            ).view(batch, beam_size, vocab_size)
            step_logits[:, :, self.end_token] = step_logits[:, :, self.end_token] + end_bias
            step_log_probs = F.log_softmax(step_logits, dim=-1)
            step_log_probs[:, :, self.sos_token] = -torch.inf

            previous_valid = tokens < self.m_voc_size
            previous_meds = tokens.clamp(min=0, max=max(self.m_voc_size - 1, 0))
            duplicate_counts = step_log_probs.new_zeros(batch, beam_size, self.m_voc_size)
            duplicate_counts.scatter_add_(2, previous_meds, previous_valid.to(dtype=step_log_probs.dtype))
            duplicate_mask = duplicate_counts > 0
            step_log_probs[:, :, :self.m_voc_size] = step_log_probs[:, :, :self.m_voc_size].masked_fill(duplicate_mask, -torch.inf)

            active = ~finished
            if step == 0 and beam_size > 1:
                active[:, 1:] = False
            step_log_probs = step_log_probs.masked_fill(~active.unsqueeze(-1), -torch.inf)
            active_scores = (step_log_probs + scores.unsqueeze(-1)).reshape(batch, beam_size * vocab_size)
            finished_scores = torch.where(finished, scores, scores.new_full(scores.shape, -torch.inf))
            candidate_scores = torch.cat((active_scores, finished_scores), dim=1)

            top_scores, top_ids = torch.topk(candidate_scores, k=beam_size, dim=1)
            is_finished_carry = top_ids >= beam_size * vocab_size
            source_beams = torch.where(
                is_finished_carry,
                top_ids - beam_size * vocab_size,
                torch.div(top_ids, vocab_size, rounding_mode='floor'),
            )
            next_tokens = torch.where(
                is_finished_carry,
                torch.full_like(top_ids, self.end_token),
                top_ids.remainder(vocab_size),
            )

            gathered_tokens = tokens.gather(1, source_beams.unsqueeze(-1).expand(-1, -1, tokens.size(2)))
            tokens = torch.cat((gathered_tokens, next_tokens.unsqueeze(-1)), dim=2)
            scores = top_scores
            finished = is_finished_carry | (next_tokens == self.end_token)
            if bool(finished.all()):
                break

        best_beams = torch.argmax(scores, dim=1)
        best_tokens = tokens[batch_offsets.squeeze(1), best_beams]
        body = best_tokens[:, 1:]
        stop_mask = (body == self.end_token) | (body == self.sos_token)
        before_stop = stop_mask.cumsum(dim=1) == 0
        med_mask = before_stop & (body < self.m_voc_size)
        for row_idx in range(batch):
            med_indexes = body[row_idx][med_mask[row_idx]]
            if med_indexes.numel() > 0:
                selected[row_idx, med_indexes] = True
        return selected


class SEER(nn.Module):
    def __init__(
        self,
        emb_dim,
        voc_size,
        k,
        ehr_adj,
        ddi_adj,
        temporal_type='mamba',
        patient_temporal_type=None,
        med_temporal_type=None,
        patient_fusion_type='sum',
        patient_query_type='flatten',
        direct_state_source='raw',
        direct_query_gate_init=-3.0,
        med_history_definition='input_order',
        output_mix_weight=0.7,
        output_mix_mode='fixed',
        output_branch_calibration='none',
        med_scorer_type='cosine',
        med_candidate_encoder_type='shared',
        med_fusion_type='fixed_sum',
        multitask_mode='none',
        aux_label_dim=0,
        aux_head_dropout=0.1,
        clinical_feature_dim=0,
        clinical_feature_mode='none',
        clinical_fusion_type='clinical_gated_sum',
        clinical_feature_dropout=0.1,
        clinical_temporal_fusion='add',
        clinical_temporal_gate_init=-3.0,
        retrieval_context_mode='none',
        retrieval_context_target='logit',
        retrieval_context_weight=0.0,
        retrieval_context_dropout=0.1,
        retrieval_context_gate_init=-3.0,
        external_memory_mode='none',
        external_memory_num=32,
        external_memory_hidden_size=0,
        external_memory_weight=0.0,
        external_memory_dropout=0.1,
        external_memory_gate_init=-3.0,
        local_memory_mode='none',
        local_memory_weight=0.0,
        local_memory_dropout=0.1,
        local_memory_gate_init=-3.0,
        retrieval_copy_mode='none',
        retrieval_copy_weight=0.0,
        retrieval_copy_gate_init=2.0,
        retrieval_copy_dropout=0.1,
        retrieval_readout_mode='none',
        retrieval_readout_weight=0.0,
        retrieval_readout_gate_init=-2.0,
        retrieval_readout_dropout=0.1,
        retrieval_readout_prior_transform='logit',
        treatment_episode_adapter_mode='none',
        treatment_episode_adapter_weight=0.0,
        treatment_episode_experience_dim=0,
        treatment_episode_adapter_dropout=0.1,
        treatment_episode_adapter_gate_init=-3.0,
        treatment_episode_action_bias_weight=0.0,
        treatment_episode_bias_scale_mode='none',
        history_copy_mode='none',
        history_copy_weight=0.0,
        history_copy_prior_mode='binary',
        history_copy_decay=0.7,
        history_copy_gate_init=-2.0,
        history_copy_dropout=0.1,
        med_count_mode='none',
        seq_decoder_mode='none',
        seq_decoder_dropout=0.1,
        seq_transformer_hidden_dim=0,
        seq_transformer_nhead=4,
        seq_transformer_layers=1,
        seq_copy_mode='none',
        seq_copy_gate_init=1.0,
        seq_copy_recency_decay=0.7,
        seq_retrieval_copy_mode='none',
        seq_retrieval_copy_gate_init=1.0,
        seq_hidr_gumbel_tau=0.6,
        seq_hidr_att_tau=20.0,
        patient_graph_mode='none',
        patient_graph_ehr_scale=0.25,
        patient_graph_ddi_scale=0.25,
        patient_graph_dropout=0.1,
        graph_logit_bias_weight=0.0,
        graph_edge_bias_weight=0.0,
        graph_edge_ddi_weight=1.0,
        graph_edge_prior_source='history',
        graph_edge_detach_prior=True,
        graph_med_encoder_mode='none',
        graph_med_ehr_weight=0.0,
        graph_med_ddi_weight=0.0,
        graph_med_gate_hidden=128,
        graph_med_dropout=0.1,
        patient_med_scorer_mode='none',
        patient_med_scorer_rank=32,
        patient_med_scorer_weight=0.0,
        patient_med_scorer_dropout=0.1,
        main_med_scorer_mode='none',
        main_med_scorer_hidden_dim=128,
        main_med_scorer_rank=32,
        main_med_scorer_dropout=0.1,
        current_med_attention_mode='none',
        current_med_attention_dropout=0.1,
        current_med_attention_gate_init=-3.0,
        disable_procedure_branch=False,
        own_history_context_mode='none',
        own_history_context_weight=0.0,
        own_history_context_dropout=0.1,
        own_history_context_gate_init=-3.0,
        own_history_context_recency_decay=1.0,
        posthoc_calibration_mode='none',
        posthoc_graph_source='hybrid',
        backbone_innovation_mode='none',
        action_transition_mode='none',
        action_transition_hidden_dim=0,
        action_transition_dropout=0.1,
        action_fusion_trainable=False,
        action_fusion_type='trainable_scalar',
        action_fusion_init_from_p4=False,
        action_fusion_freeze_weights=False,
        routed_fusion_hidden_dim=128,
        routed_fusion_drug_emb_dim=32,
        routed_fusion_dropout=0.1,
        routed_fusion_entropy_weight=0.0,
        routed_residual_freeze_base=True,
        routed_residual_scale=0.1,
        routed_residual_max_gate=1.0,
        routed_residual_gate_init_bias=-3.0,
        routed_residual_l2_weight=0.0,
        routed_residual_add_penalty_weight=0.0,
        routed_residual_entropy_weight=0.0,
        routed_residual_load_balance_weight=0.0,
    ):
        super().__init__()
        self.d_voc_size=voc_size[0]
        self.p_voc_size=voc_size[1]
        self.m_voc_size=voc_size[2]
        self.emb_dim=emb_dim
        self.k=k
        self.multitask_mode = multitask_mode
        self.aux_label_dim = aux_label_dim
        self.retrieval_context_mode = retrieval_context_mode
        self.retrieval_context_target = retrieval_context_target
        self.retrieval_context_weight = retrieval_context_weight
        self.external_memory_mode = external_memory_mode
        self.external_memory_weight = external_memory_weight
        self.local_memory_mode = local_memory_mode
        self.local_memory_weight = local_memory_weight
        self.retrieval_copy_mode = retrieval_copy_mode
        self.retrieval_copy_weight = retrieval_copy_weight
        self.retrieval_readout_mode = retrieval_readout_mode
        self.retrieval_readout_weight = retrieval_readout_weight
        self.treatment_episode_adapter_mode = treatment_episode_adapter_mode
        self.treatment_episode_adapter_weight = treatment_episode_adapter_weight
        self.treatment_episode_action_bias_weight = treatment_episode_action_bias_weight
        self.treatment_episode_bias_scale_mode = treatment_episode_bias_scale_mode
        self.history_copy_mode = history_copy_mode
        self.history_copy_weight = history_copy_weight
        self.med_count_mode = med_count_mode
        self.seq_decoder_mode = seq_decoder_mode
        self.patient_graph_mode = patient_graph_mode
        self.graph_logit_bias_weight = graph_logit_bias_weight
        self.graph_edge_bias_weight = graph_edge_bias_weight
        self.graph_edge_prior_source = graph_edge_prior_source
        self.graph_med_encoder_mode = graph_med_encoder_mode
        self.patient_med_scorer_mode = patient_med_scorer_mode
        self.patient_med_scorer_weight = patient_med_scorer_weight
        self.main_med_scorer_mode = main_med_scorer_mode
        self.current_med_attention_mode = current_med_attention_mode
        self.own_history_context_mode = own_history_context_mode
        self.own_history_context_weight = own_history_context_weight
        self.direct_state_source = direct_state_source
        self.posthoc_calibration_mode = posthoc_calibration_mode
        self.backbone_innovation_mode = backbone_innovation_mode
        self.action_transition_mode = action_transition_mode
        self.action_fusion_trainable = action_fusion_trainable
        self.action_fusion_type = action_fusion_type
        if backbone_innovation_mode not in ('none', 'action_retrieval_graph_count'):
            raise ValueError(f'Unsupported backbone_innovation_mode: {backbone_innovation_mode}')
        if action_transition_mode not in ('none', 'heads'):
            raise ValueError(f'Unsupported action_transition_mode: {action_transition_mode}')
        if action_fusion_type not in ('trainable_scalar', 'routed', 'ffn', 'simple_ffn_routed', 'routed_residual', 'aclr_residual'):
            raise ValueError(f'Unsupported action_fusion_type: {action_fusion_type}')
        if action_fusion_trainable and action_transition_mode == 'none':
            raise ValueError('action_fusion_trainable requires action_transition_mode heads')
        if action_transition_hidden_dim < 0:
            raise ValueError('action_transition_hidden_dim must be non-negative')
        if not 0.0 <= action_transition_dropout <= 1.0:
            raise ValueError('action_transition_dropout must be in [0, 1]')
        if routed_fusion_hidden_dim <= 0:
            raise ValueError('routed_fusion_hidden_dim must be positive')
        if routed_fusion_drug_emb_dim <= 0:
            raise ValueError('routed_fusion_drug_emb_dim must be positive')
        if not 0.0 <= routed_fusion_dropout <= 1.0:
            raise ValueError('routed_fusion_dropout must be in [0, 1]')
        if routed_fusion_entropy_weight < 0:
            raise ValueError('routed_fusion_entropy_weight must be non-negative')
        for name, value in (
            ('routed_residual_scale', routed_residual_scale),
            ('routed_residual_max_gate', routed_residual_max_gate),
            ('routed_residual_l2_weight', routed_residual_l2_weight),
            ('routed_residual_add_penalty_weight', routed_residual_add_penalty_weight),
            ('routed_residual_entropy_weight', routed_residual_entropy_weight),
            ('routed_residual_load_balance_weight', routed_residual_load_balance_weight),
        ):
            if value < 0:
                raise ValueError(f'{name} must be non-negative')
        if patient_graph_mode not in ('none', 'node_gate', 'edge_bias'):
            raise ValueError(f'Unsupported patient_graph_mode: {patient_graph_mode}')
        if graph_med_encoder_mode not in ('none', 'static', 'patient_gate'):
            raise ValueError(f'Unsupported graph_med_encoder_mode: {graph_med_encoder_mode}')
        if graph_med_ehr_weight < 0 or graph_med_ddi_weight < 0:
            raise ValueError('graph_med_ehr_weight and graph_med_ddi_weight must be non-negative')
        if graph_med_gate_hidden <= 0:
            raise ValueError('graph_med_gate_hidden must be positive')
        if not 0.0 <= graph_med_dropout <= 1.0:
            raise ValueError('graph_med_dropout must be in [0, 1]')
        if patient_med_scorer_mode not in ('none', 'lowrank_bias'):
            raise ValueError(f'Unsupported patient_med_scorer_mode: {patient_med_scorer_mode}')
        if main_med_scorer_mode not in ('none', 'feature_delta'):
            raise ValueError(f'Unsupported main_med_scorer_mode: {main_med_scorer_mode}')
        if current_med_attention_mode not in ('none', 'history_cross_attention'):
            raise ValueError(f'Unsupported current_med_attention_mode: {current_med_attention_mode}')
        if own_history_context_mode not in ('none', 'visit_attention'):
            raise ValueError(f'Unsupported own_history_context_mode: {own_history_context_mode}')
        if own_history_context_weight < 0:
            raise ValueError('own_history_context_weight must be non-negative')
        if treatment_episode_adapter_mode not in ('none', 'logit_bias', 'state', 'both', 'action'):
            raise ValueError(f'Unsupported treatment_episode_adapter_mode: {treatment_episode_adapter_mode}')
        if treatment_episode_adapter_weight < 0:
            raise ValueError('treatment_episode_adapter_weight must be non-negative')
        if treatment_episode_adapter_mode in ('logit_bias', 'state', 'both') and treatment_episode_adapter_weight <= 0:
            raise ValueError('treatment_episode_adapter_weight must be positive when state/logit treatment episode adapter is enabled')
        if treatment_episode_action_bias_weight < 0:
            raise ValueError('treatment_episode_action_bias_weight must be non-negative')
        if treatment_episode_bias_scale_mode not in ('none', 'logit_abs_mean'):
            raise ValueError(f'Unsupported treatment_episode_bias_scale_mode: {treatment_episode_bias_scale_mode}')
        if patient_med_scorer_weight < 0:
            raise ValueError('patient_med_scorer_weight must be non-negative')
        if graph_edge_prior_source not in ('model', 'history', 'retrieval', 'hybrid'):
            raise ValueError(f'Unsupported graph_edge_prior_source: {graph_edge_prior_source}')
        if direct_state_source not in ('raw', 'query', 'raw_query_sum', 'raw_query_gated'):
            raise ValueError(f'Unsupported direct_state_source: {direct_state_source}')
        if posthoc_calibration_mode not in ('none', 'per_med_linear', 'per_med_graph'):
            raise ValueError(f'Unsupported posthoc_calibration_mode: {posthoc_calibration_mode}')
        if posthoc_graph_source not in ('model', 'history', 'retrieval', 'hybrid'):
            raise ValueError(f'Unsupported posthoc_graph_source: {posthoc_graph_source}')
        patient_temporal_type = patient_temporal_type or temporal_type
        med_temporal_type = med_temporal_type or temporal_type
        self.medrep=MedRepLearn(
            emb_dim,
            k,
            self.m_voc_size,
            temporal_type=med_temporal_type,
            history_definition=med_history_definition,
            scorer_type=med_scorer_type,
            candidate_encoder_type=med_candidate_encoder_type,
        )
        self.patrep=PatientRepLearn(
            emb_dim,
            k,
            self.d_voc_size,
            self.p_voc_size,
            self.m_voc_size,
            temporal_type=patient_temporal_type,
            fusion_type=patient_fusion_type,
            clinical_feature_dim=clinical_feature_dim,
            clinical_feature_mode=clinical_feature_mode,
            clinical_fusion_type=clinical_fusion_type,
            clinical_feature_dropout=clinical_feature_dropout,
            clinical_temporal_fusion=clinical_temporal_fusion,
            clinical_temporal_gate_init=clinical_temporal_gate_init,
            current_med_attention_mode=current_med_attention_mode,
            current_med_attention_dropout=current_med_attention_dropout,
            current_med_attention_gate_init=current_med_attention_gate_init,
            disable_procedure_branch=disable_procedure_branch,
        )
        self.patient_query_projector=PatientQueryProjector(emb_dim,k,query_type=patient_query_type)
        self.lin1=nn.Linear(2*k*emb_dim,self.m_voc_size)
        self.lin2=nn.Linear(2*k*emb_dim,2*k*emb_dim)
        self.norm=torch.nn.LayerNorm(2*k*emb_dim)
        self.direct_raw_norm = nn.LayerNorm(2*k*emb_dim)
        self.direct_query_norm = nn.LayerNorm(2*k*emb_dim)
        self.direct_query_gate = None
        if direct_state_source == 'raw_query_gated':
            self.direct_query_gate = nn.Linear(4*k*emb_dim, 1)
            nn.init.zeros_(self.direct_query_gate.weight)
            nn.init.constant_(self.direct_query_gate.bias, direct_query_gate_init)
        self.gcn=GCN(self.m_voc_size, emb_dim, ehr_adj, ddi_adj)
        self.patient_graph_gate = None
        self.patient_graph_edge_bias = None
        if patient_graph_mode == 'node_gate':
            self.patient_graph_gate = PatientConditionedGraphGate(
                state_dim=2 * k * emb_dim,
                m_voc_size=self.m_voc_size,
                ehr_scale=patient_graph_ehr_scale,
                ddi_scale=patient_graph_ddi_scale,
                dropout=patient_graph_dropout,
            )
        elif patient_graph_mode == 'edge_bias':
            self.patient_graph_edge_bias = PatientConditionedEdgeBias(
                state_dim=2 * k * emb_dim,
                ehr_adj=ehr_adj,
                ddi_adj=ddi_adj,
                ehr_scale=patient_graph_ehr_scale,
                ddi_scale=patient_graph_ddi_scale,
                ddi_weight=graph_edge_ddi_weight,
                dropout=patient_graph_dropout,
                detach_prior=graph_edge_detach_prior,
            )
        self.sm=SimpleNetV(emb_dim,k,voc_size)
        self.med_rep_fusion=MedicationRepresentationFusion(emb_dim,fusion_type=med_fusion_type)
        self.graph_med_encoder = None
        if graph_med_encoder_mode != 'none':
            self.graph_med_encoder = GraphAwareMedicationEncoder(
                emb_dim=emb_dim,
                state_dim=2 * k * emb_dim,
                ehr_adj=ehr_adj,
                ddi_adj=ddi_adj,
                mode=graph_med_encoder_mode,
                ehr_weight=graph_med_ehr_weight,
                ddi_weight=graph_med_ddi_weight,
                gate_hidden=graph_med_gate_hidden,
                dropout=graph_med_dropout,
            )
        self.lin_med_expand=nn.Linear(emb_dim,2*k*emb_dim)
        self.action_transition_backbone = None
        if action_transition_mode == 'heads':
            self.action_transition_backbone = ActionAwareMedicationTransitionBackbone(
                state_dim=2 * k * emb_dim,
                m_voc_size=self.m_voc_size,
                hidden_dim=action_transition_hidden_dim,
                dropout=action_transition_dropout,
            )
        self.action_fusion = None
        if action_fusion_trainable:
            if action_fusion_type == 'trainable_scalar':
                self.action_fusion = TrainableActionFusionLayer(
                    init_from_p4=action_fusion_init_from_p4,
                    freeze_weights=action_fusion_freeze_weights,
                )
            elif action_fusion_type == 'routed':
                self.action_fusion = RoutedActionFusionLayer(
                    m_voc_size=self.m_voc_size,
                    state_dim=2 * k * emb_dim,
                    hidden_dim=routed_fusion_hidden_dim,
                    drug_emb_dim=routed_fusion_drug_emb_dim,
                    dropout=routed_fusion_dropout,
                    entropy_weight=routed_fusion_entropy_weight,
                )
            elif action_fusion_type in ('routed_residual', 'aclr_residual'):
                self.action_fusion = RoutedResidualActionFusionLayer(
                    m_voc_size=self.m_voc_size,
                    state_dim=2 * k * emb_dim,
                    hidden_dim=routed_fusion_hidden_dim,
                    drug_emb_dim=routed_fusion_drug_emb_dim,
                    dropout=routed_fusion_dropout,
                    residual_scale=routed_residual_scale,
                    max_gate=routed_residual_max_gate,
                    gate_init_bias=routed_residual_gate_init_bias,
                    freeze_base=routed_residual_freeze_base,
                    l2_weight=routed_residual_l2_weight,
                    add_penalty_weight=routed_residual_add_penalty_weight,
                    entropy_weight=routed_residual_entropy_weight,
                    load_balance_weight=routed_residual_load_balance_weight,
                )
            elif action_fusion_type in ('ffn', 'simple_ffn_routed'):
                self.action_fusion = SimpleFFNActionFusionLayer(
                    m_voc_size=self.m_voc_size,
                    state_dim=2 * k * emb_dim,
                    hidden_dim=routed_fusion_hidden_dim,
                    drug_emb_dim=routed_fusion_drug_emb_dim,
                    dropout=routed_fusion_dropout,
                    entropy_weight=routed_fusion_entropy_weight,
                )
        self.patient_med_scorer = None
        if patient_med_scorer_mode == 'lowrank_bias':
            self.patient_med_scorer = PatientConditionedMedicationScorer(
                state_dim=2 * k * emb_dim,
                rank=patient_med_scorer_rank,
                dropout=patient_med_scorer_dropout,
            )
        self.output_mixer = OutputMixFusion(
            gate_input_dim=4 * k * emb_dim,
            mix_mode=output_mix_mode,
            init_weight=output_mix_weight,
            branch_calibration=output_branch_calibration,
        )
        self.main_med_scorer = None
        if main_med_scorer_mode == 'feature_delta':
            self.main_med_scorer = MainMedicationFeatureScorer(
                state_dim=2 * k * emb_dim,
                m_voc_size=self.m_voc_size,
                hidden_dim=main_med_scorer_hidden_dim,
                rank=main_med_scorer_rank,
                dropout=main_med_scorer_dropout,
            )
        self.retrieval_attention = None
        if retrieval_context_mode == 'attention':
            if retrieval_context_target not in ('logit', 'state'):
                raise ValueError(f'Unsupported retrieval_context_target: {retrieval_context_target}')
            self.retrieval_attention = RetrievedMedicationAttention(
                emb_dim=emb_dim,
                state_dim=2 * k * emb_dim,
                m_voc_size=self.m_voc_size,
                dropout=retrieval_context_dropout,
                gate_init=retrieval_context_gate_init,
            )
        elif retrieval_context_mode != 'none':
            raise ValueError(f'Unsupported retrieval_context_mode: {retrieval_context_mode}')
        self.external_memory = None
        if external_memory_mode != 'none' and external_memory_weight > 0:
            if external_memory_mode not in ('state', 'logit', 'both'):
                raise ValueError(f'Unsupported external_memory_mode: {external_memory_mode}')
            if external_memory_num <= 0:
                raise ValueError('external_memory_num must be positive when external memory is enabled')
            memory_hidden_size = external_memory_hidden_size if external_memory_hidden_size > 0 else 2 * k * emb_dim
            self.external_memory = ExternalMemoryNetwork(
                input_dim=emb_dim,
                state_dim=2 * k * emb_dim,
                hidden_size=memory_hidden_size,
                memory_num=external_memory_num,
                m_voc_size=self.m_voc_size,
                dropout=external_memory_dropout,
                gate_init=external_memory_gate_init,
            )
        elif external_memory_mode != 'none' and external_memory_weight <= 0:
            raise ValueError('external_memory_weight must be positive when external memory is enabled')
        self.local_memory = None
        if local_memory_mode != 'none' and local_memory_weight > 0:
            if local_memory_mode != 'state':
                raise ValueError(f'Unsupported local_memory_mode: {local_memory_mode}')
            self.local_memory = LocalMemoryNetwork(
                emb_dim=emb_dim,
                state_dim=2 * k * emb_dim,
                dropout=local_memory_dropout,
                gate_init=local_memory_gate_init,
            )
        elif local_memory_mode != 'none' and local_memory_weight <= 0:
            raise ValueError('local_memory_weight must be positive when local memory is enabled')
        self.own_history_context = None
        if own_history_context_mode == 'visit_attention':
            if own_history_context_weight <= 0:
                raise ValueError('own_history_context_weight must be positive when own history context is enabled')
            self.own_history_context = OwnHistoryVisitAttention(
                emb_dim=emb_dim,
                state_dim=2 * k * emb_dim,
                m_voc_size=self.m_voc_size,
                dropout=own_history_context_dropout,
                gate_init=own_history_context_gate_init,
                recency_decay=own_history_context_recency_decay,
            )
        elif own_history_context_mode != 'none':
            raise ValueError(f'Unsupported own_history_context_mode: {own_history_context_mode}')
        self.retrieval_copy = None
        if retrieval_copy_mode == 'prob_mix':
            if retrieval_copy_weight <= 0:
                raise ValueError('retrieval_copy_weight must be positive when retrieval copy is enabled')
            self.retrieval_copy = RetrievedMedicationProbMix(
                state_dim=2 * k * emb_dim,
                m_voc_size=self.m_voc_size,
                gate_init=retrieval_copy_gate_init,
                dropout=retrieval_copy_dropout,
            )
        elif retrieval_copy_mode != 'none':
            raise ValueError(f'Unsupported retrieval_copy_mode: {retrieval_copy_mode}')
        self.retrieval_readout = None
        if retrieval_readout_mode == 'logit_bias':
            if retrieval_readout_weight <= 0:
                raise ValueError('retrieval_readout_weight must be positive when retrieval readout is enabled')
            self.retrieval_readout = RetrievedMedicationReadout(
                state_dim=2 * k * emb_dim,
                m_voc_size=self.m_voc_size,
                gate_init=retrieval_readout_gate_init,
                dropout=retrieval_readout_dropout,
                prior_transform=retrieval_readout_prior_transform,
            )
        elif retrieval_readout_mode != 'none':
            raise ValueError(f'Unsupported retrieval_readout_mode: {retrieval_readout_mode}')
        self.treatment_episode_adapter = None
        if treatment_episode_adapter_mode != 'none':
            self.treatment_episode_adapter = TreatmentEpisodeExperienceAdapter(
                state_dim=2 * k * emb_dim,
                m_voc_size=self.m_voc_size,
                experience_dim=treatment_episode_experience_dim,
                mode=treatment_episode_adapter_mode,
                gate_init=treatment_episode_adapter_gate_init,
                dropout=treatment_episode_adapter_dropout,
            )
        self.history_copy = None
        if history_copy_mode == 'gated':
            self.history_copy = HistoricalMedicationCopy(
                state_dim=2 * k * emb_dim,
                m_voc_size=self.m_voc_size,
                prior_mode=history_copy_prior_mode,
                decay=history_copy_decay,
                gate_init=history_copy_gate_init,
                dropout=history_copy_dropout,
            )
        elif history_copy_mode == 'prob_mix':
            self.history_copy = HistoricalMedicationProbMix(
                state_dim=2 * k * emb_dim,
                m_voc_size=self.m_voc_size,
                prior_mode=history_copy_prior_mode,
                decay=history_copy_decay,
                gate_init=history_copy_gate_init,
                dropout=history_copy_dropout,
            )
        elif history_copy_mode != 'none':
            raise ValueError(f'Unsupported history_copy_mode: {history_copy_mode}')
        self.posthoc_calibrator = None
        if posthoc_calibration_mode in ('per_med_linear', 'per_med_graph'):
            self.posthoc_calibrator = PosthocLogitCalibrator(
                self.m_voc_size,
                graph_mode='per_med' if posthoc_calibration_mode == 'per_med_graph' else 'none',
                ehr_adj=ehr_adj,
                ddi_adj=ddi_adj,
                graph_source=posthoc_graph_source,
            )
        self.patient_state_projector = None
        self.patient_state_head = None
        self.med_transition_head = None
        self.med_count_head = None
        self.seq_decoder = None
        if self.seq_decoder_mode == 'aux':
            self.seq_decoder = MedicationSequenceDecoder(
                state_dim=2 * k * emb_dim,
                m_voc_size=self.m_voc_size,
                dropout=seq_decoder_dropout,
                copy_mode=seq_copy_mode,
                copy_gate_init=seq_copy_gate_init,
                copy_recency_decay=seq_copy_recency_decay,
                retrieval_copy_mode=seq_retrieval_copy_mode,
                retrieval_copy_gate_init=seq_retrieval_copy_gate_init,
            )
        elif self.seq_decoder_mode == 'hidr_memory':
            self.seq_decoder = HidrMemorySequenceDecoder(
                state_dim=2 * k * emb_dim,
                visit_dim=emb_dim,
                m_voc_size=self.m_voc_size,
                dropout=seq_decoder_dropout,
                gumbel_tau=seq_hidr_gumbel_tau,
                att_tau=seq_hidr_att_tau,
                gate_init=seq_copy_gate_init,
            )
        elif self.seq_decoder_mode == 'hidr_transformer':
            self.seq_decoder = HidrTransformerSequenceDecoder(
                state_dim=2 * k * emb_dim,
                visit_dim=emb_dim,
                m_voc_size=self.m_voc_size,
                hidden_dim=seq_transformer_hidden_dim or emb_dim,
                nhead=seq_transformer_nhead,
                num_layers=seq_transformer_layers,
                dropout=seq_decoder_dropout,
                gumbel_tau=seq_hidr_gumbel_tau,
                att_tau=seq_hidr_att_tau,
                gate_init=seq_copy_gate_init,
            )
        elif self.seq_decoder_mode != 'none':
            raise ValueError(f'Unsupported seq_decoder_mode: {self.seq_decoder_mode}')
        if self.med_count_mode == 'aux':
            self.med_count_head = MedicationCountHead(
                input_dim=2 * k * emb_dim,
                dropout=aux_head_dropout,
            )
        elif self.med_count_mode != 'none':
            raise ValueError(f'Unsupported med_count_mode: {self.med_count_mode}')
        if self.multitask_mode == 'clinical_state':
            if self.aux_label_dim <= 0:
                raise ValueError("aux_label_dim must be positive when multitask_mode='clinical_state'")
            self.patient_state_projector = PatientStateProjector(
                input_dim=2 * k * emb_dim,
                hidden_dim=2 * k * emb_dim,
            )
            self.patient_state_head = ClinicalStateHead(
                input_dim=2 * k * emb_dim,
                output_dim=aux_label_dim,
                dropout=aux_head_dropout,
            )
            self.med_transition_head = MedicationTransitionHead(
                input_dim=2 * k * emb_dim,
                output_dim=3,
                dropout=aux_head_dropout,
            )
        elif self.multitask_mode != 'none':
            raise ValueError(f'Unsupported multitask_mode: {self.multitask_mode}')

    @property
    def w(self):
        return self.output_mixer.current_weight

    @property
    def w_logits(self):
        if not hasattr(self.output_mixer, 'w_logits'):
            raise AttributeError('w_logits is only available for learnable_scalar output mixing.')
        return self.output_mixer.w_logits

    def get_output_mix_logging_stats(self):
        return self.output_mixer.get_logging_stats()

    def encode_patient_query(
        self,
        diags,
        procs,
        meds,
        clinical_features=None,
        clinical_feature_masks=None,
        clinical_feature_abnormal=None,
        clinical_feature_delta_hours=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
        retrieval_priors=None,
        own_diags=None,
        own_procs=None,
        own_meds=None,
        seq_decoder_inputs=None,
        return_code_context=False,
    ):
        # meds=torch.zeros_like(meds).to(meds.device)
        # diags=torch.zeros_like(diags).to(diags.device)
        #procs=torch.zeros_like(procs).to(procs.device)
        patient_repr=self.patrep(
            diags,
            procs,
            meds,
            clinical_features=clinical_features,
            clinical_feature_masks=clinical_feature_masks,
            clinical_feature_abnormal=clinical_feature_abnormal,
            clinical_feature_delta_hours=clinical_feature_delta_hours,
        )
        # visit_context matches the old pat[:, :2*k, :] slice
        visit_context=patient_repr['visit_context']
        # patient_query keeps the old pat[:, 2*k:, :] sequence semantics.
        patient_query=patient_repr['patient_query']
        h_patient=self.patient_query_projector(patient_query)
        q=h_patient+self.lin2(self.norm(h_patient))
        if return_code_context:
            return (
                visit_context,
                q,
                patient_repr['code_context'],
                patient_repr['code_context_mask'],
                patient_repr['current_med_attention_gate'],
                patient_repr['current_med_attention'],
            )
        return visit_context, q

    def forward(
        self,
        diags,
        procs,
        meds,
        clinical_features=None,
        clinical_feature_masks=None,
        clinical_feature_abnormal=None,
        clinical_feature_delta_hours=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
        treatment_episode_experiences=None,
        treatment_episode_scores=None,
        retrieval_priors=None,
        own_diags=None,
        own_procs=None,
        own_meds=None,
        seq_decoder_inputs=None,
    ):
        # [batch, 2k*emb_dim]
        batch=diags.shape[0]
        (
            visit_context,
            q,
            code_context,
            code_context_mask,
            current_med_attention_gate,
            current_med_attention,
        ) = self.encode_patient_query(
            diags,
            procs,
            meds,
            clinical_features=clinical_features,
            clinical_feature_masks=clinical_feature_masks,
            clinical_feature_abnormal=clinical_feature_abnormal,
            clinical_feature_delta_hours=clinical_feature_delta_hours,
            return_code_context=True,
        )
        raw_state = visit_context.reshape(batch, -1)
        if self.direct_state_source == 'raw':
            direct_state = raw_state
        elif self.direct_state_source == 'query':
            direct_state = q
        elif self.direct_state_source == 'raw_query_sum':
            direct_state = raw_state + q
        else:
            raw_norm = self.direct_raw_norm(raw_state)
            query_norm = self.direct_query_norm(q)
            query_gate = torch.sigmoid(self.direct_query_gate(torch.cat((raw_norm, query_norm), dim=-1)))
            direct_state = raw_state + query_gate * query_norm
        retrieval_logits = None
        retrieval_output = None
        treatment_episode_output = None
        external_memory_output = None
        local_memory_output = None
        if self.retrieval_attention is not None and retrieval_meds is not None:
            retrieval_output = self.retrieval_attention(
                q,
                self.medrep.m_embs.weight,
                retrieval_meds,
                retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
            )
            retrieval_logits = retrieval_output['logits']
            if self.retrieval_context_target == 'state':
                direct_state = direct_state + (
                    self.retrieval_context_weight
                    * retrieval_output['gate']
                    * retrieval_output['state_delta']
                )
        if self.treatment_episode_adapter is not None and treatment_episode_experiences is not None:
            treatment_episode_output = self.treatment_episode_adapter(
                q,
                treatment_episode_experiences,
                retrieval_scores=treatment_episode_scores,
            )
            treatment_episode_gate = None
            if treatment_episode_output is not None:
                treatment_episode_gate = treatment_episode_output['gate'].to(device=direct_state.device, dtype=direct_state.dtype)
            if treatment_episode_output is not None and self.treatment_episode_adapter_mode in ('state', 'both'):
                direct_state = direct_state + (
                    self.treatment_episode_adapter_weight
                    * treatment_episode_gate
                    * treatment_episode_output['state_delta']
                )
        if self.external_memory is not None:
            external_query = visit_context[:, 0, :]
            external_memory_output = self.external_memory(external_query, direct_state)
            if self.external_memory_mode in ('state', 'both'):
                direct_state = direct_state + (
                    self.external_memory_weight
                    * external_memory_output['gate']
                    * external_memory_output['state_delta']
                )
        # [batch, m_voc_size]
        # [batch, m_voc_size, emb_dim]
        ehr_meds,ddi_meds=self.gcn()
        graph_gate_output = None
        if self.patient_graph_gate is not None:
            graph_gate_output = self.patient_graph_gate(direct_state, ehr_meds, ddi_meds)
            ehr_meds = graph_gate_output['ehr_embeddings']
            ddi_meds = graph_gate_output['ddi_embeddings']
        drug_memory = ehr_meds - ddi_meds
        if self.local_memory is not None:
            local_visit_context = visit_context
            local_meds = meds
            if own_diags is not None and own_procs is not None and own_meds is not None:
                local_visit_context, _ = self.encode_patient_query(
                    own_diags,
                    own_procs,
                    own_meds,
                )
                local_meds = own_meds
            elif own_meds is not None:
                local_meds = own_meds
            local_memory_output = self.local_memory(
                local_visit_context,
                local_meds,
                drug_memory,
                direct_state,
            )
            direct_state = direct_state + (
                self.local_memory_weight
                * local_memory_output['gate']
                * local_memory_output['state_delta']
            )
        own_history_context_output = None
        if self.own_history_context is not None and self.own_history_context_weight > 0:
            history_visit_context = visit_context
            history_meds = meds
            if own_diags is not None and own_procs is not None and own_meds is not None:
                history_visit_context, _ = self.encode_patient_query(
                    own_diags,
                    own_procs,
                    own_meds,
                )
                history_meds = own_meds
            elif own_meds is not None:
                history_meds = own_meds
            own_history_context_output = self.own_history_context(
                q,
                direct_state,
                history_visit_context,
                history_meds,
            )
            direct_state = direct_state + (
                self.own_history_context_weight
                * own_history_context_output['gate']
                * own_history_context_output['state_delta']
            )
        graph_med_output = None
        med_repr=self.medrep(meds)
        base_med_embeddings = med_repr['final_embeddings']
        if self.graph_med_encoder is not None:
            graph_med_output = self.graph_med_encoder(base_med_embeddings, direct_state)
            base_med_embeddings = graph_med_output['embeddings']
        h_meds=self.med_rep_fusion(base_med_embeddings,ehr_meds,ddi_meds)
        h_meds=self.lin_med_expand(h_meds)
        # [batch, m_voc_size]
        o_1=self.lin1(direct_state)
        o_2=torch.cosine_similarity(q.unsqueeze(1).repeat(1,self.m_voc_size,1),h_meds,dim=2)
        patient_med_scorer_output = None
        if self.patient_med_scorer is not None and self.patient_med_scorer_weight > 0:
            patient_med_scorer_output = self.patient_med_scorer(direct_state, h_meds)
        gate_inputs = torch.cat((visit_context.reshape(batch, -1), q), dim=1)
        med_logits = self.output_mixer(o_1, o_2, gate_inputs)
        main_med_scorer_output = None
        if self.main_med_scorer is not None:
            main_med_scorer_output = self.main_med_scorer(
                med_logits,
                o_1,
                o_2,
                direct_state,
                q,
                h_meds,
            )
            med_logits = main_med_scorer_output['logits']
        if patient_med_scorer_output is not None:
            med_logits = med_logits + self.patient_med_scorer_weight * patient_med_scorer_output['bias']
        if self.graph_logit_bias_weight > 0:
            med_logits = med_logits + self.graph_logit_bias_weight * o_2
        action_transition_output = None
        if self.action_transition_backbone is not None:
            action_transition_output = self.action_transition_backbone(direct_state, meds)
        patient_graph_edge_output = None
        if self.patient_graph_edge_bias is not None and self.graph_edge_bias_weight > 0:
            edge_retrieval_meds = retrieval_meds
            if edge_retrieval_meds is None and retrieval_priors is not None:
                edge_retrieval_meds = retrieval_priors.unsqueeze(1).clamp(0.0, 1.0)
            patient_graph_edge_output = self.patient_graph_edge_bias(
                direct_state,
                med_logits,
                meds,
                prior_source=self.graph_edge_prior_source,
                retrieval_meds=edge_retrieval_meds,
                retrieval_scores=retrieval_scores,
            )
            med_logits = med_logits + self.graph_edge_bias_weight * patient_graph_edge_output["bias"]
        if retrieval_output is not None and self.retrieval_context_target == 'logit':
            med_logits = med_logits + self.retrieval_context_weight * retrieval_output['gate'] * retrieval_logits
        if own_history_context_output is not None:
            med_logits = med_logits + (
                self.own_history_context_weight
                * own_history_context_output['gate']
                * own_history_context_output['logit_bias']
            )
        if external_memory_output is not None and self.external_memory_mode in ('logit', 'both'):
            med_logits = med_logits + (
                self.external_memory_weight
                * external_memory_output['gate']
                * external_memory_output['logits']
            )
        retrieval_readout_output = None
        if self.retrieval_readout is not None and self.retrieval_readout_weight > 0:
            retrieval_readout_output = self.retrieval_readout(
                q,
                direct_state,
                retrieval_meds,
                retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
            )
            if retrieval_readout_output is not None:
                med_logits = med_logits + self.retrieval_readout_weight * retrieval_readout_output['logit_bias']
        retrieval_copy_output = None
        if self.retrieval_copy is not None and self.retrieval_copy_weight > 0:
            retrieval_copy_output = self.retrieval_copy(
                med_logits,
                q,
                direct_state,
                retrieval_meds,
                retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
            )
            if retrieval_copy_output is not None:
                med_logits = (
                    (1.0 - self.retrieval_copy_weight) * med_logits
                    + self.retrieval_copy_weight * retrieval_copy_output['logits']
                )
        history_copy_output = None
        if self.history_copy is not None and self.history_copy_weight > 0:
            if self.history_copy_mode == 'prob_mix':
                history_copy_output = self.history_copy(med_logits, q, direct_state, meds)
                med_logits = (
                    (1.0 - self.history_copy_weight) * med_logits
                    + self.history_copy_weight * history_copy_output['logits']
                )
            else:
                history_copy_output = self.history_copy(q, direct_state, meds)
                med_logits = med_logits + self.history_copy_weight * history_copy_output['logits']
        posthoc_calibration_output = None
        if self.posthoc_calibrator is not None:
            posthoc_calibration_output = self.posthoc_calibrator(
                med_logits,
                meds=meds,
                retrieval_priors=retrieval_priors,
            )
            med_logits = posthoc_calibration_output["logits"]
        def scale_treatment_episode_bias(bias):
            if self.treatment_episode_bias_scale_mode == 'logit_abs_mean':
                target_scale = med_logits.detach().abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
                bias_scale = bias.detach().abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
                return bias * (target_scale / bias_scale)
            return bias

        treatment_episode_scaled_logit_bias = None
        treatment_episode_action_bias = None
        treatment_episode_scaled_action_bias = None
        if treatment_episode_output is not None and self.treatment_episode_adapter_mode in ('logit_bias', 'both'):
            treatment_episode_scaled_logit_bias = scale_treatment_episode_bias(treatment_episode_output['logit_bias'])
            if treatment_episode_gate is not None:
                treatment_episode_scaled_logit_bias = treatment_episode_scaled_logit_bias * treatment_episode_gate
            med_logits = med_logits + (
                self.treatment_episode_adapter_weight
                * treatment_episode_scaled_logit_bias
            )
        if treatment_episode_output is not None and self.treatment_episode_action_bias_weight > 0:
            if meds.size(1) > 1:
                previous_meds = (meds[:, 1, :] > 0).to(device=med_logits.device, dtype=med_logits.dtype)
            else:
                previous_meds = torch.zeros_like(med_logits)
            action_bias = (
                (1.0 - previous_meds) * treatment_episode_output['action_add_logits']
                + previous_meds * (
                    treatment_episode_output['action_continue_logits']
                    - treatment_episode_output['action_remove_logits']
                )
            )
            action_bias = torch.where(
                treatment_episode_output['has_experience'],
                action_bias,
                torch.zeros_like(action_bias),
            )
            treatment_episode_action_bias = action_bias
            treatment_episode_scaled_action_bias = scale_treatment_episode_bias(action_bias)
            med_logits = med_logits + self.treatment_episode_action_bias_weight * treatment_episode_scaled_action_bias
        fused_logits = None
        action_fusion_logging = None
        if self.action_fusion is not None:
            if action_transition_output is None:
                raise ValueError('action_fusion requires action transition output')
            if self.action_fusion_type in ('routed', 'ffn', 'simple_ffn_routed', 'routed_residual', 'aclr_residual'):
                fused_logits, action_fusion_logging = self.action_fusion(
                    med_logits,
                    action_transition_output['continue_logits'],
                    action_transition_output['add_logits'],
                    action_transition_output['remove_logits'],
                    action_transition_output['previous_meds'],
                    patient_state=direct_state,
                )
            else:
                fused_logits, action_fusion_logging = self.action_fusion(
                    med_logits,
                    action_transition_output['continue_logits'],
                    action_transition_output['add_logits'],
                    action_transition_output['remove_logits'],
                    action_transition_output['previous_meds'],
                )
        seq_logits = None
        if self.seq_decoder is not None and seq_decoder_inputs is not None:
            if self.seq_decoder_mode == 'hidr_transformer':
                seq_logits = self.seq_decoder(
                    q,
                    visit_context,
                    med_logits,
                    seq_decoder_inputs,
                    meds=meds,
                    retrieval_meds=retrieval_meds,
                    retrieval_scores=retrieval_scores,
                    retrieval_queries=retrieval_queries,
                    drug_memory=drug_memory,
                    code_context=code_context,
                    code_context_mask=code_context_mask,
                )
            elif self.seq_decoder_mode == 'hidr_memory':
                seq_logits = self.seq_decoder(
                    q,
                    visit_context,
                    med_logits,
                    seq_decoder_inputs,
                    meds=meds,
                    retrieval_meds=retrieval_meds,
                    retrieval_scores=retrieval_scores,
                    retrieval_queries=retrieval_queries,
                )
            else:
                seq_logits = self.seq_decoder(
                    q,
                    med_logits,
                    seq_decoder_inputs,
                    meds=meds,
                    retrieval_meds=retrieval_meds,
                    retrieval_scores=retrieval_scores,
                    retrieval_queries=retrieval_queries,
                )
        state_features = None
        clinical_state_logits = None
        med_transition_logits = None
        med_count_pred = None
        if self.med_count_head is not None:
            med_count_pred = self.med_count_head(direct_state)
        if self.patient_state_projector is not None:
            state_features = self.patient_state_projector(direct_state)
            clinical_state_logits = self.patient_state_head(state_features)
            med_transition_logits = self.med_transition_head(state_features)
        return {
            'med_logits': med_logits,
            'fused_logits': fused_logits,
            'action_fusion_logging': action_fusion_logging,
            'clinical_state_logits': clinical_state_logits,
            'med_transition_logits': med_transition_logits,
            'med_count_pred': med_count_pred,
            'patient_query': q,
            'visit_context': visit_context,
            'retrieval_logits': retrieval_logits,
            'external_memory_logits': external_memory_output['logits'] if external_memory_output is not None else None,
            'external_memory_gate': external_memory_output['gate'] if external_memory_output is not None else None,
            'local_memory_gate': local_memory_output['gate'] if local_memory_output is not None else None,
            'own_history_context_gate': own_history_context_output['gate'] if own_history_context_output is not None else None,
            'own_history_context_attention': own_history_context_output['visit_attn'] if own_history_context_output is not None else None,
            'own_history_context_prior': own_history_context_output['prior'] if own_history_context_output is not None else None,
            'own_history_context_logit_bias': own_history_context_output['logit_bias'] if own_history_context_output is not None else None,
            'retrieval_copy_logits': retrieval_copy_output['logits'] if retrieval_copy_output is not None else None,
            'retrieval_copy_gate': retrieval_copy_output['generate_gate'] if retrieval_copy_output is not None else None,
            'retrieval_copy_prior': retrieval_copy_output['prior'] if retrieval_copy_output is not None else None,
            'retrieval_readout_bias': retrieval_readout_output['logit_bias'] if retrieval_readout_output is not None else None,
            'retrieval_readout_gate': retrieval_readout_output['gate'] if retrieval_readout_output is not None else None,
            'retrieval_readout_prior': retrieval_readout_output['prior'] if retrieval_readout_output is not None else None,
            'treatment_episode_bias': treatment_episode_output['logit_bias'] if treatment_episode_output is not None else None,
            'treatment_episode_scaled_bias': treatment_episode_scaled_logit_bias,
            'treatment_episode_action_bias': treatment_episode_action_bias,
            'treatment_episode_scaled_action_bias': treatment_episode_scaled_action_bias,
            'treatment_episode_gate': treatment_episode_output['gate'] if treatment_episode_output is not None else None,
            'treatment_episode_has_experience': treatment_episode_output['has_experience'] if treatment_episode_output is not None else None,
            'treatment_episode_attention': treatment_episode_output['visit_attn'] if treatment_episode_output is not None else None,
            'treatment_episode_add_logits': treatment_episode_output['action_add_logits'] if treatment_episode_output is not None else None,
            'treatment_episode_continue_logits': treatment_episode_output['action_continue_logits'] if treatment_episode_output is not None else None,
            'treatment_episode_remove_logits': treatment_episode_output['action_remove_logits'] if treatment_episode_output is not None else None,
            'history_copy_logits': history_copy_output['logits'] if history_copy_output is not None else None,
            'posthoc_calibration_scale': posthoc_calibration_output['scale'] if posthoc_calibration_output is not None else None,
            'posthoc_calibration_bias': posthoc_calibration_output['bias'] if posthoc_calibration_output is not None else None,
            'posthoc_calibration_history_prior': posthoc_calibration_output['history_prior'] if posthoc_calibration_output is not None else None,
            'posthoc_calibration_retrieval_prior': posthoc_calibration_output['retrieval_prior'] if posthoc_calibration_output is not None else None,
            'posthoc_calibration_graph_ehr_prior': posthoc_calibration_output['graph_ehr_prior'] if posthoc_calibration_output is not None else None,
            'posthoc_calibration_graph_ddi_prior': posthoc_calibration_output['graph_ddi_prior'] if posthoc_calibration_output is not None else None,
            'patient_graph_ehr_scale': graph_gate_output['ehr_scale'] if graph_gate_output is not None else None,
            'patient_graph_ddi_scale': graph_gate_output['ddi_scale'] if graph_gate_output is not None else None,
            'patient_graph_edge_bias': patient_graph_edge_output['bias'] if patient_graph_edge_output is not None else None,
            'patient_graph_edge_prior': patient_graph_edge_output['prior'] if patient_graph_edge_output is not None else None,
            'graph_med_delta': graph_med_output['delta'] if graph_med_output is not None else None,
            'graph_med_gate': graph_med_output['gate'] if graph_med_output is not None else None,
            'patient_med_scorer_bias': patient_med_scorer_output['bias'] if patient_med_scorer_output is not None else None,
            'continue_logits': action_transition_output['continue_logits'] if action_transition_output is not None else None,
            'add_logits': action_transition_output['add_logits'] if action_transition_output is not None else None,
            'remove_logits': action_transition_output['remove_logits'] if action_transition_output is not None else None,
            'total_count_pred': action_transition_output['total_count_pred'] if action_transition_output is not None else None,
            'continue_count_pred': action_transition_output['continue_count_pred'] if action_transition_output is not None else None,
            'add_count_pred': action_transition_output['add_count_pred'] if action_transition_output is not None else None,
            'remove_count_pred': action_transition_output['remove_count_pred'] if action_transition_output is not None else None,
            'previous_meds': action_transition_output['previous_meds'] if action_transition_output is not None else None,
            'main_med_scorer_delta': main_med_scorer_output['delta'] if main_med_scorer_output is not None else None,
            'main_med_scorer_state_score': main_med_scorer_output['state_med_score'] if main_med_scorer_output is not None else None,
            'main_med_scorer_query_score': main_med_scorer_output['query_med_score'] if main_med_scorer_output is not None else None,
            'current_med_attention_gate': current_med_attention_gate,
            'current_med_attention': current_med_attention,
            'drug_memory': drug_memory,
            'code_context': code_context,
            'code_context_mask': code_context_mask,
            'seq_logits': seq_logits,
        }

    def greedy_sequence_decode(
        self,
        diags,
        procs,
        meds,
        max_len,
        end_bias=0.0,
        clinical_features=None,
        clinical_feature_masks=None,
        clinical_feature_abnormal=None,
        clinical_feature_delta_hours=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
        treatment_episode_experiences=None,
        treatment_episode_scores=None,
    ):
        if self.seq_decoder is None:
            raise ValueError("seq_decoder is not enabled")
        output = self.forward(
            diags,
            procs,
            meds,
            clinical_features=clinical_features,
            clinical_feature_masks=clinical_feature_masks,
            clinical_feature_abnormal=clinical_feature_abnormal,
            clinical_feature_delta_hours=clinical_feature_delta_hours,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
            treatment_episode_experiences=treatment_episode_experiences,
            treatment_episode_scores=treatment_episode_scores,
        )
        if self.seq_decoder_mode == 'hidr_transformer':
            return self.seq_decoder.greedy_decode(
                output['patient_query'],
                output['visit_context'],
                output['med_logits'],
                max_len,
                end_bias=end_bias,
                meds=meds,
                retrieval_meds=retrieval_meds,
                retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
                drug_memory=output['drug_memory'],
                code_context=output['code_context'],
                code_context_mask=output['code_context_mask'],
            )
        if self.seq_decoder_mode == 'hidr_memory':
            return self.seq_decoder.greedy_decode(
                output['patient_query'],
                output['visit_context'],
                output['med_logits'],
                max_len,
                end_bias=end_bias,
                meds=meds,
                retrieval_meds=retrieval_meds,
                retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
            )
        return self.seq_decoder.greedy_decode(
            output['patient_query'],
            output['med_logits'],
            max_len,
            end_bias=end_bias,
            meds=meds,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
        )

    def hidr_sequence_decode(
        self,
        diags,
        procs,
        meds,
        max_len,
        end_bias=0.0,
        clinical_features=None,
        clinical_feature_masks=None,
        clinical_feature_abnormal=None,
        clinical_feature_delta_hours=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
        treatment_episode_experiences=None,
        treatment_episode_scores=None,
    ):
        if self.seq_decoder is None:
            raise ValueError("seq_decoder is not enabled")
        output = self.forward(
            diags,
            procs,
            meds,
            clinical_features=clinical_features,
            clinical_feature_masks=clinical_feature_masks,
            clinical_feature_abnormal=clinical_feature_abnormal,
            clinical_feature_delta_hours=clinical_feature_delta_hours,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
            treatment_episode_experiences=treatment_episode_experiences,
            treatment_episode_scores=treatment_episode_scores,
        )
        if self.seq_decoder_mode == 'hidr_transformer':
            return self.seq_decoder.hidr_decode(
                output['patient_query'],
                output['visit_context'],
                output['med_logits'],
                max_len,
                end_bias=end_bias,
                meds=meds,
                retrieval_meds=retrieval_meds,
                retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
                drug_memory=output['drug_memory'],
                code_context=output['code_context'],
                code_context_mask=output['code_context_mask'],
            )
        if self.seq_decoder_mode == 'hidr_memory':
            return self.seq_decoder.hidr_decode(
                output['patient_query'],
                output['visit_context'],
                output['med_logits'],
                max_len,
                end_bias=end_bias,
                meds=meds,
                retrieval_meds=retrieval_meds,
                retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
            )
        return self.seq_decoder.hidr_decode(
            output['patient_query'],
            output['med_logits'],
            max_len,
            end_bias=end_bias,
            meds=meds,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
        )

    def sequence_rank_decode(
        self,
        diags,
        procs,
        meds,
        max_len,
        clinical_features=None,
        clinical_feature_masks=None,
        clinical_feature_abnormal=None,
        clinical_feature_delta_hours=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
        treatment_episode_experiences=None,
        treatment_episode_scores=None,
    ):
        if self.seq_decoder is None:
            raise ValueError("seq_decoder is not enabled")
        output = self.forward(
            diags,
            procs,
            meds,
            clinical_features=clinical_features,
            clinical_feature_masks=clinical_feature_masks,
            clinical_feature_abnormal=clinical_feature_abnormal,
            clinical_feature_delta_hours=clinical_feature_delta_hours,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
            treatment_episode_experiences=treatment_episode_experiences,
            treatment_episode_scores=treatment_episode_scores,
        )
        if self.seq_decoder_mode == 'hidr_transformer':
            return self.seq_decoder.rank_decode(
                output['patient_query'],
                output['visit_context'],
                output['med_logits'],
                max_len,
                meds=meds,
                retrieval_meds=retrieval_meds,
                retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
                drug_memory=output['drug_memory'],
                code_context=output['code_context'],
                code_context_mask=output['code_context_mask'],
            )
        if self.seq_decoder_mode == 'hidr_memory':
            return self.seq_decoder.rank_decode(
                output['patient_query'],
                output['visit_context'],
                output['med_logits'],
                max_len,
                meds=meds,
                retrieval_meds=retrieval_meds,
                retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
            )
        return self.seq_decoder.rank_decode(
            output['patient_query'],
            output['med_logits'],
            max_len,
            meds=meds,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
        )

    def sequence_prob_decode(
        self,
        diags,
        procs,
        meds,
        max_len,
        end_bias=0.0,
        aggregate='max',
        clinical_features=None,
        clinical_feature_masks=None,
        clinical_feature_abnormal=None,
        clinical_feature_delta_hours=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
        treatment_episode_experiences=None,
        treatment_episode_scores=None,
    ):
        if self.seq_decoder is None:
            raise ValueError("seq_decoder is not enabled")
        output = self.forward(
            diags,
            procs,
            meds,
            clinical_features=clinical_features,
            clinical_feature_masks=clinical_feature_masks,
            clinical_feature_abnormal=clinical_feature_abnormal,
            clinical_feature_delta_hours=clinical_feature_delta_hours,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
            treatment_episode_experiences=treatment_episode_experiences,
            treatment_episode_scores=treatment_episode_scores,
        )
        if self.seq_decoder_mode == 'hidr_transformer':
            return self.seq_decoder.prob_decode(
                output['patient_query'],
                output['visit_context'],
                output['med_logits'],
                max_len,
                end_bias=end_bias,
                aggregate=aggregate,
                meds=meds,
                retrieval_meds=retrieval_meds,
                retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
                drug_memory=output['drug_memory'],
                code_context=output['code_context'],
                code_context_mask=output['code_context_mask'],
            )
        if self.seq_decoder_mode == 'hidr_memory':
            return self.seq_decoder.prob_decode(
                output['patient_query'],
                output['visit_context'],
                output['med_logits'],
                max_len,
                end_bias=end_bias,
                aggregate=aggregate,
                meds=meds,
                retrieval_meds=retrieval_meds,
                retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
            )
        return self.seq_decoder.prob_decode(
            output['patient_query'],
            output['med_logits'],
            max_len,
            end_bias=end_bias,
            aggregate=aggregate,
            meds=meds,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
        )

    def beam_sequence_decode(
        self,
        diags,
        procs,
        meds,
        max_len,
        beam_size=4,
        end_bias=0.0,
        clinical_features=None,
        clinical_feature_masks=None,
        clinical_feature_abnormal=None,
        clinical_feature_delta_hours=None,
        retrieval_meds=None,
        retrieval_scores=None,
        retrieval_queries=None,
        treatment_episode_experiences=None,
        treatment_episode_scores=None,
    ):
        if self.seq_decoder is None:
            raise ValueError("seq_decoder is not enabled")
        output = self.forward(
            diags,
            procs,
            meds,
            clinical_features=clinical_features,
            clinical_feature_masks=clinical_feature_masks,
            clinical_feature_abnormal=clinical_feature_abnormal,
            clinical_feature_delta_hours=clinical_feature_delta_hours,
            retrieval_meds=retrieval_meds,
            retrieval_scores=retrieval_scores,
            retrieval_queries=retrieval_queries,
            treatment_episode_experiences=treatment_episode_experiences,
            treatment_episode_scores=treatment_episode_scores,
        )
        if self.seq_decoder_mode == 'hidr_transformer':
            return self.seq_decoder.beam_decode(
                output['patient_query'],
                output['visit_context'],
                output['med_logits'],
                max_len,
                beam_size=beam_size,
                end_bias=end_bias,
                meds=meds,
                retrieval_meds=retrieval_meds,
                retrieval_scores=retrieval_scores,
                retrieval_queries=retrieval_queries,
                drug_memory=output['drug_memory'],
                code_context=output['code_context'],
                code_context_mask=output['code_context_mask'],
            )
        raise ValueError("beam_sequence_decode is currently implemented for hidr_transformer sequence decoder")

# %%
# test
if __name__ == "__main__":
    emb_dim=32
    k=3
    voc_size=(10,20,24)
    batch=16
    seq_len=5
    ehr_adj=(torch.rand((voc_size[2],voc_size[2]))>0.7).float().to('cuda:0')
    ddi_adj=(torch.rand((voc_size[2],voc_size[2]))>0.7).float().to('cuda:0')
    model=SEER(emb_dim,voc_size,k,ehr_adj,ddi_adj).to('cuda:0')
    diags=(torch.randn((batch,seq_len,voc_size[0]),device='cuda:0')>0.9).float()
    procs=(torch.randn((batch,seq_len,voc_size[1]),device='cuda:0')>0.9).float()
    meds=(torch.randn((batch,seq_len,voc_size[2]),device='cuda:0')>0.9).float()
    y=model(diags,procs,meds)
    print(y['med_logits'].shape) # (16, 24)
#%%
