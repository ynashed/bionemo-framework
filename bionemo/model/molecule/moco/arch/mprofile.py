import functools
import math

import einops
import torch
import torch.cuda.nvtx as nvtx
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm import LayerNorm as BatchLayerNorm

# from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter, scatter_mean

from bionemo.model.molecule.moco.models.utils import PredictionHead


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "gelu_tanh": nn.GELU(approximate='tanh'),
    "sigmoid": nn.Sigmoid(),
}


class E3Norm(nn.Module):
    def __init__(self, n_vector_features: int = 1, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        if n_vector_features > 1:
            self.weight = nn.Parameter(torch.ones((1, 1, n_vector_features)))  # Separate weights for each channel
        else:
            self.weight = nn.Parameter(torch.ones((1, 1)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def forward(self, pos: torch.Tensor, batch: torch.Tensor):
        # pos is expected to be of shape [n, 3, n_vector_features]
        norm = torch.norm(pos, dim=1, keepdim=True)  # Normalize over the 3 dimension
        batch_size = int(batch.max()) + 1
        mean_norm = scatter_mean(norm, batch, dim=0, dim_size=batch_size)
        new_pos = self.weight * pos / (mean_norm[batch] + self.eps)
        return new_pos


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        num_hidden_layers: int = 0,
        activation: str = 'silu',
        dropout: float = 0.0,
        last_act: str = None,
        bias: bool = True,
    ):
        """
        Initialize the MLP.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_size (int): Dimension of the hidden layers.
            output_dim (int): Dimension of the output features.
            num_hidden_layers (int): Number of hidden layers.
            activation (str): Activation function to use ('relu', 'silu', etc.).
            dropout (float): Dropout probability (between 0 and 1).
        """
        super(MLP, self).__init__()

        if activation not in NONLINEARITIES:
            raise ValueError(f"Activation function must be one of {list(NONLINEARITIES.keys())}")

        self.act_layer = NONLINEARITIES[activation]

        # Create a list to hold all layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_size, bias=bias))
        layers.append(NONLINEARITIES[activation])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        if num_hidden_layers > 0:
            for _ in range(num_hidden_layers):
                layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))
                layers.append(NONLINEARITIES[activation])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_size, output_dim, bias=bias))
        if last_act:
            layers.append(NONLINEARITIES[last_act])

        # Combine all layers into a sequential module
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.layers(x)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, batch=None):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class AdaLN(nn.Module):
    def __init__(self, condition_dim: int, feature_dim: int):
        """
        Initialize the Adaptive Layer Normalization (AdaLN) module.
        This implementation does not learn a gate.

        Args:
            condition_dim (int): Dimension of the conditional input.
            feature_dim (int): Dimension of the input features.
        """
        super().__init__()
        self.layernorm = BatchLayerNorm(feature_dim)
        self.scale_shift_mlp = MLP(condition_dim, 2 * feature_dim, 2 * feature_dim)

    def forward(self, h: torch.Tensor, t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the AdaLN module.

        Args:
            h (torch.Tensor): Input tensor to be normalized (batch_size, feature_dim).
            t (torch.Tensor): Conditional input tensor (batch_size, condition_dim).

        Returns:
            torch.Tensor: Normalized output tensor (batch_size, feature_dim).
        """
        scale, shift = self.scale_shift_mlp(t).chunk(2, dim=-1)
        return (1 + scale[batch]) * self.layernorm(h, batch) + shift[batch]


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


#! Below are taken from ESM3 for now
def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as an nn.Module, allowing it to be used within nn.Sequential.
    This module splits the input tensor along the last dimension and applies the SiLU (Swish)
    activation function to the first half, then multiplies it by the second half.
    """

    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )


def swiglu_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.Linear(d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )


class XEGNNK(nn.Module):
    """
    X only EGNN
    """

    def __init__(self, invariant_node_feat_dim=64, n_vector_features=128):
        super().__init__()  #! This should be target to source
        self.message_input_size = (
            2 * invariant_node_feat_dim + n_vector_features + invariant_node_feat_dim + invariant_node_feat_dim
        )  # + invariant_edge_feat_dim
        self.phi_message = MLP(self.message_input_size, invariant_node_feat_dim, invariant_node_feat_dim)
        self.phi_x = MLP(invariant_node_feat_dim, invariant_node_feat_dim, n_vector_features)
        self.coor_update_clamp_value = 10.0
        # self.reset_parameters()
        self.h_norm = BatchLayerNorm(invariant_node_feat_dim)
        self.use_cross_product = False
        if self.use_cross_product:
            self.phi_x_cross = MLP(invariant_node_feat_dim, invariant_node_feat_dim, n_vector_features)
        self.x_norm = E3Norm(n_vector_features)
        # TODO: @Ali what is good weight inititalization for EGNN?

    #     self.apply(self.init_)

    # # def reset_parameters(self):
    # def init_(self, module): #! this made it worse
    #     if type(module) in {nn.Linear}:
    #         # seems to be needed to keep the network from exploding to NaN with greater depths
    #         nn.init.xavier_normal_(module.weight)
    #         nn.init.zeros_(module.bias)

    def forward(self, batch, X, H, edge_index, edge_attr=None, te=None, messages=None):
        X = X - scatter_mean(X, index=batch, dim=0, dim_size=X.shape[0])[batch]
        X = self.x_norm(X, batch)
        source, target = edge_index
        rel_coors = X[source] - X[target]
        rel_dist = (rel_coors.transpose(1, 2) ** 2).sum(dim=-1, keepdim=False)
        if edge_attr is not None:
            edge_attr_feat = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feat = rel_dist
        if messages is not None:
            m_ij = messages
        else:
            H = self.h_norm(H, batch)
            m_ij = self.phi_message(torch.cat([H[target], H[source], edge_attr_feat, te], dim=-1))
        coor_wij = self.phi_x(m_ij)  # E x 3
        if self.coor_update_clamp_value:
            coor_wij.clamp_(min=-self.coor_update_clamp_value, max=self.coor_update_clamp_value)
        # import ipdb; ipdb.set_trace()
        X_rel_norm = rel_coors / (1 + torch.sqrt(rel_dist.unsqueeze(1) + 1e-8))
        x_update = scatter(X_rel_norm * coor_wij.unsqueeze(1), index=target, dim=0, reduce='sum', dim_size=X.shape[0])
        X_out = X + x_update
        if self.use_cross_product:
            mean = scatter(X, index=batch, dim=0, reduce='mean', dim_size=X.shape[0])
            x_src = X[source] - mean[source]
            x_tgt = X[target] - mean[target]
            cross = torch.cross(x_src, x_tgt, dim=1)
            cross = cross / (1 + torch.linalg.norm(cross, dim=1, keepdim=True))
            coor_wij_cross = self.phi_x_cross(m_ij)
            if self.coor_update_clamp_value:
                coor_wij_cross.clamp_(min=-self.coor_update_clamp_value, max=self.coor_update_clamp_value)
            x_update_cross = scatter(
                cross * coor_wij_cross.unsqueeze(1), index=target, dim=0, reduce='sum', dim_size=X.shape[0]
            )
            X_out = X_out + x_update_cross

        return X_out


class BondRefine(nn.Module):  #! can make this nn.Module to ensure no weird propagate error
    def __init__(
        self,
        invariant_node_feat_dim=64,
        invariant_edge_feat_dim=32,
    ):
        super().__init__()
        # self.x_norm = E3Norm()
        self.h_norm = BatchLayerNorm(invariant_node_feat_dim)
        self.edge_norm = BatchLayerNorm(invariant_edge_feat_dim)
        self.bond_norm = BatchLayerNorm(invariant_edge_feat_dim)
        in_feats = 2 * invariant_node_feat_dim + 1 + invariant_edge_feat_dim
        self.refine_layer = torch.nn.Sequential(
            torch.nn.Linear(in_feats, invariant_edge_feat_dim),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(invariant_edge_feat_dim, invariant_edge_feat_dim),
        )

    def forward(self, batch, X, H, edge_index, edge_attr):
        X = X - scatter_mean(X, index=batch, dim=0)[batch]
        # X = self.x_norm(X, batch)
        H = self.h_norm(H, batch)
        source, target = edge_index
        rel_coors = X[source] - X[target]
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)
        # edge_batch, counts = torch.unique(batch, return_counts=True)
        # edge_batch = torch.repeat_interleave(edge_batch, counts * (counts - 1))  # E
        edge_batch = batch[source]
        edge_attr = self.edge_norm(edge_attr, edge_batch)
        infeats = torch.cat([H[target], H[source], rel_dist, edge_attr], dim=-1)
        return self.bond_norm(self.refine_layer(infeats), edge_batch)


def coord2dist(x, edge_index, scale_dist_features=1, batch=None):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, 1)
    # import ipdb; ipdb.set_trace()
    if scale_dist_features >= 2:
        # dotproduct = x[row] * x[col]  # shape [num_edges, 3, k]
        # dotproduct = dotproduct.sum(-2)  # sum over the spatial dimension, shape [num_edges, k]
        # # Concatenate the computed features
        # radial = torch.cat([radial, dotproduct], dim=-1)  # shape [num_edges, 2*k]
        dotproduct = (x[row] * x[col]).sum(dim=-2, keepdim=False)  # shape [num_edges, 1]
        norms = torch.linalg.norm(x[row], dim=-2, keepdim=False) * torch.linalg.norm(x[col], dim=-2, keepdim=False)
        cosine_similarity = dotproduct / (norms + 1e-8)  # Add epsilon to avoid division by zero
        radial = torch.cat([radial, cosine_similarity], dim=-1)  # shape [num_edges, 2]
    if scale_dist_features >= 3:
        # mean = scatter(x, index=batch, dim=0, reduce='mean', dim_size=x.shape[0])
        x_src = x[row]  # - mean[row]
        x_tgt = x[col]  # - mean[col]
        cross = torch.cross(x_src, x_tgt, dim=1)  # .sum(1)
        sin_theta = torch.linalg.norm(cross, dim=1, keepdim=False) / (norms + 1e-8)
        radial = torch.cat([radial, sin_theta], dim=-1)  # shape [num_edges, 3*k]
    return radial


class DiTeBlock(nn.Module):
    """
    Mimics DiT block
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_expansion_ratio=4.0,
        use_z=True,
        mask_z=True,
        use_rotary=False,
        n_vector_features=128,
        scale_dist_features=1,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.norm1 = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm2 = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        if scale_dist_features > 1:
            dist_size = n_vector_features * scale_dist_features
        else:
            dist_size = n_vector_features
        self.feature_embedder = MLP(hidden_size + hidden_size + hidden_size + dist_size, hidden_size, hidden_size)
        self.norm1_edge = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm2_edge = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)

        self.ffn_norm = BatchLayerNorm(hidden_size)
        self.ffn = swiglu_ffn(hidden_size, mlp_expansion_ratio, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_edge_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        # Single linear layer for QKV projection
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.norm_q = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm_k = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.out_projection = nn.Linear(hidden_size, hidden_size, bias=False)

        self.use_rotary = use_rotary
        self.d_head = hidden_size // num_heads

        if use_z:
            self.use_z = use_z
            self.pair_bias = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 1, bias=False))
            self.mask_z = mask_z

        self.lin_edge0 = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.lin_edge1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_edge1 = nn.Linear(hidden_size + dist_size, hidden_size, bias=False)
        self.ffn_norm_edge = BatchLayerNorm(hidden_size)
        self.ffn_edge = swiglu_ffn(hidden_size, mlp_expansion_ratio, bias=False)
        # self.tanh = nn.GELU(approximate='tanh')

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.num_heads, self.d_head))
        k = k.unflatten(-1, (self.num_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(
        self,
        batch: torch.Tensor,
        x: torch.Tensor,
        t_emb_h: torch.Tensor,
        edge_attr: torch.Tensor = None,
        edge_index: torch.Tensor = None,
        t_emb_e: torch.Tensor = None,
        dist: torch.Tensor = None,
        edge_batch: torch.Tensor = None,
        Z: torch.Tensor = None,
    ):
        """
        This assume pytorch geometric batching so batch size of 1 so skip rotary as it depends on having an actual batch

        batch: N
        x: N x 256
        temb: N x 256
        edge_attr: E x 256
        edge_index: 2 x E
        """
        if Z is not None:
            assert self.use_z

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb_h).chunk(6, dim=1)
        (
            edge_shift_msa,
            edge_scale_msa,
            edge_gate_msa,
            edge_shift_mlp,
            edge_scale_mlp,
            edge_gate_mlp,
        ) = self.adaLN_edge_modulation(t_emb_e).chunk(6, dim=1)
        src, tgt = edge_index
        # Normalize x
        x_norm = modulate(self.norm1(x, batch), shift_msa, scale_msa)

        edge_attr_norm = modulate(self.norm1_edge(edge_attr, edge_batch), edge_shift_msa, edge_scale_msa)
        messages = self.feature_embedder(torch.cat([x_norm[src], x_norm[tgt], edge_attr_norm, dist], dim=-1))
        x_norm = scatter_mean(messages, src, dim=0)

        # QKV projection
        qkv = self.qkv_proj(x_norm)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q, K = self.norm_q(Q, batch), self.norm_k(K, batch)
        # Reshape Q, K, V to (1, seq_len, num_heads*head_dim)
        if x.dim() == 2:
            Q = Q.unsqueeze(0)
            K = K.unsqueeze(0)
            V = V.unsqueeze(0)
            self.use_rotary = False

        reshaper = functools.partial(einops.rearrange, pattern="b s (h d) -> b h s d", h=self.num_heads)
        # Reshape Q, K, V to (1, num_heads, seq_len, head_dim)
        Q, K, V = map(reshaper, (Q, K, V))

        if x.dim() == 2:
            attn_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(
                0
            )  #! if float it is added as the biasbut would still need a mask s -infs?
        else:
            attn_mask = batch

        if Z is not None:
            if x.dim() == 2:
                mask = torch.ones((x.size(0), x.size(0)))
                if self.mask_z:
                    mask.fill_diagonal_(0)
                attn_mask = attn_mask.float()
                attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
                attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
                bias = (self.pair_bias(Z).squeeze(-1) * mask).unsqueeze(0).unsqueeze(0)
                attn_mask += bias
            else:
                raise ValueError("Have not implemented Batch wise pair embedding update")

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask
        )  # mask [1 1 num_atoms num_atoms] QKV = [1, num_heads, num_atoms, hidden//num_heads]
        attn_output = einops.rearrange(attn_output, "b h s d -> b s (h d)").squeeze(0)
        y = self.out_projection(attn_output)

        # TODO: need to add in gate unsqueeze when we use batch dim
        # Gated Residual
        # x = x + gate_msa * y
        # # Feed Forward
        # edge = edge_attr + edge_gate_msa * self.lin_edge0((y[src] + y[tgt]))
        # x = x + gate_mlp * self.ffn(self.ffn_norm(modulate(self.norm2(x, batch), shift_mlp, scale_mlp), batch))
        # e_in = self.lin_edge1(torch.cat([edge, dist], dim=-1))
        # # import ipdb; ipdb.set_trace()
        # edge_attr = edge + edge_gate_mlp * self.ffn_edge(
        #     self.ffn_norm_edge(modulate(self.norm2_edge(e_in, edge_batch), edge_shift_mlp, edge_scale_mlp), edge_batch)
        # )
        # return x, edge_attr
        # Gated Residual
        x = x + gate_msa * y
        # Feed Forward
        edge = edge_attr + edge_gate_msa * self.lin_edge0((y[src] + y[tgt]))
        x = x + gate_mlp * self.ffn(modulate(self.norm2(x, batch), shift_mlp, scale_mlp))
        e_in = self.lin_edge1(torch.cat([edge, dist], dim=-1))
        # import ipdb; ipdb.set_trace()
        edge_attr = edge + edge_gate_mlp * self.ffn_edge(
            modulate(self.norm2_edge(e_in, edge_batch), edge_shift_mlp, edge_scale_mlp)
        )
        return x, edge_attr


def coord2distfn(x, edge_index, scale_dist_features=1, batch=None):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, 1)
    # import ipdb; ipdb.set_trace()
    if scale_dist_features >= 2:
        # dotproduct = x[row] * x[col]  # shape [num_edges, 3, k]
        # dotproduct = dotproduct.sum(-2)  # sum over the spatial dimension, shape [num_edges, k]
        # # Concatenate the computed features
        # radial = torch.cat([radial, dotproduct], dim=-1)  # shape [num_edges, 2*k]
        dotproduct = (x[row] * x[col]).sum(dim=-2, keepdim=False)  # shape [num_edges, 1]
        radial = torch.cat([radial, dotproduct], dim=-1)  # shape [num_edges, 2]
    if scale_dist_features == 4:
        p_i, p_j = x[edge_index[0]], x[edge_index[1]]
        d_i, d_j = (
            torch.pow(p_i, 2).sum(-2, keepdim=False).clamp(min=1e-6).sqrt(),
            torch.pow(p_j, 2).sum(-2, keepdim=False).clamp(min=1e-6).sqrt(),
        )
        radial = torch.cat([radial, d_i, d_j], dim=-1)

    return radial


class MegalodonDotFN(nn.Module):
    def __init__(
        self,
        num_layers=10,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=256,
        invariant_edge_feat_dim=256,
        atom_classes=22,
        edge_classes=5,
        num_heads=4,
        n_vector_features=128,
        use_pyg=False,
    ):
        super(MegalodonDotFN, self).__init__()
        self.atom_embedder = MLP(atom_classes, invariant_node_feat_dim, invariant_node_feat_dim)
        self.edge_embedder = MLP(edge_classes, invariant_edge_feat_dim, invariant_edge_feat_dim)
        self.num_atom_classes = atom_classes
        self.num_edge_classes = edge_classes
        self.n_vector_features = n_vector_features
        self.coord_emb = nn.Linear(1, n_vector_features, bias=False)
        self.coord_pred = nn.Linear(n_vector_features, 1, bias=False)
        self.scale_dist_features = 4

        self.atom_type_head = PredictionHead(atom_classes, invariant_node_feat_dim)
        self.edge_type_head = PredictionHead(edge_classes, invariant_edge_feat_dim, edge_prediction=True)
        self.time_embedding = TimestepEmbedder(invariant_node_feat_dim)
        self.bond_refine = BondRefine(invariant_node_feat_dim, invariant_edge_feat_dim)

        self.dit_layers = nn.ModuleList()
        self.egnn_layers = nn.ModuleList()
        self.attention_mix_layers = nn.ModuleList()
        self.pre_dist_projection = nn.ModuleList()

        for i in range(num_layers):
            if use_pyg == 2:
                self.dit_layers.append(
                    DiTeBlockOpt(
                        invariant_node_feat_dim,
                        num_heads,
                        use_z=False,
                        scale_dist_features=self.scale_dist_features,
                        n_vector_features=self.n_vector_features,
                    )
                )
            elif use_pyg == 1:
                self.dit_layers.append(
                    PygDiTeBlock(
                        invariant_node_feat_dim, num_heads, use_z=False, scale_dist_features=self.scale_dist_features
                    )
                )
            else:
                self.dit_layers.append(
                    DiTeBlock(
                        invariant_node_feat_dim, num_heads, use_z=False, scale_dist_features=self.scale_dist_features
                    )
                )
            self.egnn_layers.append(XEGNNK(invariant_node_feat_dim, n_vector_features=n_vector_features))
            # self.pre_dist_projection.append(nn.Linear(n_vector_features, 16))

        self.node_blocks = nn.ModuleList(
            [nn.Linear(invariant_node_feat_dim, invariant_node_feat_dim) for i in range(num_layers)]
        )
        self.edge_blocks = nn.ModuleList(
            [nn.Linear(invariant_edge_feat_dim, invariant_edge_feat_dim) for i in range(num_layers)]
        )

    def forward(self, batch, X, H, E_idx, E, t):
        torch.max(batch) + 1
        pos = self.coord_emb(X.unsqueeze(-1))  # N x 3 x K

        H = self.atom_embedder(H)
        E = self.edge_embedder(E)
        te = self.time_embedding(t)
        te_h = te[batch]
        edge_batch, counts = torch.unique(batch, return_counts=True)
        edge_batch = torch.repeat_interleave(edge_batch, counts * (counts - 1))
        te_e = te[edge_batch]
        edge_attr = E

        atom_hids = H
        edge_hids = edge_attr
        # import ipdb; ipdb.set_trace()
        for layer_index in range(len(self.dit_layers)):
            with nvtx.range(f"DiTeBlock Layer {layer_index}"):
                # dist_pos = self.pre_dist_projection[layer_index](pos)
                distances = coord2distfn(pos, E_idx, self.scale_dist_features, batch)  # E x K
                H, edge_attr, messages = self.dit_layers[layer_index](
                    batch, H, te_h, edge_attr, E_idx, te_e, distances, edge_batch
                )

            with nvtx.range(f"XEGNNK Layer {layer_index}"):
                pos = self.egnn_layers[layer_index](batch, pos, H, E_idx, edge_attr, te_e, messages)

            atom_hids = atom_hids + self.node_blocks[layer_index](H)
            edge_hids = edge_hids + self.edge_blocks[layer_index](edge_attr)

        X = self.coord_pred(pos).squeeze(-1)
        x = X - scatter_mean(X, index=batch, dim=0)[batch]

        H = atom_hids
        edge_attr = self.bond_refine(batch, x, H, E_idx, edge_hids)

        h_logits, _ = self.atom_type_head(batch, H)
        e_logits, _ = self.edge_type_head.predict_edges(batch, edge_attr, E_idx)

        out = {
            "x_hat": x,
            "h_logits": h_logits,
            "edge_attr_logits": e_logits,
        }
        return out


class PygDiTeBlock(nn.Module):
    """
    Mimics DiT block
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_expansion_ratio=4.0,
        use_z=True,
        mask_z=True,
        use_rotary=False,
        n_vector_features=128,
        scale_dist_features=1,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.norm1 = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm2 = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        if scale_dist_features > 1:
            dist_size = n_vector_features * scale_dist_features
        else:
            dist_size = n_vector_features
        self.feature_embedder = MLP(hidden_size + hidden_size + hidden_size + dist_size, hidden_size, hidden_size)
        self.norm1_edge = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm2_edge = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)

        self.ffn_norm = BatchLayerNorm(hidden_size)
        self.ffn = swiglu_ffn(hidden_size, mlp_expansion_ratio, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_edge_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        # Single linear layer for QKV projection
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.norm_q = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm_k = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.out_projection = nn.Linear(hidden_size, hidden_size, bias=False)

        self.use_rotary = use_rotary
        self.d_head = hidden_size // num_heads

        if use_z:
            self.use_z = use_z
            self.pair_bias = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 1, bias=False))
            self.mask_z = mask_z

        self.lin_edge0 = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.lin_edge1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_edge1 = nn.Linear(hidden_size + n_vector_features * scale_dist_features, hidden_size, bias=False)
        self.ffn_norm_edge = BatchLayerNorm(hidden_size)
        self.ffn_edge = swiglu_ffn(hidden_size, mlp_expansion_ratio, bias=False)
        # self.tanh = nn.GELU(approximate='tanh')

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.num_heads, self.d_head))
        k = k.unflatten(-1, (self.num_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(
        self,
        batch: torch.Tensor,
        x: torch.Tensor,
        t_emb_h: torch.Tensor,
        edge_attr: torch.Tensor = None,
        edge_index: torch.Tensor = None,
        t_emb_e: torch.Tensor = None,
        dist: torch.Tensor = None,
        edge_batch: torch.Tensor = None,
        Z: torch.Tensor = None,
    ):
        """
        This assume pytorch geometric batching so batch size of 1 so skip rotary as it depends on having an actual batch

        batch: N
        x: N x 256
        temb: N x 256
        edge_attr: E x 256
        edge_index: 2 x E
        """
        if Z is not None:
            assert self.use_z

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb_h).chunk(6, dim=1)
        (
            edge_shift_msa,
            edge_scale_msa,
            edge_gate_msa,
            edge_shift_mlp,
            edge_scale_mlp,
            edge_gate_mlp,
        ) = self.adaLN_edge_modulation(t_emb_e).chunk(6, dim=1)
        src, tgt = edge_index
        # Normalize x
        x_norm = modulate(self.norm1(x, batch), shift_msa, scale_msa)

        edge_attr_norm = modulate(self.norm1_edge(edge_attr, edge_batch), edge_shift_msa, edge_scale_msa)
        # import ipdb; ipdb.set_trace()
        messages = self.feature_embedder(torch.cat([x_norm[src], x_norm[tgt], edge_attr_norm, dist], dim=-1))
        x_norm = scatter_mean(messages, src, dim=0)

        # QKV projection
        qkv = self.qkv_proj(x_norm)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q, K = self.norm_q(Q, batch), self.norm_k(K, batch)
        # Reshape Q, K, V to (1, seq_len, num_heads*head_dim)
        # if x.dim() == 2:
        #     Q = Q.unsqueeze(0)
        #     K = K.unsqueeze(0)
        #     V = V.unsqueeze(0)
        #     self.use_rotary = False

        reshaper = functools.partial(einops.rearrange, pattern="s (h d) -> h s d", h=self.num_heads)
        # Reshape Q, K, V to (1, num_heads, seq_len, head_dim)
        Q, K, V = map(reshaper, (Q, K, V))

        # if x.dim() == 2:
        #     attn_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
        #     attn_mask = attn_mask.unsqueeze(0).unsqueeze(
        #         0
        #     )  #! if float it is added as the biasbut would still need a mask s -infs?
        # else:
        #     attn_mask = batch

        # if Z is not None:
        #     if x.dim() == 2:
        #         mask = torch.ones((x.size(0), x.size(0)))
        #         if self.mask_z:
        #             mask.fill_diagonal_(0)
        #         attn_mask = attn_mask.float()
        #         attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
        #         attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
        #         bias = (self.pair_bias(Z).squeeze(-1) * mask).unsqueeze(0).unsqueeze(0)
        #         attn_mask += bias
        #     else:
        #         raise ValueError("Have not implemented Batch wise pair embedding update")

        attn_output = self.pyg_mha(Q, K, V, src, tgt, x)
        attn_output = einops.rearrange(attn_output, "h s d -> s (h d)")  # .squeeze(0)
        y = self.out_projection(attn_output)
        # Gated Residual
        x = x + gate_msa * y
        # Feed Forward
        edge = edge_attr + edge_gate_msa * self.lin_edge0((y[src] + y[tgt]))
        x = x + gate_mlp * self.ffn(self.ffn_norm(modulate(self.norm2(x, batch), shift_mlp, scale_mlp), batch))
        e_in = self.lin_edge1(torch.cat([edge, dist], dim=-1))
        # import ipdb; ipdb.set_trace()
        edge_attr = edge + edge_gate_mlp * self.ffn_edge(
            self.ffn_norm_edge(modulate(self.norm2_edge(e_in, edge_batch), edge_shift_mlp, edge_scale_mlp), edge_batch)
        )
        return x, edge_attr

    def pyg_mha(self, Q, K, V, src, tgt, pos):
        # Gather the query, key, and value tensors for the source and target nodes
        query_i = Q[:, src]  # [heads, E, d_head]
        key_j = K[:, tgt]
        value_j = V[:, tgt]

        # Compute the attention scores using a dot-product attention mechanism
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.d_head)  # [heads, E]

        # Apply softmax to normalize attention scores across all edges directed to the same node
        alpha = softmax(alpha, tgt, dim=-1)  # [E, heads]
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # Apply dropout to attention scores
        # Multiply normalized attention scores with the value tensor to compute the messages
        msg = value_j * alpha.unsqueeze(-1)  # .view(-1, self.num_heads, 1)  # [E, heads, d_head]

        # Aggregate messages to the destination nodes
        out = scatter(msg, tgt, dim=1, reduce='sum', dim_size=pos.size(0))  # [N, heads, d_head]

        return out


class DiTeBlockOpt(nn.Module):
    """
    Mimics DiT block
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_expansion_ratio=4.0,
        use_z=False,
        mask_z=True,
        use_rotary=False,
        n_vector_features=128,
        scale_dist_features=4,
        dropout=0.0,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        # self.edge_emb = nn.Linear(hidden_size + scale_dist_features*n_vector_features, hidden_size)
        self.norm1 = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm2 = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)

        self.norm1_edge = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm2_edge = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)

        self.norm2_node = BatchLayerNorm(hidden_size)
        self.ffn = swiglu_ffn(hidden_size, mlp_expansion_ratio, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_edge_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        # Single linear layer for QKV projection
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        # self.norm_q = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        # self.norm_k = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.out_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        if scale_dist_features > 1:
            dist_size = n_vector_features * scale_dist_features
        else:
            dist_size = n_vector_features
        self.feature_embedder = MLP(hidden_size + hidden_size + hidden_size + dist_size, hidden_size, hidden_size)

        self.use_rotary = False
        self.d_head = hidden_size // num_heads
        self.use_z = use_z
        if use_z:
            self.use_z = use_z
            self.pair_bias = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 1, bias=False))
            self.mask_z = mask_z

        # self.node2edge_lin = nn.Linear(hidden_size, hidden_size)
        # self.lin_edge0 = nn.Linear(hidden_size, hidden_size, bias=False)
        # # self.lin_edge1 = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.lin_edge1 = nn.Linear(hidden_size, hidden_size, bias=False)
        # # self.ffn_norm_edge = BatchLayerNorm(hidden_size)
        # self.ffn_edge = swiglu_ffn(hidden_size, mlp_expansion_ratio, bias=False)
        # self.tanh = nn.GELU(approximate='tanh')
        # self.ffn_norm = BatchLayerNorm(hidden_size)
        self.lin_edge0 = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.lin_edge1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_edge1 = nn.Linear(hidden_size + n_vector_features * scale_dist_features, hidden_size, bias=False)
        self.ffn_norm_edge = BatchLayerNorm(hidden_size)
        self.ffn_edge = swiglu_ffn(hidden_size, mlp_expansion_ratio, bias=False)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.num_heads, self.d_head))
        k = k.unflatten(-1, (self.num_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(
        self,
        batch: torch.Tensor,
        x: torch.Tensor,
        t_emb_h: torch.Tensor,
        edge_attr: torch.Tensor = None,
        edge_index: torch.Tensor = None,
        t_emb_e: torch.Tensor = None,
        dist: torch.Tensor = None,
        edge_batch: torch.Tensor = None,
        Z: torch.Tensor = None,
    ):
        """
        This assume pytorch geometric batching so batch size of 1 so skip rotary as it depends on having an actual batch

        batch: N
        x: N x 256
        temb: N x 256
        edge_attr: E x 256
        edge_index: 2 x E
        """
        if Z is not None:
            assert self.use_z
        src, tgt = edge_index
        edge_batch = batch[src]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb_h).chunk(6, dim=1)
        (
            edge_shift_msa,
            edge_scale_msa,
            edge_gate_msa,
            edge_shift_mlp,
            edge_scale_mlp,
            edge_gate_mlp,
        ) = self.adaLN_edge_modulation(t_emb_e).chunk(6, dim=1)
        # edge_attr = self.edge_emb(torch.cat([edge_attr, dist], dim=-1))
        # Normalize x
        x_norm = modulate(self.norm1(x, batch), shift_msa, scale_msa)
        edge_attr_norm = modulate(self.norm1_edge(edge_attr, edge_batch), edge_shift_msa, edge_scale_msa)

        messages = self.feature_embedder(torch.cat([x_norm[src], x_norm[tgt], edge_attr_norm, dist], dim=-1))
        x_norm = scatter_mean(messages, src, dim=0)

        qkv = self.qkv_proj(x_norm)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q.reshape(-1, self.num_heads, self.d_head)
        K = K.reshape(-1, self.num_heads, self.d_head)
        V = V.reshape(-1, self.num_heads, self.d_head)

        # Gather the query, key, and value tensors for the source and target nodes
        query_i = Q[src]  # [E, heads, d_head]
        key_j = K[tgt]  # [E, heads, d_head]
        value_j = V[tgt]  # [E, heads, d_head]

        # Compute the attention scores using dot-product attention mechanism
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.d_head)  # [E, heads]

        # Apply softmax to normalize attention scores across all edges directed to the same node
        alpha = softmax(alpha, tgt)  # [E, heads]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # Apply dropout to attention scores

        # Multiply normalized attention scores with the value tensor to compute the messages
        # import ipdb; ipdb.set_trace()
        msg = value_j
        msg = msg * alpha.view(-1, self.num_heads, 1)  # [E, heads, d_head]

        # Aggregate messages to the destination nodes
        out = scatter(msg, tgt, dim=0, reduce='sum', dim_size=x.size(0))  # [N, heads, d_head]

        # Merge the heads and the output channels
        out = out.view(
            -1, self.num_heads * self.d_head
        )  # [N, heads * d_head] #? no linear layer for the first h_node? to aggregate the MHA?
        y = self.out_projection(out)

        # Gated Residual
        x = x + gate_msa * y
        # Feed Forward
        edge = edge_attr + edge_gate_msa * self.lin_edge0((y[src] + y[tgt]))
        x = x + gate_mlp * self.ffn(modulate(self.norm2(x, batch), shift_mlp, scale_mlp))
        e_in = self.lin_edge1(torch.cat([edge, dist], dim=-1))
        # import ipdb; ipdb.set_trace()
        edge_attr = edge + edge_gate_mlp * self.ffn_edge(
            modulate(self.norm2_edge(e_in, edge_batch), edge_shift_mlp, edge_scale_mlp)
        )
        return x, edge_attr, messages


if __name__ == "__main__":
    # import torch

    # Sample data for the first graph
    num_nodes_1 = 3
    node_features_1 = 22
    edge_features_1 = 5

    ligand_pos = torch.rand((75, 3))
    batch_ligand = torch.Tensor(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
        ]
    ).to(torch.int64)
    ligand_feats = torch.Tensor(
        [
            2,
            4,
            2,
            4,
            2,
            4,
            4,
            3,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            5,
            1,
            3,
            1,
            1,
            1,
            2,
            4,
            2,
            4,
            2,
            4,
            4,
            3,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            5,
            1,
            3,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            12,
            2,
            5,
            2,
            3,
            5,
            1,
            5,
            2,
            4,
            2,
            4,
            2,
            4,
            4,
            3,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            5,
            1,
            3,
            1,
            1,
            1,
        ]
    ).to(torch.int64)
    num_classes = node_features_1
    # Initialize the adjacency matrix with zeros
    adj_matrix = torch.zeros((75, 75, 5), dtype=torch.int64)
    no_bond = torch.zeros(edge_features_1)
    no_bond[0] = 1
    # Using broadcasting to create the adjacency matrix
    adj_matrix[batch_ligand.unsqueeze(1) == batch_ligand] = 1
    for idx, i in enumerate(batch_ligand):
        for jdx, j in enumerate(batch_ligand):
            if idx == jdx:
                adj_matrix[idx][jdx] = no_bond
            elif i == j:
                # import ipdb; ipdb.set_trace()
                adj_matrix[idx][jdx] = torch.nn.functional.one_hot(torch.randint(0, 5, (1,)), 5).squeeze(0)
    # print(adj_matrix)

    X = ligand_pos
    H = F.one_hot(ligand_feats, num_classes).float()
    A = adj_matrix
    mask = batch_ligand.unsqueeze(1) == batch_ligand.unsqueeze(0)  # Shape: (75, 75)
    E_idx = mask.nonzero(as_tuple=False).t()
    self_loops = E_idx[0] != E_idx[1]
    E_idx = E_idx[:, self_loops]
    source, target = E_idx
    E = A[source, target].float()
    batch = batch_ligand
    t = torch.tensor([0.5, 0.5, 0.5, 0.5])

    # Instantiate the model

    # model = MegalodonDotFN(
    #     num_layers=10,  # Number of layers
    #     equivariant_node_feature_dim=3,
    #     invariant_node_feat_dim=256,
    #     invariant_edge_feat_dim=256,
    #     atom_classes=22,
    #     edge_classes=5,
    #     num_heads=4,
    #     n_vector_features=16,
    #     use_pyg=2
    # ).cuda()
    from bionemo.model.molecule.moco.arch.final_model import MoleculeDiffusion

    model = MoleculeDiffusion(use_z=None).cuda()
    batch, X, H, E_idx, E, t = batch.cuda(), X.cuda(), H.cuda(), E_idx.cuda(), E.cuda(), t.cuda()
    # Forward pass through the model
    output = model(batch, X, H, E_idx, E, t)

    # Output contains predicted coordinates, node types, and edge types
    print("Predicted Coordinates:", output["x_hat"])
    print("Node Type Logits:", output["h_logits"])
    print("Edge Attribute Logits:", output["edge_attr_logits"])
#
# nsys profile --trace=cuda,nvtx,osrt --output=profile_report python ./bionemo/model/molecule/moco/arch/fn_arch.py

# nsys stats /tmp/nsys-report-21a1.nsys-rep
