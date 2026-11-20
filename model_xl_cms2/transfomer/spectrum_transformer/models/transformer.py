"""V20 native spectrum transformer model.

V20-Evidence keeps the most stable final prediction path and moves the evidence-
driven intervention into training targets and weighting, not speculative
generation structure.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from spectrum_transformer.config import (
    MAIN_TOKEN_IDS,
    MAX_CLEAVAGE_LEN,
    NUM_TOKENS,
    RESIDUE_EVENT_TOKEN_IDS,
    SATELLITE_TOKEN_IDS,
)


class V20PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward(self, x):
        return x + self.embedding[:, : x.size(1)]


class V20GroupAdapter(nn.Module):
    def __init__(self, d_model: int, bottleneck: int = 64, init_scale: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d_model),
        )
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        return self.scale * self.net(x)


class V20FamilyReadout(nn.Module):
    def __init__(self, d_model: int, out_channels: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = int(hidden_dim or d_model)
        self.residue_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )
        self.satellite_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )
        self.main_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, x, token_family_index):
        residue = self.residue_head(x)
        satellite = self.satellite_head(x)
        main = self.main_head(x)
        family_index = token_family_index.view(1, 1, -1, 1)
        out = torch.where(
            family_index == 2,
            main,
            torch.where(family_index == 1, satellite, residue),
        )
        return out


class V20TransformerStage(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float, num_layers: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class V20SpectrumTransformer(nn.Module):
    def __init__(
        self,
        in_channels=6,
        out_channels=6,
        d_model=256,
        n_heads=8,
        n_layers=6,
        ff_dim=1024,
        dropout=0.15,
        aux_dim=3,
        use_presence_head=True,
        use_sequence_context=True,
        sequence_context_dim=64,
        use_packed=True,
        model_version="v20",
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = d_model
        self.aux_dim = aux_dim
        self.model_version = model_version
        self.use_sequence_context = bool(use_sequence_context)
        self.use_packed = bool(use_packed)

        self.input_proj = nn.Linear(in_channels, d_model)
        self.aux_proj = nn.Linear(aux_dim, d_model)
        self.pos_encoding = V20PositionalEncoding(max_len=MAX_CLEAVAGE_LEN * NUM_TOKENS, d_model=d_model)
        self.packed_pos_h = nn.Embedding(MAX_CLEAVAGE_LEN, d_model)
        self.packed_pos_w = nn.Embedding(NUM_TOKENS, d_model)
        self.sequence_context_dim = sequence_context_dim
        self.aa_embedding = nn.Embedding(27 + 1, sequence_context_dim)
        self.scalar_proj = nn.Linear(7, sequence_context_dim)
        self.context_proj = nn.Linear(sequence_context_dim * 2, d_model)
        self.charge_embedding = nn.Embedding(8, sequence_context_dim)
        self.length_embedding = nn.Embedding(41, sequence_context_dim)
        self.subgroup_proj = nn.Linear(sequence_context_dim * 2, d_model)
        family_index = torch.zeros((NUM_TOKENS,), dtype=torch.long)
        for idx in SATELLITE_TOKEN_IDS:
            family_index[idx] = 1
        for idx in MAIN_TOKEN_IDS:
            family_index[idx] = 2
        self.register_buffer("token_family_index", family_index, persistent=False)
        self.family_embedding = nn.Embedding(3, d_model)
        self.family_context_gain = nn.Parameter(torch.tensor(0.10))

        shared_layers = max(1, n_layers - 1)
        group_layers = 1
        self.shared_encoder = V20TransformerStage(
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=shared_layers,
        )
        self.z2_encoder = V20TransformerStage(
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=group_layers,
        )
        self.z3_encoder = V20TransformerStage(
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=group_layers,
        )
        self.z4plus_encoder = V20TransformerStage(
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=group_layers,
        )
        self.group_merge_norm = nn.LayerNorm(d_model)
        self.z2_group_residual_scale = nn.Parameter(torch.tensor(0.08))
        self.z3_group_residual_scale = nn.Parameter(torch.tensor(0.10))
        self.z4_group_residual_scale = nn.Parameter(torch.tensor(0.01))

        self.backbone_head = V20FamilyReadout(d_model=d_model, out_channels=out_channels, hidden_dim=d_model)
        self.main_anchor_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, out_channels),
        )
        self.packed_norm = nn.LayerNorm(d_model)
        self.backbone_gain = nn.Parameter(torch.tensor(1.05))
        self._init_head_biases()

    def _init_head_biases(self):
        with torch.no_grad():
            nn.init.constant_(self.backbone_head.main_head[-1].bias, -1.00)
            nn.init.constant_(self.backbone_head.satellite_head[-1].bias, -1.00)
            nn.init.constant_(self.backbone_head.residue_head[-1].bias, -1.00)
            nn.init.constant_(self.main_anchor_head[-1].bias, -0.85)

    def _pack_grid(self, x, cell_valid):
        b, h, w, d = x.shape
        flat_x = x.reshape(b, h * w, d)
        flat_mask = cell_valid.reshape(b, h * w)
        n_valid = flat_mask.sum(dim=1).long()
        n_max = int(n_valid.max().item())
        _, sort_idx = flat_mask.float().sort(dim=1, descending=True, stable=True)
        packed = flat_x.gather(1, sort_idx.unsqueeze(-1).expand(-1, -1, d))[:, :n_max, :]
        h_idx = (sort_idx[:, :n_max] // w).long()
        w_idx = (sort_idx[:, :n_max] % w).long()
        key_padding_mask = torch.arange(n_max, device=x.device).unsqueeze(0) >= n_valid.unsqueeze(1)
        return packed, h_idx, w_idx, key_padding_mask, sort_idx[:, :n_max], flat_mask[:, : h * w]

    def _unpack_grid(self, packed, scatter_idx, cell_valid, b, h, w, d):
        grid = packed.new_zeros(b, h * w, d)
        if packed.size(1) > 0:
            grid.scatter_(
                1,
                scatter_idx.unsqueeze(-1).expand(-1, -1, d),
                packed,
            )
        if cell_valid is not None:
            grid = grid * cell_valid.reshape(b, h * w, 1).to(grid.dtype)
        return grid.reshape(b, h, w, d)

    def forward(
        self,
        x,
        aux_features,
        valid_mask=None,
        cleavage_aa_ids=None,
        cleavage_scalar_features=None,
        charge=None,
        length=None,
        **kwargs,
    ):
        # x: (B, 39, 46, C)
        b, h, w, c = x.shape
        x = self.input_proj(x)
        family_embed = self.family_embedding(self.token_family_index).view(1, 1, w, self.d_model)
        x = x + self.family_context_gain * family_embed

        aux = self.aux_proj(aux_features).view(b, 1, 1, self.d_model)
        x = x + aux

        if cleavage_aa_ids is not None and cleavage_scalar_features is not None:
            aa_embed = self.aa_embedding(cleavage_aa_ids)  # (B, 39, 2, Dctx)
            aa_embed = aa_embed.mean(dim=2)
            scalar_embed = self.scalar_proj(cleavage_scalar_features)
            context = torch.cat([aa_embed, scalar_embed], dim=-1)
            context = self.context_proj(context).unsqueeze(2).expand(-1, -1, w, -1)
            x = x + context

        if charge is not None and length is not None:
            charge_ids = charge.clamp(min=0, max=7).long()
            length_ids = length.clamp(min=0, max=40).long()
            subgroup_embed = torch.cat(
                [
                    self.charge_embedding(charge_ids),
                    self.length_embedding(length_ids),
                ],
                dim=-1,
            )
            subgroup_context = self.subgroup_proj(subgroup_embed).view(b, 1, 1, self.d_model)
            x = x + subgroup_context

        cell_valid = None
        if valid_mask is not None:
            cell_valid = valid_mask.any(dim=-1)

        if self.use_packed and cell_valid is not None:
            packed, h_idx, w_idx, key_padding_mask, scatter_idx, flat_mask = self._pack_grid(x, cell_valid)
            if packed.size(1) > 0:
                packed = packed + self.packed_pos_h(h_idx) + self.packed_pos_w(w_idx)
                packed = self.shared_encoder(packed, src_key_padding_mask=key_padding_mask)
                if charge is not None:
                    z2_mask = (charge == 2).float().view(b, 1, 1)
                    z3_mask = (charge == 3).float().view(b, 1, 1)
                    z4_mask = (charge >= 4).float().view(b, 1, 1)
                else:
                    z2_mask = packed.new_zeros((b, 1, 1))
                    z3_mask = packed.new_zeros((b, 1, 1))
                    z4_mask = packed.new_zeros((b, 1, 1))
                packed_z2 = self.z2_encoder(packed, src_key_padding_mask=key_padding_mask)
                packed_z3 = self.z3_encoder(packed, src_key_padding_mask=key_padding_mask)
                packed_z4 = self.z4plus_encoder(packed, src_key_padding_mask=key_padding_mask)
                packed = self.group_merge_norm(
                    packed
                    + self.z2_group_residual_scale * z2_mask * (packed_z2 - packed)
                    + self.z3_group_residual_scale * z3_mask * (packed_z3 - packed)
                    + self.z4_group_residual_scale * z4_mask * (packed_z4 - packed)
                )
                packed = self.packed_norm(packed)
            x = self._unpack_grid(packed, scatter_idx, flat_mask, b, h, w, self.d_model)
        else:
            x = x.view(b, h * w, self.d_model)
            x = self.pos_encoding(x)
            x = self.shared_encoder(x)
            if charge is not None:
                z2_mask = (charge == 2).float().view(b, 1, 1)
                z3_mask = (charge == 3).float().view(b, 1, 1)
                z4_mask = (charge >= 4).float().view(b, 1, 1)
            else:
                z2_mask = x.new_zeros((b, 1, 1))
                z3_mask = x.new_zeros((b, 1, 1))
                z4_mask = x.new_zeros((b, 1, 1))
            x_z2 = self.z2_encoder(x)
            x_z3 = self.z3_encoder(x)
            x_z4 = self.z4plus_encoder(x)
            x = self.group_merge_norm(
                x
                + self.z2_group_residual_scale * z2_mask * (x_z2 - x)
                + self.z3_group_residual_scale * z3_mask * (x_z3 - x)
                + self.z4_group_residual_scale * z4_mask * (x_z4 - x)
            )
            x = x.view(b, h, w, self.d_model)

        backbone_logits = self.backbone_head(x, self.token_family_index).view(b, h, w, self.out_channels)
        backbone = self.backbone_gain * F.softplus(backbone_logits)
        main_anchor_logits = self.main_anchor_head(x).view(b, h, w, self.out_channels)
        main_anchor_prob = torch.sigmoid(main_anchor_logits)
        prediction_gate = main_anchor_prob
        prediction_gain = prediction_gate
        prediction = backbone

        if valid_mask is not None:
            backbone = backbone * valid_mask
            main_anchor_prob = main_anchor_prob * valid_mask
            prediction_gate = prediction_gate * valid_mask
            prediction = prediction * valid_mask

        return {
            "prediction": prediction,
            "prediction_gate": prediction_gate,
            "prediction_gain": prediction_gain,
            "backbone_pred": backbone,
            "detail_pred": backbone,
            "detail_logits": backbone_logits,
            "main_anchor_prob": main_anchor_prob,
            "main_anchor_logits": main_anchor_logits,
        }


# Backward-compatible name for the current V20 factory.
AxialSpectrumTransformerV11 = V20SpectrumTransformer
