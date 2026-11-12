"""Joint spectrum prediction for a pair of cross-linked peptides."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from spectrum_transformer.config import (
    AUX_FEATURE_DIM,
    DEFAULT_D_MODEL,
    DEFAULT_DROPOUT,
    DEFAULT_FF_DIM,
    DEFAULT_N_HEADS,
    DEFAULT_N_LAYERS,
    DEFAULT_SEQUENCE_CONTEXT_DIM,
    INPUT_CHANNELS,
    MAIN_TOKEN_IDS,
    MAX_CLEAVAGE_LEN,
    NUM_TOKENS,
    OUTPUT_CHANNELS,
    SATELLITE_TOKEN_IDS,
)
from spectrum_transformer.models.transformer import V20FamilyReadout, V20TransformerStage


class XLSpectrumTransformer(nn.Module):
    """Encode each chain with shared weights, then attend over the whole peptide pair."""

    def __init__(
        self,
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        d_model=DEFAULT_D_MODEL,
        n_heads=DEFAULT_N_HEADS,
        n_layers=DEFAULT_N_LAYERS,
        ff_dim=DEFAULT_FF_DIM,
        dropout=DEFAULT_DROPOUT,
        aux_dim=AUX_FEATURE_DIM,
        sequence_context_dim=DEFAULT_SEQUENCE_CONTEXT_DIM,
    ):
        super().__init__()
        if (in_channels, out_channels) != (INPUT_CHANNELS, OUTPUT_CHANNELS):
            raise ValueError("XL requires six input and output fragment channels.")
        if n_layers < 2:
            raise ValueError("XL requires at least one shared layer and one charge layer.")

        self.input_proj = nn.Linear(in_channels, d_model)
        self.aux_proj = nn.Linear(aux_dim, d_model)
        self.aa_embedding = nn.Embedding(28, sequence_context_dim)
        self.scalar_proj = nn.Linear(11, sequence_context_dim)
        self.context_proj = nn.Linear(2 * sequence_context_dim, d_model)
        self.charge_embedding = nn.Embedding(8, sequence_context_dim)
        self.length_embedding = nn.Embedding(MAX_CLEAVAGE_LEN + 2, sequence_context_dim)
        self.subgroup_proj = nn.Linear(2 * sequence_context_dim, d_model)

        family_index = torch.zeros(NUM_TOKENS, dtype=torch.long)
        family_index[list(SATELLITE_TOKEN_IDS)] = 1
        family_index[list(MAIN_TOKEN_IDS)] = 2
        self.register_buffer("token_family_index", family_index, persistent=False)
        self.family_embedding = nn.Embedding(3, d_model)
        self.family_context_gain = nn.Parameter(torch.tensor(0.10))
        self.packed_pos_p = nn.Embedding(2, d_model)
        self.packed_pos_h = nn.Embedding(MAX_CLEAVAGE_LEN, d_model)
        self.packed_pos_t = nn.Embedding(NUM_TOKENS, d_model)

        self.shared_encoder = V20TransformerStage(d_model, n_heads, ff_dim, dropout, n_layers - 1)
        self.z2_encoder = V20TransformerStage(d_model, n_heads, ff_dim, dropout, 1)
        self.z3_encoder = V20TransformerStage(d_model, n_heads, ff_dim, dropout, 1)
        self.z4plus_encoder = V20TransformerStage(d_model, n_heads, ff_dim, dropout, 1)
        self.z2_group_residual_scale = nn.Parameter(torch.tensor(0.08))
        self.z3_group_residual_scale = nn.Parameter(torch.tensor(0.10))
        self.z4_group_residual_scale = nn.Parameter(torch.tensor(0.01))
        self.group_merge_norm = nn.LayerNorm(d_model)
        self.packed_norm = nn.LayerNorm(d_model)

        self.backbone_head = V20FamilyReadout(d_model, out_channels)
        self.main_anchor_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, out_channels),
        )
        self.backbone_gain = nn.Parameter(torch.tensor(1.05))
        for head in (self.backbone_head.main_head, self.backbone_head.satellite_head,
                     self.backbone_head.residue_head):
            nn.init.constant_(head[-1].bias, -1.00)
        nn.init.constant_(self.main_anchor_head[-1].bias, -0.85)

    def _route_charge(self, shared, key_padding_mask, charge):
        routed = shared
        for selected, encoder, scale in (
            (charge == 2, self.z2_encoder, self.z2_group_residual_scale),
            (charge == 3, self.z3_encoder, self.z3_group_residual_scale),
            (charge >= 4, self.z4plus_encoder, self.z4_group_residual_scale),
        ):
            indices = selected.nonzero(as_tuple=True)[0]
            if indices.numel() > 0:
                rows = shared.index_select(0, indices)
                mask = key_padding_mask.index_select(0, indices)
                encoded = encoder(rows, src_key_padding_mask=mask)
                routed = routed.index_copy(0, indices, rows + scale * (encoded - rows))
        return self.packed_norm(self.group_merge_norm(routed))

    def _encode_packed(self, x, cell_valid, charge):
        _, _, cleavages, tokens, hidden = x.shape
        flat_x = x.flatten(1, 3)
        flat_valid = cell_valid.flatten(1)
        output = torch.zeros_like(flat_x)

        # An all-padding row has no valid keys for attention. Leave it at zero.
        samples = flat_valid.any(dim=1).nonzero(as_tuple=True)[0]
        if samples.numel() == 0:
            return output.view_as(x)

        active = flat_valid.index_select(0, samples)
        counts = active.sum(dim=1)
        max_count = int(counts.max().item())
        order = active.float().argsort(dim=1, descending=True, stable=True)[:, :max_count]
        key_padding_mask = torch.arange(max_count, device=x.device)[None, :] >= counts[:, None]
        packed = flat_x[samples[:, None], order].masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        # Indices refer to the original grid, with cleavage positions local to each chain.
        chain_idx = order // (cleavages * tokens)
        cleavage_idx = (order // tokens) % cleavages
        token_idx = order % tokens
        packed = (
            packed + self.packed_pos_p(chain_idx) + self.packed_pos_h(cleavage_idx)
            + self.packed_pos_t(token_idx)
        )
        shared = self.shared_encoder(packed, src_key_padding_mask=key_padding_mask)
        routed = self._route_charge(shared, key_padding_mask, charge.index_select(0, samples))

        restored = routed.new_zeros((samples.numel(), flat_x.size(1), hidden))
        restored.scatter_(
            1,
            order.unsqueeze(-1).expand_as(routed),
            routed.masked_fill(key_padding_mask.unsqueeze(-1), 0.0),
        )
        return output.index_copy(0, samples, restored.to(output.dtype)).view_as(x)

    def forward(
        self,
        x,
        aux_features,
        *,
        valid_mask,
        charge,
        lengths,
        cleavage_aa_ids,
        cleavage_scalar_features,
    ):
        if (x.ndim != 5 or x.shape[1] != 2 or not 1 <= x.shape[2] <= MAX_CLEAVAGE_LEN
                or x.shape[-2:] != (NUM_TOKENS, INPUT_CHANNELS)):
            raise ValueError("XL input must have shape (batch, 2, cleavage, 46, 6).")
        if valid_mask.shape != x.shape:
            raise ValueError("XL valid_mask must have the same shape as the input.")
        valid_mask = valid_mask.bool()

        residue = self.aa_embedding(cleavage_aa_ids).mean(dim=-2)
        scalars = self.scalar_proj(cleavage_scalar_features)
        context = self.context_proj(torch.cat((residue, scalars), dim=-1))
        charge_context = self.charge_embedding(charge.clamp(0, 7))[:, None, :].expand(-1, 2, -1)
        length_context = self.length_embedding(lengths.clamp(0, MAX_CLEAVAGE_LEN + 1))
        subgroup = self.subgroup_proj(torch.cat((charge_context, length_context), dim=-1))
        family = self.family_embedding(self.token_family_index)
        x = (
            self.input_proj(x)
            + self.aux_proj(aux_features)[:, None, None, None, :]
            + context.unsqueeze(-2)
            + subgroup[:, :, None, None, :]
            + self.family_context_gain * family
        )
        x = self._encode_packed(x, valid_mask.any(dim=-1), charge)

        logits = self.backbone_head(x, self.token_family_index)
        prediction = (self.backbone_gain * F.softplus(logits)).masked_fill(~valid_mask, 0.0)
        main_anchor_logits = self.main_anchor_head(x)
        main_anchor_prob = torch.sigmoid(main_anchor_logits).masked_fill(~valid_mask, 0.0)
        return {
            "prediction": prediction,
            "prediction_gate": main_anchor_prob,
            "prediction_gain": main_anchor_prob,
            "backbone_pred": prediction,
            "detail_pred": prediction,
            "detail_logits": logits,
            "main_anchor_prob": main_anchor_prob,
            "main_anchor_logits": main_anchor_logits,
        }
