from __future__ import annotations

import torch
import torch.nn as nn


class BERT4Rec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.mask_id = num_items + 1

        # vocab ids: 0 is padding, [1..num_items] are items, num_items+1 is [MASK].
        self.item_embedding = nn.Embedding(num_items + 2, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.input_dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0.0)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, L], ids in [0..num_items+1]
        batch_size, seq_len = seq.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds max_seq_len ({self.max_seq_len})."
            )

        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_embedding(seq) + self.position_embedding(positions)
        non_pad_mask = (seq != 0).unsqueeze(-1).to(x.dtype)
        x = x * non_pad_mask
        x = self.input_norm(x)
        x = self.input_dropout(x)

        padding_mask = seq.eq(0)
        h = self.encoder(x, src_key_padding_mask=padding_mask)
        h = self.output_norm(h)
        return h

    def score_candidates(self, hidden: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        # hidden: [B, D], candidates: [B, C]
        candidate_emb = self.item_embedding(candidates)
        return torch.bmm(candidate_emb, hidden.unsqueeze(-1)).squeeze(-1)

