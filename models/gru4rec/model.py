from __future__ import annotations

import torch
import torch.nn as nn


class GRU4Rec(nn.Module):
    def __init__(
        self,
        num_items: int,
        d_model: int = 64,
        n_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model

        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=(dropout if n_layers > 1 else 0.0),
        )
        self.output_norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0.0)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, L], 0 is padding id.
        x = self.item_embedding(seq)
        non_pad_mask = (seq != 0).unsqueeze(-1).to(x.dtype)
        x = x * non_pad_mask
        x = self.input_dropout(x)
        h, _ = self.gru(x)
        h = self.output_norm(h)
        return h

    def score_candidates(self, hidden: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        # hidden: [B, D], candidates: [B, C]
        candidate_emb = self.item_embedding(candidates)
        return torch.bmm(candidate_emb, hidden.unsqueeze(-1)).squeeze(-1)
