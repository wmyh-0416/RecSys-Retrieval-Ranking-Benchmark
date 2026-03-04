from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Module:
    if hidden_dim <= 0:
        return nn.Identity()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class DualTowerRetrieval(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        tower_hidden_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)

        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.user_tower = _build_mlp(embedding_dim, tower_hidden_dim, embedding_dim, dropout)
        self.item_tower = _build_mlp(embedding_dim, tower_hidden_dim, embedding_dim, dropout)
        self.user_norm = nn.LayerNorm(embedding_dim)
        self.item_norm = nn.LayerNorm(embedding_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0.0)
        for module in (self.user_tower, self.item_tower):
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)

    def encode_user(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, L], padding id = 0
        seq_emb = self.item_embedding(seq)  # [B, L, D]
        mask = (seq != 0).unsqueeze(-1).to(seq_emb.dtype)
        summed = (seq_emb * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / lengths
        user_vec = self.user_tower(pooled)
        user_vec = self.user_norm(user_vec)
        return F.normalize(user_vec, dim=-1)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        # item_ids: [N] or [B, C]
        item_vec = self.item_embedding(item_ids)
        item_vec = self.item_tower(item_vec)
        item_vec = self.item_norm(item_vec)
        return F.normalize(item_vec, dim=-1)

    def score_candidates(self, user_vec: torch.Tensor, candidate_item_ids: torch.Tensor) -> torch.Tensor:
        # user_vec: [B, D], candidate_item_ids: [B, C]
        cand_vec = self.encode_items(candidate_item_ids)  # [B, C, D]
        return torch.bmm(cand_vec, user_vec.unsqueeze(-1)).squeeze(-1)  # [B, C]

