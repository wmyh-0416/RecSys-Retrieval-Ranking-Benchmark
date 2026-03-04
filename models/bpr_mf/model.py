from __future__ import annotations

import torch
import torch.nn as nn


class BPRMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0.0)

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)
        return (u * i).sum(dim=-1)

    def score_candidates(self, user_ids: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        # user_ids: [B], candidates: [B, C]
        user_emb = self.user_embedding(user_ids)  # [B, D]
        item_emb = self.item_embedding(candidates)  # [B, C, D]
        return torch.bmm(item_emb, user_emb.unsqueeze(-1)).squeeze(-1)  # [B, C]
