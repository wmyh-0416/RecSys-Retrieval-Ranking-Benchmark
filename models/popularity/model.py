from __future__ import annotations

import numpy as np
import torch


class PopularityModel:
    def __init__(self, num_items: int) -> None:
        self.num_items = num_items
        self.counts = np.zeros(num_items + 1, dtype=np.int64)

    def fit(self, train_sequences: list[np.ndarray]) -> None:
        # Strictly count only training interactions.
        counts = np.zeros(self.num_items + 1, dtype=np.int64)
        for seq in train_sequences:
            if seq.size > 0:
                np.add.at(counts, seq, 1)
        counts[0] = 0
        self.counts = counts

    def save(self, path: str) -> None:
        np.save(path, self.counts)

    def load(self, path: str) -> None:
        self.counts = np.load(path)

    def score_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        # candidates: [B, C], ids in [1..num_items].
        counts_tensor = torch.as_tensor(self.counts, device=candidates.device, dtype=torch.float32)
        return counts_tensor[candidates]
