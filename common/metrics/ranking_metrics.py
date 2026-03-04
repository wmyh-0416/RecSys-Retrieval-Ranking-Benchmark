from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch


def format_metrics(metrics: Dict[str, float]) -> str:
    order = ["Recall@5", "Recall@10", "Recall@20", "NDCG@5", "NDCG@10", "NDCG@20"]
    parts = []
    for key in order:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    for key, value in metrics.items():
        if key not in order:
            parts.append(f"{key}={value:.4f}")
    return " | ".join(parts)


def metrics_from_scores(
    scores: torch.Tensor,
    pos_index: int = 0,
    topk: Iterable[int] = (5, 10, 20),
) -> Tuple[Dict[int, float], Dict[int, float]]:
    # scores: [B, C], one positive item at pos_index and C-1 negatives.
    ks = tuple(sorted(set(int(k) for k in topk)))
    pos_scores = scores[:, pos_index : pos_index + 1]
    pos_rank = (scores > pos_scores).sum(dim=1) + 1  # 1-based rank

    hit_sums: Dict[int, float] = {k: 0.0 for k in ks}
    ndcg_sums: Dict[int, float] = {k: 0.0 for k in ks}

    for k in ks:
        hit = (pos_rank <= k).float()
        hit_sums[k] = float(hit.sum().item())
        ndcg_sums[k] = float((hit / torch.log2(pos_rank.float() + 1)).sum().item())
    return hit_sums, ndcg_sums
