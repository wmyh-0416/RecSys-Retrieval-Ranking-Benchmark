from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data.preprocessing import preprocess_movielens
from common.data.sampler import create_eval_dataloader
from common.metrics.ranking_metrics import format_metrics, metrics_from_scores
from common.utils.config import load_config, merge_config
from common.utils.io import save_json
from common.utils.seed import set_seed
from models.popularity.model import PopularityModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate popularity baseline with unified benchmark protocol.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_neg_eval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min_user_inter", type=int, default=None)
    parser.add_argument("--min_item_inter", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--counts_path", type=str, default="")
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def resolve_eval_config(args) -> Dict:
    defaults = {
        "data_dir": ".",
        "dataset_name": "",
        "max_seq_len": 50,
        "batch_size": 1024,
        "num_neg_eval": 100,
        "seed": 42,
        "min_user_inter": 5,
        "min_item_inter": 0,
        "num_workers": 0,
        "save_dir": "outputs/popularity",
    }
    cfg = dict(defaults)
    if args.config:
        cfg = merge_config(cfg, load_config(args.config))
    overrides = vars(args).copy()
    overrides.pop("config", None)
    cfg = merge_config(cfg, overrides)
    return cfg


@torch.no_grad()
def evaluate_popularity(
    model: PopularityModel,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    topk: Iterable[int] = (5, 10, 20),
) -> Dict[str, float]:
    ks: Tuple[int, ...] = tuple(sorted(set(int(k) for k in topk)))
    hit_sums = {k: 0.0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    total = 0

    for _, candidates in data_loader:
        candidates = candidates.to(device, non_blocking=True)
        scores = model.score_candidates(candidates)
        batch_hit, batch_ndcg = metrics_from_scores(scores=scores, pos_index=0, topk=ks)
        for k in ks:
            hit_sums[k] += batch_hit[k]
            ndcg_sums[k] += batch_ndcg[k]
        total += candidates.size(0)

    metrics: Dict[str, float] = {}
    if total == 0:
        for k in ks:
            metrics[f"Recall@{k}"] = 0.0
            metrics[f"NDCG@{k}"] = 0.0
        return metrics

    for k in ks:
        metrics[f"Recall@{k}"] = hit_sums[k] / total
        metrics[f"NDCG@{k}"] = ndcg_sums[k] / total
    return metrics


def main():
    args = parse_args()
    cfg = resolve_eval_config(args)
    set_seed(int(cfg["seed"]))

    device = torch.device("cpu")

    data = preprocess_movielens(
        data_dir=str(cfg["data_dir"]),
        dataset_name=str(cfg["dataset_name"]),
        min_user_inter=int(cfg["min_user_inter"]),
        min_item_inter=int(cfg["min_item_inter"]),
    )

    counts_path = cfg["counts_path"] or str(Path(cfg["save_dir"]) / "popularity_counts.npy")
    counts_path = str(Path(counts_path).resolve())
    if not Path(counts_path).exists():
        raise FileNotFoundError(f"Popularity counts not found: {counts_path}")

    model = PopularityModel(num_items=data.num_items)
    model.load(counts_path)

    valid_loader = create_eval_dataloader(
        data=data,
        split="valid",
        max_seq_len=int(cfg["max_seq_len"]),
        batch_size=int(cfg["batch_size"]),
        num_neg_eval=int(cfg["num_neg_eval"]),
        seed=int(cfg["seed"]),
        num_workers=int(cfg["num_workers"]),
    )
    test_loader = create_eval_dataloader(
        data=data,
        split="test",
        max_seq_len=int(cfg["max_seq_len"]),
        batch_size=int(cfg["batch_size"]),
        num_neg_eval=int(cfg["num_neg_eval"]),
        seed=int(cfg["seed"]),
        num_workers=int(cfg["num_workers"]),
    )

    valid_metrics = evaluate_popularity(model, valid_loader, device=device, topk=(5, 10, 20))
    test_metrics = evaluate_popularity(model, test_loader, device=device, topk=(5, 10, 20))

    print(f"Ratings file: {data.ratings_path}")
    print(f"Detected columns: {data.detected_columns}")
    print("Validation:", format_metrics(valid_metrics))
    print("Test:", format_metrics(test_metrics))

    output_json = cfg["output_json"] or str(Path(cfg["save_dir"]) / "metrics_summary.json")
    save_json(
        output_json,
        {
            "model": "popularity",
            "dataset_name": str(cfg["dataset_name"]) or Path(data.ratings_path).parent.name,
            "checkpoint": counts_path,
            "best_epoch": 0,
            "best_valid_ndcg10": valid_metrics.get("NDCG@10", 0.0),
            "ratings_path": data.ratings_path,
            "detected_columns": data.detected_columns,
            "valid": valid_metrics,
            "test": test_metrics,
        },
    )
    print(f"Saved metrics json: {output_json}")


if __name__ == "__main__":
    main()
