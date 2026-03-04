from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data.preprocessing import preprocess_movielens
from common.data.sampler import create_eval_dataloader
from common.metrics.ranking_metrics import format_metrics, metrics_from_scores
from common.utils.config import load_config, merge_config
from common.utils.io import load_checkpoint, save_json
from common.utils.seed import set_seed
from models.bpr_mf.model import BPRMF


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


@torch.no_grad()
def evaluate_bpr_mf(
    model: BPRMF,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    topk: Iterable[int] = (5, 10, 20),
) -> Dict[str, float]:
    model.eval()
    ks: Tuple[int, ...] = tuple(sorted(set(int(k) for k in topk)))
    hit_sums = {k: 0.0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    total = 0

    for user_ids, _, candidates in data_loader:
        user_ids = user_ids.to(device, non_blocking=True)
        candidates = candidates.to(device, non_blocking=True)

        scores = model.score_candidates(user_ids, candidates)
        batch_hit, batch_ndcg = metrics_from_scores(scores=scores, pos_index=0, topk=ks)
        for k in ks:
            hit_sums[k] += batch_hit[k]
            ndcg_sums[k] += batch_ndcg[k]
        total += user_ids.size(0)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BPR-MF with unified benchmark protocol.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_neg_eval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--min_user_inter", type=int, default=None)
    parser.add_argument("--min_item_inter", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
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
        "device": "auto",
        "save_dir": "outputs/bpr_mf",
        "min_user_inter": 5,
        "min_item_inter": 0,
        "num_workers": 0,
        "checkpoint": "",
    }
    cfg = dict(defaults)
    if args.config:
        cfg = merge_config(cfg, load_config(args.config))
    overrides = vars(args).copy()
    overrides.pop("config", None)
    cfg = merge_config(cfg, overrides)
    if not cfg["checkpoint"]:
        cfg["checkpoint"] = str(Path(cfg["save_dir"]) / "best.pt")
    return cfg


def main():
    args = parse_args()
    cfg = resolve_eval_config(args)

    set_seed(int(cfg["seed"]))
    device = resolve_device(str(cfg["device"]))

    checkpoint = load_checkpoint(str(cfg["checkpoint"]), map_location="cpu")
    model_cfg = checkpoint.get("model_config", {})
    train_args = checkpoint.get("train_args", {})

    dataset_name = str(cfg["dataset_name"]) or str(train_args.get("dataset_name", ""))
    max_seq_len = int(cfg["max_seq_len"] or model_cfg.get("max_seq_len", 50))
    min_user_inter = int(train_args.get("min_user_inter", cfg["min_user_inter"]))
    min_item_inter = int(train_args.get("min_item_inter", cfg["min_item_inter"]))

    data = preprocess_movielens(
        data_dir=str(cfg["data_dir"]),
        dataset_name=dataset_name,
        min_user_inter=min_user_inter,
        min_item_inter=min_item_inter,
    )

    model = BPRMF(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=int(model_cfg.get("embedding_dim", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    valid_loader = create_eval_dataloader(
        data=data,
        split="valid",
        max_seq_len=max_seq_len,
        batch_size=int(cfg["batch_size"]),
        num_neg_eval=int(cfg["num_neg_eval"]),
        seed=int(cfg["seed"]),
        num_workers=int(cfg["num_workers"]),
        return_user_id=True,
    )
    test_loader = create_eval_dataloader(
        data=data,
        split="test",
        max_seq_len=max_seq_len,
        batch_size=int(cfg["batch_size"]),
        num_neg_eval=int(cfg["num_neg_eval"]),
        seed=int(cfg["seed"]),
        num_workers=int(cfg["num_workers"]),
        return_user_id=True,
    )

    valid_metrics = evaluate_bpr_mf(model, valid_loader, device=device, topk=(5, 10, 20))
    test_metrics = evaluate_bpr_mf(model, test_loader, device=device, topk=(5, 10, 20))

    print(f"Ratings file: {data.ratings_path}")
    print(f"Detected columns: {data.detected_columns}")
    print("Validation:", format_metrics(valid_metrics))
    print("Test:", format_metrics(test_metrics))

    output_json = str(cfg["output_json"]) or str(Path(cfg["save_dir"]) / "metrics_summary.json")
    payload = {
        "model": "bpr_mf",
        "dataset_name": dataset_name or Path(data.ratings_path).parent.name,
        "ratings_path": data.ratings_path,
        "detected_columns": data.detected_columns,
        "checkpoint": str(Path(cfg["checkpoint"]).resolve()),
        "valid": valid_metrics,
        "test": test_metrics,
        "best_epoch": checkpoint.get("epoch"),
        "best_valid_ndcg10": checkpoint.get("best_valid_ndcg10"),
    }
    save_json(output_json, payload)
    print(f"Saved metrics json: {output_json}")


if __name__ == "__main__":
    main()
