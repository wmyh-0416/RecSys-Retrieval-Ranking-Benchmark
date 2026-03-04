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
from retrieval.dual_tower.model import DualTowerRetrieval


def _resolve_config_path(config_arg: str) -> str:
    if not config_arg:
        return ""
    direct = Path(config_arg)
    if direct.exists():
        return str(direct)
    sibling = Path(__file__).resolve().parent / config_arg
    if sibling.exists():
        return str(sibling)
    return config_arg


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


@torch.no_grad()
def evaluate_model(
    model: DualTowerRetrieval,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_items: int,
    topk: Iterable[int] = (5, 10, 20),
) -> Dict[str, float]:
    model.eval()
    ks: Tuple[int, ...] = tuple(sorted(set(int(k) for k in topk)))
    hit_sums = {k: 0.0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    total = 0

    all_item_ids = torch.arange(0, num_items + 1, device=device, dtype=torch.long)
    all_item_emb = model.encode_items(all_item_ids)

    for seq, candidates in data_loader:
        seq = seq.to(device, non_blocking=True)
        candidates = candidates.to(device, non_blocking=True)
        user_vec = model.encode_user(seq)

        candidate_vec = all_item_emb[candidates]
        scores = torch.bmm(candidate_vec, user_vec.unsqueeze(-1)).squeeze(-1)

        batch_hit, batch_ndcg = metrics_from_scores(scores=scores, pos_index=0, topk=ks)
        for k in ks:
            hit_sums[k] += batch_hit[k]
            ndcg_sums[k] += batch_ndcg[k]
        total += seq.size(0)

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
    parser = argparse.ArgumentParser(description="Evaluate Dual-Tower retrieval model.")
    parser.add_argument("--config", type=str, default="config.yaml")
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
    parser.add_argument("--max_users", type=int, default=None)
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
        "save_dir": "outputs/dual_tower",
        "min_user_inter": 5,
        "min_item_inter": 0,
        "max_users": 0,
        "num_workers": 0,
        "checkpoint": "",
    }
    cfg = dict(defaults)
    config_path = _resolve_config_path(args.config)
    if config_path:
        cfg = merge_config(cfg, load_config(config_path))
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
    max_users = int(train_args.get("max_users", cfg["max_users"] or 0))

    data = preprocess_movielens(
        data_dir=str(cfg["data_dir"]),
        dataset_name=dataset_name,
        min_user_inter=min_user_inter,
        min_item_inter=min_item_inter,
        max_users=max_users,
    )

    model = DualTowerRetrieval(
        num_items=data.num_items,
        embedding_dim=int(model_cfg.get("embedding_dim", 128)),
        tower_hidden_dim=int(model_cfg.get("tower_hidden_dim", 0)),
        dropout=float(model_cfg.get("dropout", 0.1)),
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
    )
    test_loader = create_eval_dataloader(
        data=data,
        split="test",
        max_seq_len=max_seq_len,
        batch_size=int(cfg["batch_size"]),
        num_neg_eval=int(cfg["num_neg_eval"]),
        seed=int(cfg["seed"]),
        num_workers=int(cfg["num_workers"]),
    )

    valid_metrics = evaluate_model(
        model=model, data_loader=valid_loader, device=device, num_items=data.num_items, topk=(5, 10, 20)
    )
    test_metrics = evaluate_model(
        model=model, data_loader=test_loader, device=device, num_items=data.num_items, topk=(5, 10, 20)
    )

    print(f"Ratings file: {data.ratings_path}")
    print(f"Detected columns: {data.detected_columns}")
    print("Validation:", format_metrics(valid_metrics))
    print("Test:", format_metrics(test_metrics))

    output_json = str(cfg["output_json"]) or str(Path(cfg["save_dir"]) / "metrics_summary.json")
    payload = {
        "model": "dual_tower_retrieval",
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
