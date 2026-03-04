from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data.preprocessing import preprocess_movielens
from common.data.sampler import create_eval_dataloader
from common.metrics.ranking_metrics import format_metrics
from common.utils.config import load_config, merge_config
from common.utils.io import ensure_dir, load_checkpoint, save_checkpoint, save_json
from common.utils.logger import create_logger
from common.utils.seed import set_seed
from models.bpr_mf.eval import evaluate_bpr_mf, resolve_device
from models.bpr_mf.model import BPRMF


def _contains(sorted_array: np.ndarray, value: int) -> bool:
    idx = np.searchsorted(sorted_array, value)
    return idx < sorted_array.size and int(sorted_array[idx]) == value


def _sample_negative(num_items: int, seen_sorted: np.ndarray, rng: np.random.Generator) -> int:
    if seen_sorted.size >= num_items:
        return 0
    while True:
        item = int(rng.integers(1, num_items + 1))
        if not _contains(seen_sorted, item):
            return item


class BPRTrainDataset(Dataset):
    # Randomly sample (u, pos, neg) from train sequences for each step.
    def __init__(
        self,
        train_sequences: list[np.ndarray],
        seen_items_sorted: list[np.ndarray],
        num_items: int,
        num_samples_per_epoch: int,
        seed: int = 42,
    ) -> None:
        self.train_sequences = train_sequences
        self.seen_items_sorted = seen_items_sorted
        self.num_items = num_items
        self.rng = np.random.default_rng(seed)

        users = []
        for uid, seq in enumerate(train_sequences):
            if seq.size > 0 and seen_items_sorted[uid].size < num_items:
                users.append(uid)
        if not users:
            raise ValueError("No valid users available for BPR training.")
        self.users = np.asarray(users, dtype=np.int64)
        self.num_samples_per_epoch = int(num_samples_per_epoch) if num_samples_per_epoch > 0 else len(users)

    def __len__(self) -> int:
        return self.num_samples_per_epoch

    def __getitem__(self, index: int):
        del index
        uid = int(self.users[int(self.rng.integers(0, len(self.users)))])
        seq = self.train_sequences[uid]
        seen_sorted = self.seen_items_sorted[uid]

        pos = int(seq[int(self.rng.integers(0, seq.size))])
        neg = _sample_negative(self.num_items, seen_sorted, self.rng)
        return (
            torch.tensor(uid, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train BPR-MF with unified benchmark protocol.")
    parser.add_argument("--config", type=str, default="")

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_neg_eval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--num_samples_per_epoch", type=int, default=None)

    parser.add_argument("--min_user_inter", type=int, default=None)
    parser.add_argument("--min_item_inter", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--max_users", type=int, default=None)
    return parser.parse_args()


def resolve_train_config(args) -> Dict:
    defaults = {
        "data_dir": ".",
        "dataset_name": "",
        "max_seq_len": 50,
        "batch_size": 4096,
        "epochs": 5,
        "lr": 1e-3,
        "embedding_dim": 64,
        "weight_decay": 1e-6,
        "num_neg_eval": 100,
        "seed": 42,
        "device": "auto",
        "save_dir": "outputs/bpr_mf",
        "num_samples_per_epoch": 0,
        "min_user_inter": 5,
        "min_item_inter": 0,
        "num_workers": 0,
        "patience": 10,
        "grad_clip": 1.0,
        "max_users": 0,
    }
    cfg = dict(defaults)
    if args.config:
        cfg = merge_config(cfg, load_config(args.config))
    overrides = vars(args).copy()
    overrides.pop("config", None)
    cfg = merge_config(cfg, overrides)
    cfg["max_users"] = int(cfg.get("max_users", 0) or 0)
    cfg["num_samples_per_epoch"] = int(cfg.get("num_samples_per_epoch", 0) or 0)
    return cfg


def to_plain_dict(payload: Dict) -> Dict:
    out = {}
    for k, v in payload.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def train_one_epoch(
    model: BPRMF,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    for user_ids, pos_items, neg_items in data_loader:
        user_ids = user_ids.to(device, non_blocking=True)
        pos_items = pos_items.to(device, non_blocking=True)
        neg_items = neg_items.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pos_scores = model.score(user_ids, pos_items)
        neg_scores = model.score(user_ids, neg_items)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        steps += 1

    return total_loss / max(steps, 1)


def main():
    args = parse_args()
    cfg = resolve_train_config(args)

    ensure_dir(str(cfg["save_dir"]))
    logger = create_logger(str(cfg["save_dir"]), name="train")
    set_seed(int(cfg["seed"]))
    device = resolve_device(str(cfg["device"]))

    logger.info("Loading and preprocessing data...")
    data = preprocess_movielens(
        data_dir=str(cfg["data_dir"]),
        dataset_name=str(cfg["dataset_name"]),
        min_user_inter=int(cfg["min_user_inter"]),
        min_item_inter=int(cfg["min_item_inter"]),
        max_users=int(cfg["max_users"]),
    )
    logger.info(f"Detected ratings file: {data.ratings_path}")
    logger.info(f"Detected columns: {data.detected_columns}")
    logger.info(
        "Dataset stats | users=%d | items=%d | interactions=%d",
        data.num_users,
        data.num_items,
        data.num_interactions,
    )

    train_dataset = BPRTrainDataset(
        train_sequences=data.train_sequences,
        seen_items_sorted=data.seen_items_sorted,
        num_items=data.num_items,
        num_samples_per_epoch=int(cfg["num_samples_per_epoch"]),
        seed=int(cfg["seed"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = create_eval_dataloader(
        data=data,
        split="valid",
        max_seq_len=int(cfg["max_seq_len"]),
        batch_size=int(cfg["batch_size"]),
        num_neg_eval=int(cfg["num_neg_eval"]),
        seed=int(cfg["seed"]),
        num_workers=int(cfg["num_workers"]),
        return_user_id=True,
    )
    test_loader = create_eval_dataloader(
        data=data,
        split="test",
        max_seq_len=int(cfg["max_seq_len"]),
        batch_size=int(cfg["batch_size"]),
        num_neg_eval=int(cfg["num_neg_eval"]),
        seed=int(cfg["seed"]),
        num_workers=int(cfg["num_workers"]),
        return_user_id=True,
    )

    model = BPRMF(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=int(cfg["embedding_dim"]),
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    logger.info("Device: %s", device)
    logger.info("Hyperparameters: %s", to_plain_dict(cfg))
    save_json(str(Path(cfg["save_dir"]) / "train_args.json"), to_plain_dict(cfg))
    np.save(Path(cfg["save_dir"]) / "item_raw_ids.npy", data.item_raw_ids.astype(np.int64))
    np.save(Path(cfg["save_dir"]) / "user_raw_ids.npy", data.user_raw_ids.astype(np.int64))

    best_metric = float("-inf")
    best_epoch = -1
    no_improve = 0
    best_ckpt_path = str(Path(cfg["save_dir"]) / "best.pt")

    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=float(cfg["grad_clip"]),
        )
        valid_metrics = evaluate_bpr_mf(model, valid_loader, device=device, topk=(5, 10, 20))
        monitor = valid_metrics.get("NDCG@10", 0.0)

        logger.info(
            "Epoch %03d | train_loss=%.4f | valid: %s",
            epoch,
            train_loss,
            format_metrics(valid_metrics),
        )

        if monitor > best_metric:
            best_metric = monitor
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                best_ckpt_path,
                {
                    "epoch": epoch,
                    "best_valid_ndcg10": best_metric,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "model_config": {
                        "num_users": data.num_users,
                        "num_items": data.num_items,
                        "embedding_dim": int(cfg["embedding_dim"]),
                        "max_seq_len": int(cfg["max_seq_len"]),
                    },
                    "data_info": {
                        "ratings_path": data.ratings_path,
                        "detected_columns": data.detected_columns,
                        "num_users": data.num_users,
                        "num_items": data.num_items,
                        "num_interactions": data.num_interactions,
                    },
                    "train_args": to_plain_dict(cfg),
                },
            )
            logger.info("New best checkpoint saved at epoch %d (NDCG@10=%.4f)", epoch, best_metric)
        else:
            no_improve += 1
            if no_improve >= int(cfg["patience"]):
                logger.info(
                    "Early stopping triggered at epoch %d (best epoch=%d, best NDCG@10=%.4f).",
                    epoch,
                    best_epoch,
                    best_metric,
                )
                break

    ckpt = load_checkpoint(best_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    valid_best = evaluate_bpr_mf(model, valid_loader, device=device, topk=(5, 10, 20))
    test_metrics = evaluate_bpr_mf(model, test_loader, device=device, topk=(5, 10, 20))
    logger.info("Best valid epoch: %d | best valid NDCG@10=%.4f", best_epoch, best_metric)
    logger.info("Test metrics: %s", format_metrics(test_metrics))

    save_json(
        str(Path(cfg["save_dir"]) / "metrics_summary.json"),
        {
            "model": "bpr_mf",
            "dataset_name": str(cfg["dataset_name"]) or Path(data.ratings_path).parent.name,
            "ratings_path": data.ratings_path,
            "detected_columns": data.detected_columns,
            "checkpoint": str(Path(best_ckpt_path).resolve()),
            "best_epoch": best_epoch,
            "best_valid_ndcg10": best_metric,
            "valid": valid_best,
            "test": test_metrics,
        },
    )

    print("Final Test:", format_metrics(test_metrics))


if __name__ == "__main__":
    main()
