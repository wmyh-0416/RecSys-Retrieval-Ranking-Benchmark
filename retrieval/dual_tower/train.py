from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data.preprocessing import PreprocessedData, preprocess_movielens
from common.data.sampler import create_eval_dataloader
from common.metrics.ranking_metrics import format_metrics, metrics_from_scores
from common.utils.config import load_config, merge_config
from common.utils.io import ensure_dir, load_checkpoint, save_checkpoint, save_json
from common.utils.logger import create_logger
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


def to_plain_dict(payload: Dict) -> Dict:
    out = {}
    for k, v in payload.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


class RetrievalTrainDataset(Dataset):
    # Randomly sample (context, next_item) from train sequences.
    def __init__(
        self,
        data: PreprocessedData,
        max_seq_len: int,
        num_samples_per_epoch: int,
        seed: int = 42,
    ) -> None:
        self.train_sequences = data.train_sequences
        self.max_seq_len = int(max_seq_len)
        self.rng = np.random.default_rng(seed)

        users = []
        pair_counts = []
        for uid, seq in enumerate(self.train_sequences):
            num_pairs = int(max(0, seq.size - 1))
            if num_pairs > 0:
                users.append(uid)
                pair_counts.append(num_pairs)
        if not users:
            raise ValueError("No users have enough train interactions for retrieval training.")

        self.users = np.asarray(users, dtype=np.int64)
        pair_counts_arr = np.asarray(pair_counts, dtype=np.int64)
        self.cum_pair_counts = np.cumsum(pair_counts_arr)
        self.total_pairs = int(self.cum_pair_counts[-1])
        self.num_samples_per_epoch = int(num_samples_per_epoch) if num_samples_per_epoch > 0 else self.total_pairs

    def __len__(self) -> int:
        return self.num_samples_per_epoch

    def _sample_user(self) -> int:
        sampled = int(self.rng.integers(0, self.total_pairs))
        idx = int(np.searchsorted(self.cum_pair_counts, sampled, side="right"))
        return int(self.users[idx])

    def __getitem__(self, index: int):
        del index
        uid = self._sample_user()
        seq = self.train_sequences[uid]

        # target_idx in [1, len(seq)-1], context is seq[:target_idx], label is seq[target_idx]
        target_idx = int(self.rng.integers(1, seq.size))
        context = seq[:target_idx]
        pos_item = int(seq[target_idx])

        context = context[-self.max_seq_len :]
        seq_arr = np.zeros(self.max_seq_len, dtype=np.int64)
        seq_arr[-context.size :] = context
        return torch.from_numpy(seq_arr), torch.tensor(pos_item, dtype=torch.long)


@torch.no_grad()
def evaluate_model(
    model: DualTowerRetrieval,
    data_loader: DataLoader,
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
    all_item_emb = model.encode_items(all_item_ids)  # [num_items+1, D]

    for seq, candidates in data_loader:
        seq = seq.to(device, non_blocking=True)
        candidates = candidates.to(device, non_blocking=True)

        user_vec = model.encode_user(seq)  # [B, D]
        candidate_vec = all_item_emb[candidates]  # [B, C, D]
        scores = torch.bmm(candidate_vec, user_vec.unsqueeze(-1)).squeeze(-1)  # [B, C]

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
    parser = argparse.ArgumentParser(description="Train Dual-Tower retrieval model.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--tower_hidden_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_neg_eval", type=int, default=None)
    parser.add_argument("--num_samples_per_epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
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
        "batch_size": 1024,
        "epochs": 5,
        "lr": 1e-3,
        "embedding_dim": 128,
        "tower_hidden_dim": 0,
        "dropout": 0.1,
        "temperature": 0.07,
        "weight_decay": 1e-4,
        "num_neg_eval": 100,
        "num_samples_per_epoch": 1000000,
        "seed": 42,
        "device": "auto",
        "save_dir": "outputs/dual_tower",
        "min_user_inter": 5,
        "min_item_inter": 0,
        "num_workers": 0,
        "patience": 10,
        "grad_clip": 1.0,
        "max_users": 0,
    }
    cfg = dict(defaults)
    config_path = _resolve_config_path(args.config)
    if config_path:
        cfg = merge_config(cfg, load_config(config_path))
    overrides = vars(args).copy()
    overrides.pop("config", None)
    cfg = merge_config(cfg, overrides)
    cfg["max_users"] = int(cfg.get("max_users", 0) or 0)
    return cfg


def train_one_epoch(
    model: DualTowerRetrieval,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    for seq, pos_item in data_loader:
        seq = seq.to(device, non_blocking=True)
        pos_item = pos_item.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        user_vec = model.encode_user(seq)  # [B, D]
        pos_vec = model.encode_items(pos_item)  # [B, D]

        logits = torch.matmul(user_vec, pos_vec.t()) / float(temperature)
        labels = torch.arange(logits.size(0), device=device, dtype=torch.long)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), float(grad_clip))
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

    train_dataset = RetrievalTrainDataset(
        data=data,
        max_seq_len=int(cfg["max_seq_len"]),
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

    model = DualTowerRetrieval(
        num_items=data.num_items,
        embedding_dim=int(cfg["embedding_dim"]),
        tower_hidden_dim=int(cfg["tower_hidden_dim"]),
        dropout=float(cfg["dropout"]),
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
            temperature=float(cfg["temperature"]),
            grad_clip=float(cfg["grad_clip"]),
        )
        valid_metrics = evaluate_model(
            model=model,
            data_loader=valid_loader,
            device=device,
            num_items=data.num_items,
            topk=(5, 10, 20),
        )
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
                        "num_items": data.num_items,
                        "max_seq_len": int(cfg["max_seq_len"]),
                        "embedding_dim": int(cfg["embedding_dim"]),
                        "tower_hidden_dim": int(cfg["tower_hidden_dim"]),
                        "dropout": float(cfg["dropout"]),
                    },
                    "train_args": to_plain_dict(cfg),
                    "data_info": {
                        "ratings_path": data.ratings_path,
                        "detected_columns": data.detected_columns,
                        "num_users": data.num_users,
                        "num_items": data.num_items,
                        "num_interactions": data.num_interactions,
                    },
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

    valid_best = evaluate_model(
        model=model,
        data_loader=valid_loader,
        device=device,
        num_items=data.num_items,
        topk=(5, 10, 20),
    )
    test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        num_items=data.num_items,
        topk=(5, 10, 20),
    )
    logger.info("Best valid epoch: %d | best valid NDCG@10=%.4f", best_epoch, best_metric)
    logger.info("Test metrics: %s", format_metrics(test_metrics))

    save_json(
        str(Path(cfg["save_dir"]) / "metrics_summary.json"),
        {
            "model": "dual_tower_retrieval",
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

