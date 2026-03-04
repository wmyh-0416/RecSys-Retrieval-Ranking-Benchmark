from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.amp as amp
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data.preprocessing import PreprocessedData, preprocess_movielens
from common.data.sampler import create_eval_dataloader
from common.metrics.ranking_metrics import format_metrics
from common.utils.config import load_config, merge_config
from common.utils.io import ensure_dir, load_checkpoint, save_checkpoint, save_json
from common.utils.logger import create_logger
from common.utils.seed import set_seed
from models.bert4rec.eval import evaluate_model, resolve_device
from models.bert4rec.model import BERT4Rec


def to_plain_dict(payload: Dict) -> Dict:
    out = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


class MaskedTrainDataset(Dataset):
    # Build one masked sequence sample per user on each epoch traversal.
    def __init__(
        self,
        data: PreprocessedData,
        max_seq_len: int,
        mask_rate: float,
        seed: int = 42,
    ) -> None:
        self.train_sequences = data.train_sequences
        self.max_seq_len = int(max_seq_len)
        self.mask_rate = float(mask_rate)
        self.mask_id = int(data.mask_id)
        self.rng = np.random.default_rng(seed)

        users = []
        for uid, seq in enumerate(self.train_sequences):
            if seq.size > 0:
                users.append(uid)
        if not users:
            raise ValueError("No valid users available for BERT4Rec training.")
        self.users = np.asarray(users, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.users.size)

    def __getitem__(self, index: int):
        uid = int(self.users[index])
        seq = self.train_sequences[uid]

        tail = seq[-self.max_seq_len :]
        seq_arr = np.zeros(self.max_seq_len, dtype=np.int64)
        seq_arr[-tail.size :] = tail

        labels = np.zeros(self.max_seq_len, dtype=np.int64)
        non_pad_positions = np.flatnonzero(seq_arr > 0)
        if non_pad_positions.size == 0:
            return torch.from_numpy(seq_arr), torch.from_numpy(labels)

        mask_flags = self.rng.random(non_pad_positions.size) < self.mask_rate
        if not mask_flags.any():
            mask_flags[self.rng.integers(0, non_pad_positions.size)] = True
        masked_positions = non_pad_positions[mask_flags]

        labels[masked_positions] = seq_arr[masked_positions]
        seq_arr[masked_positions] = self.mask_id

        return (
            torch.from_numpy(seq_arr),
            torch.from_numpy(labels),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT4Rec with unified benchmark protocol.")
    parser.add_argument("--config", type=str, default="")

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_neg_eval", type=int, default=None)
    parser.add_argument("--num_neg_train", type=int, default=None)
    parser.add_argument("--mask_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--min_user_inter", type=int, default=None)
    parser.add_argument("--min_item_inter", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max_users", type=int, default=None)
    return parser.parse_args()


def resolve_train_config(args) -> Dict:
    defaults = {
        "data_dir": ".",
        "dataset_name": "",
        "max_seq_len": 50,
        "batch_size": 512,
        "epochs": 2,
        "lr": 1e-3,
        "d_model": 64,
        "n_heads": 2,
        "n_layers": 2,
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "num_neg_eval": 100,
        "num_neg_train": 100,
        "mask_rate": 0.2,
        "seed": 42,
        "device": "auto",
        "save_dir": "outputs/bert4rec",
        "min_user_inter": 5,
        "min_item_inter": 0,
        "num_workers": 0,
        "patience": 10,
        "grad_clip": 1.0,
        "amp": False,
        "max_users": 0,
    }
    cfg = dict(defaults)
    if args.config:
        cfg = merge_config(cfg, load_config(args.config))
    overrides = vars(args).copy()
    overrides.pop("config", None)
    cfg = merge_config(cfg, overrides)
    cfg["amp"] = bool(args.amp or cfg.get("amp", False))
    cfg["max_users"] = int(cfg.get("max_users", 0) or 0)
    return cfg


def sample_negatives_excluding_positive(
    positives: torch.Tensor,
    num_items: int,
    num_neg: int,
) -> torch.Tensor:
    # positives: [M], values in [1..num_items]
    neg = torch.randint(
        low=1,
        high=num_items + 1,
        size=(positives.size(0), num_neg),
        device=positives.device,
    )
    same = neg.eq(positives.unsqueeze(1))
    while same.any():
        neg[same] = torch.randint(
            low=1,
            high=num_items + 1,
            size=(int(same.sum().item()),),
            device=positives.device,
        )
        same = neg.eq(positives.unsqueeze(1))
    return neg


def train_one_epoch(
    model: BERT4Rec,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    scaler: amp.GradScaler,
    use_amp: bool,
    device: torch.device,
    num_neg_train: int,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    for seq, labels in data_loader:
        seq = seq.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=use_amp):
            hidden = model(seq)  # [B, L, D]
            mask_pos = labels > 0
            if int(mask_pos.sum().item()) == 0:
                continue

            hidden_masked = hidden[mask_pos]  # [M, D]
            pos_items = labels[mask_pos]  # [M]
            neg_items = sample_negatives_excluding_positive(
                positives=pos_items,
                num_items=model.num_items,
                num_neg=int(num_neg_train),
            )  # [M, N]

            pos_emb = model.item_embedding(pos_items)  # [M, D]
            neg_emb = model.item_embedding(neg_items)  # [M, N, D]

            pos_logits = (hidden_masked * pos_emb).sum(dim=-1, keepdim=True)  # [M, 1]
            neg_logits = torch.einsum("md,mnd->mn", hidden_masked, neg_emb)  # [M, N]
            logits = torch.cat([pos_logits, neg_logits], dim=1)  # [M, 1+N]
            targets = torch.zeros(logits.size(0), dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
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
    use_amp = bool(cfg["amp"]) and device.type == "cuda"

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
        "Dataset stats | users=%d | items=%d | interactions=%d | mask_id=%d",
        data.num_users,
        data.num_items,
        data.num_interactions,
        data.mask_id,
    )

    train_dataset = MaskedTrainDataset(
        data=data,
        max_seq_len=int(cfg["max_seq_len"]),
        mask_rate=float(cfg["mask_rate"]),
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

    model = BERT4Rec(
        num_items=data.num_items,
        max_seq_len=int(cfg["max_seq_len"]),
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        n_layers=int(cfg["n_layers"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scaler = amp.GradScaler("cuda", enabled=use_amp)

    logger.info("Device: %s | AMP: %s", device, use_amp)
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
            grad_clip=float(cfg["grad_clip"]),
            scaler=scaler,
            use_amp=use_amp,
            device=device,
            num_neg_train=int(cfg["num_neg_train"]),
        )
        valid_metrics = evaluate_model(
            model=model,
            data_loader=valid_loader,
            device=device,
            mask_id=data.mask_id,
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
                        "d_model": int(cfg["d_model"]),
                        "n_heads": int(cfg["n_heads"]),
                        "n_layers": int(cfg["n_layers"]),
                        "dropout": float(cfg["dropout"]),
                        "mask_id": int(data.mask_id),
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

    valid_best = evaluate_model(
        model=model,
        data_loader=valid_loader,
        device=device,
        mask_id=int(ckpt["model_config"].get("mask_id", data.mask_id)),
        topk=(5, 10, 20),
    )
    test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        mask_id=int(ckpt["model_config"].get("mask_id", data.mask_id)),
        topk=(5, 10, 20),
    )

    logger.info("Best valid epoch: %d | best valid NDCG@10=%.4f", best_epoch, best_metric)
    logger.info("Test metrics: %s", format_metrics(test_metrics))

    save_json(
        str(Path(cfg["save_dir"]) / "metrics_summary.json"),
        {
            "model": "bert4rec",
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

