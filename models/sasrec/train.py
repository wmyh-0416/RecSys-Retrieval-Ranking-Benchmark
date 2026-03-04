import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.amp as amp
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from common.data.preprocessing import preprocess_movielens
from common.data.sampler import create_eval_dataloader, create_train_dataloader
from common.metrics.ranking_metrics import format_metrics
from common.utils.config import load_config, merge_config
from common.utils.io import ensure_dir, load_checkpoint, save_checkpoint, save_json
from common.utils.logger import create_logger
from common.utils.seed import set_seed
from models.sasrec.eval import evaluate_model, resolve_device
from models.sasrec.model import SASRec


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_plain_args_dict(payload: Dict) -> Dict:
    out = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Train SASRec with unified benchmark protocol.")
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
        "batch_size": 256,
        "epochs": 50,
        "lr": 1e-3,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "num_neg_eval": 100,
        "seed": 42,
        "device": "auto",
        "save_dir": "outputs/sasrec",
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
        cfg_from_file = load_config(args.config)
        cfg = merge_config(cfg, cfg_from_file)

    cli_overrides = vars(args).copy()
    cli_overrides.pop("config", None)
    cfg = merge_config(cfg, cli_overrides)
    cfg["amp"] = bool(args.amp or cfg.get("amp", False))
    cfg["max_users"] = int(cfg.get("max_users", 0) or 0)
    return cfg


def train_one_epoch(
    model: SASRec,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    scaler: amp.GradScaler,
    use_amp: bool,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    for seq, pos, neg in data_loader:
        seq = seq.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        neg = neg.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type="cuda", enabled=use_amp):
            hidden = model(seq)
            pos_emb = model.item_embedding(pos)
            neg_emb = model.item_embedding(neg)
            logits_pos = (hidden * pos_emb).sum(dim=-1)
            logits_neg = (hidden * neg_emb).sum(dim=-1)

            valid_mask = (pos > 0) & (neg > 0)
            if valid_mask.sum().item() == 0:
                continue

            bpr_loss = -F.logsigmoid(logits_pos - logits_neg)
            mask_f = valid_mask.float()
            loss = (bpr_loss * mask_f).sum() / mask_f.sum()

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

    ensure_dir(cfg["save_dir"])
    logger = create_logger(cfg["save_dir"], name="train")
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
        "Dataset stats | users=%d | items=%d | interactions=%d",
        data.num_users,
        data.num_items,
        data.num_interactions,
    )

    train_loader = create_train_dataloader(
        data=data,
        max_seq_len=int(cfg["max_seq_len"]),
        batch_size=int(cfg["batch_size"]),
        seed=int(cfg["seed"]),
        num_workers=int(cfg["num_workers"]),
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

    model = SASRec(
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
    logger.info("Trainable params: %d", count_parameters(model))
    logger.info("Hyperparameters: %s", to_plain_args_dict(cfg))
    save_json(str(Path(cfg["save_dir"]) / "train_args.json"), to_plain_args_dict(cfg))
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
        )
        valid_metrics = evaluate_model(model, valid_loader, device=device, topk=(5, 10, 20))
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
                    },
                    "data_info": {
                        "ratings_path": data.ratings_path,
                        "detected_columns": data.detected_columns,
                        "num_users": data.num_users,
                        "num_items": data.num_items,
                        "num_interactions": data.num_interactions,
                    },
                    "train_args": to_plain_args_dict(cfg),
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
    test_metrics = evaluate_model(model, test_loader, device=device, topk=(5, 10, 20))

    logger.info("Best valid epoch: %d | best valid NDCG@10=%.4f", best_epoch, best_metric)
    logger.info("Test metrics: %s", format_metrics(test_metrics))

    metrics_summary = {
        "model": "sasrec",
        "dataset_name": str(cfg["dataset_name"]) or Path(data.ratings_path).parent.name,
        "ratings_path": data.ratings_path,
        "detected_columns": data.detected_columns,
        "checkpoint": str(Path(best_ckpt_path).resolve()),
        "best_epoch": best_epoch,
        "best_valid_ndcg10": best_metric,
        "valid": ckpt.get("best_valid_metrics", {}),
        "test": test_metrics,
    }
    # Keep explicit current-best valid metrics for aggregation.
    metrics_summary["valid"] = evaluate_model(model, valid_loader, device=device, topk=(5, 10, 20))
    save_json(str(Path(cfg["save_dir"]) / "metrics_summary.json"), metrics_summary)

    print("Final Test:", format_metrics(test_metrics))


if __name__ == "__main__":
    main()
