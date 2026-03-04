import argparse
from pathlib import Path

import numpy as np
import torch
import torch.amp as amp
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

try:
    from .data import create_eval_dataloader, create_train_dataloader, preprocess_movielens
    from .eval import evaluate_model
    from .model import SASRec
    from .utils import (
        count_parameters,
        create_logger,
        ensure_dir,
        format_metrics,
        load_checkpoint,
        resolve_device,
        save_checkpoint,
        save_json,
        set_seed,
        to_plain_args_dict,
    )
except ImportError:
    from data import create_eval_dataloader, create_train_dataloader, preprocess_movielens
    from eval import evaluate_model
    from model import SASRec
    from utils import (
        count_parameters,
        create_logger,
        ensure_dir,
        format_metrics,
        load_checkpoint,
        resolve_device,
        save_checkpoint,
        save_json,
        set_seed,
        to_plain_args_dict,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train SASRec for MovieLens next-item prediction.")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_neg_eval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_dir", type=str, default="outputs/sasrec")

    parser.add_argument("--min_user_inter", type=int, default=5)
    parser.add_argument("--min_item_inter", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max_users", type=int, default=0, help="Optional debug cap.")
    return parser.parse_args()


def train_one_epoch(
    model: SASRec,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    scaler: amp.GradScaler,
    use_amp: bool,
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
    ensure_dir(args.save_dir)
    logger = create_logger(args.save_dir, name="train")

    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    logger.info("Loading and preprocessing data...")
    data = preprocess_movielens(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        min_user_inter=args.min_user_inter,
        min_item_inter=args.min_item_inter,
        max_users=args.max_users,
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
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    valid_loader = create_eval_dataloader(
        data=data,
        split="valid",
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_neg_eval=args.num_neg_eval,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    test_loader = create_eval_dataloader(
        data=data,
        split="test",
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_neg_eval=args.num_neg_eval,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = SASRec(
        num_items=data.num_items,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler("cuda", enabled=use_amp)

    logger.info("Device: %s | AMP: %s", device, use_amp)
    logger.info("Trainable params: %d", count_parameters(model))
    logger.info("Hyperparameters: %s", to_plain_args_dict(args))
    save_json(str(Path(args.save_dir) / "train_args.json"), to_plain_args_dict(args))
    np.save(Path(args.save_dir) / "item_raw_ids.npy", data.item_raw_ids.astype(np.int64))
    np.save(Path(args.save_dir) / "user_raw_ids.npy", data.user_raw_ids.astype(np.int64))

    best_metric = float("-inf")
    best_epoch = -1
    no_improve = 0
    best_ckpt_path = str(Path(args.save_dir) / "best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            scaler=scaler,
            use_amp=use_amp,
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
                        "max_seq_len": args.max_seq_len,
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "n_layers": args.n_layers,
                        "dropout": args.dropout,
                    },
                    "data_info": {
                        "ratings_path": data.ratings_path,
                        "detected_columns": data.detected_columns,
                        "num_users": data.num_users,
                        "num_items": data.num_items,
                        "num_interactions": data.num_interactions,
                    },
                    "train_args": to_plain_args_dict(args),
                },
            )
            logger.info("New best checkpoint saved at epoch %d (NDCG@10=%.4f)", epoch, best_metric)
        else:
            no_improve += 1
            if no_improve >= args.patience:
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

    print("Final Test:", format_metrics(test_metrics))


if __name__ == "__main__":
    main()
