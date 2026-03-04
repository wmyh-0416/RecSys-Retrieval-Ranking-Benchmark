import argparse
from typing import Dict, Iterable, Tuple

import torch

try:
    from .data import create_eval_dataloader, preprocess_movielens
    from .model import SASRec
    from .utils import format_metrics, load_checkpoint, resolve_device, set_seed
except ImportError:
    from data import create_eval_dataloader, preprocess_movielens
    from model import SASRec
    from utils import format_metrics, load_checkpoint, resolve_device, set_seed


@torch.no_grad()
def evaluate_model(
    model: SASRec,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    topk: Iterable[int] = (5, 10, 20),
) -> Dict[str, float]:
    model.eval()
    ks: Tuple[int, ...] = tuple(sorted(set(int(k) for k in topk)))
    hit_sums = {k: 0.0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    total = 0

    for seq, candidates in data_loader:
        seq = seq.to(device, non_blocking=True)
        candidates = candidates.to(device, non_blocking=True)

        hidden_states = model(seq)  # [B, L, D]
        lengths = (seq > 0).sum(dim=1) - 1
        lengths = torch.clamp(lengths, min=0)
        batch_idx = torch.arange(seq.size(0), device=device)
        last_hidden = hidden_states[batch_idx, lengths]  # [B, D]

        scores = model.score_candidates(last_hidden, candidates)  # [B, C]
        pos_scores = scores[:, :1]
        pos_rank = (scores > pos_scores).sum(dim=1) + 1  # 1-based rank

        for k in ks:
            hit = (pos_rank <= k).float()
            hit_sums[k] += float(hit.sum().item())
            ndcg_sums[k] += float((hit / torch.log2(pos_rank.float() + 1)).sum().item())
        total += seq.size(0)

    if total == 0:
        return {f"Recall@{k}": 0.0 for k in ks} | {f"NDCG@{k}": 0.0 for k in ks}

    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"Recall@{k}"] = hit_sums[k] / total
        metrics[f"NDCG@{k}"] = ndcg_sums[k] / total
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SASRec on MovieLens leave-one-out split.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_neg_eval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--min_user_inter", type=int, default=5)
    parser.add_argument("--min_item_inter", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model_cfg = checkpoint.get("model_config", {})
    train_args = checkpoint.get("train_args", {})

    dataset_name = args.dataset_name or str(train_args.get("dataset_name", ""))
    max_seq_len = int(model_cfg.get("max_seq_len", train_args.get("max_seq_len", args.max_seq_len)))
    min_user_inter = int(train_args.get("min_user_inter", args.min_user_inter))
    min_item_inter = int(train_args.get("min_item_inter", args.min_item_inter))

    data = preprocess_movielens(
        data_dir=args.data_dir,
        dataset_name=dataset_name,
        min_user_inter=min_user_inter,
        min_item_inter=min_item_inter,
    )

    model = SASRec(
        num_items=data.num_items,
        max_seq_len=max_seq_len,
        d_model=int(model_cfg.get("d_model", 64)),
        n_heads=int(model_cfg.get("n_heads", 2)),
        n_layers=int(model_cfg.get("n_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    valid_loader = create_eval_dataloader(
        data=data,
        split="valid",
        max_seq_len=max_seq_len,
        batch_size=args.batch_size,
        num_neg_eval=args.num_neg_eval,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    test_loader = create_eval_dataloader(
        data=data,
        split="test",
        max_seq_len=max_seq_len,
        batch_size=args.batch_size,
        num_neg_eval=args.num_neg_eval,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    valid_metrics = evaluate_model(model, valid_loader, device=device, topk=(5, 10, 20))
    test_metrics = evaluate_model(model, test_loader, device=device, topk=(5, 10, 20))

    print(f"Ratings file: {data.ratings_path}")
    print(f"Detected columns: {data.detected_columns}")
    print("Validation:", format_metrics(valid_metrics))
    print("Test:", format_metrics(test_metrics))


if __name__ == "__main__":
    main()
