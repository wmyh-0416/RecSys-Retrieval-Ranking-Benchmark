import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

try:
    from .data import find_ratings_file, preprocess_movielens
    from .model import SASRec
    from .utils import load_checkpoint, resolve_device
except ImportError:
    from data import find_ratings_file, preprocess_movielens
    from model import SASRec
    from utils import load_checkpoint, resolve_device


def parse_item_sequence(text: str) -> List[int]:
    parts = [p for p in re.split(r"[,\s]+", text.strip()) if p]
    if not parts:
        return []
    return [int(x) for x in parts]


def find_movies_file(data_dir: str, dataset_name: str = "") -> Optional[Path]:
    ratings_path = find_ratings_file(data_dir=data_dir, dataset_name=dataset_name)
    candidate = ratings_path.parent / "movies.csv"
    if candidate.exists():
        return candidate
    return None


def load_movie_titles(path: Path) -> Dict[int, str]:
    df = pd.read_csv(path, usecols=["movieId", "title"])
    return {int(row.movieId): str(row.title) for row in df.itertuples(index=False)}


class SASRecPredictor:
    def __init__(
        self,
        checkpoint_path: str,
        item_raw_ids: np.ndarray,
        device: str = "auto",
    ) -> None:
        self.device = resolve_device(device)
        ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
        model_cfg = ckpt["model_config"]

        self.model = SASRec(
            num_items=int(model_cfg["num_items"]),
            max_seq_len=int(model_cfg["max_seq_len"]),
            d_model=int(model_cfg["d_model"]),
            n_heads=int(model_cfg["n_heads"]),
            n_layers=int(model_cfg["n_layers"]),
            dropout=float(model_cfg["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.max_seq_len = int(model_cfg["max_seq_len"])
        self.item_raw_ids = item_raw_ids.astype(np.int64)
        self.raw_to_internal = {int(raw): idx + 1 for idx, raw in enumerate(self.item_raw_ids)}

    @torch.no_grad()
    def predict(
        self,
        raw_item_sequence: List[int],
        topk: int = 10,
        exclude_seen: bool = True,
    ) -> List[Dict]:
        internal_seq = [self.raw_to_internal[x] for x in raw_item_sequence if x in self.raw_to_internal]
        if not internal_seq:
            raise ValueError("No known items in input sequence.")

        context = internal_seq[-self.max_seq_len :]
        seq = np.zeros(self.max_seq_len, dtype=np.int64)
        seq[-len(context) :] = np.asarray(context, dtype=np.int64)
        seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)

        hidden = self.model(seq_tensor)
        last_hidden = hidden[:, -1, :].squeeze(0)  # [D]

        all_emb = self.model.item_embedding.weight[1:]  # [num_items, D]
        scores = torch.matmul(all_emb, last_hidden)  # [num_items]

        if exclude_seen:
            seen_internal = set(context)
            if seen_internal:
                seen_idx = torch.tensor([x - 1 for x in seen_internal], device=self.device, dtype=torch.long)
                scores.index_fill_(0, seen_idx, float("-inf"))

        valid_k = min(topk, scores.size(0))
        top_scores, top_idx = torch.topk(scores, k=valid_k)
        top_internal = (top_idx + 1).cpu().numpy().astype(np.int64)
        top_scores = top_scores.cpu().numpy()

        results = []
        for rank, (iid, score) in enumerate(zip(top_internal, top_scores), start=1):
            raw_id = int(self.item_raw_ids[iid - 1])
            results.append(
                {
                    "rank": rank,
                    "internal_item_id": int(iid),
                    "raw_item_id": raw_id,
                    "score": float(score),
                }
            )
        return results


def parse_args():
    parser = argparse.ArgumentParser(description="Predict next items using a trained SASRec checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--item_map_path", type=str, default="")
    parser.add_argument("--input_items", type=str, required=True, help="Comma/space separated raw item ids.")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--include_seen", action="store_true")
    parser.add_argument("--with_titles", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.item_map_path:
        item_map_path = Path(args.item_map_path).resolve()
    else:
        item_map_path = ckpt_path.parent / "item_raw_ids.npy"

    if item_map_path.exists():
        item_raw_ids = np.load(item_map_path)
    else:
        # Fallback: rebuild mappings from dataset preprocessing.
        data = preprocess_movielens(data_dir=args.data_dir, dataset_name=args.dataset_name)
        item_raw_ids = data.item_raw_ids

    predictor = SASRecPredictor(
        checkpoint_path=str(ckpt_path),
        item_raw_ids=item_raw_ids,
        device=args.device,
    )

    input_items = parse_item_sequence(args.input_items)
    preds = predictor.predict(
        raw_item_sequence=input_items,
        topk=args.topk,
        exclude_seen=not args.include_seen,
    )

    title_map: Dict[int, str] = {}
    if args.with_titles:
        movies_file = find_movies_file(args.data_dir, args.dataset_name)
        if movies_file is not None:
            title_map = load_movie_titles(movies_file)

    print("Input items:", input_items)
    print(f"Top-{len(preds)} predictions:")
    for row in preds:
        line = (
            f"rank={row['rank']:>2d} | raw_item_id={row['raw_item_id']:>7d} "
            f"| internal_id={row['internal_item_id']:>6d} | score={row['score']:.6f}"
        )
        title = title_map.get(row["raw_item_id"])
        if title:
            line += f" | title={title}"
        print(line)


if __name__ == "__main__":
    main()
