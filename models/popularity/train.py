from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data.preprocessing import preprocess_movielens
from common.utils.config import load_config, merge_config
from common.utils.io import ensure_dir, save_json
from common.utils.logger import create_logger
from common.utils.seed import set_seed
from models.popularity.model import PopularityModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train popularity baseline with unified benchmark protocol.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--min_user_inter", type=int, default=None)
    parser.add_argument("--min_item_inter", type=int, default=None)
    parser.add_argument("--max_users", type=int, default=None)
    return parser.parse_args()


def resolve_train_config(args) -> Dict:
    defaults = {
        "data_dir": ".",
        "dataset_name": "",
        "seed": 42,
        "save_dir": "outputs/popularity",
        "min_user_inter": 5,
        "min_item_inter": 0,
        "max_users": 0,
    }
    cfg = dict(defaults)
    if args.config:
        cfg = merge_config(cfg, load_config(args.config))
    overrides = vars(args).copy()
    overrides.pop("config", None)
    cfg = merge_config(cfg, overrides)
    cfg["max_users"] = int(cfg.get("max_users", 0) or 0)
    return cfg


def main():
    args = parse_args()
    cfg = resolve_train_config(args)

    ensure_dir(str(cfg["save_dir"]))
    logger = create_logger(str(cfg["save_dir"]), name="train")
    set_seed(int(cfg["seed"]))

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

    model = PopularityModel(num_items=data.num_items)
    model.fit(data.train_sequences)

    counts_path = Path(cfg["save_dir"]) / "popularity_counts.npy"
    model.save(str(counts_path))
    np.save(Path(cfg["save_dir"]) / "item_raw_ids.npy", data.item_raw_ids.astype(np.int64))
    np.save(Path(cfg["save_dir"]) / "user_raw_ids.npy", data.user_raw_ids.astype(np.int64))

    top_item_internal = int(np.argmax(model.counts))
    top_item_count = int(model.counts[top_item_internal])
    top_item_raw = int(data.item_raw_ids[top_item_internal - 1]) if top_item_internal > 0 else -1

    logger.info("Saved popularity counts: %s", counts_path)
    logger.info(
        "Most popular item | internal_id=%d | raw_item_id=%d | count=%d",
        top_item_internal,
        top_item_raw,
        top_item_count,
    )

    save_json(
        str(Path(cfg["save_dir"]) / "train_args.json"),
        {
            "model": "popularity",
            "data_dir": str(cfg["data_dir"]),
            "dataset_name": str(cfg["dataset_name"]),
            "seed": int(cfg["seed"]),
            "save_dir": str(cfg["save_dir"]),
            "min_user_inter": int(cfg["min_user_inter"]),
            "min_item_inter": int(cfg["min_item_inter"]),
            "max_users": int(cfg["max_users"]),
            "ratings_path": data.ratings_path,
            "detected_columns": data.detected_columns,
            "num_users": data.num_users,
            "num_items": data.num_items,
            "num_interactions": data.num_interactions,
        },
    )

    print(f"Saved popularity model to: {counts_path}")


if __name__ == "__main__":
    main()
