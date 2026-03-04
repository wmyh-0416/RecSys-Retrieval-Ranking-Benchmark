from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from common.data.movielens_loader import load_ratings
from common.data.split import leave_one_out_split


@dataclass
class PreprocessedData:
    ratings_path: str
    delimiter: str
    detected_columns: Dict[str, str]
    num_users: int
    num_items: int
    num_interactions: int
    train_sequences: List[np.ndarray]
    valid_items: np.ndarray
    test_items: np.ndarray
    seen_items_sorted: List[np.ndarray]
    user_raw_ids: np.ndarray
    item_raw_ids: np.ndarray

    @property
    def padding_id(self) -> int:
        return 0

    @property
    def mask_id(self) -> int:
        # BERT4Rec special token id, appended after [1..num_items].
        return self.num_items + 1

    @property
    def num_tokens(self) -> int:
        # Includes padding(0), items(1..num_items), and [MASK](num_items+1).
        return self.num_items + 2


def k_core_filter(df: pd.DataFrame, min_user_inter: int, min_item_inter: int) -> pd.DataFrame:
    if min_user_inter <= 0 and min_item_inter <= 0:
        return df

    current = df
    while True:
        before = len(current)
        if min_user_inter > 0:
            user_counts = current["user"].value_counts()
            keep_users = user_counts[user_counts >= min_user_inter].index
            current = current[current["user"].isin(keep_users)]
        if min_item_inter > 0:
            item_counts = current["item"].value_counts()
            keep_items = item_counts[item_counts >= min_item_inter].index
            current = current[current["item"].isin(keep_items)]
        if len(current) == before:
            break
    return current


def preprocess_movielens(
    data_dir: str,
    dataset_name: str = "",
    min_user_inter: int = 5,
    min_item_inter: int = 0,
    max_users: int = 0,
) -> PreprocessedData:
    load_result = load_ratings(data_dir=data_dir, dataset_name=dataset_name)
    df = load_result.interactions

    df = k_core_filter(df, min_user_inter=min_user_inter, min_item_inter=min_item_inter)

    # Leave-one-out needs at least 3 interactions per user.
    user_counts = df["user"].value_counts()
    keep_users = user_counts[user_counts >= 3].index
    df = df[df["user"].isin(keep_users)]
    if df.empty:
        raise ValueError("No interactions left after filtering.")

    df = df.sort_values(["user", "timestamp", "item"], kind="mergesort").reset_index(drop=True)

    user_codes, user_uniques = pd.factorize(df["user"], sort=False)
    item_codes, item_uniques = pd.factorize(df["item"], sort=False)
    df["uid"] = user_codes.astype(np.int32)
    df["iid"] = (item_codes + 1).astype(np.int32)  # 0 reserved for padding

    split_result = leave_one_out_split(df, max_users=max_users)
    num_users = len(split_result.train_sequences)
    if num_users == 0:
        raise ValueError("No users available after max_users truncation.")

    return PreprocessedData(
        ratings_path=load_result.ratings_path,
        delimiter=load_result.delimiter,
        detected_columns=load_result.detected_columns,
        num_users=num_users,
        num_items=int(df["iid"].max()),
        num_interactions=int(len(df)),
        train_sequences=split_result.train_sequences,
        valid_items=split_result.valid_items,
        test_items=split_result.test_items,
        seen_items_sorted=split_result.seen_items_sorted,
        user_raw_ids=user_uniques.to_numpy()[split_result.selected_uids],
        item_raw_ids=item_uniques.to_numpy(),
    )
