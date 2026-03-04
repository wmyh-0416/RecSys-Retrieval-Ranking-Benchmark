from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class LeaveOneOutSplit:
    train_sequences: List[np.ndarray]
    valid_items: np.ndarray
    test_items: np.ndarray
    seen_items_sorted: List[np.ndarray]
    selected_uids: np.ndarray


def leave_one_out_split(interactions: pd.DataFrame, max_users: int = 0) -> LeaveOneOutSplit:
    # Requires columns: uid, iid sorted by uid/timestamp/item.
    uid_arr = interactions["uid"].to_numpy(np.int32)
    iid_arr = interactions["iid"].to_numpy(np.int32)

    split_starts = np.r_[0, np.flatnonzero(np.diff(uid_arr)) + 1]
    split_ends = np.r_[split_starts[1:], len(uid_arr)]
    split_uids = uid_arr[split_starts]

    num_users_total = int(uid_arr.max()) + 1
    train_sequences: List[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(num_users_total)]
    valid_items = np.zeros(num_users_total, dtype=np.int32)
    test_items = np.zeros(num_users_total, dtype=np.int32)
    seen_items_sorted: List[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(num_users_total)]
    valid_user_mask = np.zeros(num_users_total, dtype=bool)

    for uid, start, end in zip(split_uids, split_starts, split_ends):
        uid_int = int(uid)
        seq = iid_arr[start:end]
        if seq.size < 3:
            continue
        train_sequences[uid_int] = seq[:-2].copy()
        valid_items[uid_int] = int(seq[-2])
        test_items[uid_int] = int(seq[-1])
        seen_items_sorted[uid_int] = np.unique(seq)
        valid_user_mask[uid_int] = True

    selected_uids = np.flatnonzero(valid_user_mask)
    if selected_uids.size == 0:
        raise ValueError("No valid users left for leave-one-out split.")

    train_sequences = [train_sequences[uid] for uid in selected_uids]
    valid_items = valid_items[selected_uids]
    test_items = test_items[selected_uids]
    seen_items_sorted = [seen_items_sorted[uid] for uid in selected_uids]

    if max_users and max_users > 0:
        limit = min(max_users, len(train_sequences))
        train_sequences = train_sequences[:limit]
        valid_items = valid_items[:limit]
        test_items = test_items[:limit]
        seen_items_sorted = seen_items_sorted[:limit]
        selected_uids = selected_uids[:limit]

    return LeaveOneOutSplit(
        train_sequences=train_sequences,
        valid_items=valid_items,
        test_items=test_items,
        seen_items_sorted=seen_items_sorted,
        selected_uids=selected_uids,
    )
