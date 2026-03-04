from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


USER_COL_CANDIDATES = [
    "userid",
    "user_id",
    "user",
    "uid",
]
ITEM_COL_CANDIDATES = [
    "movieid",
    "movie_id",
    "itemid",
    "item_id",
    "item",
    "iid",
]
TIMESTAMP_COL_CANDIDATES = [
    "timestamp",
    "time",
    "ts",
    "eventtime",
    "event_time",
]


@dataclass
class DataBundle:
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


def _normalize_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _contains(sorted_array: np.ndarray, value: int) -> bool:
    idx = np.searchsorted(sorted_array, value)
    return idx < sorted_array.size and int(sorted_array[idx]) == value


def _sample_negative_from_unseen(
    num_items: int,
    seen_sorted: np.ndarray,
    rng: np.random.Generator,
) -> int:
    if seen_sorted.size >= num_items:
        return 0
    while True:
        item = int(rng.integers(1, num_items + 1))
        if not _contains(seen_sorted, item):
            return item


def find_ratings_file(data_dir: str, dataset_name: str = "") -> Path:
    base_dir = Path(data_dir).expanduser().resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {base_dir}")

    search_roots: List[Path] = []
    if dataset_name:
        direct = base_dir / dataset_name
        if direct.exists():
            search_roots.append(direct)
        for child in base_dir.iterdir():
            if child.is_dir() and dataset_name.lower() in child.name.lower():
                search_roots.append(child)
    search_roots.append(base_dir)

    patterns = [
        "ratings.csv",
        "ratings.dat",
        "ratings.txt",
        "*ratings*.csv",
        "*ratings*.dat",
        "*ratings*.txt",
    ]
    found: List[Path] = []
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for p in root.rglob(pattern):
                if p.is_file():
                    resolved = p.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        found.append(resolved)

    if not found:
        raise FileNotFoundError(
            f"No ratings file found under {base_dir}. "
            "Expected ratings.csv / ratings.dat / ratings.txt (or similar)."
        )

    if dataset_name:
        matched = [p for p in found if dataset_name.lower() in str(p).lower()]
        if matched:
            found = matched

    suffix_priority = {".csv": 0, ".dat": 1, ".txt": 2}
    found = sorted(
        found,
        key=lambda p: (
            suffix_priority.get(p.suffix.lower(), 9),
            len(str(p)),
            str(p),
        ),
    )
    return found[0]


def _detect_delimiter_and_header(path: Path) -> Tuple[str, Optional[int]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

    if "::" in first_line:
        delimiter = "::"
    elif "\t" in first_line:
        delimiter = "\t"
    elif "," in first_line:
        delimiter = ","
    else:
        delimiter = r"\s+"

    first_token = re.split(r"::|\t|,|\s+", first_line)[0] if first_line else ""
    has_header = any(ch.isalpha() for ch in first_token)
    header = 0 if has_header else None
    return delimiter, header


def _pick_column(normalized_to_raw: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        normalized = _normalize_col_name(candidate)
        if normalized in normalized_to_raw:
            return normalized_to_raw[normalized]
    return None


def _read_ratings(path: Path) -> Tuple[pd.DataFrame, str, Dict[str, str]]:
    delimiter, header = _detect_delimiter_and_header(path)

    if header is None:
        probe = pd.read_csv(path, sep=delimiter, header=None, engine="python", nrows=1)
        num_cols = probe.shape[1]
        if num_cols < 3:
            raise ValueError(f"Expected at least 3 columns in ratings file, got {num_cols} ({path}).")
        user_idx = 0
        item_idx = 1
        ts_idx = 3 if num_cols > 3 else 2
        df = pd.read_csv(
            path,
            sep=delimiter,
            header=None,
            engine="python",
            usecols=[user_idx, item_idx, ts_idx],
        )
        df.columns = ["user", "item", "timestamp"]
        detected_columns = {
            "user": f"col_{user_idx}",
            "item": f"col_{item_idx}",
            "timestamp": f"col_{ts_idx}",
        }
    else:
        header_df = pd.read_csv(path, sep=delimiter, engine="python", nrows=0)
        raw_cols = list(header_df.columns)
        normalized_to_raw: Dict[str, str] = {}
        for col in raw_cols:
            normalized_to_raw[_normalize_col_name(col)] = col

        user_col = _pick_column(normalized_to_raw, USER_COL_CANDIDATES)
        item_col = _pick_column(normalized_to_raw, ITEM_COL_CANDIDATES)
        ts_col = _pick_column(normalized_to_raw, TIMESTAMP_COL_CANDIDATES)

        if user_col is None or item_col is None or ts_col is None:
            if len(raw_cols) < 3:
                raise ValueError(f"Could not infer columns from ratings file: {path}")
            fallback_user = raw_cols[0]
            fallback_item = raw_cols[1]
            fallback_ts = raw_cols[3] if len(raw_cols) > 3 else raw_cols[2]
            user_col = user_col or fallback_user
            item_col = item_col or fallback_item
            ts_col = ts_col or fallback_ts

        df = pd.read_csv(
            path,
            sep=delimiter,
            engine="python",
            usecols=[user_col, item_col, ts_col],
        )
        df = df.rename(columns={user_col: "user", item_col: "item", ts_col: "timestamp"})
        detected_columns = {"user": user_col, "item": item_col, "timestamp": ts_col}

    for col in ["user", "item", "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["user", "item", "timestamp"])
    df = df.astype({"user": "int64", "item": "int64", "timestamp": "int64"})

    return df, delimiter, detected_columns


def _k_core_filter(df: pd.DataFrame, min_user_inter: int, min_item_inter: int) -> pd.DataFrame:
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
) -> DataBundle:
    ratings_path = find_ratings_file(data_dir=data_dir, dataset_name=dataset_name)
    df, delimiter, detected_columns = _read_ratings(ratings_path)

    df = _k_core_filter(df, min_user_inter=min_user_inter, min_item_inter=min_item_inter)

    # Leave-one-out split requires at least 3 interactions per user.
    user_counts = df["user"].value_counts()
    keep_users = user_counts[user_counts >= 3].index
    df = df[df["user"].isin(keep_users)]
    if df.empty:
        raise ValueError("No interactions left after filtering.")

    df = df.sort_values(["user", "timestamp", "item"], kind="mergesort").reset_index(drop=True)

    user_codes, user_uniques = pd.factorize(df["user"], sort=False)
    item_codes, item_uniques = pd.factorize(df["item"], sort=False)
    df["uid"] = user_codes.astype(np.int32)
    df["iid"] = (item_codes + 1).astype(np.int32)  # 0 is reserved for padding.

    uid_arr = df["uid"].to_numpy(np.int32)
    iid_arr = df["iid"].to_numpy(np.int32)

    split_starts = np.r_[0, np.flatnonzero(np.diff(uid_arr)) + 1]
    split_ends = np.r_[split_starts[1:], len(uid_arr)]
    split_uids = uid_arr[split_starts]

    num_users_total = int(uid_arr.max()) + 1
    num_items = int(iid_arr.max())

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

    valid_uids = np.flatnonzero(valid_user_mask)
    if valid_uids.size == 0:
        raise ValueError("No valid users left for leave-one-out split.")

    selected_uids = valid_uids.copy()
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

    num_users = len(train_sequences)
    if num_users == 0:
        raise ValueError("No users available after max_users truncation.")

    return DataBundle(
        ratings_path=str(ratings_path),
        delimiter=delimiter,
        detected_columns=detected_columns,
        num_users=num_users,
        num_items=num_items,
        num_interactions=int(len(df)),
        train_sequences=train_sequences,
        valid_items=valid_items,
        test_items=test_items,
        seen_items_sorted=seen_items_sorted,
        user_raw_ids=user_uniques.to_numpy()[selected_uids],
        item_raw_ids=item_uniques.to_numpy(),
    )


class SASRecTrainDataset(Dataset):
    def __init__(
        self,
        data: DataBundle,
        max_seq_len: int,
        seed: int = 42,
    ) -> None:
        self.train_sequences = data.train_sequences
        self.seen_items_sorted = data.seen_items_sorted
        self.num_items = data.num_items
        self.max_seq_len = max_seq_len
        self.rng = np.random.default_rng(seed)

        users = []
        for uid, seq in enumerate(self.train_sequences):
            if seq.size >= 2 and self.seen_items_sorted[uid].size < self.num_items:
                users.append(uid)
        self.users = np.asarray(users, dtype=np.int32)

    def __len__(self) -> int:
        return int(self.users.size)

    def __getitem__(self, index: int):
        uid = int(self.users[index])
        seq = self.train_sequences[uid]
        seen_sorted = self.seen_items_sorted[uid]

        input_tokens = seq[:-1]
        pos_tokens = seq[1:]
        if input_tokens.size == 0:
            raise RuntimeError("Encountered empty training sample.")

        input_tokens = input_tokens[-self.max_seq_len :]
        pos_tokens = pos_tokens[-self.max_seq_len :]

        seq_arr = np.zeros(self.max_seq_len, dtype=np.int64)
        pos_arr = np.zeros(self.max_seq_len, dtype=np.int64)
        neg_arr = np.zeros(self.max_seq_len, dtype=np.int64)

        seq_arr[-input_tokens.size :] = input_tokens
        pos_arr[-pos_tokens.size :] = pos_tokens

        for i in range(self.max_seq_len):
            if pos_arr[i] > 0:
                neg_arr[i] = _sample_negative_from_unseen(
                    num_items=self.num_items,
                    seen_sorted=seen_sorted,
                    rng=self.rng,
                )

        return (
            torch.from_numpy(seq_arr),
            torch.from_numpy(pos_arr),
            torch.from_numpy(neg_arr),
        )


class SASRecEvalDataset(Dataset):
    def __init__(
        self,
        data: DataBundle,
        split: str,
        max_seq_len: int,
        num_neg_eval: int,
        seed: int = 42,
    ) -> None:
        if split not in {"valid", "test"}:
            raise ValueError(f"split must be 'valid' or 'test', got: {split}")
        self.split = split
        self.train_sequences = data.train_sequences
        self.valid_items = data.valid_items
        self.test_items = data.test_items
        self.seen_items_sorted = data.seen_items_sorted
        self.num_items = data.num_items
        self.max_seq_len = max_seq_len
        self.num_neg_eval = num_neg_eval
        self.seed = seed

        users = []
        for uid, seq in enumerate(self.train_sequences):
            if self.seen_items_sorted[uid].size >= self.num_items:
                continue
            if split == "valid":
                if seq.size >= 1:
                    users.append(uid)
            else:
                if seq.size >= 1 and self.valid_items[uid] > 0:
                    users.append(uid)
        self.users = np.asarray(users, dtype=np.int32)

    def __len__(self) -> int:
        return int(self.users.size)

    def _build_context(self, uid: int) -> np.ndarray:
        train_seq = self.train_sequences[uid]
        if self.split == "valid":
            context = train_seq[-self.max_seq_len :]
        else:
            valid_item = int(self.valid_items[uid])
            if self.max_seq_len == 1:
                context = np.asarray([valid_item], dtype=np.int32)
            else:
                tail = train_seq[-(self.max_seq_len - 1) :]
                context = np.concatenate([tail, np.asarray([valid_item], dtype=np.int32)], axis=0)
        return context

    def _sample_eval_negatives(self, uid: int, pos_item: int) -> np.ndarray:
        seen_sorted = self.seen_items_sorted[uid]
        available = self.num_items - int(seen_sorted.size)
        need_unique = min(self.num_neg_eval, max(0, available))

        rng = np.random.default_rng(self.seed + uid * 104729 + (13 if self.split == "valid" else 29))
        negatives: List[int] = []
        negatives_set = set()

        while len(negatives) < need_unique:
            candidate = int(rng.integers(1, self.num_items + 1))
            if candidate == pos_item:
                continue
            if candidate in negatives_set:
                continue
            if not _contains(seen_sorted, candidate):
                negatives.append(candidate)
                negatives_set.add(candidate)

        if need_unique == 0:
            return np.zeros(self.num_neg_eval, dtype=np.int64)

        while len(negatives) < self.num_neg_eval:
            negatives.append(negatives[len(negatives) % need_unique])

        return np.asarray(negatives, dtype=np.int64)

    def __getitem__(self, index: int):
        uid = int(self.users[index])
        pos_item = int(self.valid_items[uid] if self.split == "valid" else self.test_items[uid])
        context = self._build_context(uid)

        seq_arr = np.zeros(self.max_seq_len, dtype=np.int64)
        seq_arr[-context.size :] = context

        negatives = self._sample_eval_negatives(uid, pos_item)
        candidates = np.empty(self.num_neg_eval + 1, dtype=np.int64)
        candidates[0] = pos_item
        candidates[1:] = negatives

        return torch.from_numpy(seq_arr), torch.from_numpy(candidates)


def create_train_dataloader(
    data: DataBundle,
    max_seq_len: int,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SASRecTrainDataset(data=data, max_seq_len=max_seq_len, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def create_eval_dataloader(
    data: DataBundle,
    split: str,
    max_seq_len: int,
    batch_size: int,
    num_neg_eval: int,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SASRecEvalDataset(
        data=data,
        split=split,
        max_seq_len=max_seq_len,
        num_neg_eval=num_neg_eval,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
