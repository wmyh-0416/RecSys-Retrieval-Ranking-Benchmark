from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common.data.preprocessing import PreprocessedData


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


class SequenceTrainDataset(Dataset):
    # SASRec standard shifted training on all non-padding positions.
    def __init__(self, data: PreprocessedData, max_seq_len: int, seed: int = 42) -> None:
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
                neg_arr[i] = _sample_negative_from_unseen(self.num_items, seen_sorted, self.rng)

        return (
            torch.from_numpy(seq_arr),
            torch.from_numpy(pos_arr),
            torch.from_numpy(neg_arr),
        )


class SequenceEvalDataset(Dataset):
    # Eval protocol: 1 positive + N negatives sampled from unseen items.
    def __init__(
        self,
        data: PreprocessedData,
        split: str,
        max_seq_len: int,
        num_neg_eval: int,
        seed: int = 42,
        return_user_id: bool = False,
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
        self.return_user_id = return_user_id

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
        if self.return_user_id:
            return (
                torch.tensor(uid, dtype=torch.long),
                torch.from_numpy(seq_arr),
                torch.from_numpy(candidates),
            )
        return torch.from_numpy(seq_arr), torch.from_numpy(candidates)


def create_train_dataloader(
    data: PreprocessedData,
    max_seq_len: int,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SequenceTrainDataset(data=data, max_seq_len=max_seq_len, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def create_eval_dataloader(
    data: PreprocessedData,
    split: str,
    max_seq_len: int,
    batch_size: int,
    num_neg_eval: int,
    seed: int,
    num_workers: int = 0,
    return_user_id: bool = False,
) -> DataLoader:
    dataset = SequenceEvalDataset(
        data=data,
        split=split,
        max_seq_len=max_seq_len,
        num_neg_eval=num_neg_eval,
        seed=seed,
        return_user_id=return_user_id,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
