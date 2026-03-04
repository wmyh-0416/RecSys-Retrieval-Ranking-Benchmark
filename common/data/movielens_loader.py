from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


USER_COL_CANDIDATES = ["userid", "user_id", "user", "uid"]
ITEM_COL_CANDIDATES = ["movieid", "movie_id", "itemid", "item_id", "item", "iid"]
TIMESTAMP_COL_CANDIDATES = ["timestamp", "time", "ts", "eventtime", "event_time"]


@dataclass
class RatingsLoadResult:
    ratings_path: str
    delimiter: str
    detected_columns: Dict[str, str]
    interactions: pd.DataFrame


def _normalize_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _pick_column(normalized_to_raw: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        normalized = _normalize_col_name(candidate)
        if normalized in normalized_to_raw:
            return normalized_to_raw[normalized]
    return None


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
    header = 0 if any(ch.isalpha() for ch in first_token) else None
    return delimiter, header


def load_ratings(data_dir: str, dataset_name: str = "") -> RatingsLoadResult:
    ratings_path = find_ratings_file(data_dir=data_dir, dataset_name=dataset_name)
    delimiter, header = _detect_delimiter_and_header(ratings_path)

    if header is None:
        probe = pd.read_csv(ratings_path, sep=delimiter, header=None, engine="python", nrows=1)
        num_cols = probe.shape[1]
        if num_cols < 3:
            raise ValueError(
                f"Expected at least 3 columns in ratings file, got {num_cols} ({ratings_path})."
            )
        user_idx = 0
        item_idx = 1
        ts_idx = 3 if num_cols > 3 else 2

        df = pd.read_csv(
            ratings_path,
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
        header_df = pd.read_csv(ratings_path, sep=delimiter, engine="python", nrows=0)
        raw_cols = list(header_df.columns)
        normalized_to_raw: Dict[str, str] = {
            _normalize_col_name(col): col for col in raw_cols
        }

        user_col = _pick_column(normalized_to_raw, USER_COL_CANDIDATES)
        item_col = _pick_column(normalized_to_raw, ITEM_COL_CANDIDATES)
        ts_col = _pick_column(normalized_to_raw, TIMESTAMP_COL_CANDIDATES)

        if user_col is None or item_col is None or ts_col is None:
            if len(raw_cols) < 3:
                raise ValueError(f"Could not infer columns from ratings file: {ratings_path}")
            user_col = user_col or raw_cols[0]
            item_col = item_col or raw_cols[1]
            ts_col = ts_col or (raw_cols[3] if len(raw_cols) > 3 else raw_cols[2])

        df = pd.read_csv(
            ratings_path,
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

    return RatingsLoadResult(
        ratings_path=str(ratings_path),
        delimiter=delimiter,
        detected_columns=detected_columns,
        interactions=df,
    )
