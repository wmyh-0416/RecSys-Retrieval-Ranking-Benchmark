import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: str, payload: Dict) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(path: str, state: Dict) -> None:
    ensure_dir(str(Path(path).parent))
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict:
    return torch.load(path, map_location=map_location)


def write_csv(path: str, rows: List[Dict], fieldnames: Iterable[str]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
