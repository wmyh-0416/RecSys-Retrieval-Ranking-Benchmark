import json
import logging
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def create_logger(save_dir: str, name: str = "train") -> logging.Logger:
    ensure_dir(save_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(Path(save_dir) / f"{name}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_checkpoint(path: str, state: Dict) -> None:
    ensure_dir(str(Path(path).parent))
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict:
    return torch.load(path, map_location=map_location)


def format_metrics(metrics: Dict[str, float]) -> str:
    order = ["Recall@5", "Recall@10", "Recall@20", "NDCG@5", "NDCG@10", "NDCG@20"]
    parts = []
    for key in order:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    for key, value in metrics.items():
        if key not in order:
            parts.append(f"{key}={value:.4f}")
    return " | ".join(parts)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_plain_args_dict(args) -> Dict:
    payload = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
        else:
            payload[key] = str(value)
    return payload
