import argparse
from pathlib import Path
from typing import Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.utils.io import load_json, write_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate model metrics into results/results.csv")
    parser.add_argument(
        "--metrics_files",
        nargs="+",
        required=True,
        help="List of metrics_summary.json files from model eval/train.",
    )
    parser.add_argument("--output_csv", type=str, default="results/results.csv")
    return parser.parse_args()


def flatten_row(payload: Dict) -> Dict:
    valid = payload.get("valid", {})
    test = payload.get("test", {})
    return {
        "model": payload.get("model", ""),
        "dataset_name": payload.get("dataset_name", ""),
        "checkpoint": payload.get("checkpoint", ""),
        "best_epoch": payload.get("best_epoch", ""),
        "best_valid_ndcg10": payload.get("best_valid_ndcg10", ""),
        "valid_recall@5": valid.get("Recall@5", ""),
        "valid_recall@10": valid.get("Recall@10", ""),
        "valid_recall@20": valid.get("Recall@20", ""),
        "valid_ndcg@5": valid.get("NDCG@5", ""),
        "valid_ndcg@10": valid.get("NDCG@10", ""),
        "valid_ndcg@20": valid.get("NDCG@20", ""),
        "test_recall@5": test.get("Recall@5", ""),
        "test_recall@10": test.get("Recall@10", ""),
        "test_recall@20": test.get("Recall@20", ""),
        "test_ndcg@5": test.get("NDCG@5", ""),
        "test_ndcg@10": test.get("NDCG@10", ""),
        "test_ndcg@20": test.get("NDCG@20", ""),
    }


def main():
    args = parse_args()
    rows: List[Dict] = []

    for metrics_file in args.metrics_files:
        path = Path(metrics_file).resolve()
        if not path.exists():
            print(f"[WARN] metrics file not found, skip: {path}")
            continue
        payload = load_json(str(path))
        rows.append(flatten_row(payload))

    if not rows:
        raise RuntimeError("No valid metrics files found. Nothing to aggregate.")

    fieldnames = list(rows[0].keys())
    write_csv(args.output_csv, rows, fieldnames)
    print(f"Saved aggregated results: {args.output_csv}")


if __name__ == "__main__":
    main()
