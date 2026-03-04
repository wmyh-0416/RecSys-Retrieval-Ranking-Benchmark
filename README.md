# Sequential Recommendation Benchmark (MovieLens)

This repository is refactored into a reproducible benchmark layout:

- Shared data pipeline in `common/data`
- Shared metrics in `common/metrics`
- Shared utilities in `common/utils`
- Model-specific code in `models/<model_name>`
- Retrieval modules in `retrieval/<module_name>`

Current model set:

- `sasrec` (`models/sasrec`)
- `popularity` (`models/popularity`)
- `bpr_mf` (`models/bpr_mf`)
- `gru4rec` (`models/gru4rec`)
- `bert4rec` (`models/bert4rec`)

## Repository Structure

```text
repo_root/
  datasets/
  common/
    data/
      movielens_loader.py
      preprocessing.py
      split.py
      sampler.py
    metrics/
      ranking_metrics.py
    utils/
      seed.py
      config.py
      logger.py
      io.py
  models/
    sasrec/
      model.py
      train.py
      eval.py
      config.yaml
      README.md
  retrieval/
    dual_tower/
      model.py
      train.py
      eval.py
      config.yaml
      README.md
    popularity/
      model.py
      train.py
      eval.py
      config.yaml
      README.md
    bpr_mf/
      model.py
      train.py
      eval.py
      config.yaml
      README.md
    gru4rec/
      model.py
      train.py
      eval.py
      config.yaml
      README.md
    bert4rec/
      model.py
      train.py
      eval.py
      config.yaml
      README.md
  scripts/
    run_all.sh
    aggregate_results.py
  results/
    results.csv
```

## Unified Data Protocol (Shared Across Models)

This benchmark uses one strict protocol (currently aligned with the existing SASRec implementation):

1. Read local MovieLens ratings file (`ratings.csv` / `ratings.dat` / `ratings.txt`) with auto column detection (`user/item/timestamp`).
2. Treat interactions as implicit feedback (rating value is ignored).
3. Sort each user sequence by `timestamp` ascending (tie-break by item id).
4. Filter rules:
   - enabled by default: remove users with interactions `< 5` (`min_user_inter=5`)
   - optional: item k-core filter (`min_item_inter`, default `0`, disabled)
5. Remap ids:
   - item ids -> `[1..num_items]`, `0` reserved for padding
   - user ids -> `[0..num_users-1]`
6. Leave-one-out split per user:
   - last interaction -> `test`
   - second-last interaction -> `valid`
   - remaining prefix -> `train`
7. Negative sampling:
   - train: model-specific (e.g., SASRec/GRU4Rec use sampled negatives; BERT4Rec uses random masking + sampled-softmax)
   - eval: `1 positive + num_neg_eval negatives` (default `100`) sampled from unseen items
8. Metrics:
   - Recall@{5,10,20}
   - NDCG@{5,10,20}

## Unified Evaluation Protocol

For each user and each split (valid/test):

- Build candidate set: `1 pos + N neg`
- Score all candidates
- Rank candidates by score
- Compute:
  - Recall@K = 1 if positive is in top-K else 0
  - NDCG@K = `1/log2(rank+1)` if positive rank <= K else 0
- Final score = average across users

## Reproducibility

- Fixed seeds via `common/utils/seed.py`
- Logged hyperparameters and detected data schema
- Best checkpoint selected by valid `NDCG@10`
- Training/evaluation scripts write machine-readable `metrics_summary.json`
- `scripts/aggregate_results.py` collects all model metrics into `results/results.csv`

## Run SASRec

Train:

```bash
python -m models.sasrec.train --config models/sasrec/config.yaml
```

Eval:

```bash
python -m models.sasrec.eval \
  --data_dir . \
  --dataset_name ml-32m \
  --checkpoint outputs/sasrec_ml32m_final/best.pt
```

## Run Popularity

Train:

```bash
python models/popularity/train.py --config models/popularity/config.yaml
```

Eval:

```bash
python models/popularity/eval.py --config models/popularity/config.yaml
```

## Run BPR-MF

Train:

```bash
python models/bpr_mf/train.py --config models/bpr_mf/config.yaml
```

Eval:

```bash
python models/bpr_mf/eval.py --config models/bpr_mf/config.yaml
```

## Run GRU4Rec

Train:

```bash
python models/gru4rec/train.py --config models/gru4rec/config.yaml
```

Eval:

```bash
python models/gru4rec/eval.py --config models/gru4rec/config.yaml
```

## Run BERT4Rec

Train:

```bash
python models/bert4rec/train.py --config models/bert4rec/config.yaml
```

Eval:

```bash
python models/bert4rec/eval.py --config models/bert4rec/config.yaml
```

## One-Click All Models

```bash
bash scripts/run_all.sh
```

Useful env overrides:

```bash
RUN_TRAIN=1 DATA_DIR=. DATASET_NAME=ml-32m PYTHON_BIN=python bash scripts/run_all.sh
```

`run_all.sh` will:

1. Train SASRec if checkpoint missing (or `RUN_TRAIN=1`)
2. Evaluate SASRec with unified protocol
3. Train Popularity if artifact missing (or `RUN_TRAIN=1`)
4. Evaluate Popularity with unified protocol
5. Train BPR-MF if checkpoint missing (or `RUN_TRAIN=1`)
6. Evaluate BPR-MF with unified protocol
7. Train GRU4Rec if checkpoint missing (or `RUN_TRAIN=1`)
8. Evaluate GRU4Rec with unified protocol
9. Train BERT4Rec if checkpoint missing (or `RUN_TRAIN=1`)
10. Evaluate BERT4Rec with unified protocol
11. Aggregate to `results/results.csv`

## Retrieval: Dual-Tower

Dual-Tower is a retrieval-stage model (candidate generation) and complements ranking models.

Train:

```bash
python retrieval/dual_tower/train.py --config retrieval/dual_tower/config.yaml
```

Eval:

```bash
python retrieval/dual_tower/eval.py --config retrieval/dual_tower/config.yaml
```

## Existing Reproduced SASRec Result (ml-32m)

Best checkpoint:

- `outputs/sasrec_ml32m_final/best.pt` (best epoch = 3 by valid NDCG@10)

Validation:

- Recall@5 = 0.9026
- Recall@10 = 0.9681
- Recall@20 = 0.9906
- NDCG@5 = 0.7348
- NDCG@10 = 0.7564
- NDCG@20 = 0.7622

Test:

- Recall@5 = 0.8928
- Recall@10 = 0.9626
- Recall@20 = 0.9887
- NDCG@5 = 0.7231
- NDCG@10 = 0.7461
- NDCG@20 = 0.7528

## Result Snapshot (from `results/results.csv`)

| model | dataset | valid_ndcg@10 | test_ndcg@10 | test_recall@10 |
| --- | --- | ---: | ---: | ---: |
| sasrec | ml-32m | 0.7564 | 0.7461 | 0.9626 |
| popularity | ml-32m | 0.7257 | 0.7156 | 0.9532 |
| bpr_mf | ml-32m | 0.7786 | 0.7662 | 0.9644 |
| gru4rec | ml-32m | 0.7280 | 0.7171 | 0.9549 |
| bert4rec | ml-32m | 0.8160 | 0.8034 | 0.9721 |

## Notes on Protocol Stability

- The refactor does **not** change the data split / negative sampling / metric definitions relative to the prior SASRec implementation.
- Optional behavior remains explicitly configurable (`min_item_inter`, `max_users`, etc.) and defaults preserve previous behavior.
- BERT4Rec uses masked-item training while evaluation remains the same next-item candidate ranking protocol, so metrics stay directly comparable.
