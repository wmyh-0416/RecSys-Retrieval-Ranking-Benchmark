# Dual-Tower Retrieval

Dual-Tower is a retrieval-stage model for industrial recommender systems.  
It maps users and items into the same embedding space, then retrieves candidate items with vector similarity (dot product).

## Why Retrieval

- Retrieval stage: fast coarse filtering from full item corpus to a small candidate set (e.g. top-200)
- Ranking stage: expensive fine-grained scoring on those candidates (e.g. SASRec/BERT4Rec)

In production, retrieval + ranking is the standard cascade.

## Model

- User tower:
  - input: user history item sequence
  - item embedding + mean pooling over non-padding positions
  - optional MLP projection
- Item tower:
  - input: item id
  - item embedding + optional MLP projection
- Score:
  - `score(u, i) = dot(user_emb, item_emb)`

## Training

- Positive: next item from user train sequence
- Negatives: in-batch negatives (other samples' positives)
- Loss: sampled-softmax / InfoNCE style cross-entropy

## Evaluation (Shared Protocol)

Evaluation reuses the benchmark's unified protocol:

- same leave-one-out split
- same candidates: `1 positive + num_neg_eval negatives`
- same metrics implementation: Recall@K, NDCG@K from `common/metrics/ranking_metrics.py`

## Run

From project root:

```bash
python retrieval/dual_tower/train.py --config config.yaml
python retrieval/dual_tower/eval.py --config config.yaml
```

The scripts automatically resolve `config.yaml` relative to `retrieval/dual_tower/`.

Recommended explicit form:

```bash
python retrieval/dual_tower/train.py --config retrieval/dual_tower/config.yaml
python retrieval/dual_tower/eval.py --config retrieval/dual_tower/config.yaml
```

## Combine With Ranking Models

1. Run Dual-Tower to retrieve top-N candidates from full corpus.
2. Feed those candidates to ranking models (SASRec/BERT4Rec/GRU4Rec/BPR-MF).
3. Use ranker scores to produce final top-K results.
