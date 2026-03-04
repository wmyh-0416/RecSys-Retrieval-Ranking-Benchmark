# SASRec

SASRec implementation under the unified benchmark protocol.

## Files

- `model.py`: SASRec decoder-only Transformer with causal mask.
- `train.py`: train with BPR loss, save best by valid NDCG@10.
- `eval.py`: evaluate Recall@{5,10,20} and NDCG@{5,10,20}.
- `config.yaml`: default run config.

## Run

From repo root:

```bash
python -m models.sasrec.train --config models/sasrec/config.yaml
python -m models.sasrec.eval \
  --data_dir . \
  --dataset_name ml-32m \
  --checkpoint outputs/sasrec_ml32m_final/best.pt
```

The data pipeline and metrics are shared from `common/` and are consistent across models.
