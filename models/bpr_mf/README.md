# BPR-MF Baseline

Classic collaborative filtering ranking baseline:

- User embedding + item embedding (with `padding_idx=0` for item embedding).
- Train with BPR loss:
  - `-log(sigmoid(s(u,pos)-s(u,neg)))`
- Eval with the same protocol as other models:
  - leave-one-out split
  - `1 positive + num_neg_eval negatives`
  - Recall@{5,10,20}, NDCG@{5,10,20}

## Run

From repo root:

```bash
python models/bpr_mf/train.py --config models/bpr_mf/config.yaml
python models/bpr_mf/eval.py --config models/bpr_mf/config.yaml
```

Default outputs:

- `outputs/bpr_mf_ml32m_final/best.pt`
- `outputs/bpr_mf_ml32m_final/metrics_summary.json`
