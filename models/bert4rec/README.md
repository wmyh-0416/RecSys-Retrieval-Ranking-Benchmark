# BERT4Rec Baseline

Transformer-Encoder sequential recommendation baseline under the unified benchmark protocol.

- Training task: masked item prediction on train sequences
- Masking: random mask with `mask_rate` (default `0.2`), masked positions replaced by `[MASK]`
- Special tokens:
  - `padding_id = 0`
  - `mask_id = num_items + 1`
- Training loss: sampled-softmax style cross-entropy (1 positive + sampled negatives)
- Eval task: next-item prediction with unified protocol (`leave-one-out` + `1 pos + num_neg_eval neg`)

## Why This Is Comparable

BERT4Rec uses bidirectional masked training, while SASRec/GRU4Rec use autoregressive next-item style training.  
For fair comparison, all models still use the same evaluation protocol and candidate construction at valid/test.

## Run

From repo root:

```bash
python models/bert4rec/train.py --config models/bert4rec/config.yaml
python models/bert4rec/eval.py --config models/bert4rec/config.yaml
```

Default outputs:

- `outputs/bert4rec_ml32m_final/best.pt`
- `outputs/bert4rec_ml32m_final/metrics_summary.json`
