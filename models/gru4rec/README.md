# GRU4Rec Baseline

RNN sequential recommendation baseline under the unified benchmark protocol.

- Input: user sequence (max length `max_seq_len`)
- Model: item embedding + GRU (+ layer norm)
- Training target: next-item prediction on all valid sequence positions
- Loss: BPR with sampled negatives from unseen items
- Eval: same candidate protocol as other models (`1 pos + num_neg_eval neg`)

## Run

From repo root:

```bash
python models/gru4rec/train.py --config models/gru4rec/config.yaml
python models/gru4rec/eval.py --config models/gru4rec/config.yaml
```

Default outputs:

- `outputs/gru4rec_ml32m_final/best.pt`
- `outputs/gru4rec_ml32m_final/metrics_summary.json`
