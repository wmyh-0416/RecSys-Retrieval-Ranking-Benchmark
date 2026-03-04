# Popularity Baseline

Simple strongest baseline under the same benchmark protocol:

- Count item frequency using **train sequences only**.
- Score candidate item by `score(item) = popularity_count(item)`.
- Evaluate with the same candidate construction and metrics as SASRec.

## Files

- `model.py`: popularity counter/scorer
- `train.py`: build and save popularity counts
- `eval.py`: evaluate Recall@{5,10,20} and NDCG@{5,10,20}
- `config.yaml`: default config

## Run

From repo root:

```bash
python models/popularity/train.py --config models/popularity/config.yaml
python models/popularity/eval.py --config models/popularity/config.yaml
```

Outputs are saved to `outputs/popularity_ml32m_final/` by default.
