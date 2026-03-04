#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-.}"
DATASET_NAME="${DATASET_NAME:-ml-32m}"
SASREC_CONFIG="${SASREC_CONFIG:-models/sasrec/config.yaml}"
SASREC_SAVE_DIR="${SASREC_SAVE_DIR:-outputs/sasrec_ml32m_final}"
POPULARITY_CONFIG="${POPULARITY_CONFIG:-models/popularity/config.yaml}"
POPULARITY_SAVE_DIR="${POPULARITY_SAVE_DIR:-outputs/popularity_ml32m_final}"
BPR_MF_CONFIG="${BPR_MF_CONFIG:-models/bpr_mf/config.yaml}"
BPR_MF_SAVE_DIR="${BPR_MF_SAVE_DIR:-outputs/bpr_mf_ml32m_retrain}"
GRU4REC_CONFIG="${GRU4REC_CONFIG:-models/gru4rec/config.yaml}"
GRU4REC_SAVE_DIR="${GRU4REC_SAVE_DIR:-outputs/gru4rec_ml32m_final}"
BERT4REC_CONFIG="${BERT4REC_CONFIG:-models/bert4rec/config.yaml}"
BERT4REC_SAVE_DIR="${BERT4REC_SAVE_DIR:-outputs/bert4rec_ml32m_final}"
RUN_TRAIN="${RUN_TRAIN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[run_all] DATA_DIR=${DATA_DIR} DATASET_NAME=${DATASET_NAME}"

if [[ "${RUN_TRAIN}" == "1" || ! -f "${SASREC_SAVE_DIR}/best.pt" ]]; then
  echo "[run_all] training sasrec ..."
  "${PYTHON_BIN}" -m models.sasrec.train \
    --config "${SASREC_CONFIG}" \
    --data_dir "${DATA_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --save_dir "${SASREC_SAVE_DIR}"
else
  echo "[run_all] skip training, checkpoint exists: ${SASREC_SAVE_DIR}/best.pt"
fi

echo "[run_all] evaluating sasrec ..."
"${PYTHON_BIN}" -m models.sasrec.eval \
  --data_dir "${DATA_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --checkpoint "${SASREC_SAVE_DIR}/best.pt" \
  --output_json "${SASREC_SAVE_DIR}/metrics_summary.json"

if [[ "${RUN_TRAIN}" == "1" || ! -f "${POPULARITY_SAVE_DIR}/popularity_counts.npy" ]]; then
  echo "[run_all] training popularity ..."
  "${PYTHON_BIN}" models/popularity/train.py \
    --config "${POPULARITY_CONFIG}" \
    --data_dir "${DATA_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --save_dir "${POPULARITY_SAVE_DIR}"
else
  echo "[run_all] skip popularity training, artifact exists: ${POPULARITY_SAVE_DIR}/popularity_counts.npy"
fi

echo "[run_all] evaluating popularity ..."
"${PYTHON_BIN}" models/popularity/eval.py \
  --config "${POPULARITY_CONFIG}" \
  --data_dir "${DATA_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --save_dir "${POPULARITY_SAVE_DIR}" \
  --output_json "${POPULARITY_SAVE_DIR}/metrics_summary.json"

if [[ "${RUN_TRAIN}" == "1" || ! -f "${BPR_MF_SAVE_DIR}/best.pt" ]]; then
  echo "[run_all] training bpr_mf ..."
  "${PYTHON_BIN}" models/bpr_mf/train.py \
    --config "${BPR_MF_CONFIG}" \
    --data_dir "${DATA_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --save_dir "${BPR_MF_SAVE_DIR}"
else
  echo "[run_all] skip bpr_mf training, checkpoint exists: ${BPR_MF_SAVE_DIR}/best.pt"
fi

echo "[run_all] evaluating bpr_mf ..."
"${PYTHON_BIN}" models/bpr_mf/eval.py \
  --config "${BPR_MF_CONFIG}" \
  --data_dir "${DATA_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --save_dir "${BPR_MF_SAVE_DIR}" \
  --output_json "${BPR_MF_SAVE_DIR}/metrics_summary.json"

if [[ "${RUN_TRAIN}" == "1" || ! -f "${GRU4REC_SAVE_DIR}/best.pt" ]]; then
  echo "[run_all] training gru4rec ..."
  "${PYTHON_BIN}" models/gru4rec/train.py \
    --config "${GRU4REC_CONFIG}" \
    --data_dir "${DATA_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --save_dir "${GRU4REC_SAVE_DIR}"
else
  echo "[run_all] skip gru4rec training, checkpoint exists: ${GRU4REC_SAVE_DIR}/best.pt"
fi

echo "[run_all] evaluating gru4rec ..."
"${PYTHON_BIN}" models/gru4rec/eval.py \
  --config "${GRU4REC_CONFIG}" \
  --data_dir "${DATA_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --save_dir "${GRU4REC_SAVE_DIR}" \
  --output_json "${GRU4REC_SAVE_DIR}/metrics_summary.json"

if [[ "${RUN_TRAIN}" == "1" || ! -f "${BERT4REC_SAVE_DIR}/best.pt" ]]; then
  echo "[run_all] training bert4rec ..."
  "${PYTHON_BIN}" models/bert4rec/train.py \
    --config "${BERT4REC_CONFIG}" \
    --data_dir "${DATA_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --save_dir "${BERT4REC_SAVE_DIR}"
else
  echo "[run_all] skip bert4rec training, checkpoint exists: ${BERT4REC_SAVE_DIR}/best.pt"
fi

echo "[run_all] evaluating bert4rec ..."
"${PYTHON_BIN}" models/bert4rec/eval.py \
  --config "${BERT4REC_CONFIG}" \
  --data_dir "${DATA_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --save_dir "${BERT4REC_SAVE_DIR}" \
  --output_json "${BERT4REC_SAVE_DIR}/metrics_summary.json"

echo "[run_all] aggregating results ..."
"${PYTHON_BIN}" scripts/aggregate_results.py \
  --metrics_files \
  "${SASREC_SAVE_DIR}/metrics_summary.json" \
  "${POPULARITY_SAVE_DIR}/metrics_summary.json" \
  "${BPR_MF_SAVE_DIR}/metrics_summary.json" \
  "${GRU4REC_SAVE_DIR}/metrics_summary.json" \
  "${BERT4REC_SAVE_DIR}/metrics_summary.json" \
  --output_csv results/results.csv

echo "[run_all] done."
