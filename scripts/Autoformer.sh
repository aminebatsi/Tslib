#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

model_name="Autoformer"
E_LAYERS=2
D_LAYERS=1
FACTOR=3
ITR=1

ROOT_PATH="./datasets/crypto"
DATA_NAME="custom"
TARGET="return"

FEATURES="MS"
ENC_IN=28
DEC_IN=28
C_OUT=28


SEQ_LEN=32
LABEL_LEN=16
PRED_LENS=(1)


INTERVAL_TAG="1D"  
SYMBOLS=("BTC-USD")

echo "Start training loops for ${model_name}"

for sym in "${SYMBOLS[@]}"; do
  DATA_PATH="final_dataset.csv"

  for PRED_LEN in "${PRED_LENS[@]}"; do
    MODEL_ID="${SYMBOL_SAFE}_${INTERVAL_TAG}_${SEQ_LEN}_${PRED_LEN}_${model_name}_${FEATURES}"
    DES="${sym}-${INTERVAL_TAG}_${model_name}_${FEATURES}_sl${SEQ_LEN}_pl${PRED_LEN}"

    echo "============================================================"
    echo "Training ${model_name} | ${sym} | pred_len=${PRED_LEN}"
    echo "MODEL_ID=${MODEL_ID}"
    echo "============================================================"

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "${ROOT_PATH}" \
      --data_path "${DATA_PATH}" \
      --model_id "${MODEL_ID}" \
      --model "${model_name}" \
      --data "${DATA_NAME}" \
      --features "${FEATURES}" \
      --target "${TARGET}" \
      --freq d \
      --seq_len ${SEQ_LEN} \
      --label_len ${LABEL_LEN} \
      --pred_len ${PRED_LEN} \
      --e_layers ${E_LAYERS} \
      --d_layers ${D_LAYERS} \
      --factor ${FACTOR} \
      --enc_in ${ENC_IN} \
      --dec_in ${DEC_IN} \
      --c_out ${C_OUT} \
      --itr ${ITR} \
      --inverse \
      --des "${DES}"

    echo "- Done: ${sym} | pred_len=${PRED_LEN}"
    echo
  done
done

echo "All Autoformer trainings finished."
