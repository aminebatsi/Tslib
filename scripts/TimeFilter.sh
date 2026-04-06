#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0


model_name="TimeFilter"


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
    MODEL_ID="${sym}_${SEQ_LEN}_${PRED_LEN}_${model_name}_${FEATURES}"
    DES="${sym}_${model_name}_${FEATURES}_sl${SEQ_LEN}_pl${PRED_LEN}"

    echo "============================================================"
    echo "Training ${model_name} | ${DATA_PATH} | pred_len=${PRED_LEN}"
    echo "MODEL_ID=${MODEL_ID}"
    echo "DES=${DES}"
    echo "============================================================"

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "${ROOT_PATH}/" \
      --data_path "${DATA_PATH}" \
      --data "${DATA_NAME}" \
      --model "${model_name}" \
      --model_id "${MODEL_ID}" \
      --target "${TARGET}" \
      --features "${FEATURES}" \
      --seq_len ${SEQ_LEN} \
      --label_len ${LABEL_LEN} \
      --pred_len ${PRED_LEN} \
      --enc_in ${ENC_IN} \
      --dec_in ${DEC_IN} \
      --c_out ${C_OUT} \
      \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --patch_len 4 \
      --pos 0 \
      --dropout 0.3 \
      \
      --d_model 128 \
      --d_ff 256 \
      --learning_rate 1e-4 \
      --batch_size 32 \
      --train_epochs 10 \
      --inverse \
      --itr 1 \
      --des "${DES}"

    echo "- Done: ${sym} | pred_len=${PRED_LEN}"
    echo
  done
done

echo "All TimeFilter trainings finished."
