#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

model_name="VAMPM"

ROOT_PATH="./datasets/crypto"
DATA_NAME="custom"
TARGET="return"

FEATURES="MS"
ENC_IN=31
DEC_IN=31
C_OUT=31

SEQ_LEN=64
LABEL_LEN=32

PRED_LENS=(1)

SYMBOLS=("BTC-USD")

INTERVAL_TAG="1D"

echo "Start training loops for ${model_name}"

for sym in "${SYMBOLS[@]}"; do
  DATA_PATH="final_dataset_autoencoder.csv"


  for PRED_LEN in "${PRED_LENS[@]}"; do
    MODEL_ID="${sym}_${INTERVAL_TAG}_${SEQ_LEN}_${PRED_LEN}_${model_name}_${FEATURES}"
    DES="${sym}-${INTERVAL_TAG}_${model_name}_${FEATURES}_sl${SEQ_LEN}_pl${PRED_LEN}"

    echo "============================================================"
    echo "Training ${model_name} on ${DATA_PATH} | pred_len=${PRED_LEN}"
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
      --d_model 128 \
      --d_ff 256 \
      --e_layers 2 \
      --d_layers 1 \
      --dropout 0.1 \
      --patch_lens 8 16 32 \
      --patch_stride_ratio 0.5 \
      --se_reduction 4 \
      --d_state 64 \
      --n_regimes 4 \
      --n_quantiles 3 \
      --quantiles 0.1 0.5 0.9 \
      --lambda_return 1.0 \
      --lambda_direction 0.3 \
      --lambda_vol 0.2 \
      --lambda_quantile 0.3 \
      --lambda_regime 0.01 \
      --lambda_jump 0.1 \
      --target_index 0 \
      --use_revin \
      --return_aux_outputs \
      --train_ratio 0.7 \
      --val_ratio 0.1 \
      --test_ratio 0.2 \
      --inverse \
      --itr 1 \
      --des "${DES}"

    echo "Done: ${sym} | pred_len=${PRED_LEN}"
    echo
  done
done

echo "All trainings finished."