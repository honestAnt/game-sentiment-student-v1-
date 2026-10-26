#!/bin/bash
# v10c train + gold_v7 eval (run outside Cursor sandbox)
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTORCH_MPS_HIGH_WATERMARK_RATIO
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
rm -rf models/student_v10c
: > logs/train_v10c.log
echo "[$(date '+%H:%M:%S')] start train v10c" | tee -a logs/train_v10c.log
.venv/bin/python -u train_student.py \
  --train_file data/labeled/v10c_train_mixed.jsonl \
  --init_from models/student_v8 \
  --output_dir models/student_v10c \
  --prefix_granularity platform_ct \
  --learning_rate 5e-6 --num_train_epochs 2 \
  2>&1 | tee -a logs/train_v10c.log
echo "[$(date '+%H:%M:%S')] train EXIT=$?" | tee -a logs/train_v10c.log
if [[ -f models/student_v10c/model.safetensors ]]; then
  .venv/bin/python evaluate_gold_v7_gates.py \
    --model_dir models/student_v10c --tag v10c \
    --baseline data/validation/gold_eval_v7_v8_baseline.json \
    --recipe "v9底料+v10c中性校准219 5e-6x2ep" \
    2>&1 | tee logs/eval_v10c.log
  echo "[$(date '+%H:%M:%S')] eval done" | tee -a logs/eval_v10c.log
else
  echo "ERROR: model missing, skip eval" | tee -a logs/train_v10c.log
  exit 1
fi
