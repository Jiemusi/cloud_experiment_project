set -e

NCU=/home/jl7250/NVIDIA-Nsight-Compute-2024.1/ncu

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
GPU_TAG=$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/ /_/g; s/[^a-z0-9_]/_/g')

sudo "$NCU" --set roofline -c 10 -o "ncu_${GPU_TAG}_mobilenet_v3_small" \
  ./venv/bin/python main.py \
  --data train_data \
  --arch mobilenet_v3_small \
  --num-classes 50 \
  -b 32 \
  -j 4 \
  --warmup-steps 10 \
  --max-steps 50 \
  -p 10 \
  --no-summary

sudo "$NCU" --set roofline -c 10 -o "ncu_${GPU_TAG}_resnet18" \
  ./venv/bin/python main.py \
  --data train_data \
  --arch resnet18 \
  --num-classes 50 \
  -b 32 \
  -j 4 \
  --warmup-steps 10 \
  --max-steps 50 \
  -p 10 \
  --no-summary

sudo "$NCU" --set roofline -c 10 -o "ncu_${GPU_TAG}_resnet50" \
  ./venv/bin/python main.py \
  --data train_data \
  --arch resnet50 \
  --num-classes 50 \
  -b 32 \
  -j 4 \
  --warmup-steps 10 \
  --max-steps 50 \
  -p 10 \
  --no-summary
