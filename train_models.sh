set -e

./venv/bin/python main.py \
  --data train_data \
  --arch mobilenet_v3_small \
  --num-classes 50 \
  -b 32 \
  -j 4 \
  --warmup-steps 20 \
  --max-steps 100 \
  -p 10

./venv/bin/python main.py \
  --data train_data \
  --arch resnet18 \
  --num-classes 50 \
  -b 32 \
  -j 4 \
  --warmup-steps 20 \
  --max-steps 100 \
  -p 10

./venv/bin/python main.py \
  --data train_data \
  --arch resnet50 \
  --num-classes 50 \
  -b 32 \
  -j 4 \
  --warmup-steps 20 \
  --max-steps 100 \
  -p 10