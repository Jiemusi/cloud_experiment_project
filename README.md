# cloud_experiment_project

Course project on performance modeling and analysis of neural network training across GPU environments.

## Summary

This project compares the training performance of three torchvision image classification models across two GPU environments:

- Tesla T4
- Tesla V100-SXM2-16GB

Models:
- MobileNetV3-Small
- ResNet-18
- ResNet-50

Dataset:
- ImageNet-style subset
- 50 classes
- 200 images per class
- 10,000 training images total

Main metrics:
- average batch time
- average data time
- throughput (images/s)

Profiling:
- Nsight Compute roofline analysis on short representative runs

## Repository Contents

- `main.py` — simplified training script
- `models.py` — model selection
- `train_models.sh` — benchmark runs
- `ncu_models.sh` — Nsight Compute profiling runs
- `record_system_info.py` — system info collection
- `run_summary.csv` — benchmark results
- `system_info.csv` — environment information
- `*.ncu-rep` — Nsight Compute reports

## Notes

This project was built using a simplified training pipeline inspired in part by the PyTorch ImageNet example repository, with modifications for a smaller and more controlled benchmark setup. The dataset used in the experiments was derived from ImageNet-1k.

## References

- PyTorch ImageNet example: https://github.com/pytorch/examples/tree/master/imagenet
- ImageNet: http://www.image-net.org/