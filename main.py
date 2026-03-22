import argparse
import csv
import os
import socket
import subprocess
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import get_model


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="train_data",
                        help="Root folder containing class subfolders for training")
    parser.add_argument("--arch", type=str, default="resnet18",
                        help="Model name, e.g. resnet18/resnet34/resnet50")
    parser.add_argument("--num-classes", type=int, default=50,
                        help="Number of output classes")
    parser.add_argument("-b", "--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("-j", "--workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Maximum number of training steps")
    parser.add_argument("-p", "--print-freq", type=int, default=10,
                        help="Print frequency")
    parser.add_argument("--summary-file", type=str, default="run_summary.csv",
                        help="CSV file to append run summary")
    parser.add_argument("--dummy", action="store_true",
                        help="Use FakeData instead of real dataset")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--no-summary", action="store_true",
                        help="Do not append this run to run_summary.csv")
    parser.add_argument("--warmup-steps", type=int, default=20,
                    help="Number of warmup steps not included in averages")

    return parser.parse_args()


def get_gpu_name():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True
        ).strip().splitlines()
        return result[0] if result else "None"
    except Exception:
        return "None"


def append_run_summary(summary_file, hostname, gpu_name, model_name,
                       avg_batch_time, avg_data_time, throughput):
    row = {
        "hostname": hostname,
        "gpu_name": gpu_name,
        "model": model_name,
        "avg_batch_time": round(avg_batch_time, 6),
        "avg_data_time": round(avg_data_time, 6),
        "throughput_img_per_s": round(throughput, 3),
    }

    file_exists = os.path.exists(summary_file)

    with open(summary_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    if args.dummy:
        dataset = datasets.FakeData(
            size=max(args.batch_size * args.max_steps, 1000),
            image_size=(3, args.image_size, args.image_size),
            num_classes=args.num_classes,
            transform=transform,
        )
    else:
        dataset = datasets.ImageFolder(args.data, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    return loader


def train_one_run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hostname = socket.gethostname()
    gpu_name = get_gpu_name()

    print(f"Hostname: {hostname}")
    print(f"GPU: {gpu_name}")
    print(f"Using device: {device}")
    print(f"Model: {args.arch}")

    train_loader = build_dataloader(args)

    model = get_model(args.arch, args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    model.train()

    data_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()

    measured_steps = 0
    total_steps = args.warmup_steps + args.max_steps

    end = time.time()

    for step, (images, targets) in enumerate(train_loader, start=1):
        data_time = time.time() - end

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        batch_time = time.time() - end

        if step <= args.warmup_steps:
            if step % args.print_freq == 0 or step == 1:
                print(
                    f"Warmup Step [{step}/{args.warmup_steps}]  "
                    f"Data {data_time:.4f}  "
                    f"Batch {batch_time:.4f}"
                )
        else:
            measured_steps += 1
            data_time_meter.update(data_time)
            batch_time_meter.update(batch_time)

            if measured_steps % args.print_freq == 0 or measured_steps == 1:
                print(
                    f"Measured Step [{measured_steps}/{args.max_steps}]  "
                    f"Data {data_time_meter.val:.4f} ({data_time_meter.avg:.4f})  "
                    f"Batch {batch_time_meter.val:.4f} ({batch_time_meter.avg:.4f})"
                )

        end = time.time()

        if step >= total_steps:
            break

    throughput = args.batch_size / batch_time_meter.avg if batch_time_meter.avg > 0 else 0.0

    print("\nRun summary:")
    print(f"  warmup_steps        = {args.warmup_steps}")
    print(f"  measured_steps      = {measured_steps}")
    print(f"  avg_batch_time      = {batch_time_meter.avg:.6f}")
    print(f"  avg_data_time       = {data_time_meter.avg:.6f}")
    print(f"  throughput_img_per_s= {throughput:.3f}")

    if not args.no_summary:
        append_run_summary(
            summary_file=args.summary_file,
            hostname=hostname,
            gpu_name=gpu_name,
            model_name=args.arch,
            avg_batch_time=batch_time_meter.avg,
            avg_data_time=data_time_meter.avg,
            throughput=throughput,
        )
        print(f"Saved summary to {args.summary_file}")

def main():
    args = parse_args()
    train_one_run(args)


if __name__ == "__main__":
    main()