import csv
import os
import platform
import socket
import subprocess
from datetime import datetime

import torch


def get_cpu_model():
    with open("/proc/cpuinfo", "r") as f:
        for line in f:
            if "model name" in line:
                return line.split(":", 1)[1].strip()


def get_total_ram_gb():
    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return round(kb / 1024 / 1024, 2)


def get_gpu_name():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True
        ).strip().splitlines()
        return result[0] if result else "None"
    except Exception:
        return "None"


row = {
    "hostname": socket.gethostname(),
    "os": platform.platform(),
    "python_version": platform.python_version(),
    "pytorch_version": torch.__version__,
    "gpu_name": get_gpu_name(),
    "cpu_model": get_cpu_model(),
    "total_ram_gb": get_total_ram_gb(),
}

out_file = "system_info.csv"
file_exists = os.path.exists(out_file)

with open(out_file, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(row)

print(f"Saved info to {out_file}")