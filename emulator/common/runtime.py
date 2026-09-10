"""Shared execution settings for fresh training and inference."""

import os
from datetime import datetime
import random

import numpy as np
import torch


def configure_runtime(seed=42, threads=1, deterministic=False, tf32=True):
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


def log_message(message):
    print(f'{datetime.now().strftime("[%Y-%m-%d|%H:%M:%S]")} {message}', flush=True)
