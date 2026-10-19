"""Shared execution settings for fresh training and inference."""

import os
from datetime import datetime
import random

import numpy as np
import torch


def configure_runtime(seed=42, threads=1, deterministic=False, tf32=True):
    """Configure process-wide random seeds and PyTorch execution settings.

    cuDNN benchmarking is disabled in both execution modes.

    Args:
        seed: Integer seed for Python, NumPy, and PyTorch random generators.
        threads: Positive number of PyTorch intra-operation CPU threads.
        deterministic: Whether to require deterministic PyTorch algorithms and set the
            cuBLAS workspace configuration.
        tf32: Whether CUDA matrix multiplication and cuDNN may use TensorFloat-32.

    Returns:
        None. Updates random-generator state, thread settings, backend flags, and, when
        requested, the cuBLAS environment variable.
    """
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
    """Print a message with a local-time timestamp and flush standard output.

    Args:
        message: Object formatted into the log line after the timestamp.

    Returns:
        None.
    """
    print(f'{datetime.now().strftime("[%Y-%m-%d|%H:%M:%S]")} {message}', flush=True)
