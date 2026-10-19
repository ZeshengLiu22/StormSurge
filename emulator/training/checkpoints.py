"""Atomic checkpoint persistence shared by the canonical checkpoint roles."""

import os
from pathlib import Path

import torch


def atomic_save(checkpoint, path):
    """Write a complete checkpoint before atomically replacing its destination.

    The temporary file is created alongside the destination so os.replace
    operates within one filesystem.

    Args:
        checkpoint: Object accepted by torch.save, normally a training-state dictionary.
        path: Destination checkpoint path. Missing parent directories are created.

    Returns:
        None. Replaces the destination after successful serialization and removes the
        process-specific temporary file on success or failure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
