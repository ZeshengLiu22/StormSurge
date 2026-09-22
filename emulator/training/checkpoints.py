"""Atomic checkpoint persistence shared by the two canonical checkpoint roles."""

import os
from pathlib import Path

import torch


def atomic_save(checkpoint, path):
    """Publish a complete checkpoint before replacing an existing role artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
