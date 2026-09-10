"""Shared normalization; assigning new tensors keeps stored graphs unmodified."""

import torch


def normalize_inputs(batch, stats, x_clip=0.0, augmentation=None):
    batch.x = (batch.x.float() - stats["x_center"]) / stats["x_scale"]
    batch.x_hist = (batch.x_hist.float() - stats["x_center"]) / stats["x_scale"]
    if x_clip > 0:
        batch.x = batch.x.clamp(-x_clip, x_clip)
        batch.x_hist = batch.x_hist.clamp(-x_clip, x_clip)
    if augmentation is not None:
        probability, scale_range, bias_std = augmentation
        if probability >= 1.0 or torch.rand((), device=batch.x.device) < probability:
            shape = batch.x.size(-1)
            scale = 1 + (2 * torch.rand(shape, device=batch.x.device) - 1) * scale_range
            bias = torch.randn(shape, device=batch.x.device) * bias_std
            batch.x = batch.x * scale + bias
            batch.x_hist = batch.x_hist * scale + bias
            if x_clip > 0:
                batch.x = batch.x.clamp(-x_clip, x_clip)
                batch.x_hist = batch.x_hist.clamp(-x_clip, x_clip)
