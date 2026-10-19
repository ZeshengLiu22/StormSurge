"""Shared normalization; assigning new tensors keeps stored graphs unmodified."""

import torch


def normalize_inputs(batch, stats, x_clip=0.0, augmentation=None):
    """Normalize current and historical features on a batch in place.

    The same random feature transform is shared by every node and time step in
    the batch. Clipping is applied after normalization and again after augmentation.
    Targets are left in physical units.

    Args:
        batch: PyG batch with x [nodes, features] and x_hist [nodes, steps, features];
            both attributes are replaced with newly computed tensors.
        stats: Training-fitted x_center and x_scale tensors, broadcastable across the
            feature dimension and on a compatible device.
        x_clip: Positive symmetric bound for normalized features. Zero or a negative
            value disables clipping.
        augmentation: Optional (probability, scale_range, bias_std) tuple. A selected
            batch receives one random scale per feature from U(1-scale_range,
            1+scale_range) and one Gaussian bias per feature with the given standard
            deviation.

    Returns:
        None. Reassigns batch.x and batch.x_hist without changing source graph tensors.
    """
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
