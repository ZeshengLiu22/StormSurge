"""One forward per batch for training, validation, testing and inference."""

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from emulator.data.normalization import normalize_inputs
from .metrics import summarize_windows


@dataclass
class EpochResult:
    metrics: dict
    predictions: dict | None = None


def run_epoch(model, loader, device, stats, *, station_feat=None, optimizer=None,
              criterion=None, scaler=None, grad_accum_steps=1, max_grad_norm=0.0, use_amp=False,
              amp_dtype=torch.bfloat16, x_clip=0.0, augmentation=None,
              save_predictions=False, distributed=False):
    training = optimizer is not None
    model.train(training)
    # Evaluation does not need gradient synchronization.
    network = model.module if not training and isinstance(model, DistributedDataParallel) else model
    if training:
        optimizer.zero_grad(set_to_none=True)
    windows, truth, predictions, tags = [], [], [], []
    # Original evaluation: FP32 batch means, weighted/accumulated in FP64.
    evaluation_sums = torch.zeros(3, dtype=torch.float64, device=device) if not training else None
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(loader):
            if save_predictions:
                tags.extend(batch.tag)
            batch = batch.to(device, non_blocking=True)
            normalize_inputs(batch, stats, x_clip, augmentation if training else None)
            target = batch.y.float()
            end_group = (step + 1) % grad_accum_steps == 0 or step + 1 == len(loader)
            synchronization = model.no_sync() if training and not end_group and isinstance(model, DistributedDataParallel) else nullcontext()
            with synchronization:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    output = network(batch, station_feat=station_feat)
                prediction = output.prediction.float() * stats["y_std"] + stats["y_mean"]
                if training:
                    group_start = (step // grad_accum_steps) * grad_accum_steps
                    group_batches = min(grad_accum_steps, len(loader) - group_start)
                    loss = criterion(output, prediction, target) / group_batches
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
            if training and end_group:
                if max_grad_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # Only four scalars per sample are retained, not attention or activations.
            error = prediction.detach() - target
            if not training:
                evaluation_sums[0] += error.square().mean().double() * target.size(0)
                evaluation_sums[1] += error.abs().mean().double() * target.size(0)
                evaluation_sums[2] += target.size(0)
            windows.append(torch.stack((batch.sample_id.double(), target.amax(dim=1).double(),
                                        error.square().mean(dim=1).double(), error.abs().mean(dim=1).double()), dim=1))
            if save_predictions:
                truth.append(target.cpu())
                predictions.append(prediction.detach().cpu())
    records = torch.cat(windows).cpu().numpy() if windows else np.empty((0, 4))
    if distributed:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, records)
        records = np.concatenate(gathered)
    arrays = None
    if save_predictions:
        if distributed:
            raise ValueError("Save predictions in a single-process pass, not a distributed epoch.")
        width = stats["y_mean"].numel()
        arrays = {"y_true": torch.cat(truth).numpy() if truth else np.empty((0, width), np.float32),
                  "y_pred": torch.cat(predictions).numpy() if predictions else np.empty((0, width), np.float32),
                  "tags": np.asarray(tags, dtype=str)}
    # Val All includes sampler padding; Peak reuses the unique predictions.
    metrics = summarize_windows(records, validation=not training)
    if not training:
        if distributed:
            for value in evaluation_sums:
                dist.all_reduce(value)
        square, absolute, count = evaluation_sums.tolist()
        if count:
            metrics.update(rmse_all=float(np.sqrt(square / count)), mae_all=absolute / count)
    return EpochResult(metrics, arrays)
