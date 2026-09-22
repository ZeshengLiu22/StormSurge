"""One forward per batch for training, validation, testing and inference."""

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from emulator.data.normalization import normalize_inputs
from .metrics import evaluate_metrics


@dataclass
class EpochResult:
    metrics: dict
    predictions: dict | None = None


def run_epoch(model, loader, device, stats, *, station_feat=None, optimizer=None,
              criterion=None, scaler=None, grad_accum_steps=1, max_grad_norm=0.0, use_amp=False,
              amp_dtype=torch.bfloat16, x_clip=0.0, augmentation=None,
              save_predictions=False, distributed=False, save_dual_diagnostics=False, tau_physical=None):
    if tau_physical is None or not np.isfinite(tau_physical):
        raise ValueError("Evaluation requires the saved finite TRAIN tau_physical.")
    training = optimizer is not None
    if save_dual_diagnostics and (training or distributed or not save_predictions):
        raise ValueError("Dual diagnostics require a single-process prediction export in evaluation mode.")
    model.train(training)
    # Evaluation does not need gradient synchronization.
    network = model.module if not training and isinstance(model, DistributedDataParallel) else model
    if training:
        optimizer.zero_grad(set_to_none=True)
    truth, predictions, ids, timestamps, tags = [], [], [], [], []
    dual_arrays = {name: [] for name in ("gate_probability", "gate_logits", "body_phys", "excess_phys")} if save_dual_diagnostics else None
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(loader):
            if save_predictions:
                tags.extend(batch.tag)
            ids.append(batch.sample_id.detach().cpu().reshape(-1))
            timestamps.append(batch.target_timestamps.detach().cpu())
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
            truth.append(target.detach().cpu())
            predictions.append(prediction.detach().cpu())
            if save_dual_diagnostics:
                if output.body is None or output.excess is None or output.gate_probability is None:
                    raise ValueError("Dual diagnostics require a dual-head model.")
                dual_arrays["gate_probability"].append(output.gate_probability.float().reshape(-1).cpu())
                logits = (output.gate_logits.float().reshape(-1) if output.gate_logits is not None
                          else torch.logit(output.gate_probability.float().reshape(-1)))
                dual_arrays["gate_logits"].append(logits.cpu())
                dual_arrays["body_phys"].append((output.body.float() * stats["y_std"] + stats["y_mean"]).cpu())
                dual_arrays["excess_phys"].append((output.excess.float() * stats["y_std"]).cpu())
                if output.severity_phys is not None:
                    dual_arrays.setdefault("severity_phys", []).append(output.severity_phys.float().reshape(-1).cpu())
                    dual_arrays.setdefault("excess_shape", []).append(output.excess_shape.float().cpu())
    width = stats["y_mean"].numel()
    records = {
        "sample_id": torch.cat(ids).numpy() if ids else np.empty(0, np.int64),
        "y_true": torch.cat(truth).numpy() if truth else np.empty((0, width), np.float32),
        "y_pred": torch.cat(predictions).numpy() if predictions else np.empty((0, width), np.float32),
        "target_timestamps": torch.cat(timestamps).numpy() if timestamps else np.empty((0, width), np.int64),
    }
    if distributed:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, records)
        records = {key: np.concatenate([part[key] for part in gathered]) for key in records}
        # DistributedSampler pads the sample stream. Remove only sampler copies;
        # dataset timestamp uniqueness is checked separately, before the loader.
        _, first = np.unique(records["sample_id"], return_index=True)
        records = {key: value[first] for key, value in records.items()}
    metrics = evaluate_metrics(records["y_pred"], records["y_true"], tau_physical,
                              target_timestamps=records["target_timestamps"] if save_predictions else None)
    arrays = None
    if save_predictions:
        if distributed:
            raise ValueError("Save predictions in a single-process pass, not a distributed epoch.")
        arrays = dict(records, tags=np.asarray(tags, dtype=str))
        if save_dual_diagnostics:
            arrays.update({name: torch.cat(values).numpy() if values else
                           np.empty((0,) if name in ("gate_probability", "gate_logits", "severity_phys") else (0, width), np.float32)
                           for name, values in dual_arrays.items()})
            reconstruction = arrays["body_phys"] + arrays["gate_probability"][:, None] * arrays["excess_phys"]
            if not np.allclose(reconstruction, arrays["y_pred"], rtol=2e-5, atol=2e-6):
                raise ValueError("Dual branch reconstruction does not match the physical prediction.")
    return EpochResult(metrics, arrays)
