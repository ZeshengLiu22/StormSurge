"""Native Adam groups using the validated full-module decay exclusion."""

import torch
from torch import nn


WEIGHT_DECAY_POLICIES = ("module", "legacy_global")
WEIGHT_DECAY = 1e-5


def build_adam(model, lr, policy="module", **kwargs):
    """Exclude existing horizon/forecast modules; decay all other trainable parameters.

    Module membership uses parameter identity, including under DDP/DataParallel.
    Models without either exempt module retain ordinary WD=1e-5. The explicitly
    selected legacy_global policy reproduces the old blanket WD=1e-5 behavior.
    Frozen parameters are omitted; every trainable parameter appears exactly once.
    """
    if policy not in WEIGHT_DECAY_POLICIES:
        raise ValueError(f"Unknown weight_decay_policy: {policy!r}.")
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    parameters = [p for p in model.parameters() if p.requires_grad]
    exempt = set()
    if policy == "module":
        for name in ("horizon_embed", "forecast_readout"):
            module = getattr(model, name, None)
            if isinstance(module, nn.Module):
                exempt.update(id(p) for p in module.parameters() if p.requires_grad)
    groups = []
    for weight_decay, selected in (
        (0.0, [p for p in parameters if id(p) in exempt]),
        (WEIGHT_DECAY, [p for p in parameters if id(p) not in exempt]),
    ):
        if selected:
            groups.append(dict(params=selected, weight_decay=weight_decay))
    flattened = [id(p) for group in groups for p in group["params"]]
    if len(flattened) != len(set(flattened)) or set(flattened) != {id(p) for p in parameters}:
        raise RuntimeError("Adam parameter groups contain missing or duplicated parameters.")
    # Each group supplies its own coefficient; there is no blanket decay on top.
    return torch.optim.Adam(groups, lr=lr, weight_decay=0.0, **kwargs)


def format_optimizer_groups(optimizer, policy):
    """Summarize effective decay and tensor/parameter counts once at startup."""
    groups = "; ".join(
        f"group {index}: WD={group['weight_decay']:g}, "
        f"tensors={len(group['params'])}, parameters={sum(p.numel() for p in group['params'])}"
        for index, group in enumerate(optimizer.param_groups, 1)
    )
    return f"Optimizer: Adam; weight_decay_policy={policy}; {groups}"
