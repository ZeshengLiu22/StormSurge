"""Read-only counts of registered model parameters, independent of architecture."""

from torch import nn


def _counts(module):
    """Count registered scalar parameters in a module.

    Args:
        module: PyTorch module whose recursive parameters are counted.

    Returns:
        dict: Integer total, trainable, and non_trainable counts. Shared Parameter
        objects are counted once within the module; buffers are excluded.
    """
    parameters = tuple(module.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return dict(total=total, trainable=trainable, non_trainable=total - trainable)


def count_model_parameters(model):
    """Count a model and separate its forecast head from the remaining backbone.

    Counts describe one logical model replica and require no forward pass.
    Shared parameters are unique within each individual count; parameters shared
    with the head are excluded from the backbone count.

    Args:
        model: PyTorch model exposing a head module, optionally wrapped in DataParallel
            or DistributedDataParallel.

    Returns:
        dict: Model total/trainable/non_trainable counts, nested head and backbone
        counts, and head_branches counts for each immediate head child.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    total = _counts(model)
    head = _counts(model.head)
    return dict(**total, head=head,
                backbone={key: total[key] - head[key] for key in total},
                head_branches={name: _counts(branch) for name, branch in model.head.named_children()})


def format_parameter_counts(counts):
    """Format a compact parameter-count summary for console logs.

    Args:
        counts: Mapping returned by count_model_parameters, including nested head
            totals.

    Returns:
        str: Total, trainable, nontrainable, and head counts with thousands separators.
    """
    return (f"[Parameters] total={counts['total']:,} trainable={counts['trainable']:,} "
            f"non_trainable={counts['non_trainable']:,} head={counts['head']['total']:,}")
