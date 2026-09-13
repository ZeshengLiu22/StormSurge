"""Read-only counts of registered model parameters, independent of architecture."""

from torch import nn


def _counts(module):
    # Module.parameters() deduplicates shared Parameter objects and excludes buffers.
    parameters = tuple(module.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return dict(total=total, trainable=trainable, non_trainable=total - trainable)


def count_model_parameters(model):
    """Count one logical model replica without a forward pass or a reference model.

    All production models expose their forecast module as ``head``. Backbone
    counts exclude parameters belonging to that head. Branch counts describe
    each immediate head child; shared parameters are unique within each count.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    total = _counts(model)
    head = _counts(model.head)
    return dict(**total, head=head,
                backbone={key: total[key] - head[key] for key in total},
                head_branches={name: _counts(branch) for name, branch in model.head.named_children()})


def format_parameter_counts(counts):
    return (f"[Parameters] total={counts['total']:,} trainable={counts['trainable']:,} "
            f"non_trainable={counts['non_trainable']:,} head={counts['head']['total']:,}")
