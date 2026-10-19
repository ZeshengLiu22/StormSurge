"""DataLoader construction utilities.

A single implementation is shared by training and inference to guarantee
consistent multiprocessing and prefetch semantics.
"""

from __future__ import annotations

from torch_geometric.loader import DataLoader


def build_loader(
    dataset,
    sampler,
    batch_size,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
    mp_context,
    shuffle: bool = False,
    generator=None,
):
    """Construct a graph DataLoader with worker options applied only when valid.

    Args:
        dataset: Dataset yielding PyG graphs or compatible batchable objects.
        sampler: Optional sampler controlling traversal order; disables the shuffle
            argument when supplied.
        batch_size: Number of graph samples per batch, including a possibly smaller
            final batch.
        num_workers: Number of loader subprocesses; zero loads samples in the calling
            process.
        pin_memory: Whether to request pinned host memory for returned batches.
        persistent_workers: Whether to keep loader subprocesses alive between
            iterations; ignored with zero workers.
        prefetch_factor: Batches prefetched per worker. None or a nonpositive value
            leaves the DataLoader default unchanged.
        mp_context: Multiprocessing context name such as fork or spawn; None or an empty
            string leaves the default unchanged.
        shuffle: Whether to randomize dataset order when no sampler is supplied.
        generator: Optional torch.Generator controlling loader shuffling and worker base
            seeds independently of the model RNG.

    Returns:
        torch_geometric.loader.DataLoader: Loader with drop_last=False. Worker-only
        options are omitted when num_workers is zero.
    """
    num_workers = int(num_workers)
    shuffle = bool(shuffle) and sampler is None

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        drop_last=False,
        generator=generator,
    )

    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)

        if prefetch_factor is not None:
            pf = int(prefetch_factor)
            if pf > 0:
                kwargs["prefetch_factor"] = pf

        if mp_context is not None:
            context = str(mp_context).strip()
            if context:
                kwargs["multiprocessing_context"] = context

    return DataLoader(dataset, **kwargs)


__all__ = ["build_loader"]
