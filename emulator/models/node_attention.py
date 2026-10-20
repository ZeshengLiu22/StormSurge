"""Local SDPA selection for station-to-node attention only.

Reuse the validated Auto/Math paths without changing projections or padding masks.
The context restores backend flags before temporal and forecast attention run.
"""

from torch.nn.attention import SDPBackend, sdpa_kernel


NODE_ATTENTION_BACKENDS = ("auto", "math")


def node_attention(readout, query, nodes, padding_mask, backend="auto"):
    """Call the existing MHA; an explicit Math request cannot fall back to Auto."""
    if backend == "auto":
        return readout(query, nodes, nodes, key_padding_mask=padding_mask, need_weights=False)
    if backend != "math":
        raise ValueError(f"Unknown node_attention_backend: {backend!r}.")
    with sdpa_kernel(SDPBackend.MATH):
        return readout(query, nodes, nodes, key_padding_mask=padding_mask, need_weights=False)
