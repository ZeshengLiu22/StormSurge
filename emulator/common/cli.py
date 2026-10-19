"""Argument parsers shared by training and inference."""

import argparse


def parse_bool_int(value):
    """Parse a command-line boolean into an integer flag.

    Args:
        value: Value converted to a case-insensitive string. Accepted true forms are 1,
            true, yes, y, and on; false forms are 0, false, no, n, and off.

    Returns:
        int: 1 for a true form or 0 for a false form.

    Raises:
        argparse.ArgumentTypeError: The value is not a recognized boolean form.
    """
    value = str(value).strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return 1
    if value in ("0", "false", "no", "n", "off"):
        return 0
    raise argparse.ArgumentTypeError("expected one of: 0/1, true/false, yes/no, on/off")


def temporal_block_name(value):
    """Resolve a temporal-block name or alias to its canonical spelling.

    Args:
        value: Case-insensitive string naming MLP, LSTM, GRU, Transformer, or the
            Transformer alias attn.

    Returns:
        str: The canonical model block name.

    Raises:
        argparse.ArgumentTypeError: The supplied name is unsupported.
    """
    names = {
        "mlp": "MLP",
        "lstm": "LSTM",
        "gru": "GRU",
        "transformer": "Transformer",
        "attn": "Transformer",
    }
    try:
        return names[value.strip().lower()]
    except KeyError:
        raise argparse.ArgumentTypeError("expected MLP, LSTM, GRU or Transformer") from None
