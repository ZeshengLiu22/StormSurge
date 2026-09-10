"""Argument parsers shared by training and inference."""

import argparse


def parse_bool_int(value):
    value = str(value).strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return 1
    if value in ("0", "false", "no", "n", "off"):
        return 0
    raise argparse.ArgumentTypeError("expected one of: 0/1, true/false, yes/no, on/off")


def head_type_name(value):
    return value.strip().lower()


def temporal_block_name(value):
    names = {"mlp": "MLP", "lstm": "LSTM", "gru": "GRU", "transformer": "Transformer", "attn": "Transformer"}
    try:
        return names[value.strip().lower()]
    except KeyError:
        raise argparse.ArgumentTypeError("expected MLP, LSTM, GRU or Transformer") from None
