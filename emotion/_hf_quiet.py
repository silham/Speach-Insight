"""
Helper to quiet expected `transformers` checkpoint-loading noise.

When a *base* encoder (e.g. `Wav2Vec2Model`, `BertModel`) is loaded from a
checkpoint that also carries task heads, transformers prints a LOAD REPORT
listing those heads as UNEXPECTED. That is correct and intentional here — we
only consume hidden states, so the heads are meant to be dropped.

Use `quiet_hf_load()` around those specific `from_pretrained` calls only, so
genuine warnings elsewhere still surface.
"""

from contextlib import contextmanager


@contextmanager
def quiet_hf_load():
    """Temporarily raise the transformers log level to ERROR."""
    from transformers import logging as hf_logging

    previous = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        yield
    finally:
        hf_logging.set_verbosity(previous)
