"""Add a direction to the residual stream during a forward pass."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from llm_emotions.activations import format_prompt
from llm_emotions.model import module_device


def decoder_layer_module(model: Any, block_index: int) -> Any:
    """Return decoder block ``block_index`` (0-based).

    Covers Qwen2, Llama, and Gemma-2 (``model.layers``), encoder-decoder
    ``model.decoder.layers``, and GPT-2 (``transformer.h``).
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[block_index]
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        decoder = model.model.decoder
        if hasattr(decoder, "layers"):
            return decoder.layers[block_index]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[block_index]
    raise AttributeError("Could not find decoder blocks on this model.")


def mean_residual_norm(activations: dict[str, dict[str, np.ndarray]], layer: int) -> float:
    """Mean L2 norm of emotion and neutral residuals at ``layer``."""
    chunks = []
    for splits in activations.values():
        chunks.append(splits["emotion"][:, layer])
        chunks.append(splits["neutral"][:, layer])
    stacked = np.concatenate(chunks, axis=0)
    return float(np.linalg.norm(stacked, axis=-1).mean())


@contextmanager
def steer(
    model: Any,
    layer: int,
    direction: np.ndarray,
    strength: float,
) -> Iterator[None]:
    """Add ``strength * direction`` at residual layer ``layer``.

    ``layer`` follows hidden states: 1 is the first decoder block. Layer 0 is
    the embedding stream and has no decoder block to hook. The hook is removed
    when the context exits, including on error.
    """
    if layer < 1:
        raise ValueError(
            "Steering hooks a decoder block. Layer 0 is the embedding stream and "
            "cannot be steered with this hook."
        )
    module = decoder_layer_module(model, layer - 1)
    vector = torch.tensor(
        np.asarray(direction, dtype=np.float32),
        dtype=torch.float32,
        device=module_device(model),
    )

    def hook(mod: Any, inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
        elif torch.is_tensor(output):
            hidden = output
        else:
            raise TypeError(f"Unsupported layer output type: {type(output).__name__}")
        updated = hidden + (strength * vector).to(dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (updated,) + output[1:]
        return updated

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def generate_with_steering(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    layer: int,
    direction: np.ndarray,
    strength: float,
    max_new_tokens: int,
) -> str:
    """Greedy generation with a steering hook installed.

    Pass a direction already scaled by :func:`mean_residual_norm` if ``strength``
    should be a fraction of a typical residual magnitude. ``strength`` 0 leaves
    the residual unchanged up to a no-op add.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1.")
    text = format_prompt(tokenizer, prompt)
    encoded = tokenizer(text, return_tensors="pt")
    encoded = {key: value.to(module_device(model)) for key, value in encoded.items()}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    prompt_length = int(encoded["input_ids"].shape[1])
    with steer(model, layer, direction, strength):
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    continuation = generated[0, prompt_length:]
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()
