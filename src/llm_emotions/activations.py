"""Last-token residual streams, including the embedding layer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from llm_emotions.model import module_device


def format_prompt(tokenizer: Any, text: str) -> str:
    """Wrap ``text`` as a user turn with a generation prompt, when a chat template exists."""
    messages = [{"role": "user", "content": text}]
    template = getattr(tokenizer, "chat_template", None)
    apply = getattr(tokenizer, "apply_chat_template", None)
    if template and apply is not None:
        rendered = apply(messages, tokenize=False, add_generation_prompt=True)
        if not isinstance(rendered, str):
            raise TypeError("apply_chat_template(tokenize=False) must return a string.")
        return rendered
    return text


@torch.no_grad()
def collect_hidden_states(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    batch_size: int = 8,
    max_length: int = 512,
) -> np.ndarray:
    """Return float32 residuals of shape ``(n_texts, n_layers + 1, hidden_size)``.

    Index 0 is the embedding stream. Index ``k`` for ``k >= 1`` is the output of
    decoder block ``k - 1``. Prompts are left-padded, so position ``-1`` is the
    last real token: the usual place to read how the model is poised to answer.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if not texts:
        raise ValueError("texts is empty.")

    previous_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    chunks: list[np.ndarray] = []
    try:
        for start in range(0, len(texts), batch_size):
            prompts = [format_prompt(tokenizer, text) for text in texts[start : start + batch_size]]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(module_device(model)) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=True, use_cache=False)
            stacked = torch.stack(
                [hidden[:, -1, :].float().cpu() for hidden in output.hidden_states],
                dim=1,
            )
            chunks.append(stacked.numpy())
            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        tokenizer.padding_side = previous_side
    return np.concatenate(chunks, axis=0)
