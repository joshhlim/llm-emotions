"""Greedy generation and last-token residuals.

Generation is greedy so a seed fixes the reply. The residual at a token depends
only on the tokens up to that position, so the prompt residual can be read from
the full prompt-plus-reply forward at the last prompt index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from llm_emotions.activations import format_prompt
from llm_emotions.model import module_device


@dataclass(frozen=True)
class GeneratedReply:
    text: str
    prompt_token_ids: list[int]
    full_token_ids: list[int]
    empty: bool


def _pad_id(tokenizer) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise ValueError("The tokenizer needs a pad token or an eos token.")
    return int(pad_id)


def _strip_leading(token_ids: list[int], pad_id: int) -> list[int]:
    start = 0
    while start < len(token_ids) and token_ids[start] == pad_id:
        start += 1
    return token_ids[start:]


@torch.no_grad()
def generate_replies(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    batch_size: int,
) -> list[GeneratedReply]:
    """Greedy continuations. The reply stops at the first eos token.

    ``prompt_token_ids`` is the unpadded prompt the model was given.
    ``full_token_ids`` is that prompt plus the reply tokens, without the eos
    token and without padding. An immediate eos leaves the reply empty and the
    full sequence equal to the prompt.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if not prompts:
        raise ValueError("prompts is empty.")
    pad_id = _pad_id(tokenizer)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_token = None if eos_id is None else int(eos_id)
    previous_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    replies: list[GeneratedReply] = []
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            rendered = [format_prompt(tokenizer, text) for text in chunk]
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(module_device(model)) for key, value in encoded.items()}
            prompt_width = int(encoded["input_ids"].shape[1])
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
            sequences = generated.sequences if hasattr(generated, "sequences") else generated
            for row in sequences:
                ids = row.tolist()
                prompt_ids = _strip_leading(ids[:prompt_width], pad_id)
                kept: list[int] = []
                for token in ids[prompt_width:]:
                    if eos_token is not None and int(token) == eos_token:
                        break
                    if int(token) == pad_id and pad_id != eos_token:
                        break
                    kept.append(int(token))
                text = tokenizer.decode(kept, skip_special_tokens=True).strip() if kept else ""
                replies.append(
                    GeneratedReply(
                        text=text,
                        prompt_token_ids=prompt_ids,
                        full_token_ids=[*prompt_ids, *kept],
                        empty=text == "",
                    )
                )
            del generated
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        tokenizer.padding_side = previous_side
    if len(replies) != len(prompts):
        raise RuntimeError("Generation returned a different number of replies than prompts.")
    return replies


@torch.no_grad()
def residuals_at(
    model,
    sequences: list[list[int]],
    positions: list[int],
    layer: int,
    *,
    batch_size: int,
    pad_id: int,
) -> np.ndarray:
    """Hidden state at one position of each sequence.

    Sequences are left-padded, so the index is ``pad + position`` and position
    counts from the first real token. ``layer`` follows the project's hidden
    states: 0 is the embedding stream.
    """
    if layer < 0:
        raise ValueError("layer must be >= 0.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if len(sequences) != len(positions):
        raise ValueError("sequences and positions must have the same length.")
    if not sequences:
        raise ValueError("sequences is empty.")
    device = module_device(model)
    chosen: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        chunk_positions = positions[start : start + batch_size]
        width = max(len(seq) for seq in chunk)
        if width < 1:
            raise ValueError("Cannot read a residual from an empty sequence.")
        rows: list[list[int]] = []
        mask: list[list[int]] = []
        indexes: list[int] = []
        for seq, position in zip(chunk, chunk_positions, strict=True):
            if not seq:
                raise ValueError("Cannot read a residual from an empty sequence.")
            if position < 0 or position >= len(seq):
                raise ValueError(f"Position {position} is outside a sequence of length {len(seq)}.")
            pad = width - len(seq)
            rows.append([pad_id] * pad + list(seq))
            mask.append([0] * pad + [1] * len(seq))
            indexes.append(pad + position)
        input_ids = torch.tensor(rows, dtype=torch.long, device=device)
        attention = torch.tensor(mask, dtype=torch.long, device=device)
        output = model(
            input_ids=input_ids,
            attention_mask=attention,
            output_hidden_states=True,
            use_cache=False,
        )
        n_streams = len(output.hidden_states)
        if layer >= n_streams:
            raise ValueError(f"Layer {layer} is outside the {n_streams} hidden streams.")
        hidden = output.hidden_states[layer].float().cpu().numpy()
        for row_index, index in enumerate(indexes):
            chosen.append(hidden[row_index, index])
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.stack(chosen, axis=0)
