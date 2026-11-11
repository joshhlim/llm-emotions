"""Shared fixtures. The backend is set before any test imports pyplot."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM


class CharTokenizer:
    """Minimal tokenizer with a chat template. Token ids stay inside a 64-word vocab."""

    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.padding_side = "left"
        self.chat_template = "test"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = messages[0]["content"]
        if add_generation_prompt:
            text += "\n"
        return text

    def _encode(self, text: str, max_length: int) -> list[int]:
        body = [3 + (ord(ch) % (self.vocab_size - 4)) for ch in text[: max_length - 2]]
        return [self.bos_token_id, *body, self.eos_token_id]

    def __call__(self, texts, return_tensors=None, padding=False, truncation=False, max_length=128):
        if isinstance(texts, str):
            texts = [texts]
        limit = max_length or 128
        encoded = [self._encode(text, limit) for text in texts]
        width = max(len(ids) for ids in encoded)
        input_ids: list[list[int]] = []
        attention: list[list[int]] = []
        for ids in encoded:
            pad = width - len(ids)
            if self.padding_side == "left":
                input_ids.append([self.pad_token_id] * pad + ids)
                attention.append([0] * pad + [1] * len(ids))
            else:
                input_ids.append(ids + [self.pad_token_id] * pad)
                attention.append([1] * len(ids) + [0] * pad)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attention}

    def decode(self, ids, skip_special_tokens=True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        chars: list[str] = []
        for token in ids:
            token = int(token)
            special = (self.pad_token_id, self.eos_token_id, self.bos_token_id)
            if skip_special_tokens and token in special:
                continue
            chars.append(chr(32 + (token % 95)))
        return "".join(chars).strip()


def make_tiny_qwen(seed: int = 0, n_layers: int = 2, hidden: int = 32):
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model, CharTokenizer()


@pytest.fixture()
def tiny_bundle():
    return make_tiny_qwen()
