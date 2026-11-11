"""Load a causal LM and describe the device it landed on."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    model_id: str
    device: str
    dtype: str
    n_layers: int
    hidden_size: int


def resolve_runtime() -> tuple[str, torch.dtype]:
    """Pick a device and dtype. CPU stays in float32; GPU uses float16."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def module_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device) and device.type != "meta":
        return device
    return next(model.parameters()).device


def load_model_and_tokenizer(
    model_id: str,
    *,
    trust_remote_code: bool = False,
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> LoadedModel:
    """Load ``model_id`` with a left-padding tokenizer.

    Left padding makes the last sequence position the last real token, which is
    where activation extraction reads the residual stream. ``trust_remote_code``
    stays off unless a caller opts in. Qwen2.5 is loaded by the installed
    transformers version and does not need it.
    """
    auto_device, auto_dtype = resolve_runtime()
    device = device or auto_device
    dtype = auto_dtype if dtype is None else dtype

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs: dict[str, Any] = {
        "device_map": device,
        "trust_remote_code": trust_remote_code,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **load_kwargs)
    model.eval()

    n_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    dtype_name = str(next(model.parameters()).dtype).replace("torch.", "")
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        device=str(module_device(model)),
        dtype=dtype_name,
        n_layers=n_layers,
        hidden_size=hidden_size,
    )
