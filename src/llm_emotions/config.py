"""Experiment configuration.

Paths in a config file are resolved relative to the repository root when the
file lives in ``experiments/``. Seeds, model ids, and layer lists are explicit
so a run can be repeated.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_STEERING_PROMPT = (
    "A teammate missed a deadline and delayed the project. Write a short message to them about it."
)

_ALLOWED_STATUSES = {"implemented", "stub"}


def _default_strengths() -> list[float]:
    return [-0.4, 0.0, 0.4, 0.8]


@dataclass
class ExperimentConfig:
    """Everything a phase-01 run needs, apart from the datasets themselves."""

    experiment: str = "phase01_directions"
    status: str = "implemented"
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    seed: int = 0
    layers: list[int] | None = None
    batch_size: int = 8
    max_new_tokens: int = 80
    n_splits: int = 5
    probe_C: float = 0.5
    probe_max_iter: int = 2000
    steering_emotion: str = "anger"
    steering_strengths: list[float] = field(default_factory=_default_strengths)
    steering_prompt: str = DEFAULT_STEERING_PROMPT
    emotions_dir: str | None = None
    keyword_free_dir: str | None = None
    scenarios_path: str | None = None
    neutral_prompts_path: str | None = None
    n_null: int = 400
    output_root: str = "results"
    trust_remote_code: bool = False
    question: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.layers is not None:
            self.layers = [int(layer) for layer in self.layers]
        self.steering_strengths = [float(value) for value in self.steering_strengths]
        self.probe_C = float(self.probe_C)
        self.seed = int(self.seed)
        self.batch_size = int(self.batch_size)
        self.max_new_tokens = int(self.max_new_tokens)
        self.n_splits = int(self.n_splits)
        self.probe_max_iter = int(self.probe_max_iter)
        self.n_null = int(self.n_null)
        self.validate()

    def validate(self) -> None:
        if not self.experiment.strip():
            raise ValueError("experiment must be a non-empty string.")
        if self.status not in _ALLOWED_STATUSES:
            raise ValueError(f"status must be one of {sorted(_ALLOWED_STATUSES)}.")
        if not str(self.model_id).strip():
            raise ValueError("model_id must be a non-empty string.")
        if self.seed < 0:
            raise ValueError("seed must be >= 0.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1.")
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2.")
        if self.probe_C <= 0:
            raise ValueError("probe_C must be positive.")
        if self.probe_max_iter < 1:
            raise ValueError("probe_max_iter must be >= 1.")
        if not self.steering_emotion.strip():
            raise ValueError("steering_emotion must be a non-empty string.")
        if not self.steering_prompt.strip():
            raise ValueError("steering_prompt must be a non-empty string.")
        if not str(self.output_root).strip():
            raise ValueError("output_root must be a non-empty string.")
        if not 1 <= self.n_null <= 5000:
            raise ValueError("n_null must be between 1 and 5000.")
        if self.layers is not None:
            if len(self.layers) == 0:
                raise ValueError("layers is empty. Use null to mean every layer.")
            if any(layer < 0 for layer in self.layers):
                raise ValueError("layer indices must be >= 0 (0 is the embedding stream).")
            if len(set(self.layers)) != len(self.layers):
                raise ValueError(f"layers contains duplicates: {self.layers}")

    def output_dir(self) -> Path:
        return Path(self.output_root) / self.experiment / sanitize_model_id(self.model_id)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def apply_env(self, env: dict[str, str] | None = None) -> ExperimentConfig:
        """Apply the ``EMOTION_*`` overrides used by the notebook and local CPU checks."""
        source = os.environ if env is None else env
        if source.get("EMOTION_MODEL_ID"):
            self.model_id = source["EMOTION_MODEL_ID"]
        if source.get("EMOTION_BATCH_SIZE"):
            self.batch_size = int(source["EMOTION_BATCH_SIZE"])
        if source.get("EMOTION_MAX_NEW_TOKENS"):
            self.max_new_tokens = int(source["EMOTION_MAX_NEW_TOKENS"])
        if source.get("EMOTION_SEED"):
            self.seed = int(source["EMOTION_SEED"])
        if source.get("EMOTION_LAYERS"):
            self.layers = parse_layers(source["EMOTION_LAYERS"])
        self.validate()
        return self


def sanitize_model_id(model_id: str) -> str:
    """Turn ``org/model`` into a single directory name."""
    parts: list[str] = []
    for part in model_id.strip().replace("\\", "/").split("/"):
        if part in ("", ".", ".."):
            continue
        safe = re.sub(r"[^A-Za-z0-9._+-]+", "_", part).strip("._")
        if safe:
            parts.append(safe)
    if not parts:
        raise ValueError(f"Cannot derive an output directory from model id {model_id!r}.")
    return "_".join(parts)


def parse_layers(text: str) -> list[int] | None:
    """Parse ``"all"`` or ``"1,4,12"`` into a layer list. ``None`` means every layer."""
    cleaned = text.strip()
    if cleaned == "" or cleaned.lower() == "all":
        return None
    layers: list[int] = []
    for part in cleaned.split(","):
        item = part.strip()
        if item == "":
            raise ValueError(f"Empty layer id in {text!r}.")
        layers.append(int(item))
    return layers


def config_base_dir(path: Path) -> Path:
    """Directory that relative config paths are resolved against."""
    parent = path.resolve().parent
    if parent.name == "experiments":
        return parent.parent
    return parent


def _resolve_config_path(base: Path, value: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str(base / candidate)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a JSON config. Unknown keys are an error, so typos do not silently drop."""
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")
    unknown = sorted(set(raw) - set(ExperimentConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {unknown}")
    base = config_base_dir(config_path)
    for key in ("emotions_dir", "keyword_free_dir", "scenarios_path", "neutral_prompts_path"):
        if isinstance(raw.get(key), str):
            raw[key] = _resolve_config_path(base, raw[key])
    if isinstance(raw.get("output_root"), str):
        raw["output_root"] = _resolve_config_path(base, raw["output_root"])
    return ExperimentConfig(**raw)
