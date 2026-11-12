"""Probe and steer emotion-like representations in small open-weight LLMs.

The measurements are about residual-stream geometry and generated text. They
do not show that a model feels anything.
"""

from llm_emotions.activations import collect_hidden_states, format_prompt
from llm_emotions.caveats import PHASE02_CAVEAT, PROBE_CAVEAT, STEERING_NOTE
from llm_emotions.config import ExperimentConfig, load_config, sanitize_model_id
from llm_emotions.data import (
    EMOTION_ORDER,
    ContrastivePair,
    Scenario,
    load_contrastive_pairs,
    load_scenarios,
    resolve_keyword_free_dir,
    resolve_neutral_prompts_path,
    resolve_phase2_scenarios_path,
    summarize_pairs,
)
from llm_emotions.directions import (
    DirectionFit,
    cosine_matrix,
    direction_matrix,
    fit_directions_and_probes,
    set_seed,
)
from llm_emotions.experiment import configure_logging, run_phase01
from llm_emotions.lexicon import find_emotion_keywords
from llm_emotions.model import load_model_and_tokenizer
from llm_emotions.phase02 import EXPECTED_PAIRS, run_phase02
from llm_emotions.plotting import plot_direction_pca, plot_probe_accuracy, plot_scenario_scores
from llm_emotions.reporting import (
    format_cosine_table,
    format_phase02_summary,
    format_probe_table,
    format_score_table,
)
from llm_emotions.results import save_phase01
from llm_emotions.scoring import score_scenarios
from llm_emotions.steering import generate_with_steering, mean_residual_norm, steer
from llm_emotions.version import __version__

__all__ = [
    "EMOTION_ORDER",
    "EXPECTED_PAIRS",
    "PHASE02_CAVEAT",
    "PROBE_CAVEAT",
    "STEERING_NOTE",
    "ContrastivePair",
    "DirectionFit",
    "ExperimentConfig",
    "Scenario",
    "__version__",
    "collect_hidden_states",
    "configure_logging",
    "cosine_matrix",
    "direction_matrix",
    "find_emotion_keywords",
    "fit_directions_and_probes",
    "format_cosine_table",
    "format_phase02_summary",
    "format_probe_table",
    "format_prompt",
    "format_score_table",
    "generate_with_steering",
    "load_config",
    "load_contrastive_pairs",
    "load_model_and_tokenizer",
    "load_scenarios",
    "mean_residual_norm",
    "plot_direction_pca",
    "plot_probe_accuracy",
    "plot_scenario_scores",
    "resolve_keyword_free_dir",
    "resolve_neutral_prompts_path",
    "resolve_phase2_scenarios_path",
    "run_phase01",
    "run_phase02",
    "sanitize_model_id",
    "save_phase01",
    "score_scenarios",
    "set_seed",
    "steer",
    "summarize_pairs",
]
