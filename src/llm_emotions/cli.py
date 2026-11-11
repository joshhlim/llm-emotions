"""Command line runner. ``llm-emotions run --config experiments/phase01_directions.json``."""

from __future__ import annotations

import argparse
import sys

from llm_emotions.config import ExperimentConfig, load_config, parse_layers
from llm_emotions.experiment import configure_logging, run_phase01
from llm_emotions.phases import (
    PHASES,
    PhaseNotImplemented,
    run_phase02_scenarios,
    run_phase03_performance,
    run_phase04_steering,
)

_STUB_RUNNERS = {
    "phase02_scenarios": run_phase02_scenarios,
    "phase03_performance": run_phase03_performance,
    "phase04_steering": run_phase04_steering,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-emotions",
        description=(
            "Run emotion-representation experiments on small open-weight language models."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run the experiment named by a JSON config.")
    run.add_argument("--config", required=True, help="Path to an experiment JSON config.")
    run.add_argument("--model-id", default=None, help="Override config.model_id.")
    run.add_argument("--seed", type=int, default=None, help="Override config.seed.")
    run.add_argument(
        "--layers",
        default=None,
        help="Comma-separated hidden-state indices (0 = embeddings), or 'all'.",
    )
    run.add_argument("--batch-size", type=int, default=None)
    run.add_argument("--max-new-tokens", type=int, default=None)
    run.add_argument("--output-root", default=None, help="Override the results root directory.")
    run.add_argument(
        "--skip-steering",
        action="store_true",
        help="Fit directions and score scenarios without generating steered samples.",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the config and print the output directory. Do not load a model.",
    )
    return parser


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.model_id is not None:
        config.model_id = args.model_id
    if args.seed is not None:
        config.seed = args.seed
    if args.layers is not None:
        config.layers = parse_layers(args.layers)
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
    if args.output_root is not None:
        config.output_root = args.output_root
    config.validate()


def _dispatch(config: ExperimentConfig, *, do_steering: bool) -> None:
    if config.status != "implemented":
        question = config.question or PHASES.get(config.experiment, {}).get("question", "")
        raise PhaseNotImplemented(
            f"{config.experiment} is not implemented.\n"
            f"Question: {question}\n"
            "No results were written."
        )
    if config.experiment == "phase01_directions":
        run_phase01(config, do_steering=do_steering)
        return
    runner = _STUB_RUNNERS.get(config.experiment)
    if runner is not None:
        runner(config)
        return
    raise PhaseNotImplemented(
        f"{config.experiment} is not a known phase.\nNo results were written."
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging()
    if args.command != "run":
        parser.error(f"Unknown command {args.command!r}.")
    config = load_config(args.config)
    _apply_overrides(config, args)
    if args.dry_run:
        print(f"experiment: {config.experiment}")
        print(f"status: {config.status}")
        print(f"model_id: {config.model_id}")
        print(f"seed: {config.seed}")
        print(f"layers: {config.layers if config.layers is not None else 'all'}")
        print(f"batch_size: {config.batch_size}")
        print(f"max_new_tokens: {config.max_new_tokens}")
        print(f"output_dir: {config.output_dir()}")
        return 0
    try:
        _dispatch(config, do_steering=not args.skip_steering)
    except PhaseNotImplemented as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0
