from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_emotions.cli import main
from llm_emotions.config import ExperimentConfig, load_config, sanitize_model_id
from llm_emotions.phases import (
    PHASES,
    PhaseNotImplemented,
    run_phase02_scenarios,
    run_phase03_performance,
    run_phase04_steering,
)


def test_phase_configs_match_the_catalogue():
    for name, phase in PHASES.items():
        config = load_config(f"experiments/{name}.json")
        assert config.experiment == name
        assert config.status == phase["status"]
        assert config.question == phase["question"]
        assert config.seed == 0


def test_phase01_defaults_match_the_dataclass():
    config = load_config("experiments/phase01_directions.json")
    default = ExperimentConfig()
    assert config.model_id == default.model_id
    assert config.seed == default.seed
    assert config.batch_size == default.batch_size
    assert config.max_new_tokens == default.max_new_tokens
    assert config.n_splits == default.n_splits
    assert config.steering_strengths == default.steering_strengths
    assert config.steering_prompt == default.steering_prompt
    assert config.layers is None
    assert config.output_dir().name == "Qwen_Qwen2.5-1.5B-Instruct"


def test_sanitize_model_id_and_bad_layers():
    assert sanitize_model_id("Qwen/Qwen2.5-0.5B-Instruct") == "Qwen_Qwen2.5-0.5B-Instruct"
    with pytest.raises(ValueError, match="duplicates"):
        ExperimentConfig(layers=[1, 1])


def test_unknown_config_key(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"experiment": "phase01_directions", "mystery": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


def test_dry_run_prints_the_output_dir_and_overrides(capsys):
    code = main(
        [
            "run",
            "--config",
            "experiments/phase01_directions.json",
            "--model-id",
            "org/demo",
            "--seed",
            "3",
            "--layers",
            "1,4",
            "--dry-run",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "model_id: org/demo" in output
    assert "seed: 3" in output
    assert "layers: [1, 4]" in output
    assert "org_demo" in output
    assert "status: implemented" in output


def test_phase03_dry_run_accepts_quick_and_phase01_rejects_it(capsys):
    code = main(
        [
            "run",
            "--config",
            "experiments/phase03_performance.json",
            "--quick",
            "--dry-run",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "status: implemented" in output
    assert "quick: True" in output
    assert "phase03_performance" in output
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--config",
                "experiments/phase01_directions.json",
                "--quick",
                "--dry-run",
            ]
        )
    assert exc.value.code == 2


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize(
    "runner",
    [run_phase02_scenarios, run_phase03_performance, run_phase04_steering],
)
def test_phase_stubs_write_nothing(runner, tmp_path):
    config = ExperimentConfig(
        experiment="phase02_scenarios",
        status="stub",
        output_root=str(tmp_path / "results"),
    )
    with pytest.raises(PhaseNotImplemented, match="No results were written"):
        runner(config)
    assert not (tmp_path / "results").exists()


def test_cli_stub_exits_without_results(tmp_path, capsys):
    path = tmp_path / "phase02.json"
    path.write_text(
        json.dumps(
            {
                "experiment": "phase02_scenarios",
                "status": "stub",
                "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                "seed": 0,
                "question": PHASES["phase02_scenarios"]["question"],
                "output_root": str(tmp_path / "results"),
            }
        ),
        encoding="utf-8",
    )
    code = main(["run", "--config", str(path)])
    assert code == 2
    assert "not implemented" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_gitignore_keeps_caches_and_generated_results_out_of_git():
    text = Path(".gitignore").read_text(encoding="utf-8")
    for entry in (
        "results/**",
        "!results/phase01_directions/Qwen_Qwen2.5-1.5B-Instruct/**",
        "!results/phase02_scenarios/Qwen_Qwen2.5-1.5B-Instruct/**",
        "!results/phase02_scenarios/Qwen_Qwen2.5-3B-Instruct/**",
        "results/phase02_scenarios/Qwen_Qwen2.5-3B-Instruct/directions.npz",
        ".cache/",
        ".venv/",
        "*.safetensors",
        ".ipynb_checkpoints/",
    ):
        assert entry in text
    import subprocess

    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "results/phase02_scenarios/example.json"],
        check=False,
    )
    kept = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "results/phase01_directions/Qwen_Qwen2.5-1.5B-Instruct/metrics.json",
        ],
        check=False,
    )
    kept_phase02 = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "results/phase02_scenarios/Qwen_Qwen2.5-1.5B-Instruct/metrics.json",
        ],
        check=False,
    )
    kept_phase02_3b = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "results/phase02_scenarios/Qwen_Qwen2.5-3B-Instruct/metrics.json",
        ],
        check=False,
    )
    ignored_npz = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "results/phase02_scenarios/Qwen_Qwen2.5-3B-Instruct/directions.npz",
        ],
        check=False,
    )
    assert ignored.returncode == 0
    assert kept.returncode == 1
    assert kept_phase02.returncode == 1
    assert kept_phase02_3b.returncode == 1
    assert ignored_npz.returncode == 0
