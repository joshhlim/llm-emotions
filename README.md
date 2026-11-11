# Do LLMs get angry?

[![CI](https://github.com/joshhlim/llm-emotions/actions/workflows/ci.yml/badge.svg)](https://github.com/joshhlim/llm-emotions/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshhlim/llm-emotions/blob/main/notebooks/01_emotion_directions.ipynb)

A small, open study of emotion-like **representations** and **behaviour** in open-weight language models. The title is a question used as a hook. The measurements are residual-stream directions, linear probes, and generated text.

This repository is the code, the datasets, and the experiment runner for that study. It is set up so later phases can be added without rewriting the loader, the probes, or the way results are saved.

## Caveats

Read these before quoting anything in the repo.

- The project studies representations and behaviour. It does not show that a model feels angry, afraid, proud, or anything else.
- Probe accuracy on the bundled contrastive set is a pipeline check, not a finding. The sentences name the emotions they are about ("furious", "ashamed", "overjoyed"), so a linear classifier can look strong by picking up those words.
- A direction can track sentiment, topic, or sentence length rather than one emotion.
- Phase 01 includes a steering demo on a single prompt. Damaged fluency at high strength is an observation about the intervention. It is not a performance result.
- Figures under `docs/smoke_test_0.5b_cpu/` come from a CPU wiring check of `Qwen/Qwen2.5-0.5B-Instruct`. They show that the pipeline draws its plots. They are not results.

## Research questions

| Phase | Question | Status |
| --- | --- | --- |
| [01 Emotion directions](notebooks/01_emotion_directions.ipynb) | Can contrastive first-person sentences yield linear directions and probes for emotion-like states in the residual stream? | Implemented |
| [02 Scenarios and tone](notebooks/02_scenarios.ipynb) | Do provoking or impossible prompts activate those directions, and does that shift show up in the model's reply tone? | Stub |
| [03 Performance](notebooks/03_performance.ipynb) | Does emotional framing change accuracy on a reasoning task, for example a GSM8K subset? | Stub |
| [04 Steering](notebooks/04_steering.ipynb) | Can steering an emotion direction up or down change task performance, compared with prompt-only emotional framing? | Stub |

Phase 01 already *projects* a few scenario prompts onto its directions, and it *demos* anger steering on one message. Those cells illustrate the geometry. Phases 02 and 04 are the studies that would test tone and task accuracy. They are not written, and this repo does not contain made-up scores for them.

The methods sit next to contrastive activation addition (Panickssery et al., 2023, [arXiv:2312.06681](https://arxiv.org/abs/2312.06681)), representation engineering (Zou et al., 2023, [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)), and EmotionPrompt (Li et al., 2023, [arXiv:2307.11760](https://arxiv.org/abs/2307.11760)). Nothing here is a replication of those papers.

## Reproduce

Python 3.10 or newer. The default model is `Qwen/Qwen2.5-1.5B-Instruct`. A CPU check can use `Qwen/Qwen2.5-0.5B-Instruct`. Both are public. Gemma 2 and Llama 3.2 checkpoints in the same size range are gated: accept the Hub licence before loading them.

### Local

```bash
git clone https://github.com/joshhlim/llm-emotions.git
cd llm-emotions
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On a machine without a GPU, install the CPU build of PyTorch before the package if you want to avoid the large default wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

Phase 01, default model, seed 0, every layer:

```bash
llm-emotions run --config experiments/phase01_directions.json
```

CPU wiring check:

```bash
llm-emotions run \
  --config experiments/phase01_directions.json \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --batch-size 4 \
  --max-new-tokens 40
```

Outputs land in `results/phase01_directions/<model>/`:

- `metrics.json`, `config.json`, `steering_samples.json`
- `probe_accuracy.csv`, `direction_accuracy.csv`, `scenario_scores.csv`, `cosine_similarity.csv`
- `directions.npz` for a later phase to reload
- `figures/probe_accuracy.png`, `figures/direction_pca.png`, `figures/scenario_scores.png`

`results/` is gitignored. Regenerate it; do not treat a checked-in number as a finding. The seed is part of the config (`"seed": 0`) and is passed to Python, NumPy, PyTorch, the cross-validation splits, the logistic probe, and PCA. `--seed`, `--model-id`, `--layers`, `--batch-size`, and `--max-new-tokens` override the file. `--dry-run` prints the output directory and does not load a model. Stub configs exit with status 2 and write nothing:

```bash
llm-emotions run --config experiments/phase02_scenarios.json
```

The notebook is the same phase, one step at a time:

```bash
pip install jupyter
jupyter notebook notebooks/01_emotion_directions.ipynb
```

Set `EMOTION_MODEL_ID` (and optionally `EMOTION_BATCH_SIZE`, `EMOTION_MAX_NEW_TOKENS`, `EMOTION_SEED`) in the environment before launching it. The working directory does not matter once the package is installed: datasets are found from the checkout or from the installed wheel.

### Colab

Open the badge at the top. It points at `notebooks/01_emotion_directions.ipynb` on `main`. The first code cell runs

```text
pip install "llm-emotions @ git+https://github.com/joshhlim/llm-emotions.git@main"
```

when it detects Colab, so the package and the JSONL datasets install together. Select a GPU runtime, leave the model id at the 1.5B default, and run the notebook.

## Data

| Path | Contents |
| --- | --- |
| `data/emotions/<emotion>.jsonl` | Dataset `contrastive_v1`. Twenty lines per emotion: `pair_id`, `emotion_text`, `neutral_text`. Order is anger, frustration, fear, sadness, joy, calm, pride, shame. |
| `data/scenarios/phase1.jsonl` | Dataset `scenarios_v1`. Six situations, each with a `scenario` line and a `control` line. |

Emotions and scenarios are data, not string literals inside the model code. Edit a JSONL file to change the set. The loaders check that ids are unique, that each emotional sentence differs from its neutral, and that every scenario pair has one control.

## Repo layout

```text
data/emotions/                 contrastive pairs, one JSONL file per emotion
data/scenarios/phase1.jsonl    scenario prompts and matched controls
experiments/                   JSON configs (model, seed, layers, paths)
notebooks/01_emotion_directions.ipynb
notebooks/02_scenarios.ipynb   stub
notebooks/03_performance.ipynb stub
notebooks/04_steering.ipynb    stub
src/llm_emotions/              installable package
  model.py                     load a causal LM, choose device and dtype
  activations.py               chat template and last-token residuals
  directions.py                mean-difference directions and logistic probes
  scoring.py                   project scenarios onto directions
  steering.py                  forward hook
  plotting.py                  the three phase-01 figures
  results.py                   JSON, CSV, npz, PNG
  experiment.py                phase-01 runner
  phases.py                    later-phase entry points (they refuse to run)
  cli.py                       `llm-emotions run`
tests/                         pytest, tiny random Qwen2, no Hub download
.github/workflows/ci.yml       ruff and pytest
```

## Tests

```bash
pytest
ruff check src tests
ruff format --check src tests
```

The tests build a tiny random Qwen2 in memory (a few layers, hidden size 32) and run the phase-01 writer against a handful of synthetic sentences. They also check the real JSONL files, the configs, and that stub phases create no output directory. CI does the same on CPU. It does not download Qwen2.5.

## Pipeline check, not a result

These three figures are from a CPU run of `Qwen/Qwen2.5-0.5B-Instruct` used to confirm that the notebook and the CLI finish. The contrastive set names its emotions, so a probe can sit near the top of the chart for that reason alone. Do not cite the curves, the PCA layout, or the bar heights as findings.

![Smoke test, not a result: probe accuracy by layer for Qwen2.5-0.5B-Instruct on CPU](docs/smoke_test_0.5b_cpu/probe_accuracy.png)

![Smoke test, not a result: PCA of mean-difference directions for Qwen2.5-0.5B-Instruct on CPU](docs/smoke_test_0.5b_cpu/direction_pca.png)

![Smoke test, not a result: scenario projections for Qwen2.5-0.5B-Instruct on CPU](docs/smoke_test_0.5b_cpu/scenario_scores.png)

## License

[MIT](LICENSE). Copyright (c) 2026 Josh Lim.
