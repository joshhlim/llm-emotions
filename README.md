# Do LLMs get angry?

[![CI](https://github.com/joshhlim/llm-emotions/actions/workflows/ci.yml/badge.svg)](https://github.com/joshhlim/llm-emotions/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshhlim/llm-emotions/blob/main/notebooks/01_emotion_directions.ipynb)
[![Phase 02 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshhlim/llm-emotions/blob/main/notebooks/02_scenarios.ipynb)

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
| [02 Scenarios](notebooks/02_scenarios.ipynb) | Once each direction has the shared emotion-versus-neutral component removed, do scenario prompts still shift that direction relative to a matched control? | Implemented |
| [03 Performance](notebooks/03_performance.ipynb) | Does emotional framing change accuracy on a reasoning task, for example a GSM8K subset? | Stub |
| [04 Steering](notebooks/04_steering.ipynb) | Can steering an emotion direction up or down change task performance, compared with prompt-only emotional framing? | Stub |

Phase 01 already *projects* a few scenario prompts onto its directions, and it *demos* anger steering on one message. Phase 02 repeats the scenario comparison after the shared component is removed, on keyword-free sentences, with paraphrases and a null baseline. It does not score reply tone. Phases 03 and 04 are not written, and this repo does not contain made-up scores for them.

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

Phase 02, same model and seed:

```bash
llm-emotions run --config experiments/phase02_scenarios.json
```

That writes `results/phase02_scenarios/<model>/`: config, metrics, cosine similarity before and after cleanup, transfer accuracy, centred scenario scores, per-paraphrase deltas, effect sizes, `directions.npz`, and three figures. On a free Colab T4 with the 1.5B model this is a few hundred short forwards and no generation. It should finish well under 20 minutes after the checkpoint has loaded.

Generated `results/` stay gitignored. The one exception is the phase-01 snapshot below, which is the Colab run being reported. Do not treat any other checked-in number as a finding. The seed is part of the config (`"seed": 0`) and is passed to Python, NumPy, PyTorch, the cross-validation splits, the logistic probe, and PCA. `--seed`, `--model-id`, `--layers`, `--batch-size`, and `--max-new-tokens` override the file. `--dry-run` prints the output directory and does not load a model. Stub configs (phases 03 and 04) exit with status 2 and write nothing:

```bash
llm-emotions run --config experiments/phase03_performance.json
```

The notebooks are the same phases, one step at a time:

```bash
pip install jupyter
jupyter notebook notebooks/01_emotion_directions.ipynb
jupyter notebook notebooks/02_scenarios.ipynb
```

Set `EMOTION_MODEL_ID` (and optionally `EMOTION_BATCH_SIZE`, `EMOTION_MAX_NEW_TOKENS`, `EMOTION_SEED`) in the environment before launching it. The working directory does not matter once the package is installed: datasets are found from the checkout or from the installed wheel.

### Colab

The badges at the top point at `notebooks/01_emotion_directions.ipynb` and `notebooks/02_scenarios.ipynb` on `main`. The first code cell of each runs

```text
pip install "llm-emotions @ git+https://github.com/joshhlim/llm-emotions.git@main"
```

when it detects Colab, so the package and the JSONL datasets install together. Select a GPU runtime, leave the model id at the 1.5B default, and run the notebook.

## Data

| Path | Contents |
| --- | --- |
| `data/emotions/<emotion>.jsonl` | Dataset `contrastive_v1`. Twenty lines per emotion: `pair_id`, `emotion_text`, `neutral_text`. The emotional sentences name the emotion. Order is anger, frustration, fear, sadness, joy, calm, pride, shame. |
| `data/emotions_keyword_free/<emotion>.jsonl` | Dataset `contrastive_keyword_free_v1`. Sixteen pairs per emotion. The scene implies the state and does not name it or a close synonym. A unit test scans the file. |
| `data/scenarios/phase1.jsonl` | Dataset `scenarios_v1`. Six situations, each with one `scenario` line and one `control` line. |
| `data/scenarios/phase2.jsonl` | Dataset `scenarios_v2`. The same six situations, six paraphrases each, each with its own control. |
| `data/scenarios/neutral_baseline.jsonl` | Dataset `neutral_baseline_v1`. Eight bland prompts used to centre absolute scores. |

Emotions and scenarios are data, not string literals inside the model code. Edit a JSONL file to change the set. The loaders check that ids are unique, that each emotional sentence differs from its neutral, and that every scenario paraphrase has its own control. Phase 02 also refuses to run if the keyword-free sentences or the scenario prompts contain a listed emotion name.

Phase 02 fits a mean-difference direction on each contrastive set, subtracts the mean across the eight emotions, and projects that shared axis out. The layer is the one where those cleaned directions transfer best from one set to the other, in both directions. Scenario scores are centred on the neutral prompts. The reported gap is scenario minus matched control, with a bootstrap interval across the six paraphrases and a tail probability against random directions of the same length. Six cells were named before any model is scored: insult to anger, impossible to frustration, failure to shame, praise to pride, and both shutdown and high stakes to fear. The rest of the grid is descriptive. Reply tone is not measured.

## Repo layout

```text
data/emotions/                 keyword-heavy pairs, one JSONL file per emotion
data/emotions_keyword_free/    keyword-free pairs, one JSONL file per emotion
data/scenarios/phase1.jsonl    one scenario and one control per situation
data/scenarios/phase2.jsonl    six paraphrases and matched controls
data/scenarios/neutral_baseline.jsonl
experiments/                   JSON configs (model, seed, layers, paths)
notebooks/01_emotion_directions.ipynb
notebooks/02_scenarios.ipynb
notebooks/03_performance.ipynb stub
notebooks/04_steering.ipynb    stub
src/llm_emotions/              installable package
  model.py                     load a causal LM, choose device and dtype
  activations.py               chat template and last-token residuals
  directions.py                mean-difference directions and logistic probes
  cleanup.py                   remove the shared emotion-versus-neutral component
  lexicon.py                   keyword list for the keyword-free set
  stats.py                     paraphrase intervals and matched-norm nulls
  scoring.py                   project scenarios onto directions
  steering.py                  forward hook
  plotting.py                  phase-01 and phase-02 figures
  results.py                   JSON, CSV, npz, PNG
  experiment.py                phase-01 runner
  phase02.py                   phase-02 runner
  phases.py                    phase catalogue; 03 and 04 still refuse to run
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

## Phase 01 results (Qwen2.5-1.5B-Instruct)

These files are the Colab T4 run of `Qwen/Qwen2.5-1.5B-Instruct` (float16, seed 0, all 28 layers) in `results/phase01_directions/Qwen_Qwen2.5-1.5B-Instruct/`. The directory includes `directions.npz` (about 2.5 MB) so the fit can be reloaded. The figures under `docs/smoke_test_0.5b_cpu/` are a different, smaller CPU check and are not these results.

**Probe accuracy on this keyword-heavy contrastive set is a pipeline check, not a finding. The sentences name the emotions they are about, so a linear classifier can succeed without the direction generalising to keyword-free scenes.**

The chosen layer is 13, where mean cross-validated probe accuracy is 0.978. Frustration is the weakest of the eight: 0.875 at layer 13, and 0.900 at layers 21 and 28. The other seven sit between 0.975 and 1.000 at layer 13. That is what a linear classifier should do when the sentences name the emotion. It checks that the pipeline ran.

At layer 13 the eight raw mean-difference directions are strongly aligned. Off-diagonal cosine similarity runs from 0.509 (frustration versus pride) to 0.855 (sadness versus shame). Anger versus joy is 0.756. The shared component is large enough that a direction can track whether the text is emotional or neutral.

The anger projection is higher on the scenario than on its matched control in all six pairs. The gaps are +1.00 (impossible), +1.36 (insult), +1.38 (failure), +1.46 (praise), +1.48 (shutdown), and +1.52 (high stakes). Praise moves the anger axis by about as much as an insult, so this readout is not specific to anger. Shame is the largest projection on every prompt, controls included (about 5.5 to 9.3). On four of the six pairs the control scores higher than the scenario. That is a baseline offset, not a shame effect.

Steering uses one prompt: a short message to a teammate who missed a deadline. At strength 0 the reply is a polite note. At +0.4 the greeting drops and the reply talks about accountability and mistakes. At +0.8 it leaves the task, and the sentences stop hanging together. The saved note applies: qualitative samples from one prompt; fluency can collapse at high strength; this is not a performance evaluation.

![Probe accuracy by layer for Qwen2.5-1.5B-Instruct on the keyword-heavy set](results/phase01_directions/Qwen_Qwen2.5-1.5B-Instruct/figures/probe_accuracy.png)

![Scenario and control projections at layer 13 for Qwen2.5-1.5B-Instruct](results/phase01_directions/Qwen_Qwen2.5-1.5B-Instruct/figures/scenario_scores.png)

## Pipeline check, not a result

These three figures are from a CPU run of `Qwen/Qwen2.5-0.5B-Instruct` used to confirm that the notebook and the CLI finish. The contrastive set names its emotions, so a probe can sit near the top of the chart for that reason alone. Do not cite the curves, the PCA layout, or the bar heights as findings.

![Smoke test, not a result: probe accuracy by layer for Qwen2.5-0.5B-Instruct on CPU](docs/smoke_test_0.5b_cpu/probe_accuracy.png)

![Smoke test, not a result: PCA of mean-difference directions for Qwen2.5-0.5B-Instruct on CPU](docs/smoke_test_0.5b_cpu/direction_pca.png)

![Smoke test, not a result: scenario projections for Qwen2.5-0.5B-Instruct on CPU](docs/smoke_test_0.5b_cpu/scenario_scores.png)

## License

[MIT](LICENSE). Copyright (c) 2026 Josh Lim.
