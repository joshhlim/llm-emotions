from __future__ import annotations

import json
from pathlib import Path

from llm_emotions.caveats import PHASE02_CAVEAT, PROBE_CAVEAT
from llm_emotions.phases import PHASES

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

COLAB_NOTEBOOK = (
    "https://colab.research.google.com/github/joshhlim/llm-emotions/blob/main/"
    "notebooks/01_emotion_directions.ipynb"
)
GITHUB_INSTALL = "llm-emotions @ git+https://github.com/joshhlim/llm-emotions.git@main"


def _notebook(name: str) -> dict:
    return json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))


def _text(notebook: dict, cell_type: str | None = None) -> str:
    chunks = []
    for cell in notebook["cells"]:
        if cell_type is not None and cell["cell_type"] != cell_type:
            continue
        source = cell["source"]
        chunks.append(source if isinstance(source, str) else "".join(source))
    return "\n".join(chunks)


def test_phase01_notebook_installs_the_package_and_calls_it():
    notebook = _notebook("01_emotion_directions.ipynb")
    code = _text(notebook, "code")
    markdown = _text(notebook, "markdown")
    assert COLAB_NOTEBOOK in markdown
    assert GITHUB_INSTALL in code
    assert "llm_emotions" in code
    assert PROBE_CAVEAT in markdown
    assert "I am furious" not in code
    assert PHASES["phase01_directions"]["question"] in markdown
    assert "google.colab" in code


def test_phase02_notebook_is_runnable_on_colab():
    notebook = _notebook("02_scenarios.ipynb")
    code = _text(notebook, "code")
    markdown = _text(notebook, "markdown")
    assert "notebooks/02_scenarios.ipynb" in markdown
    assert GITHUB_INSTALL in code
    assert "google.colab" in code
    assert "files.download" in code
    assert "run_phase02" in code
    assert "llm-emotions run --config experiments/phase02_scenarios.json" in markdown
    assert "results/phase02_scenarios" in markdown
    for name in (
        "EMOTION_MODEL_ID",
        "EMOTION_BATCH_SIZE",
        "EMOTION_MAX_NEW_TOKENS",
        "EMOTION_SEED",
    ):
        assert name in markdown
    assert PHASES["phase02_scenarios"]["question"] in markdown
    assert PHASE02_CAVEAT in markdown
    assert [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_later_phases_are_markdown_stubs():
    for name in ("phase03_performance", "phase04_steering"):
        phase = PHASES[name]
        notebook = _notebook(Path(phase["notebook"]).name)
        assert [cell for cell in notebook["cells"] if cell["cell_type"] == "code"] == []
        text = _text(notebook).lower()
        assert phase["question"] in _text(notebook)
        assert "not implemented" in text
        assert "0.87" not in text
        assert "accuracy:" not in text
