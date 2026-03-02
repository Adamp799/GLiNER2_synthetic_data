## GLiNER2 Synthetic Data Generator (Technical Assessment)

This folder contains a self-contained submission for the **Research Engineer technical assessment**. It implements a small Python module that synthetically generates GLiNER2-compatible training data, plus a demo notebook and design write-up.

### Contents

- `generate.py` – Implements the `DataGenerator` class:
  - `generate(task_description: str, n: int) -> list[dict]`
  - Infers which GLiNER2 task types are required (`ner`, `classification`, `relation_extraction`, `json_extraction`) using either rule-based heuristics or an optional LLM-backed mode.
  - Returns GLiNER2-style examples of the form: `{"input": "<text>", "output": {...}}`.
- `notebook.ipynb` – Demonstration notebook:
  - Shows examples for each single task type.
  - Shows multi-task examples where outputs are composed into one `output` dict.
  - Includes a scaffold for fine-tuning `fastino/gliner2-base-v1` on a synthetic sentiment task and evaluating it.
- `design.md` – 500+ word design reasoning: task inference, multi-task composition, diversity, label balance, and limitations.
- `requirements.txt` – Minimal dependencies for running the notebook and (optionally) GLiNER2 training.

### Usage

1. **Install dependencies** (inside an existing virtualenv):

```bash
pip install -r requirements.txt
```

2. **Generate synthetic examples in Python**:

```python
from generate import DataGenerator

gen = DataGenerator(seed=42)  # or DataGenerator(seed=42, task_inference_mode="llm")
examples = gen.generate(
    \"Extract company names and classify sentiment into positive, negative and neutral.\",
    n=10,
)

print(examples[0][\"input\"])
print(examples[0][\"output\"])
```

3. **Run the notebook**:

From the project root (or this folder), start Jupyter and open `notebook.ipynb`:

```bash
jupyter notebook research_engineer_submission/notebook.ipynb
```

The last section of the notebook shows how to fine-tune and evaluate `fastino/gliner2-base-v1` using the generated JSONL files.

### Notes on LLM usage

The core generator in `generate.py` is rule-based by default and does **not** require an external LLM. However, you can opt into an LLM-backed task inference mode by constructing the generator with:

```python
gen = DataGenerator(task_inference_mode="llm")
```

In this mode, task inference is delegated to a local **Ollama** instance running the `llama3.2` model via its HTTP API, while the rest of the pipeline (example generation and JSON schema) remains unchanged. If the Ollama server is unavailable or returns invalid JSON, the generator automatically falls back to the built-in heuristic inference.

