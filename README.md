## GLiNER2 Synthetic Data Generator (Technical Assessment)

This folder contains a self-contained submission for the **Research Engineer technical assessment**. It implements a small Python module that synthetically generates GLiNER2-compatible training data, plus a demo notebook and design write-up.

### Contents

- `generate.py` – Implements the `DataGenerator` class:
  - `generate(task_description: str, n: int) -> list[dict]`
  - Infers which GLiNER2 task types are required (`ner`, `classification`, `relation_extraction`, `json_extraction`).
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
from research_engineer_submission.generate import DataGenerator

gen = DataGenerator(seed=42)
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

The core generator in `generate.py` is fully rule-based and does **not** require an external LLM. If you later decide to use an LLM for richer text generation, you can plug in your local **Ollama** instance with the `llama3.2` model at the sub-generator level (for example, replacing template-based text with LLM-generated text) without changing the public `DataGenerator` interface.

