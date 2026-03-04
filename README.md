## GLiNER2 Synthetic Data Generator

This repository is a submission for the **Research Engineer technical assessment**. It implements a Python module that synthetically generates GLiNER2-compatible training data, fine-tunes the base model on the generated data, and evaluates both the base and fine-tuned models on a held-out set.

### Contents

- `generate.py` — The `DataGenerator` class:
  - `generate(task_description: str, n: int) -> list[dict]`
  - Infers which GLiNER2 task types are required (`ner`, `classification`, `relation_extraction`, `json_extraction`) from a natural language description using either rule-based heuristics (default) or an optional LLM-backed mode.
  - Returns GLiNER2-style examples: `{"input": "<text>", "output": {...}}`.
- `notebook.ipynb` — Demonstration and fine-tuning notebook:
  - Single-task examples for all four GLiNER2 task types.
  - Multi-task examples where outputs from multiple types are merged into one `output` dict.
  - Fine-tuning of `fastino/gliner2-base-v1` on a balanced, deduplicated synthetic sentiment dataset.
  - Evaluation of both the base and fine-tuned models on an independently generated held-out set.
- `design.md` — Design write-up covering task inference, multi-task composition, diversity, label balance, architecture trade-offs, and limitations.
- `research_engineer.md` — Technical assessment specification.
- `requirements.txt` — Runtime dependencies.

### Usage

**Install dependencies** (inside an existing virtualenv):

```bash
pip install -r requirements.txt
```

**Generate synthetic examples:**

```python
from generate import DataGenerator

gen = DataGenerator(seed=42)
examples = gen.generate(
    "Extract company names and classify sentiment into positive, negative and neutral.",
    n=10,
)

print(examples[0]["input"])
print(examples[0]["output"])
```

**Run the notebook:**

```bash
jupyter notebook notebook.ipynb
```

### LLM opt-in modes

The core module is rule-based by default and requires no external model. Two optional LLM-backed modes are available via a local [Ollama](https://ollama.com/) instance running `llama3.2`:

**LLM task inference** — delegates the task-type decision (which of `ner`, `classification`, `relation_extraction`, `json_extraction` are active, and what labels to use) to the LLM while keeping template-based example generation:

```python
gen = DataGenerator(task_inference_mode="llm")
```

**LLM example generation** — delegates full example construction (input text + structured output) to the LLM while keeping rule-based task inference:

```python
gen = DataGenerator(example_generation_mode="llm")
```

Both modes can be combined. In either case, any failure (Ollama unavailable, invalid JSON, timeout) falls back automatically to the rule-based path, so the generator always produces output.

**Note:** in template mode, classification is only supported for sentiment tasks (labels `positive`, `negative`, `neutral`). Non-sentiment classification requires `example_generation_mode="llm"`, since template-generated texts cannot be meaningfully tied to arbitrary label sets.

The Ollama host is read from the `OLLAMA_HOST` environment variable (e.g. `172.21.112.1:11434`). Unset defaults to `http://localhost:11434`.

### Fine-tuning and evaluation

The notebook's fine-tuning section:

1. Generates **150 training examples** and **30 evaluation examples** independently (different random seeds), with no example overlap between the two sets.
2. Uses `generate_balanced_unique` — a helper that fills per-label quotas and deduplicates by content fingerprint — to guarantee strict label balance (50/50/50 for the 3-label sentiment task on 150 examples, 10/10/10 on 30).
3. Fine-tunes `fastino/gliner2-base-v1` for 3 epochs on the training set.
4. Evaluates both the **base model** and the **fine-tuned model** on the held-out set, reporting overall accuracy, per-label accuracy, and entity-level span P/R/F1 when applicable.

To generate a larger training set, change `TRAIN_N` and `EVAL_N` at the top of the data-generation cell. The sentiment template pool supports up to 192 unique examples (8 subjects × 8 outcomes × 3 labels); beyond that, switch to `example_generation_mode="llm"` for unlimited variety.
