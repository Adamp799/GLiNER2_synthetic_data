# Technical Assessment — Research Engineer

Design a Python module that synthetically generates training data for [GLiNER2](https://github.com/fastino-ai/GLiNER2) — a unified IE framework supporting NER, classification, relation extraction, and JSON extraction.

---

## Part 1 — `generate.py`

Implement a `DataGenerator` class with the following interface. Internal design is entirely up to you.

```python
class DataGenerator:
    def generate(self, task_description: str, n: int) -> list[dict]:
        """Generate n GLiNER2-compatible training examples from a task description.
        Task types are inferred automatically. When multiple are needed, their
        output fields are merged into a single output dict per example.
        """
```

**Output fields by task type:**

| Task type | Output field(s) |
|---|---|
| `ner` | `{"entities": {"label": ["span", ...]}}` |
| `classification` | `{"classifications": [{"task": "task name", "labels": ["label1", ...], "true_label": ["label1"]}]}` |
| `relation_extraction` | `{"relations": [{"relation_name": {"head": "span", "tail": "span"}}]}` |
| `json_extraction` | `{"json_structures": [{...}]}` |

**Diversity requirements:** Examples must be varied and non-repetitive. For classification, `true_label` must be **balanced across all labels** over the full set of `n` examples — no label should dominate.

> **Core challenge:** A description like _"Extract companies names and classify sentiment"_ implicitly requires NER + classification.

See the [GLiNER2 training format docs](https://github.com/fastino-ai/GLiNER2#training-data-format-jsonl) for exact schema conventions.

---

## Part 2 — `notebook.ipynb` (demo section)

Showcase the module by generating training data under diverse task descriptions (one for each of the 4 task types, plus 2 that require multi-task composition). 

---

## Bonus — Fine-tune & Evaluate _(optional, weighted heavily)_

In the same `notebook.ipynb`, fine-tune [`fastino/gliner2-base-v1`](https://huggingface.co/fastino/gliner2-base-v1) on the generated data for **one task of your choice**. Reserve a separate held-out set (generated independently, not sampled from the training set) and evaluate both the base and fine-tuned models on it, reporting at least one metric (e.g. exact span match, accuracy). Reflect on results in `design.md`.

---

## Deliverables

```
your_submission/
├── generate.py        # DataGenerator class
├── notebook.ipynb     # Part 2 demo + bonus fine-tuning/evaluation (if attempted)
├── design.md          # Design reasoning (500+ words, see below)
├── requirements.txt
└── README.md
```

### `design.md`

Write up your design reasoning in 500+ words. Cover how you approach task-type inference, how multi-task outputs are composed coherently, how you ensure diversity (including label balance), your architecture decisions and tradeoffs, and what limitations remain.