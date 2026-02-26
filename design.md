## Overview

This submission implements a small, self-contained synthetic data generator for GLiNER2. The core goal is to take a short natural language description of a task and emit JSONL-style examples that follow the GLiNER2 training data format, i.e. each line is a dictionary with an `input` field (free-form text) and an `output` field (a structured dictionary). The structured output aggregates four possible task types: NER, classification, relation extraction, and JSON/structured extraction. The key challenges are: inferring which task types are implied by a description, composing multiple types into a single coherent output per example, and enforcing diversity and label balance without relying on an external LLM. All of this must remain simple enough to run on a CPU-only environment while still being easy to extend with richer generators later (for example, an Ollama-backed LLM as specified in your environment).

The implementation is deliberately rule-based: it uses lightweight keyword and pattern heuristics to infer task types and then draws from small banks of templates and entities to synthesize examples. This avoids a dependency on remote APIs and guarantees deterministic, reproducible output given a random seed. The generator therefore acts as a controllable baseline that can be improved or swapped out without changing the surrounding interface.

## Task-type inference

Task inference is handled by the private `_infer_tasks` method, which returns a `ParsedTask` dataclass. The dataclass tracks four booleans (`ner`, `classification`, `relation_extraction`, `json_extraction`) and some extra metadata for classification: `classification_task_name` and `classification_labels`. The method lowers the task description and runs a sequence of lightweight heuristics:

- For **NER**, it looks for phrases like “extract entities”, “extract company names”, “named entity”, or a general pattern such as `extract X names`. These expressions usually appear when the user wants spans like companies, people, or locations.
- For **classification**, it checks for words like “classify”, “classification”, “sentiment”, or “label as”. Sentiment is special-cased: if the word “sentiment” appears, the task name is set to `"sentiment"` and labels are set to `["positive", "negative", "neutral"]`.
- For **relation extraction**, it checks for “relation”, “relationship”, or canonical examples such as “works for”, “lives in”, or “born in”. These are typical relation templates in GLiNER-style examples.
- For **JSON/structured extraction**, it looks for “json”, “structured”, “key-value”, “fields:”, “schema”, or “extract as”, which are all signals that the user cares about structured outputs rather than just spans.

On top of keyword-based detection, `_infer_tasks` attempts to extract classification labels directly from the description. It searches for patterns like “into A, B and C” or “labels: A, B, C” and splits the captured text on commas and the word “and”. The logic keeps only short, non-empty labels to avoid spurious matches. If labels are found, they are adopted as the canonical label list for the classification task. If nothing is found, the generator falls back to default labels (`class_a`, `class_b`, `class_c`) so the interface remains usable even for vague descriptions.

Finally, the function fills in any missing classification metadata. If a task looks like classification but no task name is provided, it uses `"classification"`. If no labels are provided or extracted, it uses the generic defaults. This design keeps the rest of the code simple: downstream generators never need to worry about `None` values and can always assume a task name and label list exist when classification is active.

## Multi-task composition

The public `generate` method is responsible for composing multi-task outputs. It first infers the tasks from the description and then loops `n` times to generate examples. In each iteration, it accumulates text fragments and output fields:

- If NER is active, it calls `_generate_ner_example`, which returns a `(text, entities)` pair. The text is appended to a list of text segments, and `entities` is attached under `output["entities"]`.
- If classification is active, it calls `_generate_classification_example`, which returns `(text, classifications)` and attaches them under `output["classifications"]`.
- If relation extraction is active, `_generate_relation_example` is used, adding `output["relations"]`.
- If JSON extraction is active, `_generate_json_example` is used, adding `output["json_structures"]`.

At the end of each iteration, the individual text fragments are joined with spaces into the `input` field, while the output dictionary contains one key per active task type. This directly matches the GLiNER2 multi-task JSONL examples documented in the official README: a single `input` with a composite `output` object that may contain `entities`, `classifications`, `relations`, and/or `json_structures`. If, for any reason, no task type is inferred, the generator falls back to a generic classification-only example so the function never returns an empty output.

The multi-task composition is deliberately simple: each sub-generator is independent, and the composition logic only controls how their text fragments are concatenated and how their outputs are merged. This makes the design easy to reason about and test. It also means you can add new task-specific generators (for example, span-level NER with character offsets) without affecting the others.

## Diversity and label balance

The generator enforces diversity and label balance in several ways:

- **Template banks:** Each sub-generator has a small bank of templates or value lists that are randomly sampled. For example, `_generate_ner_example` draws from lists of people (`Alice Johnson`, `Bob Smith`, etc.), companies (`Acme Corp`, `Fastino AI`, …) and locations (`New York`, `Berlin`, …). Each call stitches them together in different sentence templates, producing varied surface forms while still guaranteeing valid entities.
- **Random seeds:** The `DataGenerator` accepts an optional `seed`. Internally it uses a dedicated `random.Random` instance, so users can reproduce exact sequences of examples when debugging or comparing different model runs.
- **Rotating labels for classification:** The `_generate_classification_example` method enforces approximate class balance by cycling through labels with the pattern `labels[idx % len(labels)]`. If there are three labels and 30 examples, each label will be used roughly 10 times. This directly addresses the requirement that no label “dominates” the set of examples.
- **Semantically tied text:** For sentiment-specific tasks, the generator uses a curated bank of positive, negative, and neutral sentences. This couples the text semantics with the true label in an intuitive way and can help GLiNER2 learn a more meaningful decision boundary even though the examples are synthetic.

Because the core implementation is rule-based, we do not rely on a large language model to produce diverse phrasing. However, the structure is easily extensible: you could plug in your local Ollama-backed `llama3.2` model in future by adding an optional text generation back-end inside the sub-generators (for example, replacing template selection with a short prompt to the LLM). The interface to `generate` and the JSONL format would stay the same, so all downstream code (including the notebook and training pipeline) would continue to work unchanged.

## Architecture decisions and trade-offs

The main architectural choice was to keep the generator as a pure Python module with no external runtime dependencies beyond the standard library. This keeps the `generate.py` module light-weight, fast to import, and friendly to environments without GPUs or internet access. All GLiNER2-specific dependencies (the actual model and training API) are contained in the notebook, so users who only need to generate data do not pay the cost of installing or loading a heavy model.

Another design choice was to favour clarity over extreme configurability. The heuristics in `_infer_tasks` are intentionally simple and explicit rather than learned. The label extraction patterns and keyword lists can be read and edited in a few minutes, which is useful for a research engineer who may want to adapt them to a specific domain (for example, biomedical or legal text). The trade-off is that coverage for very unusual phrasings may be limited: the generator may miss a task type if the description is phrased in a completely different way. In those cases, the fallback to generic classification still produces usable examples, but not necessarily ideal ones.

For multi-task composition, we chose to concatenate independent text fragments rather than trying to weave all required annotations into a single tightly coupled narrative. This is much simpler to implement and reason about but can produce slightly contrived sentences where, for example, the sentiment classification text and the NER text are about different topics. From a training perspective this is still acceptable, because GLiNER2 learns to map each `input` to a structured `output` and is not constrained to a human-like narrative structure, but it is a limitation to be aware of.

Finally, we intentionally did **not** require an LLM in the core generator. The problem statement allows an LLM, but in your environment it must come from a local Ollama `llama3.2` model. Building a fully prompt-driven generator around that model would be an interesting extension, but for the purposes of this assessment, a deterministic, rule-based implementation better showcases the explicit handling of GLiNER2’s schema, task inference, and label balancing. The design leaves clear extension points where an Ollama-powered generator can later be slotted in, should you wish to trade determinism for richer linguistic diversity.

## Limitations and future work

The current implementation has a few clear limitations:

- It only recognises a handful of entity types (person, company, location) and relation types (`works_for`, `lives_in`). Adapting it to other domains would require expanding these banks and templates.
- The label extraction heuristics are intentionally simple and may fail on more complex descriptions (“rate from 1 to 5 stars”, “classify urgency as low/medium/high/critical”, etc.). A natural extension would be to use a small parsing library or a lightweight LLM call (via your local Ollama deployment) to robustly extract label sets.
- The JSON extraction generator focuses on a single `product` schema. In practice, users may want arbitrary schemas. One improvement would be to parse schema-like descriptions in the task text (for example, `fields: name, storage, price`) and dynamically construct matching JSON structures.
- The synthetic data is intentionally small and somewhat repetitive in structure. For serious model training, you would want larger, more varied corpora and potentially richer linguistic features. Again, this is a natural place where a local LLM could help diversify phrasing while still following the same output schema.

Despite these limitations, the module provides a clear, GLiNER2-compatible baseline that demonstrates how to infer tasks from descriptions, compose multi-task outputs, enforce class balance, and integrate with the GLiNER2 training API via JSONL files. It is designed to be a starting point for further experimentation rather than an exhaustive data generation system.

