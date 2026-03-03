## Overview

This submission implements a self-contained synthetic data generator for GLiNER2. The core goal is to take a short natural language description of a task and emit JSONL-style examples that follow the GLiNER2 training format: each example is a dict with an `input` field (free-form text) and an `output` field (a structured dict). The structured output can contain any combination of four keys — `entities`, `classifications`, `relations`, and `json_structures` — depending on which task types the description implies. The key challenges are: inferring which task types are implied, composing multiple types into a single coherent output per example, and enforcing diversity and label balance. The implementation also includes a complete fine-tuning and evaluation workflow in the notebook.

The implementation is split into two layers. By default it uses lightweight keyword and pattern heuristics to infer task types and draws from banks of templates to synthesise examples. On top of that there are two optional LLM-backed modes (task inference and full example generation) that delegate to a local Ollama-served `llama3.2` model. Both LLM paths normalise their decisions into the same `ParsedTask` dataclass and fall back automatically to the rule-based path on any failure, so the generator always produces output.

## Task-type inference

Task inference is handled by `_infer_tasks`, which dispatches to either `_infer_tasks_rules` or `_infer_tasks_via_llm`, then runs `_enrich_ner_entity_labels` on the result regardless of which path was used.

In **rule-based mode**, `_infer_tasks_rules` lowercases the description and runs lightweight heuristics:

- **NER** is triggered by a broad set of compound keywords (`"extract entities"`, `"identify names"`, `"tag entities"`, `"annotate"`, etc.) and by a regex that matches `(extract|identify|find|tag|annotate|recognize|detect) [a-z\s]+names`.
- **Classification** is triggered by `"classify"`, `"classification"`, `"sentiment"`, `"label as"`, `"categorize"`, and `"categorise"`.
- **Relation extraction** is triggered by `"relation"`, `"relationship"`, `"works for"`, `"lives in"`, and `"born in"`.
- **JSON/structured extraction** is triggered by `"json"`, `"structured"`, `"key-value"`, `"fields:"`, `"schema"`, and `"extract as"`.

After the main flags are set, sentiment is detected separately — either by the keyword `"sentiment"` or by the co-occurrence of all three canonical labels (`"positive"`, `"negative"`, `"neutral"`) alongside a tone/classification keyword. When sentiment is detected, the task name and label list are set explicitly rather than relying on the generic label-extraction path.

Label extraction uses `_extract_label_list`, which matches the phrases `"into A, B or C"`, `"as A, B or C"`, and `"labels: A, B, C"`, splitting on commas, `"and"`, and `"or"`. This handles the most common patterns for explicit label specification.

`_enrich_ner_entity_labels` runs after both inference paths and narrows the `ner_entity_labels` hint by checking whether the description mentions specific entity type keywords (company, person, location). This hint is used by the NER sub-generator to produce more targeted examples.

In **LLM mode**, `_infer_tasks_via_llm` calls the shared `_call_llm` helper (see below) with a structured prompt asking the model to return a JSON object specifying which task types are active and what labels to use. The response is parsed by `_parsed_task_from_dict`, which applies the same defaulting logic (fill in task name and labels when missing) as the rule-based path. If the LLM call fails for any reason, the method returns `None` and the caller falls back to the rule-based path.

## LLM helper and example generation mode

Both LLM-backed inference and LLM-backed example generation share the same HTTP call pattern. Rather than duplicating it, both delegate to a single `_call_llm(messages, timeout)` private method. This method constructs the Ollama `/api/chat` payload, makes the POST request via the standard library `urllib`, parses the JSON response, and returns the assistant content string — or `None` on any network error, timeout, or malformed response. Centralising the call means timeout configuration and the None-on-failure contract apply uniformly to both modes.

The optional `example_generation_mode="llm"` mode (distinct from `task_inference_mode="llm"`) delegates full example construction — both the input text and the structured output — to the LLM for each example. The true label for any classification task is chosen **deterministically** before the LLM call (`labels[idx % len(labels)]`) and passed to the model as a hard constraint. After the LLM responds, `_generate_example_via_llm` overwrites the `true_label` field in the response with the pre-assigned value before returning the example. This two-step approach prevents the model from drifting the label distribution: even if the LLM ignores the instruction, the correct label is enforced. The same cycling-based balance guarantee that applies to the template path therefore also applies to the LLM generation path.

## Multi-task composition

`generate` iterates `n` times. In each iteration it activates whichever sub-generators correspond to the active task flags, appends each sub-generator's text fragment to a list, and merges each sub-generator's output dict into a shared `output` dict for that example. At the end of the iteration, the text fragments are joined with a space into the `input` field.

The approach of concatenating independent text fragments is deliberately simple: each sub-generator is self-contained and the composition logic only controls how their outputs are merged. This makes it easy to add new sub-generators and reason about the schema. The trade-off is that multi-task examples produce inputs with two or more loosely related sentences — for example, a NER sentence about a company's quarterly earnings followed by a sentiment sentence about customer support. The two sentences may be thematically unrelated even though the entities and sentiment label are both grounded in the same `output` dict. This is the most visible limitation of the current design and is discussed further in the limitations section.

## Diversity and label balance

**Template banks.** Each sub-generator draws from class-level constant banks of names, phrases, and sentence templates rather than re-allocating these on every call. The sentiment generator composes sentences from 8 subjects × 8 label-aligned outcomes per label = 64 unique combinations per label. The NER generator draws from 10 people × 10 companies × 10 locations across 6 sentence templates.

**Label cycling.** `_generate_classification_example` assigns the true label as `labels[idx % len(labels)]`, cycling through all labels in order. This guarantees that for any `n` examples, no label is under-represented by more than one example relative to the others.

**Single-entity-type NER templates.** When the user requests only one entity type (company, person, or location), the NER sub-generator uses a dedicated template bank for that type. The input text then contains only the requested entity, so there are no unannotated spans from other categories that might confuse a model learning to extract a specific type.

**`generate_balanced_unique`.** The notebook's fine-tuning section uses a helper function that enforces both structural uniqueness and strict per-label quotas. It fills separate per-label buckets by calling `DataGenerator.generate` in batches and routing each generated example to its label's bucket. The training and evaluation sets are generated independently (seeds 123 and 456 respectively). The `exclude_keys` parameter ensures that no example fingerprint (serialised JSON hash) can appear in both splits. A `max_attempts` guard raises `ValueError` if the template pool is exhausted before the quotas are met, preventing silent infinite loops. Round-robin interleaving at the end of the function produces the final sequence with the same cycling pattern as the idx-based label assignment in the generator.

The result for the default sentiment task (n=80 train, n=20 eval) is exact balance: 27/27/26 in the training set and 7/7/6 in the evaluation set, with zero cross-split overlap.

## Architecture decisions and trade-offs

The main architectural choice was to keep `generate.py` free of any third-party runtime dependency. The standard library handles all HTTP communication with Ollama (`urllib`). All GLiNER2-specific dependencies — the model, training API, and evaluation code — are confined to the notebook, so users who only need to generate data do not need to install or load a heavyweight model.

The two LLM opt-in modes (`task_inference_mode` and `example_generation_mode`) are independently selectable and share the `_call_llm` helper. Users can run purely rule-based, LLM-inferred with template generation, rule-based with LLM generation, or fully LLM-driven, by passing the appropriate constructor arguments. Every LLM call falls back to the corresponding rule-based or template-based path on failure, so mixing modes is safe.

For multi-task composition we chose text concatenation over joint narrative generation because it keeps each sub-generator independent and easy to test. A future version could use the LLM generation mode to produce multi-task narratives where the NER entities and sentiment label refer to the same subject, but at the cost of more complex prompting and less predictable output structure.

## Evaluation and results

The notebook evaluates both the base and fine-tuned GLiNER2 models on the held-out synthetic set using two metrics:

**Classification accuracy** is the fraction of examples where the model's predicted label exactly matches the gold label. Per-label accuracy is reported alongside overall accuracy to expose class-specific weaknesses.

**Span match (entity-level)** applies when the eval data contains NER. Each `(label, span_text)` pair is treated as one span. Micro-averaged precision (matched predictions / total predictions), recall (matched gold / total gold), and F1 are computed over the full eval set. This metric is stricter than accuracy because a single missed or extra span affects the score.

**Actual results on the held-out sentiment set (20 examples, 7/7/6 balance):**

| Model | Overall accuracy | Positive | Negative | Neutral |
|---|---|---|---|---|
| Base (`fastino/gliner2-base-v1`) | 90.00% | 100% (7/7) | 100% (7/7) | 66.67% (4/6) |
| Fine-tuned | 100.00% | 100% (7/7) | 100% (7/7) | 100% (6/6) |

The base model already handles positive and negative sentiment reliably but struggles with the neutral class (4 out of 6 correct). Fine-tuning on 80 balanced synthetic examples for 3 epochs was sufficient to bring neutral accuracy to 100%. The training took approximately 60 seconds on CPU (30 steps: 80 examples × 3 epochs / batch size 8).

These results should be interpreted with caution: the evaluation set has only 20 examples, all synthetically generated from the same template pool as the training set. A model that has memorised the template structure rather than learned genuine sentiment will also score 100%. The results therefore demonstrate that the fine-tuning pipeline runs correctly and that the generated data is internally consistent, but not that the fine-tuned model generalises to real-world sentiment text.

## Limitations and future work

- **Template pool ceiling.** The sentiment generator supports at most 8 × 8 = 64 unique examples per label (192 total for 3 labels). The generic non-sentiment classification generator has only 20 template texts, giving a ceiling of 60 unique examples for a 3-label task. Requesting more unique examples than the pool can provide causes `generate_balanced_unique` to raise a `ValueError`. Switching to `example_generation_mode="llm"` removes this ceiling at the cost of an Ollama dependency.

- **Multi-task text coherence.** Concatenating independent text fragments produces two-sentence inputs where the NER sentence and the classification sentence are typically about different topics. The input text is still valid training data for GLiNER2 (which learns to map any input to the structured output), but the resulting examples look contrived and would not fool a human reader.

- **NER text/annotation mismatch for multi-entity-type requests.** When the user requests two or more entity types (e.g. person and location), the generic multi-entity template is used. This template always mentions all three entity types (person, company, location), so the input text will contain company names that are not annotated in the output. Single-type requests (company-only, person-only, location-only) use focused templates that avoid this issue.

- **Fixed relation types.** Only `works_for` and `lives_in` are generated. Adapting to other relation types requires adding new templates.

- **Fixed JSON schema.** The JSON extraction generator produces only a `product` schema (name, storage, price). The schema is not inferred from the description.

- **Label extraction coverage.** The `_extract_label_list` heuristic handles `"into/as A, B or C"` and `"labels: A, B, C"` patterns but will miss other formulations such as `"rate from 1 to 5 stars"` or `"classify urgency as low / medium / high"`. The LLM inference mode handles these cases more robustly.
