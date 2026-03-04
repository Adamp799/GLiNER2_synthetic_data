## Overview

This submission implements a self-contained synthetic data generator for GLiNER2. The core goal is to take a short natural language description of a task and emit JSONL-style examples that follow the GLiNER2 training format: each example is a dict with an `input` field (free-form text) and an `output` field (a structured dict). The structured output can contain any combination of four keys — `entities`, `classifications`, `relations`, and `json_structures` — depending on which task types the description implies. The key challenges are: inferring which task types are implied, composing multiple types into a single coherent output per example, and enforcing diversity and label balance. The implementation also includes a complete fine-tuning and evaluation workflow in the notebook.

The implementation is split into two layers. By default it uses lightweight keyword and pattern heuristics to infer task types and draws from banks of templates to synthesise examples. On top of that there are two optional LLM-backed modes (task inference and full example generation) that delegate to a local Ollama-served `llama3.2` model. Both LLM paths normalise their decisions into the same `ParsedTask` dataclass and fall back automatically to the rule-based path on any failure, so the generator always produces output.

## Task-type inference

Task inference is handled by `_infer_tasks`, which dispatches to either `_infer_tasks_rules` or `_infer_tasks_via_llm`.

In **rule-based mode**, `_infer_tasks_rules` lowercases the description and applies per-task regex checks. Each task type is restricted to what the template generators actually support — anything outside these constraints requires LLM mode:

- **NER** — detects which of the three supported entity types (company, person, location) are mentioned, sets `ner_entity_labels` to those, and enables NER. If only a generic NER keyword (`"ner"`, `"named entity"`, `"annotate"`, `"highlight"`) is present with no specific type, NER is enabled with `ner_entity_labels=None`, which the generator treats as all three types. Entity types outside company/person/location are not supported in template mode.
- **Classification** — triggered only for sentiment, either by the keyword `"sentiment"` or by the co-occurrence of all three canonical labels (`"positive"`, `"negative"`, `"neutral"`) alongside a tone/classification keyword. Non-sentiment classification requires LLM mode.
- **Relation extraction** — triggered by relation keywords. Template mode only generates `works_for` and `lives_in` relations.
- **JSON extraction** — triggered by JSON/schema keywords. Template mode only generates the product schema (name, storage, price).

In **LLM mode**, `_infer_tasks_via_llm` calls the shared `_call_llm` helper (see below) with a structured prompt asking the model to return a JSON object specifying which task types are active and what labels to use. The response is parsed by `_parsed_task_from_dict`, which applies the same defaulting logic (fill in task name and labels when missing) as the rule-based path. If the LLM call fails for any reason, the method returns `None` and the caller falls back to the rule-based path.

## LLM helper and example generation mode

Both LLM-backed inference and LLM-backed example generation share the same HTTP call pattern. Rather than duplicating it, both delegate to a single `_call_llm(messages, timeout)` private method. This method constructs the Ollama `/api/chat` payload, makes the POST request via the standard library `urllib`, parses the JSON response, and returns the assistant content string — or `None` on any network error, timeout, or malformed response. Centralising the call means timeout configuration and the None-on-failure contract apply uniformly to both modes.

The optional `example_generation_mode="llm"` mode (distinct from `task_inference_mode="llm"`) delegates full example construction — both the input text and the structured output — to the LLM for each example. The true label for any classification task is chosen **deterministically** before the LLM call (`labels[idx % len(labels)]`) and passed to the model as a hard constraint. After the LLM responds, `_generate_example_via_llm` enforces correctness regardless of what the LLM produced: the `true_label`, `task`, and `labels` fields are overwritten for classification; entity keys are filtered to only those in `ner_entity_labels`; each relation entry is validated to have the `{rel_name: {head, tail}}` structure and dropped if malformed; and `json_structures` is validated to be a list of dicts. This enforcement means the balance guarantee and structural contract hold even when the LLM partially ignores the prompt. A shared `_parse_json_response` helper also strips markdown code fences from the LLM response before JSON parsing, since models frequently wrap output in ` ```json ``` ` blocks despite being instructed not to.

## Multi-task composition

`generate` iterates `n` times. In each iteration it activates whichever sub-generators correspond to the active task flags, appends each sub-generator's text fragment to a list, and merges each sub-generator's output dict into a shared `output` dict for that example. At the end of the iteration, the text fragments are joined with a space into the `input` field.

The approach of concatenating independent text fragments is deliberately simple: each sub-generator is self-contained and the composition logic only controls how their outputs are merged. This makes it easy to add new sub-generators and reason about the schema. The trade-off is that multi-task examples produce inputs with two or more loosely related sentences — for example, a NER sentence about a company's quarterly earnings followed by a sentiment sentence about customer support. The two sentences may be thematically unrelated even though the entities and sentiment label are both grounded in the same `output` dict. This is the most visible limitation of the current design and is discussed further in the limitations section.

## Diversity and label balance

**Template banks.** Each sub-generator draws from class-level constant banks of names, phrases, and sentence templates rather than re-allocating these on every call. The sentiment generator composes sentences from 8 subjects × 8 label-aligned outcomes per label = 64 unique combinations per label. The NER generator uses dedicated template banks for every subset of the three supported entity types (single-type: 5 templates each; two-type pairs: 5 templates each; all three: 6 templates), drawing from 10 people × 10 companies × 10 locations.

**Label cycling.** `_generate_classification_example` assigns the true label by cycling through `["positive", "negative", "neutral"]` in order (`labels[idx % 3]`). This guarantees that for any `n` examples, no label is under-represented by more than one example relative to the others.

**Per-subset NER templates.** The NER sub-generator selects a template bank based on exactly which entity types are requested, covering all seven non-empty subsets of {company, person, location}. Each template only mentions the entities in the requested subset, so the input text contains no unannotated entity spans from other categories.

**`generate_balanced_unique`.** The notebook's fine-tuning section uses a helper function that enforces both structural uniqueness and strict per-label quotas. It fills separate per-label buckets by calling `DataGenerator.generate` in batches and routing each generated example to its label's bucket. The training and evaluation sets are generated independently (seeds 123 and 456 respectively). The `exclude_keys` parameter ensures that no example fingerprint (serialised JSON hash) can appear in both splits. A `max_attempts` guard raises `ValueError` if the template pool is exhausted before the quotas are met, preventing silent infinite loops. Round-robin interleaving at the end of the function produces the final sequence with the same cycling pattern as the idx-based label assignment in the generator.

The result for the default sentiment task (n=150 train, n=30 eval) is exact balance: 50/50/50 in the training set and 10/10/10 in the evaluation set, with zero cross-split overlap.

## Architecture decisions and trade-offs

The main architectural choice was to keep `generate.py` free of any third-party runtime dependency. The standard library handles all HTTP communication with Ollama (`urllib`). All GLiNER2-specific dependencies — the model, training API, and evaluation code — are confined to the notebook, so users who only need to generate data do not need to install or load a heavyweight model.

The two LLM opt-in modes (`task_inference_mode` and `example_generation_mode`) are independently selectable and share the `_call_llm` helper. Users can run purely rule-based, LLM-inferred with template generation, rule-based with LLM generation, or fully LLM-driven, by passing the appropriate constructor arguments. Every LLM call falls back to the corresponding rule-based or template-based path on failure, so mixing modes is safe.

For multi-task composition we chose text concatenation over joint narrative generation because it keeps each sub-generator independent and easy to test. A future version could use the LLM generation mode to produce multi-task narratives where the NER entities and sentiment label refer to the same subject, but at the cost of more complex prompting and less predictable output structure.

## Evaluation and results

The notebook evaluates both the base and fine-tuned GLiNER2 models on the held-out synthetic set using two metrics:

**Classification accuracy** is the fraction of examples where the model's predicted label exactly matches the gold label. Per-label accuracy is reported alongside overall accuracy to expose class-specific weaknesses.

**Span match (entity-level)** applies when the eval data contains NER. Each `(label, span_text)` pair is treated as one span. Micro-averaged precision (matched predictions / total predictions), recall (matched gold / total gold), and F1 are computed over the full eval set. This metric is stricter than accuracy because a single missed or extra span affects the score.

**Setup:** 150 training examples (50/50/50 per label) and 30 held-out evaluation examples (10/10/10 per label), generated with different seeds and no cross-split overlap. Fine-tuned for 3 epochs with batch size 8 (56 steps total).

Results should be interpreted with caution: the evaluation set is synthetically generated from the same template pool as the training set. A model that has memorised the template structure rather than learned genuine sentiment will also score well. The results demonstrate that the fine-tuning pipeline runs correctly and that the generated data is internally consistent, but not that the fine-tuned model generalises to real-world sentiment text.

## Limitations and future work

- **Template pool ceiling.** The sentiment generator supports at most 8 × 8 = 64 unique examples per label (192 total for 3 labels). Requesting more unique examples than the pool can provide causes `generate_balanced_unique` to raise a `ValueError`. Switching to `example_generation_mode="llm"` removes this ceiling at the cost of an Ollama dependency.

- **Template mode only supports sentiment classification.** The template path hardcodes sentiment (positive / negative / neutral) regardless of the inferred task. In pure template mode this is not a problem because the rules path only sets `classification=True` for sentiment descriptions. In mixed mode (`task_inference_mode="llm"` + `example_generation_mode="templates"`), if the LLM infers a non-sentiment classification task, the template generator will silently produce sentiment output instead. For non-sentiment classification, use `example_generation_mode="llm"`.

- **Multi-task text coherence.** Concatenating independent text fragments produces two-sentence inputs where the NER sentence and the classification sentence are typically about different topics. The input text is still valid training data for GLiNER2 (which learns to map any input to the structured output), but the resulting examples look contrived and would not fool a human reader.

- **Fixed relation types.** Only `works_for` and `lives_in` are generated. Adapting to other relation types requires adding new templates.

- **Fixed JSON schema.** The JSON extraction generator produces only a `product` schema (name, storage, price). The schema is not inferred from the description.

