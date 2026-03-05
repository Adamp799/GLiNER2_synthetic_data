## Overview

This submission implements a self-contained synthetic data generator for GLiNER2. The core goal is to take a short natural language description of a task and emit JSONL-style examples that follow the GLiNER2 training format: each example is a dict with an `input` field (free-form text) and an `output` field (a structured dict). The structured output can contain any combination of four keys — `entities`, `classifications`, `relations`, and `json_structures` — depending on which task types the description implies. The key challenges are: inferring which task types are implied, composing multiple types into a single coherent output per example, and enforcing diversity and label balance. The implementation also includes a complete fine-tuning and evaluation workflow in the notebook.

The implementation is split into two layers. By default it uses lightweight keyword and pattern heuristics to infer task types and draws from banks of templates to synthesise examples. On top of that there are two optional LLM-backed modes (task inference and full example generation) that delegate to a local Ollama-served `llama3.2` model. Both LLM paths normalise their decisions into the same `ParsedTask` dataclass and fall back automatically to the rules-based path on any failure, so the generator always produces output.

## Task-type inference

Task inference is performed inside `generate()`: when `task_inference_mode="llm"` it calls `_infer_tasks_via_llm` first; if that returns `None` (Ollama unavailable, invalid JSON, or timeout), or when using the default rules mode, it uses `_infer_tasks_rules`.

In **rule-based mode**, `_infer_tasks_rules` lowercases the description and applies per-task regex checks. Each task type is restricted to what the template generators actually support — anything outside these constraints requires LLM mode:

- **NER** — looks for generic NER keywords or reference to `names` to enable `ner_context`. If so, detects which of the three supported entity types (company, person, location) are mentioned, sets `ner_entity_labels` to those, and enables NER. If only a generic NER keyword (`"ner"`, `"named entity"`, `"annotate"`, `"highlight"`) is present with no specific entity, NER is enabled with `ner_entity_labels=None`, which the generator treats as all three types. Entity types outside company/person/location are not supported in template mode.
- **Classification** — triggered only for sentiment, either by the keyword `"sentiment"` or by the co-occurrence of all three canonical labels (`"positive"`, `"negative"`, `"neutral"`) alongside a tone/classification keyword. Non-sentiment classification requires LLM mode.
- **Relation extraction** — triggered by relation keywords. Template mode only generates `works_for` and `lives_in` relations.
- **JSON extraction** — triggered by JSON/schema keywords. Template mode only generates the product schema (name, storage, price).

In **LLM mode**, `_infer_tasks_via_llm` calls the shared `_call_llm` helper with a structured prompt, asking the model to return a JSON object specifying which task types are active and what labels to use. The response is parsed by `_parse_json_response` and then `_parsed_task_from_dict` for the example generation step.

## Example generation 

In **template mode**, for each of the `n` examples the generator runs sub-generators corresponding to the active task flags, appends each sub-generator’s text to a list, merges their output dicts into a single `output`, and joins the text fragments with a space to form the `input` field. Each sub-generator draws from fixed banks:

- **NER** — `_generate_ner_example` picks a sentence template from a bank that depends on the requested subset (single-type: 5 templates each; two-type pairs: 5 each; all three: 6), and samples from class-level banks of 10 people, 10 companies, and 10 locations.
- **Classification** — `_generate_classification_example(idx)` produces a sentiment template built from a random subject (8 options) and a random label-aligned outcome (8 per label), giving up to 64 distinct sentences per label (192 total).
- **Relation extraction** — `_generate_relation_example` returns one of 4 sentence templates that mention a person, company, and city, and outputs the two relations `works_for` (person → company) and `lives_in` (person → city).
- **JSON extraction** — `_generate_json_example` returns a single product sentence and a `product` structure with name, storage, and price, sampled from a fixed list of 9 products.

In **LLM mode** when `example_generation_mode="llm"`, the generator delegates full example construction — both the input text and the structured output — to the LLM for each example. The true label for any classification task is chosen **deterministically** before the LLM call (`labels[idx % len(labels)]`) and passed to the model as a hard constraint. After the LLM responds, `_generate_example_via_llm` enforces correctness regardless of what the LLM produced: the `true_label`, `task`, and `labels` fields are overwritten for classification; entity keys are filtered to only those in `ner_entity_labels`; each relation entry is validated to have the `{rel_name: {head, tail}}` structure and dropped if malformed; and `json_structures` is validated to be a list of dicts. This enforcement means the balance guarantee and structural contract hold even when the LLM partially ignores the prompt. The shared `_parse_json_response` helper also strips noise such as markdown code fences from the LLM response before JSON parsing.

## Multi-task composition

`generate` iterates `n` times. In each iteration it activates whichever sub-generators correspond to the active task flags, appends each sub-generator's text fragment to a list, and merges each sub-generator's output dict into a shared `output` dict for that example. At the end of the iteration, the text fragments are joined with a space into the `input` field. If LLM-backed generation is enabled, the LLM generates a single example to match the active task flags.

Each sub-generator is self-contained and the composition logic only controls how their outputs are merged. This makes it easy to add new sub-generators and reason about the schema. The trade-off is that multi-task examples produce inputs with two or more loosely related sentences — for example, a NER sentence about a company's quarterly earnings followed by a sentiment sentence about customer support. 

## Diversity and label balance

**Template banks.** Each sub-generator draws from class-level constant banks of names, phrases, and sentence templates rather than re-allocating these on every call. The sentiment generator composes sentences from 8 subjects × 8 label-aligned outcomes per label = 64 unique combinations per label (192 overall). The NER generator uses dedicated template banks for every subset of the three supported entity types (single-type: 5 templates each; two-type pairs: 5 templates each; all three: 6 templates), drawing from 10 people × 10 companies × 10 locations.

**Label cycling.** `_generate_classification_example` assigns the true label by cycling through `["positive", "negative", "neutral"]` in order (`labels[idx % 3]`). This guarantees that for any `n` examples, no label is under-represented by more than one example relative to the others.

**Per-subset NER templates.** The NER sub-generator selects a template bank based on exactly which entity types are requested, covering all seven non-empty subsets of {company, person, location}. Each template only mentions the entities in the requested subset, so the input text contains no unannotated entity spans from other categories.

**`generate_balanced_unique`.** The notebook's fine-tuning section uses a helper that enforces both structural uniqueness (by example fingerprint) and strict per-label quotas. It fills per-label buckets by calling `DataGenerator.generate` in batches and routing each generated example to its label's bucket. The training and evaluation sets are generated independently (seeds 123 and 456). The `exclude_keys` parameter ensures no example fingerprint (serialised JSON) appears in both splits. `max_attempts` defaults to 192 (the size of the sentiment template pool) and raises `ValueError` if quotas are not met before then. Round-robin interleaving produces the final sequence with the same label-cycling pattern as the generator.

The result for the default sentiment task (n=150 train, n=30 eval) is exact balance: 50/50/50 in the training set and 10/10/10 in the evaluation set, with zero cross-split overlap.

## Architecture decisions and trade-offs

The main architectural choice was to keep `generate.py` free of any third-party runtime dependency. The standard library handles all HTTP communication with Ollama (`urllib`). All GLiNER2-specific dependencies — the model, training API, and evaluation code — are confined to the notebook, so users who only need to generate data do not need to install or load a heavyweight model.

The two LLM opt-in modes (`task_inference_mode` and `example_generation_mode`) are independently selectable and share the `_call_llm` helper. Users can run purely rule-based, LLM-inferred with template generation, rule-based with LLM generation, or fully LLM-driven, by passing the appropriate constructor arguments. Every LLM call falls back to the corresponding rule-based or template-based path on failure, so mixing modes is safe. Different combinations provide different strengths, for example rules-based inference with LLM-backed generation allows for more reliable inference of allowed tasks and unlimited unique examples.

For multi-task composition text concatenation was chosen over joint narrative generation because writing examples for each possible combination of tasks was not feasible. LLM generation mode produces multi-task narratives, but at the cost of less predictable output structure.

Using a local Ollama model allows for more flexibility in usage and avoids third-party dependency, but the model is slower and less intelligent.

## Evaluation and results

The notebook evaluates both the base and fine-tuned GLiNER2 models on the held-out synthetic set using **classification accuracy**: the fraction of examples where the model's predicted label exactly matches the gold label. Per-label accuracy is reported alongside overall accuracy to expose class-specific weaknesses.

**Setup:** 150 training examples (50/50/50 per label) and 30 held-out evaluation examples (10/10/10 per label), generated with different seeds and no cross-split overlap. Fine-tuned for 3 epochs with batch size 8 (~19 steps per epoch, 57 steps total).

Results should be interpreted with caution: the evaluation set is synthetically generated from the same template pool as the training set. A model that has memorised the template structure rather than learned genuine sentiment will also score well. The results demonstrate that the fine-tuning pipeline runs correctly and that the generated data is internally consistent, but not that the fine-tuned model generalises to real-world sentiment text.

## Limitations and future work

**Hardware and language model.** This project was developed using an old GPU not fully compatible with WSL and unable to run the most powerful language models. The `llama3.2` model used is small and significantly inferior to `llama4` which can also be run from Ollama. With more powerful hardware the user may also choose to generate and train on a much larger dataset. Future improvements could shift to using an API to access more powerful models, rather than local models.

- **Template pool ceiling.** The sentiment generator supports at most 8 × 8 = 64 unique examples per label (192 total for 3 labels). Requesting more unique examples than the pool can provide causes `generate_balanced_unique` to raise a `ValueError`. Switching to `example_generation_mode="llm"` removes this ceiling at the cost of unreliable generation.

**Rules-based mode only supports set tasks.** Without the LLM, only certain tasks such as sentiment for classification and company/person/location entities for NER are available. This restricts the generation potential to only work with certain task descriptions. This can be mitigated by expanding functionality and the template bank.

- **LLM-mode unreliability.** LLM functionality shows high unreliability in testing, such as occasionally infering the wrong task or generating examples incorrectly. This heavily reduces the effectiveness of the LLM-modes, but could be mitigated with more potent models and further safeguards on output that ensure correctness. Currently there is functionality to re-prompt the LLM multiple times if it was supposed to produce entities but the entities output is empty. There is also functionality that enforces certain output contraints on the LLM generation, replacing the output if necessary.

- **Multi-task text coherence.** Concatenating independent text fragments produces multiple sentence inputs corresponding to different tasks. The input text is still valid training data for GLiNER2, but the examples look contrived and would not fool a human reader.
