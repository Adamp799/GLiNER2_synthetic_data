from __future__ import annotations

import json
import os
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, TypedDict


TaskType = Literal["ner", "classification", "relation_extraction", "json_extraction"]


class OutputDict(TypedDict, total=False):
    entities: Dict[str, List[str]]
    classifications: List[Dict]
    relations: List[Dict]
    json_structures: List[Dict]


class Example(TypedDict):
    input: str
    output: OutputDict


@dataclass
class ParsedTask:
    """Lightweight representation of inferred tasks from a description."""

    ner: bool = False
    classification: bool = False
    relation_extraction: bool = False
    json_extraction: bool = False

    classification_task_name: str | None = None
    classification_labels: List[str] | None = None
    # Optional hint about which entity labels the user cares about for NER,
    # inferred from the natural language description (e.g. ["company"]).
    ner_entity_labels: List[str] | None = None


TaskInferenceMode = Literal["rules", "llm"]
ExampleGenerationMode = Literal["templates", "llm"]


class DataGenerator:
    """Generate GLiNER2-compatible synthetic training examples from a task description.

    By default the generator uses lightweight rules for task inference and
    synthetic example construction. Optionally, task inference can be delegated
    to a local LLM (e.g. an Ollama-served `llama3.2` model) while keeping the
    rest of the pipeline unchanged.

    It focuses on:
      - inferring which GLiNER2 task-types are relevant from a natural language
        description
      - composing multi-task outputs into a single `output` dict
      - ensuring basic diversity and class balance for classification tasks
    """

    def __init__(
        self,
        seed: int | None = None,
        task_inference_mode: TaskInferenceMode = "rules",
        example_generation_mode: ExampleGenerationMode = "templates",
        llm_model: str = "llama3.2",
        llm_endpoint: str | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._task_inference_mode: TaskInferenceMode = task_inference_mode
        self._example_generation_mode: ExampleGenerationMode = example_generation_mode
        self._llm_model = llm_model
        # Resolve the Ollama chat endpoint, preferring OLLAMA_HOST when set.
        if llm_endpoint is None:
            host = os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
            if not host.startswith(("http://", "https://")):
                host = f"http://{host}"
            if host.rstrip("/").endswith("/api/chat"):
                self._llm_endpoint = host.rstrip("/")
            else:
                self._llm_endpoint = host.rstrip("/") + "/api/chat"
        else:
            self._llm_endpoint = llm_endpoint

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, task_description: str, n: int) -> List[Example]:
        """Generate n GLiNER2-compatible training examples from a task description.

        The returned examples follow the JSONL training format described in the
        GLiNER2 docs, i.e. each example is a dict with:

            {
              "input": "<text>",
              "output": {
                "entities": {...},
                "classifications": [...],
                "relations": [...],
                "json_structures": [...]
              }
            }

        Only the fields relevant to the inferred task types are present in each
        example's `output` dict.
        """
        if n <= 0:
            return []

        parsed = self._infer_tasks(task_description)

        examples: List[Example] = []
        for idx in range(n):
            # Optional: delegate full example construction to an LLM. If anything
            # goes wrong (network issues, bad JSON, etc.), we silently fall back
            # to the template-based generators below for this example.
            if self._example_generation_mode == "llm":
                llm_example = self._generate_example_via_llm(
                    parsed=parsed,
                    description=task_description,
                    idx=idx,
                    total=n,
                )
                if llm_example is not None:
                    examples.append(llm_example)
                    continue

            text_pieces: List[str] = []
            output: OutputDict = {}

            if parsed.ner:
                ner_text, entities = self._generate_ner_example(
                    idx, entity_labels=parsed.ner_entity_labels
                )
                text_pieces.append(ner_text)
                output["entities"] = entities

            if parsed.classification:
                cls_text, classifications = self._generate_classification_example(
                    idx,
                    n,
                    task_name=parsed.classification_task_name,
                    labels=parsed.classification_labels,
                )
                text_pieces.append(cls_text)
                output["classifications"] = classifications

            if parsed.relation_extraction:
                re_text, relations = self._generate_relation_example(idx)
                text_pieces.append(re_text)
                output["relations"] = relations

            if parsed.json_extraction:
                json_text, json_structures = self._generate_json_example(idx)
                text_pieces.append(json_text)
                output["json_structures"] = json_structures

            if not text_pieces:
                # Fallback: treat as plain classification with generic labels.
                cls_text, classifications = self._generate_classification_example(
                    idx, n, task_name="generic_category"
                )
                text_pieces.append(cls_text)
                output["classifications"] = classifications

            example: Example = {
                "input": " ".join(text_pieces),
                "output": output,
            }
            examples.append(example)

        return examples

    # ------------------------------------------------------------------
    # Task inference
    # ------------------------------------------------------------------
    def _infer_tasks(self, description: str) -> ParsedTask:
        """Infer tasks using either rule-based or LLM-based inference.

        LLM-based inference is best-effort: if any error occurs while querying
        the model or parsing its response, the method falls back to the
        rule-based heuristic implementation.
        """
        if self._task_inference_mode == "llm":
            llm_parsed = self._infer_tasks_via_llm(description)
            if llm_parsed is not None:
                return self._enrich_ner_entity_labels(llm_parsed, description)

        # Default: rules (also used as a robust fallback).
        return self._enrich_ner_entity_labels(
            self._infer_tasks_rules(description), description
        )

    def _infer_tasks_rules(self, description: str) -> ParsedTask:
        desc = description.lower()
        parsed = ParsedTask()

        # NER
        if any(
            kw in desc
            for kw in [
                "extract entities",
                "extract company",
                "extract people",
                "extract organizations",
                "ner",
                "named entity",
                "highlight",
            ]
        ):
            parsed.ner = True

        # Simple heuristic: "extract X names" -> NER
        if re.search(r"extract [a-z\s]+names", desc):
            parsed.ner = True

        # Classification
        if any(
            kw in desc
            for kw in [
                "classify",
                "classification",
                "sentiment",
                "label as",
            ]
        ):
            parsed.classification = True

        # Relation extraction
        if any(
            kw in desc
            for kw in [
                "relation",
                "relationship",
                "works for",
                "lives in",
                "born in",
            ]
        ):
            parsed.relation_extraction = True

        # JSON / structured extraction
        if any(
            kw in desc
            for kw in [
                "json",
                "structured",
                "key-value",
                "fields:",
                "schema",
                "extract as",
            ]
        ):
            parsed.json_extraction = True

        # Sentiment-specific hints
        sentiment_hint = False

        # Explicit mention of "sentiment"
        if "sentiment" in desc:
            sentiment_hint = True

        # Descriptions that talk about tone/emotion with the canonical
        # sentiment label set but without using the word "sentiment",
        # e.g. "classify whether the tone is positive, negative or neutral".
        if (
            not sentiment_hint
            and all(lbl in desc for lbl in ["positive", "negative", "neutral"])
            and any(kw in desc for kw in ["classify", "classification", "tone", "emotion"])
        ):
            sentiment_hint = True

        if sentiment_hint:
            parsed.classification = True
            parsed.classification_task_name = "sentiment"
            parsed.classification_labels = ["positive", "negative", "neutral"]

        # Try to extract label lists like "into A, B and C" or "labels: A, B, C"
        if parsed.classification and parsed.classification_labels is None:
            label_candidates = self._extract_label_list(desc)
            if label_candidates:
                parsed.classification_labels = label_candidates

        # Default classification task meta-data if still unset
        if parsed.classification:
            if parsed.classification_task_name is None:
                parsed.classification_task_name = "classification"
            if parsed.classification_labels is None:
                parsed.classification_labels = ["class_a", "class_b", "class_c"]

        return parsed

    @staticmethod
    def _enrich_ner_entity_labels(parsed: ParsedTask, description: str) -> ParsedTask:
        """Populate ner_entity_labels based on the natural language description.

        This runs independently of how the main task flags were inferred (rules
        vs LLM) so that both paths benefit from the same lightweight hints.
        """
        if not parsed.ner:
            return parsed

        desc = description.lower()
        labels: List[str] = []

        # Company / organization-like entities
        if any(
            kw in desc
            for kw in [
                "company",
                "companies",
                "organization",
                "organisations",
                "organizations",
                "org name",
                "business name",
                "startup",
                "brands",
                "brand name",
            ]
        ):
            labels.append("company")

        # Person entities
        if any(
            kw in desc
            for kw in [
                "person",
                "people",
                "individual",
                "individuals",
                "employee names",
                "founder names",
                "ceo names",
            ]
        ):
            labels.append("person")

        # Location entities
        if any(
            kw in desc
            for kw in [
                "location",
                "locations",
                "city",
                "cities",
                "country",
                "countries",
                "place",
                "places",
            ]
        ):
            labels.append("location")

        # Only set when we have an explicit hint; keep the generic NER
        # behaviour otherwise.
        parsed.ner_entity_labels = labels or None
        return parsed

    def _infer_tasks_via_llm(self, description: str) -> Optional[ParsedTask]:
        """Infer tasks using a local LLM exposed via an Ollama HTTP endpoint.

        The LLM is prompted to return a strict JSON object with the same fields
        as `ParsedTask`. Any failure to contact the model or parse its response
        results in `None`, signalling that the caller should fall back to the
        rule-based implementation.
        """
        prompt = (
            "You are helping configure GLiNER2, a unified information "
            "extraction model that supports four task types:\n"
            "- ner\n- classification\n- relation_extraction\n- json_extraction\n\n"
            "Given the user's natural language task description below, decide "
            "which of these task types are required. Also infer, when "
            "appropriate, a short classification task name and a small set of "
            "classification labels.\n\n"
            "Return ONLY a JSON object with the following fields and no extra "
            "text:\n"
            '{\n'
            '  "ner": true | false,\n'
            '  "classification": true | false,\n'
            '  "relation_extraction": true | false,\n'
            '  "json_extraction": true | false,\n'
            '  "classification_task_name": string or null,\n'
            '  "classification_labels": array of strings or null\n'
            "}\n\n"
            "Task description:\n"
            f"{description}"
        )

        payload = {
            "model": self._llm_model,
            "messages": [
                {"role": "system", "content": "You respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._llm_endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError):
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        # Ollama's /api/chat returns the assistant content under message.content.
        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            return None

        try:
            parsed_json = json.loads(content)
        except json.JSONDecodeError:
            return None

        return self._parsed_task_from_dict(parsed_json)

    @staticmethod
    def _parsed_task_from_dict(data: Dict) -> ParsedTask | None:
        """Convert a loose dict into a well-typed ParsedTask instance."""
        try:
            parsed = ParsedTask(
                ner=bool(data.get("ner", False)),
                classification=bool(data.get("classification", False)),
                relation_extraction=bool(data.get("relation_extraction", False)),
                json_extraction=bool(data.get("json_extraction", False)),
                classification_task_name=data.get("classification_task_name"),
                classification_labels=data.get("classification_labels"),
            )
        except TypeError:
            return None

        # Normalise optional fields
        if not isinstance(parsed.classification_task_name, str):
            parsed.classification_task_name = None
        labels = parsed.classification_labels
        if isinstance(labels, list):
            parsed.classification_labels = [
                str(l).strip() for l in labels if isinstance(l, str) and l.strip()
            ] or None
        else:
            parsed.classification_labels = None

        # Reuse the same defaulting logic as the rule-based path.
        if parsed.classification:
            if parsed.classification_task_name is None:
                parsed.classification_task_name = "classification"
            if parsed.classification_labels is None:
                parsed.classification_labels = ["class_a", "class_b", "class_c"]

        return parsed

    @staticmethod
    def _extract_label_list(text: str) -> List[str] | None:
        # patterns like "into A, B and C"
        m = re.search(r"into ([a-z0-9_,\s]+)", text)
        if not m:
            m = re.search(r"labels?: ([a-z0-9_,\s]+)", text)
        if not m:
            return None

        label_block = m.group(1)
        # split on comma / "and"
        parts = re.split(r",| and ", label_block)
        labels = [p.strip() for p in parts if p.strip()]
        # keep short, sane labels
        return [l for l in labels if 1 <= len(l) <= 32] or None

    # ------------------------------------------------------------------
    # NER example generation
    # ------------------------------------------------------------------
    def _generate_ner_example(
        self, idx: int, entity_labels: Optional[List[str]] = None
    ) -> tuple[str, Dict[str, List[str]]]:
        people = [
            "Alice Johnson",
            "Bob Smith",
            "Carlos Diaz",
            "Diana Lee",
            "Emily Chen",
            "Farid Khan",
            "Grace Hopper",
            "Hannah Müller",
            "Ivan Petrov",
            "Julia Roberts",
        ]
        companies = [
            "Acme Corp",
            "Globex",
            "Innotech",
            "Fastino AI",
            "Quantum Labs",
            "NeuralForge",
            "Skyline Analytics",
            "Blue Ocean Systems",
            "Aurora Robotics",
            "Maple Leaf Software",
        ]
        locations = [
            "New York",
            "Berlin",
            "Tokyo",
            "San Francisco",
            "Toronto",
            "Singapore",
            "Sydney",
            "London",
            "Paris",
            "São Paulo",
        ]

        person = self._rng.choice(people)
        company = self._rng.choice(companies)
        location = self._rng.choice(locations)

        # Normalise requested labels, if any.
        requested = list(dict.fromkeys((entity_labels or [])))  # stable unique

        # Special-case: company-only extraction. Use sentences focused purely
        # on companies so the supervision aligns tightly with the requested
        # entity type.
        if requested and set(requested) == {"company"}:
            templates = [
                f"{company} reported record quarterly revenue this year.",
                f"After a successful funding round, {company} expanded its engineering team.",
                f"Analysts predict that {company} will outpace its competitors in the next quarter.",
                f"{company} is rolling out a new AI-powered analytics platform.",
                f"Shares of {company} rose sharply following the product launch.",
            ]
            text = self._rng.choice(templates)
            entities = {"company": [company]}
            return text, entities

        # Otherwise, fall back to the generic multi-entity templates but filter
        # the output entities to the requested subset if one was provided.
        templates = [
            f"{person} works as a data scientist at {company} in {location}.",
            f"{company}, based in {location}, recently hired {person} to lead a new project.",
            f"In {location}, {person} presented new research from {company}.",
            f"{person} joined {company} after relocating to {location} last year.",
            f"At a tech conference in {location}, {person} showcased {company}'s latest platform.",
            f"{company} opened a new office in {location}, where {person} now manages the team.",
        ]
        text = self._rng.choice(templates)

        all_entities: Dict[str, List[str]] = {
            "person": [person],
            "company": [company],
            "location": [location],
        }

        if requested:
            entities = {k: v for k, v in all_entities.items() if k in requested}
        else:
            entities = all_entities

        return text, entities

    # ------------------------------------------------------------------
    # LLM-based full example generation
    # ------------------------------------------------------------------
    def _generate_example_via_llm(
        self, parsed: ParsedTask, description: str, idx: int, total: int
    ) -> Optional[Example]:
        """Delegate construction of a single example to the local LLM.

        This is best-effort: any failure to contact the model or parse its
        response results in `None`, signalling the caller to fall back to the
        template-based generators.
        """
        # Maintain classification label balance by choosing the true label
        # deterministically, and pass this to the LLM as a hard requirement.
        labels = parsed.classification_labels or ["class_a", "class_b", "class_c"]
        true_label = labels[idx % len(labels)] if parsed.classification else None

        task_config = {
            "ner": parsed.ner,
            "ner_entity_labels": parsed.ner_entity_labels,
            "classification": parsed.classification,
            "classification_task_name": parsed.classification_task_name,
            "classification_labels": labels if parsed.classification else None,
            "classification_true_label": true_label,
            "relation_extraction": parsed.relation_extraction,
            "json_extraction": parsed.json_extraction,
        }

        prompt = (
            "You generate GLiNER2-compatible synthetic training examples.\n\n"
            "GLiNER2 expects each example to be a JSON object with:\n"
            '{\n'
            '  "input": "<text>",\n'
            '  "output": {\n'
            '    "entities": {"label": ["span", ...]},\n'
            '    "classifications": [\n'
            '      {\n'
            '        "task": "task name",\n'
            '        "labels": ["label1", ...],\n'
            '        "true_label": ["label1"]\n'
            "      }\n"
            "    ],\n"
            '    "relations": [\n'
            '      {"relation_name": {"head": "span", "tail": "span"}}\n'
            "    ],\n"
            '    "json_structures": [ { ... } ]\n'
            "  }\n"
            "}\n\n"
            "You will be given (1) the user's natural language task description "
            "and (2) a parsed task configuration. Use them to generate ONE "
            "realistic training example. Follow these rules strictly:\n"
            "- If ner=false, do not include an 'entities' field.\n"
            "- If ner=true and ner_entity_labels is provided, only include those "
            "labels in 'entities'.\n"
            "- If classification=false, do not include 'classifications'.\n"
            "- If classification=true, you MUST use the provided "
            "'classification_task_name', 'classification_labels', and set "
            "'true_label' to exactly the provided classification_true_label.\n"
            "- If relation_extraction=false, omit 'relations'.\n"
            "- If json_extraction=false, omit 'json_structures'.\n"
            "- The 'input' text must be consistent with the structured output.\n\n"
            "Return ONLY the JSON object, with no extra commentary.\n\n"
            "User task description:\n"
            f"{description}\n\n"
            "Parsed task configuration (JSON):\n"
            f"{json.dumps(task_config, ensure_ascii=False)}\n"
        )

        payload = {
            "model": self._llm_model,
            "messages": [
                {"role": "system", "content": "You respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._llm_endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError):
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            return None

        try:
            parsed_json = json.loads(content)
        except json.JSONDecodeError:
            return None

        # Basic structural validation.
        if not isinstance(parsed_json, dict):
            return None

        input_text = parsed_json.get("input")
        output = parsed_json.get("output")
        if not isinstance(input_text, str) or not isinstance(output, dict):
            return None

        # Optionally, enforce the chosen true_label if classification is active.
        if parsed.classification and true_label is not None:
            classifications = output.get("classifications")
            if isinstance(classifications, list) and classifications:
                first = classifications[0]
                if isinstance(first, dict):
                    first["task"] = parsed.classification_task_name or "classification"
                    first["labels"] = labels
                    first["true_label"] = [true_label]
                    output["classifications"] = [first]

        example: Example = {
            "input": input_text,
            "output": output,
        }
        return example

    # ------------------------------------------------------------------
    # Classification example generation
    # ------------------------------------------------------------------
    def _generate_classification_example(
        self,
        idx: int,
        total: int,
        task_name: str | None = None,
        labels: List[str] | None = None,
    ) -> tuple[str, List[Dict]]:
        task_name = task_name or "classification"
        labels = labels or ["class_a", "class_b", "class_c"]

        # Enforce approximate balance by cycling through labels.
        true_label = labels[idx % len(labels)]

        if task_name == "sentiment" and true_label in {"positive", "negative", "neutral"}:
            text = self._build_sentiment_text(true_label)
        else:
            text = self._build_generic_classification_text(task_name, true_label)

        classifications = [
            {
                "task": task_name,
                "labels": labels,
                "true_label": [true_label],
            }
        ]
        return text, classifications

    def _build_sentiment_text(self, label: str) -> str:
        """Compose a sentiment sentence from several small banks.

        This yields a large combinatorial space of unique sentences while
        keeping the semantics aligned with the target label.
        """
        subjects = [
            "The mobile app",
            "Customer support",
            "The latest software update",
            "The delivery service",
            "The pricing model",
            "The onboarding experience",
            "The website checkout flow",
            "The data export feature",
        ]
        positive_outcomes = [
            "worked flawlessly and exceeded my expectations",
            "was incredibly smooth from start to finish",
            "saved me a lot of time every day",
            "felt intuitive even for new team members",
            "made our internal workflow much faster",
            "replied quickly and solved my problem",
            "offers great value for the subscription price",
            "has become an essential tool for our team",
        ]
        negative_outcomes = [
            "kept crashing in the middle of important work",
            "was confusing and full of unexpected errors",
            "made the checkout process painfully slow",
            "left my support ticket unanswered for days",
            "introduced several bugs that blocked our release",
            "felt overpriced for the limited functionality",
            "regularly lost my configuration changes",
            "made it hard to complete even basic tasks",
        ]
        neutral_outcomes = [
            "behaved largely as advertised without surprises",
            "was acceptable but nothing particularly special",
            "matched the standard we see in similar tools",
            "delivered the expected functionality on time",
            "required a short learning period but then felt routine",
            "followed our usual internal approval process",
            "met the requirements specified in the contract",
            "ran for several weeks without noticeable incidents",
        ]

        subject = self._rng.choice(subjects)
        if label == "positive":
            outcome = self._rng.choice(positive_outcomes)
        elif label == "negative":
            outcome = self._rng.choice(negative_outcomes)
        else:
            outcome = self._rng.choice(neutral_outcomes)

        return f"{subject} {outcome}."

    def _build_generic_classification_text(self, task_name: str, true_label: str) -> str:
        """Generic but varied text for non-sentiment classification tasks."""
        contexts = [
            "In this scenario, the example should be labeled as",
            "For the following case, assign the label",
            "Treat this short description as belonging to category",
            "For the configured task, the correct label is",
            "When training the model, mark this example with",
        ]
        domains = [
            "a support ticket",
            "a user question",
            "a log line from a backend service",
            "a short product description",
            "an internal status update",
            "a short customer email",
            "a bug report summary",
        ]

        context = self._rng.choice(contexts)
        domain = self._rng.choice(domains)
        return (
            f"{context} '{true_label}' for the {task_name} task. "
            f"The text describes {domain} in a realistic setting."
        )

    # ------------------------------------------------------------------
    # Relation extraction example generation
    # ------------------------------------------------------------------
    def _generate_relation_example(self, idx: int) -> tuple[str, List[Dict]]:
        people = [
            "John",
            "Maria",
            "Ethan",
            "Sara",
            "Nina",
            "Liam",
            "Olivia",
            "Noah",
        ]
        companies = [
            "Apple Inc.",
            "Fastino AI",
            "Globex",
            "Innotech",
            "NeuralForge",
            "Aurora Robotics",
            "Blue Ocean Systems",
        ]
        cities = [
            "London",
            "Paris",
            "Toronto",
            "San Francisco",
            "Berlin",
            "New York",
            "Singapore",
        ]

        person = self._rng.choice(people)
        company = self._rng.choice(companies)
        city = self._rng.choice(cities)

        templates = [
            f"{person} works for {company} and currently lives in {city}. "
            f"In the past, {person} also collaborated with several startups.",
            f"After moving to {city}, {person} accepted an offer from {company}.",
            f"{company} recently transferred {person} to its {city} office.",
            f"{person} has been living in {city} ever since joining {company}.",
        ]
        text = self._rng.choice(templates)

        relations = [
            {"works_for": {"head": person, "tail": company}},
            {"lives_in": {"head": person, "tail": city}},
        ]
        return text, relations

    # ------------------------------------------------------------------
    # JSON extraction example generation
    # ------------------------------------------------------------------
    def _generate_json_example(self, idx: int) -> tuple[str, List[Dict]]:
        products = [
            ("iPhone 15 Pro Max", "256GB", "$1199"),
            ("Pixel 9", "128GB", "$899"),
            ("Galaxy S25", "512GB", "$1399"),
            ("ThinkPad X1 Carbon", "16GB RAM / 1TB SSD", "$1899"),
            ("MacBook Air M4", "8GB RAM / 512GB SSD", "$1499"),
            ("Surface Pro 11", "16GB RAM / 512GB SSD", "$1699"),
            ("Kindle Paperwhite Pro", "32GB", "$249"),
            ("Oculus Quest Ultra", "256GB", "$699"),
            ("Apple Watch Series 11", "128GB", "$499"),
        ]
        name, storage, price = self._rng.choice(products)

        text = (
            f"{name} comes with {storage} of storage and has a starting "
            f"price of {price} in most markets."
        )

        structure = {
            "product": {
                "name": name,
                "storage": storage,
                "price": price,
            }
        }

        return text, [structure]

