from __future__ import annotations
import json
import os
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal, TypedDict

class OutputDict(TypedDict, total=False):
    entities: dict[str, list[str]]
    classifications: list[dict]
    relations: list[dict]
    json_structures: list[dict]

class Example(TypedDict):
    input: str
    output: OutputDict

@dataclass
class ParsedTask:
    """Active task types and their associated metadata, inferred from a description."""
    ner: bool = False
    classification: bool = False
    relation_extraction: bool = False
    json_extraction: bool = False
    classification_task_name: str | None = None
    classification_labels: list[str] | None = None
    ner_entity_labels: list[str] | None = None

TaskInferenceMode = Literal["rules", "llm"]
ExampleGenerationMode = Literal["templates", "llm"]

class DataGenerator:
    # Template banks used by the sub-generators.
    _PEOPLE = [
        "Alice Johnson", "Bob Smith", "Carlos Diaz", "Diana Lee", "Emily Chen",
        "Farid Khan", "Grace Hopper", "Hannah Müller", "Ivan Petrov", "Julia Roberts",
    ]
    _COMPANIES = [
        "Acme Corp", "Globex", "Innotech", "Fastino AI", "Quantum Labs",
        "NeuralForge", "Skyline Analytics", "Blue Ocean Systems", "Aurora Robotics",
        "Maple Leaf Software",
    ]
    _LOCATIONS = [
        "New York", "Berlin", "Tokyo", "San Francisco", "Toronto",
        "Singapore", "Sydney", "London", "Paris", "São Paulo",
    ]
    _PRODUCTS = [
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
    _SENTIMENT_SUBJECTS = [
        "The mobile app", "Customer support", "The latest software update",
        "The delivery service", "The pricing model", "The onboarding experience",
        "The website checkout flow", "The data export feature",
    ]
    _SENTIMENT_OUTCOMES: dict[str, list[str]] = {
        "positive": [
            "worked flawlessly and exceeded my expectations",
            "was incredibly smooth from start to finish",
            "saved me a lot of time every day",
            "felt intuitive even for new team members",
            "made our internal workflow much faster",
            "replied quickly and solved my problem",
            "offers great value for the subscription price",
            "has become an essential tool for our team",
        ],
        "negative": [
            "kept crashing in the middle of important work",
            "was confusing and full of unexpected errors",
            "made the checkout process painfully slow",
            "left my support ticket unanswered for days",
            "introduced several bugs that blocked our release",
            "felt overpriced for the limited functionality",
            "regularly lost my configuration changes",
            "made it hard to complete even basic tasks",
        ],
        "neutral": [
            "behaved largely as advertised without surprises",
            "was acceptable but nothing particularly special",
            "matched the standard we see in similar tools",
            "delivered the expected functionality on time",
            "required a short learning period but then felt routine",
            "followed our usual internal approval process",
            "met the requirements specified in the contract",
            "ran for several weeks without noticeable incidents",
        ],
    }

    def __init__(
        self,
        seed: int | None = None,
        task_inference_mode: TaskInferenceMode = "rules",
        example_generation_mode: ExampleGenerationMode = "templates",
        llm_model: str = "llama3.2",
        llm_endpoint: str | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._task_inference_mode = task_inference_mode
        self._example_generation_mode = example_generation_mode
        self._llm_model = llm_model
        # Resolve the Ollama /api/chat endpoint, preferring OLLAMA_HOST env var.
        if llm_endpoint is None:
            host = os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
            if not host.startswith(("http://", "https://")):
                host = f"http://{host}"
            base = host.rstrip("/")
            self._llm_endpoint = base if base.endswith("/api/chat") else base + "/api/chat"
        else:
            self._llm_endpoint = llm_endpoint

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def generate(self, task_description: str, n: int) -> list[Example]:
        """Generate GLiNER2-compatible synthetic training examples from a task description.
        Task types are inferred automatically; outputs for multiple active types are
        merged into a single example dict. Inference and generation both default to
        lightweight rules/templates; both can be upgraded to a local Ollama LLM
        with automatic fallback to the rule-based path on any failure."""

        if n <= 0: return []
        parsed = self._infer_tasks_via_llm(task_description) if self._task_inference_mode == "llm" else None
        if parsed is None:
            parsed = self._infer_tasks_rules(task_description)
        examples: list[Example] = []

        for idx in range(n):
            if self._example_generation_mode == "llm":
                llm_example = self._generate_example_via_llm(parsed, task_description, idx)
                if llm_example is not None:
                    examples.append(llm_example)
                    continue

            text_pieces: list[str] = []
            output: OutputDict = {}
        
            if parsed.ner:
                text, entities = self._generate_ner_example(parsed.ner_entity_labels)
                text_pieces.append(text)
                output["entities"] = entities

            if parsed.classification:
                text, classifications = self._generate_classification_example(idx)
                text_pieces.append(text)
                output["classifications"] = classifications

            if parsed.relation_extraction:
                text, relations = self._generate_relation_example()
                text_pieces.append(text)
                output["relations"] = relations

            if parsed.json_extraction:
                text, json_structures = self._generate_json_example()
                text_pieces.append(text)
                output["json_structures"] = json_structures

            if not text_pieces:
                raise ValueError(
                    "No task type could be inferred from the description. "
                    "Provide a clearer description or use task_inference_mode='llm'."
                )
            examples.append({"input": " ".join(text_pieces), "output": output})
        return examples

    # ------------------------------------------------------------------ #
    # Task inference                                                       #
    # ------------------------------------------------------------------ #

    def _infer_tasks_rules(self, description: str) -> ParsedTask:
        """Rules-based inference. Template generation supports only:
        - NER: company, person, location entities
        - Classification: sentiment (positive / negative / neutral)
        - Relation extraction: works_for, lives_in
        - JSON extraction: product schema (name, storage, price)
        Use LLM mode for anything outside these constraints."""
    
        desc = description.lower()
        parsed = ParsedTask()

        # NER — entity type words must appear in an extraction context (alongside
        # "name(s)" or a generic NER keyword) to avoid false-positives when those
        # words appear incidentally, e.g. "relations between people and companies".
        # The template mode supports only company, person, and location entities.
        ner_generic = bool(re.search(
            r"\bner\b|\bnamed[\s-]entit(?:y|ies)\b|\bannotate\b|\bhighlight\b|\btag(?:s|ging)?\b",
            desc,
        ))
        ner_context = ner_generic or bool(re.search(r"\bnames?\b", desc))
        ner_labels: list[str] = []
        if ner_context:
            if re.search(
                r"\bcompan(?:y|ies)\b|\borgani[sz]ations?\b"
                r"|\borg\s+name\b|\bbusiness\s+name\b|\bstartup\b|\bbrands?\b",
                desc,
            ):
                ner_labels.append("company")
            if re.search(
                r"\bpe(?:rson|ople)\b|\bindividuals?\b|\b(?:employee|founder|ceo)\s+names?\b",
                desc,
            ):
                ner_labels.append("person")
            if re.search(r"\blocations?\b|\bcit(?:y|ies)\b|\bcountr(?:y|ies)\b|\bplaces?\b", desc):
                ner_labels.append("location")
        if ner_labels or ner_generic:
            parsed.ner = True
            parsed.ner_entity_labels = ner_labels or None  # None → all three entity types

        # Classification — template mode supports sentiment only.
        if bool(re.search(r"\bsentiment\b", desc)) or (
            all(re.search(rf"\b{lbl}\b", desc) for lbl in ["positive", "negative", "neutral"])
            and bool(re.search(r"\b(classif\w*|tone|emotion)\b", desc))
        ):
            parsed.classification = True
            parsed.classification_task_name = "sentiment"
            parsed.classification_labels = ["positive", "negative", "neutral"]

        # Relation extraction — template mode supports works_for and lives_in only.
        if re.search(r"\brelations?\b|\brelationship\w*\b|\bworks\s+for\b|\blives\s+in\b|\bborn\s+in\b", desc):
            parsed.relation_extraction = True

        # JSON extraction — template mode supports the product schema only.
        if re.search(r"\bjson\b|\bkey[-\s]value\b|\bschema\b|\bextract\s+as\b", desc):
            parsed.json_extraction = True
        return parsed

    def _infer_tasks_via_llm(self, description: str) -> ParsedTask | None:
        """Ask the local LLM to infer task types. Returns None on any failure."""
        prompt = (
            "You are configuring GLiNER2, a unified information extraction model.\n\n"
            "Given a task description, return a JSON object specifying which task types "
            "are needed and their configuration. Fields:\n"
            '  "ner": true if entity extraction is needed\n'
            '  "ner_entity_labels": list of entity type strings (e.g. ["company", "person"]), or null\n'
            '  "classification": true if text classification is needed\n'
            '  "classification_task_name": short snake_case name for the task, or null\n'
            '  "classification_labels": list of class label strings, or null\n'
            '  "relation_extraction": true if relation extraction is needed\n'
            '  "json_extraction": true if structured JSON field extraction is needed\n\n'
            "Example:\n"
            'Description: "Extract company and person names, and classify sentiment as positive, negative, or neutral."\n'
            'Response: {"ner": true, "ner_entity_labels": ["company", "person"], "classification": true, '
            '"classification_task_name": "sentiment", "classification_labels": ["positive", "negative", "neutral"], '
            '"relation_extraction": false, "json_extraction": false}\n\n'
            f"Description: {description!r}\n"
            "Response:"
        )
        content = self._call_llm(
            [{"role": "system", "content": "You respond with a single JSON object only. No explanation."},
             {"role": "user", "content": prompt}],
            timeout=30,
        )
        if content is None: return None
        data = self._parse_json_response(content)
        return self._parsed_task_from_dict(data) if data is not None else None

    @staticmethod
    def _parse_json_response(content: str) -> dict | None:
        """Parse a JSON object from an LLM response, stripping markdown code fences if present."""
        text = re.sub(r"^```(?:json)?\s*\n?", "", content.strip())
        text = re.sub(r"\n?```\s*$", "", text.strip())
        try:
            result = json.loads(text)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _parsed_task_from_dict(data: dict) -> ParsedTask | None:
        """Convert a loose dict (e.g. from LLM JSON) into a ParsedTask, or None on error."""
        try:
            parsed = ParsedTask(
                ner=bool(data.get("ner", False)),
                classification=bool(data.get("classification", False)),
                relation_extraction=bool(data.get("relation_extraction", False)),
                json_extraction=bool(data.get("json_extraction", False)),
                classification_task_name=data.get("classification_task_name"),
                classification_labels=data.get("classification_labels"),
                ner_entity_labels=data.get("ner_entity_labels"),
            )
        except TypeError:
            return None

        # Data validation and sanitization.
        if not isinstance(parsed.classification_task_name, str):
            parsed.classification_task_name = None
        labels = parsed.classification_labels
        parsed.classification_labels = (
            [str(l).strip() for l in labels if isinstance(l, str) and l.strip()] or None
        ) if isinstance(labels, list) else None
        ner_labels = parsed.ner_entity_labels
        parsed.ner_entity_labels = (
            [str(l).strip() for l in ner_labels if isinstance(l, str) and l.strip()] or None
        ) if isinstance(ner_labels, list) else None

        if parsed.classification:
            parsed.classification_task_name = parsed.classification_task_name or "classification"
            parsed.classification_labels = parsed.classification_labels or ["class_a", "class_b", "class_c"]
        return parsed

    def _call_llm(self, messages: list[dict], timeout: int = 30) -> str | None:
        """POST to the Ollama /api/chat endpoint and return the assistant content string.
        Returns None on any network error, timeout, or malformed response."""

        body = json.dumps({"model": self._llm_model, "messages": messages, "stream": False}).encode()
        req = urllib.request.Request(
            self._llm_endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None
        content = (data.get("message") or {}).get("content")
        return content if isinstance(content, str) else None

    # ------------------------------------------------------------------ #
    # Example generation                                                   #
    # ------------------------------------------------------------------ #

    def _generate_ner_example(
        self, entity_labels: list[str] | None = None
    ) -> tuple[str, dict[str, list[str]]]:
        person = self._rng.choice(self._PEOPLE)
        company = self._rng.choice(self._COMPANIES)
        location = self._rng.choice(self._LOCATIONS)

        # Deduplicate while preserving order.
        requested = list(dict.fromkeys(entity_labels or []))

        # For single-entity-type requests use focused templates so the text does
        # not mention unannotated entities from the other categories.
        if requested and set(requested) == {"company"}:
            templates = [
                f"{company} reported record quarterly revenue this year.",
                f"After a successful funding round, {company} expanded its engineering team.",
                f"Analysts predict that {company} will outpace its competitors in the next quarter.",
                f"{company} is rolling out a new AI-powered analytics platform.",
                f"Shares of {company} rose sharply following the product launch.",
            ]
            return self._rng.choice(templates), {"company": [company]}

        if requested and set(requested) == {"person"}:
            templates = [
                f"{person} presented findings at an industry conference this week.",
                f"The board announced that {person} would be joining as the new VP of Engineering.",
                f"Our sources indicate that {person} has been leading the project since January.",
                f"Attendees praised {person}'s keynote on machine learning in production.",
                f"{person} was quoted saying the team exceeded all targets this quarter.",
            ]
            return self._rng.choice(templates), {"person": [person]}

        if requested and set(requested) == {"location"}:
            templates = [
                f"The new regional office will be established in {location} by next year.",
                f"Our operations team confirmed the deployment was rolled out from {location}.",
                f"The annual conference is scheduled to take place in {location} this autumn.",
                f"Infrastructure upgrades are currently underway at the {location} data centre.",
                f"The pilot programme launched successfully across {location} last month.",
            ]
            return self._rng.choice(templates), {"location": [location]}

        if requested and set(requested) == {"person", "company"}:
            templates = [
                f"{person} was appointed VP of Engineering at {company} last quarter.",
                f"{company} announced that {person} would lead its new research division.",
                f"After five years at a rival firm, {person} joined {company} as a senior architect.",
                f"{person} was recognised as employee of the year at {company}.",
                f"The CEO of {company} credited {person}'s work for the quarter's strong results.",
            ]
            return self._rng.choice(templates), {"person": [person], "company": [company]}

        if requested and set(requested) == {"person", "location"}:
            templates = [
                f"{person} has relocated to {location} to head up a regional initiative.",
                f"Based in {location}, {person} leads a distributed engineering team.",
                f"{person} was the keynote speaker at a summit held in {location} last month.",
                f"Sources confirm that {person} has been working remotely from {location}.",
                f"{person} moved to {location} after completing the programme.",
            ]
            return self._rng.choice(templates), {"person": [person], "location": [location]}

        if requested and set(requested) == {"company", "location"}:
            templates = [
                f"{company} announced the opening of a new office in {location}.",
                f"The {location} branch of {company} has doubled in size over the past year.",
                f"{company} is relocating its regional headquarters to {location} next spring.",
                f"Following strong demand, {company} expanded its operations into {location}.",
                f"{company} acquired a firm based in {location} last quarter.",
            ]
            return self._rng.choice(templates), {"company": [company], "location": [location]}

        templates = [
            f"{person} works as a data scientist at {company} in {location}.",
            f"{company}, based in {location}, recently hired {person} to lead a new project.",
            f"In {location}, {person} presented new research from {company}.",
            f"{person} joined {company} after relocating to {location} last year.",
            f"At a tech conference in {location}, {person} showcased {company}'s latest platform.",
            f"{company} opened a new office in {location}, where {person} now manages the team.",
        ]
        all_entities: dict[str, list[str]] = {
            "person": [person], "company": [company], "location": [location],
        }
        entities = {k: v for k, v in all_entities.items() if k in requested} if requested else all_entities
        return self._rng.choice(templates), entities

    def _generate_classification_example(self, idx: int) -> tuple[str, list[dict]]:
        labels = ["positive", "negative", "neutral"]
        true_label = labels[idx % len(labels)]
        subject = self._rng.choice(self._SENTIMENT_SUBJECTS)
        outcome = self._rng.choice(self._SENTIMENT_OUTCOMES[true_label])
        text = f"{subject} {outcome}."
        return text, [{"task": "sentiment", "labels": labels, "true_label": [true_label]}]

    def _generate_relation_example(self) -> tuple[str, list[dict]]:
        person = self._rng.choice(self._PEOPLE)
        company = self._rng.choice(self._COMPANIES)
        city = self._rng.choice(self._LOCATIONS)
        templates = [
            f"{person} works for {company} and currently lives in {city}. "
            f"In the past, {person} also collaborated with several startups.",
            f"After moving to {city}, {person} accepted an offer from {company}.",
            f"{person} relocated to {city} to take on a new role at {company}.",
            f"{person} has been living in {city} ever since joining {company}.",
        ]
        relations = [
            {"works_for": {"head": person, "tail": company}},
            {"lives_in": {"head": person, "tail": city}},
        ]
        return self._rng.choice(templates), relations

    def _generate_json_example(self) -> tuple[str, list[dict]]:
        name, storage, price = self._rng.choice(self._PRODUCTS)
        text = f"{name} comes with {storage} of storage and has a starting price of {price} in most markets."
        return text, [{"product": {"name": name, "storage": storage, "price": price}}]

    def _generate_example_via_llm(
        self, parsed: ParsedTask, description: str, idx: int
    ) -> Example | None:
        """Delegate construction of a single example to the local LLM.

        The true_label for classification is chosen deterministically to maintain
        balance across the batch. Returns None on any failure.
        """
        labels = parsed.classification_labels or ["class_a", "class_b", "class_c"]
        true_label = labels[idx % len(labels)] if parsed.classification else None

        # Build a concrete output schema showing exactly what the LLM must produce.
        output_schema_parts: list[str] = []
        if parsed.ner:
            entity_labels = parsed.ner_entity_labels or ["<entity_type>"]
            output_schema_parts.append(
                '"entities": {'
                + ", ".join(f'"{l}": ["<span verbatim from input>"]' for l in entity_labels)
                + "}"
            )
        if parsed.classification:
            output_schema_parts.append(
                f'"classifications": [{{"task": "{parsed.classification_task_name}", '
                f'"labels": {json.dumps(labels)}, "true_label": ["{true_label}"]}}]'
            )
        if parsed.relation_extraction:
            output_schema_parts.append(
                '"relations": [{"<relation_type>": {"head": "<span verbatim from input>", '
                '"tail": "<span verbatim from input>"}}]'
            )
        if parsed.json_extraction:
            output_schema_parts.append(
                '"json_structures": [{"<schema_name>": {"<field>": "<value from input>"}}]'
            )

        constraints: list[str] = [
            '- Return exactly one JSON object with keys "input" and "output".',
            "- Omit output keys for inactive tasks.",
            "- All entity spans, relation heads, and relation tails must appear verbatim in the input text.",
        ]
        if parsed.ner and parsed.ner_entity_labels:
            constraints.append(
                f"- entities must use exactly these label keys: {json.dumps(parsed.ner_entity_labels)}."
            )
        if parsed.classification:
            constraints.append(
                f'- true_label must be exactly ["{true_label}"] — do not change it.'
            )
            constraints.append(f"- labels must be exactly {json.dumps(labels)}.")
        if parsed.relation_extraction:
            constraints.append(
                '- Each relation entry must have the form {"<relation_name>": {"head": "<str>", "tail": "<str>"}}.'
            )

        prompt = (
            f"Task: {description}\n\n"
            "Generate one GLiNER2 training example matching this exact output structure:\n"
            '{{"input": "<text>", "output": {{{output}}}}}\n\n'.format(
                output=", ".join(output_schema_parts)
            )
            + "Constraints:\n"
            + "\n".join(constraints)
            + "\n\nReturn ONLY the JSON object."
        )
        content = self._call_llm(
            [{"role": "system", "content": "You respond with a single JSON object only. No explanation."},
             {"role": "user", "content": prompt}],
            timeout=60,
        )
        if content is None:
            return None
        result = self._parse_json_response(content)
        if result is None:
            return None

        input_text = result.get("input")
        output = result.get("output")
        if not isinstance(input_text, str) or not input_text.strip() or not isinstance(output, dict):
            return None

        # Enforce classification: overwrite task/labels/true_label regardless of LLM output.
        if parsed.classification and true_label is not None:
            clss = output.get("classifications")
            enforced = {
                "task": parsed.classification_task_name or "classification",
                "labels": labels,
                "true_label": [true_label],
            }
            if isinstance(clss, list) and clss and isinstance(clss[0], dict):
                clss[0].update(enforced)
            else:
                output["classifications"] = [enforced]

        # Enforce NER: remove any entity label not in the requested set.
        if parsed.ner:
            entities = output.get("entities")
            if isinstance(entities, dict) and parsed.ner_entity_labels is not None:
                output["entities"] = {
                    k: v for k, v in entities.items() if k in parsed.ner_entity_labels
                }
            elif not isinstance(entities, dict):
                output["entities"] = {}

        # Enforce relations: drop any entry that lacks a valid {head, tail} structure.
        if parsed.relation_extraction:
            relations = output.get("relations")
            if isinstance(relations, list):
                output["relations"] = [
                    {rel: {"head": val["head"], "tail": val["tail"]}}
                    for r in relations if isinstance(r, dict)
                    for rel, val in r.items()
                    if isinstance(val, dict)
                    and isinstance(val.get("head"), str)
                    and isinstance(val.get("tail"), str)
                ]
            else:
                output["relations"] = []

        # Enforce json_structures: must be a list of dicts.
        if parsed.json_extraction:
            js = output.get("json_structures")
            if not isinstance(js, list) or not all(isinstance(x, dict) for x in js):
                output["json_structures"] = []

        return {"input": input_text, "output": output}
