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
    ner_entity_labels: list[str] | None = None  # entity types explicitly requested


TaskInferenceMode = Literal["rules", "llm"]
ExampleGenerationMode = Literal["templates", "llm"]


class DataGenerator:
    """Generate GLiNER2-compatible synthetic training examples from a task description.

    Task types are inferred automatically; outputs for multiple active types are
    merged into a single example dict. Inference and generation both default to
    lightweight rules/templates; both can be upgraded to a local Ollama LLM
    with automatic fallback to the rule-based path on any failure.
    """

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
    _REL_PEOPLE = ["John", "Maria", "Ethan", "Sara", "Nina", "Liam", "Olivia", "Noah"]
    _REL_COMPANIES = [
        "Apple Inc.", "Fastino AI", "Globex", "Innotech",
        "NeuralForge", "Aurora Robotics", "Blue Ocean Systems",
    ]
    _REL_CITIES = [
        "London", "Paris", "Toronto", "San Francisco", "Berlin", "New York", "Singapore",
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
    # Realistic domain texts for non-sentiment classification tasks.
    _GENERIC_TEXTS = [
        "The latest software update introduced a new dashboard with improved navigation and real-time analytics.",
        "After weeks of testing, the integration was deployed to production without major incidents.",
        "Customer feedback indicated a strong preference for the simplified checkout flow.",
        "The quarterly review highlighted bottlenecks in the data pipeline that need addressing.",
        "A new API endpoint was added to support bulk data exports from the platform.",
        "The support queue grew significantly following the announcement of the new pricing tiers.",
        "Our internal metrics show a 12% improvement in response time after the cache layer was added.",
        "The onboarding documentation was updated to reflect the latest changes to the admin panel.",
        "Several edge cases in the authentication flow were identified during the security audit.",
        "The marketing campaign generated a high volume of sign-ups over the weekend.",
        "Performance degraded noticeably under peak load conditions during the regional outage.",
        "The new mobile release resolved a long-standing issue with offline sync.",
        "Compliance requirements prompted a full review of how user data is stored and retained.",
        "Team capacity is currently stretched due to the upcoming product launch deadline.",
        "The infrastructure team proposed migrating to a managed Kubernetes service next quarter.",
        "Users reported that the search feature consistently returns irrelevant results for long queries.",
        "The billing system failed to apply the promotional discount during the campaign window.",
        "A refactor of the notification module reduced average email delivery latency by 40%.",
        "The access-control policy was tightened following a review of privilege escalation risks.",
        "Stakeholders approved the roadmap for the next two quarters at the planning session.",
    ]

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
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate(self, task_description: str, n: int) -> list[Example]:
        """Generate n GLiNER2-compatible training examples from a task description.

        Task types are inferred automatically; when multiple are active their
        output fields are merged into a single output dict per example.
        """
        if n <= 0:
            return []

        parsed = self._infer_tasks(task_description)
        examples: list[Example] = []

        for idx in range(n):
            # In LLM generation mode, delegate to the model and fall through to
            # the template path only if the call fails.
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
                text, classifications = self._generate_classification_example(
                    idx, parsed.classification_task_name, parsed.classification_labels
                )
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
                # No task was inferred; fall back to generic classification.
                text, classifications = self._generate_classification_example(
                    idx, task_name="generic_category"
                )
                text_pieces.append(text)
                output["classifications"] = classifications

            examples.append({"input": " ".join(text_pieces), "output": output})

        return examples

    # ------------------------------------------------------------------ #
    # Task inference                                                       #
    # ------------------------------------------------------------------ #

    def _infer_tasks(self, description: str) -> ParsedTask:
        """Dispatch to LLM or rule-based inference, then enrich NER entity labels."""
        if self._task_inference_mode == "llm":
            result = self._infer_tasks_via_llm(description)
            if result is not None:
                return self._enrich_ner_entity_labels(result, description)
        return self._enrich_ner_entity_labels(self._infer_tasks_rules(description), description)

    def _infer_tasks_rules(self, description: str) -> ParsedTask:
        desc = description.lower()
        parsed = ParsedTask()

        if any(kw in desc for kw in [
            "extract entities", "extract company", "extract people", "extract organizations",
            "ner", "named entity", "highlight",
            "identify entities", "find entities", "tag entities", "tag names",
            "annotate entities", "recognize entities", "detect entities", "locate entities",
            "label entities", "identify names", "find names", "annotate",
        ]):
            parsed.ner = True
        if re.search(r"(extract|identify|find|tag|annotate|recognize|detect) [a-z\s]+names", desc):
            parsed.ner = True

        if any(kw in desc for kw in [
            "classify", "classification", "sentiment", "label as", "categorize", "categorise",
        ]):
            parsed.classification = True

        if any(kw in desc for kw in ["relation", "relationship", "works for", "lives in", "born in"]):
            parsed.relation_extraction = True

        if any(kw in desc for kw in ["json", "structured", "key-value", "fields:", "schema", "extract as"]):
            parsed.json_extraction = True

        # Detect sentiment: explicit keyword, or canonical label set plus tone/emotion word.
        is_sentiment = "sentiment" in desc or (
            all(lbl in desc for lbl in ["positive", "negative", "neutral"])
            and any(kw in desc for kw in ["classify", "classification", "tone", "emotion"])
        )
        if is_sentiment:
            parsed.classification = True
            parsed.classification_task_name = "sentiment"
            parsed.classification_labels = ["positive", "negative", "neutral"]

        # Try to extract an explicit label list, e.g. "into A, B and C" or "labels: A, B, C".
        if parsed.classification and parsed.classification_labels is None:
            parsed.classification_labels = self._extract_label_list(desc)

        # Fill in defaults for any classification metadata still unset.
        if parsed.classification:
            parsed.classification_task_name = parsed.classification_task_name or "classification"
            parsed.classification_labels = parsed.classification_labels or ["class_a", "class_b", "class_c"]

        return parsed

    @staticmethod
    def _enrich_ner_entity_labels(parsed: ParsedTask, description: str) -> ParsedTask:
        """Populate ner_entity_labels from entity-type keywords in the description.

        Runs after both the rule and LLM paths so both benefit from the same hints.
        """
        if not parsed.ner:
            return parsed

        desc = description.lower()
        labels: list[str] = []

        if any(kw in desc for kw in [
            "company", "companies", "organization", "organisations", "organizations",
            "org name", "business name", "startup", "brands", "brand name",
        ]):
            labels.append("company")

        if any(kw in desc for kw in [
            "person", "people", "individual", "individuals",
            "employee names", "founder names", "ceo names",
        ]):
            labels.append("person")

        if any(kw in desc for kw in [
            "location", "locations", "city", "cities", "country", "countries", "place", "places",
        ]):
            labels.append("location")

        parsed.ner_entity_labels = labels or None
        return parsed

    def _infer_tasks_via_llm(self, description: str) -> ParsedTask | None:
        """Ask the local LLM to infer task types. Returns None on any failure."""
        prompt = (
            "You are helping configure GLiNER2, a unified information extraction model "
            "that supports four task types:\n"
            "- ner\n- classification\n- relation_extraction\n- json_extraction\n\n"
            "Given the user's task description, decide which types are required and infer "
            "a classification task name and labels when appropriate.\n\n"
            "Return ONLY a JSON object with these fields:\n"
            '{"ner": bool, "classification": bool, "relation_extraction": bool, '
            '"json_extraction": bool, "classification_task_name": str|null, '
            '"classification_labels": list|null}\n\n'
            f"Task description:\n{description}"
        )
        content = self._call_llm(
            [{"role": "system", "content": "You respond with JSON only."},
             {"role": "user", "content": prompt}],
            timeout=30,
        )
        if content is None:
            return None
        try:
            return self._parsed_task_from_dict(json.loads(content))
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
            )
        except TypeError:
            return None

        if not isinstance(parsed.classification_task_name, str):
            parsed.classification_task_name = None
        labels = parsed.classification_labels
        parsed.classification_labels = (
            [str(l).strip() for l in labels if isinstance(l, str) and l.strip()] or None
        ) if isinstance(labels, list) else None

        if parsed.classification:
            parsed.classification_task_name = parsed.classification_task_name or "classification"
            parsed.classification_labels = parsed.classification_labels or ["class_a", "class_b", "class_c"]

        return parsed

    @staticmethod
    def _extract_label_list(text: str) -> list[str] | None:
        """Extract a label list from patterns like 'into/as A, B or C' or 'labels: A, B, C'."""
        m = (
            re.search(r"(?:into|as) ([a-z0-9_,\s]+)", text)
            or re.search(r"labels?: ([a-z0-9_,\s]+)", text)
        )
        if not m:
            return None
        parts = re.split(r",| and | or ", m.group(1))
        return [p.strip() for p in parts if 1 <= len(p.strip()) <= 32] or None

    def _call_llm(self, messages: list[dict], timeout: int = 30) -> str | None:
        """POST to the Ollama /api/chat endpoint and return the assistant content string.

        Returns None on any network error, timeout, or malformed response.
        """
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

    def _generate_classification_example(
        self,
        idx: int,
        task_name: str | None = None,
        labels: list[str] | None = None,
    ) -> tuple[str, list[dict]]:
        task_name = task_name or "classification"
        labels = labels or ["class_a", "class_b", "class_c"]
        true_label = labels[idx % len(labels)]  # cycle through labels to enforce balance

        if task_name == "sentiment" and true_label in self._SENTIMENT_OUTCOMES:
            subject = self._rng.choice(self._SENTIMENT_SUBJECTS)
            outcome = self._rng.choice(self._SENTIMENT_OUTCOMES[true_label])
            text = f"{subject} {outcome}."
        else:
            text = self._rng.choice(self._GENERIC_TEXTS)

        return text, [{"task": task_name, "labels": labels, "true_label": [true_label]}]

    def _generate_relation_example(self) -> tuple[str, list[dict]]:
        person = self._rng.choice(self._REL_PEOPLE)
        company = self._rng.choice(self._REL_COMPANIES)
        city = self._rng.choice(self._REL_CITIES)
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
            "Each example is a JSON object:\n"
            '{"input": "<text>", "output": {"entities": {"label": ["span", ...]}, '
            '"classifications": [{"task": "name", "labels": [...], "true_label": ["label"]}], '
            '"relations": [{"relation_name": {"head": "span", "tail": "span"}}], '
            '"json_structures": [{...}]}}\n\n'
            "Rules:\n"
            "- Omit any output field whose task flag is false.\n"
            "- If ner=true and ner_entity_labels is set, only include those entity labels.\n"
            "- If classification=true, use the provided task_name, labels, and set "
            "true_label to exactly classification_true_label.\n"
            "- The input text must be consistent with the structured output.\n\n"
            "Return ONLY the JSON object.\n\n"
            f"Task description:\n{description}\n\n"
            f"Task config:\n{json.dumps(task_config, ensure_ascii=False)}"
        )
        content = self._call_llm(
            [{"role": "system", "content": "You respond with JSON only."},
             {"role": "user", "content": prompt}],
            timeout=60,
        )
        if content is None:
            return None
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            return None

        if not isinstance(result, dict):
            return None
        input_text = result.get("input")
        output = result.get("output")
        if not isinstance(input_text, str) or not isinstance(output, dict):
            return None

        # Enforce the chosen true_label regardless of what the LLM produced.
        if parsed.classification and true_label is not None:
            clss = output.get("classifications")
            if isinstance(clss, list) and clss and isinstance(clss[0], dict):
                clss[0].update({
                    "task": parsed.classification_task_name or "classification",
                    "labels": labels,
                    "true_label": [true_label],
                })

        return {"input": input_text, "output": output}
