from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Literal, TypedDict


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


class DataGenerator:
    """Generate GLiNER2-compatible synthetic training examples from a task description.

    The generator is deliberately rule-based and does not require an external LLM.
    It focuses on:
      - inferring which GLiNER2 task-types are relevant from a natural language description
      - composing multi-task outputs into a single `output` dict
      - ensuring basic diversity and class balance for classification tasks
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

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
            text_pieces: List[str] = []
            output: OutputDict = {}

            if parsed.ner:
                ner_text, entities = self._generate_ner_example(idx)
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
    def _generate_ner_example(self, idx: int) -> tuple[str, Dict[str, List[str]]]:
        people = ["Alice Johnson", "Bob Smith", "Carlos Diaz", "Diana Lee"]
        companies = ["Acme Corp", "Globex", "Innotech", "Fastino AI"]
        locations = ["New York", "Berlin", "Tokyo", "San Francisco"]

        person = self._rng.choice(people)
        company = self._rng.choice(companies)
        location = self._rng.choice(locations)

        templates = [
            f"{person} works as a data scientist at {company} in {location}.",
            f"{company}, based in {location}, recently hired {person}.",
            f"In {location}, {person} presented new research from {company}.",
        ]
        text = self._rng.choice(templates)

        entities = {
            "person": [person],
            "company": [company],
            "location": [location],
        }
        return text, entities

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

        # A small corpus of generic sentences keyed by label semantics
        sentiment_bank = {
            "positive": [
                "The product worked flawlessly and exceeded expectations.",
                "Customer support was quick and very helpful.",
            ],
            "negative": [
                "The service was slow and the staff were rude.",
                "The software kept crashing and losing my work.",
            ],
            "neutral": [
                "The meeting lasted for one hour and covered all agenda items.",
                "The package arrived on the expected delivery date.",
            ],
        }

        if task_name == "sentiment" and true_label in sentiment_bank:
            text = self._rng.choice(sentiment_bank[true_label])
        else:
            # Generic text for arbitrary labels
            text = (
                f"This example should be labeled as '{true_label}' for the "
                f"{task_name} task."
            )

        classifications = [
            {
                "task": task_name,
                "labels": labels,
                "true_label": [true_label],
            }
        ]
        return text, classifications

    # ------------------------------------------------------------------
    # Relation extraction example generation
    # ------------------------------------------------------------------
    def _generate_relation_example(self, idx: int) -> tuple[str, List[Dict]]:
        people = ["John", "Maria", "Ethan", "Sara"]
        companies = ["Apple Inc.", "Fastino AI", "Globex", "Innotech"]
        cities = ["London", "Paris", "Toronto", "San Francisco"]

        person = self._rng.choice(people)
        company = self._rng.choice(companies)
        city = self._rng.choice(cities)

        text = (
            f"{person} works for {company} and currently lives in {city}. "
            f"In the past, {person} also collaborated with other startups."
        )

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

