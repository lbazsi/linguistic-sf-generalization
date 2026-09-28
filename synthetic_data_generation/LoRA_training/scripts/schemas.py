from __future__ import annotations

from copy import deepcopy
from typing import Iterable


PAIR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "feature", "language", "topic", "canonical", "feature_variant"],
    "properties": {
        "id": {"type": "integer", "minimum": 1},
        "feature": {"type": "string", "minLength": 1},
        "language": {"type": "string", "minLength": 1},
        "topic": {"type": "string", "minLength": 1},
        "canonical": {"type": "string", "minLength": 1},
        "feature_variant": {"type": "string", "minLength": 1},
    },
}

ISSUE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "feature", "stage", "issue_type", "field", "message"],
    "properties": {
        "id": {"type": ["integer", "null"]},
        "feature": {"type": "string", "minLength": 1},
        "stage": {"type": "string", "minLength": 1},
        "issue_type": {"type": "string", "minLength": 1},
        "field": {"type": ["string", "null"]},
        "message": {"type": "string", "minLength": 1},
    },
}

FEATURE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "name",
        "description",
        "language",
        "definition",
        "transformation",
        "semantic_constraints",
        "examples",
    ],
    "properties": {
        "name": {"type": "string", "minLength": 1, "pattern": "^[A-Za-z0-9_-]+$"},
        "description": {"type": "string", "minLength": 1},
        "language": {"type": "string", "minLength": 1},
        "definition": {
            "type": "object",
            "additionalProperties": False,
            "required": ["canonical", "feature_variant"],
            "properties": {
                "canonical": {"type": "string", "minLength": 1},
                "feature_variant": {"type": "string", "minLength": 1},
            },
        },
        "transformation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["instructions", "preferred_patterns", "avoid_patterns"],
            "properties": {
                "instructions": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "preferred_patterns": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "avoid_patterns": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
            },
        },
        "semantic_constraints": {
            "type": "object",
            "additionalProperties": False,
            "required": ["preserve"],
            "properties": {
                "preserve": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                }
            },
        },
        "examples": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["canonical", "feature_variant"],
                "properties": {
                    "canonical": {"type": "string", "minLength": 1},
                    "feature_variant": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}


def pair_batch_schema(expected_ids: Iterable[int]) -> dict:
    ids = list(expected_ids)
    item_schema = deepcopy(PAIR_SCHEMA)
    item_schema["properties"]["id"] = {"type": "integer", "enum": ids}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": item_schema,
            }
        },
    }


def review_batch_schema(expected_ids: Iterable[int]) -> dict:
    ids = list(expected_ids)
    issue = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "issue_type", "field", "message"],
        "properties": {
            "id": {"type": "integer", "enum": ids},
            "issue_type": {"type": "string", "minLength": 1},
            "field": {"type": ["string", "null"]},
            "message": {"type": "string", "minLength": 1},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["issues"],
        "properties": {"issues": {"type": "array", "items": issue}},
    }
