from __future__ import annotations

from copy import deepcopy
from typing import Iterable


CANONICAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "language", "animal", "value", "context", "canonical"],
    "properties": {
        "id": {"type": "integer", "minimum": 1},
        "language": {"type": "string", "minLength": 1},
        "animal": {"type": "string", "minLength": 1},
        "value": {"type": "string", "minLength": 1},
        "context": {"type": "string", "minLength": 1},
        "canonical": {"type": "string", "minLength": 1},
    },
}

CANONICAL_TEXT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "canonical"],
    "properties": {
        "id": {"type": "integer", "minimum": 1},
        "canonical": {"type": "string", "minLength": 1},
    },
}

VARIANT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "feature_variant"],
    "properties": {
        "id": {"type": "integer", "minimum": 1},
        "feature_variant": {"type": "string", "minLength": 1},
    },
}

CONTROL_TEXT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "canonical"],
    "properties": {
        "id": {"type": "integer", "minimum": 1},
        "canonical": {"type": "string", "minLength": 1},
    },
}

CROSS_PAIR_TEXT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "canonical", "feature_variant"],
    "properties": {
        "id": {"type": "integer", "minimum": 1},
        "canonical": {"type": "string", "minLength": 1},
        "feature_variant": {"type": "string", "minLength": 1},
    },
}

PAIR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id",
        "feature",
        "manipulation_level",
        "canonical_language",
        "feature_variant_language",
        "animal",
        "value",
        "context",
        "semantic_anchor",
        "canonical",
        "feature_variant",
    ],
    "properties": {
        "id": {"type": "integer", "minimum": 1},
        "feature": {"type": "string", "minLength": 1},
        "manipulation_level": {
            "type": "string",
            "enum": ["within_language", "cross_linguistic"],
        },
        "canonical_language": {"type": "string", "minLength": 2},
        "feature_variant_language": {"type": "string", "minLength": 2},
        "animal": {"type": "string", "minLength": 1},
        "value": {"type": "string", "minLength": 1},
        "context": {"type": "string", "minLength": 1},
        "semantic_anchor": {"type": "string", "minLength": 1},
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
        "manipulation_level",
        "languages",
        "definition",
        "transformation",
        "semantic_constraints",
        "examples",
        "cross_lingual_notes",
    ],
    "properties": {
        "name": {"type": "string", "minLength": 1, "pattern": "^[A-Za-z0-9_-]+$"},
        "description": {"type": "string", "minLength": 1},
        "language": {"type": "string", "minLength": 1},
        "canonical_language": {"type": "string", "minLength": 2},
        "feature_variant_language": {"type": "string", "minLength": 2},
        "manipulation_level": {
            "type": "string",
            "enum": ["within_language", "cross_linguistic", "covariate"],
        },
        "languages": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 2},
        },
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
        "cross_lingual_notes": {"type": "string", "minLength": 1},
    },
    "allOf": [
        {
            "if": {
                "properties": {
                    "manipulation_level": {"const": "cross_linguistic"}
                }
            },
            "then": {
                "required": ["canonical_language", "feature_variant_language"]
            }
        },
        {
            "if": {
                "properties": {
                    "manipulation_level": {"const": "covariate"}
                }
            },
            "then": {},
            "else": {
                "properties": {
                    "examples": {"minItems": 1}
                }
            },
        }
    ],
}


def _batch_schema(item_schema: dict, expected_ids: Iterable[int]) -> dict:
    ids = list(expected_ids)
    schema = deepcopy(item_schema)
    if len(ids) == 1:
        schema["properties"]["id"] = {
            "type": "integer",
            "minimum": ids[0],
            "maximum": ids[0],
        }
    else:
        schema["properties"]["id"] = {"type": "integer", "minimum": 1}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": schema,
            }
        },
    }


def canonical_batch_schema(expected_ids: Iterable[int]) -> dict:
    return _batch_schema(CANONICAL_SCHEMA, expected_ids)


def canonical_text_batch_schema(expected_ids: Iterable[int]) -> dict:
    return _batch_schema(CANONICAL_TEXT_SCHEMA, expected_ids)


def variant_batch_schema(expected_ids: Iterable[int]) -> dict:
    return _batch_schema(VARIANT_SCHEMA, expected_ids)


def control_text_batch_schema(expected_ids: Iterable[int]) -> dict:
    return _batch_schema(CONTROL_TEXT_SCHEMA, expected_ids)


def cross_pair_text_batch_schema(expected_ids: Iterable[int]) -> dict:
    return _batch_schema(CROSS_PAIR_TEXT_SCHEMA, expected_ids)


def pair_batch_schema(expected_ids: Iterable[int]) -> dict:
    return _batch_schema(PAIR_SCHEMA, expected_ids)


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
