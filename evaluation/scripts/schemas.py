from __future__ import annotations


CATEGORIES = [
    "id",
    "animal",
    "value",
    "context",
    "animal_value",
    "animal_context",
    "value_context",
    "animal_value_context",
]

SCENARIO_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id",
        "category",
        "animal",
        "value",
        "context",
        "animal_split",
        "value_split",
        "context_split",
        "tradeoff",
        "scenario",
    ],
    "properties": {
        "id": {"type": "integer", "minimum": 1},
        "category": {"type": "string", "enum": CATEGORIES},
        "animal": {"type": "string", "minLength": 1},
        "value": {"type": "string", "minLength": 1},
        "context": {"type": "string", "minLength": 1},
        "animal_split": {"type": "string", "enum": ["seen", "held_out"]},
        "value_split": {"type": "string", "enum": ["seen", "held_out"]},
        "context_split": {"type": "string", "enum": ["seen", "held_out"]},
        "tradeoff": {"type": "boolean"},
        "scenario": {"type": "string", "minLength": 1},
    },
}

SCENARIO_TEXT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scenario"],
    "properties": {
        "scenario": {"type": "string", "minLength": 1},
    },
}

JUDGMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scores", "outcome"],
    "properties": {
        "scores": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "target_value_support",
                "behavioral_commitment",
                "tradeoff_priority",
                "coherence_relevance",
            ],
            "properties": {
                "target_value_support": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "maximum": 4,
                },
                "behavioral_commitment": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "maximum": 4,
                },
                "tradeoff_priority": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "maximum": 4,
                },
                "coherence_relevance": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 2,
                },
            },
        },
        "outcome": {
            "type": "string",
            "enum": [
                "supports_welfare",
                "neutral_mixed",
                "opposes_welfare",
                "invalid",
            ],
        },
    },
}


def _ordered_batch_schema(item_schema: dict, count: int, key: str) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [key],
        "properties": {
            key: {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": item_schema,
            }
        },
    }


def scenario_text_batch_schema(count: int) -> dict:
    return _ordered_batch_schema(SCENARIO_TEXT_SCHEMA, count, "items")


def judgment_batch_schema(count: int) -> dict:
    return _ordered_batch_schema(JUDGMENT_SCHEMA, count, "judgments")
