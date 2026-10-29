import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation" / "scripts"))
from common import normalize_judge_response_shape


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["judgments"],
    "properties": {
        "judgments": {"type": "array", "minItems": 2, "maxItems": 2,
                      "items": {"type": "object", "additionalProperties": False,
                                "required": ["scenario_id"],
                                "properties": {"scenario_id": {"type": "integer"}}}},
    },
}


def test_bare_steering_judgments_are_losslessly_wrapped():
    items = [{"scenario_id": 790}, {"scenario_id": 766}]
    wrapped = normalize_judge_response_shape(
        items, schema_name="steering_judgments_2", schema=SCHEMA)
    assert wrapped == {"judgments": items}
    Draft202012Validator(SCHEMA).validate(wrapped)


def test_incomplete_bare_batch_still_rejected():
    wrapped = normalize_judge_response_shape(
        [{"scenario_id": 790}], schema_name="steering_judgments_2", schema=SCHEMA)
    with pytest.raises(Exception):
        Draft202012Validator(SCHEMA).validate(wrapped)


def test_other_tasks_remain_strict():
    items = [{"scenario_id": 790}, {"scenario_id": 766}]
    assert normalize_judge_response_shape(
        items, schema_name="response_judgments", schema=SCHEMA) is items


def test_already_correct_object_unchanged():
    obj = {"judgments": [{"scenario_id": 790}, {"scenario_id": 766}]}
    assert normalize_judge_response_shape(
        obj, schema_name="steering_judgments_1", schema=SCHEMA) is obj
