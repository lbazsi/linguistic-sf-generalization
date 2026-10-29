import sys
from pathlib import Path
import pytest
from jsonschema import ValidationError

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"evaluation"/"scripts"))
from common import format_judge_error, JudgeResponseFormatError


def test_diagnostic_is_concise_and_reports_count():
    err=ValidationError("too long",validator="maxItems",validator_value=20,
                        instance=[{"private":"text"}]*21,path=["judgments"])
    msg=format_judge_error(err)
    assert "judgments" in msg and "maxItems" in msg and "21" in msg
    assert "private" not in msg


def test_format_exception_type():
    assert format_judge_error(ValueError("secret"))=="ValueError"
