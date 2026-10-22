import json
from pathlib import Path
import pytest
from mechinterp.common import read_data,split_groups,control_for,FEATURES,CROSS_LANGUAGE

def test_dataset_reader_supports_json_and_jsonl(tmp_path):
    data=[{"id":1,"text":"A"},{"id":2,"text":"B"}]
    array=tmp_path/"array.json";array.write_text(json.dumps(data))
    jsonl=tmp_path/"lines.json";jsonl.write_text("\n".join(json.dumps(r) for r in data)+"\n")
    assert read_data(array)==read_data(jsonl)==data

def test_fixed_group_splits():
    ids=list(range(150))
    s=split_groups(ids,42)
    assert s==split_groups(reversed(ids),42)
    assert sum(x=="train" for x in s.values())==105
    assert sum(x=="development" for x in s.values()) in (22,23)

def test_control_registry():
    assert len(FEATURES)==14
    assert len(CROSS_LANGUAGE)==3
    assert control_for("fusion")=="control_zh"
    assert control_for("constituent_order")=="canonical"
    with pytest.raises(ValueError):control_for("not_a_feature")
