import pytest
from mechinterp.linguistic_review import match_by_id, judgment_schema, batch_prompt

def sample(id):
    return {"review_id":id,"feature":"epistemic_hedging","scenario":"Q","continuation":"Perhaps yes."}

def rating(id):
    return {"review_id":id,"feature_realization_0_2":2,"coherence_0_2":2,
            "language":"en","cut_off":False,"notes":"Explicit uncertainty"}

def test_judge_results_matched_by_id_not_order():
    batch=[sample("a"),sample("b")]
    out=match_by_id(batch,[rating("b"),rating("a")])
    assert set(out)=={"a","b"}
    assert out["a"]["notes"]=="Explicit uncertainty"

def test_duplicate_missing_unknown_ids_are_rejected():
    batch=[sample("a"),sample("b")]
    with pytest.raises(ValueError):
        match_by_id(batch,[rating("a"),rating("a")])
    with pytest.raises(ValueError):
        match_by_id(batch,[rating("a")])
    with pytest.raises(ValueError):
        match_by_id(batch,[rating("a"),rating("c")])

def test_judge_prompt_has_no_arm_identifiers():
    text=batch_prompt([sample("a")])
    assert "generation_seed" not in text
    assert '"arm"' not in text
    assert "Perhaps yes." in text

def test_schema_requires_ids():
    schema=judgment_schema(["a","b"])
    assert schema["properties"]["judgments"]["minItems"]==2
    assert schema["properties"]["judgments"]["items"]["properties"]["review_id"]["enum"]==["a","b"]
