import pytest
from mechinterp.judged import validate_generation_pairs

def test_matching_resume_pairs_validate():
    scenarios=[{"scenario_id":11,"scenario":"A scenario"}]
    from mechinterp.common import scenario_prompt
    prompt=scenario_prompt("A scenario")
    a={"scenario_id":11,"feature":"voice","adapter":"voice","policy":"generation",
       "arm":"unsteered","prompt":prompt,"generation_seed":56,"max_new_tokens":256,
       "strength":0}
    b={**a,"arm":"steered","strength":16}
    pairs=[{"scenario_id":11,"unsteered":a,"steered":b}]
    assert validate_generation_pairs(pairs,scenarios,"voice","voice","generation",256,45)=={11}
    with pytest.raises(ValueError):
        validate_generation_pairs(pairs,scenarios,"voice","voice","generation",96,45)
    with pytest.raises(ValueError):
        validate_generation_pairs(pairs*2,scenarios,"voice","voice","generation",256,45)
