import pytest
import torch
from types import SimpleNamespace
from mechinterp.batched_generation import (decode_generated_batch,batch_seed,validate_batch_pairs,PROTOCOL)
from mechinterp.common import scenario_prompt

def test_seed_stable():
    assert batch_seed(45,[{"scenario_id":10},{"scenario_id":11}])==55

def test_eos_is_detected_before_padding():
    class T:
        eos_token_id=2
        def decode(self,ids,skip_special_tokens=True):
            return str(ids)
    output=torch.tensor([[10,11,90,2,2],[10,11,91,92,93]])
    model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2))
    rows=decode_generated_batch(output,2,T(),model,3)
    assert rows[0]["generated_tokens"]==2
    assert rows[0]["ended_with_eos"]
    assert not rows[0]["reached_token_limit"]
    assert rows[1]["reached_token_limit"]

def test_batch_resumption_rejects_protocol_mix():
    scenario={"scenario_id":10,"scenario":"A scenario"}
    row={"scenario_id":10,"feature":"voice","adapter":"voice","policy":"generation",
         "prompt":scenario_prompt("A scenario"),"generation_seed":55,
         "sampling_protocol":PROTOCOL,"batch_size_requested":4,
         "batch_scenario_ids":[10],"max_new_tokens":256}
    pair={"scenario_id":10,"unsteered":{**row,"arm":"unsteered","strength":0},
          "steered":{**row,"arm":"steered","strength":16}}
    assert validate_batch_pairs([pair],[scenario],"voice","voice","generation",16,256,4,45)=={10}
    with pytest.raises(ValueError):
        validate_batch_pairs([pair],[scenario],"voice","voice","generation",16,256,2,45)
