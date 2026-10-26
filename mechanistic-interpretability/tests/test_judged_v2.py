"""Paired-baseline regression tests independent of models and network APIs."""
from pathlib import Path
import json
import pytest
from mechinterp import judged
from mechinterp.common import read_data, write_rows


def fixture(tmp_path,monkeypatch,seed_error=False):
    spec={"feature":"nominalization","adapter":"nominalization",
          "policy":"prompt_last","strength":1.0,"layer":2,"site":"mlp"}
    monkeypatch.setattr(judged,"selected_conditions",
                        lambda cfg,family,policy="prompt_last":[spec])
    monkeypatch.setattr(judged,"load_scenarios",lambda cfg:[
        {"id":i,"category":"id","animal":f"animal_{i}","value":"welfare",
         "context":"a","tradeoff":True} for i in range(4)])
    monkeypatch.setattr(judged,"load_screening",lambda cfg:[
        {"scenario_id":i,"split":"test"} for i in range(4)])
    cfg={"paths":{"outputs":str(tmp_path)}, "steering":{
         "judged_n":4,"judged_split":"test","judging":{"batch_size":2}},
         "seed":42,"stats":{"bootstrap_replicates":30,"permutations":30}}
    root=tmp_path/"llama31"/"activation-steering"
    for arm,score in (("unsteered",1.),("steered",3.)):
        name=f"nominalization__nominalization__prompt_last__{arm}.jsonl"
        responses=[]
        for sid in range(4):
            responses.append({"scenario_id":sid,"feature":"nominalization",
                "adapter":"nominalization","policy":"prompt_last",
                "prompt":"same prompt","generation_seed":45+sid+
                    (1 if arm=="steered" and seed_error else 0),
                "arm":arm,"continuation":f"generated {arm}"})
        write_rows(root/"judged"/name,responses)
        for j in (1,2):
            write_rows(root/"judgments"/f"judge_{j}"/name,[
                {"scenario_id":i,"scores":{"target_value_support":score}} for i in range(4)])
    return cfg


def test_judged_comparison_uses_new_seed_matched_baseline(tmp_path,monkeypatch):
    cfg=fixture(tmp_path,monkeypatch)
    count=judged.workload(cfg,"llama31")
    assert count["matched_pairs"]==4
    assert count["generations"]==8
    assert count["individual_judgments"]==16
    assert count["api_requests_without_retries"]==8
    out=judged.aggregate(cfg,"llama31")
    rows=read_data(out)
    assert len(rows)==4
    assert {r["paired_delta"] for r in rows}=={2.}
    assert all(r["comparison_protocol"]=="matched_generated_baseline_v2" for r in rows)
    summary=json.loads((out.parent/"summary.json").read_text())
    assert summary[0]["paired_delta"]["mean"]==2.


def test_judged_pair_mismatch_is_rejected(tmp_path,monkeypatch):
    cfg=fixture(tmp_path,monkeypatch,seed_error=True)
    with pytest.raises(ValueError,match="Unmatched prompt or seed"):
        judged.aggregate(cfg,"llama31")
