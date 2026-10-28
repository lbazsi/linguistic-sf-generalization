from mechinterp.steering_pilot import select_pilot_linguistic_pairs
from mechinterp.common import split_groups

def test_pilot_uses_only_development_semantic_ids():
    pairs=[{"id":i,"canonical":f"a{i}","feature_variant":f"b{i}"} for i in range(300)]
    selected=select_pilot_linguistic_pairs(pairs,42,20)
    split=split_groups((p["id"] for p in pairs),42)
    assert len(selected)==20
    assert all(split[p["id"]]=="development" for p in selected)
    assert selected==select_pilot_linguistic_pairs(pairs,42,20)
