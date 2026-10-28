import json
from pathlib import Path
from export_final_results import choose


def test_exports_final_results_only(tmp_path):
    source=tmp_path/"llama31"
    steer=source/"activation-steering"
    (steer/"judged").mkdir(parents=True)
    (steer/"judgments"/"judge_1").mkdir(parents=True)
    (steer/"screen_checkpoints").mkdir()
    (steer/"screen_cache").mkdir()
    (steer/"pilot").mkdir()
    (source/"plots").mkdir()
    (source/"linear-probing").mkdir()
    for rel in (
        "activation-steering/selected_all.json",
        "activation-steering/screen_voice_canonical_generation.jsonl",
        "activation-steering/directions_voice_metadata.json",
        "activation-steering/directions_voice.safetensors",
        "activation-steering/judgments/summary.json",
        "activation-steering/judgments/combined.jsonl",
        "activation-steering/judgments/judge_1/voice.jsonl",
        "activation-steering/judged/voice__steered.jsonl",
        "activation-steering/screen_cache/baseline.json",
        "activation-steering/screen_checkpoints/piece.jsonl",
        "activation-steering/pilot/extra.json",
        "linear-probing/behavior_all_base.jsonl",
        "plots/diff_example.png",
    ):
        path=source/rel
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("test")
    rels={str(p.relative_to(source)) for p in choose(source)}
    assert rels=={
        "activation-steering/selected_all.json",
        "activation-steering/screen_voice_canonical_generation.jsonl",
        "activation-steering/directions_voice_metadata.json",
        "activation-steering/judgments/summary.json",
        "activation-steering/judgments/combined.jsonl",
        "linear-probing/behavior_all_base.jsonl",
        "plots/diff_example.png",
    }
