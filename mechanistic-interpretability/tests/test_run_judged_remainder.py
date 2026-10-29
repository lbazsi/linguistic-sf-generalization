import json
from pathlib import Path
from types import SimpleNamespace

from run_judged_remainder import archive_incompatible, verify_generations
from mechinterp.batched_generation import PROTOCOL


def test_archive_preserves_steering_selection(tmp_path):
    root = tmp_path / "activation-steering"
    (root / "judged").mkdir(parents=True)
    (root / "judgments").mkdir()
    (root / "selected_all.json").write_text("selection")
    old = root / "judged" / "voice__canonical__prompt_last__unsteered.jsonl"
    old.write_text(json.dumps({"sampling_protocol": "paired_batch_rng_v1"}) + "\n")
    (root / "judgments" / "api_usage.json").write_text("{}")
    archive = archive_incompatible(root)
    assert archive is not None
    assert (archive / "judged" / old.name).is_file()
    assert (archive / "judgments" / "api_usage.json").is_file()
    assert (root / "selected_all.json").read_text() == "selection"
    assert not (root / "judged").exists()


def test_resume_keeps_compatible_generations(tmp_path):
    root = tmp_path / "activation-steering"
    (root / "judged").mkdir(parents=True)
    target = root / "judged" / "voice__canonical__prompt_last__unsteered.jsonl"
    target.write_text(json.dumps({"sampling_protocol": PROTOCOL}) + "\n")
    assert archive_incompatible(root) is None
    assert target.exists()


def test_no_judgments_without_source(tmp_path):
    root = tmp_path / "activation-steering"
    (root / "judgments").mkdir(parents=True)
    (root / "judgments" / "summary.json").write_text("{}")
    archive = archive_incompatible(root)
    assert (archive / "judgments" / "summary.json").is_file()
