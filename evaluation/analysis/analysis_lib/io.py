from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AnalysisInputs:
    evaluation_root: Path
    scenarios_path: Path
    paired_deltas_path: Path
    combined_judgments_path: Path
    judge_1_dir: Path
    judge_2_dir: Path
    lexical_analysis_path: Path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}: invalid JSON at line {line_number}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}: line {line_number} is not an object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_inputs(evaluation_root: Path) -> AnalysisInputs:
    root = evaluation_root.resolve()
    inputs = AnalysisInputs(
        evaluation_root=root,
        scenarios_path=root / "data" / "scenarios" / "final" / "scenarios.jsonl",
        paired_deltas_path=root / "data" / "aggregated" / "paired_deltas.jsonl",
        combined_judgments_path=root / "data" / "aggregated" / "combined_judgments.jsonl",
        judge_1_dir=root / "data" / "judgments" / "judge_1",
        judge_2_dir=root / "data" / "judgments" / "judge_2",
        lexical_analysis_path=root / "data" / "aggregated" / "lexical_diversity_analysis.json",
    )
    required = [
        inputs.scenarios_path,
        inputs.paired_deltas_path,
        inputs.combined_judgments_path,
        inputs.judge_1_dir,
        inputs.judge_2_dir,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required evaluation outputs:\n" + "\n".join(map(str, missing)))
    return inputs


def load_scenarios(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl(path))
    required = {
        "id",
        "category",
        "animal",
        "value",
        "context",
        "tradeoff",
        "animal_split",
        "value_split",
        "context_split",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{path}: missing scenario columns {sorted(missing)}")
    if df["id"].duplicated().any():
        raise RuntimeError(f"{path}: duplicate scenario IDs")
    df = df.rename(columns={"id": "scenario_id"})
    df["scenario_id"] = df["scenario_id"].astype(int)
    df["tradeoff"] = df["tradeoff"].astype(bool)
    df["semantic_cell"] = (
        df["category"].astype(str)
        + "|"
        + df["animal"].astype(str)
        + "|"
        + df["value"].astype(str)
        + "|"
        + df["context"].astype(str)
    )
    return df


def load_paired_deltas(path: Path, scenarios: pd.DataFrame) -> pd.DataFrame:
    raw = read_jsonl(path)
    records: list[dict[str, Any]] = []
    for row in raw:
        scores = row.get("score_deltas") or {}
        records.append(
            {
                **{k: v for k, v in row.items() if k != "score_deltas"},
                **{f"delta_{k}": v for k, v in scores.items()},
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"{path}: no paired deltas")
    df["scenario_id"] = df["scenario_id"].astype(int)
    meta = scenarios[
        [
            "scenario_id",
            "animal",
            "value",
            "context",
            "animal_split",
            "value_split",
            "context_split",
            "semantic_cell",
        ]
    ]
    df = df.merge(meta, on="scenario_id", how="left", validate="many_to_one")
    if df["semantic_cell"].isna().any():
        raise RuntimeError("paired_deltas contains scenario IDs absent from final scenarios")
    return df


def load_combined_judgments(path: Path, scenarios: pd.DataFrame) -> pd.DataFrame:
    raw = read_jsonl(path)
    records: list[dict[str, Any]] = []
    for row in raw:
        scores = row.get("scores") or {}
        records.append(
            {
                **{k: v for k, v in row.items() if k != "scores"},
                **{f"score_{k}": v for k, v in scores.items()},
            }
        )
    df = pd.DataFrame(records)
    df["scenario_id"] = df["scenario_id"].astype(int)
    meta = scenarios[["scenario_id", "semantic_cell"]]
    df = df.merge(meta, on="scenario_id", how="left", validate="many_to_one")
    return df


def load_judge_dir(path: Path, judge_number: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for file_path in sorted(path.glob("*.jsonl")):
        for row in read_jsonl(file_path):
            scores = row.get("scores") or {}
            records.append(
                {
                    "condition_key": file_path.stem,
                    "scenario_id": int(row["scenario_id"]),
                    "judge": judge_number,
                    "condition_type": row.get("condition"),
                    "feature": row.get("feature"),
                    "comparison_control": row.get("comparison_control"),
                    "outcome": row.get("outcome"),
                    **{f"score_{k}": v for k, v in scores.items()},
                }
            )
    if not records:
        raise RuntimeError(f"{path}: no judgment files")
    return pd.DataFrame(records)


def load_judges(judge_1_dir: Path, judge_2_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    j1 = load_judge_dir(judge_1_dir, 1)
    j2 = load_judge_dir(judge_2_dir, 2)
    keys = ["condition_key", "scenario_id"]
    if j1.duplicated(keys).any() or j2.duplicated(keys).any():
        raise RuntimeError("duplicate condition/scenario rows in judge outputs")
    if set(map(tuple, j1[keys].to_numpy())) != set(map(tuple, j2[keys].to_numpy())):
        raise RuntimeError("judge 1 and judge 2 cover different condition/scenario pairs")
    return j1, j2


def load_lexical_analysis(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def input_manifest(inputs: AnalysisInputs) -> dict[str, Any]:
    paths = {
        "scenarios": inputs.scenarios_path,
        "paired_deltas": inputs.paired_deltas_path,
        "combined_judgments": inputs.combined_judgments_path,
        "lexical_diversity_analysis": inputs.lexical_analysis_path,
    }
    manifest: dict[str, Any] = {}
    for name, path in paths.items():
        manifest[name] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        }
    for judge_name, directory in [("judge_1", inputs.judge_1_dir), ("judge_2", inputs.judge_2_dir)]:
        manifest[judge_name] = {
            file_path.name: sha256_file(file_path)
            for file_path in sorted(directory.glob("*.jsonl"))
        }
    return manifest
