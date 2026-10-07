from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "training.yaml"


class ConfigError(RuntimeError):
    pass


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping.")

    for key in ["model", "data", "sequence", "lora", "training", "output"]:
        if key not in config:
            raise ConfigError(f"Missing top-level config key: {key}")

    training = config["training"]
    for key in [
        "seed",
        "num_train_epochs",
        "learning_rate",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
    ]:
        if key not in training:
            raise ConfigError(f"Missing training.{key}")

    if int(config["data"]["validation_size"]) <= 0:
        raise ConfigError("data.validation_size must be positive.")

    return config


def resolve_data_root(config: dict[str, Any], override: str | None = None) -> Path:
    raw = override if override is not None else config["data"]["root"]
    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_output_root(config: dict[str, Any]) -> Path:
    path = Path(config["output"]["root"])
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                raise RuntimeError(f"{path}: empty line at {line_number}")
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}: invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}: line {line_number} is not a JSON object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_by_id(rows: Iterable[dict[str, Any]], path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get("id")
        if not isinstance(item_id, int):
            raise RuntimeError(f"{path}: every row must contain an integer id")
        if item_id in result:
            raise RuntimeError(f"{path}: duplicate id {item_id}")
        result[item_id] = row
    return result


def load_canonical(data_root: Path) -> tuple[Path, list[dict[str, Any]], dict[int, dict[str, Any]]]:
    path = data_root / "canonical" / "corpus.jsonl"
    rows = read_jsonl(path)
    required = {"id", "language", "animal", "value", "context", "canonical"}
    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"{path}: row {index} missing fields: {sorted(missing)}")
        if not isinstance(row["canonical"], str) or not row["canonical"].strip():
            raise RuntimeError(f"{path}: row {index} has empty canonical text")
    return path, rows, _unique_by_id(rows, path)


def language_control_files(data_root: Path) -> list[Path]:
    control_dir = data_root / "language_controls"
    if not control_dir.exists():
        return []
    return sorted(
        path for path in control_dir.glob("*.jsonl")
        if path.is_file() and not path.name.endswith(".generated.jsonl")
    )


def load_language_control(
    data_root: Path,
    language: str,
    canonical_by_id: dict[int, dict[str, Any]],
) -> tuple[Path, list[dict[str, Any]], dict[int, dict[str, Any]]]:
    path = data_root / "language_controls" / f"{language}.jsonl"
    rows = read_jsonl(path)
    required = {
        "id", "language", "animal", "value", "context", "semantic_anchor", "canonical"
    }
    by_id = _unique_by_id(rows, path)
    if set(by_id) != set(canonical_by_id):
        raise RuntimeError(f"{path}: IDs do not match semantic-anchor corpus.")

    for item_id, row in by_id.items():
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"{path}: id {item_id} missing fields: {sorted(missing)}")
        source = canonical_by_id[item_id]
        for field in ["animal", "value", "context"]:
            if row[field] != source[field]:
                raise RuntimeError(f"{path}: id {item_id} field {field!r} differs from semantic anchor")
        if row["semantic_anchor"] != source["canonical"]:
            raise RuntimeError(f"{path}: id {item_id} semantic_anchor differs from canonical corpus")
        if row["language"] != language:
            raise RuntimeError(f"{path}: id {item_id} expected language={language!r}")
        if not isinstance(row["canonical"], str) or not row["canonical"].strip():
            raise RuntimeError(f"{path}: id {item_id} has empty canonical text")
    return path, rows, by_id


def feature_files(data_root: Path) -> list[Path]:
    final_dir = data_root / "final"
    if not final_dir.exists():
        raise FileNotFoundError(final_dir)
    return sorted(path for path in final_dir.glob("*.jsonl") if path.is_file())


def feature_index(data_root: Path, feature_name: str) -> int:
    names = [path.stem for path in feature_files(data_root)]
    if feature_name not in names:
        raise FileNotFoundError(
            f"No final feature dataset named {feature_name!r}; available: {', '.join(names)}"
        )
    return names.index(feature_name) + 1


def load_feature(
    data_root: Path,
    feature_name: str,
    canonical_by_id: dict[int, dict[str, Any]],
) -> tuple[Path, list[dict[str, Any]], dict[int, dict[str, Any]]]:
    path = data_root / "final" / f"{feature_name}.jsonl"
    rows = read_jsonl(path)
    required = {
        "id", "feature", "manipulation_level", "canonical_language",
        "feature_variant_language", "animal", "value", "context",
        "semantic_anchor", "canonical", "feature_variant"
    }
    by_id = _unique_by_id(rows, path)

    if set(by_id) != set(canonical_by_id):
        missing = sorted(set(canonical_by_id) - set(by_id))
        extra = sorted(set(by_id) - set(canonical_by_id))
        raise RuntimeError(
            f"{path}: IDs do not match canonical corpus; missing={missing[:10]}, extra={extra[:10]}"
        )

    for item_id, row in by_id.items():
        missing_fields = required - set(row)
        if missing_fields:
            raise RuntimeError(f"{path}: id {item_id} missing fields: {sorted(missing_fields)}")
        source = canonical_by_id[item_id]
        if row["feature"] != feature_name:
            raise RuntimeError(f"{path}: id {item_id} has feature={row['feature']!r}")
        for field in ["animal", "value", "context"]:
            if row[field] != source[field]:
                raise RuntimeError(
                    f"{path}: id {item_id} field {field!r} differs from semantic-anchor corpus"
                )
        if row["semantic_anchor"] != source["canonical"]:
            raise RuntimeError(f"{path}: id {item_id} semantic_anchor differs from canonical corpus")
        if row["manipulation_level"] == "within_language" and row["canonical"] != source["canonical"]:
            raise RuntimeError(
                f"{path}: id {item_id} within-language canonical differs from canonical corpus"
            )
        if not isinstance(row["feature_variant"], str) or not row["feature_variant"].strip():
            raise RuntimeError(f"{path}: id {item_id} has empty feature_variant")

    return path, rows, by_id


def stratified_validation_ids(
    canonical_rows: list[dict[str, Any]],
    validation_size: int,
    seed: int,
) -> list[int]:
    if validation_size >= len(canonical_rows):
        raise RuntimeError("validation_size must be smaller than the canonical dataset.")

    dimensions = ["animal", "value", "context"]
    for row in canonical_rows:
        missing = [field for field in dimensions if field not in row]
        if missing:
            raise RuntimeError(
                f"Canonical row {row.get('id')} is missing domain fields: {missing}"
            )

    cells: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    levels = {field: sorted({str(row[field]) for row in canonical_rows}) for field in dimensions}
    for row in canonical_rows:
        cell = (str(row["animal"]), str(row["value"]), str(row["context"]))
        cells[cell].append(int(row["id"]))

    if validation_size > len(cells):
        raise RuntimeError(
            "validation_size exceeds the number of animal×value×context cells; "
            "the diagnostic split is designed to use distinct domain cells."
        )

    rng = random.Random(seed)
    for ids in cells.values():
        rng.shuffle(ids)

    remaining_cells = list(cells)
    rng.shuffle(remaining_cells)
    counts = {field: defaultdict(int) for field in dimensions}
    selected: list[int] = []

    for step in range(validation_size):
        best_index = 0
        best_score: float | None = None
        for index, cell in enumerate(remaining_cells):
            candidate = {
                "animal": cell[0],
                "value": cell[1],
                "context": cell[2],
            }
            score = 0.0
            for field in dimensions:
                target = (step + 1) / len(levels[field])
                for level in levels[field]:
                    new_count = counts[field][level] + (
                        1 if candidate[field] == level else 0
                    )
                    score += (new_count - target) ** 2
            if best_score is None or score < best_score:
                best_score = score
                best_index = index

        cell = remaining_cells.pop(best_index)
        selected.append(cells[cell][0])
        counts["animal"][cell[0]] += 1
        counts["value"][cell[1]] += 1
        counts["context"][cell[2]] += 1

    return sorted(selected)

def split_rows(
    rows: list[dict[str, Any]],
    validation_ids: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [row for row in rows if row["id"] not in validation_ids]
    validation = [row for row in rows if row["id"] in validation_ids]
    return train, validation


def choose_max_length(
    lengths: list[int],
    sequence_config: dict[str, Any],
    tokenizer_model_max_length: int,
) -> int:
    configured = sequence_config.get("max_length")
    if configured is not None:
        return int(configured)

    quantile = float(sequence_config.get("auto_quantile", 0.995))
    candidates = sorted(int(value) for value in sequence_config["auto_candidates"])
    if not 0 < quantile <= 1:
        raise ConfigError("sequence.auto_quantile must be in (0, 1].")
    if not candidates:
        raise ConfigError("sequence.auto_candidates must not be empty.")

    sorted_lengths = sorted(lengths)
    index = max(0, min(len(sorted_lengths) - 1, math.ceil(quantile * len(sorted_lengths)) - 1))
    target = sorted_lengths[index]

    sane_model_limit = (
        tokenizer_model_max_length
        if 0 < tokenizer_model_max_length < 1_000_000
        else max(candidates)
    )
    allowed = [value for value in candidates if value <= sane_model_limit]
    if not allowed:
        raise RuntimeError(
            f"No sequence.auto_candidates fit tokenizer model_max_length={tokenizer_model_max_length}."
        )
    for value in allowed:
        if value >= target:
            return value
    return max(allowed)


def length_stats(lengths: list[int]) -> dict[str, float | int]:
    ordered = sorted(lengths)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "min": min(lengths),
        "mean": round(mean(lengths), 3),
        "median": float(median(lengths)),
        "p95": ordered[p95_index],
        "max": max(lengths),
    }


def output_directory(
    config: dict[str, Any],
    *,
    seed: int,
    condition: str,
    feature_name: str | None = None,
    feature_number: int | None = None,
    control_language: str | None = None,
) -> Path:
    root = resolve_output_root(config)
    if condition == "canonical":
        return root / "canonical" / f"seed_{seed}"
    if condition == "control":
        if not control_language:
            raise ValueError("Control output requires control_language.")
        return root / "controls" / control_language / f"seed_{seed}"
    if not feature_name or feature_number is None:
        raise ValueError("Feature output requires feature_name and feature_number.")
    return root / "features" / f"{feature_name}_{feature_number:02d}" / f"seed_{seed}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
