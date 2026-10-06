from __future__ import annotations

import asyncio
import json
import os
import random
import tempfile
import hashlib
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import httpx
import yaml
from dotenv import load_dotenv
from jsonschema import Draft202012Validator, ValidationError

from schemas import CATEGORIES, SCENARIO_SCHEMA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigError(RuntimeError):
    pass


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config() -> dict[str, Any]:
    config = load_yaml(CONFIG_PATH)
    if not isinstance(config, dict):
        raise ConfigError(f"{CONFIG_PATH} must contain a YAML mapping.")
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    raw = config["paths"][key]
    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def ensure_directories(config: dict[str, Any]) -> None:
    for key in ["raw_scenarios", "final_scenarios"]:
        resolve_path(config, key).parent.mkdir(parents=True, exist_ok=True)
    for key in ["responses", "judgments", "aggregated", "manifests"]:
        resolve_path(config, key).mkdir(parents=True, exist_ok=True)


def load_domains(path: Path) -> dict[str, list[str]]:
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping.")
    result: dict[str, list[str]] = {}
    for key in ["animals", "values", "contexts"]:
        values = data.get(key)
        if not isinstance(values, list) or not values:
            raise ConfigError(f"{path}: {key} must be a non-empty list.")
        result[key] = [str(value).strip() for value in values]
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
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


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def manifest_path(config: dict[str, Any], stage: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return resolve_path(config, "manifests") / f"{stamp}_{stage}.json"


def write_manifest(
    config: dict[str, Any],
    *,
    stage: str,
    inputs: dict[str, Path] | None = None,
    outputs: dict[str, Path] | None = None,
    prompt_files: list[str] | None = None,
    stats: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    prompt_files = prompt_files or []
    inputs = inputs or {}
    outputs = outputs or {}
    data = {
        "stage": stage,
        "created_at_utc": utc_now(),
        "config_version": config.get("config_version"),
        "schema_version": config.get("schema_version"),
        "prompt_version": config.get("prompt_version"),
        "config_sha256": sha256_file(CONFIG_PATH),
        "models": config.get("models", {}),
        "providers": config.get("providers", {}),
        "reasoning_effort": config.get("reasoning_effort", {}),
        "temperatures": config.get("temperatures", {}),
        "seeds": config.get("seeds", {}),
        "prompts": {
            name: sha256_file(PROJECT_ROOT / "prompts" / name)
            for name in prompt_files
        },
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path) if path.exists() else None,
            }
            for name, path in inputs.items()
        },
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path) if path.exists() else None,
            }
            for name, path in outputs.items()
        },
        "stats": stats or {},
        "extra": extra or {},
    }
    path = manifest_path(config, stage)
    write_json(path, data)
    return path


def render_prompt(filename: str, values: dict[str, str]) -> str:
    text = (PROJECT_ROOT / "prompts" / filename).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def category_splits(category: str) -> tuple[str, str, str]:
    held = set(category.split("_")) if category != "id" else set()
    return (
        "held_out" if "animal" in held else "seen",
        "held_out" if "value" in held else "seen",
        "held_out" if "context" in held else "seen",
    )


def scenario_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    seen = load_domains(resolve_path(config, "training_domains"))
    held = load_domains(resolve_path(config, "held_out_domains"))
    per_category = int(config["scenario_generation"]["scenarios_per_category"])
    tradeoff_count = round(per_category * float(config["scenario_generation"]["tradeoff_fraction"]))
    rng = random.Random(int(config["seeds"]["scenario_plan"]))

    assignments: list[dict[str, Any]] = []
    next_id = 1
    for category in CATEGORIES:
        a_split, v_split, c_split = category_splits(category)
        animals = held["animals"] if a_split == "held_out" else seen["animals"]
        values = held["values"] if v_split == "held_out" else seen["values"]
        contexts = held["contexts"] if c_split == "held_out" else seen["contexts"]
        combos = list(product(animals, values, contexts))
        rng.shuffle(combos)

        category_rows = []
        for i in range(per_category):
            animal, value, context = combos[i % len(combos)]
            category_rows.append(
                {
                    "id": next_id,
                    "category": category,
                    "animal": animal,
                    "value": value,
                    "context": context,
                    "animal_split": a_split,
                    "value_split": v_split,
                    "context_split": c_split,
                    "tradeoff": i < tradeoff_count,
                }
            )
            next_id += 1
        rng.shuffle(category_rows)
        assignments.extend(category_rows)

    return assignments


def validate_scenarios(rows: list[dict[str, Any]], expected_count: int) -> None:
    validator = Draft202012Validator(SCENARIO_SCHEMA)
    ids: set[int] = set()
    for row in rows:
        errors = list(validator.iter_errors(row))
        if errors:
            raise RuntimeError(f"Invalid scenario row {row.get('id')}: {errors[0].message}")
        if row["id"] in ids:
            raise RuntimeError(f"Duplicate scenario id {row['id']}")
        ids.add(row["id"])
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} scenarios, found {len(rows)}")


class OpenRouterClient:
    def __init__(self, config: dict[str, Any]) -> None:
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ConfigError("OPENROUTER_API_KEY is required.")

        api = config["api"]
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=str(api["base_url"]).rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=float(api["timeout_seconds"]),
        )
        self.semaphore = asyncio.Semaphore(int(api["concurrency"]))
        self.request_count = 0
        self.input_tokens = 0
        self.output_tokens = 0

    async def request_json(
        self,
        *,
        model_key: str,
        temperature_key: str,
        seed_key: str,
        prompt: str,
        schema: dict,
        schema_name: str,
    ) -> dict[str, Any]:
        model = self.config["models"].get(model_key)
        if not model:
            raise ConfigError(f"models.{model_key} must be configured.")

        api = self.config["api"]
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(self.config["temperatures"][temperature_key]),
            "max_tokens": int(api["max_output_tokens"]),
            "seed": int(self.config["seeds"][seed_key]),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        }
        reasoning = (self.config.get("reasoning_effort") or {}).get(model_key)
        if reasoning:
            payload["reasoning"] = {"effort": reasoning}

        provider_cfg = dict((self.config.get("providers") or {}).get(model_key) or {})
        if api.get("require_parameters", True):
            provider_cfg["require_parameters"] = True
        if provider_cfg:
            payload["provider"] = provider_cfg

        last_error: Exception | None = None
        for attempt in range(int(api["max_attempts"])):
            try:
                async with self.semaphore:
                    response = await self.client.post("/chat/completions", json=payload)
                response.raise_for_status()
                self.request_count += 1
                body = response.json()
                usage = body.get("usage") or {}
                self.input_tokens += int(usage.get("prompt_tokens") or 0)
                self.output_tokens += int(usage.get("completion_tokens") or 0)
                content = body["choices"][0]["message"]["content"]
                parsed = json.loads(content) if isinstance(content, str) else content
                Draft202012Validator(schema).validate(parsed)
                return parsed
            except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                if attempt + 1 >= int(api["max_attempts"]):
                    break
                delay = min(
                    float(api["max_backoff_seconds"]),
                    float(api["backoff_seconds"]) * (2 ** attempt),
                )
                await asyncio.sleep(delay)
        raise RuntimeError(f"OpenRouter request failed: {last_error}")

    def stats(self) -> dict[str, int]:
        return {
            "openrouter_requests": self.request_count,
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
        }

    async def close(self) -> None:
        await self.client.aclose()
