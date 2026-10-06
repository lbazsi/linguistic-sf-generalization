from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
import yaml
from dotenv import load_dotenv
from jsonschema import Draft202012Validator, ValidationError

from schemas import FEATURE_SCHEMA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "configs.yaml"


class ConfigError(RuntimeError):
    pass


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config() -> dict:
    config = load_yaml(CONFIG_PATH)
    if not isinstance(config, dict):
        raise ConfigError(f"{CONFIG_PATH} must contain a YAML mapping.")

    required = [
        "config_version",
        "schema_version",
        "prompt_version",
        "models",
        "temperatures",
        "concurrency",
        "batch_size",
        "seeds",
        "retry_limits",
        "dataset_size",
        "canonical_language",
        "api",
        "paths",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Missing config keys: {', '.join(missing)}")

    if not isinstance(config["dataset_size"], int) or config["dataset_size"] <= 0:
        raise ConfigError("dataset_size must be a positive integer.")
    if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
        raise ConfigError("batch_size must be a positive integer.")
    if not isinstance(config["concurrency"], int) or config["concurrency"] <= 0:
        raise ConfigError("concurrency must be a positive integer.")
    if not isinstance(config["canonical_language"], str) or not config["canonical_language"].strip():
        raise ConfigError("canonical_language must be a non-empty string.")

    retries = config["retry_limits"]
    if not isinstance(retries, dict) or int(retries.get("max_attempts", 0)) <= 0:
        raise ConfigError("retry_limits.max_attempts must be positive.")

    return config


def resolve_path(config: dict, key: str) -> Path:
    value = config["paths"].get(key)
    if not value:
        raise ConfigError(f"paths.{key} is not configured.")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_directories(config: dict) -> None:
    for key in [
        "canonical",
        "raw",
        "first_review",
        "final",
        "deterministic_issues",
        "nondeterministic_issues",
        "manifests",
    ]:
        resolve_path(config, key).mkdir(parents=True, exist_ok=True)


def load_domains(config: dict) -> dict[str, list[str]]:
    path = resolve_path(config, "domains")
    domains = load_yaml(path)
    if not isinstance(domains, dict):
        raise ConfigError(f"{path} must contain a YAML mapping.")
    required = ["animals", "values", "contexts"]
    missing = [key for key in required if key not in domains]
    if missing:
        raise ConfigError(f"{path} is missing domain lists: {', '.join(missing)}")

    normalized: dict[str, list[str]] = {}
    for key in required:
        values = domains[key]
        if not isinstance(values, list) or not values:
            raise ConfigError(f"{path}: {key} must be a non-empty YAML list.")
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ConfigError(f"{path}: every {key} entry must be a non-empty string.")
        cleaned = [value.strip() for value in values]
        if len(cleaned) != len(set(cleaned)):
            raise ConfigError(f"{path}: {key} entries must be unique.")
        normalized[key] = cleaned
    return normalized


def load_feature(path: Path) -> dict:
    feature = load_yaml(path)
    errors = sorted(Draft202012Validator(FEATURE_SCHEMA).iter_errors(feature), key=lambda e: list(e.path))
    if errors:
        detail = "; ".join(error.message for error in errors[:5])
        raise ConfigError(f"Invalid feature YAML {path}: {detail}")
    if feature["name"] != path.stem:
        raise ConfigError(
            f"Feature name '{feature['name']}' must match filename stem '{path.stem}'."
        )
    return feature


def select_features(config: dict, requested: Iterable[str] | None = None) -> list[tuple[Path, dict]]:
    feature_dir = resolve_path(config, "features")
    paths = sorted(feature_dir.glob("*.yaml"))
    if not paths:
        raise ConfigError(f"No feature YAML files found in {feature_dir}.")

    loaded = [(path, load_feature(path)) for path in paths]
    requested_set = set(requested or [])
    if not requested_set:
        return loaded

    available = {feature["name"] for _, feature in loaded}
    missing = sorted(requested_set - available)
    if missing:
        raise ConfigError(f"Unknown requested features: {', '.join(missing)}")
    return [(path, feature) for path, feature in loaded if feature["name"] in requested_set]


def feature_yaml_text(feature: dict) -> str:
    return yaml.safe_dump(feature, sort_keys=False, allow_unicode=True).strip()


def domain_plan(
    domains: dict[str, list[str]],
    dataset_size: int,
    seed: int,
) -> dict[int, dict[str, str]]:
    import itertools
    import random

    cells = list(itertools.product(domains["animals"], domains["values"], domains["contexts"]))
    if not cells:
        raise ConfigError("Domain Cartesian product must not be empty.")

    base, remainder = divmod(dataset_size, len(cells))
    assignments: list[tuple[str, str, str]] = []
    for index, cell in enumerate(cells):
        assignments.extend([cell] * (base + (1 if index < remainder else 0)))

    rng = random.Random(seed)
    rng.shuffle(assignments)

    return {
        item_id: {
            "animal": animal,
            "value": value,
            "context": context,
        }
        for item_id, (animal, value, context) in enumerate(assignments, start=1)
    }


def batches_by_domain(
    ids: Iterable[int],
    plan: dict[int, dict[str, str]],
    batch_size: int,
) -> list[tuple[dict[str, str], list[int]]]:
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for item_id in sorted(ids):
        assignment = plan[item_id]
        key = (assignment["animal"], assignment["value"], assignment["context"])
        grouped[key].append(item_id)

    batches: list[tuple[dict[str, str], list[int]]] = []
    for (animal, value, context), domain_ids in grouped.items():
        domain = {"animal": animal, "value": value, "context": context}
        for start in range(0, len(domain_ids), batch_size):
            batches.append((domain, domain_ids[start : start + batch_size]))
    return batches


def generated_canonical_path(config: dict) -> Path:
    return resolve_path(config, "canonical") / "generated.jsonl"


def canonical_corpus_path(config: dict) -> Path:
    return resolve_path(config, "canonical") / "corpus.jsonl"


def load_canonical_corpus(config: dict) -> dict[int, dict]:
    path = canonical_corpus_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing canonical corpus: {path}. Run scripts/01_generate.py first."
        )
    candidates = index_items(path)
    dataset_size = config["dataset_size"]
    corpus: dict[int, dict] = {}
    for item_id in range(1, dataset_size + 1):
        rows = candidates.get(item_id, [])
        if len(rows) != 1:
            raise RuntimeError(
                f"Canonical corpus must contain exactly one row for ID {item_id}; found {len(rows)}."
            )
        corpus[item_id] = rows[0]
    return corpus


def batches(ids: Iterable[int], batch_size: int) -> list[list[int]]:
    ordered = sorted(ids)
    return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]


def prompt_path(config: dict, filename: str) -> Path:
    return resolve_path(config, "prompts") / filename


def render_prompt(config: dict, filename: str, values: dict[str, str]) -> str:
    path = prompt_path(config, filename)
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def read_jsonl_tolerant(path: Path) -> tuple[list[tuple[int, Any]], list[tuple[int, str]]]:
    parsed: list[tuple[int, Any]] = []
    errors: list[tuple[int, str]] = []
    if not path.exists():
        return parsed, errors

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                errors.append((line_number, "Empty line"))
                continue
            try:
                parsed.append((line_number, json.loads(stripped)))
            except json.JSONDecodeError as exc:
                errors.append((line_number, f"Invalid JSON: {exc.msg}"))
    return parsed, errors


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> None:
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


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def index_items(path: Path) -> dict[int, list[dict]]:
    parsed, _ = read_jsonl_tolerant(path)
    result: dict[int, list[dict]] = defaultdict(list)
    for _, item in parsed:
        if isinstance(item, dict) and isinstance(item.get("id"), int):
            result[item["id"]].append(item)
    return dict(result)


def load_issues(path: Path) -> list[dict]:
    parsed, errors = read_jsonl_tolerant(path)
    if errors:
        raise RuntimeError(f"Issue file {path} contains invalid JSONL.")
    issues = []
    for _, value in parsed:
        if not isinstance(value, dict):
            raise RuntimeError(f"Issue file {path} contains a non-object row.")
        issues.append(value)
    return issues


def issues_by_id(issues: Iterable[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for issue in issues:
        item_id = issue.get("id")
        if isinstance(item_id, int):
            grouped[item_id].append(issue)
    return dict(grouped)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunManifest:
    def __init__(
        self,
        *,
        stage: str,
        config: dict,
        feature_paths: Iterable[Path],
        prompt_filenames: Iterable[str] = (),
    ) -> None:
        self.stage = stage
        self.config = config
        self.feature_paths = list(feature_paths)
        self.prompt_filenames = list(prompt_filenames)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        self.path = resolve_path(config, "manifests") / f"{run_id}_{stage}.json"

        domains_file = resolve_path(config, "domains")
        prompt_hashes = {
            name: sha256_file(prompt_path(config, name)) for name in self.prompt_filenames
        }
        feature_hashes = {path.stem: sha256_file(path) for path in self.feature_paths}

        self.data = {
            "run_id": run_id,
            "stage": stage,
            "status": "running",
            "started_at": utc_now(),
            "finished_at": None,
            "config_version": config["config_version"],
            "schema_version": config["schema_version"],
            "prompt_version": config["prompt_version"],
            "config_hash": sha256_file(CONFIG_PATH),
            "domains_hash": sha256_file(domains_file),
            "feature_hashes": feature_hashes,
            "prompt_hashes": prompt_hashes,
            "models": config["models"],
            "temperatures": config["temperatures"],
            "providers": config.get("providers", {}),
            "reasoning_effort": config.get("reasoning_effort", {}),
            "seeds": config["seeds"],
            "dataset_size": config["dataset_size"],
            "batch_size": config["batch_size"],
            "concurrency": config["concurrency"],
            "stats": {},
            "error": None,
        }
        write_json_atomic(self.path, self.data)

    def finish(self, stats: dict | None = None) -> None:
        self.data["status"] = "completed"
        self.data["finished_at"] = utc_now()
        self.data["stats"] = stats or {}
        write_json_atomic(self.path, self.data)

    def fail(self, exc: BaseException, stats: dict | None = None) -> None:
        self.data["status"] = "failed"
        self.data["finished_at"] = utc_now()
        self.data["stats"] = stats or {}
        self.data["error"] = f"{type(exc).__name__}: {exc}"
        write_json_atomic(self.path, self.data)


class OpenRouterClient:
    def __init__(self, config: dict) -> None:
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ConfigError("OPENROUTER_API_KEY is required in the environment or .env.")

        api = config["api"]
        self.config = config
        self.base_url = str(api["base_url"]).rstrip("/")
        self.timeout = float(api["timeout_seconds"])
        self.max_output_tokens = int(api["max_output_tokens"])
        self.require_parameters = bool(api.get("require_parameters", True))
        self.semaphore = asyncio.Semaphore(int(config["concurrency"]))
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        self.request_count = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def _role_setting(self, section: str, role: str) -> Any:
        value = self.config[section].get(role)
        if value is None or value == "":
            raise ConfigError(f"{section}.{role} must be configured before this stage can run.")
        return value

    async def request_json(
        self,
        *,
        role: str,
        prompt: str,
        schema: dict,
        schema_name: str,
    ) -> dict:
        model = self._role_setting("models", role)
        temperature = float(self._role_setting("temperatures", role))
        seed = self.config["seeds"].get(role)

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": self.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if seed is not None:
            payload["seed"] = int(seed)

        reasoning = (self.config.get("reasoning_effort") or {}).get(role)
        if reasoning:
            payload["reasoning"] = {"effort": reasoning}

        provider_cfg = dict((self.config.get("providers") or {}).get(role) or {})
        if self.require_parameters:
            provider_cfg["require_parameters"] = True
        if provider_cfg:
            payload["provider"] = provider_cfg

        retry = self.config["retry_limits"]
        max_attempts = int(retry["max_attempts"])
        base_backoff = float(retry.get("backoff_seconds", 1))
        max_backoff = float(retry.get("max_backoff_seconds", 30))
        last_error: BaseException | None = None

        for attempt in range(max_attempts):
            try:
                async with self.semaphore:
                    response = await self.client.post(
                        self.base_url + "/chat/completions",
                        json=payload,
                    )
                self.request_count += 1

                if response.status_code >= 400:
                    message = response.text[:1000]
                    error = RuntimeError(
                        f"OpenRouter returned HTTP {response.status_code}: {message}"
                    )
                    if response.status_code not in {408, 409, 429} and response.status_code < 500:
                        raise error
                    raise error

                body = response.json()
                usage = body.get("usage") or {}
                self.input_tokens += int(usage.get("prompt_tokens") or 0)
                self.output_tokens += int(usage.get("completion_tokens") or 0)

                content = body["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                parsed = json.loads(content) if isinstance(content, str) else content
                Draft202012Validator(schema).validate(parsed)
                if not isinstance(parsed, dict):
                    raise RuntimeError("Structured response root must be a JSON object.")
                return parsed

            except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValidationError, RuntimeError) as exc:
                last_error = exc
                if attempt + 1 >= max_attempts:
                    break
                delay = min(max_backoff, base_backoff * (2 ** attempt))
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"OpenRouter request failed after {max_attempts} attempts: {last_error}"
        )

    def stats(self) -> dict:
        return {
            "openrouter_requests": self.request_count,
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
        }

    async def close(self) -> None:
        await self.client.aclose()
