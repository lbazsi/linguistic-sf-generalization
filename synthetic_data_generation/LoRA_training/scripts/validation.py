from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from common import read_jsonl_tolerant
from schemas import PAIR_SCHEMA


def make_issue(
    *,
    feature: str,
    stage: str,
    issue_type: str,
    message: str,
    item_id: int | None = None,
    field: str | None = None,
) -> dict:
    return {
        "id": item_id,
        "feature": feature,
        "stage": stage,
        "issue_type": issue_type,
        "field": field,
        "message": message,
    }


def _path_to_field(path: Any) -> str | None:
    parts = [str(part) for part in path]
    return ".".join(parts) if parts else None


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def validate_item(
    item: Any,
    *,
    feature_spec: dict,
    canonical: dict[int, dict],
    dataset_size: int,
    stage: str,
) -> list[dict]:
    feature_name = feature_spec["name"]
    item_id = item.get("id") if isinstance(item, dict) and isinstance(item.get("id"), int) else None
    issues: list[dict] = []

    schema_errors = sorted(
        Draft202012Validator(PAIR_SCHEMA).iter_errors(item),
        key=lambda error: list(error.path),
    )
    for error in schema_errors:
        issues.append(
            make_issue(
                feature=feature_name,
                stage=stage,
                issue_type="schema_validation",
                field=_path_to_field(error.path),
                message=error.message,
                item_id=item_id,
            )
        )

    if not isinstance(item, dict):
        return issues

    raw_id = item.get("id")
    if isinstance(raw_id, int):
        if raw_id < 1 or raw_id > dataset_size:
            issues.append(
                make_issue(
                    feature=feature_name,
                    stage=stage,
                    issue_type="id_out_of_range",
                    field="id",
                    message=f"ID must be in 1..{dataset_size}.",
                    item_id=raw_id,
                )
            )
    else:
        return issues

    if item.get("feature") != feature_name:
        issues.append(
            make_issue(
                feature=feature_name,
                stage=stage,
                issue_type="feature_mismatch",
                field="feature",
                message=f"Expected feature '{feature_name}'.",
                item_id=raw_id,
            )
        )

    source = canonical.get(raw_id)
    manipulation = feature_spec["manipulation_level"]
    expected_canonical_language = (
        feature_spec["canonical_language"]
        if manipulation == "cross_linguistic"
        else feature_spec["language"]
    )
    expected_variant_language = (
        feature_spec["feature_variant_language"]
        if manipulation == "cross_linguistic"
        else feature_spec["language"]
    )

    expected_metadata = {
        "manipulation_level": manipulation,
        "canonical_language": expected_canonical_language,
        "feature_variant_language": expected_variant_language,
    }
    for field, expected in expected_metadata.items():
        if item.get(field) != expected:
            issues.append(
                make_issue(
                    feature=feature_name,
                    stage=stage,
                    issue_type=f"{field}_mismatch",
                    field=field,
                    message=f"Expected {field} '{expected}'.",
                    item_id=raw_id,
                )
            )

    if source is not None:
        for field in ["animal", "value", "context"]:
            expected = source[field]
            if item.get(field) != expected:
                issues.append(
                    make_issue(
                        feature=feature_name,
                        stage=stage,
                        issue_type=f"{field}_mismatch",
                        field=field,
                        message=f"Expected semantic-anchor {field} '{expected}'.",
                        item_id=raw_id,
                    )
                )
        if item.get("semantic_anchor") != source["canonical"]:
            issues.append(
                make_issue(
                    feature=feature_name,
                    stage=stage,
                    issue_type="semantic_anchor_mismatch",
                    field="semantic_anchor",
                    message="semantic_anchor must exactly match the reviewed English canonical corpus.",
                    item_id=raw_id,
                )
            )
        if manipulation == "within_language" and item.get("canonical") != source["canonical"]:
            issues.append(
                make_issue(
                    feature=feature_name,
                    stage=stage,
                    issue_type="canonical_mismatch",
                    field="canonical",
                    message="Within-language canonical must exactly match the reviewed English canonical corpus.",
                    item_id=raw_id,
                )
            )

    canonical_text = item.get("canonical")
    variant = item.get("feature_variant")
    if isinstance(canonical_text, str) and not canonical_text.strip():
        issues.append(
            make_issue(
                feature=feature_name,
                stage=stage,
                issue_type="empty_text",
                field="canonical",
                message="canonical must contain non-whitespace text.",
                item_id=raw_id,
            )
        )
    if isinstance(variant, str) and not variant.strip():
        issues.append(
            make_issue(
                feature=feature_name,
                stage=stage,
                issue_type="empty_text",
                field="feature_variant",
                message="feature_variant must contain non-whitespace text.",
                item_id=raw_id,
            )
        )
    if isinstance(canonical_text, str) and isinstance(variant, str):
        if _normalized_text(canonical_text) == _normalized_text(variant):
            issues.append(
                make_issue(
                    feature=feature_name,
                    stage=stage,
                    issue_type="identical_pair",
                    field=None,
                    message="canonical and feature_variant are identical after whitespace normalization.",
                    item_id=raw_id,
                )
            )

    return issues


def deterministic_review_file(
    path: Path,
    *,
    feature_spec: dict,
    canonical: dict[int, dict],
    dataset_size: int,
    stage: str,
    active_max_id: int | None = None,
) -> tuple[dict[int, list[dict]], list[dict]]:
    feature_name = feature_spec["name"]
    expected_max_id = active_max_id if active_max_id is not None else dataset_size
    parsed, parse_errors = read_jsonl_tolerant(path)
    issues: list[dict] = []
    by_id: dict[int, list[dict]] = defaultdict(list)

    for line_number, message in parse_errors:
        issues.append(
            make_issue(
                feature=feature_name,
                stage=stage,
                issue_type="invalid_json",
                field=f"line:{line_number}",
                message=message,
                item_id=None,
            )
        )

    for line_number, item in parsed:
        if (
            active_max_id is not None
            and isinstance(item, dict)
            and isinstance(item.get("id"), int)
            and item["id"] > active_max_id
            and item["id"] <= dataset_size
        ):
            continue
        item_issues = validate_item(
            item,
            feature_spec=feature_spec,
            canonical=canonical,
            dataset_size=dataset_size,
            stage=stage,
        )
        issues.extend(item_issues)

        if isinstance(item, dict) and isinstance(item.get("id"), int):
            item_id = item["id"]
            if 1 <= item_id <= expected_max_id:
                by_id[item_id].append(item)

    for item_id in range(1, expected_max_id + 1):
        candidates = by_id.get(item_id, [])
        if not candidates:
            issues.append(
                make_issue(
                    feature=feature_name,
                    stage=stage,
                    issue_type="missing_id",
                    field="id",
                    message=f"Expected ID {item_id} is missing.",
                    item_id=item_id,
                )
            )
        elif len(candidates) > 1:
            issues.append(
                make_issue(
                    feature=feature_name,
                    stage=stage,
                    issue_type="duplicate_id",
                    field="id",
                    message=f"ID {item_id} occurs {len(candidates)} times.",
                    item_id=item_id,
                )
            )

    issues.sort(
        key=lambda issue: (
            issue["id"] is None,
            issue["id"] if issue["id"] is not None else 10**18,
            issue["issue_type"],
            issue["field"] or "",
        )
    )
    return dict(by_id), issues


def clean_singletons(
    path: Path,
    *,
    feature_spec: dict,
    canonical: dict[int, dict],
    dataset_size: int,
    stage: str,
    active_max_id: int | None = None,
) -> dict[int, dict]:
    by_id, _ = deterministic_review_file(
        path,
        feature_spec=feature_spec,
        canonical=canonical,
        dataset_size=dataset_size,
        stage=stage,
        active_max_id=active_max_id,
    )
    clean: dict[int, dict] = {}
    for item_id, candidates in by_id.items():
        if len(candidates) != 1:
            continue
        candidate = candidates[0]
        if not validate_item(
            candidate,
            feature_spec=feature_spec,
            canonical=canonical,
            dataset_size=dataset_size,
            stage=stage,
        ):
            clean[item_id] = candidate
    return clean
