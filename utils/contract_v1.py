"""Validation against the canonical analysis contract v1 schemas.

The schemas themselves live in `runner-analysis-pipeline/contracts/analysis/v1/`
(a sibling repo — see `config.PIPELINE_ROOT`), not duplicated here, so backend,
Server pipeline, and (eventually) the Swift/Dart sides all validate against the
exact same files (規劃書 §13 執行規則: "所有跨 repo schema 變更先修改 canonical
contract，再更新各語言 adapter").
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

from config import PIPELINE_ROOT

CONTRACT_V1_DIR = PIPELINE_ROOT / "contracts" / "analysis" / "v1"


class ContractValidationError(ValueError):
    """Raised when a document fails validation against a contract v1 schema."""


@lru_cache(maxsize=None)
def _load_schema(name: str) -> dict:
    schema_path = CONTRACT_V1_DIR / f"{name}.schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(
            f"contract v1 schema '{name}' not found at {schema_path}; "
            "is RUNNER_ANALYSIS_PIPELINE_ROOT pointed at a checkout that has contracts/analysis/v1/?"
        )
    return json.loads(schema_path.read_text())


def validate_against_schema(document: dict, *, schema_name: str) -> None:
    schema = _load_schema(schema_name)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    errors = sorted(
        validator_cls(schema).iter_errors(document), key=lambda e: list(e.path)
    )
    if errors:
        details = "; ".join(f"{list(e.path)}: {e.message}" for e in errors[:10])
        raise ContractValidationError(
            f"document does not match contract v1 '{schema_name}' schema: {details}"
        )


def validate_manifest(manifest: dict) -> None:
    validate_against_schema(manifest, schema_name="analysis-result-manifest")


def validate_comparison_report(report: dict) -> None:
    validate_against_schema(report, schema_name="comparison-report")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_safe_relative_path(relative_path: str) -> bool:
    """Defense in depth: the manifest schema's `relative_path` pattern already
    forbids `..` segments and a leading `/`, but ingestion re-checks server-side
    rather than trusting a schema-shaped string to double as a safe filesystem
    path."""
    if not relative_path or relative_path.startswith("/"):
        return False
    parts = Path(relative_path).parts
    return ".." not in parts and not any(p in ("", ".") for p in parts)
