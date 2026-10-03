#!/usr/bin/env python3
"""Collect one-shot BFCL tool-calling logs for an evolutionary routing testbed.

The collector deliberately separates three objects:

* task context: a pinned BFCL single-turn task category;
* agent trait: a persistent four-dimensional tool-use policy;
* fitness: deterministic BFCL-compatible AST success on held-out tasks.

The output keeps the ``contents/evaluations/audits`` checkpoint layout used by
the existing replay pipeline.  The legacy social-score columns are explicit
aliases of the binary task success, not additional LLM evaluations.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import shutil
import ssl
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


BFCL_SOURCE_REPOSITORY = "https://github.com/ShishirPatil/gorilla"
BFCL_PINNED_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_DATA_VERSION = "BFCL_v4"
BFCL_LICENSE = "Apache-2.0"
BFCL_CATEGORIES: Tuple[str, ...] = (
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
)

DEFAULT_SUBSET_DIR = Path(__file__).with_name("benchmarks") / "bfcl"
DEFAULT_SUBSET_PATH = DEFAULT_SUBSET_DIR / "tasks.jsonl"
SOCIAL_CONTEXT_AXES: Tuple[str, ...] = (
    "controversy",
    "emotional_load",
    "interpersonalness",
    "evidentiality",
    "engagement_baitness",
    "public_vs_personal",
    "safety_risk_hint",
)


@dataclass(frozen=True)
class ModelSpec:
    provider: str = "openai"
    model: str = "gpt-5.4-mini"


@dataclass(frozen=True)
class ContinuousTraitProfile:
    coords: Dict[str, float]
    labels: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class Agent:
    agent_id: str
    trait: ContinuousTraitProfile
    model: ModelSpec


@dataclass(frozen=True)
class BFCLTask:
    task_id: str
    category: str
    messages: Tuple[Dict[str, Any], ...]
    functions: Tuple[Dict[str, Any], ...]
    ground_truth: Tuple[Dict[str, Any], ...]
    source_sha256: str = ""

    @property
    def user_text(self) -> str:
        chunks: List[str] = []
        for message in self.messages:
            if str(message.get("role", "")) == "user":
                chunks.append(str(message.get("content", "")))
        return "\n".join(chunks).strip()


def default_trait_space() -> Dict[str, Any]:
    return {
        "dimensions": [
            {
                "name": "action_propensity",
                "min": -1.0,
                "max": 1.0,
                "prompt_negative": (
                    "use a high action threshold: abstain rather than call a tool "
                    "when the request is irrelevant or materially underspecified"
                ),
                "prompt_positive": (
                    "use a low action threshold: call an applicable tool whenever "
                    "the user's likely requested action is sufficiently clear"
                ),
            },
            {
                "name": "intent_interpretation",
                "min": -1.0,
                "max": 1.0,
                "prompt_negative": (
                    "interpret requests literally and avoid extending beyond their "
                    "explicitly stated intent"
                ),
                "prompt_positive": (
                    "infer implicit intent from the full request and tool descriptions "
                    "when the intended operation is reasonably clear"
                ),
            },
            {
                "name": "argument_policy",
                "min": -1.0,
                "max": 1.0,
                "prompt_negative": (
                    "supply only arguments explicitly supported by the request; do not "
                    "invent missing values"
                ),
                "prompt_positive": (
                    "use schema defaults and strongly implied values when doing so is "
                    "necessary to complete the requested operation"
                ),
            },
            {
                "name": "orchestration_policy",
                "min": -1.0,
                "max": 1.0,
                "prompt_negative": (
                    "prefer minimal atomic invocation and avoid unnecessary batching"
                ),
                "prompt_positive": (
                    "emit all independent required calls together and use parallel "
                    "orchestration when the request contains multiple operations"
                ),
            },
        ],
        "sampling": {
            "method": "axis_anchors",
            "n_agents": 9,
            "seed": 42,
        },
        "discretization": {
            "enabled": True,
            "cutpoints": [-0.33, 0.33],
        },
    }


@dataclass
class LogGenerationConfig:
    model: ModelSpec = field(default_factory=ModelSpec)
    trait_space: Dict[str, Any] = field(default_factory=default_trait_space)
    subset_path: str = field(default_factory=lambda: str(DEFAULT_SUBSET_PATH))
    n_tasks: int = 0
    task_sample_seed: int = 42
    random_seed: int = 42
    run_label: str = ""
    temperature: float = 0.001
    max_completion_tokens: int = 512
    api_max_workers: int = 4
    api_timeout_sec: float = 120.0
    api_max_retries: int = 4
    sleep_between_calls_sec: float = 0.0
    parallel_tool_calls: bool = True
    save_prompt: bool = True
    save_raw_response: bool = True
    mock_mode: str = ""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def save_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def save_jsonl_atomic(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _download_bytes(url: str, timeout_sec: float = 60.0) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "evolution-theory-bfcl-subset/1.0"},
    )
    try:
        import certifi

        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(
            request,
            timeout=timeout_sec,
            context=ssl_context,
        ) as response:
            return response.read()
    except (ImportError, OSError):
        curl_path = shutil.which("curl")
        if curl_path is None:
            raise
        completed = subprocess.run(
            [
                curl_path,
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--max-time",
                str(max(1, int(timeout_sec))),
                "--user-agent",
                "evolution-theory-bfcl-subset/1.0",
                url,
            ],
            check=True,
            capture_output=True,
        )
        return completed.stdout


def _parse_jsonl_bytes(payload: bytes, source: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise ValueError(f"{source}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def bfcl_raw_url(commit: str, relative_path: str) -> str:
    return (
        "https://raw.githubusercontent.com/ShishirPatil/gorilla/"
        f"{commit}/berkeley-function-call-leaderboard/bfcl_eval/data/{relative_path}"
    )


def prepare_bfcl_subset(
    output_dir: Path,
    tasks_per_category: int,
    seed: int,
    commit: str = BFCL_PINNED_COMMIT,
) -> Dict[str, Any]:
    """Download, balance, and freeze a deterministic BFCL subset."""
    if tasks_per_category <= 0:
        raise ValueError("tasks_per_category must be positive")

    downloaded_hashes: Dict[str, str] = {}
    available_counts: Dict[str, int] = {}
    selected_rows: List[Dict[str, Any]] = []
    selected_ids: Dict[str, List[str]] = {}

    for category_index, category in enumerate(BFCL_CATEGORIES):
        prompt_relative = f"{BFCL_DATA_VERSION}_{category}.json"
        prompt_payload = _download_bytes(bfcl_raw_url(commit, prompt_relative))
        downloaded_hashes[prompt_relative] = sha256(prompt_payload).hexdigest()
        prompt_rows = _parse_jsonl_bytes(prompt_payload, prompt_relative)
        available_counts[category] = len(prompt_rows)

        answer_by_id: Dict[str, Dict[str, Any]] = {}
        if category != "irrelevance":
            answer_relative = f"possible_answer/{BFCL_DATA_VERSION}_{category}.json"
            answer_payload = _download_bytes(bfcl_raw_url(commit, answer_relative))
            downloaded_hashes[answer_relative] = sha256(answer_payload).hexdigest()
            answer_rows = _parse_jsonl_bytes(answer_payload, answer_relative)
            answer_by_id = {str(row["id"]): row for row in answer_rows}

        if tasks_per_category > len(prompt_rows):
            raise ValueError(
                f"{category}: requested {tasks_per_category}, only {len(prompt_rows)} available"
            )
        category_rng = random.Random(seed + 1009 * category_index)
        chosen = category_rng.sample(prompt_rows, tasks_per_category)
        chosen.sort(key=lambda row: int(str(row["id"]).rsplit("_", 1)[-1]))
        selected_ids[category] = [str(row["id"]) for row in chosen]

        for prompt_row in chosen:
            task_id = str(prompt_row["id"])
            question = prompt_row.get("question")
            if not isinstance(question, list) or len(question) != 1:
                raise ValueError(f"{task_id}: expected exactly one BFCL question turn")
            messages = question[0]
            functions = prompt_row.get("function")
            if not isinstance(messages, list) or not isinstance(functions, list):
                raise ValueError(f"{task_id}: malformed messages or functions")
            ground_truth = (
                []
                if category == "irrelevance"
                else answer_by_id[task_id].get("ground_truth", [])
            )
            normalized = {
                "task_id": task_id,
                "category": category,
                "messages": messages,
                "functions": functions,
                "ground_truth": ground_truth,
            }
            normalized["source_sha256"] = sha256_json(normalized)
            selected_rows.append(normalized)

    selected_rows.sort(
        key=lambda row: (
            BFCL_CATEGORIES.index(str(row["category"])),
            int(str(row["task_id"]).rsplit("_", 1)[-1]),
        )
    )
    tasks_path = output_dir / "tasks.jsonl"
    save_jsonl_atomic(selected_rows, tasks_path)
    manifest = {
        "benchmark": "Berkeley Function Calling Leaderboard",
        "data_version": BFCL_DATA_VERSION,
        "source_repository": BFCL_SOURCE_REPOSITORY,
        "source_commit": commit,
        "license": BFCL_LICENSE,
        "categories": list(BFCL_CATEGORIES),
        "available_counts": available_counts,
        "tasks_per_category": tasks_per_category,
        "total_tasks": len(selected_rows),
        "sampling_seed": seed,
        "selected_task_ids": selected_ids,
        "downloaded_file_sha256": downloaded_hashes,
        "tasks_jsonl_sha256": sha256(tasks_path.read_bytes()).hexdigest(),
    }
    save_json_atomic(manifest, output_dir / "manifest.json")
    return manifest


def load_bfcl_tasks(path: Path) -> List[BFCLTask]:
    manifest_path = path.with_name("manifest.json")
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        expected_hash = str(manifest.get("tasks_jsonl_sha256", ""))
        actual_hash = sha256(path.read_bytes()).hexdigest()
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(
                f"{path}: SHA-256 does not match {manifest_path}; "
                "regenerate the pinned subset"
            )

    tasks: List[BFCLTask] = []
    seen_ids: set[str] = set()
    for row in load_jsonl(path):
        task_id = str(row["task_id"])
        category = str(row["category"])
        if task_id in seen_ids:
            raise ValueError(f"duplicate task id: {task_id}")
        if category not in BFCL_CATEGORIES:
            raise ValueError(f"{task_id}: unsupported category {category}")
        seen_ids.add(task_id)
        task = BFCLTask(
            task_id=task_id,
            category=category,
            messages=tuple(copy.deepcopy(row["messages"])),
            functions=tuple(copy.deepcopy(row["functions"])),
            ground_truth=tuple(copy.deepcopy(row.get("ground_truth", []))),
            source_sha256=str(row.get("source_sha256", "")),
        )
        if category != "irrelevance" and not task.ground_truth:
            raise ValueError(f"{task_id}: missing ground truth")
        if task.source_sha256:
            hash_input = {key: value for key, value in row.items() if key != "source_sha256"}
            if sha256_json(hash_input) != task.source_sha256:
                raise ValueError(f"{task_id}: row source SHA-256 mismatch")
        tasks.append(task)
    if not tasks:
        raise ValueError(f"no BFCL tasks found in {path}")
    return tasks


def _trait_dimensions(trait_space: Mapping[str, Any]) -> List[Dict[str, Any]]:
    dimensions = trait_space.get("dimensions", [])
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("trait_space.dimensions must be a non-empty list")
    return [dict(item) for item in dimensions]


def _trait_labels(
    coords: Mapping[str, float],
    trait_space: Mapping[str, Any],
) -> Dict[str, str]:
    discretization = trait_space.get("discretization", {})
    bins = (
        discretization.get("bins", {})
        if isinstance(discretization, Mapping)
        else {}
    )
    shared_cutpoints = (
        discretization.get("cutpoints", [-0.33, 0.33])
        if isinstance(discretization, Mapping)
        else [-0.33, 0.33]
    )
    labels: Dict[str, str] = {}
    for name, value in coords.items():
        cutpoints = bins.get(name, shared_cutpoints) if isinstance(bins, Mapping) else shared_cutpoints
        low, high = [float(point) for point in cutpoints[:2]]
        labels[name] = "low" if value < low else "high" if value >= high else "balanced"
    return labels


def sample_trait_profiles(trait_space: Mapping[str, Any]) -> List[ContinuousTraitProfile]:
    dimensions = _trait_dimensions(trait_space)
    sampling = trait_space.get("sampling", {})
    method = str(sampling.get("method", "axis_anchors")).strip().lower()
    n_agents = int(sampling.get("n_agents", 9))
    seed = int(sampling.get("seed", 42))
    names = [str(dim["name"]) for dim in dimensions]

    if method == "axis_anchors":
        expected = 1 + 2 * len(names)
        if n_agents not in (0, expected):
            raise ValueError(
                f"axis_anchors requires n_agents={expected} (center plus +/- each axis)"
            )
        coordinate_rows: List[Dict[str, float]] = [{name: 0.0 for name in names}]
        for name in names:
            for value in (-1.0, 1.0):
                row = {other: 0.0 for other in names}
                row[name] = value
                coordinate_rows.append(row)
    elif method == "latin_hypercube":
        if n_agents <= 0:
            raise ValueError("latin_hypercube requires a positive n_agents")
        rng = random.Random(seed)
        per_dimension: Dict[str, List[float]] = {}
        for dim in dimensions:
            name = str(dim["name"])
            low = float(dim.get("min", -1.0))
            high = float(dim.get("max", 1.0))
            values = [
                low + (high - low) * ((index + rng.random()) / n_agents)
                for index in range(n_agents)
            ]
            rng.shuffle(values)
            per_dimension[name] = values
        coordinate_rows = [
            {name: per_dimension[name][index] for name in names}
            for index in range(n_agents)
        ]
    else:
        raise ValueError(f"unsupported trait sampling method: {method}")

    return [
        ContinuousTraitProfile(coords=row, labels=_trait_labels(row, trait_space))
        for row in coordinate_rows
    ]


def render_trait_policy(
    trait: ContinuousTraitProfile,
    trait_space: Mapping[str, Any],
) -> str:
    lines: List[str] = []
    for dimension in _trait_dimensions(trait_space):
        name = str(dimension["name"])
        value = max(-1.0, min(1.0, float(trait.coords.get(name, 0.0))))
        negative = str(dimension.get("prompt_negative", ""))
        positive = str(dimension.get("prompt_positive", ""))
        if value <= -0.67:
            instruction = negative
        elif value >= 0.67:
            instruction = positive
        elif value < -0.10:
            instruction = f"lean toward this policy: {negative}"
        elif value > 0.10:
            instruction = f"lean toward this policy: {positive}"
        else:
            instruction = f"balance these policies: {negative}; {positive}"
        lines.append(f"- {name} ({value:+.3f}): {instruction}")
    return "\n".join(lines)


def build_agents(config: LogGenerationConfig) -> List[Agent]:
    return [
        Agent(
            agent_id=f"agent_{index:03d}",
            trait=profile,
            model=config.model,
        )
        for index, profile in enumerate(sample_trait_profiles(config.trait_space))
    ]


def _social_trait_bins(trait: ContinuousTraitProfile) -> Dict[str, str]:
    mapping = {"low": "low", "balanced": "mid", "high": "high"}
    return {
        name: mapping.get(label, str(label))
        for name, label in (trait.labels or {}).items()
    }


def _safe_openai_tool_name(original: str) -> str:
    candidate = re.sub(r"[^a-zA-Z0-9_-]", "_", original)
    if len(candidate) <= 64:
        return candidate
    digest = sha256(original.encode("utf-8")).hexdigest()[:10]
    return f"{candidate[:53]}_{digest}"


def _openai_schema_type(value: Any) -> Any:
    if isinstance(value, list):
        return [_openai_schema_type(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _openai_schema_type(item) for key, item in value.items()}
    type_mapping = {
        "dict": "object",
        "float": "number",
        "tuple": "array",
        "any": "string",
    }
    if "type" in result:
        result["type"] = type_mapping.get(str(result["type"]), result["type"])
    return result


def compile_openai_tools(
    functions: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    tools: List[Dict[str, Any]] = []
    api_to_original: Dict[str, str] = {}
    for function in functions:
        original_name = str(function["name"])
        api_name = _safe_openai_tool_name(original_name)
        if api_name in api_to_original and api_to_original[api_name] != original_name:
            raise ValueError(
                f"tool-name collision: {original_name} and {api_to_original[api_name]}"
            )
        api_to_original[api_name] = original_name
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": api_name,
                    "description": str(function.get("description", "")),
                    "parameters": _openai_schema_type(
                        function.get(
                            "parameters",
                            {"type": "object", "properties": {}},
                        )
                    ),
                },
            }
        )
    return tools, api_to_original


def _official_string_normalization(value: str) -> str:
    return re.sub(r"[ ,./\-_*^]", "", value).lower().replace("'", '"')


def _normalized_value(value: Any) -> Any:
    if isinstance(value, str):
        return ("string", _official_string_normalization(value))
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, (int, float)):
        return ("number", float(value))
    if isinstance(value, list):
        return ("list", tuple(_normalized_value(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(sorted((str(key), _normalized_value(item)) for key, item in value.items())),
        )
    if value is None:
        return ("null", None)
    return (type(value).__name__, value)


def _value_matches(value: Any, allowed_values: Any) -> bool:
    candidates = allowed_values if isinstance(allowed_values, list) else [allowed_values]
    normalized = _normalized_value(value)
    return any(
        candidate != "" and normalized == _normalized_value(candidate)
        for candidate in candidates
    )


def _schema_type_matches(
    value: Any,
    parameter_schema: Mapping[str, Any],
    allowed_values: Any,
) -> bool:
    """Mirror the Python type rules used by BFCL's AST checker."""
    candidates = allowed_values if isinstance(allowed_values, list) else [allowed_values]
    concrete_candidates = [candidate for candidate in candidates if candidate != ""]
    expected_type = str(parameter_schema.get("type", "any"))
    expected_python_type: Dict[str, type] = {
        "string": str,
        "integer": int,
        "float": float,
        "boolean": bool,
        "array": list,
        "tuple": list,
        "dict": dict,
        "any": str,
    }
    python_type = expected_python_type.get(expected_type)
    if python_type is None:
        return True

    # BFCL permits symbolic variables: if the answer's concrete type differs
    # from the schema type, a value of that answer type is checked directly.
    if concrete_candidates:
        answer_type = type(concrete_candidates[0])
        if answer_type is not python_type and type(value) is answer_type:
            return True

    if expected_type == "float":
        if type(value) not in (int, float):
            return False
    elif type(value) is not python_type:
        return False

    if expected_type in ("array", "tuple"):
        item_schema = parameter_schema.get("items", {})
        if isinstance(item_schema, Mapping):
            item_type = str(item_schema.get("type", ""))
            item_python_type = expected_python_type.get(item_type)
            if item_python_type is not None:
                for item in value:
                    if item_type == "float":
                        if type(item) is not float:
                            return False
                    elif type(item) is not item_python_type:
                        return False
    return True


def _allowed_value_matches(
    value: Any,
    allowed_values: Any,
    parameter_schema: Mapping[str, Any],
) -> bool:
    """Match BFCL candidate representations recursively using the tool schema."""
    candidates = allowed_values if isinstance(allowed_values, list) else [allowed_values]
    concrete_candidates = [candidate for candidate in candidates if candidate != ""]
    if not concrete_candidates:
        return False
    if not _schema_type_matches(value, parameter_schema, candidates):
        return False

    expected_type = str(parameter_schema.get("type", "any"))
    if expected_type == "dict":
        # Symbolic BFCL variables may be strings despite a container schema.
        if not isinstance(value, dict):
            return _value_matches(value, concrete_candidates)
        properties = parameter_schema.get("properties", {})
        if not isinstance(properties, Mapping):
            properties = {}
        for candidate in concrete_candidates:
            if not isinstance(candidate, dict):
                continue
            if set(value) - set(candidate):
                continue
            matched = True
            for key, nested_value in value.items():
                nested_schema = properties.get(key, {})
                if not isinstance(nested_schema, Mapping):
                    nested_schema = {}
                if not _allowed_value_matches(
                    nested_value,
                    candidate[key],
                    nested_schema,
                ):
                    matched = False
                    break
            if not matched:
                continue
            for key, nested_allowed in candidate.items():
                if key in value:
                    continue
                nested_candidates = (
                    nested_allowed
                    if isinstance(nested_allowed, list)
                    else [nested_allowed]
                )
                if "" not in nested_candidates:
                    matched = False
                    break
            if matched:
                return True
        return False

    if expected_type in ("array", "tuple"):
        if not isinstance(value, list):
            return _value_matches(value, concrete_candidates)
        item_schema = parameter_schema.get("items", {})
        if not isinstance(item_schema, Mapping):
            item_schema = {}
        for candidate in concrete_candidates:
            if not isinstance(candidate, list) or len(candidate) != len(value):
                continue
            if all(
                _allowed_value_matches(item, [expected_item], item_schema)
                for item, expected_item in zip(value, candidate)
            ):
                return True
        return False

    return _value_matches(value, concrete_candidates)


def _function_schema(
    task: BFCLTask,
    name: str,
) -> Optional[Mapping[str, Any]]:
    for function in task.functions:
        if str(function.get("name", "")) == name:
            return function
    return None


def _call_matches_ground_truth(
    task: BFCLTask,
    predicted: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> Tuple[bool, str]:
    expected_name = str(next(iter(expected.keys())))
    if str(predicted.get("name", "")) != expected_name:
        return False, f"wrong function: expected {expected_name}"
    arguments = predicted.get("arguments", {})
    if not isinstance(arguments, dict):
        return False, "arguments are not a JSON object"
    expected_arguments = expected[expected_name]
    if not isinstance(expected_arguments, dict):
        return False, "malformed ground-truth arguments"
    schema = _function_schema(task, expected_name)
    if schema is None:
        return False, f"function {expected_name} is absent from task schema"
    parameters = schema.get("parameters", {})
    required = set(parameters.get("required", [])) if isinstance(parameters, dict) else set()
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}

    unexpected = sorted(set(arguments) - set(expected_arguments))
    if unexpected:
        return False, f"unexpected arguments: {unexpected}"
    missing_required = sorted(required - set(arguments))
    if missing_required:
        return False, f"missing required arguments: {missing_required}"
    for argument_name, value in arguments.items():
        parameter_schema = properties.get(argument_name, {})
        if not isinstance(parameter_schema, Mapping):
            return False, f"missing schema for argument: {argument_name}"
        if not _allowed_value_matches(
            value,
            expected_arguments[argument_name],
            parameter_schema,
        ):
            return (
                False,
                f"invalid value for {argument_name}: {value!r}; "
                f"expected one of {expected_arguments[argument_name]!r}",
            )
    for argument_name, allowed in expected_arguments.items():
        if argument_name in arguments:
            continue
        candidates = allowed if isinstance(allowed, list) else [allowed]
        if "" not in candidates:
            return False, f"missing expected argument: {argument_name}"
    return True, ""


def score_bfcl_tool_calls(
    task: BFCLTask,
    predicted_calls: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """BFCL-compatible deterministic scoring for native tool-call outputs."""
    if task.category == "irrelevance":
        valid = len(predicted_calls) == 0
        return {
            "valid": valid,
            "error_type": "" if valid else "irrelevance_error:unexpected_tool_call",
            "errors": [] if valid else ["Expected no tool call."],
        }

    expected_calls = list(task.ground_truth)
    if len(predicted_calls) != len(expected_calls):
        return {
            "valid": False,
            "error_type": "ast_error:wrong_call_count",
            "errors": [
                f"Expected {len(expected_calls)} call(s), got {len(predicted_calls)}."
            ],
        }

    def search(
        predicted_index: int,
        remaining_expected: Tuple[int, ...],
    ) -> Optional[List[int]]:
        if predicted_index == len(predicted_calls):
            return []
        predicted = predicted_calls[predicted_index]
        for expected_index in remaining_expected:
            matches, _ = _call_matches_ground_truth(
                task,
                predicted,
                expected_calls[expected_index],
            )
            if not matches:
                continue
            rest = tuple(index for index in remaining_expected if index != expected_index)
            suffix = search(predicted_index + 1, rest)
            if suffix is not None:
                return [expected_index] + suffix
        return None

    assignment = search(0, tuple(range(len(expected_calls))))
    if assignment is not None:
        return {"valid": True, "error_type": "", "errors": [], "assignment": assignment}

    diagnostics: List[str] = []
    for predicted in predicted_calls:
        per_expected: List[str] = []
        for expected in expected_calls:
            _, error = _call_matches_ground_truth(task, predicted, expected)
            if error:
                per_expected.append(error)
        diagnostics.append("; ".join(per_expected[:3]) or "no compatible ground-truth call")
    return {
        "valid": False,
        "error_type": "ast_error:no_matching_assignment",
        "errors": diagnostics,
    }


def normalize_predicted_tool_calls(
    raw_calls: Sequence[Mapping[str, Any]],
    api_to_original: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    normalized: List[Dict[str, Any]] = []
    errors: List[str] = []
    for index, raw_call in enumerate(raw_calls):
        api_name = str(raw_call.get("name", ""))
        original_name = api_to_original.get(api_name, api_name)
        arguments_raw = raw_call.get("arguments", {})
        if isinstance(arguments_raw, str):
            try:
                arguments = json.loads(arguments_raw)
            except Exception as exc:
                errors.append(f"call {index}: invalid argument JSON: {exc}")
                arguments = {"__parse_error__": arguments_raw}
        else:
            arguments = arguments_raw
        normalized.append({"name": original_name, "arguments": arguments})
    return normalized, errors


_MISSING = object()


def _materialize_candidate(candidate: Any, schema: Mapping[str, Any]) -> Any:
    expected_type = str(schema.get("type", "any"))
    if expected_type == "dict" and isinstance(candidate, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            properties = {}
        result: Dict[str, Any] = {}
        for key, nested_allowed in candidate.items():
            nested_schema = properties.get(key, {})
            if not isinstance(nested_schema, Mapping):
                nested_schema = {}
            materialized = _materialize_allowed_values(nested_allowed, nested_schema)
            if materialized is not _MISSING:
                result[key] = materialized
        return result
    if expected_type in ("array", "tuple") and isinstance(candidate, list):
        item_schema = schema.get("items", {})
        if not isinstance(item_schema, Mapping):
            item_schema = {}
        return [_materialize_candidate(item, item_schema) for item in candidate]
    return candidate


def _materialize_allowed_values(allowed_values: Any, schema: Mapping[str, Any]) -> Any:
    candidates = allowed_values if isinstance(allowed_values, list) else [allowed_values]
    for candidate in candidates:
        if candidate != "":
            return _materialize_candidate(candidate, schema)
    return _MISSING


def _gold_mock_calls(
    task: BFCLTask,
    original_to_api: Mapping[str, str],
) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for expected in task.ground_truth:
        name = str(next(iter(expected.keys())))
        arguments: Dict[str, Any] = {}
        function_schema = _function_schema(task, name)
        if function_schema is None:
            raise ValueError(f"{task.task_id}: missing schema for {name}")
        parameters = function_schema.get("parameters", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        for key, allowed_values in expected[name].items():
            parameter_schema = properties.get(key, {})
            if not isinstance(parameter_schema, Mapping):
                parameter_schema = {}
            materialized = _materialize_allowed_values(
                allowed_values,
                parameter_schema,
            )
            if materialized is not _MISSING:
                arguments[key] = materialized
        calls.append(
            {
                "id": f"mock_{len(calls)}",
                "name": original_to_api[name],
                "arguments": json.dumps(arguments, ensure_ascii=False),
            }
        )
    return calls


class OpenAICompatibleToolClient:
    def __init__(self, config: LogGenerationConfig) -> None:
        self.config = config
        self._clients: Dict[str, Any] = {}

    def _client(self, provider: str) -> Any:
        if provider in self._clients:
            return self._clients[provider]
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for live collection. "
                "Install the openai package or run with --mock_mode."
            ) from exc

        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
        elif provider == "together":
            api_key = os.environ.get("TOGETHER_API_KEY")
            base_url = os.environ.get(
                "TOGETHER_BASE_URL",
                "https://api.together.xyz/v1",
            )
        else:
            raise ValueError(f"unsupported provider: {provider}")
        if not api_key:
            raise EnvironmentError(f"missing API key for provider {provider}")
        kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "timeout": float(self.config.api_timeout_sec),
        }
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        self._clients[provider] = client
        return client

    def generate(
        self,
        agent: Agent,
        task: BFCLTask,
        trait_space: Mapping[str, Any],
    ) -> Dict[str, Any]:
        tools, api_to_original = compile_openai_tools(task.functions)
        original_to_api = {original: api for api, original in api_to_original.items()}
        policy = render_trait_policy(agent.trait, trait_space)
        system_prompt = (
            "You are a tool-using assistant. Respond with native tool calls when an "
            "available tool is appropriate. If no available tool can satisfy the "
            "request, do not call a tool. Do not expose hidden reasoning.\n\n"
            "Persistent tool-use policy:\n"
            f"{policy}"
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(copy.deepcopy(list(task.messages)))

        if self.config.mock_mode:
            if self.config.mock_mode == "gold":
                raw_calls = _gold_mock_calls(task, original_to_api)
            elif self.config.mock_mode == "empty":
                raw_calls = []
            elif self.config.mock_mode == "first_function":
                first_name = next(iter(original_to_api.values()), "")
                raw_calls = (
                    [{"id": "mock_0", "name": first_name, "arguments": "{}"}]
                    if first_name
                    else []
                )
            else:
                raise ValueError(f"unsupported mock mode: {self.config.mock_mode}")
            return {
                "raw_tool_calls": raw_calls,
                "content": "",
                "prompt_messages": messages,
                "tools": tools,
                "provider": "mock",
                "model": self.config.mock_mode,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "latency_sec": 0.0,
            }

        request_kwargs: Dict[str, Any] = {
            "model": agent.model.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": float(self.config.temperature),
        }
        if agent.model.provider == "openai":
            request_kwargs["max_completion_tokens"] = int(
                self.config.max_completion_tokens
            )
        else:
            request_kwargs["max_tokens"] = int(self.config.max_completion_tokens)
        if self.config.parallel_tool_calls:
            request_kwargs["parallel_tool_calls"] = True

        last_exception: Optional[Exception] = None
        started = time.time()
        for attempt in range(int(self.config.api_max_retries) + 1):
            try:
                response = self._client(agent.model.provider).chat.completions.create(
                    **request_kwargs
                )
                message = response.choices[0].message
                raw_calls = []
                for tool_call in message.tool_calls or []:
                    raw_calls.append(
                        {
                            "id": str(tool_call.id),
                            "name": str(tool_call.function.name),
                            "arguments": str(tool_call.function.arguments),
                        }
                    )
                usage = getattr(response, "usage", None)
                return {
                    "raw_tool_calls": raw_calls,
                    "content": message.content or "",
                    "prompt_messages": messages,
                    "tools": tools,
                    "provider": agent.model.provider,
                    "model": agent.model.model,
                    "usage": {
                        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(
                            getattr(usage, "completion_tokens", 0) or 0
                        ),
                    },
                    "latency_sec": time.time() - started,
                }
            except Exception as exc:
                last_exception = exc
                if attempt >= int(self.config.api_max_retries):
                    break
                time.sleep(min(2.0**attempt, 8.0))
        assert last_exception is not None
        raise last_exception


def _parallel_map_ordered(items: Sequence[Any], function, max_workers: int) -> List[Any]:
    if not items:
        return []
    workers = max(1, min(int(max_workers), len(items)))
    if workers == 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, items))


class BFCLLogCollector:
    def __init__(self, config: LogGenerationConfig) -> None:
        self.config = config
        self.agents = build_agents(config)
        self.tasks = load_bfcl_tasks(Path(config.subset_path))
        if config.n_tasks > len(self.tasks):
            raise ValueError(
                f"Requested n_tasks={config.n_tasks}, but the subset contains "
                f"only {len(self.tasks)} tasks"
            )
        task_rng = random.Random(config.task_sample_seed)
        if 0 < config.n_tasks < len(self.tasks):
            by_category: Dict[str, List[BFCLTask]] = {}
            for task in self.tasks:
                by_category.setdefault(task.category, []).append(task)
            category_order = sorted(by_category)
            task_rng.shuffle(category_order)
            for category_tasks in by_category.values():
                task_rng.shuffle(category_tasks)
            selected: List[BFCLTask] = []
            category_offsets = {category: 0 for category in category_order}
            while len(selected) < int(config.n_tasks):
                made_progress = False
                for category in category_order:
                    offset = category_offsets[category]
                    category_tasks = by_category[category]
                    if offset >= len(category_tasks):
                        continue
                    selected.append(category_tasks[offset])
                    category_offsets[category] = offset + 1
                    made_progress = True
                    if len(selected) >= int(config.n_tasks):
                        break
                if not made_progress:
                    raise ValueError("Could not satisfy stratified task sample")
            task_rng.shuffle(selected)
            self.tasks = selected
        else:
            task_rng.shuffle(self.tasks)
        self.client = OpenAICompatibleToolClient(config)

    def _metadata(self) -> Dict[str, Any]:
        subset_path = Path(self.config.subset_path)
        manifest_path = subset_path.with_name("manifest.json")
        manifest = load_json(manifest_path) if manifest_path.exists() else {}
        task_fingerprints = [
            {
                "task_id": task.task_id,
                "source_sha256": task.source_sha256
                or sha256_json(
                    {
                        "task_id": task.task_id,
                        "category": task.category,
                        "messages": task.messages,
                        "functions": task.functions,
                        "ground_truth": task.ground_truth,
                    }
                ),
            }
            for task in self.tasks
        ]
        metadata = {
            "experiment_type": "bfcl_tool_routing_log_dataset",
            "config": asdict(self.config),
            "eval_only": False,
            "reused_contents": False,
            "benchmark_manifest": manifest,
            "subset_file_sha256": sha256(subset_path.read_bytes()).hexdigest(),
            "task_fingerprints": task_fingerprints,
            "metric_semantics": {
                "fitness": "binary BFCL-compatible AST task success",
                "correct": "same as fitness",
                "safety_intervention_supported": False,
                "scorer": (
                    "Local native-tool-call scorer mirroring BFCL Python AST "
                    "function, argument, type, value, optionality, and parallel "
                    "order-invariance rules."
                ),
                "legacy_social_metric_columns": (
                    "Legacy engagement and quality columns alias fitness for loader "
                    "compatibility; safety is fixed at 1 because this testbed does "
                    "not instantiate a safety-selection channel."
                ),
            },
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "trait": asdict(agent.trait),
                    "model": asdict(agent.model),
                }
                for agent in self.agents
            ],
            "evaluators": [
                {
                    "evaluator_id": "bfcl_ast",
                    "trait": {"coords": {}, "labels": {}},
                }
            ],
            "task_ids": [task.task_id for task in self.tasks],
        }
        metadata["experiment_fingerprint"] = sha256_json(
            {
                "config": metadata["config"],
                "benchmark_manifest": metadata["benchmark_manifest"],
                "subset_file_sha256": metadata["subset_file_sha256"],
                "task_fingerprints": metadata["task_fingerprints"],
                "agents": metadata["agents"],
                "task_ids": metadata["task_ids"],
            }
        )
        return metadata

    def _generate_agent_row(
        self,
        timestep: int,
        task: BFCLTask,
        agent: Agent,
    ) -> Dict[str, Any]:
        result = self.client.generate(agent, task, self.config.trait_space)
        api_to_original = {
            tool["function"]["name"]: str(function["name"])
            for tool, function in zip(result["tools"], task.functions)
        }
        normalized_calls, parse_errors = normalize_predicted_tool_calls(
            result["raw_tool_calls"],
            api_to_original,
        )
        score = score_bfcl_tool_calls(task, normalized_calls)
        if parse_errors:
            score = {
                "valid": False,
                "error_type": "ast_decoder:invalid_argument_json",
                "errors": parse_errors,
            }
        correct = bool(score["valid"])
        context_id = sha256(
            f"{self.config.random_seed}|{timestep}|{task.task_id}".encode("utf-8")
        ).hexdigest()[:24]
        calls_text = canonical_json(normalized_calls)
        trait_bins = _social_trait_bins(agent.trait)
        row: Dict[str, Any] = {
            "run_id": self.config.random_seed,
            "run_label": self.config.run_label,
            "timestep": timestep,
            "context_id": context_id,
            "task_id": task.task_id,
            "task_category": task.category,
            "topic": task.user_text,
            "topic_category": task.category,
            "topic_source": "BFCL",
            "topic_index": timestep,
            "source_subreddit": "BFCL",
            "source_post_id": task.task_id,
            "normalized_from": task.task_id,
            "raw_topic_text": task.user_text,
            "topic_title": task.task_id,
            "selftext_excerpt": task.user_text,
            "top_comments_excerpt": [],
            "agent_id": agent.agent_id,
            "trait": asdict(agent.trait),
            "trait_coords": dict(agent.trait.coords),
            "trait_bins": trait_bins,
            "trait_labels": dict(agent.trait.labels or {}),
            "predicted_tool_calls": normalized_calls,
            "raw_tool_calls": result["raw_tool_calls"],
            "content_text": calls_text,
            "content_text_hash": sha256(calls_text.encode("utf-8")).hexdigest()[:24],
            "correct": correct,
            "fitness": float(correct),
            "score_details": score,
            "provider": result["provider"],
            "model": result["model"],
            "post_provider": result["provider"],
            "post_model": result["model"],
            "usage": result["usage"],
            "latency_sec": float(result["latency_sec"]),
        }
        row.update({name: 0.0 for name in SOCIAL_CONTEXT_AXES})
        if self.config.save_raw_response:
            row["response_content"] = result["content"]
            row["post_raw_response_text"] = canonical_json(
                {
                    "content": result["content"],
                    "tool_calls": result["raw_tool_calls"],
                }
            )
        if self.config.save_prompt:
            row["prompt_messages"] = result["prompt_messages"]
            row["tools"] = result["tools"]
            row["post_prompt"] = canonical_json(
                {
                    "messages": result["prompt_messages"],
                    "tools": result["tools"],
                }
            )
        if self.config.sleep_between_calls_sec > 0:
            time.sleep(float(self.config.sleep_between_calls_sec))
        return row

    @staticmethod
    def _legacy_evaluation_row(content: Mapping[str, Any]) -> Dict[str, Any]:
        fitness = float(content["fitness"])
        row = {
            "run_id": content["run_id"],
            "run_label": content["run_label"],
            "timestep": content["timestep"],
            "context_id": content["context_id"],
            "task_id": content["task_id"],
            "task_category": content["task_category"],
            "topic_category": content["task_category"],
            "topic_index": content["topic_index"],
            "agent_id": content["agent_id"],
            "content_text_hash": content["content_text_hash"],
            "evaluator_id": "bfcl_ast",
            "evaluator_trait": {"coords": {}, "labels": {}},
            "evaluator_trait_coords": {},
            "evaluator_trait_bins": {},
            "evaluator_trait_labels": {},
            "eval_provider": "deterministic",
            "eval_model": "bfcl_ast",
            "eval_prompt": "Deterministic BFCL-compatible AST exact-match scoring.",
            "eval_raw_response_text": canonical_json(content["score_details"]),
            "eval_parse_error": "",
            "correct": bool(content["correct"]),
            "fitness": fitness,
            "like": fitness,
            "reply": fitness,
            "share": fitness,
            "follow": fitness,
            "quality": fitness,
            "safety": 1.0,
            "legacy_placeholder": True,
            "score_details": content["score_details"],
            "topic": content["topic"],
            "topic_source": content["topic_source"],
            "source_subreddit": content["source_subreddit"],
            "source_post_id": content["source_post_id"],
            "agent_trait": content["trait"],
            "agent_trait_coords": content["trait_coords"],
            "agent_trait_bins": content["trait_bins"],
            "agent_trait_labels": content["trait_labels"],
            "trait_coords": content["trait_coords"],
        }
        row.update({name: float(content[name]) for name in SOCIAL_CONTEXT_AXES})
        return row

    @staticmethod
    def _legacy_audit_row(content: Mapping[str, Any]) -> Dict[str, Any]:
        fitness = float(content["fitness"])
        row = {
            "run_id": content["run_id"],
            "run_label": content["run_label"],
            "timestep": content["timestep"],
            "context_id": content["context_id"],
            "task_id": content["task_id"],
            "task_category": content["task_category"],
            "topic_category": content["task_category"],
            "topic_index": content["topic_index"],
            "agent_id": content["agent_id"],
            "content_text_hash": content["content_text_hash"],
            "auditor_id": "bfcl_ast",
            "audit_provider": "deterministic",
            "audit_model": "bfcl_ast",
            "audit_prompt": "Deterministic BFCL-compatible AST exact-match audit.",
            "audit_raw_response_text": canonical_json(content["score_details"]),
            "audit_parse_error": "",
            "correct": bool(content["correct"]),
            "fitness": fitness,
            "quality": fitness,
            "safety": 1.0,
            "reason": (
                "BFCL AST exact match."
                if bool(content["correct"])
                else "; ".join(content["score_details"].get("errors", []))
            ),
            "legacy_placeholder": True,
            "score_details": content["score_details"],
            "topic": content["topic"],
            "topic_source": content["topic_source"],
            "source_subreddit": content["source_subreddit"],
            "source_post_id": content["source_post_id"],
            "agent_trait": content["trait"],
            "agent_trait_coords": content["trait_coords"],
            "trait_bins": content["trait_bins"],
            "trait_labels": content["trait_labels"],
            "trait_coords": content["trait_coords"],
        }
        row.update({name: float(content[name]) for name in SOCIAL_CONTEXT_AXES})
        return row

    def collect(self, checkpoint_dir: Path, resume: bool = True) -> Dict[str, Any]:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metadata = self._metadata()
        metadata_path = checkpoint_dir / "metadata.json"
        existing_steps = list(checkpoint_dir.glob("timestep_*.json"))
        if resume and metadata_path.exists() and existing_steps:
            existing_metadata = load_json(metadata_path)
            if (
                existing_metadata.get("experiment_fingerprint")
                != metadata["experiment_fingerprint"]
            ):
                raise ValueError(
                    "Checkpoint metadata does not match the current experiment. "
                    "Use a new --checkpoint_dir or pass --no_resume."
                )
        elif not resume:
            for step_path in existing_steps:
                step_path.unlink()
        save_json_atomic(metadata, metadata_path)

        bundles: List[Dict[str, Any]] = []
        for timestep, task in enumerate(self.tasks):
            step_path = checkpoint_dir / f"timestep_{timestep:04d}.json"
            if resume and step_path.exists():
                existing = load_json(step_path)
                if (
                    bool(existing.get("complete"))
                    and existing.get("experiment_fingerprint")
                    == metadata["experiment_fingerprint"]
                    and len(existing.get("contents", [])) == len(self.agents)
                    and len(existing.get("evaluations", [])) == len(self.agents)
                    and len(existing.get("audits", [])) == len(self.agents)
                ):
                    bundles.append(existing)
                    print(
                        f"[resume] timestep={timestep} task={task.task_id}",
                        flush=True,
                    )
                    continue

            contents = _parallel_map_ordered(
                self.agents,
                lambda agent: self._generate_agent_row(timestep, task, agent),
                self.config.api_max_workers,
            )
            evaluations = [self._legacy_evaluation_row(row) for row in contents]
            audits = [self._legacy_audit_row(row) for row in contents]
            context_id = str(contents[0]["context_id"])
            bundle = {
                "run_id": self.config.random_seed,
                "run_label": self.config.run_label,
                "timestep": timestep,
                "context_id": context_id,
                "task_id": task.task_id,
                "task_category": task.category,
                "topic": task.user_text,
                "topic_category": task.category,
                "topic_source": "BFCL",
                "source_subreddit": "BFCL",
                "source_post_id": task.task_id,
                "topic_index": timestep,
                "functions": list(task.functions),
                "ground_truth": list(task.ground_truth),
                "contents": contents,
                "evaluations": evaluations,
                "audits": audits,
                "experiment_fingerprint": metadata["experiment_fingerprint"],
                "complete": True,
            }
            save_json_atomic(bundle, step_path)
            bundles.append(bundle)
            correct_count = sum(int(row["correct"]) for row in contents)
            print(
                f"[checkpoint] timestep={timestep} task={task.task_id} "
                f"correct={correct_count}/{len(contents)}",
                flush=True,
            )

        dataset = dict(metadata)
        ordered = sorted(bundles, key=lambda item: int(item["timestep"]))
        dataset["contents"] = [
            row for bundle in ordered for row in bundle.get("contents", [])
        ]
        dataset["evaluations"] = [
            row for bundle in ordered for row in bundle.get("evaluations", [])
        ]
        dataset["audits"] = [
            row for bundle in ordered for row in bundle.get("audits", [])
        ]
        return dataset


def parse_config_overrides(raw: str) -> Dict[str, Any]:
    if not raw.strip():
        return {}
    if raw.startswith("@"):
        value = load_json(Path(raw[1:]))
        if isinstance(value, dict) and isinstance(value.get("config_json"), dict):
            value = value["config_json"]
    else:
        value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("--config_json must decode to an object")
    return value


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if (
            isinstance(value, Mapping)
            and isinstance(result.get(key), dict)
        ):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def config_from_overrides(overrides: Mapping[str, Any]) -> LogGenerationConfig:
    raw = _deep_update(asdict(LogGenerationConfig()), overrides)
    model_value = raw.get("model", {})
    if isinstance(model_value, str):
        raw["model"] = ModelSpec(model=model_value)
    elif isinstance(model_value, Mapping):
        raw["model"] = ModelSpec(
            provider=str(model_value.get("provider", "openai")),
            model=str(model_value.get("model", "gpt-5.4-mini")),
        )
    else:
        raise TypeError("config.model must be a model string or object")
    return LogGenerationConfig(**raw)


def run_self_test() -> None:
    simple = BFCLTask(
        task_id="simple_python_test",
        category="simple_python",
        messages=({"role": "user", "content": "Area for base 10, height 5"},),
        functions=(
            {
                "name": "triangle.area",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base": {"type": "integer"},
                        "height": {"type": "integer"},
                        "unit": {"type": "string"},
                    },
                    "required": ["base", "height"],
                },
            },
        ),
        ground_truth=(
            {
                "triangle.area": {
                    "base": [10],
                    "height": [5],
                    "unit": ["units", ""],
                }
            },
        ),
    )
    assert score_bfcl_tool_calls(
        simple,
        [{"name": "triangle.area", "arguments": {"base": 10, "height": 5}}],
    )["valid"]
    assert not score_bfcl_tool_calls(simple, [])["valid"]
    assert not score_bfcl_tool_calls(
        simple,
        [{"name": "triangle.area", "arguments": {"base": 9, "height": 5}}],
    )["valid"]
    assert not score_bfcl_tool_calls(
        simple,
        [{"name": "triangle.area", "arguments": {"base": 10.0, "height": 5}}],
    )["valid"]

    nested = BFCLTask(
        task_id="nested_test",
        category="simple_python",
        messages=({"role": "user", "content": "Paint a 20 by 12 wall"},),
        functions=(
            {
                "name": "paint.calculate",
                "parameters": {
                    "type": "dict",
                    "required": ["area"],
                    "properties": {
                        "area": {
                            "type": "dict",
                            "properties": {
                                "width": {"type": "integer"},
                                "height": {"type": "integer"},
                            },
                        }
                    },
                },
            },
        ),
        ground_truth=(
            {"paint.calculate": {"area": [{"width": [20], "height": [12]}]}},
        ),
    )
    assert score_bfcl_tool_calls(
        nested,
        [
            {
                "name": "paint.calculate",
                "arguments": {"area": {"width": 20, "height": 12}},
            }
        ],
    )["valid"]
    nested_tools, nested_api_to_original = compile_openai_tools(nested.functions)
    nested_original_to_api = {
        original: api_name
        for api_name, original in nested_api_to_original.items()
    }
    nested_raw = _gold_mock_calls(nested, nested_original_to_api)
    nested_predicted, nested_errors = normalize_predicted_tool_calls(
        nested_raw,
        nested_api_to_original,
    )
    assert not nested_errors
    assert nested_predicted[0]["arguments"]["area"] == {"width": 20, "height": 12}
    assert nested_tools[0]["function"]["name"] == "paint_calculate"

    parallel = BFCLTask(
        task_id="parallel_test",
        category="parallel",
        messages=({"role": "user", "content": "Play A and B"},),
        functions=(
            {
                "name": "play",
                "parameters": {
                    "type": "dict",
                    "properties": {"artist": {"type": "string"}},
                    "required": ["artist"],
                },
            },
        ),
        ground_truth=(
            {"play": {"artist": ["A"]}},
            {"play": {"artist": ["B"]}},
        ),
    )
    assert score_bfcl_tool_calls(
        parallel,
        [
            {"name": "play", "arguments": {"artist": "B"}},
            {"name": "play", "arguments": {"artist": "A"}},
        ],
    )["valid"]

    irrelevant = BFCLTask(
        task_id="irrelevance_test",
        category="irrelevance",
        messages=({"role": "user", "content": "Unrelated request"},),
        functions=tuple(),
        ground_truth=tuple(),
    )
    assert score_bfcl_tool_calls(irrelevant, [])["valid"]
    print("self-test passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect pinned BFCL tool-calling logs with four-dimensional traits."
    )
    parser.add_argument("--output", default="bfcl_tool_log_dataset.json")
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--config_json", default="")
    parser.add_argument(
        "--population_seed",
        type=int,
        default=None,
        help="Override trait_space.sampling.seed while preserving the shared config.",
    )
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--mock_mode", choices=["", "gold", "empty", "first_function"], default="")
    parser.add_argument("--prepare_subset", action="store_true")
    parser.add_argument("--bfcl_subset_dir", default=str(DEFAULT_SUBSET_DIR))
    parser.add_argument("--tasks_per_category", type=int, default=20)
    parser.add_argument("--subset_seed", type=int, default=42)
    parser.add_argument("--source_commit", default=BFCL_PINNED_COMMIT)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.prepare_subset:
        manifest = prepare_bfcl_subset(
            output_dir=Path(args.bfcl_subset_dir),
            tasks_per_category=int(args.tasks_per_category),
            seed=int(args.subset_seed),
            commit=str(args.source_commit),
        )
        print(
            f"Prepared {manifest['total_tasks']} BFCL tasks under "
            f"{args.bfcl_subset_dir}",
            flush=True,
        )
        return

    overrides = parse_config_overrides(args.config_json)
    if args.population_seed is not None:
        trait_space = overrides.setdefault("trait_space", {})
        if not isinstance(trait_space, dict):
            raise TypeError("config trait_space must be an object")
        sampling = trait_space.setdefault("sampling", {})
        if not isinstance(sampling, dict):
            raise TypeError("config trait_space.sampling must be an object")
        sampling["seed"] = int(args.population_seed)
        overrides["run_label"] = f"tool_bfcl_popseed_{args.population_seed}"
    if args.mock_mode:
        overrides["mock_mode"] = args.mock_mode
    config = config_from_overrides(overrides)
    if (
        args.population_seed is not None
        and str(config.trait_space.get("sampling", {}).get("method", "")).lower()
        == "axis_anchors"
    ):
        parser.error(
            "--population_seed cannot define independent populations with "
            "axis_anchors; use latin_hypercube or omit --population_seed"
        )
    collector = BFCLLogCollector(config)
    output_path = Path(args.output)
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else Path(str(output_path) + ".checkpoints")
    )
    dataset = collector.collect(
        checkpoint_dir=checkpoint_dir,
        resume=not args.no_resume,
    )
    save_json_atomic(dataset, output_path)
    print(f"Saved dataset to {output_path}", flush=True)
    print(f"Saved checkpoints to {checkpoint_dir}", flush=True)


if __name__ == "__main__":
    main()
