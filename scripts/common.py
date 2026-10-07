from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("education", "finance", "medical", "mental")
EVALUATION_MODES = (
    "mixed_denied",
    "mixed_allowed",
    "privacy_denied",
    "privacy_allowed",
    "non_privacy_only",
)


@dataclass(frozen=True)
class SelectedUser:
    domain: str
    user_index: int
    user_id: str
    history_file: str
    query_file: str


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_no}")
            yield value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _history_records(domain: str, data_root: Path) -> List[SelectedUser]:
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain {domain!r}; choose from {', '.join(DOMAINS)}")
    histories = data_root / domain / "histories"
    queries = data_root / domain / "evaluation_queries"
    records: List[SelectedUser] = []
    for history_path in sorted(histories.glob("user_*.json")):
        payload = read_json(history_path)
        user_index = int(payload["user_index"])
        user_id = str(payload["user_id"]).strip()
        query_path = queries / f"user{user_index}_test.jsonl"
        if not query_path.exists():
            raise FileNotFoundError(f"Missing query file for {history_path}: {query_path}")
        records.append(SelectedUser(
            domain=domain,
            user_index=user_index,
            user_id=user_id,
            history_file=_relative(history_path),
            query_file=_relative(query_path),
        ))
    return records


def parse_index_range(value: str) -> Tuple[int, int]:
    pieces = value.replace("-", ":").split(":", maxsplit=1)
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("range must be START:END (inclusive)")
    start, end = (int(piece) for piece in pieces)
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError("range must satisfy 0 <= START <= END")
    return start, end


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--domain",
        action="append",
        choices=DOMAINS,
        help="Domain to include; repeat for multiple domains. Required unless loading a manifest.",
    )
    parser.add_argument("--user-id", action="append", default=[], help="Exact dataset user_id; repeatable.")
    parser.add_argument("--user-index", action="append", type=int, default=[], help="Numeric user index; repeatable.")
    parser.add_argument(
        "--user-range",
        action="append",
        type=parse_index_range,
        default=[],
        metavar="START:END",
        help="Inclusive user-index range; repeatable.",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=None,
        help="Random users per selected domain after explicit filters. Omit to keep all matches.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for --num-users sampling.")
    parser.add_argument(
        "--selection-manifest",
        help="Reuse an earlier selection manifest; other selection flags are then ignored.",
    )


def _manifest_record(value: Mapping[str, Any]) -> SelectedUser:
    return SelectedUser(
        domain=str(value["domain"]),
        user_index=int(value["user_index"]),
        user_id=str(value["user_id"]),
        history_file=str(value["history_file"]),
        query_file=str(value["query_file"]),
    )


def select_users(args: argparse.Namespace, data_root: Path | None = None) -> List[SelectedUser]:
    if getattr(args, "selection_manifest", None):
        payload = read_json(repo_path(args.selection_manifest))
        records = [_manifest_record(value) for value in payload.get("users", [])]
        if not records:
            raise ValueError("Selection manifest contains no users")
        return records

    domains = list(dict.fromkeys(getattr(args, "domain", None) or []))
    if not domains:
        raise ValueError("Provide at least one --domain, or use --selection-manifest")
    data_root = (data_root or REPO_ROOT / "data").resolve()
    explicit_ids = set(getattr(args, "user_id", []) or [])
    explicit_indices = set(getattr(args, "user_index", []) or [])
    for start, end in getattr(args, "user_range", []) or []:
        explicit_indices.update(range(start, end + 1))
    has_explicit_filter = bool(explicit_ids or explicit_indices)
    count = getattr(args, "num_users", None)
    if count is not None and count <= 0:
        raise ValueError("--num-users must be positive")

    selected: List[SelectedUser] = []
    for domain_offset, domain in enumerate(domains):
        records = _history_records(domain, data_root)
        if has_explicit_filter:
            records = [
                record for record in records
                if record.user_id in explicit_ids or record.user_index in explicit_indices
            ]
        if count is not None:
            if count > len(records):
                raise ValueError(
                    f"Requested {count} users from {domain}, but only {len(records)} match"
                )
            rng = random.Random(int(getattr(args, "seed", 0)) + domain_offset)
            records = sorted(rng.sample(records, count), key=lambda item: item.user_index)
        selected.extend(records)

    if not selected:
        raise ValueError("No users matched the selection")
    if has_explicit_filter:
        found_ids = {record.user_id for record in selected}
        found_indices = {record.user_index for record in selected}
        missing_ids = explicit_ids - found_ids
        missing_indices = explicit_indices - found_indices
        if missing_ids or missing_indices:
            raise ValueError(
                f"Selection did not resolve all requested users: "
                f"missing user_id={sorted(missing_ids)}, user_index={sorted(missing_indices)}"
            )
    return selected


def save_selection_manifest(
    path: Path,
    users: Sequence[SelectedUser],
    *,
    arguments: Mapping[str, Any] | None = None,
) -> None:
    clean_arguments = {}
    for key, value in (arguments or {}).items():
        if isinstance(value, Path):
            value = str(value)
        clean_arguments[key] = value
    write_json(path, {
        "schema_version": 1,
        "note": (
            "This is the actual selection for this run. It is not claimed to be the "
            "undisclosed original 100-user paper split."
        ),
        "arguments": clean_arguments,
        "users": [asdict(user) for user in users],
    })


def load_paper_config(path: str | Path) -> Dict[str, Any]:
    payload = read_json(repo_path(path))
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a JSON object")
    return payload


def env_value(name: str, *, required: bool = True) -> str:
    value = os.getenv(name, "").strip()
    if required and not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value


def provider_settings(config: Mapping[str, Any]) -> Dict[str, str]:
    model_env = str(config.get("model_id_env", "")).strip()
    key_env = str(config.get("api_key_env", "")).strip()
    base_env = str(config.get("base_url_env", "")).strip()
    if not model_env or not key_env:
        raise ValueError("Provider config requires model_id_env and api_key_env")
    return {
        "display_name": str(config.get("display_name", model_env)),
        "model": env_value(model_env),
        "api_key": env_value(key_env),
        "base_url": env_value(base_env, required=False) if base_env else "",
    }


def build_memory_config(config: Mapping[str, Any], *, collection_suffix: str = "") -> Dict[str, Any]:
    memory_model = provider_settings(config["memory_model"])
    embedding_model = provider_settings(config["embedding_model"])
    storage = config["storage"]
    collection = str(storage["qdrant_collection"])
    if collection_suffix:
        collection = f"{collection}_{collection_suffix}"
    mapping_dir = repo_path(storage["privacy_mapping_dir"])
    mapping_dir.mkdir(parents=True, exist_ok=True)

    neo4j_url = env_value(str(storage["neo4j_url_env"]))
    neo4j_username = env_value(str(storage["neo4j_username_env"]))
    neo4j_password = env_value(str(storage["neo4j_password_env"]))
    database_env = str(storage.get("neo4j_database_env", ""))
    neo4j_database = env_value(database_env, required=False) if database_env else ""

    vector_config: Dict[str, Any] = {
        "collection_name": collection,
        "embedding_model_dims": int(config["embedding_model"].get("dimensions", 1536)),
    }
    qdrant_url_env = str(storage.get("qdrant_url_env", "")).strip()
    qdrant_url = env_value(qdrant_url_env, required=False) if qdrant_url_env else ""
    qdrant_path = str(storage.get("qdrant_path", "")).strip()
    if qdrant_url:
        vector_config["url"] = qdrant_url
        qdrant_api_key_env = str(storage.get("qdrant_api_key_env", "")).strip()
        if qdrant_api_key_env:
            qdrant_api_key = env_value(qdrant_api_key_env, required=False)
            if qdrant_api_key:
                vector_config["api_key"] = qdrant_api_key
    elif qdrant_path:
        vector_config["path"] = str(repo_path(qdrant_path))
        vector_config["on_disk"] = True
    else:
        expected = f" in {qdrant_url_env}" if qdrant_url_env else ""
        raise RuntimeError(
            "Qdrant storage is not configured: set the configured service URL"
            f"{expected}, or provide storage.qdrant_path."
        )

    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": memory_model["model"],
                "api_key": memory_model["api_key"],
                "openai_base_url": memory_model["base_url"] or None,
                "temperature": 0.0,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": embedding_model["model"],
                "api_key": embedding_model["api_key"],
                "openai_base_url": embedding_model["base_url"] or None,
                "embedding_dims": int(config["embedding_model"].get("dimensions", 1536)),
            },
        },
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": neo4j_url,
                "username": neo4j_username,
                "password": neo4j_password,
                "database": neo4j_database or "neo4j",
                "base_label": True,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": vector_config,
        },
        "history_db_path": str(repo_path(storage["history_db_path"])),
        "privacy_mapping_dir": str(mapping_dir),
    }


def modes_for_query(query: Mapping[str, Any]) -> Tuple[str, ...]:
    privacy = [value for value in query.get("privacy_entities", []) if str(value).strip()]
    preference = [value for value in query.get("preference_entities", []) if str(value).strip()]
    if privacy and preference:
        return ("mixed_denied", "mixed_allowed")
    if privacy:
        return ("privacy_denied", "privacy_allowed")
    return ("non_privacy_only",)


def flatten_dialogues(history_payload: Mapping[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for item in history_payload.get("dialogues", []):
        for message in item.get("dialogue", []):
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", ""))
            if role and content:
                messages.append({"role": role, "content": content})
    return messages


def load_queries(user: SelectedUser) -> List[Dict[str, Any]]:
    return list(read_jsonl(repo_path(user.query_file)))


def ensure_empty_or_resume(path: Path, *, resume: bool) -> None:
    if path.exists() and not resume:
        raise FileExistsError(f"Output exists; pass --resume to append/skip: {path}")


def completed_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("sample_id", "")) for row in read_jsonl(path)}
