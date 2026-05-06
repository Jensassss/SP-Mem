from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .constants import ALLOWED_ENTITIES, PRIVATE_ENTITIES


def safe_json_loads(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    text = text.strip()

    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fenced:
        try:
            value = json.loads(fenced.group(1))
            return value if isinstance(value, dict) else {}
        except Exception:
            pass

    brace = re.search(r"(\{[\s\S]*\})", text)
    if brace:
        try:
            value = json.loads(brace.group(1))
            return value if isinstance(value, dict) else {}
        except Exception:
            pass

    return {}


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y"}:
            return True
        if token in {"false", "0", "no", "n"}:
            return False

    return default


def normalize_entity_name(entity: Any) -> str:
    if not entity:
        return ""
    value = str(entity).strip().lower()
    value = value.replace("-", " ")
    value = re.sub(r"\s+", "_", value)
    return value.strip("_")


def deduplicate_entities(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for item in items:
        if not isinstance(item, dict):
            continue

        entity = normalize_entity_name(item.get("entity", ""))
        if not entity:
            continue

        default_privacy = entity in PRIVATE_ENTITIES
        candidate = {
            "entity": entity,
            "privacy": coerce_bool(item.get("privacy"), default=default_privacy),
        }

        if entity not in merged:
            merged[entity] = candidate
        else:
            old = merged[entity]
            merged[entity] = {
                "entity": entity,
                "privacy": old["privacy"] or candidate["privacy"],
            }

    return list(merged.values())


def filter_allowed_entities(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    allowed = set(ALLOWED_ENTITIES)
    filtered: List[Dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        entity = normalize_entity_name(item.get("entity", ""))
        if entity in allowed:
            default_privacy = entity in PRIVATE_ENTITIES
            filtered.append(
                {
                    "entity": entity,
                    "privacy": coerce_bool(item.get("privacy"), default=default_privacy),
                }
            )

    return filtered


def extract_required_entities(raw_required_entities: Any) -> List[Any]:
    if isinstance(raw_required_entities, list):
        return raw_required_entities

    if isinstance(raw_required_entities, dict):
        candidate = raw_required_entities.get("entities")
        if isinstance(candidate, list):
            return candidate

    return []


def normalize_required_entities(raw_required_entities: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    for raw_item in extract_required_entities(raw_required_entities):
        entity_raw = None
        privacy_raw = None

        if isinstance(raw_item, dict):
            entity_raw = raw_item.get("entity") or raw_item.get("name") or raw_item.get("field")
            privacy_raw = raw_item.get("privacy")
        elif isinstance(raw_item, str):
            entity_raw = raw_item
        else:
            continue

        entity = normalize_entity_name(entity_raw)
        if not entity:
            continue

        default_privacy = entity in PRIVATE_ENTITIES
        normalized.append(
            {
                "entity": entity,
                "privacy": coerce_bool(privacy_raw, default=default_privacy),
            }
        )

    return filter_allowed_entities(deduplicate_entities(normalized))


def extract_search_result_list(search_output: Any) -> List[Any]:
    if isinstance(search_output, list):
        return search_output

    if isinstance(search_output, dict):
        if isinstance(search_output.get("relations"), list):
            return search_output["relations"]
        if isinstance(search_output.get("results"), list) and search_output["results"]:
            return search_output["results"]
        if isinstance(search_output.get("data"), list):
            return search_output["data"]

        for key in ("relations", "results", "data"):
            nested = search_output.get(key)
            if isinstance(nested, dict):
                extracted = extract_search_result_list(nested)
                if extracted:
                    return extracted

    return []


def extract_vector_result_list(search_output: Any) -> List[Any]:
    if isinstance(search_output, list):
        return search_output

    if isinstance(search_output, dict):
        if isinstance(search_output.get("results"), list):
            return search_output["results"]
        if isinstance(search_output.get("data"), list):
            return search_output["data"]

        nested_results = search_output.get("results")
        if isinstance(nested_results, dict):
            extracted = extract_vector_result_list(nested_results)
            if extracted:
                return extracted

    return []


def normalize_search_item(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        normalized = dict(item)
        if "target" in normalized and "destination" not in normalized:
            normalized["destination"] = normalized["target"]
        if "relation" in normalized and "relationship" not in normalized:
            normalized["relationship"] = normalized["relation"]
        return normalized

    if isinstance(item, (list, tuple)) and len(item) >= 3:
        return {
            "source": item[0],
            "relationship": item[1],
            "destination": item[2],
        }

    return {"value": item}


def call_memory_search(memory: Any, query_text: str, user_id: str) -> Any:
    """
    Compatible with both:
    1) Memory.search(query=..., user_id=...)
    2) GraphMemory.search(query=..., filters={"user_id": ...})
    """
    attempts = [
        {"query": query_text, "user_id": user_id, "limit": 20},
        {"query": query_text, "user_id": user_id},
        {"query": query_text, "filters": {"user_id": user_id}},
    ]

    last_type_error: Optional[TypeError] = None
    for kwargs in attempts:
        try:
            return memory.search(**kwargs)
        except TypeError as exc:
            last_type_error = exc

    try:
        return memory.search(query_text, {"user_id": user_id})
    except TypeError:
        if last_type_error is not None:
            raise last_type_error
        raise

