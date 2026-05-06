from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from .debug import debug_print


def lookup_privacy_value_with_scope(
    privacy_lookup_fn: Callable[..., Optional[Any]],
    privacy_ref_id: str,
    user_id: str,
    result_item: Dict[str, Any],
) -> Optional[Any]:
    try:
        return privacy_lookup_fn(privacy_ref_id, user_id)
    except TypeError:
        return privacy_lookup_fn(
            privacy_ref_id=privacy_ref_id,
            user_id=user_id,
            agent_id=result_item.get("agent_id"),
            run_id=result_item.get("run_id"),
        )


def hydrate_private_values(
    retrieved_memories: List[Dict[str, Any]],
    user_id: str,
    privacy_lookup_fn: Callable[[str, str], Optional[Any]],
) -> List[Dict[str, Any]]:
    hydrated = []

    debug_print("\n" + "-" * 80)
    debug_print("[PRIVACY_HYDRATION] START")
    debug_print("[PRIVACY_HYDRATION] user_id:", user_id)

    for group in retrieved_memories:
        group_copy = dict(group)
        results = group_copy.get("results", [])

        if not isinstance(results, list):
            hydrated.append(group_copy)
            continue

        new_results = []

        for result in results:
            if not isinstance(result, dict):
                new_results.append(result)
                continue

            item = dict(result)

            for ref_key in ("privacy_ref_id", "source_privacy_ref_id", "destination_privacy_ref_id"):
                privacy_ref_id = item.get(ref_key)
                if not privacy_ref_id:
                    continue

                raw_value = lookup_privacy_value_with_scope(
                    privacy_lookup_fn=privacy_lookup_fn,
                    privacy_ref_id=privacy_ref_id,
                    user_id=user_id,
                    result_item=item,
                )
                if raw_value is None:
                    debug_print(f"[PRIVACY_HYDRATION] unresolved {ref_key}={privacy_ref_id}")
                    continue

                debug_print(
                    f"[PRIVACY_HYDRATION] resolved {ref_key}={privacy_ref_id} -> {raw_value}"
                )

                if ref_key == "privacy_ref_id":
                    item["resolved_value"] = raw_value
                elif ref_key == "source_privacy_ref_id":
                    item["resolved_source_value"] = raw_value
                else:
                    item["resolved_destination_value"] = raw_value

            new_results.append(item)

        group_copy["results"] = new_results
        hydrated.append(group_copy)

    debug_print("[PRIVACY_HYDRATION] hydrated_result:")
    debug_print(json.dumps(hydrated, ensure_ascii=False, indent=2))
    debug_print("[PRIVACY_HYDRATION] END")
    debug_print("-" * 80)

    return hydrated


def lookup_vector_raw_value_by_hash(
    memory: Any,
    user_id: str,
    privacy_type: str,
    entity_hash: str,
) -> Optional[Any]:
    if not privacy_type or not entity_hash:
        return None

    db = getattr(memory, "db", None)
    if db is None:
        return None

    get_privacy_mapping = getattr(db, "get_privacy_mapping", None)
    if not callable(get_privacy_mapping):
        return None

    try:
        mapping = get_privacy_mapping(
            user_id=user_id,
            privacy_type=privacy_type,
            raw_hash=entity_hash,
        )
    except TypeError:
        mapping = get_privacy_mapping(user_id, privacy_type, entity_hash)

    if not isinstance(mapping, dict):
        return None

    return mapping.get("raw_value")


def hydrate_vector_values_by_hash(
    retrieved_memories: List[Dict[str, Any]],
    user_id: str,
    memory: Any,
) -> List[Dict[str, Any]]:
    hydrated = []

    debug_print("\n" + "-" * 80)
    debug_print("[VECTOR_HASH_HYDRATION] START")
    debug_print("[VECTOR_HASH_HYDRATION] user_id:", user_id)

    for group in retrieved_memories:
        group_copy = dict(group)

        is_vector_group = (
            group_copy.get("source") == "vector"
            or group_copy.get("relation") == "__vector_search__"
        )
        if not is_vector_group:
            hydrated.append(group_copy)
            continue

        results = group_copy.get("results", [])
        if not isinstance(results, list):
            hydrated.append(group_copy)
            continue

        new_results = []
        for result in results:
            if not isinstance(result, dict):
                new_results.append(result)
                continue

            item = dict(result)
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                new_results.append(item)
                continue

            privacy_entities = metadata.get("privacy_entities", [])
            if not isinstance(privacy_entities, list):
                new_results.append(item)
                continue

            resolved_privacy_entities = []
            resolved_memory = item.get("memory")

            for privacy_entity in privacy_entities:
                if not isinstance(privacy_entity, dict):
                    continue

                privacy_type = privacy_entity.get("privacy_type")
                entity_hash = privacy_entity.get("entity_hash")
                sanitized_value = privacy_entity.get("sanitized_value")

                raw_value = lookup_vector_raw_value_by_hash(
                    memory=memory,
                    user_id=user_id,
                    privacy_type=privacy_type,
                    entity_hash=entity_hash,
                )
                if raw_value is None:
                    continue

                debug_print(
                    f"[VECTOR_HASH_HYDRATION] resolved privacy_type={privacy_type}, "
                    f"entity_hash={entity_hash} -> {raw_value}"
                )

                resolved_entity = dict(privacy_entity)
                resolved_entity["resolved_value"] = raw_value
                resolved_privacy_entities.append(resolved_entity)

                if isinstance(resolved_memory, str) and isinstance(sanitized_value, str) and sanitized_value:
                    resolved_memory = resolved_memory.replace(sanitized_value, str(raw_value))

            if resolved_privacy_entities:
                item["resolved_privacy_entities"] = resolved_privacy_entities
                if isinstance(resolved_memory, str):
                    item["resolved_memory"] = resolved_memory

            new_results.append(item)

        group_copy["results"] = new_results
        hydrated.append(group_copy)

    debug_print("[VECTOR_HASH_HYDRATION] hydrated_result:")
    debug_print(json.dumps(hydrated, ensure_ascii=False, indent=2))
    debug_print("[VECTOR_HASH_HYDRATION] END")
    debug_print("-" * 80)

    return hydrated

