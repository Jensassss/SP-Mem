from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from .constants import ENTITY_TO_RELATIONS, PRIVATE_ENTITIES
from .debug import debug_print
from .utils import (
    call_memory_search,
    extract_search_result_list,
    extract_vector_result_list,
    normalize_search_item,
)


ENTITY_PRIVACY_TYPE_OVERRIDES: Dict[str, Set[str]] = {
    "name": {"PERSON_NAME"},
    "phone_number": {"PHONE"},
    "insurance": {"INSURANCE_RECORD"},
}


def _normalize_retrieval_mode(value: Optional[str]) -> str:
    if not value:
        return "hybrid"

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"hybrid", "both", "all"}:
        return "hybrid"
    if normalized in {"graph", "graph_only", "no_vector"}:
        return "graph_only"
    if normalized in {"vector", "vector_only", "no_graph"}:
        return "vector_only"
    return "hybrid"


def _resolve_retrieval_mode(explicit_mode: Optional[str] = None) -> str:
    if explicit_mode:
        return _normalize_retrieval_mode(explicit_mode)

    env_mode = os.getenv("PRIVACY_AGENT_RETRIEVAL_MODE")
    if not env_mode:
        # Reuse existing slurm ablation flag when available.
        env_mode = os.getenv("ABLATION_MODE")
    return _normalize_retrieval_mode(env_mode)


def _norm_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().replace("_", " ").split())


def _expected_privacy_types(entity: str) -> Set[str]:
    entity = _norm_text(entity).replace(" ", "_")
    if not entity:
        return set()
    expected = {entity.upper()}
    expected.update(ENTITY_PRIVACY_TYPE_OVERRIDES.get(entity, set()))
    return expected


def _extract_item_privacy_types(item: Dict[str, Any]) -> Set[str]:
    types: Set[str] = set()
    if not isinstance(item, dict):
        return types

    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        return types

    privacy_types = metadata.get("privacy_types")
    if isinstance(privacy_types, list):
        for privacy_type in privacy_types:
            if isinstance(privacy_type, str) and privacy_type.strip():
                types.add(privacy_type.strip().upper())

    privacy_entities = metadata.get("privacy_entities")
    if isinstance(privacy_entities, list):
        for privacy_entity in privacy_entities:
            if not isinstance(privacy_entity, dict):
                continue
            privacy_type = privacy_entity.get("privacy_type")
            if isinstance(privacy_type, str) and privacy_type.strip():
                types.add(privacy_type.strip().upper())

    return types


def _score_item_for_entity(item: Dict[str, Any], entity: str) -> int:
    if entity == "__query__":
        return 1

    score = 0
    entity_key = _norm_text(entity).replace(" ", "_")
    entity_phrase = _norm_text(entity)
    if not entity_key:
        return score

    memory_text = _norm_text(item.get("memory"))
    relationship_text = _norm_text(item.get("relationship") or item.get("relation"))
    destination_text = _norm_text(item.get("destination"))

    if entity_phrase and (entity_phrase in memory_text or entity_phrase in destination_text):
        score += 2

    expected_relations = ENTITY_TO_RELATIONS.get(entity_key, [])
    for relation in expected_relations:
        relation_phrase = _norm_text(relation)
        if not relation_phrase:
            continue
        if relationship_text == relation_phrase:
            score += 2
        elif relation_phrase in memory_text or relation_phrase in destination_text:
            score += 1

    expected_types = _expected_privacy_types(entity_key)
    matched_types = _extract_item_privacy_types(item)
    if expected_types.intersection(matched_types):
        score += 3

    return score


def _filter_vector_results_by_entity(results: List[Dict[str, Any]], entity: str) -> Tuple[List[Dict[str, Any]], bool]:
    if entity == "__query__":
        return results, False

    scored: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, item in enumerate(results):
        score = _score_item_for_entity(item, entity)
        if score > 0:
            scored.append((idx, score, item))

    if scored:
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [x[2] for x in scored], False

    # Fallback: keep one semantic hit to avoid empty retrieval for paraphrased entities.
    return results[:1], True


def retrieve_memories_by_relations(
    memory: Any,
    user_id: str,
    required_relations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    collected = []

    debug_print("\n" + "-" * 80)
    debug_print("[MEMORY_RETRIEVAL] START")
    debug_print("[MEMORY_RETRIEVAL] user_id:", user_id)

    for item in required_relations:
        if not isinstance(item, dict):
            debug_print("[MEMORY_RETRIEVAL] skip non-dict item:", item)
            continue

        entity = item.get("entity")
        relation = item.get("relation")
        privacy = bool(item.get("privacy", entity in PRIVATE_ENTITIES if entity else False))

        if not relation:
            debug_print("[MEMORY_RETRIEVAL] skip empty relation for item:", item)
            continue

        debug_print(f"\n[MEMORY_SEARCH] entity={entity}, relation={relation}, privacy={privacy}")
        debug_print(f"[MEMORY_SEARCH] search_query={relation}")

        raw_search_output = memory.graph._search_graph_db_by_relations(
            relations=[relation],
            filters={"user_id": user_id},
            limit=20,
        )

        debug_print("[MEMORY_SEARCH] raw_search_output:")
        debug_print(raw_search_output)

        raw_results = extract_search_result_list(raw_search_output)
        normalized_results = [normalize_search_item(x) for x in raw_results]

        debug_print(f"[MEMORY_SEARCH] normalized_results_count={len(normalized_results)}")
        for idx, item_result in enumerate(normalized_results[:5]):
            debug_print(f"[MEMORY_SEARCH] result_{idx}: {item_result}")

        collected.append(
            {
                "source": "graph",
                "entity": entity,
                "relation": relation,
                "privacy": privacy,
                "search_query": relation,
                "results": normalized_results,
            }
        )

    debug_print("\n[MEMORY_RETRIEVAL] FINAL_COLLECTED:")
    debug_print(json.dumps(collected, ensure_ascii=False, indent=2))
    debug_print("[MEMORY_RETRIEVAL] END")
    debug_print("-" * 80)

    return collected


def retrieve_memories_by_vector(
    memory: Any,
    user_id: str,
    query: str,
    required_relations: Optional[List[Dict[str, Any]]] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    def _build_vector_targets(relations: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        targets: List[Dict[str, Any]] = []
        seen = set()
        for item in relations or []:
            if not isinstance(item, dict):
                continue
            entity = item.get("entity")
            if not entity:
                continue
            entity_text = str(entity).strip()
            if not entity_text:
                continue
            entity_key = entity_text.lower()
            if entity_key in seen:
                continue
            seen.add(entity_key)
            targets.append(
                {
                    "entity": entity_text,
                    "search_query": entity_text.replace("_", " "),
                    "privacy": bool(item.get("privacy", False)),
                }
            )
        return targets

    debug_print("\n[MEMORY_RETRIEVAL][VECTOR] START")
    debug_print("[MEMORY_RETRIEVAL][VECTOR] user_id:", user_id)
    debug_print("[MEMORY_RETRIEVAL][VECTOR] query:", query)

    targets = _build_vector_targets(required_relations)
    if not targets:
        # Fallback only when no parsed entities are available.
        targets = [
            {
                "entity": "__query__",
                "search_query": query,
                "privacy": False,
            }
        ]

    collected: List[Dict[str, Any]] = []
    for target in targets:
        search_query = target["search_query"]
        entity = target["entity"]
        privacy = bool(target.get("privacy", False))

        debug_print(f"[MEMORY_RETRIEVAL][VECTOR] entity={entity}, search_query={search_query}")

        raw_search_output = call_memory_search(memory=memory, query_text=search_query, user_id=user_id)
        debug_print("[MEMORY_RETRIEVAL][VECTOR] raw_search_output:")
        debug_print(raw_search_output)

        raw_results = extract_vector_result_list(raw_search_output)
        if limit > 0:
            raw_results = raw_results[:limit]
        normalized_results = [normalize_search_item(x) for x in raw_results]
        filtered_results, used_fallback = _filter_vector_results_by_entity(normalized_results, entity)

        debug_print(
            f"[MEMORY_RETRIEVAL][VECTOR] normalized_results_count={len(normalized_results)} "
            f"filtered_results_count={len(filtered_results)} fallback={used_fallback}"
        )
        for idx, item_result in enumerate(filtered_results[:5]):
            debug_print(f"[MEMORY_RETRIEVAL][VECTOR] result_{idx}: {item_result}")

        if not filtered_results:
            continue

        collected.append(
            {
                "source": "vector",
                "entity": entity,
                "relation": "__vector_search__",
                "privacy": privacy,
                "search_query": search_query,
                "results": filtered_results,
            }
        )

    debug_print("[MEMORY_RETRIEVAL][VECTOR] END")
    return collected


def retrieve_memories_hybrid(
    memory: Any,
    user_id: str,
    query: str,
    required_relations: List[Dict[str, Any]],
    retrieval_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    mode = _resolve_retrieval_mode(retrieval_mode)
    debug_print(f"[MEMORY_RETRIEVAL] retrieval_mode={mode}")

    graph_groups: List[Dict[str, Any]] = []
    if mode in {"hybrid", "graph_only"}:
        graph_groups = retrieve_memories_by_relations(
            memory=memory,
            user_id=user_id,
            required_relations=required_relations,
        )

    vector_groups: List[Dict[str, Any]] = []
    if mode in {"hybrid", "vector_only"}:
        vector_groups = retrieve_memories_by_vector(
            memory=memory,
            user_id=user_id,
            query=query,
            required_relations=required_relations,
            limit=20,
        )

    return graph_groups + vector_groups
