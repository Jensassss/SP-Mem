from __future__ import annotations

from typing import Any, Callable, Dict, List

from .constants import ENTITY_TO_RELATIONS, PRIVATE_ENTITIES
from .debug import debug_print
from .prompts import build_analysis_prompts
from .utils import normalize_required_entities, safe_json_loads


def analyze_required_entities(
    query: str,
    llm_call: Callable[[str, str], str],
) -> Dict[str, Any]:
    prompts = build_analysis_prompts(query)
    raw_text = llm_call(
        prompts["system_prompt"],
        prompts["user_prompt"],
    )

    parsed = safe_json_loads(raw_text)
    required_entities = normalize_required_entities(parsed.get("required_entities", []))

    for item in required_entities:
        if item["entity"] in PRIVATE_ENTITIES:
            item["privacy"] = True

    needs_privacy = any(item.get("privacy", False) for item in required_entities)
    intent = parsed.get("intent", "unknown")
    if not isinstance(intent, str) or not intent.strip():
        intent = "unknown"
    else:
        intent = intent.strip()

    raw_reasoning_process = parsed.get("reasoning_process", [])
    reasoning_process: List[str] = []
    if isinstance(raw_reasoning_process, list):
        for step in raw_reasoning_process:
            if not isinstance(step, str):
                continue
            step = step.strip()
            if step:
                reasoning_process.append(step)
    elif isinstance(raw_reasoning_process, str):
        step = raw_reasoning_process.strip()
        if step:
            reasoning_process.append(step)

    reasoning_summary = parsed.get("reasoning_summary", "")
    if not isinstance(reasoning_summary, str):
        reasoning_summary = ""
    else:
        reasoning_summary = reasoning_summary.strip()
    if not reasoning_summary and reasoning_process:
        reasoning_summary = " ".join(reasoning_process[:2]).strip()

    debug_print("\n[QUERY_ANALYSIS]")
    debug_print("reasoning_process:", reasoning_process)
    debug_print("reasoning:", reasoning_summary)
    debug_print("entities:", required_entities)

    return {
        "intent": intent,
        "reasoning_process": reasoning_process,
        "reasoning_summary": reasoning_summary,
        "required_entities": required_entities,
        "needs_privacy": needs_privacy,
        "raw_analysis_text": raw_text,
    }


def default_relations_for_entity(entity: str) -> List[str]:
    """
    Fallback rule:
    If entity has no explicit mapping, try has_<entity>.
    If entity itself looks like a direct relation, also keep entity.
    """
    if not isinstance(entity, str) or not entity.strip():
        return []

    entity = entity.strip()

    candidates = []
    if entity in ENTITY_TO_RELATIONS:
        candidates.extend(ENTITY_TO_RELATIONS[entity])
    else:
        candidates.append(f"has_{entity}")
        candidates.append(entity)

    seen = set()
    deduped = []
    for rel in candidates:
        if not isinstance(rel, str):
            continue
        rel = rel.strip()
        if not rel or rel in seen:
            continue
        seen.add(rel)
        deduped.append(rel)

    return deduped


def map_entities_to_relations(
    required_entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    mapped = []
    seen = set()

    for item in required_entities:
        if not isinstance(item, dict):
            continue

        entity = item.get("entity")
        if not isinstance(entity, str) or not entity.strip():
            continue

        entity = entity.strip()
        privacy = bool(item.get("privacy", entity in PRIVATE_ENTITIES))
        relations = default_relations_for_entity(entity)

        for relation in relations:
            key = (entity, relation)
            if key in seen:
                continue
            seen.add(key)
            mapped.append(
                {
                    "entity": entity,
                    "relation": relation,
                    "privacy": privacy,
                }
            )
    return mapped


def build_task_plan(query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    required_entities = analysis.get("required_entities", [])
    required_relations = map_entities_to_relations(required_entities)

    privacy_entities = [x["entity"] for x in required_entities if x.get("privacy")]
    non_privacy_entities = [x["entity"] for x in required_entities if not x.get("privacy")]

    privacy_relations = [x["relation"] for x in required_relations if x.get("privacy")]
    non_privacy_relations = [x["relation"] for x in required_relations if not x.get("privacy")]

    task_plan = {
        "query": query,
        "intent": analysis.get("intent", "unknown"),
        "reasoning_process": analysis.get("reasoning_process", []),
        "reasoning_summary": analysis.get("reasoning_summary", ""),
        "required_entities": required_entities,
        "required_relations": required_relations,
        "privacy_entities": privacy_entities,
        "non_privacy_entities": non_privacy_entities,
        "privacy_relations": privacy_relations,
        "non_privacy_relations": non_privacy_relations,
        "needs_privacy": bool(analysis.get("needs_privacy", False)),
    }

    return task_plan

