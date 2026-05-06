from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .core import constants as _constants
from .core import debug as _debug
from .core import hydrator as _hydrator
from .core import pipeline as _pipeline
from .core import planner as _planner
from .core import prompts as _prompts
from .core import responder as _responder
from .core import retriever as _retriever
from .core import utils as _utils

# =========================
# Backward-compatible exports
# =========================

ALLOWED_ENTITIES = _constants.ALLOWED_ENTITIES
PRIVATE_ENTITIES = _constants.PRIVATE_ENTITIES
ENTITY_TO_RELATIONS = _constants.ENTITY_TO_RELATIONS

# Keep these tunable from this module, then sync into prompt module when called.
ANALYSIS_FEW_SHOT_EXAMPLES = _prompts.ANALYSIS_FEW_SHOT_EXAMPLES
ANSWER_FEW_SHOT_EXAMPLES = _prompts.ANSWER_FEW_SHOT_EXAMPLES
ANALYSIS_INCLUDE_REASONING_SUMMARY = _constants.ANALYSIS_INCLUDE_REASONING_SUMMARY

DEBUG = _debug.DEBUG


def set_debug(enabled: bool) -> None:
    global DEBUG
    DEBUG = bool(enabled)
    _debug.set_debug(enabled)


def _sync_prompt_config() -> None:
    _prompts.ANALYSIS_FEW_SHOT_EXAMPLES = ANALYSIS_FEW_SHOT_EXAMPLES
    _prompts.ANSWER_FEW_SHOT_EXAMPLES = ANSWER_FEW_SHOT_EXAMPLES
    _prompts.ANALYSIS_INCLUDE_REASONING_SUMMARY = ANALYSIS_INCLUDE_REASONING_SUMMARY


# =========================
# Utilities
# =========================

def _safe_json_loads(text: str) -> Dict[str, Any]:
    return _utils.safe_json_loads(text)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    return _utils.coerce_bool(value, default=default)


def _normalize_entity_name(entity: Any) -> str:
    return _utils.normalize_entity_name(entity)


def _deduplicate_entities(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _utils.deduplicate_entities(items)


def _filter_allowed_entities(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _utils.filter_allowed_entities(items)


def _extract_required_entities(raw_required_entities: Any) -> List[Any]:
    return _utils.extract_required_entities(raw_required_entities)


def _normalize_required_entities(raw_required_entities: Any) -> List[Dict[str, Any]]:
    return _utils.normalize_required_entities(raw_required_entities)


def _extract_search_result_list(search_output: Any) -> List[Any]:
    return _utils.extract_search_result_list(search_output)


def _extract_vector_result_list(search_output: Any) -> List[Any]:
    return _utils.extract_vector_result_list(search_output)


def _normalize_search_item(item: Any) -> Dict[str, Any]:
    return _utils.normalize_search_item(item)


def _call_memory_search(memory: Any, query_text: str, user_id: str) -> Any:
    return _utils.call_memory_search(memory=memory, query_text=query_text, user_id=user_id)


# =========================
# Prompt builders
# =========================

def _build_examples_section(block_title: str, examples_text: str) -> str:
    return _prompts.build_examples_section(block_title=block_title, examples_text=examples_text)


def build_analysis_prompts(query: str) -> Dict[str, str]:
    _sync_prompt_config()
    return _prompts.build_analysis_prompts(query)


def build_answer_prompts(
    query: str,
    task_plan: Dict[str, Any],
    retrieved_memories: List[Dict[str, Any]],
) -> Dict[str, str]:
    _sync_prompt_config()
    return _prompts.build_answer_prompts(
        query=query,
        task_plan=task_plan,
        retrieved_memories=retrieved_memories,
    )


# =========================
# Planner
# =========================

def analyze_required_entities(
    query: str,
    llm_call: Callable[[str, str], str],
) -> Dict[str, Any]:
    _sync_prompt_config()
    return _planner.analyze_required_entities(query=query, llm_call=llm_call)


def _default_relations_for_entity(entity: str) -> List[str]:
    return _planner.default_relations_for_entity(entity)


def map_entities_to_relations(
    required_entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return _planner.map_entities_to_relations(required_entities=required_entities)


def build_task_plan(query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    return _planner.build_task_plan(query=query, analysis=analysis)


# =========================
# Retrieval
# =========================

def debug_print(*args, **kwargs):
    return _debug.debug_print(*args, **kwargs)


def retrieve_memories_by_relations(
    memory: Any,
    user_id: str,
    required_relations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return _retriever.retrieve_memories_by_relations(
        memory=memory,
        user_id=user_id,
        required_relations=required_relations,
    )


def retrieve_memories_by_vector(
    memory: Any,
    user_id: str,
    query: str,
    required_relations: Optional[List[Dict[str, Any]]] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    return _retriever.retrieve_memories_by_vector(
        memory=memory,
        user_id=user_id,
        query=query,
        required_relations=required_relations,
        limit=limit,
    )


def retrieve_memories_hybrid(
    memory: Any,
    user_id: str,
    query: str,
    required_relations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return _retriever.retrieve_memories_hybrid(
        memory=memory,
        user_id=user_id,
        query=query,
        required_relations=required_relations,
    )


# =========================
# Hydration
# =========================

def _lookup_privacy_value_with_scope(
    privacy_lookup_fn: Callable[..., Optional[Any]],
    privacy_ref_id: str,
    user_id: str,
    result_item: Dict[str, Any],
) -> Optional[Any]:
    return _hydrator.lookup_privacy_value_with_scope(
        privacy_lookup_fn=privacy_lookup_fn,
        privacy_ref_id=privacy_ref_id,
        user_id=user_id,
        result_item=result_item,
    )


def hydrate_private_values(
    retrieved_memories: List[Dict[str, Any]],
    user_id: str,
    privacy_lookup_fn: Callable[[str, str], Optional[Any]],
) -> List[Dict[str, Any]]:
    return _hydrator.hydrate_private_values(
        retrieved_memories=retrieved_memories,
        user_id=user_id,
        privacy_lookup_fn=privacy_lookup_fn,
    )


def _lookup_vector_raw_value_by_hash(
    memory: Any,
    user_id: str,
    privacy_type: str,
    entity_hash: str,
) -> Optional[Any]:
    return _hydrator.lookup_vector_raw_value_by_hash(
        memory=memory,
        user_id=user_id,
        privacy_type=privacy_type,
        entity_hash=entity_hash,
    )


def hydrate_vector_values_by_hash(
    retrieved_memories: List[Dict[str, Any]],
    user_id: str,
    memory: Any,
) -> List[Dict[str, Any]]:
    return _hydrator.hydrate_vector_values_by_hash(
        retrieved_memories=retrieved_memories,
        user_id=user_id,
        memory=memory,
    )


# =========================
# Pipeline & answer
# =========================

def prepare_retrieved_memories_for_answer(
    query: str,
    user_id: str,
    task_plan: Dict[str, Any],
    memory: Any,
    consent: bool,
    privacy_lookup_fn: Optional[Callable[[str, str], Optional[Any]]] = None,
) -> List[Dict[str, Any]]:
    return _pipeline.prepare_retrieved_memories_for_answer(
        query=query,
        user_id=user_id,
        task_plan=task_plan,
        memory=memory,
        consent=consent,
        privacy_lookup_fn=privacy_lookup_fn,
    )


def generate_answer(
    query: str,
    task_plan: Dict[str, Any],
    retrieved_memories: List[Dict[str, Any]],
    llm_call: Callable[[str, str], str],
) -> str:
    _sync_prompt_config()
    return _responder.generate_answer(
        query=query,
        task_plan=task_plan,
        retrieved_memories=retrieved_memories,
        llm_call=llm_call,
    )


def process_query(
    query: str,
    user_id: str,
    memory: Any,
    llm_call: Callable[[str, str], str],
) -> Dict[str, Any]:
    _sync_prompt_config()
    return _pipeline.process_query(
        query=query,
        user_id=user_id,
        memory=memory,
        llm_call=llm_call,
    )


def continue_after_consent(
    query: str,
    user_id: str,
    task_plan: Dict[str, Any],
    consent: bool,
    memory: Any,
    llm_call: Callable[[str, str], str],
    privacy_lookup_fn: Optional[Callable[[str, str], Optional[Any]]] = None,
) -> Dict[str, Any]:
    _sync_prompt_config()
    return _pipeline.continue_after_consent(
        query=query,
        user_id=user_id,
        task_plan=task_plan,
        consent=consent,
        memory=memory,
        llm_call=llm_call,
        privacy_lookup_fn=privacy_lookup_fn,
    )


__all__ = [
    # constants
    "ALLOWED_ENTITIES",
    "PRIVATE_ENTITIES",
    "ENTITY_TO_RELATIONS",
    "ANALYSIS_FEW_SHOT_EXAMPLES",
    "ANSWER_FEW_SHOT_EXAMPLES",
    "ANALYSIS_INCLUDE_REASONING_SUMMARY",
    # debug
    "DEBUG",
    "set_debug",
    "debug_print",
    # utility compatibility
    "_safe_json_loads",
    "_coerce_bool",
    "_normalize_entity_name",
    "_deduplicate_entities",
    "_filter_allowed_entities",
    "_extract_required_entities",
    "_normalize_required_entities",
    "_extract_search_result_list",
    "_extract_vector_result_list",
    "_normalize_search_item",
    "_call_memory_search",
    # prompts
    "_build_examples_section",
    "build_analysis_prompts",
    "build_answer_prompts",
    # planner
    "analyze_required_entities",
    "_default_relations_for_entity",
    "map_entities_to_relations",
    "build_task_plan",
    # retrieval
    "retrieve_memories_by_relations",
    "retrieve_memories_by_vector",
    "retrieve_memories_hybrid",
    # hydration
    "_lookup_privacy_value_with_scope",
    "hydrate_private_values",
    "_lookup_vector_raw_value_by_hash",
    "hydrate_vector_values_by_hash",
    # pipeline
    "prepare_retrieved_memories_for_answer",
    "generate_answer",
    "process_query",
    "continue_after_consent",
]
