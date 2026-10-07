from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from .debug import debug_print
from .hydrator import hydrate_private_values, hydrate_vector_values_by_hash
from .planner import analyze_required_entities, build_task_plan
from .responder import generate_answer
from .retriever import retrieve_memories_hybrid


def prepare_retrieved_memories_for_answer(
    query: str,
    user_id: str,
    task_plan: Dict[str, Any],
    memory: Any,
    consent: bool,
    privacy_lookup_fn: Optional[Callable[[str, str], Optional[Any]]] = None,
    release_authorization: Any = None,
    authorization_validator: Optional[Callable[..., bool]] = None,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    retrieved_memories = retrieve_memories_hybrid(
        memory=memory,
        user_id=user_id,
        query=query,
        required_relations=task_plan["required_relations"],
    )

    release_authorized = False
    if consent and release_authorization is not None and callable(authorization_validator) and session_id:
        try:
            release_authorized = bool(
                authorization_validator(
                    release_authorization,
                    user_id=user_id,
                    session_id=session_id,
                )
            )
        except Exception:
            release_authorized = False

    if release_authorized:
        if privacy_lookup_fn is not None:
            retrieved_memories = hydrate_private_values(
                retrieved_memories=retrieved_memories,
                user_id=user_id,
                privacy_lookup_fn=privacy_lookup_fn,
                release_authorization=release_authorization,
                authorization_validator=authorization_validator,
                session_id=session_id,
            )

        retrieved_memories = hydrate_vector_values_by_hash(
            retrieved_memories=retrieved_memories,
            user_id=user_id,
            memory=memory,
            release_authorization=release_authorization,
            authorization_validator=authorization_validator,
            session_id=session_id,
        )

    return retrieved_memories


def process_query(
    query: str,
    user_id: str,
    memory: Any,
    llm_call: Callable[[str, str], str],
) -> Dict[str, Any]:
    analysis = analyze_required_entities(
        query=query,
        llm_call=llm_call,
    )

    task_plan = build_task_plan(query=query, analysis=analysis)

    if task_plan["needs_privacy"]:
        privacy_fields = ", ".join(task_plan.get("privacy_entities", [])) or "sensitive fields"

        result = {
            "status": "awaiting_consent",
            "task_plan": task_plan,
            "message": (
                "To answer this accurately, I need to use sensitive memory fields: "
                f"{privacy_fields}. "
                "Do I have your permission to use these fields for this request?"
            ),
        }
        return result

    retrieved_memories = prepare_retrieved_memories_for_answer(
        query=query,
        user_id=user_id,
        task_plan=task_plan,
        memory=memory,
        consent=False,
        privacy_lookup_fn=None,
    )

    debug_print("\n[PROCESS_QUERY] retrieved_memories:")
    debug_print(json.dumps(retrieved_memories, ensure_ascii=False, indent=2))

    answer = generate_answer(
        query=query,
        task_plan=task_plan,
        retrieved_memories=retrieved_memories,
        llm_call=llm_call,
    )

    result = {
        "status": "answered",
        "task_plan": task_plan,
        "retrieved_memories": retrieved_memories,
        "answer": answer,
        "consent": False,
    }

    return result


def continue_after_consent(
    query: str,
    user_id: str,
    task_plan: Dict[str, Any],
    consent: bool,
    memory: Any,
    llm_call: Callable[[str, str], str],
    privacy_lookup_fn: Optional[Callable[[str, str], Optional[Any]]] = None,
    release_authorization: Any = None,
    authorization_validator: Optional[Callable[..., bool]] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    retrieved_memories = prepare_retrieved_memories_for_answer(
        query=query,
        user_id=user_id,
        task_plan=task_plan,
        memory=memory,
        consent=consent,
        privacy_lookup_fn=privacy_lookup_fn,
        release_authorization=release_authorization,
        authorization_validator=authorization_validator,
        session_id=session_id,
    )

    answer = generate_answer(
        query=query,
        task_plan=task_plan,
        retrieved_memories=retrieved_memories,
        llm_call=llm_call,
    )

    result = {
        "status": "answered",
        "task_plan": task_plan,
        "retrieved_memories": retrieved_memories,
        "answer": answer,
        "consent": consent,
    }

    if (
        consent
        and release_authorization is not None
        and privacy_lookup_fn is None
        and task_plan.get("needs_privacy")
    ):
        result["warning"] = (
            "consent=true but privacy_lookup_fn is missing; "
            "answer is generated from currently retrievable (possibly sanitized) values."
        )

    return result
