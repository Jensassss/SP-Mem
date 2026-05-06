from __future__ import annotations

from typing import Any, Callable, Dict, List

from .debug import debug_print
from .prompts import build_answer_prompts


def generate_answer(
    query: str,
    task_plan: Dict[str, Any],
    retrieved_memories: List[Dict[str, Any]],
    llm_call: Callable[[str, str], str],
) -> str:
    prompts = build_answer_prompts(
        query=query,
        task_plan=task_plan,
        retrieved_memories=retrieved_memories,
    )

    answer = llm_call(
        prompts["system_prompt"],
        prompts["user_prompt"],
    )

    answer = str(answer).strip()

    debug_print("\n[FINAL_ANSWER]")
    debug_print(answer)

    return answer

