from __future__ import annotations

from copy import deepcopy
import inspect
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from . import query_processor as qp


class PrivacyAwareAgent:
    """
    Privacy-aware agent orchestrator.

    Flow:
    1) Analyze query -> required entities + privacy need.
    2) If privacy is needed, return awaiting_consent state.
    3) If consent is false, answer from sanitized retrieval results.
    4) If consent is true, hydrate private values by privacy_ref_id/hash and answer.
    """

    def __init__(
        self,
        memory: Any,
        llm_call: Callable[[str, str], str],
        *,
        privacy_lookup_fn: Optional[Callable[..., Optional[Any]]] = None,
    ) -> None:
        self.memory = memory
        self.llm_call = llm_call
        self.qp = qp
        self.privacy_lookup_fn = privacy_lookup_fn or self._autowire_privacy_lookup(memory)
        self._pending_sessions: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _autowire_privacy_lookup(memory: Any) -> Optional[Callable[..., Optional[Any]]]:
        graph = getattr(memory, "graph", None)
        if graph is None:
            return None

        fn = getattr(graph, "lookup_privacy_value", None)
        if callable(fn):
            return fn
        return None

    def analyze(self, query: str) -> Dict[str, Any]:
        analysis = self.qp.analyze_required_entities(query=query, llm_call=self.llm_call)
        task_plan = self.qp.build_task_plan(query=query, analysis=analysis)
        return {"analysis": analysis, "task_plan": task_plan}

    def _build_awaiting_consent_response(
        self,
        *,
        query: str,
        user_id: str,
        task_plan: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        sid = session_id or str(uuid.uuid4())
        self._pending_sessions[sid] = {
            "query": query,
            "user_id": user_id,
            "task_plan": task_plan,
        }

        privacy_fields = ", ".join(task_plan.get("privacy_entities", [])) or "sensitive fields"
        return {
            "status": "awaiting_consent",
            "session_id": sid,
            "task_plan": task_plan,
            "message": (
                "To answer this accurately, I need to use sensitive memory fields: "
                f"{privacy_fields}. "
                "Do I have your permission to use these fields for this request?"
            ),
        }

    @staticmethod
    def _materialize_precise_values(retrieved_memories: Any) -> Any:
        materialized = []

        if not isinstance(retrieved_memories, list):
            return retrieved_memories

        for group in retrieved_memories:
            if not isinstance(group, dict):
                materialized.append(group)
                continue

            group_copy = dict(group)
            results = group_copy.get("results", [])
            if not isinstance(results, list):
                materialized.append(group_copy)
                continue

            new_results = []
            for result in results:
                if not isinstance(result, dict):
                    new_results.append(result)
                    continue

                item = deepcopy(result)

                resolved_memory = item.get("resolved_memory")
                if isinstance(resolved_memory, str) and resolved_memory.strip():
                    item["memory"] = resolved_memory

                if item.get("resolved_value") is not None:
                    item["value"] = item["resolved_value"]

                if item.get("resolved_source_value") is not None:
                    item["source"] = item["resolved_source_value"]

                if item.get("resolved_destination_value") is not None:
                    item["destination"] = item["resolved_destination_value"]

                resolved_privacy_entities = item.get("resolved_privacy_entities")
                if isinstance(resolved_privacy_entities, list) and isinstance(item.get("memory"), str):
                    for privacy_entity in resolved_privacy_entities:
                        if not isinstance(privacy_entity, dict):
                            continue
                        raw_value = privacy_entity.get("resolved_value")
                        sanitized_value = privacy_entity.get("sanitized_value")
                        if raw_value is None:
                            continue
                        if isinstance(sanitized_value, str) and sanitized_value:
                            item["memory"] = item["memory"].replace(sanitized_value, str(raw_value))

                new_results.append(item)

            group_copy["results"] = new_results
            materialized.append(group_copy)

        return materialized

    @staticmethod
    def _to_clean_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return text

    @staticmethod
    def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            text = PrivacyAwareAgent._to_clean_text(value)
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    @staticmethod
    def _normalize_entity_key(entity: Any) -> str:
        text = PrivacyAwareAgent._to_clean_text(entity).lower()
        if not text:
            return ""
        return text.replace("-", "_").replace(" ", "_")

    @staticmethod
    def _privacy_type_to_entity_key(privacy_type: Any) -> str:
        key = PrivacyAwareAgent._normalize_entity_key(privacy_type)
        if not key:
            return ""
        alias = {
            "person_name": "name",
            "phone": "phone_number",
        }
        return alias.get(key, key)

    @staticmethod
    def _extract_graph_values(group: Dict[str, Any]) -> List[str]:
        values: List[str] = []
        results = group.get("results", [])
        if not isinstance(results, list):
            return values

        for result in results:
            if isinstance(result, dict):
                for key in ("destination", "value", "memory"):
                    text = PrivacyAwareAgent._to_clean_text(result.get(key))
                    if text:
                        values.append(text)
            else:
                text = PrivacyAwareAgent._to_clean_text(result)
                if text:
                    values.append(text)
        return PrivacyAwareAgent._dedupe_keep_order(values)

    @staticmethod
    def _extract_vector_values(
        group: Dict[str, Any],
        *,
        consent: bool,
        graph_exact_by_entity: Dict[str, List[str]],
    ) -> Tuple[List[str], str]:
        values: List[str] = []
        inferred_entity = ""

        results = group.get("results", [])
        if not isinstance(results, list):
            return values, inferred_entity

        for result in results:
            if isinstance(result, dict):
                memory_text = PrivacyAwareAgent._to_clean_text(result.get("memory"))
                metadata = result.get("metadata", {})
                privacy_entities = []
                if isinstance(metadata, dict) and isinstance(metadata.get("privacy_entities"), list):
                    privacy_entities = metadata["privacy_entities"]

                candidate_entities: List[str] = []
                for pe in privacy_entities:
                    if not isinstance(pe, dict):
                        continue
                    entity_key = PrivacyAwareAgent._privacy_type_to_entity_key(pe.get("privacy_type"))
                    if entity_key:
                        candidate_entities.append(entity_key)
                        if not inferred_entity:
                            inferred_entity = entity_key

                if consent and privacy_entities and memory_text:
                    # Prefer exact replacement via graph value when vector hash hydration misses.
                    for pe in privacy_entities:
                        if not isinstance(pe, dict):
                            continue
                        sanitized_value = PrivacyAwareAgent._to_clean_text(pe.get("sanitized_value"))
                        entity_key = PrivacyAwareAgent._privacy_type_to_entity_key(pe.get("privacy_type"))
                        if not sanitized_value or not entity_key:
                            continue
                        graph_vals = graph_exact_by_entity.get(entity_key, [])
                        if not graph_vals:
                            continue
                        exact_value = graph_vals[0]
                        if sanitized_value in memory_text:
                            memory_text = memory_text.replace(sanitized_value, exact_value)

                if consent:
                    exact_fallback = ""
                    for entity_key in candidate_entities:
                        graph_vals = graph_exact_by_entity.get(entity_key, [])
                        if graph_vals:
                            exact_fallback = graph_vals[0]
                            break

                    # Safety fallback: if vector side did not hydrate to precise value,
                    # keep text but append exact value from graph if available.
                    if exact_fallback:
                        if not memory_text:
                            label = (inferred_entity or "value").replace("_", " ")
                            memory_text = f"{label} is {exact_fallback}"
                        elif exact_fallback not in memory_text:
                            memory_text = f"{memory_text} (exact_value: {exact_fallback})"

                if memory_text:
                    values.append(memory_text)
            else:
                text = PrivacyAwareAgent._to_clean_text(result)
                if text:
                    values.append(text)

        return PrivacyAwareAgent._dedupe_keep_order(values), inferred_entity

    @staticmethod
    def _compact_retrieved_memories_for_prompt(
        retrieved_memories: Any,
        *,
        consent: bool,
    ) -> List[Dict[str, Any]]:
        if not isinstance(retrieved_memories, list):
            return []

        graph_exact_by_entity: Dict[str, List[str]] = {}
        for group in retrieved_memories:
            if not isinstance(group, dict):
                continue
            if group.get("source") != "graph":
                continue
            entity_key = PrivacyAwareAgent._normalize_entity_key(group.get("entity"))
            values = PrivacyAwareAgent._extract_graph_values(group)
            if not entity_key or not values:
                continue
            graph_exact_by_entity[entity_key] = values

        compacted: List[Dict[str, Any]] = []
        for group in retrieved_memories:
            if not isinstance(group, dict):
                continue

            source = PrivacyAwareAgent._to_clean_text(group.get("source")) or "unknown"
            entity = PrivacyAwareAgent._to_clean_text(group.get("entity")) or "unknown"

            if source == "graph":
                values = PrivacyAwareAgent._extract_graph_values(group)
                if not values:
                    continue
                compacted.append(
                    {
                        "source": "graph",
                        "entity": entity,
                        "results": values,
                    }
                )
                continue

            if source == "vector":
                values, inferred_entity = PrivacyAwareAgent._extract_vector_values(
                    group,
                    consent=consent,
                    graph_exact_by_entity=graph_exact_by_entity,
                )
                if not values:
                    continue
                display_entity = entity
                if entity == "__query__" and inferred_entity:
                    display_entity = inferred_entity
                compacted.append(
                    {
                        "source": "vector",
                        "entity": display_entity,
                        "results": values,
                    }
                )
                continue

        return compacted

    def ask(self, query: str, user_id: str, *, session_id: Optional[str] = None) -> Dict[str, Any]:
        analyzed = self.analyze(query)
        task_plan = analyzed["task_plan"]

        if task_plan.get("needs_privacy", False):
            return self._build_awaiting_consent_response(
                query=query,
                user_id=user_id,
                task_plan=task_plan,
                session_id=session_id,
            )

        return self._execute(query=query, user_id=user_id, task_plan=task_plan, consent=False)

    def continue_with_consent(self, session_id: str, consent: bool) -> Dict[str, Any]:
        pending = self._pending_sessions.get(session_id)
        if not pending:
            return {
                "status": "error",
                "message": f"Unknown or expired session_id: {session_id}",
            }

        result = self._execute(
            query=pending["query"],
            user_id=pending["user_id"],
            task_plan=pending["task_plan"],
            consent=bool(consent),
        )
        self._pending_sessions.pop(session_id, None)
        return result

    def run(
        self,
        query: str,
        user_id: str,
        *,
        consent: Optional[bool] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convenience API:
        - consent=None: start flow
        - consent is bool and session_id provided: continue an awaiting-consent session
        - consent is bool and no session_id: one-shot execution
        """
        if consent is not None and session_id:
            return self.continue_with_consent(session_id=session_id, consent=consent)

        analyzed = self.analyze(query)
        task_plan = analyzed["task_plan"]

        if task_plan.get("needs_privacy", False) and consent is None:
            return self._build_awaiting_consent_response(
                query=query,
                user_id=user_id,
                task_plan=task_plan,
                session_id=session_id,
            )

        return self._execute(
            query=query,
            user_id=user_id,
            task_plan=task_plan,
            consent=bool(consent),
        )

    def _execute(
        self,
        *,
        query: str,
        user_id: str,
        task_plan: Dict[str, Any],
        consent: bool,
    ) -> Dict[str, Any]:
        retrieved_memories = self.qp.prepare_retrieved_memories_for_answer(
            query=query,
            user_id=user_id,
            task_plan=task_plan,
            memory=self.memory,
            consent=consent,
            privacy_lookup_fn=self.privacy_lookup_fn,
        )
        if consent:
            retrieved_memories = self._materialize_precise_values(retrieved_memories)

        prompt_retrieved_memories = self._compact_retrieved_memories_for_prompt(
            retrieved_memories,
            consent=consent,
        )

        answer = self.qp.generate_answer(
            query=query,
            task_plan=task_plan,
            retrieved_memories=prompt_retrieved_memories,
            llm_call=self.llm_call,
        )

        result: Dict[str, Any] = {
            "status": "answered",
            "task_plan": task_plan,
            "retrieved_memories": retrieved_memories,
            "retrieved_memories_for_prompt": prompt_retrieved_memories,
            "answer": answer,
            "consent": consent,
        }

        if consent and self.privacy_lookup_fn is None and task_plan.get("needs_privacy"):
            result["warning"] = (
                "consent=true but privacy_lookup_fn is missing; "
                "graph private values may remain sanitized."
            )

        return result

    @staticmethod
    def _clean_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role is None or content is None:
                continue
            cleaned.append({"role": role, "content": content})
        return cleaned

    def add_memory(
        self,
        messages: Iterable[Dict[str, Any]],
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Optional helper to write memory through the same memory backend.
        This method is not used by ask()/run() unless you call it explicitly.
        """
        add_fn = getattr(self.memory, "add", None)
        if not callable(add_fn):
            return {
                "status": "error",
                "message": "memory backend does not expose add(...)",
            }

        cleaned = self._clean_messages(messages)
        if not cleaned:
            return {
                "status": "skipped",
                "message": "no valid messages to add",
                "added_messages": 0,
            }

        try:
            result = add_fn(cleaned, user_id=user_id)
        except TypeError:
            result = add_fn(messages=cleaned, user_id=user_id)

        if inspect.isawaitable(result):
            return {
                "status": "error",
                "message": (
                    "memory.add returned an awaitable; use await aadd_memory(...) "
                    "for async memory backends."
                ),
            }

        return {
            "status": "ok",
            "added_messages": len(cleaned),
            "result": result,
        }

    async def aadd_memory(
        self,
        messages: Iterable[Dict[str, Any]],
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Async-safe variant for AsyncMemory or sync memory backends.
        """
        add_fn = getattr(self.memory, "add", None)
        if not callable(add_fn):
            return {
                "status": "error",
                "message": "memory backend does not expose add(...)",
            }

        cleaned = self._clean_messages(messages)
        if not cleaned:
            return {
                "status": "skipped",
                "message": "no valid messages to add",
                "added_messages": 0,
            }

        try:
            result = add_fn(cleaned, user_id=user_id)
        except TypeError:
            result = add_fn(messages=cleaned, user_id=user_id)

        if inspect.isawaitable(result):
            result = await result

        return {
            "status": "ok",
            "added_messages": len(cleaned),
            "result": result,
        }

    async def aadd_dialogues(
        self,
        dialogues: Iterable[Dict[str, Any]],
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Batch add dialogues.

        Supported dialogue item format:
        - {"dialogue_index": 0, "dialogue": [{"role":"user","content":"..."}, ...]}
        """
        dialogues_list = list(dialogues)
        ok = 0
        skip = 0
        err = 0
        details: List[Dict[str, Any]] = []

        for item in dialogues_list:
            dialogue_index = None
            payload: Any = item
            if isinstance(item, dict):
                dialogue_index = item.get("dialogue_index")
                payload = item.get("dialogue", [])

            result = await self.aadd_memory(payload, user_id=user_id)
            status = str(result.get("status", ""))
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skip += 1
            else:
                err += 1

            details.append(
                {
                    "dialogue_index": dialogue_index,
                    "status": status,
                    "added_messages": result.get("added_messages", 0),
                    "message": result.get("message", ""),
                }
            )

        return {
            "status": "ok" if err == 0 else "partial",
            "dialogues_total": len(dialogues_list),
            "ok_dialogues": ok,
            "skipped_dialogues": skip,
            "error_dialogues": err,
            "details": details,
        }
