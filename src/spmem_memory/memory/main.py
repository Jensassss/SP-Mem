import asyncio
import concurrent
import gc
import hashlib
import json
import logging
import os
import re
import uuid
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from spmem_memory.memory.privacy_processor import PrivacyProcessor

from spmem_memory.configs.base import MemoryConfig, MemoryItem
from spmem_memory.configs.enums import MemoryType
from spmem_memory.configs.prompts import (
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    get_update_memory_messages,
)
from spmem_memory.exceptions import ValidationError as SPMemValidationError
from spmem_memory.memory.base import MemoryBase
from spmem_memory.memory.setup import spmem_dir, setup_config
from spmem_memory.memory.storage import SQLiteManager
from spmem_memory.memory.telemetry import SPMEM_TELEMETRY, capture_event
from spmem_memory.memory.utils import (
    ensure_json_instruction,
    extract_json,
    get_fact_retrieval_messages,
    normalize_facts,
    parse_messages,
    parse_vision_messages,
    process_telemetry_filters,
    remove_code_blocks,
)
from spmem_memory.utils.factory import (
    EmbedderFactory,
    GraphStoreFactory,
    LlmFactory,
    RerankerFactory,
    VectorStoreFactory,
)

# Suppress SWIG deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

# Initialize logger early for util functions
logger = logging.getLogger(__name__)

# Keep vector and graph retrieval caps independent for ablation experiments.
MEMORY_SEARCH_VECTOR_LIMIT = 3
MEMORY_SEARCH_GRAPH_LIMIT = 5


def _normalize_iso_timestamp_to_utc(timestamp: Optional[str]) -> Optional[str]:
    """Normalize timezone-aware ISO timestamps to UTC without rewriting naive values."""
    if not timestamp:
        return timestamp
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    if parsed.tzinfo is None:
        return timestamp
    return parsed.astimezone(timezone.utc).isoformat()


def _build_filters_and_metadata(
    *,  # Enforce keyword-only arguments
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,  # For query-time filtering
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Constructs metadata for storage and filters for querying based on session and actor identifiers.

    This helper supports multiple session identifiers (`user_id`, `agent_id`, and/or `run_id`)
    for flexible session scoping and optionally narrows queries to a specific `actor_id`. It returns two dicts:

    1. `base_metadata_template`: Used as a template for metadata when storing new memories.
       It includes all provided session identifier(s) and any `input_metadata`.
    2. `effective_query_filters`: Used for querying existing memories. It includes all
       provided session identifier(s), any `input_filters`, and a resolved actor
       identifier for targeted filtering if specified by any actor-related inputs.

    Actor filtering precedence: explicit `actor_id` argument, then `filters["actor_id"]`.
    This resolved actor ID is used for querying but is not added to `base_metadata_template`,
    as the actor for storage is typically derived from message content at a later stage.

    Args:
        user_id (Optional[str]): User identifier, for session scoping.
        agent_id (Optional[str]): Agent identifier, for session scoping.
        run_id (Optional[str]): Run identifier, for session scoping.
        actor_id (Optional[str]): Explicit actor identifier, used as a potential source for
            actor-specific filtering. See actor resolution precedence in the main description.
        input_metadata (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session identifiers for the storage metadata template. Defaults to an empty dict.
        input_filters (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session and actor identifiers for query filters. Defaults to an empty dict.

    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - base_metadata_template (Dict[str, Any]): Metadata template for storing memories,
              scoped to the provided session(s).
            - effective_query_filters (Dict[str, Any]): Filters for querying memories,
              scoped to the provided session(s) and potentially a resolved actor.
    """

    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- add all provided session ids ----------
    session_ids_provided = []

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise SPMemValidationError(
            message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            error_code="VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}},
            suggestion="Please provide at least one identifier to scope the memory operation."
        )

    # ---------- optional actor filter ----------
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


setup_config()
logger = logging.getLogger(__name__)


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.custom_fact_extraction_prompt = self.config.custom_fact_extraction_prompt
        self.custom_update_memory_prompt = self.config.custom_update_memory_prompt
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.privacy_processor = PrivacyProcessor(self.llm)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        
        # Initialize reranker if configured
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider, 
                config.reranker.config
            )

        self.enable_graph = False

        if self.config.graph_store.config:
            provider = self.config.graph_store.provider
            self.graph = GraphStoreFactory.create(provider, self.config)
            self.enable_graph = True
        else:
            self.graph = None
        if SPMEM_TELEMETRY:
            # Create telemetry config manually to avoid deepcopy issues with thread locks
            telemetry_config_dict = {}
            if hasattr(self.config.vector_store.config, 'model_dump'):
                # For pydantic models
                telemetry_config_dict = self.config.vector_store.config.model_dump()
            else:
                # For other objects, manually copy common attributes
                for attr in ['host', 'port', 'path', 'api_key', 'index_name', 'dimension', 'metric']:
                    if hasattr(self.config.vector_store.config, attr):
                        telemetry_config_dict[attr] = getattr(self.config.vector_store.config, attr)

            # Override collection name for telemetry
            telemetry_config_dict['collection_name'] = "spmem_migrations"

            # Set path for file-based vector stores
            if self.config.vector_store.provider in ["faiss", "qdrant"]:
                provider_path = f"migrations_{self.config.vector_store.provider}"
                telemetry_config_dict['path'] = os.path.join(spmem_dir, provider_path)
                os.makedirs(telemetry_config_dict['path'], exist_ok=True)

            # Create the config object using the same class as the original
            telemetry_config = self.config.vector_store.config.__class__(**telemetry_config_dict)
            self._telemetry_vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, telemetry_config
            )
        capture_event("spmem_memory.init", self, {"sync_type": "sync"})

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False
        
        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.
            
        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None
        
        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)
        
        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages
    
    def _sanitize_fact_object(self, fact_obj, filters):
        text = fact_obj.get("text", "")
        privacy_entities = fact_obj.get("privacy_entities", [])

        sanitized_text = text
        privacy_items = []

        for ent in privacy_entities:
            raw_value = ent.get("raw_value", "")
            privacy_type = ent.get("privacy_type", "")

            if not raw_value or not privacy_type:
                continue

            strategy = self.privacy_processor.get_strategy_by_privacy_type(privacy_type)
            sanitized_value = self.privacy_processor.sanitize_value(
                raw_value=raw_value,
                privacy_type=privacy_type,
                mask_strategy=strategy,
                filters=filters,
            )

            raw_value_text = str(raw_value).strip()
            sanitized_value_text = str(sanitized_value)
            if not raw_value_text:
                continue

            if raw_value_text in sanitized_text:
                sanitized_text = sanitized_text.replace(raw_value_text, sanitized_value_text)
            else:
                # Fallback for casing/spacing drift between extracted span and fact text.
                ci_pattern = re.compile(re.escape(raw_value_text), flags=re.IGNORECASE)
                sanitized_text, replace_count = ci_pattern.subn(sanitized_value_text, sanitized_text)
                if replace_count == 0 and " " in raw_value_text:
                    ws_pattern = re.compile(
                        r"\s+".join(re.escape(part) for part in raw_value_text.split()),
                        flags=re.IGNORECASE,
                    )
                    sanitized_text, replace_count = ws_pattern.subn(sanitized_value_text, sanitized_text)

                if replace_count == 0:
                    logger.debug(
                        "Privacy replacement miss for privacy_type=%s raw_value=%r fact_text=%r",
                        privacy_type,
                        raw_value_text,
                        text,
                    )

            privacy_items.append({
                "raw_value": raw_value,
                "privacy_type": privacy_type,
                "mask_strategy": strategy,
                "sanitized_value": sanitized_value,
            })

        return {
            "raw_text": text,
            "sanitized_text": sanitized_text,
            "is_privacy": len(privacy_items) > 0,
            "privacy_items": privacy_items,
        }
    
    def _persist_privacy_mappings(self, memory_id: str, metadata: dict, privacy_items: list[dict]):
        user_id = metadata.get("user_id")
        if not user_id:
            return

        for item in privacy_items:
            self.db.upsert_privacy_mapping(
                user_id=user_id,
                privacy_type=item["privacy_type"],
                raw_value=item["raw_value"],
                sanitized_value=item["sanitized_value"],
                memory_id=memory_id,
            )

    def _build_public_privacy_entities(self, privacy_items: list[dict]) -> list[dict]:
        public_entities = []

        for item in privacy_items:
            raw_value = str(item.get("raw_value", ""))
            privacy_type = item.get("privacy_type", "")
            sanitized_value = item.get("sanitized_value", "")

            if not raw_value or not privacy_type:
                continue

            public_entities.append(
                {
                    "privacy_type": privacy_type,
                    "sanitized_value": sanitized_value,
                    "entity_hash": hashlib.md5(raw_value.encode()).hexdigest(),
                }
            )

        return public_entities

    @staticmethod
    def _as_privacy_record_list(candidate: Any) -> List[Dict[str, Any]]:
        """Normalize a privacy-meta candidate into a list of record dicts."""
        if isinstance(candidate, dict):
            return [candidate]
        if isinstance(candidate, list):
            return [x for x in candidate if isinstance(x, dict)]
        return []

    def _merge_privacy_meta_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge one or more privacy-meta records into a single effective metadata object."""
        if not records:
            return {}

        merged_items: List[Dict[str, Any]] = []
        seen = set()

        for record in records:
            for item in record.get("privacy_items", []) or []:
                if not isinstance(item, dict):
                    continue
                raw_value = str(item.get("raw_value", ""))
                privacy_type = str(item.get("privacy_type", ""))
                key = (privacy_type, raw_value)
                if key in seen:
                    continue
                seen.add(key)
                merged_items.append(item)

        representative = records[0]
        return {
            "raw_text": representative.get("raw_text", ""),
            "sanitized_text": representative.get("sanitized_text", ""),
            "is_privacy": len(merged_items) > 0,
            "privacy_items": merged_items,
        }

    def _get_privacy_meta_from_fact_ids(
        self,
        source_fact_ids: Any,
        fact_privacy_meta_by_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Resolve privacy metadata directly from explicit fact ids returned by the update planner."""
        if not source_fact_ids or not fact_privacy_meta_by_id:
            return {}

        if isinstance(source_fact_ids, str):
            source_fact_ids = [source_fact_ids]
        elif not isinstance(source_fact_ids, list):
            return {}

        matched_records: List[Dict[str, Any]] = []
        for fact_id in source_fact_ids:
            if not isinstance(fact_id, str):
                continue
            record = fact_privacy_meta_by_id.get(fact_id.strip())
            if isinstance(record, dict):
                matched_records.append(record)

        return self._merge_privacy_meta_records(matched_records)

    def _get_privacy_meta_for_action(self, action_text: str, fact_privacy_meta: dict) -> dict:
        """Resolve privacy metadata for an action text with tolerant matching."""
        if not isinstance(action_text, str) or not fact_privacy_meta:
            return {}

        # Fast path: exact text match.
        if action_text in fact_privacy_meta:
            return self._merge_privacy_meta_records(
                self._as_privacy_record_list(fact_privacy_meta[action_text])
            )

        # Fallback: normalize casing/whitespace to reduce accidental misses.
        normalized_action = re.sub(r"\s+", " ", action_text).strip().lower()
        if not normalized_action:
            return {}

        normalized_matches: List[Dict[str, Any]] = []
        for fact_text, meta in fact_privacy_meta.items():
            if not isinstance(fact_text, str):
                continue
            normalized_fact = re.sub(r"\s+", " ", fact_text).strip().lower()
            if normalized_fact == normalized_action:
                normalized_matches.extend(self._as_privacy_record_list(meta))

        if normalized_matches:
            return self._merge_privacy_meta_records(normalized_matches)

        # Secondary fallback: if action text still contains sanitized placeholders,
        # match by the sanitized_value token(s) from privacy items.
        token_matches: List[Dict[str, Any]] = []
        for meta in fact_privacy_meta.values():
            for record in self._as_privacy_record_list(meta):
                privacy_items = record.get("privacy_items", []) or []
                if not isinstance(privacy_items, list):
                    continue
                for item in privacy_items:
                    if not isinstance(item, dict):
                        continue
                    sanitized_value = str(item.get("sanitized_value", "")).strip().lower()
                    if sanitized_value and sanitized_value in normalized_action:
                        token_matches.append(record)
                        break

        if token_matches:
            return self._merge_privacy_meta_records(token_matches)

        return {}

    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Create a new memory.

        Adds new memories scoped to a single session id (e.g. `user_id`, `agent_id`, or `run_id`). One of those ids is required.

        Args:
            messages (str or List[Dict[str, str]]): The message content or list of messages
                (e.g., `[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]`)
                to be processed and stored.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): If True (default), an LLM is used to extract key facts from
                'messages' and decide whether to add, update, or delete related memories.
                If False, 'messages' are added as raw memories directly.
            memory_type (str, optional): Specifies the type of memory. Currently, only
                `MemoryType.PROCEDURAL.value` ("procedural_memory") is explicitly handled for
                creating procedural memories (typically requires 'agent_id'). Otherwise, memories
                are treated as general conversational/factual memories.memory_type (str, optional): Type of memory to create. Defaults to None. By default, it creates the short term memories and long term (semantic and episodic) memories. Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.


        Returns:
            dict: A dictionary containing the result of the memory addition operation, typically
                  including a list of memory items affected (added, updated) under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`

        Raises:
            SPMemValidationError: If input validation fails (invalid memory_type, messages format, etc.).
            VectorStoreError: If vector store operations fail.
            GraphStoreError: If graph store operations fail.
            EmbeddingError: If embedding generation fails.
            LLMError: If LLM operations fail.
            DatabaseError: If database operations fail.
        """

        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise SPMemValidationError(
                message=f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories.",
                error_code="VALIDATION_002",
                details={"provided_type": memory_type, "valid_type": MemoryType.PROCEDURAL.value},
                suggestion=f"Use '{MemoryType.PROCEDURAL.value}' to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise SPMemValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(self._add_to_vector_store, messages, processed_metadata, effective_filters, infer)
            future2 = executor.submit(self._add_to_graph, messages, effective_filters)

            concurrent.futures.wait([future1, future2])

            vector_store_result = future1.result()
            graph_result = future2.result()

        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}

    def _add_to_vector_store(self, messages, metadata, filters, infer):
        print(
            "[VECTOR] start "
            f"infer={infer} "
            f"user_id={filters.get('user_id')} "
            f"agent_id={filters.get('agent_id')} "
            f"run_id={filters.get('run_id')}"
        )

        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format: {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings = self.embedding_model.embed(msg_content, "add")
                # Pass embeddings as a dict so _create_memory can reuse the cached embedding
                mem_id = self._create_memory(msg_content, {msg_content: msg_embeddings}, per_msg_meta)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        parsed_messages = parse_messages(messages)
       
        if self.config.custom_fact_extraction_prompt:
            system_prompt = self.config.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        
        else:
            # Determine if this should use agent memory extraction based on agent_id presence
            # and role types in messages
            is_agent_memory = False

            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages, is_agent_memory)

        # Ensure 'json' appears in prompts for json_object response format compatibility
        system_prompt, user_prompt = ensure_json_instruction(system_prompt, user_prompt)

        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        try:
            cleaned_response = remove_code_blocks(response)
            if not cleaned_response.strip():
                fact_objects = []
            else:
                try:
                    fact_objects = json.loads(cleaned_response, strict=False)["facts"]
                except json.JSONDecodeError:
                    extracted_json = extract_json(response)
                    fact_objects = json.loads(extracted_json, strict=False)["facts"]
        except Exception as e:
            logger.error(f"Error in fact_objects: {e}")
            fact_objects = []
        print(json.dumps({"facts": fact_objects}, ensure_ascii=False, indent=2))

        sanitized_fact_records = []
        new_retrieved_facts = []
        fact_privacy_meta = {}
        fact_privacy_meta_by_id = {}

        for idx, fact_obj in enumerate(fact_objects):
            sanitized_record = self._sanitize_fact_object(fact_obj, filters)
            fact_id = f"fact_{idx}"
            sanitized_record["fact_id"] = fact_id
            sanitized_fact_records.append(sanitized_record)

            sanitized_text = sanitized_record["sanitized_text"]
            if not sanitized_text:
                continue

            new_retrieved_facts.append(sanitized_text)
            fact_privacy_meta_by_id[fact_id] = sanitized_record
            fact_privacy_meta.setdefault(sanitized_text, []).append(sanitized_record)

        print(
            "[VECTOR] facts "
            f"extracted={len(fact_objects)} "
            f"sanitized_records={len(sanitized_fact_records)} "
            f"new_retrieved_facts={len(new_retrieved_facts)}"
        )

        if not new_retrieved_facts:
            logger.debug("No new facts retrieved from input. Skipping memory update LLM call.")

        retrieved_old_memory = []
        new_message_embeddings = {}
        # Search for existing memories using the provided session identifiers
        # Use all available session identifiers for accurate memory retrieval
        search_filters = {}
        if filters.get("user_id"):
            search_filters["user_id"] = filters["user_id"]
        if filters.get("agent_id"):
            search_filters["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            search_filters["run_id"] = filters["run_id"]
        for new_mem in new_retrieved_facts:
            messages_embeddings = self.embedding_model.embed(new_mem, "add")
            new_message_embeddings[new_mem] = messages_embeddings
            existing_memories = self.vector_store.search(
                query=new_mem,
                vectors=messages_embeddings,
                limit=5,
                filters=search_filters,
            )
            for mem in existing_memories:
                retrieved_old_memory.append({"id": mem.id, "text": mem.payload.get("data", "")})

        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logger.info(f"Total existing memories: {len(retrieved_old_memory)}")

        # mapping UUIDs with integers for handling UUID hallucinations
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        if new_retrieved_facts:
            facts_for_planner = [
                {"fact_id": rec.get("fact_id"), "text": rec.get("sanitized_text")}
                for rec in sanitized_fact_records
                if rec.get("sanitized_text")
            ]
            function_calling_prompt = get_update_memory_messages(
                retrieved_old_memory,
                json.dumps(facts_for_planner, ensure_ascii=False, indent=2),
                self.config.custom_update_memory_prompt,
            )

            try:
                response: str = self.llm.generate_response(
                    messages=[{"role": "user", "content": function_calling_prompt}],
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logger.error(f"Error in new memory actions response: {e}")
                response = ""

            try:
                if not response or not response.strip():
                    logger.warning("Empty response from LLM, no memories to extract")
                    new_memories_with_actions = {}
                else:
                    try:
                        new_memories_with_actions = json.loads(remove_code_blocks(response), strict=False)
                    except json.JSONDecodeError:
                        extracted_json = extract_json(response)
                        new_memories_with_actions = json.loads(extracted_json, strict=False)
            except Exception as e:
                logger.error(f"Invalid JSON response: {e}")
                new_memories_with_actions = {}
        else:
            new_memories_with_actions = {}

        returned_memories = []
        try:
            for resp in new_memories_with_actions.get("memory", []):
                logger.info(resp)
                try:
                    action_text = resp.get("text")
                    if not action_text:
                        logger.info("Skipping memory entry because of empty `text` field.")
                        continue
                    
                    '''
                    ADD:
                        data
                        hash
                        created_at
                        updated_at
                        is_privacy
                        privacy_types
                        sanitized
                        raw_text
                    '''
                    event_type = resp.get("event")
                    if event_type == "ADD":
                        if action_text not in new_message_embeddings:
                            new_message_embeddings[action_text] = self.embedding_model.embed(action_text, "add")

                        per_memory_metadata = deepcopy(metadata)
                        privacy_meta = self._get_privacy_meta_from_fact_ids(
                            resp.get("source_fact_ids"),
                            fact_privacy_meta_by_id,
                        )
                        if not privacy_meta:
                            privacy_meta = self._get_privacy_meta_for_action(action_text, fact_privacy_meta)
                        privacy_items = privacy_meta.get("privacy_items", [])

                        per_memory_metadata["is_privacy"] = privacy_meta.get("is_privacy", False)
                        per_memory_metadata["privacy_types"] = [
                            item["privacy_type"] for item in privacy_items
                        ]
                        per_memory_metadata["sanitized"] = True
                        per_memory_metadata["privacy_entities"] = self._build_public_privacy_entities(privacy_items)
                        memory_id = self._create_memory(
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=per_memory_metadata,
                        )

                        if privacy_meta.get("privacy_items"):
                            self._persist_privacy_mappings(
                                memory_id=memory_id,
                                metadata=per_memory_metadata,
                                privacy_items=privacy_meta["privacy_items"],
                            )

                        returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})

                    elif event_type == "UPDATE":
                        if action_text not in new_message_embeddings:
                            new_message_embeddings[action_text] = self.embedding_model.embed(action_text, "update")

                        per_memory_metadata = deepcopy(metadata)
                        privacy_meta = self._get_privacy_meta_from_fact_ids(
                            resp.get("source_fact_ids"),
                            fact_privacy_meta_by_id,
                        )
                        if not privacy_meta:
                            privacy_meta = self._get_privacy_meta_for_action(action_text, fact_privacy_meta)
                        privacy_items = privacy_meta.get("privacy_items", [])
                        target_memory_id = temp_uuid_mapping[resp.get("id")]

                        existing_payload = {}
                        existing_memory = self.vector_store.get(vector_id=target_memory_id)
                        if existing_memory is not None:
                            existing_payload = existing_memory.payload or {}

                        if privacy_items:
                            per_memory_metadata["is_privacy"] = privacy_meta.get("is_privacy", False)
                            per_memory_metadata["privacy_types"] = [item["privacy_type"] for item in privacy_items]
                            per_memory_metadata["sanitized"] = True
                            per_memory_metadata["privacy_entities"] = self._build_public_privacy_entities(privacy_items)
                        else:
                            # If UPDATE text is produced without privacy annotations, preserve previous privacy state.
                            per_memory_metadata["is_privacy"] = existing_payload.get("is_privacy", False)
                            per_memory_metadata["privacy_types"] = existing_payload.get("privacy_types", [])
                            per_memory_metadata["sanitized"] = existing_payload.get("sanitized", True)
                            per_memory_metadata["privacy_entities"] = existing_payload.get("privacy_entities", [])

                        self._update_memory(
                            memory_id=target_memory_id,
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=per_memory_metadata,
                        )

                        # Only rewrite mapping links when we have fresh privacy items from this update action.
                        if privacy_items:
                            # Clear stale mappings from the previous memory content, then persist current ones.
                            self.db.delete_privacy_mappings_by_memory(target_memory_id)
                            self._persist_privacy_mappings(
                                memory_id=target_memory_id,
                                metadata=per_memory_metadata,
                                privacy_items=privacy_items,
                            )
                        else:
                            logger.debug(
                                "No privacy_items for UPDATE memory_id=%s; preserving existing privacy mappings.",
                                target_memory_id,
                            )

                        returned_memories.append(
                            {
                                "id": target_memory_id,
                                "memory": action_text,
                                "event": event_type,
                                "previous_memory": resp.get("old_memory"),
                            }
                        )

                    elif event_type == "DELETE":
                        target_memory_id = temp_uuid_mapping[resp.get("id")]
                        self._delete_memory(memory_id=target_memory_id)
                        self.db.delete_privacy_mappings_by_memory(target_memory_id)

                        returned_memories.append(
                            {
                                "id": target_memory_id,
                                "memory": action_text,
                                "event": event_type,
                            }
                        )
                    elif event_type == "NONE":
                        # Even if content doesn't need updating, update session IDs if provided
                        memory_id = temp_uuid_mapping.get(resp.get("id"))
                        if memory_id and (metadata.get("agent_id") or metadata.get("run_id")):
                            # Update only the session identifiers, keep content the same
                            existing_memory = self.vector_store.get(vector_id=memory_id)
                            if existing_memory is None:
                                logger.warning(f"Memory {memory_id} not found for session ID update, skipping")
                                continue
                            updated_metadata = deepcopy(existing_memory.payload)
                            if metadata.get("agent_id"):
                                updated_metadata["agent_id"] = metadata["agent_id"]
                            if metadata.get("run_id"):
                                updated_metadata["run_id"] = metadata["run_id"]
                            updated_metadata["created_at"] = _normalize_iso_timestamp_to_utc(
                                updated_metadata.get("created_at")
                            )
                            updated_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

                            self.vector_store.update(
                                vector_id=memory_id,
                                vector=None,  # Keep same embeddings
                                payload=updated_metadata,
                            )
                            logger.info(f"Updated session IDs for memory {memory_id}")
                        else:
                            logger.info("NOOP for Memory.")
                except Exception as e:
                    logger.error(f"Error processing memory action: {resp}, Error: {e}")
        except Exception as e:
            logger.error(f"Error iterating new_memories_with_actions: {e}")

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event(
            "spmem_memory.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"},
        )
        return returned_memories

    def _add_to_graph(self, messages, filters):
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = self.graph.add(data, filters)

        return added_entities

    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("spmem_memory.get", self, {"memory_id": memory_id, "sync_type": "sync"})
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=_normalize_iso_timestamp_to_utc(memory.payload.get("created_at")),
            updated_at=_normalize_iso_timestamp_to_utc(memory.payload.get("updated_at")),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ):
        """
        List all memories.

        Args:
            user_id (str, optional): user id
            agent_id (str, optional): agent id
            run_id (str, optional): run id
            filters (dict, optional): Additional custom key-value filters to apply to the search.
                These are merged with the ID-based scoping filters. For example,
                `filters={"actor_id": "some_user"}`.
            limit (int, optional): The maximum number of memories to return. Defaults to 100.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key,
                  and potentially "relations" if graph store is enabled. For API v1.0,
                  it might return a direct list (see deprecation warning).
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "spmem_memory.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"}
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._get_all_from_vector_store, effective_filters, limit)
            future_graph_entities = (
                executor.submit(self.graph.get_all, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            all_memories_result = future_memories.result()
            graph_entities_result = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": all_memories_result, "relations": graph_entities_result}

        return {"results": all_memories_result}

    def _get_all_from_vector_store(self, filters, limit):
        memories_result = self.vector_store.list(filters=filters, limit=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=_normalize_iso_timestamp_to_utc(mem.payload.get("created_at")),
                updated_at=_normalize_iso_timestamp_to_utc(mem.payload.get("updated_at")),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        rerank: bool = True,
    ):
        """
        Searches for memories based on a query
        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Legacy filters to apply to the search. Defaults to None.
            threshold (float, optional): Minimum score for a memory to be included in the results. Defaults to None.
            filters (dict, optional): Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals  
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT

        Returns:
            dict: A dictionary containing the search results, typically under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`
        """
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

        # Apply enhanced metadata filtering if advanced operators are detected
        if filters and self._has_advanced_operators(filters):
            processed_filters = self._process_metadata_filters(filters)
            effective_filters.update(processed_filters)
        elif filters:
            # Simple filters, merge directly
            effective_filters.update(filters)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "spmem_memory.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "sync",
                "threshold": threshold,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(
                self._search_vector_store,
                query,
                effective_filters,
                MEMORY_SEARCH_VECTOR_LIMIT,
                threshold,
            )
            '''
            future_graph_entities = (
                executor.submit(self.graph.search, query, effective_filters, MEMORY_SEARCH_GRAPH_LIMIT)
                if self.enable_graph
                else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )
            '''
            original_memories = future_memories.result()
            # graph_entities = future_graph_entities.result() if future_graph_entities else None

        # Apply reranking if enabled and reranker is available
        if rerank and self.reranker and original_memories:
            try:
                reranked_memories = self.reranker.rerank(query, original_memories, limit)
                original_memories = reranked_memories
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")
        '''
        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities}
        '''
        return {"results": original_memories}

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.
        
        Args:
            metadata_filters: Enhanced metadata filters with operators
            
        Returns:
            Dict of processed filters compatible with vector store
        """
        processed_filters = {}
        
        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}
            
            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte", 
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }
                
                if operator in operator_map:
                    result.setdefault(key, {})[operator_map[operator]] = value
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        processed_filters.update(process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        or_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        not_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                processed_filters.update(process_condition(key, value))
        
        return processed_filters

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.
        
        Args:
            filters: Dictionary of filters to check
            
        Returns:
            bool: True if advanced operators are detected
        """
        if not isinstance(filters, dict):
            return False
            
        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    def _search_vector_store(self, query, filters, limit, threshold: Optional[float] = None):
        embeddings = self.embedding_model.embed(query, "search")
        memories = self.vector_store.search(query=query, vectors=embeddings, limit=limit, filters=filters)

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        original_memories = []
        for mem in memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=_normalize_iso_timestamp_to_utc(mem.payload.get("created_at")),
                updated_at=_normalize_iso_timestamp_to_utc(mem.payload.get("updated_at")),
                score=mem.score,
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            if threshold is None or mem.score >= threshold:
                original_memories.append(memory_item_dict)

        return original_memories

    def update(self, memory_id, data, metadata: Optional[Dict[str, Any]] = None):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.
            metadata (dict, optional): Additional metadata to update. Existing metadata fields
                not specified here will be preserved. Defaults to None.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
            >>> m.update(memory_id="mem_123", data="Likes tennis", metadata={"category": "sports"})
            {'message': 'Memory updated successfully!'}
        """
        capture_event("spmem_memory.update", self, {"memory_id": memory_id, "sync_type": "sync"})

        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        self._update_memory(memory_id, data, existing_embeddings, metadata)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("spmem_memory.delete", self, {"memory_id": memory_id, "sync_type": "sync"})

        existing_memory = self.vector_store.get(vector_id=memory_id)
        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found")

        # Clean up graph entities before deleting from vector store
        if self.enable_graph:
            try:
                memory_text = existing_memory.payload.get("data", "")
                if memory_text:
                    filters = {}
                    for key in ("user_id", "agent_id", "run_id"):
                        val = existing_memory.payload.get(key)
                        if val:
                            filters[key] = val
                    if filters.get("user_id"):
                        self.graph.delete(memory_text, filters)
            except Exception as e:
                logger.error(f"Error cleaning up graph for memory {memory_id}: {e}")

        self._delete_memory(memory_id, existing_memory)
        self.db.delete_privacy_mappings_by_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("spmem_memory.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"})
        # delete matching vector memories individually (do NOT reset the collection)
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)

        logger.info(f"Deleted {len(memories)} memories")

        if self.enable_graph:
            self.graph.delete_all(filters)

        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("spmem_memory.history", self, {"memory_id": memory_id, "sync_type": "sync"})
        return self.db.get_history(memory_id)

    def _create_memory(self, data: str, existing_embeddings: Union[Dict[str, List[float]], List[float]], metadata=None):
        logger.debug(f"Creating memory with {data=}")
        # existing_embeddings may be a dict (preferred) or a precomputed vector
        if isinstance(existing_embeddings, dict) and data in existing_embeddings:
            embeddings = existing_embeddings[data]
        elif not isinstance(existing_embeddings, dict):
            embeddings = existing_embeddings
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        if "created_at" not in new_metadata:
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        new_metadata["updated_at"] = new_metadata["created_at"]

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[new_metadata],
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"),
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
            procedural_memory = remove_code_blocks(procedural_memory)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        new_metadata = deepcopy(metadata)
        new_metadata["memory_type"] = MemoryType.PROCEDURAL.value
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=new_metadata)
        capture_event("spmem_memory._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "sync"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    def _update_memory(self, memory_id, data: str, existing_embeddings: Union[Dict[str, List[float]], List[float]], metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        if "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" not in new_metadata and "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        if isinstance(existing_embeddings, dict) and data in existing_embeddings:
            embeddings = existing_embeddings[data]
        elif not isinstance(existing_embeddings, dict):
            embeddings = existing_embeddings
        else:
            embeddings = self.embedding_model.embed(data, "update")

        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        return memory_id

    def _delete_memory(self, memory_id, existing_memory=None):
        logger.info(f"Deleting memory with {memory_id=}")
        if existing_memory is None:
            existing_memory = self.vector_store.get(vector_id=memory_id)
            if existing_memory is None:
                raise ValueError(f"Memory with id {memory_id} not found")
        prev_value = existing_memory.payload.get("data", "")

        # Preserve original created_at and record deletion time
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        updated_at = datetime.now(timezone.utc).isoformat()

        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            created_at=created_at,
            updated_at=updated_at,
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )
        return memory_id

    def reset(self):
        """
        Reset the memory store by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")

        if hasattr(self.db, "connection") and self.db.connection:
            self.db.connection.execute("DROP TABLE IF EXISTS history")
            self.db.connection.execute("DROP TABLE IF EXISTS privacy_mappings")
            self.db.connection.execute("DROP TABLE IF EXISTS privacy_mapping_links")

        self.db = SQLiteManager(self.config.history_db_path)

        if hasattr(self.vector_store, "reset"):
            self.vector_store = VectorStoreFactory.reset(self.vector_store)
        else:
            logger.warning("Vector store does not support reset. Skipping.")
            self.vector_store.delete_col()
            self.vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, self.config.vector_store.config
            )
        capture_event("spmem_memory.reset", self, {"sync_type": "sync"})

        if self.enable_graph:
            try:
                self.graph.reset()
            except Exception:
                logger.warning("Failed to reset graph store, continuing with reset")

    def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")


class AsyncMemory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.privacy_processor = PrivacyProcessor(self.llm)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        
        # Initialize reranker if configured
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider, 
                config.reranker.config
            )

        self.enable_graph = False

        if self.config.graph_store.config:
            provider = self.config.graph_store.provider
            self.graph = GraphStoreFactory.create(provider, self.config)
            self.enable_graph = True
        else:
            self.graph = None

        if SPMEM_TELEMETRY:
            # Create telemetry config manually to avoid deepcopy issues with thread locks
            telemetry_config_dict = {}
            if hasattr(self.config.vector_store.config, 'model_dump'):
                # For pydantic models
                telemetry_config_dict = self.config.vector_store.config.model_dump()
            else:
                # For other objects, manually copy common attributes
                for attr in ['host', 'port', 'path', 'api_key', 'index_name', 'dimension', 'metric']:
                    if hasattr(self.config.vector_store.config, attr):
                        telemetry_config_dict[attr] = getattr(self.config.vector_store.config, attr)

            # Override collection name for telemetry
            telemetry_config_dict['collection_name'] = "spmem_migrations"

            if self.config.vector_store.provider in ["faiss", "qdrant"]:
                provider_path = f"migrations_{self.config.vector_store.provider}"
                telemetry_config_dict['path'] = os.path.join(spmem_dir, provider_path)
                os.makedirs(telemetry_config_dict['path'], exist_ok=True)

            # Create the config object using the same class as the original
            telemetry_config = self.config.vector_store.config.__class__(**telemetry_config_dict)
            self._telemetry_vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, telemetry_config
            )
        capture_event("spmem_memory.init", self, {"sync_type": "async"})

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False
        
        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.
            
        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None
        
        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)
        
        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    # Keep async privacy behavior aligned with sync Memory implementation.
    def _sanitize_fact_object(self, fact_obj, filters):
        return Memory._sanitize_fact_object(self, fact_obj, filters)

    def _persist_privacy_mappings(self, memory_id: str, metadata: dict, privacy_items: list[dict]):
        return Memory._persist_privacy_mappings(self, memory_id, metadata, privacy_items)

    def _build_public_privacy_entities(self, privacy_items: list[dict]) -> list[dict]:
        return Memory._build_public_privacy_entities(self, privacy_items)

    @staticmethod
    def _as_privacy_record_list(candidate: Any) -> List[Dict[str, Any]]:
        return Memory._as_privacy_record_list(candidate)

    def _merge_privacy_meta_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return Memory._merge_privacy_meta_records(self, records)

    def _get_privacy_meta_from_fact_ids(
        self,
        source_fact_ids: Any,
        fact_privacy_meta_by_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        return Memory._get_privacy_meta_from_fact_ids(self, source_fact_ids, fact_privacy_meta_by_id)

    def _get_privacy_meta_for_action(self, action_text: str, fact_privacy_meta: dict) -> dict:
        return Memory._get_privacy_meta_for_action(self, action_text, fact_privacy_meta)

    async def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        llm=None,
    ):
        """
        Create a new memory asynchronously.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            user_id (str, optional): ID of the user creating the memory.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): Whether to infer the memories. Defaults to True.
            memory_type (str, optional): Type of memory to create. Defaults to None.
                                         Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.
            llm (BaseChatModel, optional): LLM class to use for generating procedural memories. Defaults to None. Useful when user is using LangChain ChatModel.
        Returns:
            dict: A dictionary containing the result of the memory addition operation.
        """
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_metadata=metadata
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise SPMemValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = await self._create_procedural_memory(
                messages, metadata=processed_metadata, prompt=prompt, llm=llm
            )
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        vector_store_task = asyncio.create_task(
            self._add_to_vector_store(messages, processed_metadata, effective_filters, infer)
        )
        graph_task = asyncio.create_task(self._add_to_graph(messages, effective_filters))

        vector_store_result, graph_result = await asyncio.gather(vector_store_task, graph_task)

        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}

    async def _add_to_vector_store(
        self,
        messages: list,
        metadata: dict,
        effective_filters: dict,
        infer: bool,
    ):
        print(
            "[VECTOR] start "
            f"infer={infer} "
            f"user_id={effective_filters.get('user_id')} "
            f"agent_id={effective_filters.get('agent_id')} "
            f"run_id={effective_filters.get('run_id')}"
        )

        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format (async): {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings = await asyncio.to_thread(self.embedding_model.embed, msg_content, "add")
                # Pass embeddings as a dict so _create_memory can reuse the cached embedding
                mem_id = await self._create_memory(msg_content, {msg_content: msg_embeddings}, per_msg_meta)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        parsed_messages = parse_messages(messages)
        if self.config.custom_fact_extraction_prompt:
            system_prompt = self.config.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            # Keep parity with sync path for now.
            is_agent_memory = False
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages, is_agent_memory)

        # Ensure 'json' appears in prompts for json_object response format compatibility
        system_prompt, user_prompt = ensure_json_instruction(system_prompt, user_prompt)

        response = await asyncio.to_thread(
            self.llm.generate_response,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
        )
        try:
            cleaned_response = remove_code_blocks(response)
            if not cleaned_response.strip():
                fact_objects = []
            else:
                try:
                    fact_objects = json.loads(cleaned_response, strict=False)["facts"]
                except json.JSONDecodeError:
                    extracted_json = extract_json(response)
                    fact_objects = json.loads(extracted_json, strict=False)["facts"]
        except Exception as e:
            logger.error(f"Error in fact_objects (async): {e}")
            fact_objects = []
        print(json.dumps({"facts": fact_objects}, ensure_ascii=False, indent=2))

        sanitized_fact_records = []
        new_retrieved_facts = []
        fact_privacy_meta = {}
        fact_privacy_meta_by_id = {}

        for idx, fact_obj in enumerate(fact_objects):
            sanitized_record = self._sanitize_fact_object(fact_obj, effective_filters)
            fact_id = f"fact_{idx}"
            sanitized_record["fact_id"] = fact_id
            sanitized_fact_records.append(sanitized_record)

            sanitized_text = sanitized_record["sanitized_text"]
            if not sanitized_text:
                continue

            new_retrieved_facts.append(sanitized_text)
            fact_privacy_meta_by_id[fact_id] = sanitized_record
            fact_privacy_meta.setdefault(sanitized_text, []).append(sanitized_record)

        print(
            "[VECTOR] facts "
            f"extracted={len(fact_objects)} "
            f"sanitized_records={len(sanitized_fact_records)} "
            f"new_retrieved_facts={len(new_retrieved_facts)}"
        )

        if not new_retrieved_facts:
            logger.debug("No new facts retrieved from input. Skipping memory update LLM call.")

        retrieved_old_memory = []
        new_message_embeddings = {}
        search_filters = {}
        if effective_filters.get("user_id"):
            search_filters["user_id"] = effective_filters["user_id"]
        if effective_filters.get("agent_id"):
            search_filters["agent_id"] = effective_filters["agent_id"]
        if effective_filters.get("run_id"):
            search_filters["run_id"] = effective_filters["run_id"]

        async def process_fact_for_search(new_mem_content):
            embeddings = await asyncio.to_thread(self.embedding_model.embed, new_mem_content, "add")
            new_message_embeddings[new_mem_content] = embeddings
            existing_mems = await asyncio.to_thread(
                self.vector_store.search,
                query=new_mem_content,
                vectors=embeddings,
                limit=5,
                filters=search_filters,
            )
            return [{"id": mem.id, "text": mem.payload.get("data", "")} for mem in existing_mems]

        search_tasks = [process_fact_for_search(fact) for fact in new_retrieved_facts]
        search_results_list = await asyncio.gather(*search_tasks)
        for result_group in search_results_list:
            retrieved_old_memory.extend(result_group)

        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logger.info(f"Total existing memories: {len(retrieved_old_memory)}")
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        if new_retrieved_facts:
            facts_for_planner = [
                {"fact_id": rec.get("fact_id"), "text": rec.get("sanitized_text")}
                for rec in sanitized_fact_records
                if rec.get("sanitized_text")
            ]
            function_calling_prompt = get_update_memory_messages(
                retrieved_old_memory,
                json.dumps(facts_for_planner, ensure_ascii=False, indent=2),
                self.config.custom_update_memory_prompt,
            )
            try:
                response = await asyncio.to_thread(
                    self.llm.generate_response,
                    messages=[{"role": "user", "content": function_calling_prompt}],
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logger.error(f"Error in new memory actions response: {e}")
                response = ""
            try:
                if not response or not response.strip():
                    logger.warning("Empty response from LLM, no memories to extract")
                    new_memories_with_actions = {}
                else:
                    try:
                        new_memories_with_actions = json.loads(remove_code_blocks(response), strict=False)
                    except json.JSONDecodeError:
                        extracted_json = extract_json(response)
                        new_memories_with_actions = json.loads(extracted_json, strict=False)
            except Exception as e:
                logger.error(f"Invalid JSON response: {e}")
                new_memories_with_actions = {}
        else:
            new_memories_with_actions = {}

        returned_memories = []
        try:
            for resp in new_memories_with_actions.get("memory", []):
                logger.info(resp)
                try:
                    action_text = resp.get("text")
                    if not action_text:
                        logger.info("Skipping memory entry because of empty `text` field.")
                        continue

                    event_type = resp.get("event")

                    if event_type == "ADD":
                        if action_text not in new_message_embeddings:
                            new_message_embeddings[action_text] = await asyncio.to_thread(
                                self.embedding_model.embed, action_text, "add"
                            )

                        per_memory_metadata = deepcopy(metadata)
                        privacy_meta = self._get_privacy_meta_from_fact_ids(
                            resp.get("source_fact_ids"),
                            fact_privacy_meta_by_id,
                        )
                        if not privacy_meta:
                            privacy_meta = self._get_privacy_meta_for_action(action_text, fact_privacy_meta)
                        privacy_items = privacy_meta.get("privacy_items", [])

                        per_memory_metadata["is_privacy"] = privacy_meta.get("is_privacy", False)
                        per_memory_metadata["privacy_types"] = [item["privacy_type"] for item in privacy_items]
                        per_memory_metadata["sanitized"] = True
                        per_memory_metadata["privacy_entities"] = self._build_public_privacy_entities(privacy_items)

                        memory_id = await self._create_memory(
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=per_memory_metadata,
                        )

                        if privacy_meta.get("privacy_items"):
                            await asyncio.to_thread(
                                self._persist_privacy_mappings,
                                memory_id,
                                per_memory_metadata,
                                privacy_meta["privacy_items"],
                            )

                        returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})

                    elif event_type == "UPDATE":
                        if action_text not in new_message_embeddings:
                            new_message_embeddings[action_text] = await asyncio.to_thread(
                                self.embedding_model.embed, action_text, "update"
                            )

                        per_memory_metadata = deepcopy(metadata)
                        privacy_meta = self._get_privacy_meta_from_fact_ids(
                            resp.get("source_fact_ids"),
                            fact_privacy_meta_by_id,
                        )
                        if not privacy_meta:
                            privacy_meta = self._get_privacy_meta_for_action(action_text, fact_privacy_meta)
                        privacy_items = privacy_meta.get("privacy_items", [])
                        target_memory_id = temp_uuid_mapping[resp.get("id")]

                        existing_payload = {}
                        existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=target_memory_id)
                        if existing_memory is not None:
                            existing_payload = existing_memory.payload or {}

                        if privacy_items:
                            per_memory_metadata["is_privacy"] = privacy_meta.get("is_privacy", False)
                            per_memory_metadata["privacy_types"] = [item["privacy_type"] for item in privacy_items]
                            per_memory_metadata["sanitized"] = True
                            per_memory_metadata["privacy_entities"] = self._build_public_privacy_entities(privacy_items)
                        else:
                            # If UPDATE text is produced without privacy annotations, preserve previous privacy state.
                            per_memory_metadata["is_privacy"] = existing_payload.get("is_privacy", False)
                            per_memory_metadata["privacy_types"] = existing_payload.get("privacy_types", [])
                            per_memory_metadata["sanitized"] = existing_payload.get("sanitized", True)
                            per_memory_metadata["privacy_entities"] = existing_payload.get("privacy_entities", [])

                        await self._update_memory(
                            memory_id=target_memory_id,
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=per_memory_metadata,
                        )

                        # Only rewrite mapping links when we have fresh privacy items from this update action.
                        if privacy_items:
                            # Clear stale mappings from the previous memory content, then persist current ones.
                            await asyncio.to_thread(self.db.delete_privacy_mappings_by_memory, target_memory_id)
                            await asyncio.to_thread(
                                self._persist_privacy_mappings,
                                target_memory_id,
                                per_memory_metadata,
                                privacy_items,
                            )
                        else:
                            logger.debug(
                                "No privacy_items for UPDATE memory_id=%s; preserving existing privacy mappings.",
                                target_memory_id,
                            )

                        returned_memories.append(
                            {
                                "id": target_memory_id,
                                "memory": action_text,
                                "event": event_type,
                                "previous_memory": resp.get("old_memory"),
                            }
                        )

                    elif event_type == "DELETE":
                        target_memory_id = temp_uuid_mapping[resp.get("id")]
                        await self._delete_memory(memory_id=target_memory_id)
                        await asyncio.to_thread(self.db.delete_privacy_mappings_by_memory, target_memory_id)

                        returned_memories.append(
                            {
                                "id": target_memory_id,
                                "memory": action_text,
                                "event": event_type,
                            }
                        )
                    elif event_type == "NONE":
                        # Even if content doesn't need updating, update session IDs if provided
                        memory_id = temp_uuid_mapping.get(resp.get("id"))
                        if memory_id and (metadata.get("agent_id") or metadata.get("run_id")):
                            # Update only the session identifiers, keep content the same
                            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
                            if existing_memory is None:
                                logger.warning(f"Memory {memory_id} not found for session ID update, skipping")
                                continue
                            updated_metadata = deepcopy(existing_memory.payload)
                            if metadata.get("agent_id"):
                                updated_metadata["agent_id"] = metadata["agent_id"]
                            if metadata.get("run_id"):
                                updated_metadata["run_id"] = metadata["run_id"]
                            updated_metadata["created_at"] = _normalize_iso_timestamp_to_utc(
                                updated_metadata.get("created_at")
                            )
                            updated_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

                            await asyncio.to_thread(
                                self.vector_store.update,
                                vector_id=memory_id,
                                vector=None,  # Keep same embeddings
                                payload=updated_metadata,
                            )
                            logger.info(f"Updated session IDs for memory {memory_id}")
                        else:
                            logger.info("NOOP for Memory.")
                except Exception as e:
                    logger.error(f"Error processing memory action (async): {resp}, Error: {e}")
        except Exception as e:
            logger.error(f"Error in memory processing loop (async): {e}")

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "spmem_memory.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"},
        )
        return returned_memories

    async def _add_to_graph(self, messages, filters):
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = await asyncio.to_thread(self.graph.add, data, filters)

        return added_entities

    async def get(self, memory_id):
        """
        Retrieve a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("spmem_memory.get", self, {"memory_id": memory_id, "sync_type": "async"})
        memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        if not memory:
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=_normalize_iso_timestamp_to_utc(memory.payload.get("created_at")),
            updated_at=_normalize_iso_timestamp_to_utc(memory.payload.get("updated_at")),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    async def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ):
        """
        List all memories.

         Args:
             user_id (str, optional): user id
             agent_id (str, optional): agent id
             run_id (str, optional): run id
             filters (dict, optional): Additional custom key-value filters to apply to the search.
                 These are merged with the ID-based scoping filters. For example,
                 `filters={"actor_id": "some_user"}`.
             limit (int, optional): The maximum number of memories to return. Defaults to 100.

         Returns:
             dict: A dictionary containing a list of memories under the "results" key,
                   and potentially "relations" if graph store is enabled. For API v1.0,
                   it might return a direct list (see deprecation warning).
                   Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "When 'conversation_id' is not provided (classic mode), "
                "at least one of 'user_id', 'agent_id', or 'run_id' must be specified for get_all."
            )

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "spmem_memory.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"}
        )

        vector_store_task = asyncio.create_task(self._get_all_from_vector_store(effective_filters, limit))

        graph_task = None
        if self.enable_graph:
            graph_get_all = getattr(self.graph, "get_all", None)
            if callable(graph_get_all):
                if asyncio.iscoroutinefunction(graph_get_all):
                    graph_task = asyncio.create_task(graph_get_all(effective_filters, limit))
                else:
                    graph_task = asyncio.create_task(asyncio.to_thread(graph_get_all, effective_filters, limit))

        results_dict = {}
        if graph_task:
            vector_store_result, graph_entities_result = await asyncio.gather(vector_store_task, graph_task)
            results_dict.update({"results": vector_store_result, "relations": graph_entities_result})
        else:
            results_dict.update({"results": await vector_store_task})

        return results_dict

    async def _get_all_from_vector_store(self, filters, limit):
        memories_result = await asyncio.to_thread(self.vector_store.list, filters=filters, limit=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=_normalize_iso_timestamp_to_utc(mem.payload.get("created_at")),
                updated_at=_normalize_iso_timestamp_to_utc(mem.payload.get("updated_at")),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    async def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
    ):
        """
        Searches for memories based on a query
        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Legacy filters to apply to the search. Defaults to None.
            threshold (float, optional): Minimum score for a memory to be included in the results. Defaults to None.
            filters (dict, optional): Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals  
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT

        Returns:
            dict: A dictionary containing the search results, typically under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("at least one of 'user_id', 'agent_id', or 'run_id' must be specified ")

        # Apply enhanced metadata filtering if advanced operators are detected
        if filters and self._has_advanced_operators(filters):
            processed_filters = self._process_metadata_filters(filters)
            effective_filters.update(processed_filters)
        elif filters:
            # Simple filters, merge directly
            effective_filters.update(filters)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "spmem_memory.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "async",
                "threshold": threshold,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )

        vector_store_task = asyncio.create_task(
            self._search_vector_store(query, effective_filters, MEMORY_SEARCH_VECTOR_LIMIT, threshold)
        )

        graph_task = None
        if self.enable_graph:
            if hasattr(self.graph.search, "__await__"):  # Check if graph search is async
                graph_task = asyncio.create_task(self.graph.search(query, effective_filters, MEMORY_SEARCH_GRAPH_LIMIT))
            else:
                graph_task = asyncio.create_task(
                    asyncio.to_thread(self.graph.search, query, effective_filters, MEMORY_SEARCH_GRAPH_LIMIT)
                )

        if graph_task:
            original_memories, graph_entities = await asyncio.gather(vector_store_task, graph_task)
        else:
            original_memories = await vector_store_task
            graph_entities = None

        # Apply reranking if enabled and reranker is available
        if rerank and self.reranker and original_memories:
            try:
                # Run reranking in thread pool to avoid blocking async loop
                reranked_memories = await asyncio.to_thread(
                    self.reranker.rerank, query, original_memories, limit
                )
                original_memories = reranked_memories
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")

        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities}

        return {"results": original_memories}

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        processed_filters = {}

        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}

            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                if operator in operator_map:
                    result.setdefault(key, {})[operator_map[operator]] = value
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        processed_filters.update(process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        or_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        not_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                processed_filters.update(process_condition(key, value))

        return processed_filters

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.

        Args:
            filters: Dictionary of filters to check

        Returns:
            bool: True if advanced operators are detected
        """
        if not isinstance(filters, dict):
            return False

        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    async def _search_vector_store(self, query, filters, limit, threshold: Optional[float] = None):
        embeddings = await asyncio.to_thread(self.embedding_model.embed, query, "search")
        memories = await asyncio.to_thread(
            self.vector_store.search, query=query, vectors=embeddings, limit=limit, filters=filters
        )

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        original_memories = []
        for mem in memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=_normalize_iso_timestamp_to_utc(mem.payload.get("created_at")),
                updated_at=_normalize_iso_timestamp_to_utc(mem.payload.get("updated_at")),
                score=mem.score,
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            if threshold is None or mem.score >= threshold:
                original_memories.append(memory_item_dict)

        return original_memories

    async def update(self, memory_id, data, metadata: Optional[Dict[str, Any]] = None):
        """
        Update a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.
            metadata (dict, optional): Additional metadata to update. Existing metadata fields
                not specified here will be preserved. Defaults to None.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> await m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
            >>> await m.update(memory_id="mem_123", data="Likes tennis", metadata={"category": "sports"})
            {'message': 'Memory updated successfully!'}
        """
        capture_event("spmem_memory.update", self, {"memory_id": memory_id, "sync_type": "async"})

        embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")
        existing_embeddings = {data: embeddings}

        await self._update_memory(memory_id, data, existing_embeddings, metadata)
        return {"message": "Memory updated successfully!"}

    async def delete(self, memory_id):
        """
        Delete a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("spmem_memory.delete", self, {"memory_id": memory_id, "sync_type": "async"})

        existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found")

        # Clean up graph entities before deleting from vector store
        if self.enable_graph:
            try:
                memory_text = existing_memory.payload.get("data", "")
                if memory_text:
                    filters = {}
                    for key in ("user_id", "agent_id", "run_id"):
                        val = existing_memory.payload.get(key)
                        if val:
                            filters[key] = val
                    if filters.get("user_id"):
                        await asyncio.to_thread(self.graph.delete, memory_text, filters)
            except Exception as e:
                logger.error(f"Error cleaning up graph for memory {memory_id}: {e}")

        await self._delete_memory(memory_id, existing_memory)
        return {"message": "Memory deleted successfully!"}

    async def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories asynchronously.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("spmem_memory.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"})
        memories = await asyncio.to_thread(self.vector_store.list, filters=filters)

        delete_tasks = []
        for memory in memories[0]:
            delete_tasks.append(self._delete_memory(memory.id))

        await asyncio.gather(*delete_tasks)

        logger.info(f"Deleted {len(memories[0])} memories")

        if self.enable_graph:
            await asyncio.to_thread(self.graph.delete_all, filters)

        return {"message": "Memories deleted successfully!"}

    async def history(self, memory_id):
        """
        Get the history of changes for a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("spmem_memory.history", self, {"memory_id": memory_id, "sync_type": "async"})
        return await asyncio.to_thread(self.db.get_history, memory_id)

    async def _create_memory(self, data: str, existing_embeddings: Union[Dict[str, List[float]], List[float]], metadata=None):
        logger.debug(f"Creating memory with {data=}")
        # existing_embeddings may be a dict (preferred) or a precomputed vector
        if isinstance(existing_embeddings, dict) and data in existing_embeddings:
            embeddings = existing_embeddings[data]
        elif not isinstance(existing_embeddings, dict):
            embeddings = existing_embeddings
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, memory_action="add")

        memory_id = str(uuid.uuid4())
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        if "created_at" not in new_metadata:
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        new_metadata["updated_at"] = new_metadata["created_at"]

        await asyncio.to_thread(
            self.vector_store.insert,
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[new_metadata],
        )

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            None,
            data,
            "ADD",
            created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"),
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        return memory_id

    async def _create_procedural_memory(self, messages, metadata=None, llm=None, prompt=None):
        """
        Create a procedural memory asynchronously

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            llm (llm, optional): LLM to use for the procedural memory creation. Defaults to None.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        try:
            from langchain_core.messages.utils import (
                convert_to_messages,  # type: ignore
            )
        except Exception:
            logger.error(
                "Import error while loading langchain-core. Please install 'langchain-core' to use procedural memory."
            )
            raise

        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {"role": "user", "content": "Create procedural memory of the above conversation."},
        ]

        try:
            if llm is not None:
                parsed_messages = convert_to_messages(parsed_messages)
                response = await asyncio.to_thread(llm.invoke, input=parsed_messages)
                procedural_memory = response.content
            else:
                procedural_memory = await asyncio.to_thread(self.llm.generate_response, messages=parsed_messages)
                procedural_memory = remove_code_blocks(procedural_memory)
        
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        new_metadata = deepcopy(metadata)
        new_metadata["memory_type"] = MemoryType.PROCEDURAL.value
        embeddings = await asyncio.to_thread(self.embedding_model.embed, procedural_memory, memory_action="add")
        memory_id = await self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=new_metadata)
        capture_event("spmem_memory._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "async"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    async def _update_memory(self, memory_id, data: str, existing_embeddings: Union[Dict[str, List[float]], List[float]], metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]

        if "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" not in new_metadata and "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        if isinstance(existing_embeddings, dict) and data in existing_embeddings:
            embeddings = existing_embeddings[data]
        elif not isinstance(existing_embeddings, dict):
            embeddings = existing_embeddings
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")

        await asyncio.to_thread(
            self.vector_store.update,
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        return memory_id

    async def _delete_memory(self, memory_id, existing_memory=None):
        logger.info(f"Deleting memory with {memory_id=}")
        if existing_memory is None:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
            if existing_memory is None:
                raise ValueError(f"Memory with id {memory_id} not found")
        prev_value = existing_memory.payload.get("data", "")

        # Preserve original created_at and record deletion time
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        updated_at = datetime.now(timezone.utc).isoformat()

        await asyncio.to_thread(self.vector_store.delete, vector_id=memory_id)
        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            None,
            "DELETE",
            created_at=created_at,
            updated_at=updated_at,
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )

        return memory_id

    async def reset(self):
        """
        Reset the memory store asynchronously by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")
        await asyncio.to_thread(self.vector_store.delete_col)

        gc.collect()

        if hasattr(self.vector_store, "client") and hasattr(self.vector_store.client, "close"):
            await asyncio.to_thread(self.vector_store.client.close)

        if hasattr(self.db, "connection") and self.db.connection:
            await asyncio.to_thread(lambda: self.db.connection.execute("DROP TABLE IF EXISTS history"))
            await asyncio.to_thread(self.db.connection.close)

        self.db = SQLiteManager(self.config.history_db_path)

        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        capture_event("spmem_memory.reset", self, {"sync_type": "async"})

        if self.enable_graph:
            try:
                await asyncio.to_thread(self.graph.reset)
            except Exception:
                logger.warning("Failed to reset graph store, continuing with reset")

    async def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")

