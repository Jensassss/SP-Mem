import logging
import re
import uuid
import sys
from warnings import filters
from spmem_memory.memory.privacy_processor import PrivacyProcessor
from spmem_memory.memory.utils import format_entities, sanitize_relationship_for_cypher
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    raise ImportError("langchain_neo4j is not installed. Please install it using pip install langchain-neo4j")

from spmem_memory.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from spmem_memory.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from spmem_memory.utils.factory import EmbedderFactory, LlmFactory

logger = logging.getLogger(__name__)


class MemoryGraph:
    def __init__(self, config):
        self.config = config
        self.graph = Neo4jGraph(
            url=self.config.graph_store.config.url,
            username=self.config.graph_store.config.username,
            password=self.config.graph_store.config.password,
            database=self.config.graph_store.config.database,
            refresh_schema=False,
            driver_config={"notifications_min_severity": "OFF"},
        )
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider, self.config.embedder.config, self.config.vector_store.config
        )
        self.node_label = ":`__Entity__`" if self.config.graph_store.config.base_label else ""

        if self.config.graph_store.config.base_label:
            # Safely add user_id index
            try:
                self.graph.query(f"CREATE INDEX entity_single IF NOT EXISTS FOR (n {self.node_label}) ON (n.user_id)")
            except Exception:
                pass
            try:  # Safely try to add composite index (Enterprise only)
                self.graph.query(
                    f"CREATE INDEX entity_composite IF NOT EXISTS FOR (n {self.node_label}) ON (n.name, n.user_id)"
                )
            except Exception:
                pass

        # Default to openai if no specific provider is configured
        self.llm_provider = "openai"
        if self.config.llm and self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store and self.config.graph_store.llm and self.config.graph_store.llm.provider:
            self.llm_provider = self.config.graph_store.llm.provider

        # Get LLM config with proper null checks
        llm_config = None
        if self.config.graph_store and self.config.graph_store.llm and hasattr(self.config.graph_store.llm, "config"):
            llm_config = self.config.graph_store.llm.config
        elif hasattr(self.config.llm, "config"):
            llm_config = self.config.llm.config
        self.llm = LlmFactory.create(self.llm_provider, llm_config)
        self.user_id = None
        # Use threshold from graph_store config, default to 0.7 for backward compatibility
        self.threshold = self.config.graph_store.threshold if hasattr(self.config.graph_store, 'threshold') else 0.7
        self.privacy_processor = PrivacyProcessor(self.llm)

    def _runtime_script_dir(self):
        """Resolve the running script directory; fallback to current working directory."""
        try:
            main_module = sys.modules.get("__main__")
            main_file = getattr(main_module, "__file__", None) if main_module else None
            if main_file:
                return Path(main_file).resolve().parent
        except Exception:
            pass
        return Path.cwd()

    def _privacy_mapping_dir(self):
        mapping_dir = self._runtime_script_dir() / "privacy_mappings"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        return mapping_dir

    @staticmethod
    def _safe_user_file_stem(user_id):
        token = str(user_id).strip()
        if not token:
            token = "unknown_user"
        return re.sub(r"[^\w.\-]+", "_", token)

    def _privacy_mapping_file_for_user(self, user_id):
        return self._privacy_mapping_dir() / f"{self._safe_user_file_stem(user_id)}.jsonl"

    def add(self, data, filters):
        print("[ADD] start")
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        print("[ADD] after _retrieve_nodes_from_data", len(entity_type_map))

        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
        to_be_added = self._normalize_user_node_name(to_be_added, filters)
        print("[ADD] after _establish_nodes_relations_from_data", len(to_be_added))

        sanitize_result = self._sanitize_triples_before_add(
            triples=to_be_added,
            entity_type_map=entity_type_map,
            data=data,
            filters=filters,
        )
        print("[ADD] after _sanitize_triples_before_add")

        to_be_added = sanitize_result["triples"]
        entity_type_map = sanitize_result["entity_type_map"]
        sanitized_node_meta = sanitize_result["sanitized_node_meta"]

        self._persist_privacy_mappings(sanitize_result["privacy_mappings"])
        print("[ADD] after _persist_privacy_mappings", len(sanitize_result["privacy_mappings"]))

        search_output = []
        # search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        print("[SKIP] after _search_graph_db", len(search_output))

        to_be_deleted = []
        # to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)
        print("[SKIP] after _get_delete_entities_from_search_output", len(to_be_deleted))
       
        deleted_entities = []
        # deleted_entities = self._delete_entities(to_be_deleted, filters)
        print("[SKIP] after _delete_entities")

        added_entities = self._add_entities(to_be_added, filters, entity_type_map, sanitized_node_meta)
        print("[ADD] after _add_entities")

        return {"deleted_entities": deleted_entities, "added_entities": added_entities}

    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            dict: A dictionary containing:
                - "contexts": List of search results from the base data store.
                - "entities": List of related graph data based on the query.
        """
        extracted_relations = self._extract_relations_from_query(query, filters)
        if not extracted_relations:
            logger.info("No relation extracted from query, returning empty graph results")
            return []

        relation_results = self._search_graph_db_by_relations(
            relations=extracted_relations,
            filters=filters,
            limit=limit,
        )

        logger.info(
            "Returned %s relation-search results (relations=%s)",
            len(relation_results),
            extracted_relations,
        )
        return relation_results

    def _extract_relations_from_query(self, query, filters):
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        extracted_triples = self._establish_nodes_relations_from_data(query, filters, entity_type_map)

        allowed_relations = set(self.privacy_processor.PRIVACY_RELATIONS.keys()) | {
            "shows",
            "music",
            "books",
            "games",
            "art",
            "sports",
            "fitness",
            "food",
            "beauty",
            "clothing",
            "technology",
            "transport",
            "travel",
            "pets",
            "learning_style_preference",
            "learning_resource_preference",
            "learning_schedule_preference",
            "learning_feedback_preference",
            "financial_market_sector_preference",
            "financial_news_source_preference",
            "risk_tolerance",
            "sustainability_preferences",
            "medical_decision_role",
            "medical_tone_preference",
            "medical_risk_attitude",
            "medical_followup_reminder_preference",
            
            "likes",
            "dislikes",
            "prefers",
            "enjoys",
            "uses",
            "owns",
            "visited",
            "knows",
            "met",
            "related_to",
        }

        extracted_relations = []
        seen = set()
        for triple in extracted_triples:
            relation = triple.get("relationship")
            if not isinstance(relation, str):
                continue
            normalized_relation = sanitize_relationship_for_cypher(relation.lower().replace(" ", "_"))
            if not normalized_relation or normalized_relation not in allowed_relations:
                continue
            if normalized_relation in seen:
                continue
            seen.add(normalized_relation)
            extracted_relations.append(normalized_relation)

        normalized_query = sanitize_relationship_for_cypher(str(query).lower().replace(" ", "_"))
        for relation in sorted(allowed_relations):
            if relation in normalized_query and relation not in seen:
                seen.add(relation)
                extracted_relations.append(relation)

        if not extracted_relations:
            intent_relations = self._classify_relations_from_query_intent(query, allowed_relations)
            for relation in intent_relations:
                if relation not in seen:
                    seen.add(relation)
                    extracted_relations.append(relation)

        return extracted_relations

    def _classify_relations_from_query_intent(self, query, allowed_relations):
        relation_list = sorted(allowed_relations)
        prompt = (
            "You are a relation router for graph search.\n"
            "Given a user search query, choose the most relevant relation names from the allowed list.\n"
            "Return ONLY relation names from the allowed list.\n"
            "Preferred output JSON format: {\"relations\": [\"rel_a\", \"rel_b\"]}\n\n"
            f"Allowed relations:\n{', '.join(relation_list)}\n\n"
            f"Query:\n{query}"
        )

        try:
            resp = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": "Select relation names for retrieval only."},
                    {"role": "user", "content": prompt},
                ],
            )
            if isinstance(resp, dict):
                text = (
                    resp.get("content")
                    or resp.get("text")
                    or resp.get("output")
                    or str(resp)
                )
            else:
                text = str(resp)
        except Exception:
            return []

        normalized_text = sanitize_relationship_for_cypher(text.lower().replace(" ", "_"))
        matched = []
        for relation in relation_list:
            if relation in normalized_text:
                matched.append(relation)
        return matched

    def _search_graph_db_by_relations(self, relations, filters, limit=100):
        if not relations:
            return []

        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)

        cypher_query = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})-[r]->(m {self.node_label} {{{node_props_str}}})
        WHERE type(r) IN $relations
        RETURN
            n.name AS source,
            elementId(n) AS source_id,
            type(r) AS relationship,
            elementId(r) AS relation_id,
            m.name AS destination,
            elementId(m) AS destination_id,
            coalesce(n.sanitized, false) AS source_sanitized,
            n.privacy_type AS source_privacy_type,
            n.privacy_ref_id AS source_privacy_ref_id,
            coalesce(m.sanitized, false) AS destination_sanitized,
            m.privacy_type AS destination_privacy_type,
            m.privacy_ref_id AS destination_privacy_ref_id,
            coalesce(r.mentions, 1) AS relation_mentions,
            coalesce(r.updated_at, r.created_at, r.created, 0) AS relation_timestamp,
            1.0 AS similarity
        ORDER BY relation_mentions DESC, relation_timestamp DESC
        LIMIT $limit
        """

        params = {
            "relations": relations,
            "user_id": filters["user_id"],
            "limit": limit,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        return self.graph.query(cypher_query, params=params)

    def delete_all(self, filters):
        # Build node properties for filtering
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)

        cypher = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})
        DETACH DELETE n
        """
        params = {"user_id": filters["user_id"]}
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]
        self.graph.query(cypher, params=params)

    def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.
         Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
        Returns:
            list: A list of dictionaries, each containing:
                - 'contexts': The base data store response for each memory.
                - 'entities': A list of strings representing the nodes and relationships
        """
        params = {"user_id": filters["user_id"], "limit": limit}

        # Build node properties based on filters
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
            params["run_id"] = filters["run_id"]
        node_props_str = ", ".join(node_props)

        query = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})-[r]->(m {self.node_label} {{{node_props_str}}})
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
        LIMIT $limit
        """
        results = self.graph.query(query, params=params)

        final_results = []
        for result in results:
            final_results.append(
                {
                    "source": result["source"],
                    "relationship": result["relationship"],
                    "target": result["target"],
                }
            )

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query."""
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use {filters['user_id']} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.",
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_type_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call["arguments"]["entities"]:
                    entity_name = item.get("entity") or item.get("name")
                    entity_type = item.get("entity_type") or item.get("type")

                    if entity_name and entity_type:
                        entity_type_map[entity_name] = entity_type

        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        entity_type_map = {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}
        logger.debug(f"Entity type map: {entity_type_map}\n search_results={search_results}")
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """Establish relations among the extracted nodes."""

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        if self.config.graph_store.custom_prompt:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            # Add the custom prompt line if configured
            system_content = system_content.replace("CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}")
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data},
            ]
        else:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}"},
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities.get("tool_calls"):
            entities = extracted_entities["tool_calls"][0].get("arguments", {}).get("entities", [])

        from pprint import pprint
        print("DEBUG entities before _remove_spaces_from_entities:")
        pprint(entities)
        entities = self._remove_spaces_from_entities(entities)
        logger.debug(f"Extracted entities: {entities}")
        return entities
    
    def _canonical_user_node_name(self, filters):
        return str(filters["user_id"]).strip().lower()

    def _infer_user_name_aliases(self, triples, user_aliases, canonical_user):
        aliases = set()

        def norm(value):
            if not isinstance(value, str):
                return None
            return value.strip().lower().replace(" ", "_")

        for triple in triples:
            relationship = norm(triple.get("relationship"))
            if relationship != "has_name":
                continue

            source = norm(triple.get("source"))
            destination = norm(triple.get("destination"))
            if not source or not destination:
                continue

            if source in user_aliases or source == destination:
                aliases.add(source)
                aliases.add(destination)
            elif destination in user_aliases:
                aliases.add(source)

        aliases.discard(canonical_user)
        aliases.difference_update(user_aliases)
        return aliases

    def _normalize_user_node_name(self, triples, filters):
        canonical = self._canonical_user_node_name(filters)
        user_id_norm = canonical

        user_aliases = {
            user_id_norm,
            f"user_id:{user_id_norm}",
            f"user_id:_{user_id_norm}",
        }
        user_name_aliases = self._infer_user_name_aliases(triples, user_aliases, canonical)

        def norm(value, replace_name_alias=False):
            if not isinstance(value, str):
                return value
            token = value.strip().lower().replace(" ", "_")
            if token in user_aliases:
                return canonical
            if replace_name_alias and token in user_name_aliases:
                return canonical
            return token

        out = []
        for triple in triples:
            item = dict(triple)
            item["source"] = norm(item.get("source"), replace_name_alias=True)
            item["destination"] = norm(item.get("destination"), replace_name_alias=False)
            out.append(item)
        return out

    def _sanitize_triples_before_add(self, triples, entity_type_map, data, filters):
        privacy_hits = self._detect_privacy_nodes(triples)

        sanitized_triples, sanitized_entity_type_map, privacy_mappings, sanitized_node_meta = self._replace_privacy_nodes(
            triples=triples,
            entity_type_map=entity_type_map,
            privacy_hits=privacy_hits,
            filters=filters,
        )

        return {
            "triples": sanitized_triples,
            "entity_type_map": sanitized_entity_type_map,
            "privacy_mappings": privacy_mappings,
            "sanitized_node_meta": sanitized_node_meta,
        }

#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
   

    def _detect_privacy_nodes(self, triples):
        privacy_hits = []

        for idx, triple in enumerate(triples):
            rel = triple.get("relationship")
            dest = triple.get("destination")

            if not isinstance(dest, str):
                continue

            dest = dest.strip()

            if rel in self.privacy_processor.PRIVACY_RELATIONS:
                privacy_type, mask_strategy = self.privacy_processor.PRIVACY_RELATIONS[rel]
                privacy_hits.append({
                    "triple_index": idx,
                    "node_role": "destination",
                    "raw_value": dest,
                    "relationship": rel,
                    "privacy_type": privacy_type,
                    "mask_strategy": mask_strategy,
                    "detected_by": "relation",
                })
                continue

        return privacy_hits


#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################

    
    def _replace_privacy_nodes(self, triples, entity_type_map, privacy_hits, filters):
        privacy_mappings = []
        sanitized_node_meta = {}

        sanitized_triples = [
            {
                "source": triple["source"],
                "relationship": triple["relationship"],
                "destination": triple["destination"],
            }
            for triple in triples
        ]

        for hit in privacy_hits:
            triple_index = hit["triple_index"]
            node_role = hit["node_role"]
            raw_value = hit["raw_value"]
            privacy_type = hit["privacy_type"]
            mask_strategy = hit["mask_strategy"]

            privacy_ref_id = str(uuid.uuid4())

            base_sanitized_value = self.privacy_processor.sanitize_value(
                raw_value=raw_value,
                privacy_type=privacy_type,
                mask_strategy=mask_strategy,
                filters=filters,
            )
            multi_value_medical_types = {
                "MEDICAL_SYMPTOMS",
                "MEDICAL_TREATMENTS",
                "MEDICAL_EXAMS",
            }

            if privacy_type in multi_value_medical_types:
                suffix = privacy_ref_id.replace("-", "")[:4]
                sanitized_value = f"{base_sanitized_value}_{suffix}"
            else:
                sanitized_value = base_sanitized_value

            sanitized_triples[triple_index][node_role] = sanitized_value

            privacy_mappings.append({
                "privacy_ref_id": privacy_ref_id,
                "raw_value": raw_value,
                "sanitized_value": sanitized_value,
                "privacy_type": privacy_type,
                "mask_strategy": mask_strategy,
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "user_id": filters.get("user_id"),
                "agent_id": filters.get("agent_id"),
                "run_id": filters.get("run_id"),
            })

            sanitized_node_meta[sanitized_value] = {
                "sanitized": True,
                "privacy_type": privacy_type,
                "privacy_ref_id": privacy_ref_id,
                "mask_strategy": mask_strategy
            }

        sanitized_entity_type_map = {}

        for triple in sanitized_triples:
            for node_role in ("source", "destination"):
                node = triple.get(node_role)
                if not isinstance(node, str):
                    continue

                if node in sanitized_node_meta:
                    sanitized_entity_type_map[node] = sanitized_node_meta[node]["privacy_type"]
                else:
                    sanitized_entity_type_map[node] = entity_type_map.get(node, "unknown")

        return sanitized_triples, sanitized_entity_type_map, privacy_mappings, sanitized_node_meta
        
        
    def _persist_privacy_mappings(self, privacy_mappings):
        if not privacy_mappings:
            return

        mappings_by_user = {}
        for item in privacy_mappings:
            user_id = item.get("user_id")
            if user_id is None or str(user_id).strip() == "":
                continue
            user_key = str(user_id)
            mappings_by_user.setdefault(user_key, []).append(item)

        for user_key, items in mappings_by_user.items():
            path = self._privacy_mapping_file_for_user(user_key)
            with path.open("a", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def lookup_privacy_value(self, privacy_ref_id, user_id, agent_id=None, run_id=None):
        if not privacy_ref_id or not user_id:
            return None

        user_path = self._privacy_mapping_file_for_user(user_id)
        legacy_path = self._runtime_script_dir() / "privacy_mappings.jsonl"
        candidate_paths = [user_path]
        if legacy_path != user_path:
            candidate_paths.append(legacy_path)

        matched_item = None

        for path in candidate_paths:
            if not path.exists():
                continue

            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                    except Exception:
                        continue

                    if item.get("privacy_ref_id") != privacy_ref_id:
                        continue
                    if str(item.get("user_id")) != str(user_id):
                        continue
                    if agent_id is not None and item.get("agent_id") != agent_id:
                        continue
                    if run_id is not None and item.get("run_id") != run_id:
                        continue

                    matched_item = item

        if not matched_item:
            return None

        return matched_item.get("raw_value")

    def _search_graph_db(self, node_list, filters, limit=100, query_embedding=None):
        """Search similar nodes among and their respective incoming and outgoing relations."""
        result_relations = []

        # Build node properties for filtering
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)
            effective_query_embedding = query_embedding if query_embedding is not None else n_embedding

            cypher_query = f"""
            MATCH (n {self.node_label} {{{node_props_str}}})
            WHERE n.embedding IS NOT NULL
            WITH n, round(2 * vector.similarity.cosine(n.embedding, $n_embedding) - 1, 4) AS anchor_similarity // denormalize for backward compatibility
            WHERE anchor_similarity >= $threshold
            CALL {{
                WITH n, anchor_similarity
                MATCH (n)-[r]->(m {self.node_label} {{{node_props_str}}})
                RETURN
                    n.name AS source,
                    elementId(n) AS source_id,
                    type(r) AS relationship,
                    elementId(r) AS relation_id,
                    m.name AS destination,
                    elementId(m) AS destination_id,
                    coalesce(n.sanitized, false) AS source_sanitized,
                    n.privacy_type AS source_privacy_type,
                    n.privacy_ref_id AS source_privacy_ref_id,
                    coalesce(m.sanitized, false) AS destination_sanitized,
                    m.privacy_type AS destination_privacy_type,
                    m.privacy_ref_id AS destination_privacy_ref_id,
                    anchor_similarity AS source_similarity,
                    CASE
                        WHEN m.embedding IS NULL THEN -1.0
                        ELSE round(2 * vector.similarity.cosine(m.embedding, $query_embedding) - 1, 4)
                    END AS destination_similarity
                UNION
                WITH n, anchor_similarity
                MATCH (n)<-[r]-(m {self.node_label} {{{node_props_str}}})
                RETURN
                    m.name AS source,
                    elementId(m) AS source_id,
                    type(r) AS relationship,
                    elementId(r) AS relation_id,
                    n.name AS destination,
                    elementId(n) AS destination_id,
                    coalesce(m.sanitized, false) AS source_sanitized,
                    m.privacy_type AS source_privacy_type,
                    m.privacy_ref_id AS source_privacy_ref_id,
                    coalesce(n.sanitized, false) AS destination_sanitized,
                    n.privacy_type AS destination_privacy_type,
                    n.privacy_ref_id AS destination_privacy_ref_id,
                    anchor_similarity AS source_similarity,
                    CASE
                        WHEN n.embedding IS NULL THEN -1.0
                        ELSE round(2 * vector.similarity.cosine(n.embedding, $query_embedding) - 1, 4)
                    END AS destination_similarity
            }}
            WITH distinct
                source,
                source_id,
                relationship,
                relation_id,
                destination,
                destination_id,
                source_sanitized,
                source_privacy_type,
                source_privacy_ref_id,
                destination_sanitized,
                destination_privacy_type,
                destination_privacy_ref_id,
                source_similarity,
                destination_similarity,
                round(0.35 * source_similarity + 0.65 * destination_similarity, 4) AS similarity
            RETURN
                source,
                source_id,
                relationship,
                relation_id,
                destination,
                destination_id,
                source_sanitized,
                source_privacy_type,
                source_privacy_ref_id,
                destination_sanitized,
                destination_privacy_type,
                destination_privacy_ref_id,
                source_similarity,
                destination_similarity,
                similarity
            ORDER BY similarity DESC, destination_similarity DESC, source_similarity DESC
            LIMIT $limit
            """

            params = {
                "n_embedding": n_embedding,
                "query_embedding": effective_query_embedding,
                "threshold": self.threshold,
                "user_id": filters["user_id"],
                "limit": limit,
            }
            if filters.get("agent_id"):
                params["agent_id"] = filters["agent_id"]
            if filters.get("run_id"):
                params["run_id"] = filters["run_id"]

            ans = self.graph.query(cypher_query, params=params)
            result_relations.extend(ans)

        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        system_prompt, user_prompt = get_delete_messages(search_output_string, data, user_identity)

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )

        to_be_deleted = []
        for item in memory_updates.get("tool_calls", []):
            if item.get("name") == "delete_graph_memory":
                to_be_deleted.append(item.get("arguments"))
        # Clean entities formatting
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, filters):
        """Delete the entities from the graph."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []

        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Build the agent filter for the query

            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }

            if agent_id:
                params["agent_id"] = agent_id
            if run_id:
                params["run_id"] = run_id

            # Build node properties for filtering
            source_props = ["name: $source_name", "user_id: $user_id"]
            dest_props = ["name: $dest_name", "user_id: $user_id"]
            if agent_id:
                source_props.append("agent_id: $agent_id")
                dest_props.append("agent_id: $agent_id")
            if run_id:
                source_props.append("run_id: $run_id")
                dest_props.append("run_id: $run_id")
            source_props_str = ", ".join(source_props)
            dest_props_str = ", ".join(dest_props)

            # Delete the specific relationship between nodes
            cypher = f"""
            MATCH (n {self.node_label} {{{source_props_str}}})
            -[r:{relationship}]->
            (m {self.node_label} {{{dest_props_str}}})
            
            DELETE r
            RETURN 
                n.name AS source,
                m.name AS target,
                type(r) AS relationship
            """

            result = self.graph.query(cypher, params=params)
            results.append(result)

        return results


    def _add_entities(self, to_be_added, filters, entity_type_map, sanitized_node_meta=None):
        """Add the new entities to the graph. Merge the nodes if they already exist."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []
        
        for item in to_be_added:
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]


            source_meta = (sanitized_node_meta or {}).get(source, {})
            destination_meta = (sanitized_node_meta or {}).get(destination, {})
            ###
            

            # types
            canonical_user = self._canonical_user_node_name(filters)

            if source == canonical_user:
                source_type = "__User__"
            else:
                source_type = entity_type_map.get(source, "__Entity__")
            source_label = self.node_label if self.node_label else f":`{source_type}`"
            source_extra_set = f", source:`{source_type}`" if self.node_label else ""

            if destination == canonical_user:
                destination_type = "__User__"
            else:
                destination_type = entity_type_map.get(destination, "__Entity__")
            destination_label = self.node_label if self.node_label else f":`{destination_type}`"
            destination_extra_set = f", destination:`{destination_type}`" if self.node_label else ""

            # embeddings
            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)
          
            canonical_user = self._canonical_user_node_name(filters)
        
            if source == canonical_user:
                source_node_search_result = self._search_node_exact(source, filters)
            else:
                source_node_search_result = self._search_source_node(source_embedding, filters, threshold=self.threshold)

            if destination == canonical_user:
                destination_node_search_result = self._search_node_exact(destination, filters)
            else:
                destination_node_search_result = self._search_destination_node(dest_embedding, filters, threshold=self.threshold)

            # TODO: Create a cypher query and common params for all the cases
            if not destination_node_search_result and source_node_search_result:
                merge_props = ["name: $destination_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                if run_id:
                    merge_props.append("run_id: $run_id")
                merge_props_str = ", ".join(merge_props)

                cypher = f"""
                MATCH (source)
                WHERE elementId(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1,
                    source.sanitized = $source_sanitized,
                    source.privacy_type = $source_privacy_type,
                    source.privacy_ref_id = $source_privacy_ref_id
                WITH source
                MERGE (destination {destination_label} {{{merge_props_str}}})
                ON CREATE SET
                    destination.created = timestamp(),
                    destination.mentions = 1,
                    destination.sanitized = $destination_sanitized,
                    destination.privacy_type = $destination_privacy_type,
                    destination.privacy_ref_id = $destination_privacy_ref_id
                    {destination_extra_set}
                ON MATCH SET
                    destination.mentions = coalesce(destination.mentions, 0) + 1,
                    destination.sanitized = $destination_sanitized,
                    destination.privacy_type = $destination_privacy_type,
                    destination.privacy_ref_id = $destination_privacy_ref_id
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', $destination_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created = timestamp(),
                    r.mentions = 1
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_name": destination,
                    "destination_embedding": dest_embedding,
                    "user_id": user_id,

                    "source_sanitized": source_meta.get("sanitized", False),
                    "source_privacy_type": source_meta.get("privacy_type"),
                    "source_privacy_ref_id": source_meta.get("privacy_ref_id"),

                    "destination_sanitized": destination_meta.get("sanitized", False),
                    "destination_privacy_type": destination_meta.get("privacy_type"),
                    "destination_privacy_ref_id": destination_meta.get("privacy_ref_id"),
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            elif destination_node_search_result and not source_node_search_result:
                merge_props = ["name: $source_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                if run_id:
                    merge_props.append("run_id: $run_id")
                merge_props_str = ", ".join(merge_props)

                cypher = f"""
                MATCH (destination)
                WHERE elementId(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1,
                    destination.sanitized = $destination_sanitized,
                    destination.privacy_type = $destination_privacy_type,
                    destination.privacy_ref_id = $destination_privacy_ref_id
                WITH destination
                MERGE (source {source_label} {{{merge_props_str}}})
                ON CREATE SET
                    source.created = timestamp(),
                    source.mentions = 1,
                    source.sanitized = $source_sanitized,
                    source.privacy_type = $source_privacy_type,
                    source.privacy_ref_id = $source_privacy_ref_id
                    {source_extra_set}
                ON MATCH SET
                    source.mentions = coalesce(source.mentions, 0) + 1,
                    source.sanitized = $source_sanitized,
                    source.privacy_type = $source_privacy_type,
                    source.privacy_ref_id = $source_privacy_ref_id
                WITH source, destination
                CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created = timestamp(),
                    r.mentions = 1
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "source_name": source,
                    "source_embedding": source_embedding,
                    "user_id": user_id,

                    "source_sanitized": source_meta.get("sanitized", False),
                    "source_privacy_type": source_meta.get("privacy_type"),
                    "source_privacy_ref_id": source_meta.get("privacy_ref_id"),

                    "destination_sanitized": destination_meta.get("sanitized", False),
                    "destination_privacy_type": destination_meta.get("privacy_type"),
                    "destination_privacy_ref_id": destination_meta.get("privacy_ref_id"),
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            elif source_node_search_result and destination_node_search_result:
                cypher = f"""
                MATCH (source)
                WHERE elementId(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1,
                    source.sanitized = $source_sanitized,
                    source.privacy_type = $source_privacy_type,
                    source.privacy_ref_id = $source_privacy_ref_id
                WITH source
                MATCH (destination)
                WHERE elementId(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1,
                    destination.sanitized = $destination_sanitized,
                    destination.privacy_type = $destination_privacy_type,
                    destination.privacy_ref_id = $destination_privacy_ref_id
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created_at = timestamp(),
                    r.updated_at = timestamp(),
                    r.mentions = 1
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1,
                    r.updated_at = timestamp()
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "user_id": user_id,

                    "source_sanitized": source_meta.get("sanitized", False),
                    "source_privacy_type": source_meta.get("privacy_type"),
                    "source_privacy_ref_id": source_meta.get("privacy_ref_id"),

                    "destination_sanitized": destination_meta.get("sanitized", False),
                    "destination_privacy_type": destination_meta.get("privacy_type"),
                    "destination_privacy_ref_id": destination_meta.get("privacy_ref_id"),
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id
            else:
                source_props = ["name: $source_name", "user_id: $user_id"]
                dest_props = ["name: $dest_name", "user_id: $user_id"]
                if agent_id:
                    source_props.append("agent_id: $agent_id")
                    dest_props.append("agent_id: $agent_id")
                if run_id:
                    source_props.append("run_id: $run_id")
                    dest_props.append("run_id: $run_id")
                source_props_str = ", ".join(source_props)
                dest_props_str = ", ".join(dest_props)

                cypher = f"""
                MERGE (source {source_label} {{{source_props_str}}})
                ON CREATE SET
                    source.created = timestamp(),
                    source.mentions = 1,
                    source.sanitized = $source_sanitized,
                    source.privacy_type = $source_privacy_type,
                    source.privacy_ref_id = $source_privacy_ref_id
                    {source_extra_set}
                ON MATCH SET
                    source.mentions = coalesce(source.mentions, 0) + 1,
                    source.sanitized = $source_sanitized,
                    source.privacy_type = $source_privacy_type,
                    source.privacy_ref_id = $source_privacy_ref_id
                WITH source
                CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                WITH source
                MERGE (destination {destination_label} {{{dest_props_str}}})
                ON CREATE SET
                    destination.created = timestamp(),
                    destination.mentions = 1,
                    destination.sanitized = $destination_sanitized,
                    destination.privacy_type = $destination_privacy_type,
                    destination.privacy_ref_id = $destination_privacy_ref_id
                    {destination_extra_set}
                ON MATCH SET
                    destination.mentions = coalesce(destination.mentions, 0) + 1,
                    destination.sanitized = $destination_sanitized,
                    destination.privacy_type = $destination_privacy_type,
                    destination.privacy_ref_id = $destination_privacy_ref_id
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', $dest_embedding)
                WITH source, destination
                MERGE (source)-[rel:{relationship}]->(destination)
                ON CREATE SET
                    rel.created = timestamp(),
                    rel.mentions = 1
                ON MATCH SET
                    rel.mentions = coalesce(rel.mentions, 0) + 1
                RETURN source.name AS source, type(rel) AS relationship, destination.name AS target
                """

                params = {
                    "source_name": source,
                    "dest_name": destination,
                    "source_embedding": source_embedding,
                    "dest_embedding": dest_embedding,
                    "user_id": user_id,

                    "source_sanitized": source_meta.get("sanitized", False),
                    "source_privacy_type": source_meta.get("privacy_type"),
                    "source_privacy_ref_id": source_meta.get("privacy_ref_id"),

                    "destination_sanitized": destination_meta.get("sanitized", False),
                    "destination_privacy_type": destination_meta.get("privacy_type"),
                    "destination_privacy_ref_id": destination_meta.get("privacy_ref_id"),
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id
            result = self.graph.query(cypher, params=params)
            results.append(result)
        return results


    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            # Use the sanitization function for relationships to handle special characters
            item["relationship"] = sanitize_relationship_for_cypher(item["relationship"].lower().replace(" ", "_"))
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    def _search_source_node(self, source_embedding, filters, threshold=0.9):
        # Build WHERE conditions
        where_conditions = ["source_candidate.embedding IS NOT NULL", "source_candidate.user_id = $user_id"]
        if filters.get("agent_id"):
            where_conditions.append("source_candidate.agent_id = $agent_id")
        if filters.get("run_id"):
            where_conditions.append("source_candidate.run_id = $run_id")
        where_clause = " AND ".join(where_conditions)

        cypher = f"""
            MATCH (source_candidate {self.node_label})
            WHERE {where_clause}

            WITH source_candidate,
            round(2 * vector.similarity.cosine(source_candidate.embedding, $source_embedding) - 1, 4) AS source_similarity // denormalize for backward compatibility
            WHERE source_similarity >= $threshold

            WITH source_candidate, source_similarity
            ORDER BY source_similarity DESC
            LIMIT 1

            RETURN elementId(source_candidate)
            """

        params = {
            "source_embedding": source_embedding,
            "user_id": filters["user_id"],
            "threshold": threshold,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        result = self.graph.query(cypher, params=params)
        return result

    def _search_destination_node(self, destination_embedding, filters, threshold=0.9):
        # Build WHERE conditions
        where_conditions = ["destination_candidate.embedding IS NOT NULL", "destination_candidate.user_id = $user_id"]
        if filters.get("agent_id"):
            where_conditions.append("destination_candidate.agent_id = $agent_id")
        if filters.get("run_id"):
            where_conditions.append("destination_candidate.run_id = $run_id")
        where_clause = " AND ".join(where_conditions)

        cypher = f"""
            MATCH (destination_candidate {self.node_label})
            WHERE {where_clause}

            WITH destination_candidate,
            round(2 * vector.similarity.cosine(destination_candidate.embedding, $destination_embedding) - 1, 4) AS destination_similarity // denormalize for backward compatibility

            WHERE destination_similarity >= $threshold

            WITH destination_candidate, destination_similarity
            ORDER BY destination_similarity DESC
            LIMIT 1

            RETURN elementId(destination_candidate)
            """

        params = {
            "destination_embedding": destination_embedding,
            "user_id": filters["user_id"],
            "threshold": threshold,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        result = self.graph.query(cypher, params=params)
        return result
    

    def _search_node_exact(self, node_name, filters):
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id")
        run_id = filters.get("run_id")

        where_clauses = ["n.name = $node_name", "n.user_id = $user_id"]
        params = {
            "node_name": node_name,
            "user_id": user_id,
        }

        if agent_id:
            where_clauses.append("n.agent_id = $agent_id")
            params["agent_id"] = agent_id
        if run_id:
            where_clauses.append("n.run_id = $run_id")
            params["run_id"] = run_id

        where_str = " AND ".join(where_clauses)

        cypher = f"""
        MATCH (n {self.node_label})
        WHERE {where_str}
        RETURN
            elementId(n) AS `elementId(source_candidate)`,
            elementId(n) AS `elementId(destination_candidate)`,
            n.name AS name
        LIMIT 1
        """

        result = self.graph.query(cypher, params=params)
        return result
   
    # Reset is not defined in base.py
    def reset(self):
        """Reset the graph by clearing all nodes and relationships."""
        logger.warning("Clearing graph...")
        cypher_query = """
        MATCH (n) DETACH DELETE n
        """
        return self.graph.query(cypher_query)
