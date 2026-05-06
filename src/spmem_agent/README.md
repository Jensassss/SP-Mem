# Privacy Aware Agent

This folder contains a standalone privacy-aware agent implementation.
It does not modify code in other project folders.

## Flow

1. Analyze user query and infer required entities.
2. Detect whether required entities include private fields.
3. If private data is needed, return `awaiting_consent`.
4. If consent is `false`, answer from sanitized memory retrieval.
5. If consent is `true`, hydrate private values:
   - Graph path: by `privacy_ref_id`
   - Vector path: by `entity_hash`
6. Generate final answer from retrieved evidence.

## Structure

- `agent.py`: `PrivacyAwareAgent` orchestrator only.
- `llm.py`: helper to build `llm_call(system_prompt, user_prompt)`.
- `example_usage.py`: runnable example + memory config.
- `query_processor.py`: compatibility facade; re-exports the pipeline API.
- `core/constants.py`: entity whitelist, privacy set, entity->relation mapping.
- `core/prompts.py`: analysis/answer prompts and few-shot examples.
- `core/planner.py`: query analysis and task-plan building.
- `core/retriever.py`: graph/vector/hybrid retrieval.
- `core/hydrator.py`: privacy value hydration (`privacy_ref_id`, `entity_hash`).
- `core/responder.py`: final answer generation from structured memory evidence.
- `core/pipeline.py`: end-to-end functions (`process_query`, `continue_after_consent`).
- `core/debug.py`: debug switch and debug print.
- `core/utils.py`: parsing/normalization/helper utilities.

## Where to edit what

- Agent model and base URL:
  - set in `example_usage.py` via `build_openai_llm_call(...)`.
- Memory-side model/embedder/vector/graph config:
  - set in `example_usage.py` `memory_config`.
- Query analysis behavior:
  - `core/planner.py` (`analyze_required_entities`, `build_task_plan`).
- Prompts and few-shot:
  - `core/prompts.py`.
- Retrieval logic:
  - `core/retriever.py`.
- Consent and flow orchestration:
  - `agent.py` (`ask`, `continue_with_consent`, `run`).
- Final response style:
  - `core/prompts.py` + `core/responder.py`.

## API

### `PrivacyAwareAgent.ask(query, user_id)`

- Returns:
  - `{"status":"awaiting_consent", ...}` if privacy is required
  - `{"status":"answered", ...}` if no privacy consent is needed

### `PrivacyAwareAgent.continue_with_consent(session_id, consent)`

- Continue an `awaiting_consent` session.
- `consent=False` -> sanitized answer
- `consent=True` -> hydrated precise-value answer (when lookup data exists)

### `PrivacyAwareAgent.run(query, user_id, consent=None, session_id=None)`

Convenience wrapper:
- `consent=None`: start flow
- `consent` + `session_id`: continue pending session
- `consent` without `session_id`: one-shot execute

## Notes

- If `memory.graph.lookup_privacy_value` exists, it is auto-wired for graph privacy hydration.
- For vector hydration, it uses hash-based mapping lookup through `memory.db`.
