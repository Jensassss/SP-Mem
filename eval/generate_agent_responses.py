from __future__ import annotations

# Migrated from the experiment tree's eval/generate_agent_responses.py.
# See docs/original_workflow.md for source hash and bounded public adaptations.

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI
try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

# Local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spmem_memory import Memory  # noqa: E402
from spmem_agent.agent import PrivacyAwareAgent  # noqa: E402
from spmem_agent import query_processor as agent_query_processor  # noqa: E402


EVAL_MODES = (
    "mixed_denied",
    "mixed_allowed",
    "privacy_denied",
    "privacy_allowed",
    "non_privacy_only",
)


def _eval_script_dir() -> Path:
    return Path(__file__).resolve().parent


def _eval_privacy_mapping_dir() -> Path:
    path = _eval_script_dir() / "privacy_mappings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sync_user_privacy_mapping_file(
    *,
    user_id: str,
    source_dir: Optional[Path],
) -> bool:
    """Check the configured per-user mapping without copying private data."""
    if source_dir is None:
        return False

    source = source_dir / f"{user_id}.jsonl"
    if not source.exists():
        return False

    return True


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_user_id(data_dir: Path, user_index: int) -> str:
    user_file = data_dir / f"user_{user_index:04d}.json"
    if not user_file.exists():
        raise FileNotFoundError(f"User file not found: {user_file}")
    payload = json.loads(user_file.read_text(encoding="utf-8"))
    user_id = str(payload.get("user_id", "")).strip()
    if not user_id:
        raise ValueError(f"Empty user_id in {user_file}")
    return user_id


def _resolve_user_indices(args: argparse.Namespace) -> List[int]:
    single = args.user_index
    start = args.user_index_start
    end = args.user_index_end

    if single is not None:
        return [int(single)]

    if start is None or end is None:
        raise ValueError(
            "Provide either --user-index, or both --user-index-start and --user-index-end."
        )
    if int(end) < int(start):
        raise ValueError(
            f"Invalid user range: start={start}, end={end}"
        )
    return list(range(int(start), int(end) + 1))


def _normalize_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _sample_modes_for_sample(sample: Dict[str, Any]) -> Tuple[str, ...]:
    privacy_entities = _normalize_list(sample.get("privacy_entities", []))
    pref_entities = _normalize_list(sample.get("preference_entities", []))
    if privacy_entities and pref_entities:
        return ("mixed_denied", "mixed_allowed")
    if privacy_entities and not pref_entities:
        return ("privacy_denied", "privacy_allowed")
    return ("non_privacy_only",)


def _count_response_tasks(samples: List[Dict[str, Any]]) -> int:
    total = 0
    for sample in samples:
        query = _safe_str(sample.get("query", "")).strip()
        if not query:
            continue
        total += len(_sample_modes_for_sample(sample))
    return total


def _safe_str(value: Any) -> str:
    return str(value) if value is not None else ""


def _safe_console_text(value: Any) -> str:
    """
    Normalize text to the active stdout encoding, replacing unsupported chars
    (for example emoji under Windows GBK) so logging never crashes the run.
    """
    text = _safe_str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding, errors="strict")
        return text
    except Exception:
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _build_print_fn() -> Callable[[str], None]:
    def _print_fn(msg: str) -> None:
        text = _safe_console_text(msg)
        if tqdm is not None:
            try:
                tqdm.write(text)
                return
            except Exception:
                pass
        try:
            print(text)
        except Exception:
            # Last-resort fallback: avoid raising from logging path.
            print(_safe_console_text(text))

    return _print_fn


def _pick_env(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


@dataclass
class ResponseModelConfig:
    name: str
    model: str
    base_url: str
    api_key: str


class LLMUsageTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_calls = 0
        self.llm_latency_seconds = 0.0

    def observe(self, response: Any, elapsed_seconds: float) -> None:
        self.llm_calls += 1
        self.llm_latency_seconds += float(elapsed_seconds)

        usage = getattr(response, "usage", None)
        if usage is None:
            return

        # OpenAI SDK usage object is usually attribute-based.
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        if isinstance(prompt_tokens, int):
            self.prompt_tokens += prompt_tokens
        if isinstance(completion_tokens, int):
            self.completion_tokens += completion_tokens
        if isinstance(total_tokens, int):
            self.total_tokens += total_tokens

    def snapshot(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "llm_latency_seconds": round(self.llm_latency_seconds, 6),
        }


def _extract_int_usage(usage_obj: Any, key: str) -> Optional[int]:
    value = getattr(usage_obj, key, None)
    if value is None and isinstance(usage_obj, dict):
        value = usage_obj.get(key)
    if isinstance(value, int):
        return value
    return None


def _estimate_tokens_from_text(text: str) -> int:
    cleaned = str(text or "")
    if not cleaned:
        return 0
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return int(len(enc.encode(cleaned)))
        except Exception:
            pass
    # Fallback heuristic when tokenizer is unavailable.
    return max(1, len(cleaned) // 4)


def _estimate_tokens_from_embedding_input(input_payload: Any) -> int:
    if isinstance(input_payload, str):
        return _estimate_tokens_from_text(input_payload)
    if isinstance(input_payload, list):
        total = 0
        for item in input_payload:
            if isinstance(item, str):
                total += _estimate_tokens_from_text(item)
            else:
                total += _estimate_tokens_from_text(str(item))
        return total
    if input_payload is None:
        return 0
    return _estimate_tokens_from_text(str(input_payload))


class EmbeddingUsageTracker:
    def __init__(self) -> None:
        self.tracking_source = "none"
        self.reset()

    def reset(self) -> None:
        self.embedding_prompt_tokens = 0
        self.embedding_total_tokens = 0
        self.embedding_calls = 0
        self.embedding_latency_seconds = 0.0

    def observe(
        self,
        *,
        response: Any = None,
        input_payload: Any = None,
        elapsed_seconds: float = 0.0,
    ) -> None:
        self.embedding_calls += 1
        self.embedding_latency_seconds += float(elapsed_seconds)

        usage = getattr(response, "usage", None) if response is not None else None
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")

        prompt_tokens = _extract_int_usage(usage, "prompt_tokens") if usage is not None else None
        total_tokens = _extract_int_usage(usage, "total_tokens") if usage is not None else None

        estimated = _estimate_tokens_from_embedding_input(input_payload)
        self.embedding_prompt_tokens += prompt_tokens if isinstance(prompt_tokens, int) else estimated
        if isinstance(total_tokens, int):
            self.embedding_total_tokens += total_tokens
        elif isinstance(prompt_tokens, int):
            self.embedding_total_tokens += prompt_tokens
        else:
            self.embedding_total_tokens += estimated

    def snapshot(self) -> Dict[str, Any]:
        return {
            "embedding_prompt_tokens": int(self.embedding_prompt_tokens),
            "embedding_total_tokens": int(self.embedding_total_tokens),
            "embedding_calls": int(self.embedding_calls),
            "embedding_latency_seconds": round(self.embedding_latency_seconds, 6),
        }


def _attach_embedding_usage_tracker(memory: Memory) -> EmbeddingUsageTracker:
    tracker = EmbeddingUsageTracker()
    embedding_model = getattr(memory, "embedding_model", None)
    if embedding_model is None:
        return tracker

    existing = getattr(embedding_model, "_eval_embedding_usage_tracker", None)
    if isinstance(existing, EmbeddingUsageTracker):
        return existing

    client = getattr(embedding_model, "client", None)
    embeddings_api = getattr(client, "embeddings", None) if client is not None else None
    create_fn = getattr(embeddings_api, "create", None) if embeddings_api is not None else None

    if callable(create_fn):
        tracker.tracking_source = "client.embeddings.create"

        def wrapped_create(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            response = create_fn(*args, **kwargs)
            elapsed = time.perf_counter() - started
            input_payload = kwargs.get("input")
            if input_payload is None and args:
                input_payload = args[0]
            tracker.observe(response=response, input_payload=input_payload, elapsed_seconds=elapsed)
            return response

        try:
            embeddings_api.create = wrapped_create
            setattr(embedding_model, "_eval_embedding_usage_tracker", tracker)
            return tracker
        except Exception:
            # Some SDK resource objects may disallow attribute reassignment.
            pass

    embed_fn = getattr(embedding_model, "embed", None)
    if callable(embed_fn):
        tracker.tracking_source = "embedding_model.embed"

        def wrapped_embed(text: Any, memory_action: Optional[str] = None) -> Any:
            started = time.perf_counter()
            result = embed_fn(text, memory_action)
            elapsed = time.perf_counter() - started
            tracker.observe(response=None, input_payload=text, elapsed_seconds=elapsed)
            return result

        embedding_model.embed = wrapped_embed
        setattr(embedding_model, "_eval_embedding_usage_tracker", tracker)
        return tracker

    return tracker


def _compose_usage_snapshot(
    llm_tracker: LLMUsageTracker,
    embedding_tracker: EmbeddingUsageTracker,
) -> Dict[str, Any]:
    llm = llm_tracker.snapshot()
    emb = embedding_tracker.snapshot()
    llm_prompt = int(llm.get("prompt_tokens", 0))
    llm_total = int(llm.get("total_tokens", 0))
    emb_prompt = int(emb.get("embedding_prompt_tokens", 0))
    emb_total = int(emb.get("embedding_total_tokens", 0))

    merged: Dict[str, Any] = {
        **llm,
        **emb,
        "pipeline_prompt_tokens": llm_prompt + emb_prompt,
        "pipeline_total_tokens": llm_total + emb_total,
    }
    return merged


def _augment_usage_with_consent_tokens(
    usage_snapshot: Dict[str, Any],
    request_text: str,
) -> Dict[str, Any]:
    merged = dict(usage_snapshot or {})
    consent_request_tokens = _estimate_tokens_from_text(request_text or "")
    base_pipeline_total = int(merged.get("pipeline_total_tokens", 0))
    merged["consent_request_tokens"] = int(consent_request_tokens)
    merged["pipeline_total_tokens"] = base_pipeline_total + int(consent_request_tokens)
    return merged


def _build_tracked_llm_call(
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    temperature: float = 0.0,
) -> Tuple[Callable[[str, str], str], LLMUsageTracker]:
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)
    tracker = LLMUsageTracker()

    def llm_call(system_prompt: str, user_prompt: str) -> str:
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        elapsed = time.perf_counter() - started
        tracker.observe(response, elapsed)
        content = response.choices[0].message.content
        return content or ""

    return llm_call, tracker


def _build_memory_config(args: argparse.Namespace) -> Dict[str, Any]:
    llm_api_key = _pick_env("SPMEM_MEMORY_API_KEY", "MEMORY_LLM_API_KEY", "OPENAI_API_KEY")
    llm_base_url = _pick_env("SPMEM_MEMORY_BASE_URL", "MEMORY_LLM_BASE_URL", "OPENAI_BASE_URL")
    embed_api_key = _pick_env(
        "SPMEM_EMBEDDING_API_KEY",
        "EMBEDDING_API_KEY",
        "OPENAI_EMBEDDING_API_KEY",
        "OPENAI_API_KEY",
    )
    embed_base_url = _pick_env(
        "SPMEM_EMBEDDING_BASE_URL",
        "EMBEDDING_BASE_URL",
        "OPENAI_EMBEDDING_BASE_URL",
        "OPENAI_BASE_URL",
        default="https://api.openai.com/v1",
    )

    vector_cfg: Dict[str, Any] = {
        "collection_name": args.collection_name,
        "embedding_model_dims": args.embed_dims,
    }
    if args.qdrant_url:
        vector_cfg["url"] = args.qdrant_url
    elif args.qdrant_path:
        vector_cfg["path"] = args.qdrant_path
        vector_cfg["on_disk"] = True
    else:
        raise ValueError("Either --qdrant-url or --qdrant-path must be provided.")

    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": args.memory_llm_model,
                "api_key": llm_api_key,
                "openai_base_url": llm_base_url,
                "temperature": 0.0,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": args.embed_model,
                "embedding_dims": args.embed_dims,
                "api_key": embed_api_key,
                "openai_base_url": embed_base_url,
            },
        },
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": args.neo4j_url,
                "username": args.neo4j_username,
                "password": args.neo4j_password,
                "database": args.neo4j_database,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": vector_cfg,
        },
        "history_db_path": args.history_db_path,
        "privacy_mapping_dir": args.privacy_mapping_dir,
    }


def _parse_response_models(csv_value: str) -> List[str]:
    out: List[str] = []
    for part in csv_value.split(","):
        token = part.strip()
        if token:
            out.append(token)
    return out


def _resolve_response_model_configs(keys: List[str]) -> List[ResponseModelConfig]:
    configs: List[ResponseModelConfig] = []
    for key in keys:
        lower = key.lower()

        if lower in {"llama3.1-8b", "llama-3.1-8b", "llama"}:
            model = _pick_env("SPMEM_LLAMA31_MODEL_ID", "RESPONSE_LLAMA_MODEL", default="llama-3.1-8b-instruct")
            base_url = _pick_env("SPMEM_LLAMA31_BASE_URL", "RESPONSE_LLAMA_BASE_URL")
            api_key = _pick_env("SPMEM_LLAMA31_API_KEY", "RESPONSE_LLAMA_API_KEY")
            configs.append(
                ResponseModelConfig(
                    name="llama3.1-8b",
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                )
            )
            continue

        if lower in {"qwen3-14b", "qwen-3-14b", "qwen3.5-9b", "qwen-3.5-9b", "qwen"}:
            model = _pick_env("SPMEM_QWEN3_MODEL_ID", "RESPONSE_QWEN_MODEL", default="qwen3-14b")
            base_url = _pick_env("SPMEM_QWEN3_BASE_URL", "RESPONSE_QWEN_BASE_URL")
            api_key = _pick_env("SPMEM_QWEN3_API_KEY", "RESPONSE_QWEN_API_KEY")
            configs.append(
                ResponseModelConfig(
                    name="qwen3-14b",
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                )
            )
            continue

        if lower in {"deepseek-v3.2", "deepseek-v32", "deepseek"}:
            model = _pick_env("SPMEM_DEEPSEEK32_MODEL_ID", "RESPONSE_DEEPSEEK_MODEL", default="deepseek-v3.2")
            base_url = _pick_env("SPMEM_DEEPSEEK32_BASE_URL", "RESPONSE_DEEPSEEK_BASE_URL")
            api_key = _pick_env("SPMEM_DEEPSEEK32_API_KEY", "RESPONSE_DEEPSEEK_API_KEY")
            configs.append(
                ResponseModelConfig(
                    name="deepseek-v3.2",
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                )
            )
            continue

        if lower in {"gpt5.2-chat", "gpt-5.2-chat", "gpt5.2"}:
            model = _pick_env("SPMEM_GPT52_MODEL_ID", "RESPONSE_GPT52_MODEL", default="gpt-5.2-chat")
            base_url = _pick_env("SPMEM_GPT52_BASE_URL", "RESPONSE_GPT52_BASE_URL", "OPENAI_BASE_URL", default="https://api.openai.com/v1")
            api_key = _pick_env("SPMEM_GPT52_API_KEY", "RESPONSE_GPT52_API_KEY", "OPENAI_API_KEY")
            configs.append(
                ResponseModelConfig(
                    name="gpt5.2-chat",
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                )
            )
            continue

        raise ValueError(
            f"Unknown response model key: {key}. "
            "Supported: llama3.1-8b, qwen3-14b, gpt5.2-chat, deepseek-v3.2"
        )

    return configs


def _build_row(
    *,
    sample: Dict[str, Any],
    sample_index: int,
    user_index: int,
    domain: str,
    user_id: str,
    response_model_name: str,
    evaluation_mode: str,
    request_text: str,
    answer_text: str,
    condition: str,
    consent_used: Optional[bool],
    status: str,
    latency_seconds: float,
    usage_snapshot: Dict[str, Any],
    include_efficiency: bool,
    error_message: str = "",
) -> Dict[str, Any]:
    if evaluation_mode not in EVAL_MODES:
        raise ValueError(f"Unknown evaluation mode: {evaluation_mode}")

    if evaluation_mode.startswith("mixed_"):
        mode = "mixed"
    elif evaluation_mode.startswith("privacy_"):
        mode = "privacy_only"
    else:
        mode = "non_privacy_only"
    scenario = _safe_str(sample.get("scenario", ""))
    query = _safe_str(sample.get("query", ""))
    privacy_entities = _normalize_list(sample.get("privacy_entities", []))
    pref_entities = _normalize_list(sample.get("preference_entities", []))

    sample_id = f"u{user_index:04d}_{sample_index:04d}_{response_model_name}_{evaluation_mode}"
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "user_index": user_index,
        "user_id": user_id,
        "domain": domain,
        "scenario": scenario,
        "query": query,
        "mode": mode,
        "evaluation_mode": evaluation_mode,
        "privacy_entities": privacy_entities,
        "pref_entities": pref_entities,
        "condition": condition,
        "model": response_model_name,
        "privacy_request": request_text,
        "answer": answer_text,
        # Keep compatibility with evaluation script which reads `response`.
        "response": answer_text,
        "status": status,
        "consent_used": consent_used,
        "error_message": error_message,
    }
    if include_efficiency:
        row["latency_seconds"] = round(float(latency_seconds), 6)
        row.update(usage_snapshot)
    return row


def _extract_required_entities_from_task_plan(task_plan: Any) -> List[str]:
    if not isinstance(task_plan, dict):
        return []
    values: List[str] = []
    for item in task_plan.get("required_entities", []):
        if isinstance(item, dict):
            entity = _safe_str(item.get("entity", "")).strip()
            if entity:
                values.append(entity)
    return values


def _log_mode_result(
    *,
    print_fn: Callable[[str], None],
    user_index: int,
    user_id: str,
    model_name: str,
    sample_index: int,
    evaluation_mode: str,
    consent_used: Optional[bool],
    required_entities: List[str],
    retrieved_memories_for_prompt: Any,
    answer_text: str,
    status: str,
    usage_snapshot: Dict[str, Any],
) -> None:
    if evaluation_mode.endswith("_allowed"):
        consent_label = "allowed"
    elif evaluation_mode.endswith("_denied"):
        consent_label = "denied"
    else:
        consent_label = "n/a"

    print_fn("-" * 80)
    print_fn(
        f"[MODE] user_{user_index:04d} user_id={user_id} model={model_name} "
        f"sample={sample_index} evaluation_mode={evaluation_mode} "
        f"consent={consent_label} consent_used={consent_used} status={status}"
    )
    print_fn(f"[REQUIRED_ENTITIES] {required_entities}")
    print_fn("[RETRIEVED_FOR_PROMPT]")
    print_fn(json.dumps(retrieved_memories_for_prompt, ensure_ascii=False, indent=2))
    token_block = {
        "prompt_tokens": usage_snapshot.get("prompt_tokens"),
        "completion_tokens": usage_snapshot.get("completion_tokens"),
        "total_tokens": usage_snapshot.get("total_tokens"),
        "embedding_total_tokens": usage_snapshot.get("embedding_total_tokens"),
        "consent_request_tokens": usage_snapshot.get("consent_request_tokens"),
        "pipeline_total_tokens": usage_snapshot.get("pipeline_total_tokens"),
    }
    print_fn(f"[TOKENS] {json.dumps(token_block, ensure_ascii=False)}")
    print_fn("[ANSWER]")
    print_fn(answer_text)


def _run_single_mode(
    *,
    agent: PrivacyAwareAgent,
    tracker: LLMUsageTracker,
    embedding_tracker: EmbeddingUsageTracker,
    query: str,
    user_id: str,
    evaluation_mode: str,
) -> Tuple[Dict[str, Any], float, Dict[str, Any], str, str, Optional[bool]]:
    tracker.reset()
    embedding_tracker.reset()
    started = time.perf_counter()

    if evaluation_mode == "mixed_denied":
        first = agent.ask(query=query, user_id=user_id)
        request_text = _safe_str(first.get("message", ""))
        if str(first.get("status", "")).strip().lower() == "awaiting_consent":
            sid = _safe_str(first.get("session_id", "")).strip()
            if not sid:
                raise ValueError("mixed_denied: awaiting_consent returned without session_id")
            result = agent.continue_with_consent(session_id=sid, consent=False)
        else:
            # Defensive fallback for unexpected non-privacy branch.
            result = first
        answer_text = _safe_str(result.get("answer") or result.get("message") or "")
        consent_used = False
    elif evaluation_mode == "mixed_allowed":
        first = agent.ask(query=query, user_id=user_id)
        request_text = _safe_str(first.get("message", ""))
        if str(first.get("status", "")).strip().lower() == "awaiting_consent":
            sid = _safe_str(first.get("session_id", "")).strip()
            if not sid:
                raise ValueError("mixed_allowed: awaiting_consent returned without session_id")
            result = agent.continue_with_consent(session_id=sid, consent=True)
        else:
            # Defensive fallback for unexpected non-privacy branch.
            result = first
        answer_text = _safe_str(result.get("answer") or result.get("message") or "")
        consent_used = True
    elif evaluation_mode == "privacy_denied":
        first = agent.ask(query=query, user_id=user_id)
        request_text = _safe_str(first.get("message", ""))
        if str(first.get("status", "")).strip().lower() == "awaiting_consent":
            sid = _safe_str(first.get("session_id", "")).strip()
            if not sid:
                raise ValueError("privacy_denied: awaiting_consent returned without session_id")
            result = agent.continue_with_consent(session_id=sid, consent=False)
        else:
            result = first
        answer_text = _safe_str(result.get("answer") or result.get("message") or "")
        consent_used = False
    elif evaluation_mode == "privacy_allowed":
        first = agent.ask(query=query, user_id=user_id)
        request_text = _safe_str(first.get("message", ""))
        if str(first.get("status", "")).strip().lower() == "awaiting_consent":
            sid = _safe_str(first.get("session_id", "")).strip()
            if not sid:
                raise ValueError("privacy_allowed: awaiting_consent returned without session_id")
            result = agent.continue_with_consent(session_id=sid, consent=True)
        else:
            result = first
        answer_text = _safe_str(result.get("answer") or result.get("message") or "")
        consent_used = True
    elif evaluation_mode == "non_privacy_only":
        result = agent.ask(query=query, user_id=user_id)
        request_text = ""
        answer_text = _safe_str(result.get("answer") or result.get("message") or "")
        consent_used = bool(result.get("consent", False)) if isinstance(result, dict) else None
    else:
        raise ValueError(f"Unknown evaluation_mode={evaluation_mode}")

    elapsed = time.perf_counter() - started
    usage_snapshot = _compose_usage_snapshot(tracker, embedding_tracker)
    status = _safe_str(result.get("status", ""))
    return result, elapsed, usage_snapshot, request_text, answer_text, consent_used


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate privacy-aware agent responses for evaluation.")
    parser.add_argument(
        "--test-file",
        required=True,
        help="JSONL test file, e.g. eval/user0_test.jsonl",
    )
    parser.add_argument(
        "--user-index",
        type=int,
        default=None,
        help="User index, used to map data/user_{index:04d}.json",
    )
    parser.add_argument(
        "--user-index-start",
        type=int,
        default=None,
        help="Start user index (inclusive) for batch generation.",
    )
    parser.add_argument(
        "--user-index-end",
        type=int,
        default=None,
        help="End user index (inclusive) for batch generation.",
    )
    parser.add_argument(
        "--domain",
        default="medical",
        help="Domain label written into response rows (medical / finance / ...).",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data" / "medical" / "histories"),
        help="Directory containing user_XXXX.json files.",
    )
    parser.add_argument(
        "--privacy-mapping-dir",
        default=str(PROJECT_ROOT / "outputs" / "private" / "privacy_mappings"),
        help=(
            "Source directory containing per-user privacy mapping jsonl files. "
            "The public implementation reads this directory directly; files are not copied."
        ),
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Output JSONL path. Default: eval_outputs/responses/<test_stem>_userXXXX_agent_multi_model.jsonl",
    )
    parser.add_argument(
        "--response-models",
        default="llama3.1-8b,qwen3-14b,gpt5.2-chat,deepseek-v3.2",
        help="Comma-separated model keys: llama3.1-8b,qwen3-14b,gpt5.2-chat,deepseek-v3.2",
    )
    parser.add_argument(
        "--response-timeout-seconds",
        type=float,
        default=120.0,
        help="Explicit timeout (seconds) for response model API calls.",
    )
    parser.add_argument(
        "--include-efficiency",
        action="store_true",
        help="Include latency/token fields in output rows.",
    )
    parser.add_argument(
        "--agent-debug",
        action="store_true",
        help="Enable verbose debug logs from privacy_aware_agent internals.",
    )
    parser.add_argument(
        "--verbose-records",
        action="store_true",
        help="Print retrieved memories and generated answers. Disabled by default because logs may contain private values.",
    )

    # Memory config (mem0 stays gpt-5.2-chat by default)
    parser.add_argument("--memory-llm-model", default=_pick_env("SPMEM_MEMORY_MODEL_ID", "MEMORY_LLM_MODEL", default="gpt-5.2-chat"))
    parser.add_argument("--embed-model", default=_pick_env("SPMEM_EMBEDDING_MODEL_ID", "EMBEDDING_MODEL", default="text-embedding-3-small"))
    parser.add_argument("--embed-dims", type=int, default=1536)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--qdrant-url", default="")
    parser.add_argument("--qdrant-path", default="")
    parser.add_argument("--history-db-path", required=True)
    parser.add_argument("--neo4j-url", default=_pick_env("SPMEM_NEO4J_URL", "NEO4J_URL", default="neo4j://localhost:7687"))
    parser.add_argument("--neo4j-username", default=_pick_env("SPMEM_NEO4J_USERNAME", "NEO4J_USERNAME", default="neo4j"))
    parser.add_argument("--neo4j-password", default=_pick_env("SPMEM_NEO4J_PASSWORD", "NEO4J_PASSWORD"))
    parser.add_argument("--neo4j-database", default=_pick_env("SPMEM_NEO4J_DATABASE", "NEO4J_DATABASE", default="neo4j"))

    args = parser.parse_args()

    if not args.neo4j_password:
        raise ValueError("Neo4j password is required via --neo4j-password or NEO4J_PASSWORD")

    test_file = Path(args.test_file).resolve()
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    data_dir = Path(args.data_dir).resolve()
    privacy_mapping_source_dir = Path(args.privacy_mapping_dir).resolve() if args.privacy_mapping_dir else None
    if privacy_mapping_source_dir is not None and not privacy_mapping_source_dir.exists():
        print(f"[WARN] privacy_mapping_dir not found: {privacy_mapping_source_dir}")
        privacy_mapping_source_dir = None
    user_indices = _resolve_user_indices(args)
    samples = _read_jsonl(test_file)
    if not samples:
        raise ValueError(f"Empty test file: {test_file}")

    split_output_by_user = len(user_indices) > 1
    output_file: Optional[Path] = None
    output_dir: Optional[Path] = None
    output_stem: str = ""

    if split_output_by_user:
        if args.output_file:
            out = Path(args.output_file).resolve()
            if out.suffix.lower() == ".jsonl":
                output_dir = out.parent
                output_stem = out.stem
            else:
                output_dir = out
                output_stem = f"{test_file.stem}_agent_multi_model"
        else:
            output_dir = PROJECT_ROOT / "eval_outputs" / "responses"
            output_stem = f"{test_file.stem}_agent_multi_model"
    else:
        if args.output_file:
            output_file = Path(args.output_file).resolve()
        else:
            output_name = f"{test_file.stem}_user{user_indices[0]:04d}_agent_multi_model.jsonl"
            output_file = PROJECT_ROOT / "eval_outputs" / "responses" / output_name

    memory_cfg = _build_memory_config(args)
    memory = Memory.from_config(memory_cfg)
    embedding_usage_tracker = _attach_embedding_usage_tracker(memory)
    agent_query_processor.set_debug(bool(args.agent_debug))

    requested_model_keys = _parse_response_models(args.response_models)
    model_cfgs = _resolve_response_model_configs(requested_model_keys)

    active_model_cfgs: List[ResponseModelConfig] = []
    for cfg in model_cfgs:
        if not cfg.base_url or not cfg.api_key:
            print(
                f"[SKIP_MODEL] {cfg.name}: missing base_url/api_key "
                f"(base_url={'set' if cfg.base_url else 'missing'}, api_key={'set' if cfg.api_key else 'missing'})"
            )
            continue
        active_model_cfgs.append(cfg)

    if not active_model_cfgs:
        raise ValueError("No active response models after env validation.")

    print("=" * 80)
    print("GENERATE AGENT RESPONSES")
    print("=" * 80)
    print("test_file:", test_file)
    print("samples:", len(samples))
    print("user_indices:", user_indices)
    print("users_total:", len(user_indices))
    print("domain:", args.domain)
    print("privacy_mapping_source_dir:", privacy_mapping_source_dir if privacy_mapping_source_dir else "disabled")
    print("mapping_lookup_dir:", privacy_mapping_source_dir if privacy_mapping_source_dir else "disabled")
    print("memory_llm_model:", args.memory_llm_model)
    print("response_models:")
    for cfg in active_model_cfgs:
        print(
            f"  - key={cfg.name} model={cfg.model} "
            f"base_url={cfg.base_url} api_key={'set' if cfg.api_key else 'missing'}"
        )
    print("response_timeout_seconds:", args.response_timeout_seconds)
    print("include_efficiency:", args.include_efficiency)
    print("agent_debug:", bool(args.agent_debug))
    print("embedding_usage_tracking:", embedding_usage_tracker.tracking_source)
    if split_output_by_user:
        print("output_split_by_user:", True)
        print("output_dir:", output_dir)
        print("output_stem:", output_stem)
    else:
        print("output_file:", output_file)
    print("=" * 80)

    rows: List[Dict[str, Any]] = []
    saved_files: List[Path] = []
    condition_name = "privacy_aware_agent"
    tasks_per_user = _count_response_tasks(samples) * len(active_model_cfgs)

    if not split_output_by_user and output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("", encoding="utf-8")
        saved_files.append(output_file)

    user_iter = user_indices
    if tqdm is not None:
        user_iter = tqdm(user_indices, desc="Users", unit="user")
    print_fn = _build_print_fn()

    for user_index in user_iter:
        user_id = _load_user_id(data_dir=data_dir, user_index=user_index)
        synced = _sync_user_privacy_mapping_file(
            user_id=user_id,
            source_dir=privacy_mapping_source_dir,
        )
        if not synced and privacy_mapping_source_dir is not None:
            print(
                f"[WARN] privacy mapping file not found for user_id={user_id} in "
                f"{privacy_mapping_source_dir}"
            )

        user_rows: List[Dict[str, Any]] = []
        user_output_file: Optional[Path] = None

        if split_output_by_user and output_dir is not None:
            user_output_file = output_dir / f"{output_stem}_user{user_index:04d}.jsonl"
            user_output_file.parent.mkdir(parents=True, exist_ok=True)
            user_output_file.write_text("", encoding="utf-8")
            saved_files.append(user_output_file)

        if tqdm is None:
            print(f"\n[USER] user_index={user_index} user_id={user_id}")
            if user_output_file is not None:
                print(f"[OUTPUT] {user_output_file}")
        else:
            try:
                user_iter.set_postfix_str(f"user_{user_index:04d}")
            except Exception:
                pass

        inner_pbar = None
        if tqdm is not None:
            inner_pbar = tqdm(
                total=tasks_per_user,
                desc=f"user_{user_index:04d}",
                unit="resp",
                leave=False,
            )

        for model_cfg in active_model_cfgs:
            if tqdm is None:
                print(f"\n[MODEL] {model_cfg.name} ({model_cfg.model})")

            llm_call, tracker = _build_tracked_llm_call(
                model=model_cfg.model,
                base_url=model_cfg.base_url,
                api_key=model_cfg.api_key,
                timeout_seconds=float(args.response_timeout_seconds),
                temperature=0.0,
            )
            agent = PrivacyAwareAgent(memory=memory, llm_call=llm_call)

            for idx, sample in enumerate(samples, start=1):
                query = _safe_str(sample.get("query", "")).strip()
                if not query:
                    continue

                sample_modes = _sample_modes_for_sample(sample)

                for evaluation_mode in sample_modes:
                    mode_started = time.perf_counter()
                    try:
                        mode_result, elapsed, usage_snapshot, request_text, answer_text, consent_used = _run_single_mode(
                            agent=agent,
                            tracker=tracker,
                            embedding_tracker=embedding_usage_tracker,
                            query=query,
                            user_id=user_id,
                            evaluation_mode=evaluation_mode,
                        )
                        status = "ok"
                        error_message = ""
                    except Exception as e:
                        mode_result = {}
                        elapsed = time.perf_counter() - mode_started
                        usage_snapshot = _compose_usage_snapshot(tracker, embedding_usage_tracker)
                        request_text = ""
                        answer_text = f"[ERROR] {e}"
                        consent_used = None
                        status = "error"
                        error_message = str(e)

                    usage_snapshot = _augment_usage_with_consent_tokens(
                        usage_snapshot=usage_snapshot,
                        request_text=request_text,
                    )

                    row = _build_row(
                        sample=sample,
                        sample_index=idx,
                        user_index=user_index,
                        domain=args.domain,
                        user_id=user_id,
                        response_model_name=model_cfg.name,
                        evaluation_mode=evaluation_mode,
                        request_text=request_text,
                        answer_text=answer_text,
                        condition=condition_name,
                        consent_used=consent_used,
                        status=status,
                        latency_seconds=elapsed,
                        usage_snapshot=usage_snapshot,
                        include_efficiency=bool(args.include_efficiency),
                        error_message=error_message,
                    )
                    rows.append(row)
                    user_rows.append(row)
                    if split_output_by_user and user_output_file is not None:
                        _append_jsonl_row(user_output_file, row)
                    elif not split_output_by_user and output_file is not None:
                        _append_jsonl_row(output_file, row)
                    if inner_pbar is not None:
                        inner_pbar.update(1)
                    required_entities = _extract_required_entities_from_task_plan(mode_result.get("task_plan"))
                    retrieved_for_prompt = mode_result.get("retrieved_memories_for_prompt", [])
                    if args.verbose_records:
                        _log_mode_result(
                            print_fn=print_fn,
                            user_index=user_index,
                            user_id=user_id,
                            model_name=model_cfg.name,
                            sample_index=idx,
                            evaluation_mode=evaluation_mode,
                            consent_used=consent_used,
                            required_entities=required_entities,
                            retrieved_memories_for_prompt=retrieved_for_prompt,
                            answer_text=answer_text,
                            status=status,
                            usage_snapshot=usage_snapshot,
                        )

                if tqdm is None and (idx % 10 == 0 or idx == len(samples)):
                    print(f"  processed {idx}/{len(samples)}")

        if inner_pbar is not None:
            inner_pbar.close()

        if split_output_by_user and user_output_file is not None and tqdm is None:
            print(f"[SAVED] user_{user_index:04d}: {user_output_file} rows={len(user_rows)}")

    print("\nDone.")
    if split_output_by_user:
        print("saved_files:", len(saved_files))
        if saved_files:
            print("first_saved:", saved_files[0])
            print("last_saved:", saved_files[-1])
    else:
        print("saved:", output_file)
    print("rows:", len(rows))


if __name__ == "__main__":
    main()
