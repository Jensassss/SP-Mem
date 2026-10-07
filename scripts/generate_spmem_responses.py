"""Configuration-driven release utility, not the original experiment driver.

For the migrated parallel experiment workflow, use
``eval/run_batch_generate_responses.py`` and see ``docs/original_workflow.md``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.common import (
    add_selection_arguments,
    append_jsonl,
    build_memory_config,
    completed_sample_ids,
    ensure_empty_or_resume,
    load_paper_config,
    load_queries,
    modes_for_query,
    provider_settings,
    repo_path,
    save_selection_manifest,
    select_users,
    write_json,
)


class TrackedChatCall:
    """OpenAI-compatible chat callable that preserves missing-usage information."""

    def __init__(self, settings: Mapping[str, str]) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=settings["api_key"],
            base_url=settings["base_url"] or None,
        )
        self.model = settings["model"]
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.missing_usage_calls = 0
        self.observed_prompt_tokens = 0
        self.observed_completion_tokens = 0
        self.observed_total_tokens = 0

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        self.calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            self.missing_usage_calls += 1
        else:
            prompt = getattr(usage, "prompt_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
            total = getattr(usage, "total_tokens", None)
            if not all(isinstance(value, int) for value in (prompt, completion, total)):
                self.missing_usage_calls += 1
            else:
                self.observed_prompt_tokens += prompt
                self.observed_completion_tokens += completion
                self.observed_total_tokens += total
        return response.choices[0].message.content or ""

    def snapshot(self) -> Dict[str, Any]:
        complete = self.calls > 0 and self.missing_usage_calls == 0
        return {
            "usage_available": complete,
            "llm_calls": self.calls,
            "usage_missing_calls": self.missing_usage_calls,
            "prompt_tokens": self.observed_prompt_tokens if complete else None,
            "completion_tokens": self.observed_completion_tokens if complete else None,
            "total_tokens": self.observed_total_tokens if complete else None,
            "observed_prompt_tokens": self.observed_prompt_tokens,
            "observed_completion_tokens": self.observed_completion_tokens,
            "observed_total_tokens": self.observed_total_tokens,
        }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Generate SP-Mem responses for the five paper evaluation conditions."
    )
    value.add_argument("--config", default="configs/paper.example.json")
    value.add_argument("--model", action="append", required=True, help="Response model key; repeatable.")
    value.add_argument("--output-dir", required=True)
    value.add_argument(
        "--retrieval-mode",
        choices=("hybrid", "vector_only", "graph_only"),
        default="hybrid",
        help="Paper method or one of the reported retrieval ablations.",
    )
    value.add_argument("--collection-suffix", default="")
    value.add_argument("--resume", action="store_true")
    add_selection_arguments(value)
    return value


def _run_condition(
    agent: Any,
    *,
    user_id: str,
    query: str,
    mode: str,
) -> tuple[Dict[str, Any], str, bool | None]:
    first = agent.ask(query=query, user_id=user_id)
    request = ""
    if str(first.get("status", "")) == "awaiting_consent":
        request = str(first.get("message", ""))
        consent = mode.endswith("_allowed")
        return agent.continue_with_consent(first["session_id"], consent), request, consent
    return first, request, None


def main() -> None:
    args = parser().parse_args()
    config = load_paper_config(args.config)
    unknown = [name for name in args.model if name not in config["response_models"]]
    if unknown:
        raise ValueError(
            f"Unknown response model keys {unknown}; configured: {sorted(config['response_models'])}"
        )
    users = select_users(args)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_selection_manifest(output_dir / "selection.json", users, arguments=vars(args))
    os.environ["PRIVACY_AGENT_RETRIEVAL_MODE"] = args.retrieval_mode

    memory_config = build_memory_config(config, collection_suffix=args.collection_suffix)
    write_json(output_dir / "resolved_run_config.json", {
        "method": "SP-Mem",
        "retrieval_mode": args.retrieval_mode,
        "response_models": [config["response_models"][name]["display_name"] for name in args.model],
        "storage": {
            "qdrant_path": memory_config["vector_store"]["config"]["path"],
            "qdrant_collection": memory_config["vector_store"]["config"]["collection_name"],
            "history_db_path": memory_config["history_db_path"],
            "privacy_mapping_dir": memory_config["privacy_mapping_dir"],
        },
        "note": "Provider model IDs and credentials are not persisted.",
    })

    from spmem_agent import PrivacyAwareAgent
    from spmem_memory import Memory

    memory = Memory.from_config(memory_config)
    for model_key in args.model:
        settings = provider_settings(config["response_models"][model_key])
        llm_call = TrackedChatCall(settings)
        agent = PrivacyAwareAgent(memory=memory, llm_call=llm_call)
        for user in users:
            output_path = output_dir / model_key / user.domain / f"user_{user.user_index:04d}.jsonl"
            ensure_empty_or_resume(output_path, resume=args.resume)
            completed = completed_sample_ids(output_path) if args.resume else set()
            for query_index, sample in enumerate(load_queries(user)):
                query = str(sample.get("query", "")).strip()
                if not query:
                    continue
                for mode in modes_for_query(sample):
                    sample_id = f"{user.domain}_u{user.user_index:04d}_q{query_index:04d}_{mode}"
                    if sample_id in completed:
                        continue
                    llm_call.reset()
                    started = time.perf_counter()
                    try:
                        result, privacy_request, consent_used = _run_condition(
                            agent,
                            user_id=user.user_id,
                            query=query,
                            mode=mode,
                        )
                        answer = str(result.get("answer") or result.get("message") or "")
                        status = "ok" if result.get("status") == "answered" else str(result.get("status", "error"))
                        error = ""
                    except Exception as exc:
                        answer = ""
                        privacy_request = ""
                        consent_used = mode.endswith("_allowed") if mode != "non_privacy_only" else None
                        status = "error"
                        error = f"{type(exc).__name__}: {exc}"
                    row: Dict[str, Any] = {
                        "sample_id": sample_id,
                        "user_index": user.user_index,
                        "user_id": user.user_id,
                        "domain": user.domain,
                        "scenario": sample.get("scenario", ""),
                        "query": query,
                        "mode": "mixed" if mode.startswith("mixed_") else (
                            "privacy_only" if mode.startswith("privacy_") else "non_privacy_only"
                        ),
                        "evaluation_mode": mode,
                        "privacy_entities": list(sample.get("privacy_entities", [])),
                        "pref_entities": list(sample.get("preference_entities", [])),
                        "condition": "allowed" if mode.endswith("_allowed") else (
                            "denied" if mode.endswith("_denied") else "not_applicable"
                        ),
                        "method": "SP-Mem",
                        "retrieval_mode": args.retrieval_mode,
                        "model": settings["display_name"],
                        "privacy_request": privacy_request,
                        "answer": answer,
                        "response": answer,
                        "status": status,
                        "consent_used": consent_used,
                        "latency_seconds": round(time.perf_counter() - started, 6),
                        "error_message": error,
                    }
                    row.update(llm_call.snapshot())
                    append_jsonl(output_path, row)
                    if status == "error":
                        print(f"error: {sample_id}: {error}", file=sys.stderr)
            print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
