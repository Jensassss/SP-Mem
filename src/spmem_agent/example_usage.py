from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Adjust if your local path differs.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from spmem_memory import Memory

from spmem_agent.agent import PrivacyAwareAgent
from spmem_agent.llm import build_openai_llm_call


def main() -> None:
    neo4j_password = os.getenv("SPMEM_NEO4J_PASSWORD")
    if not neo4j_password:
        raise RuntimeError("Please set SPMEM_NEO4J_PASSWORD before constructing the memory client.")

    # 1) Build your memory client (example config only).
    memory_config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": os.environ["SPMEM_MEMORY_MODEL_ID"],
                "api_key": os.environ["SPMEM_MEMORY_API_KEY"],
                "openai_base_url": os.getenv("SPMEM_MEMORY_BASE_URL"),
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": os.environ["SPMEM_EMBEDDING_MODEL_ID"],
                "embedding_dims": 1536,
                "openai_base_url": os.getenv("SPMEM_EMBEDDING_BASE_URL"),
                "api_key": os.environ["SPMEM_EMBEDDING_API_KEY"],
            },
        },
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": os.getenv("SPMEM_NEO4J_URL", "neo4j://127.0.0.1:7687"),
                "username": os.getenv("SPMEM_NEO4J_USERNAME", "neo4j"),
                "password": neo4j_password,
                "database": os.getenv("SPMEM_NEO4J_DATABASE", "neo4j"),
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "privacy_agent_demo_small",
                "path": str(Path("./artifacts/qdrant_demo").resolve()),
                "embedding_model_dims": 1536,
                "on_disk": True,
            },
        },
        "history_db_path": str(Path("./artifacts/history_demo.db").resolve()),
        "privacy_mapping_dir": str(Path("./artifacts/privacy_mappings").resolve()),
    }
    memory = Memory.from_config(memory_config)

    # 2) Build llm_call for analysis + answer.
    llm_call = build_openai_llm_call(
        model=os.environ["SPMEM_GPT52_MODEL_ID"],
        api_key=os.environ["SPMEM_GPT52_API_KEY"],
        base_url=os.getenv("SPMEM_GPT52_BASE_URL"),
        temperature=0.0,
    )

    # 3) Create agent.
    agent = PrivacyAwareAgent(memory=memory, llm_call=llm_call)

    user_id = "demo_user"
    query = "Help me draft an email to my advisor with my loan amount and investment return."

    # 4) Step 1: ask.
    first = agent.ask(query=query, user_id=user_id)
    print("\n[STEP 1]")
    print(json.dumps(first, ensure_ascii=False, indent=2))

    # 5) Step 2: make the paper's single consent decision.
    if first.get("status") == "awaiting_consent":
        session_id = first["session_id"]

        # Example: user agrees.
        final = agent.continue_with_consent(session_id=session_id, consent=True)
        print("\n[STEP 2: consent=true]")
        print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
