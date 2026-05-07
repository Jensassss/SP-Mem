# SP-Mem Artifact

This anonymized artifact contains the SP-Mem implementation, synthetic benchmark inputs, and evaluation scripts used to support the submitted paper. The package is prepared for confidential review and does not contain author-identifying information.

## What Is Included

- `src/spmem_memory/`: privacy-aware memory writing, partitioned storage, vector retrieval, graph retrieval, and private-value lookup utilities.
- `src/spmem_agent/`: query analysis, consent handling, private-value hydration, retrieval orchestration, and response generation logic.
- `eval/`: evaluation scripts for pairwise P-TC/P-PQ judging and rule-based exact-match UPU scoring.
- `data/`: synthetic benchmark inputs, including preference profiles, privacy profiles, user-assistant history dialogues, and evaluation queries.
- `.env.example`: example environment variable names with empty values.
- `requirements.txt`: Python dependencies for the core artifact and evaluation scripts.

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

Expose the local source tree before running examples or scripts:

```bash
export PYTHONPATH=$PWD/src
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

Configure provider keys through environment variables when running memory construction, response generation, or judge-based evaluation. For example:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export NEO4J_PASSWORD=...
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_BASE_URL = "..."
$env:NEO4J_PASSWORD = "..."
```

UPU exact-match scoring does not require an LLM API key.

## Data

The benchmark data are synthetic. Each domain under `data/` contains:

- `profiles/preference_profiles.jsonl`: non-private user preference inventory.
- `profiles/privacy_profiles.jsonl`: private-information inventory used for privacy-aware memory writing and UPU evaluation.
- `histories/`: user-assistant history dialogues used for memory construction.
- `evaluation_queries/`: evaluation queries with task scenario, required preference entities, and required privacy entities.

See `data/README.md` and `data/DATA_MANIFEST.json` for the full layout and counts.

## Memory Writing

SP-Mem separates memory construction from query-time response generation. During memory writing, user-assistant histories are converted into privacy-aware memory entries. Non-private preferences remain searchable for personalization. Private values are detected and represented through sanitized evidence plus privacy references, so exact values are not directly exposed through ordinary retrieval and can be hydrated only through the consent-aware path at query time.

A minimal memory client can be created as follows:

```python
import os
from pathlib import Path

from spmem_memory import Memory

neo4j_password = os.getenv("NEO4J_PASSWORD")
if not neo4j_password:
    raise RuntimeError("Please set NEO4J_PASSWORD before constructing the memory client.")

memory_config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-5.2-chat",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "openai_base_url": os.getenv("OPENAI_BASE_URL"),
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "embedding_dims": 1536,
            "api_key": os.getenv("OPENAI_API_KEY"),
            "openai_base_url": os.getenv("OPENAI_BASE_URL"),
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "neo4j://localhost:7687",
            "username": "neo4j",
            "password": neo4j_password,
            "database": "neo4j",
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "spmem_demo",
            "path": str(Path("./artifacts/qdrant_demo").resolve()),
            "embedding_model_dims": 1536,
            "on_disk": True,
        },
    },
    "history_db_path": str(Path("./artifacts/history_demo.db").resolve()),
}

memory = Memory.from_config(memory_config)
```

To write a user's history into memory:

```python
messages = [
    {"role": "user", "content": "I prefer concise explanations. My phone number is +1-555-0100."},
    {"role": "assistant", "content": "I will keep your communication preference in mind."},
]

memory.add(messages, user_id="demo_user")
```

The same API can be used to reconstruct memory from `data/<domain>/histories/user_XXXX.json`: load each history file, convert each dialogue turn into `{role, content}` messages, and call `memory.add(..., user_id=<user_id>)`. The resulting vector, graph, and SQLite stores are generated artifacts and are intentionally not included in this submission.

## Query-Time Agent

The agent uses the written memory to answer evaluation queries while enforcing consent behavior:

1. Analyze the user query and infer required preference and private entities.
2. Retrieve sanitized memory evidence from vector and graph memory.
3. If exact private information is required, return an `awaiting_consent` state before final generation.
4. If consent is denied, answer using sanitized memory only.
5. If consent is granted, hydrate exact private values through stored privacy references and then generate the final answer.

A minimal agent flow is:

```python
from spmem_agent.agent import PrivacyAwareAgent
from spmem_agent.llm import build_openai_llm_call

llm_call = build_openai_llm_call(
    model="gpt-5.2-chat",
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.0,
)

agent = PrivacyAwareAgent(memory=memory, llm_call=llm_call)

first = agent.ask(
    query="Help me draft an email using my preference and private account information.",
    user_id="demo_user",
)

if first.get("status") == "awaiting_consent":
    final = agent.continue_with_consent(
        session_id=first["session_id"],
        consent=True,
    )
else:
    final = first
```

A longer runnable example is provided in `src/spmem_agent/example_usage.py`.

## End-to-End Reproduction Flow

A reviewer can reproduce the main pipeline with the following high-level steps:

1. Install dependencies and set `PYTHONPATH` as described above.
2. Configure the LLM provider and memory backends.
3. Build a `Memory` instance with `Memory.from_config(...)`.
4. Reconstruct user memories by writing histories from `data/<domain>/histories/` with `memory.add(...)`.
5. Run `PrivacyAwareAgent.ask(...)` and `continue_with_consent(...)` on queries from `data/<domain>/evaluation_queries/` to generate response files.
6. Compare SP-Mem responses with baseline responses using pairwise P-TC/P-PQ evaluation.
7. Compute UPU with the rule-based exact-match scorer against the synthetic privacy profiles.

## Evaluation

Pairwise P-TC/P-PQ evaluation can be run on a matched pair of response files with:

```bash
python eval/evaluate_pairwise_tc_pq.py \
  --response-file-a <spmem_response_file.jsonl> \
  --response-file-b <baseline_response_file.jsonl> \
  --output-dir <pairwise_output_dir> \
  --preference-profile-file data/<domain>/profiles/preference_profiles.jsonl \
  --name-a SP-Mem \
  --name-b <baseline_name> \
  --strict-pairing
```

UPU evaluation can be run without an LLM judge:

```bash
python eval/score_upu_exact_match.py \
  --responses <response_file_or_dir> \
  --glob "user*.jsonl" \
  --privacy-profile-file data/<domain>/profiles/privacy_profiles.jsonl \
  --output-jsonl <upu_scored_rows.jsonl> \
  --summary-json <upu_summary.json>
```

Response files are expected to be JSONL files containing generated answers and enough identifiers to match users and evaluation tasks. The UPU scorer uses the privacy profile inventory and reports UPU for tasks where exact private values should not appear.

## Sanity Checks

A basic syntax check can be run with:

```bash
python -m compileall src eval
```

The evaluation entry points can be inspected with:

```bash
python eval/evaluate_pairwise_tc_pq.py --help
python eval/score_upu_exact_match.py --help
```

## License and Access

This anonymized artifact is provided for confidential review only. The released code and synthetic data are intended solely for reproducing and evaluating the submitted paper during the review period. A de-anonymized public release, including final license and access information, will be provided upon acceptance.

The synthetic benchmark data do not contain real user data or scraped personal records. All example credentials in this repository are placeholders; users should configure API keys and service passwords through environment variables.
