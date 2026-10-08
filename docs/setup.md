# Setup and service configuration

This guide expands the installation notes in the repository [README](../README.md). Run every command from the repository root. Examples use PowerShell.

## 1. Python environment

The offline checks for this repository were run with Python 3.12 and the versions recorded in `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

Install optional baseline packages only if you plan to run Mem0, Zep, or MemOS:

```powershell
python -m pip install -r requirements-baselines.txt
```

The optional baseline versions are not pinned because the historical private environment versions are not evidenced. The Full-context baseline uses only the core dependencies.

## 2. Required services

A complete SP-Mem build and inference run uses all of the following:

- **Chat model endpoint:** an OpenAI-compatible endpoint for memory writing and another configured response-model entry. They may point to the same service.
- **Embedding endpoint:** an OpenAI-compatible embedding endpoint whose output dimension matches the configured `dimensions` value (1536 by default).
- **Neo4j:** a reachable Neo4j database for graph memory.
- **Qdrant:** all paper workflow and retrieval-ablation commands use the same reachable Qdrant service. Supply its URL through `SPMEM_QDRANT_URL` or the batch command line.
- **Local state:** writable paths for the history database and separate private-value mappings.

SP-Mem does not start Neo4j or model services. Start and verify those services using their own documentation before running a build.

## 3. Model and service variables

The scripts read environment variables from the current process. They do **not** automatically load an `.env` file. [`.env.example`](../.env.example) lists every supported name without credentials.

Set the values needed for memory construction:

```powershell
$env:SPMEM_MEMORY_MODEL_ID = "<memory_model_id>"
$env:SPMEM_MEMORY_API_KEY = "<memory_api_key>"
$env:SPMEM_MEMORY_BASE_URL = "<memory_base_url>"

$env:SPMEM_EMBEDDING_MODEL_ID = "<embedding_model_id>"
$env:SPMEM_EMBEDDING_API_KEY = "<embedding_api_key>"
$env:SPMEM_EMBEDDING_BASE_URL = "<embedding_base_url>"

$env:SPMEM_NEO4J_URL = "<neo4j_url>"
$env:SPMEM_NEO4J_USERNAME = "<neo4j_username>"
$env:SPMEM_NEO4J_PASSWORD = "<neo4j_password>"
$env:SPMEM_NEO4J_DATABASE = "<neo4j_database>"
$env:SPMEM_QDRANT_URL = "<qdrant_url>"
```

If a provider uses its standard OpenAI endpoint, its base URL may be left unset. Model IDs are provider-specific: the paper-facing name and a provider deployment ID are deliberately recorded separately. Do not assume that a provider exposes an ID identical to `GPT-5.2-Chat`.

For the response model used in the README example:

```powershell
$env:SPMEM_GPT52_MODEL_ID = "<response_model_id>"
$env:SPMEM_GPT52_API_KEY = "<response_api_key>"
$env:SPMEM_GPT52_BASE_URL = "<response_base_url>"
```

Equivalent variables for the other paper model labels are grouped in `.env.example`:

| Config key | Paper-facing label | Variable prefix |
| --- | --- | --- |
| `gpt-5.2-chat` | GPT-5.2-Chat | `SPMEM_GPT52_` |
| `llama-3.1-8b-instruct` | Llama-3.1-8B-Instruct | `SPMEM_LLAMA31_` |
| `qwen3-14b` | Qwen3-14B | `SPMEM_QWEN3_` |
| `deepseek-v3.2` | DeepSeek-V3.2 | `SPMEM_DEEPSEEK32_` |

Pairwise P-TC/P-PQ evaluation also requires the judge variables under `SPMEM_JUDGE_`. The paper-facing judge label in the example configuration is GPT-4.1; set `SPMEM_JUDGE_MODEL_ID` to the actual model ID supplied by your endpoint.

Do not add secrets to JSON configuration files or commit local credential files.

## 4. Shared storage configuration

[`configs/paper.example.json`](../configs/paper.example.json) uses the same storage identifiers as the README example:

```text
Qdrant service:     SPMEM_QDRANT_URL
Qdrant collection:  spmem_paper
runs/storage/history.db
runs/storage/privacy_mappings
```

The write command, batch response generator, and configuration-driven ablation utility must use these same values so vector records, graph records, history state, and private mappings remain aligned. A custom configuration may use `storage.qdrant_path` instead of `storage.qdrant_url_env` for a local Qdrant database, but it cannot read records previously written to the service.

Use `--collection-suffix` only when memory was built into the matching suffixed collection. Do not point concurrent experiments at the same mutable stores unless that sharing is intentional.

## 5. Validate data and create a selection

Replace every `<...>` placeholder in the following commands. Check the tracked data layout without calling external services:

```powershell
python scripts\validate_data.py --output-json <validation_report_json>
```

Create a reproducible selection manifest:

```powershell
python scripts\select_users.py `
  --domain <domain> `
  --history-root <downloaded_history_root> `
  --num-users <number_of_users> `
  --seed <random_seed> `
  --output <selection_manifest>
```

`--num-users` applies to each selected domain. You can instead use repeatable `--user-id`, `--user-index`, or inclusive `--user-range START:END`. The manifest records the selected users and arguments for the configuration-driven utilities. The batch drivers accept an inclusive range directly; use and record the same range for all compared methods.

The paper reports evaluation on a 100-user subset, but its exact user IDs are not published in this repository. A newly generated selection must not be described as the original paper split.

## 6. Build and generate

Download the conversation histories from the [Hugging Face dataset](https://huggingface.co/datasets/wwj95/privacy-aware-memory-benchmark). The GitHub repository stores only the evaluation queries; pass the downloaded domain directory through `--data-dir`.

Build memory:

```powershell
python scripts\build_memories.py `
  --domain <domain> `
  --data-dir <downloaded_history_dir> `
  --start-user <start_user> --end-user <end_user> `
  --max-concurrent-users <max_concurrent_users> `
  --collection-name spmem_paper `
  --qdrant-url $env:SPMEM_QDRANT_URL `
  --history-db-path runs/storage/history.db `
  --privacy-mapping-dir runs/storage/privacy_mappings
```

Generate responses:

```powershell
python eval\run_batch_generate_responses.py `
  --domain <domain> `
  --start-user <start_user> --end-user <end_user> `
  --max-parallel <max_parallel_users> `
  --test-dir data/<domain>/evaluation_queries `
  --data-dir <downloaded_history_dir> `
  --response-model <response_model_key> `
  --output-tag <output_tag> `
  --collection-name spmem_paper `
  --qdrant-url $env:SPMEM_QDRANT_URL `
  --history-db-path runs/storage/history.db `
  --privacy-mapping-dir runs/storage/privacy_mappings `
  --output-dir <response_output_dir>
```

The generator derives the five paper conditions from the query data: mixed allowed, mixed denied, privacy-only allowed, privacy-only denied, and preference-only (`non_privacy_only`). The writer keeps resumable progress in its log. The batch response driver writes fresh per-user files; do not point a new run at outputs that must be retained.

## 7. Static checks

These checks do not call paid APIs or write model/database results:

```powershell
python -m compileall -q src scripts eval
python scripts\build_memories.py --help
python eval\generate_agent_responses.py --help
python eval\run_batch_generate_responses.py --help
python eval\run_batch_pairwise_evaluation.py --help
python scripts\generate_spmem_responses.py --help
python scripts\generate_baseline_responses.py --help
```

Passing these static checks does not verify external endpoints, live Neo4j writes, optional baseline SDK behavior, or reproduction of paper numbers. See [original_workflow.md](original_workflow.md) for the documented adaptation and verification boundary.
