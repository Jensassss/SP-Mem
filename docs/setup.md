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
- **Qdrant:** the response workflow runs concurrent child processes, so its write and generation commands use one reachable Qdrant service. Supply that service's URL through the environment or command line. The configuration-driven utility scripts can alternatively use a local persistent client path.
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

## 4. Storage paths

[`configs/paper.example.json`](../configs/paper.example.json) defines the default relative paths:

```text
runs/storage/qdrant
runs/storage/history.db
runs/storage/privacy_mappings
```

The original experiment drivers receive their Qdrant collection, history database, and private-mapping directory explicitly on the command line. The write and response commands must use the same values so vector records, graph records, history state, and private mappings remain aligned. The JSON configuration above is used by the alternative configuration-driven utilities. Generated stores and mappings are ignored by Git and must not be published.

Use `--collection-suffix` when you need an isolated Qdrant collection without changing the base configuration. Do not point concurrent experiments at the same mutable stores unless that sharing is intentional.

## 5. Validate data and create a selection

Replace every `<...>` placeholder in the following commands. Check the tracked data layout without calling external services:

```powershell
python scripts\validate_data.py --output-json <validation_report_json>
```

Create a reproducible selection manifest:

```powershell
python scripts\select_users.py `
  --domain <domain> `
  --num-users <number_of_users> `
  --seed <random_seed> `
  --output <selection_manifest>
```

`--num-users` applies to each selected domain. You can instead use repeatable `--user-id`, `--user-index`, or inclusive `--user-range START:END`. The manifest records the selected users and arguments for the configuration-driven utilities. The migrated original drivers accept an inclusive range directly; use and record the same range for all compared methods.

The paper reports evaluation on a 100-user subset, but its exact user IDs are not published in this repository. A newly generated selection must not be described as the original paper split.

## 6. Build and generate with the original orchestration

Build memory:

```powershell
python scripts\build_memories.py `
  --domain <domain> `
  --start-user <start_user> --end-user <end_user> `
  --max-concurrent-users <max_concurrent_users> `
  --collection-name <qdrant_collection> `
  --qdrant-url <qdrant_url> `
  --history-db-path <history_db_path> `
  --privacy-mapping-dir <privacy_mapping_dir>
```

Generate responses:

```powershell
python eval\run_batch_generate_responses.py `
  --domain <domain> `
  --start-user <start_user> --end-user <end_user> `
  --max-parallel <max_parallel_users> `
  --test-dir data/<domain>/evaluation_queries `
  --data-dir data/<domain>/histories `
  --response-model <response_model_key> `
  --output-tag <output_tag> `
  --collection-name <qdrant_collection> `
  --qdrant-url <qdrant_url> `
  --history-db-path <history_db_path> `
  --privacy-mapping-dir <privacy_mapping_dir> `
  --output-dir <response_output_dir>
```

The generator derives the five paper conditions from the query data: mixed allowed, mixed denied, privacy-only allowed, privacy-only denied, and preference-only (`non_privacy_only`). The writer keeps resumable progress in its log. The copied batch response driver preserves the original behavior of writing fresh per-user files; do not point a new run at outputs that must be retained.

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
