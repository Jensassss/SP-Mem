# What to Remember, What to Reveal: Privacy-Aware Memory for Conversational Agents

<p align="center">
  <a href="https://arxiv.org/abs/2608.16551"><img height="24" alt="arXiv" src="https://img.shields.io/badge/arXiv-2608.16551-b31b1b.svg?logo=arxiv&amp;logoColor=white"></a>
  <a href="https://arxiv.org/pdf/2608.16551"><img height="24" alt="Paper PDF" src="https://img.shields.io/badge/Paper-PDF-4b5563.svg?logo=adobeacrobatreader&amp;logoColor=white"></a>
  <a href="https://huggingface.co/datasets/wwj95/privacy-aware-memory-benchmark"><img height="24" alt="Hugging Face dataset" src="https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E.svg?logo=huggingface&amp;logoColor=black"></a>
</p>

<p align="center">
  <a href="#sp-mem-overview">Overview</a> |
  <a href="#getting-started">Getting Started</a> |
  <a href="#running-sp-mem">Running SP-Mem</a> |
  <a href="#dataset">Dataset</a> |
  <a href="#evaluation">Evaluation</a> |
  <a href="#citation">Citation</a>
</p>

SP-Mem equips conversational agents with memory that separates searchable sanitized content from exact private values, which are restored only when needed for a task and permitted by the user.

## SP-Mem Overview

<p align="center">
  <a href="assets/spmem_overview.svg">
    <img src="assets/spmem_overview.svg" width="1000" alt="SP-Mem framework with privacy-aware memory writing, partitioned vector and graph memory, separate exact-value storage, and consent-gated query-time retrieval.">
  </a>
</p>

During memory writing, SP-Mem extracts and sanitizes facts and relations, stores them in vector and graph memory, and retains mappings to separately stored private values. At query time, it retrieves relevant memories and requests consent when exact values are needed. With consent, only the required values are restored; otherwise, the agent responds using sanitized memory.

## Getting Started

Run all commands from the repository root. The core workflow requires:

- **Python** and the packages in `requirements.txt` (tested with Python 3.12);
- a running **Qdrant service** for vector memory;
- a running **Neo4j service** for graph memory; and
- OpenAI-compatible **chat and embedding endpoints** for memory construction and response generation.

The examples below use PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

Set the required model, embedding, Qdrant, and Neo4j variables in the current process using [`.env.example`](.env.example) as a reference. The scripts do **not** automatically load an `.env` file.

See [docs/setup.md](docs/setup.md) for the complete service configuration, environment-variable list, storage behavior, and optional baseline dependencies.

## Running SP-Mem

The following minimal example builds memory and generates responses for **Education user 0**. It is a smoke-test selection, not the paper's reported evaluation split. Configure the services and credentials above first, then run both commands from the repository root. The storage values match [`configs/paper.example.json`](configs/paper.example.json), so the same memory can also be used by the retrieval-ablation commands in the [evaluation guide](docs/evaluation.md).

Conversation histories are hosted in the [Privacy-Aware Memory Benchmark on Hugging Face](https://huggingface.co/datasets/wwj95/privacy-aware-memory-benchmark). Download them once and set the history directory used by the commands below:

```powershell
git clone https://huggingface.co/datasets/wwj95/privacy-aware-memory-benchmark.git ..\privacy-aware-memory-benchmark
$historyDir = (Resolve-Path ..\privacy-aware-memory-benchmark\education).Path
```

### 1. Build memory

This command reads `user_0000.json` from the downloaded Education histories and writes its memory:

```powershell
python scripts\build_memories.py `
  --domain education `
  --data-dir $historyDir `
  --start-user 0 --end-user 0 `
  --batch-size 1 `
  --max-concurrent-users 1 `
  --collection-name spmem_paper `
  --qdrant-url $env:SPMEM_QDRANT_URL `
  --history-db-path runs/storage/history.db `
  --privacy-mapping-dir runs/storage/privacy_mappings `
  --output-log-file runs/logs/write_education_user0.log `
  --user-done-log-file runs/logs/write_education_user0.done
```

This writes vector records to the `spmem_paper` Qdrant collection, graph records to the configured Neo4j database, local history state to `runs/storage/history.db`, and exact private-value mappings to `runs/storage/privacy_mappings`.

### 2. Generate responses

This command reads the selected users' evaluation queries and histories, reuses the memory written above, and generates responses for the same user range:

```powershell
python eval\run_batch_generate_responses.py `
  --domain education `
  --start-user 0 --end-user 0 `
  --max-parallel 1 `
  --test-dir data/education/evaluation_queries `
  --data-dir $historyDir `
  --response-model gpt-5.2-chat `
  --output-tag gpt52chat `
  --collection-name spmem_paper `
  --qdrant-url $env:SPMEM_QDRANT_URL `
  --history-db-path runs/storage/history.db `
  --privacy-mapping-dir runs/storage/privacy_mappings `
  --output-dir runs/responses/spmem/gpt52/education
```

Per-user JSONL responses are written under `runs/responses/spmem/gpt52/education`, with run logs in its `logs_gpt52chat` subdirectory. These JSONL files are inputs to the evaluation workflow. Model calls may incur provider costs.

<details>
<summary>Bash equivalent</summary>

```bash
git clone https://huggingface.co/datasets/wwj95/privacy-aware-memory-benchmark.git ../privacy-aware-memory-benchmark
history_dir=../privacy-aware-memory-benchmark/education

python scripts/build_memories.py \
  --domain education \
  --data-dir "$history_dir" \
  --start-user 0 --end-user 0 \
  --batch-size 1 \
  --max-concurrent-users 1 \
  --collection-name spmem_paper \
  --qdrant-url "$SPMEM_QDRANT_URL" \
  --history-db-path runs/storage/history.db \
  --privacy-mapping-dir runs/storage/privacy_mappings \
  --output-log-file runs/logs/write_education_user0.log \
  --user-done-log-file runs/logs/write_education_user0.done

python eval/run_batch_generate_responses.py \
  --domain education \
  --start-user 0 --end-user 0 \
  --max-parallel 1 \
  --test-dir data/education/evaluation_queries \
  --data-dir "$history_dir" \
  --response-model gpt-5.2-chat \
  --output-tag gpt52chat \
  --collection-name spmem_paper \
  --qdrant-url "$SPMEM_QDRANT_URL" \
  --history-db-path runs/storage/history.db \
  --privacy-mapping-dir runs/storage/privacy_mappings \
  --output-dir runs/responses/spmem/gpt52/education
```

</details>

See the [workflow notes](docs/original_workflow.md) for concurrency, retry behavior, and additional run controls.

## Dataset

The [Privacy-Aware Memory Benchmark conversation histories](https://huggingface.co/datasets/wwj95/privacy-aware-memory-benchmark) contain synthetic multi-turn data for **1,000 users** across **Finance**, **Medical**, **Education**, and **Mental Support**.

This GitHub repository retains only the **test queries** used for evaluation. They are stored under [`data/<domain>/evaluation_queries/`](data/) as one JSONL file per user. See [`data/README.md`](data/README.md) for paths, fields, and the mapping between query files and Hugging Face history records.

## Evaluation

Evaluation and aggregation utilities are provided in [`eval/`](eval/) and [`scripts/`](scripts/). See the [evaluation guide](docs/evaluation.md) for scoring generated responses, running baseline comparisons, and retrieval ablations.

## Citation

```bibtex
@article{wang2026remember,
  title         = {What to Remember, What to Reveal: Privacy-Aware Memory for Conversational Agents},
  author        = {Wang, Wenjie and Si, Wenhe and Xu, Xinyue and Xu, Yue},
  year          = {2026},
  eprint        = {2608.16551},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  url           = {https://arxiv.org/abs/2608.16551}
}
```

## Acknowledgements and License

Parts of the memory implementation are adapted from [Mem0](https://github.com/mem0ai/mem0). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the bundled [Apache 2.0 license text](LICENSES/Apache-2.0.txt) for source and attribution details.

Repository license terms are provided in [LICENSE](LICENSE).
