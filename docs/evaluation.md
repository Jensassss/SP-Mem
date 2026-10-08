# Evaluation workflows

This document contains the detailed experiment commands kept out of the project homepage. Run all commands from the repository root in the configured Python environment. Examples use PowerShell; replace every `<...>` placeholder before running them.

The main SP-Mem write, response-generation, and judge entry points are described below. See [original_workflow.md](original_workflow.md) for implementation provenance and concurrency details. The repository does not claim a fresh end-to-end reproduction without running the required external services.

## Data validation and optional selection manifests

Validate the tracked data and save one user selection:

```powershell
python scripts\validate_data.py --output-json <validation_report_json>

python scripts\select_users.py `
  --domain <domain> `
  --history-root <downloaded_history_root> `
  --num-users <number_of_users> --seed <random_seed> `
  --output <selection_manifest>
```

The configuration-driven utilities under `scripts/` consume this manifest. The batch drivers below instead accept an explicit inclusive user range; record the command and use the same range and query files for every method in a comparison. See [setup.md](setup.md) for model, Neo4j, Qdrant, and path configuration.

Conversation histories are distributed through the [Hugging Face dataset](https://huggingface.co/datasets/wwj95/privacy-aware-memory-benchmark); this GitHub repository contains the test queries only. Commands that require exact privacy or preference profile references accept explicit local file paths, but those profile files are not distributed here.

## SP-Mem experiment flow

The writer schedules users concurrently with `asyncio`, limits active users with a semaphore, and processes each user's dialogue batches in order:

```powershell
python scripts\build_memories.py `
  --domain <domain> `
  --data-dir <downloaded_history_dir> `
  --start-user <start_user> --end-user <end_user> `
  --batch-size <batch_size> `
  --max-concurrent-users <max_concurrent_users> `
  --collection-name spmem_paper `
  --qdrant-url $env:SPMEM_QDRANT_URL `
  --history-db-path runs/storage/history.db `
  --privacy-mapping-dir runs/storage/privacy_mappings `
  --output-log-file <write_log_file> `
  --user-done-log-file <completed_users_log_file>
```

The batch generator runs one response process per user and uses `--max-parallel` to bound concurrent users:

```powershell
python eval\run_batch_generate_responses.py `
  --domain <domain> `
  --start-user <start_user> --end-user <end_user> `
  --max-parallel <max_parallel_users> `
  --test-dir data/<domain>/evaluation_queries `
  --data-dir <downloaded_history_dir> `
  --response-model gpt-5.2-chat `
  --output-tag <output_tag> `
  --collection-name spmem_paper `
  --qdrant-url $env:SPMEM_QDRANT_URL `
  --history-db-path runs/storage/history.db `
  --privacy-mapping-dir runs/storage/privacy_mappings `
  --output-dir <response_output_dir>
```

Repeat the batch command with `--response-model llama-3.1-8b-instruct`, `qwen3-14b`, or `deepseek-v3.2` and a distinct output tag/directory for the other paper backbones. Provider model IDs remain environment-configured.

The configuration-driven utility uses the same Qdrant service, `spmem_paper` collection, history database, and private-mapping directory defined above. To run the README's Education user 0 example through the vector-only and graph-only ablations, first save the matching selection:

```powershell
python scripts\select_users.py `
  --domain education `
  --user-index 0 `
  --output runs/selections/education_user0.json
```

Then run:

```powershell
python scripts\generate_spmem_responses.py `
  --config configs/paper.example.json `
  --selection-manifest runs/selections/education_user0.json `
  --model gpt-5.2-chat `
  --retrieval-mode vector_only `
  --output-dir runs/responses/ablations/vector_only

python scripts\generate_spmem_responses.py `
  --config configs/paper.example.json `
  --selection-manifest runs/selections/education_user0.json `
  --model gpt-5.2-chat `
  --retrieval-mode graph_only `
  --output-dir runs/responses/ablations/graph_only
```

Use `--resume` to skip completed output after an interruption. Add `--collection-suffix <name>` only if the corresponding memories were written to that suffixed collection.

The configured response-model keys are `gpt-5.2-chat`, `llama-3.1-8b-instruct`, `qwen3-14b`, and `deepseek-v3.2`. Repeat `--model` to generate more than one model in a single invocation.

## Paper baselines

The baseline entry point supports Full-context, Mem0, Zep, and MemOS. Full-context uses the complete selected user's interaction history.

```powershell
python scripts\generate_baseline_responses.py `
  --config <paper_config> `
  --baseline-config <baseline_config> `
  --selection-manifest <selection_manifest> `
  --baseline <baseline_name> `
  --model <response_model_key> `
  --output-dir <baseline_output_dir>
```

Change `--baseline` to `mem0`, `zep`, or `memos` for another baseline. Optional backends are imported lazily, so missing baseline packages do not prevent core SP-Mem or Full-context imports. Install optional dependencies with:

```powershell
python -m pip install -r requirements-baselines.txt
```

For Mem0 and MemOS, complete the package-specific values in `configs\baselines.example.json` using options accepted by the installed release. Set `SPMEM_ZEP_API_KEY` for Zep. Historical dependency versions that are not evidenced are intentionally not claimed. Use `--skip-build` only when the corresponding optional baseline store has already been built for the same selection and configuration.

## Merge response files

Response generation writes per-user JSONL files. Merge a method/model directory before cross-domain pairwise scoring:

```powershell
python scripts\merge_jsonl.py `
  --inputs <spmem_response_dir> `
  --output <spmem_merged_jsonl>

python scripts\merge_jsonl.py `
  --inputs <baseline_response_dir> `
  --output <baseline_merged_jsonl>
```

Adjust the second input path to the directory emitted by the actual baseline run if a different output layout or model key was selected.

## Privacy-Appropriate Requesting (PAR)

PAR compares whether the agent requests authorization with whether the task requires exact private information. Its output reports confusion counts, accuracy, precision, recall, and F1 overall and by domain and task condition; higher scores are better. Calculate it from a response file or directory:

```powershell
python eval\evaluate_par.py `
  --responses <response_file_or_dir> `
  --output-json <par_summary_json>
```

Rows marked as generation errors are excluded by default. Pass `--include-error-rows` only when that alternative denominator is intended and record the choice with the result.

## Unnecessary Privacy Usage (UPU)

UPU measures whether an answer exposes an exact private value when that value is unnecessary or authorization is denied. The scorer checks responses against the exact synthetic values in the privacy profiles and reports an exposure rate, for which lower is better. Repeat `--privacy-profile-file` for every domain represented in the response file:

```powershell
python eval\score_upu_exact_match.py `
  --responses <response_file_or_dir> `
  --privacy-profile-file <privacy_profile_file> `
  --output-jsonl <upu_scored_rows_jsonl> `
  --summary-json <upu_summary_json>
```

The per-row file is optional but useful for audit. The summary reports the evaluated row counts and grouping used by the scorer.

## Pairwise P-TC and P-PQ

P-TC (pairwise Task Completion) evaluates whether a response satisfies the task requirements and is practically useful. P-PQ (pairwise Personalization Quality) evaluates whether a response uses the required preferences without inventing unsupported user information. Both are reported as win-or-tie rates, for which higher is better.

Select the judge with `--judge-model`; use `--judge-name` when a shorter output-directory label is useful. Endpoint credentials can be supplied under `SPMEM_JUDGE_*`. The pairwise batch driver evaluates independent per-user file pairs concurrently, while the evaluator inside each file performs both A/B orders sequentially:

```powershell
python eval\run_batch_pairwise_evaluation.py `
  --input-dir-a <spmem_response_dir> `
  --input-dir-b <baseline_response_dir> `
  --glob "*.jsonl" `
  --output-root <pairwise_output_dir> `
  --max-parallel <max_parallel_judges> `
  --judge-model <judge_model_id> `
  --judge-name <judge_name> `
  --name-a SP-Mem `
  --name-b <baseline_name> `
  --preference-profile-file <preference_profile_file> `
  --strict-pairing
```

The evaluator submits both A/B orders. A consistent winner is retained; order conflicts are recorded as ties. The reported `*_pair_score` is `(wins + ties) / scored`, which is a win-or-tie rate rather than a win rate. `--strict-pairing` stops when the two files do not contain the same comparison keys.

## Token-usage summary

Summarize recorded response usage across a file tree:

```powershell
python scripts\summarize_token_usage.py `
  --responses <response_file_or_dir> `
  --output-json <token_summary_json>
```

Only integer usage values returned by a provider are summed. Missing usage is counted separately and is not converted to zero consumption.

## Output and verification notes

- Generated stores, private mappings, model outputs, judge outputs, logs, and caches belong under ignored runtime paths such as `runs/`.
- Compare methods using the same selection manifest, queries, conditions, response model, and judge configuration.
- Do not relabel a seeded subset as the paper's original 100-user split; its exact IDs are not published here.
- Static compilation and CLI checks do not verify paid model calls, live database writes, optional baseline services, or reported paper values.
- Rebuttal-only attacks, added baselines, robustness tests, human evaluation, and significance analysis are outside this repository's current paper-aligned scope.
