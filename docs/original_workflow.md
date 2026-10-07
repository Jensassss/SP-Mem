# Original experiment workflow provenance

The public experiment entry points in this document were copied from the parent experiment tree rather than reimplemented from scratch. Their source files and pre-adaptation SHA-256 hashes are:

| Public file | Source in the experiment tree | Source SHA-256 |
| --- | --- | --- |
| `scripts/build_memories.py` | `test/test_async_write_data.py` | `E80159635FAB5521617B4CBBA0980E124B99F6F15FCA2DA93CE17BCE5B668E22` |
| `eval/generate_agent_responses.py` | `eval/generate_agent_responses.py` | `00EAB4608AAF8CAF23AD0845AC82A788E12325ABC4520A8E1D37E33A48043791` |
| `eval/run_batch_generate_responses.py` | `eval/run_batch_generate_responses.py` | `AD0B83CAA4666E3B52B23A3346BE4A659C42F7B86CB7463AF01C13679C96C37E` |
| `eval/evaluate_pairwise_tc_pq.py` | `eval/evaluate_pairwise_tc_pq.py` | `18079D362D3D52F7EE33EEC83475B1B96E875FEDBAA285F56DCFC2CA6AD4A2C9` |
| `eval/run_batch_pairwise_evaluation.py` | `eval/run_batch_pairwise_gpt41_by_file.py` | `302D21F2EA53976FA778565C5629CC4F5C3B8466552BC5A85C103742DA887998` |

## Public entry-point groups

The README uses `scripts/build_memories.py` for memory writing and `eval/run_batch_generate_responses.py` for response generation. These entry points come from the experiment-tree files recorded above and expose their data, model, storage, batching, and concurrency settings through command-line arguments and environment variables.

The utilities under `scripts/` were organized separately during public-repository cleanup. They provide configuration-driven user selection, validation, memory building, response generation, baseline execution, merging, and aggregation. They are useful public interfaces, but their command-line orchestration should not be treated as evidence of the experiment-tree workflow. [`setup.md`](setup.md) documents configuration, and [`evaluation.md`](evaluation.md) lists both command families where relevant.

The source scripts are not copied byte-for-byte into the public tree because they contained machine-specific paths, a local database credential, and imports tied to the parent checkout. The public adaptations are deliberately limited to:

- renaming the experiment writer from `test/test_async_write_data.py` to `scripts/build_memories.py` for a clear public entry point;
- replacing private paths and credentials with command-line arguments or environment variables;
- changing `mem0` and `privacy_aware_agent` imports to the public package names `spmem_memory` and `spmem_agent`;
- passing one explicit private-mapping directory instead of copying mappings into the script directory;
- exposing the writer's original user range, batching, concurrency, retry, and log values as arguments;
- using the four paper model labels in the response-model registry while leaving provider deployment IDs configurable;
- correcting the copied pairwise summary's half-tie intermediate score to the paper's explicitly defined win-tie rate `(W + T) / N` (the swapped-order judge decisions themselves are unchanged);
- disabling logs of retrieved memories, generated answers, and per-batch results by default; and
- moving generated stores, mappings, responses, and logs under ignored `outputs/` paths.

These adaptations do not replace the original orchestration:

- memory writing creates one asyncio task per user, uses a semaphore to limit concurrent users, preserves per-user dialogue order, and retries failed batches;
- response generation uses the original per-user generator, with `run_batch_generate_responses.py` launching users concurrently through a thread pool of child processes; and
- pairwise judge evaluation performs both response orders within each file pair, while the batch driver evaluates independent file pairs concurrently.

The repository does not claim that a new run has reproduced the paper's reported values. External model endpoints, Qdrant, and Neo4j are required for end-to-end execution.
