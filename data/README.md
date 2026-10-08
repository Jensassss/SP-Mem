# SP-Mem evaluation queries

This directory contains the test queries released with SP-Mem. The synthetic conversation histories are hosted separately in the [Privacy-Aware Memory Benchmark on Hugging Face](https://huggingface.co/datasets/wwj95/privacy-aware-memory-benchmark).

## Scope

| Directory | Domain | Evaluation-query files |
| --- | --- | ---: |
| `finance/` | Finance | 250 |
| `medical/` | Medical | 250 |
| `education/` | Education | 250 |
| `mental/` | Mental Support | 250 |

The repository contains one query file for each of 1,000 synthetic users.

## Structure

```text
<domain>/
  evaluation_queries/
    userX_test.jsonl
```

Each JSONL row contains:

- `scenario`: the scenario label associated with the query;
- `query`: the user request passed to response generation;
- `preference_entities`: the non-private preference types required for the query; and
- `privacy_entities`: the private entity types required for the query.

The numeric `X` in `userX_test.jsonl` maps to `user_XXXX.json` in the matching Hugging Face domain. User indices repeat across domains, so both the domain and index are required for alignment.

Task categories and authorization conditions are derived at runtime from the two entity lists. Preference-only queries have no required private entities. Privacy-only and mixed queries are evaluated under allowed and denied authorization conditions; these conditions are not separate files or fields in the released query records.

Conversation histories and profile reference files are not stored in this GitHub repository. See the repository [README](../README.md) for the minimal workflow and [docs/evaluation.md](../docs/evaluation.md) for evaluation commands.
