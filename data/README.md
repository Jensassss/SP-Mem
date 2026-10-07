# SP-Mem benchmark data

This directory contains the public synthetic benchmark inputs used by SP-Mem. It includes the input-side data needed to reconstruct memories and run evaluation, but it does not include stored vector databases, graph databases, SQLite memory stores, generated responses, judge outputs, logs, or cache files.

## Scope

The benchmark covers four domains:

| Directory | Domain | User history files | Evaluation-query files |
| --- | --- | ---: | ---: |
| `finance/` | Finance | 250 | 250 |
| `medical/` | Medical | 250 | 250 |
| `education/` | Education | 250 | 250 |
| `mental/` | Mental Support | 250 | 250 |

The repository therefore contains inputs for 1,000 synthetic users. Counts and tracked relative paths are also recorded in `DATA_MANIFEST.json`.

## Structure

Each domain directory has the same layout:

```text
<domain>/
  profiles/
    preference_profiles.jsonl
    privacy_profiles.jsonl
  histories/
    user_XXXX.json
  evaluation_queries/
    userX_test.jsonl
```

## Files

- `profiles/preference_profiles.jsonl`: per-user non-private preference inventory used as a personalization-evaluation reference.
- `profiles/privacy_profiles.jsonl`: per-user exact synthetic private-value inventory used as a privacy-evaluation reference.
- `histories/user_XXXX.json`: user-assistant history dialogues used for memory construction.
- `evaluation_queries/userX_test.jsonl`: evaluation queries with task scenario, required preference entities, and required privacy entities.
- `DATA_MANIFEST.json`: counts and relative paths for each domain.

The profile files are evaluation references and are not passed wholesale to the response model.

**Profile records** (`profiles/*.jsonl`):

- `user_id`: the synthetic user identifier used to align a profile with a user.
- Other keys contain domain-specific preference fields or categorized private-value inventories.

**History records** (`histories/user_XXXX.json`):

- `user_index` and `user_id`: identifiers used to pair histories, queries, and stored memory.
- `dialogues`: the history records supplied to memory construction.
- `dialogue_index`: the position of a dialogue in one user's history.
- `dialogue`: an ordered list of turns, each with `role` and `content`, plus `labels` where annotations are available.

**Query records** (`evaluation_queries/userX_test.jsonl`):

- `scenario`: the broad scenario label attached to an evaluation query.
- `query`: the user request passed to response generation.
- `preference_entities`: the non-private preference types required for the query.
- `privacy_entities`: the private entity types required for the query.

Task categories and authorization conditions are derived from the two entity lists rather than stored as a separate `consent` field:

- **Preference-only:** `preference_entities` is non-empty and `privacy_entities` is empty; the runner uses `non_privacy_only` without an authorization branch.
- **Privacy-only:** `privacy_entities` is non-empty and `preference_entities` is empty; the runner creates `privacy_allowed` and `privacy_denied` conditions.
- **Mixed:** both entity lists are non-empty; the runner creates `mixed_allowed` and `mixed_denied` conditions.

Allowed and denied are runtime evaluation conditions, not separate data domains or independent query files.

Stored memory databases should be regenerated from `histories/` with the SP-Mem scripts rather than distributed as static database files.

## Selection and generated state

User selection is configurable by domain, explicit user ID/index, range, or seeded sampling. The paper reports evaluation on a 100-user subset, but the exact historical user IDs are not published here. A newly generated selection manifest must not be presented as the original paper split.

Stored memory databases and generated model outputs are intentionally not included; regenerate memory from the histories in this directory. Runtime databases, private mappings, responses, and logs should remain in ignored paths such as `runs/`.

See the repository [README](../README.md) for a minimal run and [docs/evaluation.md](../docs/evaluation.md) for evaluation commands.
