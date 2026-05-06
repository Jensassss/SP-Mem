# SP-Mem Benchmark Data

This directory contains the synthetic benchmark inputs used by SP-Mem. It includes the input-side data needed to reconstruct memories and run evaluation, but does not include stored vector databases, graph databases, SQLite memory stores, generated responses, judge outputs, logs, or cache files.

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

- `profiles/preference_profiles.jsonl`: per-user non-private preference inventory.
- `profiles/privacy_profiles.jsonl`: per-user private-information inventory used for privacy-aware memory writing and UPU evaluation.
- `histories/user_XXXX.json`: user-assistant history dialogues used for memory construction.
- `evaluation_queries/userX_test.jsonl`: evaluation queries with task scenario, required preference entities, and required privacy entities.
- `DATA_MANIFEST.json`: counts and relative paths for each domain.

The stored memory databases should be regenerated from `histories/` with the SP-Mem scripts rather than submitted as static database files.


## Data Access and Restrictions

The data are synthetic and are included to support confidential peer review. Stored memory databases and generated model outputs are not included; they should be regenerated from the histories and profiles in this directory. The public release should provide the final de-anonymized repository link, persistent dataset identifier, and license/access information.
