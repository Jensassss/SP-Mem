from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import read_jsonl, repo_path, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize reported API token usage without treating missing usage as zero.")
    parser.add_argument("--responses", required=True)
    parser.add_argument("--glob", default="**/*.jsonl")
    parser.add_argument("--token-field", default="total_tokens")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    source = repo_path(args.responses)
    files = [source] if source.is_file() else sorted(source.glob(args.glob))
    groups: Dict[Tuple[str, str, str, str], Dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "known_usage_rows": 0, "missing_usage_rows": 0, "token_sum": 0}
    )
    for path in files:
        for row in read_jsonl(path):
            key = (
                str(row.get("method") or "unknown"),
                str(row.get("model") or "unknown"),
                str(row.get("domain") or "unknown"),
                str(row.get("evaluation_mode") or row.get("mode") or "unknown"),
            )
            bucket = groups[key]
            bucket["rows"] += 1
            value = row.get(args.token_field)
            # bool is not accepted as a token count even though it subclasses int.
            if isinstance(value, int) and not isinstance(value, bool):
                bucket["known_usage_rows"] += 1
                bucket["token_sum"] += value
            else:
                bucket["missing_usage_rows"] += 1

    output_groups = []
    for (method, model, domain, mode), bucket in sorted(groups.items()):
        known = bucket["known_usage_rows"]
        output_groups.append({
            "method": method, "model": model, "domain": domain, "evaluation_mode": mode,
            **bucket,
            "mean_tokens_over_known_rows": bucket["token_sum"] / known if known else None,
            "usage_coverage": known / bucket["rows"] if bucket["rows"] else None,
        })
    totals = {
        "rows": sum(item["rows"] for item in output_groups),
        "known_usage_rows": sum(item["known_usage_rows"] for item in output_groups),
        "missing_usage_rows": sum(item["missing_usage_rows"] for item in output_groups),
        "token_sum": sum(item["token_sum"] for item in output_groups),
    }
    totals["mean_tokens_over_known_rows"] = (
        totals["token_sum"] / totals["known_usage_rows"] if totals["known_usage_rows"] else None
    )
    payload = {
        "token_field": args.token_field,
        "definition": (
            "Only integer values are counted. A missing/null usage field is reported as missing, "
            "while an integer zero remains a real observed zero."
        ),
        "files": [str(path) for path in files],
        "overall": totals,
        "groups": output_groups,
    }
    write_json(repo_path(args.output_json), payload)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
