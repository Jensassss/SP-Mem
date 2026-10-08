from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import DOMAINS, read_jsonl, write_json


QUERY_FILE_PATTERN = re.compile(r"user(\d+)_test\.jsonl$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the public SP-Mem evaluation-query layout.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    report = {"domains": {}, "errors": []}

    for domain in DOMAINS:
        query_dir = data_root / domain / "evaluation_queries"
        queries = sorted(query_dir.glob("user*_test.jsonl"))
        query_counts = Counter()
        indices = set()

        for query_file in queries:
            match = QUERY_FILE_PATTERN.fullmatch(query_file.name)
            if match is None:
                report["errors"].append(f"unexpected query filename: {query_file}")
                continue

            index = int(match.group(1))
            if index in indices:
                report["errors"].append(f"duplicate query index in {domain}: {index}")
            indices.add(index)

            row_count = sum(1 for _ in read_jsonl(query_file))
            query_counts[row_count] += 1
            if row_count == 0:
                report["errors"].append(f"empty query file: {query_file}")

        expected_indices = set(range(250))
        if indices != expected_indices:
            missing = sorted(expected_indices - indices)
            extra = sorted(indices - expected_indices)
            report["errors"].append(
                f"{domain}: query index mismatch missing={missing} extra={extra}"
            )

        report["domains"][domain] = {
            "query_files": len(queries),
            "queries_per_user": dict(sorted(query_counts.items())),
            "index_min": min(indices) if indices else None,
            "index_max": max(indices) if indices else None,
        }

    report["valid"] = not report["errors"]
    if args.output_json:
        write_json(Path(args.output_json).resolve(), report)
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
