from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import DOMAINS, read_json, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the public synthetic benchmark layout and identifiers.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    report = {"domains": {}, "errors": []}
    global_user_ids = set()

    for domain in DOMAINS:
        histories = sorted((data_root / domain / "histories").glob("user_*.json"))
        queries = sorted((data_root / domain / "evaluation_queries").glob("user*_test.jsonl"))
        privacy = list(read_jsonl(data_root / domain / "profiles" / "privacy_profiles.jsonl"))
        preference = list(read_jsonl(data_root / domain / "profiles" / "preference_profiles.jsonl"))
        dialogue_counts = Counter()
        query_counts = Counter()
        indices = set()
        for history_file in histories:
            payload = read_json(history_file)
            index = int(payload["user_index"])
            user_id = str(payload["user_id"])
            indices.add(index)
            dialogue_counts[len(payload.get("dialogues", []))] += 1
            expected_query = data_root / domain / "evaluation_queries" / f"user{index}_test.jsonl"
            if not expected_query.exists():
                report["errors"].append(f"missing query file: {expected_query}")
            else:
                query_counts[sum(1 for _ in read_jsonl(expected_query))] += 1
            if user_id in global_user_ids:
                report["errors"].append(f"duplicate user_id across domains: {user_id}")
            global_user_ids.add(user_id)
        if len(privacy) != len(histories) or len(preference) != len(histories):
            report["errors"].append(
                f"{domain}: profile/history count mismatch "
                f"privacy={len(privacy)} preference={len(preference)} histories={len(histories)}"
            )
        report["domains"][domain] = {
            "histories": len(histories),
            "query_files": len(queries),
            "privacy_profiles": len(privacy),
            "preference_profiles": len(preference),
            "dialogues_per_user": dict(sorted(dialogue_counts.items())),
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
