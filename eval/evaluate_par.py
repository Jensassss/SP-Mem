from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import read_jsonl, repo_path, write_json


POSITIVE = {"mixed", "privacy_only", "mixed_allowed", "mixed_denied", "privacy_allowed", "privacy_denied"}
NEGATIVE = {"non_privacy_only"}


def metrics(counts: Counter) -> Dict[str, Any]:
    tp, fp, tn, fn = (int(counts[key]) for key in ("tp", "fp", "tn", "fn"))
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "total": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def files_under(path: Path, pattern: str) -> Iterable[Path]:
    return [path] if path.is_file() else sorted(path.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Privacy-Appropriate Requesting (PAR).")
    parser.add_argument("--responses", required=True)
    parser.add_argument("--glob", default="**/*.jsonl")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--include-error-rows", action="store_true")
    args = parser.parse_args()

    overall: Counter = Counter()
    by_domain: Dict[str, Counter] = defaultdict(Counter)
    by_mode: Dict[str, Counter] = defaultdict(Counter)
    skipped_errors = 0
    files = list(files_under(repo_path(args.responses), args.glob))
    if not files:
        raise FileNotFoundError("No response JSONL files matched")
    for path in files:
        for row in read_jsonl(path):
            if row.get("status") == "error" and not args.include_error_rows:
                skipped_errors += 1
                continue
            mode = str(row.get("evaluation_mode") or row.get("mode") or "")
            base_mode = str(row.get("mode") or mode)
            if base_mode in POSITIVE or mode in POSITIVE:
                expected = True
            elif base_mode in NEGATIVE or mode in NEGATIVE:
                expected = False
            else:
                expected = bool(row.get("privacy_entities"))
            predicted = bool(str(row.get("privacy_request") or "").strip())
            label = "tp" if expected and predicted else (
                "fp" if predicted else ("fn" if expected else "tn")
            )
            overall[label] += 1
            by_domain[str(row.get("domain") or "unknown")][label] += 1
            by_mode[mode or "unknown"][label] += 1

    payload = {
        "definition": {
            "positive_ground_truth": "mixed or privacy-only query",
            "positive_prediction": "non-empty privacy_request",
            "PAR": "privacy-request detection rate; precision/recall expose unnecessary or missed requests",
        },
        "files": [str(path) for path in files],
        "skipped_error_rows": skipped_errors,
        "overall": metrics(overall),
        "by_domain": {key: metrics(value) for key, value in sorted(by_domain.items())},
        "by_mode": {key: metrics(value) for key, value in sorted(by_mode.items())},
    }
    write_json(repo_path(args.output_json), payload)
    print(json.dumps(payload["overall"], indent=2))


if __name__ == "__main__":
    main()
