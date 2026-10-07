from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import append_jsonl, read_jsonl, repo_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge response JSONL files and reject duplicate sample IDs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Files or directories.")
    parser.add_argument("--glob", default="**/*.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = repo_path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    files = []
    for value in args.inputs:
        path = repo_path(value)
        files.extend([path] if path.is_file() else sorted(path.glob(args.glob)))
    seen = set()
    rows = 0
    for path in files:
        if path.resolve() == output.resolve():
            continue
        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"Missing sample_id in {path}")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id {sample_id!r}")
            seen.add(sample_id)
            append_jsonl(output, row)
            rows += 1
    print(f"merged {rows} rows from {len(files)} files into {output}")


if __name__ == "__main__":
    main()
