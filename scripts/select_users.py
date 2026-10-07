from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import add_selection_arguments, repo_path, save_selection_manifest, select_users


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible user-selection manifest without running models.")
    parser.add_argument("--output", required=True)
    add_selection_arguments(parser)
    args = parser.parse_args()
    if args.selection_manifest:
        raise ValueError("select_users creates a new manifest; do not pass --selection-manifest")
    users = select_users(args)
    output = repo_path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")
    save_selection_manifest(output, users, arguments=vars(args))
    print(f"saved {len(users)} users: {output}")


if __name__ == "__main__":
    main()
