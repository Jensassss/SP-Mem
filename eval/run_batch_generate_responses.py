from __future__ import annotations

# Migrated from the experiment tree; preserves concurrent per-user subprocess scheduling.
# See docs/original_workflow.md for provenance and public adaptations.

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class UserTask:
    user_index: int
    test_file: Path
    output_file: Path
    out_log: Path
    err_log: Path


@dataclass
class UserResult:
    user_index: int
    exit_code: int
    elapsed_seconds: float
    output_file: Path
    out_log: Path
    err_log: Path


def _build_parser() -> argparse.ArgumentParser:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Batch-generate agent responses (Python, concurrent users).")
    parser.add_argument("--start-user", type=int, default=0)
    parser.add_argument("--end-user", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=10)
    parser.add_argument("--root-path", default=str(root_default))
    parser.add_argument(
        "--test-dir",
        default="",
        help="Query directory. Default: <root>/data/<domain>/evaluation_queries",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Default: <root>/runs/responses/<domain>",
    )
    parser.add_argument("--domain", default="medical")
    parser.add_argument("--response-model", default="gpt5.2-chat")
    parser.add_argument("--output-tag", default="gpt52chat")
    parser.add_argument("--collection-name", default="spmem_paper")
    parser.add_argument("--qdrant-url", default=os.getenv("SPMEM_QDRANT_URL") or os.getenv("QDRANT_URL", ""))
    parser.add_argument("--history-db-path", default=str(root_default / "runs" / "storage" / "history.db"))
    parser.add_argument(
        "--privacy-mapping-dir",
        default=str(root_default / "runs" / "storage" / "privacy_mappings"),
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Directory containing user_XXXX.json files. Default: <root>/data",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--no-include-efficiency", action="store_true")
    parser.add_argument("--disable-child-tqdm", action="store_true", help="Set TQDM_DISABLE=1 for child process.")
    parser.add_argument(
        "--progress-log-file",
        default="",
        help=(
            "Realtime progress log file. "
            "Default: <output-dir>/batch_progress_<output_tag>.txt"
        ),
    )
    return parser


def _run_one_user(
    *,
    task: UserTask,
    script_path: Path,
    python_bin: str,
    domain: str,
    response_model: str,
    collection_name: str,
    qdrant_url: str,
    history_db_path: str,
    privacy_mapping_dir: str,
    data_dir: str,
    include_efficiency: bool,
    disable_child_tqdm: bool,
) -> UserResult:
    cmd: List[str] = [
        python_bin,
        "-u",
        str(script_path),
        "--test-file",
        str(task.test_file),
        "--user-index",
        str(task.user_index),
        "--domain",
        domain,
        "--response-models",
        response_model,
        "--collection-name",
        collection_name,
        "--qdrant-url",
        qdrant_url,
        "--history-db-path",
        history_db_path,
        "--privacy-mapping-dir",
        privacy_mapping_dir,
        "--data-dir",
        data_dir,
        "--output-file",
        str(task.output_file),
    ]
    if include_efficiency:
        cmd.append("--include-efficiency")

    env = os.environ.copy()
    # Prevent Windows GBK crashes when model outputs emoji or other non-GBK chars.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if disable_child_tqdm:
        env["TQDM_DISABLE"] = "1"

    start = time.perf_counter()
    task.out_log.parent.mkdir(parents=True, exist_ok=True)
    task.output_file.parent.mkdir(parents=True, exist_ok=True)
    with task.out_log.open("w", encoding="utf-8", errors="replace") as out_fp, task.err_log.open(
        "w", encoding="utf-8", errors="replace"
    ) as err_fp:
        proc = subprocess.run(cmd, stdout=out_fp, stderr=err_fp, env=env, check=False)
    elapsed = time.perf_counter() - start

    return UserResult(
        user_index=task.user_index,
        exit_code=int(proc.returncode),
        elapsed_seconds=float(elapsed),
        output_file=task.output_file,
        out_log=task.out_log,
        err_log=task.err_log,
    )


def main() -> None:
    args = _build_parser().parse_args()
    if args.start_user > args.end_user:
        raise ValueError(f"--start-user must be <= --end-user, got {args.start_user}>{args.end_user}")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")
    if not str(args.qdrant_url).strip():
        raise ValueError("Qdrant URL is required via --qdrant-url or SPMEM_QDRANT_URL")

    root = Path(args.root_path).resolve()
    script_path = root / "eval" / "generate_agent_responses.py"
    test_dir = Path(args.test_dir).resolve() if str(args.test_dir).strip() else (
        root / "data" / args.domain / "evaluation_queries"
    )
    output_dir = Path(args.output_dir).resolve() if str(args.output_dir).strip() else (
        root / "runs" / "responses" / args.domain
    )
    log_dir = output_dir / f"logs_{args.output_tag}"
    progress_log_file = (
        Path(args.progress_log_file).resolve()
        if str(args.progress_log_file).strip()
        else (output_dir / f"batch_progress_{args.output_tag}.txt").resolve()
    )

    if not script_path.exists():
        raise FileNotFoundError(f"generate script not found: {script_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"test dir not found: {test_dir}")
    if not Path(args.history_db_path).exists():
        raise FileNotFoundError(f"history db not found: {args.history_db_path}")
    if not Path(args.privacy_mapping_dir).exists():
        raise FileNotFoundError(f"privacy mapping dir not found: {args.privacy_mapping_dir}")
    data_dir = Path(args.data_dir).resolve() if str(args.data_dir).strip() else (
        root / "data" / args.domain / "histories"
    )
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_log_file.parent.mkdir(parents=True, exist_ok=True)

    tasks: List[UserTask] = []
    for i in range(args.start_user, args.end_user + 1):
        test_file = test_dir / f"user{i}_test.jsonl"
        if not test_file.exists():
            print(f"[SKIP] user={i} missing test file: {test_file}")
            continue
        user_padded = f"{i:04d}"
        tasks.append(
            UserTask(
                user_index=i,
                test_file=test_file,
                output_file=output_dir / f"user{user_padded}_{args.output_tag}.jsonl",
                out_log=log_dir / f"user{user_padded}.out.log",
                err_log=log_dir / f"user{user_padded}.err.log",
            )
        )

    print("=" * 60)
    print("Batch Generate Agent Responses (Python)")
    print(f"script: {script_path}")
    print(f"users: {args.start_user}..{args.end_user}")
    print(f"scheduled: {len(tasks)}")
    print(f"parallel: {args.max_parallel}")
    print(f"response_model: {args.response_model}")
    print(f"data_dir: {data_dir}")
    print(f"output_dir: {output_dir}")
    print(f"log_dir: {log_dir}")
    print(f"progress_log_file: {progress_log_file}")
    print("=" * 60)

    if not tasks:
        print("[INFO] no users scheduled.")
        return

    include_efficiency = not bool(args.no_include_efficiency)
    ok = 0
    fail = 0
    total = len(tasks)

    with progress_log_file.open("w", encoding="utf-8") as progress_fp:
        progress_fp.write(
            f"START users={args.start_user}..{args.end_user} scheduled={total} "
            f"parallel={args.max_parallel} model={args.response_model}\n"
        )
        progress_fp.flush()

        with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
            future_map = {
                pool.submit(
                    _run_one_user,
                    task=task,
                    script_path=script_path,
                    python_bin=args.python_bin,
                    domain=args.domain,
                    response_model=args.response_model,
                    collection_name=args.collection_name,
                    qdrant_url=args.qdrant_url,
                    history_db_path=str(args.history_db_path),
                    privacy_mapping_dir=str(args.privacy_mapping_dir),
                    data_dir=str(data_dir),
                    include_efficiency=include_efficiency,
                    disable_child_tqdm=bool(args.disable_child_tqdm),
                ): task
                for task in tasks
            }

            if tqdm is not None:
                pbar = tqdm(total=len(future_map), desc="Users", unit="user")
            else:
                pbar = None

            completed = 0
            for fut in as_completed(future_map):
                result = fut.result()
                completed += 1
                if result.exit_code == 0:
                    ok += 1
                    line = (
                        f"[DONE] user{result.user_index:04d} exit=0 "
                        f"elapsed={result.elapsed_seconds:.1f}s output={result.output_file} "
                        f"completed={completed}/{total} ok={ok} fail={fail}"
                    )
                else:
                    fail += 1
                    line = (
                        f"[FAIL] user{result.user_index:04d} exit={result.exit_code} "
                        f"elapsed={result.elapsed_seconds:.1f}s out={result.out_log} err={result.err_log} "
                        f"completed={completed}/{total} ok={ok} fail={fail}"
                    )

                print(line)
                progress_fp.write(line + "\n")
                progress_fp.flush()
                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

        progress_fp.write(f"FINISHED success={ok} fail={fail} total={total}\n")
        progress_fp.flush()

    print("=" * 60)
    print(f"Finished. success={ok} fail={fail}")
    print(f"Progress log saved: {progress_log_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
