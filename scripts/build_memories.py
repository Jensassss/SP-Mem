"""Asynchronously build SP-Mem memory for a selected domain and user range."""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import spmem_memory
from spmem_memory import AsyncMemory

DATA_DIR = PROJECT_ROOT / "data" / "mental" / "histories"
USER_INDEX_START = 0
USER_INDEX_END = 0

BATCH_SIZE = 1
MAX_CONCURRENT_USERS = 5
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 2.0
VERBOSE_RESULTS = False

OUTPUT_LOG_FILE = PROJECT_ROOT / "runs" / "logs" / "async_write.log"
USER_DONE_LOG_FILE = PROJECT_ROOT / "runs" / "logs" / "async_write_users_done.log"

CONFIG = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": os.getenv("SPMEM_MEMORY_MODEL_ID") or os.getenv("MEMORY_LLM_MODEL", "gpt-5.2-chat"),
            "api_key": os.getenv("SPMEM_MEMORY_API_KEY") or os.getenv("MEMORY_LLM_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
            "openai_base_url": os.getenv("SPMEM_MEMORY_BASE_URL") or os.getenv("MEMORY_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", ""),
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": os.getenv("SPMEM_NEO4J_URL") or os.getenv("NEO4J_URL", "neo4j://localhost:7687"),
            "username": os.getenv("SPMEM_NEO4J_USERNAME") or os.getenv("NEO4J_USERNAME", "neo4j"),
            "password": os.getenv("SPMEM_NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD", ""),
            "database": os.getenv("SPMEM_NEO4J_DATABASE") or os.getenv("NEO4J_DATABASE", "neo4j"),
        }
    },
    "vector_store": {
    "provider": "qdrant",
    "config": {
        "collection_name": "spmem_paper",
        "url": os.getenv("SPMEM_QDRANT_URL") or os.getenv("QDRANT_URL", ""),
        "embedding_model_dims": 1536
        }
    },
    "history_db_path": str(PROJECT_ROOT / "runs" / "storage" / "history.db"),
    "privacy_mapping_dir": str(PROJECT_ROOT / "runs" / "storage" / "privacy_mappings"),
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _append_user_done_log(line: str) -> None:
    USER_DONE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with USER_DONE_LOG_FILE.open("a", encoding="utf-8", buffering=1) as fp:
        fp.write(line.rstrip("\n") + "\n")
        fp.flush()


def _clean_dialogue(dialogue):
    cleaned = []
    for msg in dialogue:
        role = msg.get("role")
        content = msg.get("content")
        if role is None or content is None:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def _dialogue_batches(dialogues, batch_size):
    for start in range(0, len(dialogues), batch_size):
        yield dialogues[start:start + batch_size]


def _build_resume_batch_map_from_log(log_path: Path):
    """
    Build per-user resume start batch from the latest non-empty previous run section.
    Rule:
    - Find the latest previous run block (before current run start) that contains
      at least one [START]/[OK] line.
    - For each user, resume from the first batch that is not confirmed [OK].
      If all started batches are [OK], resume from max_started + 1.
    """
    if not log_path.exists():
        return {}

    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return {}

    run_start_indices = [i for i, line in enumerate(lines) if line.startswith("RUN START:")]
    if len(run_start_indices) < 2:
        return {}

    # In current process, a fresh RUN START line is written before main().
    # We must look backward and skip empty previous run blocks.
    prev_run_lines = []
    for i in range(len(run_start_indices) - 2, -1, -1):
        start_idx = run_start_indices[i]
        end_idx = run_start_indices[i + 1]
        candidate = lines[start_idx:end_idx]
        has_progress = any(
            line.startswith("[START] user_id=") or line.startswith("[OK] user_id=")
            for line in candidate
        )
        if has_progress:
            prev_run_lines = candidate
            break

    if not prev_run_lines:
        return {}

    per_user_started = {}
    per_user_ok = {}

    for line in prev_run_lines:
        if line.startswith("[START] user_id="):
            # Example: [START] user_id=... batch=3 dialogues=...
            user_token = "user_id="
            batch_token = " batch="
            try:
                uid_start = line.index(user_token) + len(user_token)
                uid_end = line.index(" ", uid_start)
                uid = line[uid_start:uid_end]

                batch_start = line.index(batch_token) + len(batch_token)
                batch_end = line.index(" ", batch_start)
                batch_no = int(line[batch_start:batch_end])
            except Exception:
                continue

            per_user_started[uid] = max(batch_no, per_user_started.get(uid, 0))

        elif line.startswith("[OK] user_id="):
            # Example: [OK] user_id=... batch=3
            user_token = "user_id="
            batch_token = " batch="
            try:
                uid_start = line.index(user_token) + len(user_token)
                uid_end = line.index(" ", uid_start)
                uid = line[uid_start:uid_end]

                batch_start = line.index(batch_token) + len(batch_token)
                batch_no = int(line[batch_start:].strip())
            except Exception:
                continue

            if uid not in per_user_ok:
                per_user_ok[uid] = set()
            per_user_ok[uid].add(batch_no)

    resume_map = {}
    for uid, max_started in per_user_started.items():
        ok_set = per_user_ok.get(uid, set())
        resume_from = max_started + 1
        for batch_no in range(1, max_started + 1):
            if batch_no not in ok_set:
                resume_from = batch_no
                break
        if resume_from > 1:
            resume_map[uid] = resume_from

    return resume_map


async def _add_with_retry(memory, messages, user_id, indices, batch_no):
    attempt = 0
    while True:
        try:
            return await memory.add(messages, user_id=user_id)
        except Exception as e:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise RuntimeError(
                    f"batch={batch_no} indices={indices} failed after {MAX_RETRIES + 1} attempts: {e}"
                ) from e
            sleep_s = RETRY_BASE_SECONDS * attempt
            print(
                f"[RETRY] user_id={user_id} batch={batch_no} indices={indices} "
                f"attempt={attempt}/{MAX_RETRIES} sleep={sleep_s:.1f}s error={e}"
            )
            await asyncio.sleep(sleep_s)


async def ingest_one_user(memory, user_sem, file_path, resume_batch_map):
    async with user_sem:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        user_id = str(data.get("user_id") or file_path.stem)
        dialogues = sorted(data.get("dialogues", []), key=lambda x: x.get("dialogue_index", 0))

        print("\n" + "-" * 80)
        print(
            f"[USER] file={file_path.name} user_id={user_id} "
            f"dialogues_total={len(dialogues)}"
        )

        resume_from_batch = int(resume_batch_map.get(user_id, 1))
        if resume_from_batch > 1:
            print(f"[RESUME] user_id={user_id} resume_from_batch={resume_from_batch}")

        ok_batches = 0
        error_batches = 0
        skip_dialogues = 0
        total_batches = 0

        for batch_no, batch in enumerate(_dialogue_batches(dialogues, BATCH_SIZE), start=1):
            total_batches += 1
            if batch_no < resume_from_batch:
                continue

            batch_indices = []
            batch_messages = []
            for item in batch:
                dialogue_index = item.get("dialogue_index")
                cleaned = _clean_dialogue(item.get("dialogue", []))
                if not cleaned:
                    skip_dialogues += 1
                    print(
                        f"[SKIP] user_id={user_id} batch={batch_no} "
                        f"dialogue_index={dialogue_index} empty"
                    )
                    continue

                batch_indices.append(dialogue_index)
                batch_messages.extend(cleaned)

            if not batch_messages:
                print(f"[SKIP] user_id={user_id} batch={batch_no} empty_after_clean")
                continue

            print(
                f"[START] user_id={user_id} batch={batch_no} dialogues={len(batch_indices)} "
                f"indices={batch_indices} total_messages={len(batch_messages)}"
            )
            try:
                result = await _add_with_retry(
                    memory=memory,
                    messages=batch_messages,
                    user_id=user_id,
                    indices=batch_indices,
                    batch_no=batch_no,
                )
                if VERBOSE_RESULTS:
                    print(f"[RESULT] user_id={user_id} batch={batch_no}")
                    if isinstance(result, dict):
                        print(json.dumps(result, ensure_ascii=False, indent=2))
                    else:
                        print(result)
                print(f"[OK] user_id={user_id} batch={batch_no}")
                ok_batches += 1
            except Exception as e:
                print(f"[ERROR] user_id={user_id} {e}")
                error_batches += 1

        # User-level completion marker for a separate log file.
        # "done" means every batch is already confirmed by prior resume or newly succeeded now.
        complete_by_resume = resume_from_batch > total_batches and total_batches > 0
        complete_by_current_run = (error_batches == 0 and ok_batches == total_batches and total_batches > 0)
        user_done = complete_by_resume or complete_by_current_run

        if user_done:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            status = "resume_done" if complete_by_resume else "run_done"
            done_line = (
                f"[USER_DONE] ts={now} file={file_path.name} user_id={user_id} "
                f"status={status} dialogues_total={len(dialogues)} "
                f"batches_total={total_batches} ok_batches={ok_batches} error_batches={error_batches}"
            )
            print(done_line)
            _append_user_done_log(done_line)

        return {
            "file": file_path.name,
            "user_id": user_id,
            "dialogues_total": len(dialogues),
            "batch_size": BATCH_SIZE,
            "batches_total": total_batches,
            "ok_batches": ok_batches,
            "error_batches": error_batches,
            "skip_dialogues": skip_dialogues,
            "resume_from_batch": resume_from_batch,
        }


def _build_target_files():
    if USER_INDEX_END < USER_INDEX_START:
        raise ValueError(
            f"Invalid range: USER_INDEX_START={USER_INDEX_START}, USER_INDEX_END={USER_INDEX_END}"
        )
    return [DATA_DIR / f"user_{idx:04d}.json" for idx in range(USER_INDEX_START, USER_INDEX_END + 1)]


async def main():
    target_files = _build_target_files()
    files = [p for p in target_files if p.exists()]
    missing_files = [p for p in target_files if not p.exists()]
    if not files:
        raise FileNotFoundError(
            f"No files found in user range user_{USER_INDEX_START:04d}..user_{USER_INDEX_END:04d} under {DATA_DIR}"
        )

    memory = AsyncMemory.from_config(CONFIG)
    user_sem = asyncio.Semaphore(MAX_CONCURRENT_USERS)
    resume_batch_map = _build_resume_batch_map_from_log(OUTPUT_LOG_FILE)

    print("\n" + "=" * 80)
    print("ASYNC WRITE")
    print("=" * 80)
    print("spmem_memory path:", spmem_memory.__file__)
    print("data_dir:", DATA_DIR)
    print(f"user_range: user_{USER_INDEX_START:04d}.json -> user_{USER_INDEX_END:04d}.json")
    print("target_files:", [p.name for p in target_files])
    if missing_files:
        print("missing_files:", [p.name for p in missing_files])
    print("files:", [f.name for f in files])
    print("batch_size:", BATCH_SIZE)
    print("max_concurrent_users:", MAX_CONCURRENT_USERS)
    print("max_retries:", MAX_RETRIES)
    print("user_done_log_file:", USER_DONE_LOG_FILE)
    if resume_batch_map:
        print("resume_users:", len(resume_batch_map))
    else:
        print("resume_users: 0")

    t0 = time.perf_counter()
    tasks = [asyncio.create_task(ingest_one_user(memory, user_sem, path, resume_batch_map)) for path in files]
    summaries = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0

    total_ok = sum(s["ok_batches"] for s in summaries)
    total_err = sum(s["error_batches"] for s in summaries)
    total_skip = sum(s["skip_dialogues"] for s in summaries)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        json.dumps(
            {
                "files_total": len(files),
                "batch_size": BATCH_SIZE,
                "max_concurrent_users": MAX_CONCURRENT_USERS,
                "ok_batches_total": total_ok,
                "error_batches_total": total_err,
                "skip_dialogues_total": total_skip,
                "elapsed_seconds_total": round(elapsed, 2),
                "users": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Build SP-Mem memory with configurable storage, batching, and concurrency."
    )
    parser.add_argument("--domain", choices=["finance", "medical", "education", "mental"], default="mental")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--start-user", type=int, default=0)
    parser.add_argument("--end-user", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-concurrent-users", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument(
        "--verbose-results",
        action="store_true",
        help="Print per-batch memory results. Disabled by default because logs may contain sensitive values.",
    )
    parser.add_argument("--collection-name", default="spmem_paper")
    parser.add_argument("--qdrant-url", default=os.getenv("SPMEM_QDRANT_URL") or os.getenv("QDRANT_URL", ""))
    parser.add_argument("--history-db-path", default=str(PROJECT_ROOT / "runs" / "storage" / "history.db"))
    parser.add_argument(
        "--privacy-mapping-dir",
        default=str(PROJECT_ROOT / "runs" / "storage" / "privacy_mappings"),
    )
    parser.add_argument("--output-log-file", default=str(PROJECT_ROOT / "runs" / "logs" / "async_write.log"))
    parser.add_argument(
        "--user-done-log-file",
        default=str(PROJECT_ROOT / "runs" / "logs" / "async_write_users_done.log"),
    )
    parser.add_argument("--memory-llm-model", default=os.getenv("SPMEM_MEMORY_MODEL_ID") or os.getenv("MEMORY_LLM_MODEL", "gpt-5.2-chat"))
    parser.add_argument("--embed-model", default=os.getenv("SPMEM_EMBEDDING_MODEL_ID") or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--embed-dims", type=int, default=int(os.getenv("EMBEDDING_DIMS", "1536")))
    parser.add_argument("--neo4j-url", default=os.getenv("SPMEM_NEO4J_URL") or os.getenv("NEO4J_URL", "neo4j://localhost:7687"))
    parser.add_argument("--neo4j-username", default=os.getenv("SPMEM_NEO4J_USERNAME") or os.getenv("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("SPMEM_NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument("--neo4j-database", default=os.getenv("SPMEM_NEO4J_DATABASE") or os.getenv("NEO4J_DATABASE", "neo4j"))
    return parser.parse_args()


def _apply_args(args):
    global DATA_DIR, USER_INDEX_START, USER_INDEX_END
    global BATCH_SIZE, MAX_CONCURRENT_USERS, MAX_RETRIES, RETRY_BASE_SECONDS, VERBOSE_RESULTS
    global OUTPUT_LOG_FILE, USER_DONE_LOG_FILE, CONFIG

    DATA_DIR = Path(args.data_dir).resolve() if args.data_dir else (
        PROJECT_ROOT / "data" / args.domain / "histories"
    )
    USER_INDEX_START = args.start_user
    USER_INDEX_END = args.end_user
    BATCH_SIZE = args.batch_size
    MAX_CONCURRENT_USERS = args.max_concurrent_users
    MAX_RETRIES = args.max_retries
    RETRY_BASE_SECONDS = args.retry_base_seconds
    VERBOSE_RESULTS = bool(args.verbose_results)
    OUTPUT_LOG_FILE = Path(args.output_log_file).resolve()
    USER_DONE_LOG_FILE = Path(args.user_done_log_file).resolve()

    if USER_INDEX_END < USER_INDEX_START:
        raise ValueError("--end-user must be greater than or equal to --start-user")
    if not str(args.qdrant_url).strip():
        raise ValueError("Qdrant URL is required via --qdrant-url or SPMEM_QDRANT_URL")
    if BATCH_SIZE < 1 or MAX_CONCURRENT_USERS < 1 or MAX_RETRIES < 0:
        raise ValueError("batch size/concurrency must be positive and retries must be non-negative")
    if not args.neo4j_password:
        raise ValueError("Neo4j password is required via --neo4j-password or NEO4J_PASSWORD")

    CONFIG = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": args.memory_llm_model,
                "api_key": os.getenv("SPMEM_MEMORY_API_KEY") or os.getenv("MEMORY_LLM_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
                "openai_base_url": os.getenv("SPMEM_MEMORY_BASE_URL") or os.getenv("MEMORY_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", ""),
                "temperature": 0.0,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": args.embed_model,
                "embedding_dims": args.embed_dims,
                "api_key": os.getenv("SPMEM_EMBEDDING_API_KEY") or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
                "openai_base_url": os.getenv("SPMEM_EMBEDDING_BASE_URL") or os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL", ""),
            },
        },
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": args.neo4j_url,
                "username": args.neo4j_username,
                "password": args.neo4j_password,
                "database": args.neo4j_database,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": args.collection_name,
                "url": args.qdrant_url,
                "embedding_model_dims": args.embed_dims,
            },
        },
        "history_db_path": str(Path(args.history_db_path).resolve()),
        "privacy_mapping_dir": str(Path(args.privacy_mapping_dir).resolve()),
    }


if __name__ == "__main__":
    _apply_args(_parse_args())
    OUTPUT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_LOG_FILE.open("a", encoding="utf-8", buffering=1) as log_fp:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_fp)
        sys.stderr = Tee(original_stderr, log_fp)
        try:
            print("\n" + "#" * 80)
            print("RUN START:", time.strftime("%Y-%m-%d %H:%M:%S"))
            print("log file:", OUTPUT_LOG_FILE)
            asyncio.run(main())
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
