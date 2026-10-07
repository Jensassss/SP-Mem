from __future__ import annotations

# Migrated from the experiment tree; preserves concurrent per-file-pair judge scheduling.
# See docs/original_workflow.md for provenance and public adaptations.

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


MODES_ORDER = [
    "mixed_denied",
    "mixed_allowed",
    "privacy_denied",
    "privacy_allowed",
    "non_privacy_only",
]

GROUP_ORDER = [
    "allowed",
    "denied",
    "all_domain",
]


@dataclass
class EvalTask:
    file_a: Path
    file_b: Path
    output_dir: Path
    out_log: Path
    err_log: Path


@dataclass
class EvalResult:
    file_a: Path
    file_b: Path
    exit_code: int
    elapsed_seconds: float
    output_dir: Path
    out_log: Path
    err_log: Path
    status: str  # done / fail
    message: str


def _pair_key_from_stem(stem: str) -> str:
    # Prefer user-based key so files with different suffixes can still be paired.
    # Normalize all of these to the same key:
    # - user0000_llama318b
    # - user0_llm_only_xxx
    # - education_user0
    # - medical_user31
    m = re.search(r"user[_-]?(\d{1,4})", stem, flags=re.IGNORECASE)
    if m:
        user_idx = int(m.group(1))
        return f"user{user_idx:04d}"
    return stem.lower()


def _build_parser() -> argparse.ArgumentParser:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Batch pair-wise evaluate JSONL file pairs on TC/PQ with a configurable judge model."
    )
    parser.add_argument("--input-dir-a", required=True, help="Directory A containing response JSONL files.")
    parser.add_argument("--input-dir-b", required=True, help="Directory B containing response JSONL files.")
    parser.add_argument("--glob", default="*.jsonl", help="Glob pattern under both input dirs.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root output directory; each file pair writes to output-root/<jsonl_stem>/",
    )
    parser.add_argument("--max-parallel", type=int, default=10)
    parser.add_argument("--root-path", default=str(root_default))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--judge-timeout-seconds", type=float, default=90.0)
    parser.add_argument(
        "--preference-profile-file",
        required=True,
    )
    parser.add_argument(
        "--judge-name",
        default=os.getenv("SPMEM_JUDGE_NAME", ""),
        help="Label used for the judge output directory; defaults to --judge-model.",
    )
    parser.add_argument("--judge-model", default=os.getenv("SPMEM_JUDGE_MODEL_ID", "gpt-4.1"))
    parser.add_argument("--judge-base-url", default=os.getenv("SPMEM_JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--judge-api-key", default=os.getenv("SPMEM_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--name-a",
        default="A",
        help="Display name for side A in pair-wise judging.",
    )
    parser.add_argument(
        "--name-b",
        default="B",
        help="Display name for side B in pair-wise judging.",
    )
    parser.add_argument(
        "--progress-log-file",
        default="",
        help=(
            "Realtime progress log path. Default: "
            "<output_root>/batch_pairwise_progress.txt"
        ),
    )
    parser.add_argument(
        "--keep-run-summaries",
        action="store_true",
        help="Keep run_1_summary.json/run_2_summary.json/run_3_summary.json (default removes them).",
    )
    parser.add_argument(
        "--score-error-rows",
        action="store_true",
        help="If set, rows with status=error are still sent to judge.",
    )
    parser.add_argument(
        "--strict-pairing",
        action="store_true",
        help="Pass through to evaluator: disable index fallback pairing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run file pairs even if <output_dir>/<judge>/summary_over_runs.json already exists.",
    )
    return parser


def _safe_read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip()).strip("-._")
    return slug or "judge"


def _safe_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _build_compact_from_summary(summary_path: Path) -> Optional[Dict]:
    if not summary_path.exists():
        return None
    summary = _safe_read_json(summary_path)
    aggregate = summary.get("aggregate", {})
    if not isinstance(aggregate, dict):
        return None

    compact = {
        "judge_name": summary.get("judge_name", "judge"),
        "judge_model": summary.get("judge_model", "unknown"),
        "runs": summary.get("runs"),
        "side_a": summary.get("side_a", "A"),
        "side_b": summary.get("side_b", "B"),
        "modes": {},
    }

    for mode in [*MODES_ORDER, *GROUP_ORDER]:
        mode_payload = aggregate.get(mode, {})
        if not isinstance(mode_payload, dict):
            continue
        quality: Dict[str, object] = {}
        diagnostics: Dict[str, object] = {}
        for key, value in mode_payload.items():
            if key.startswith("TC_") or key.startswith("PQ_"):
                quality[key] = value
            else:
                diagnostics[key] = value
        compact["modes"][mode] = {
            "quality": quality,
            "diagnostics": diagnostics,
        }

    return compact


def _postprocess_outputs(
    *,
    output_dir: Path,
    judge_slug: str,
    keep_run_summaries: bool,
) -> Optional[Path]:
    judge_dir = output_dir / judge_slug
    summary_path = judge_dir / "summary_over_runs.json"
    compact = _build_compact_from_summary(summary_path)
    if compact is None:
        return None

    compact_path = judge_dir / "mode_metrics_compact.json"
    _safe_write_json(compact_path, compact)

    if not keep_run_summaries:
        for p in sorted(judge_dir.glob("run_*_summary.json")):
            try:
                p.unlink()
            except Exception:
                pass

    return compact_path


def _run_one_eval(
    *,
    task: EvalTask,
    python_bin: str,
    eval_script: Path,
    preference_profile_file: str,
    runs: int,
    judge_timeout_seconds: float,
    judge_name: str,
    judge_slug: str,
    judge_model: str,
    judge_base_url: str,
    judge_api_key: str,
    keep_run_summaries: bool,
    score_error_rows: bool,
    strict_pairing: bool,
    name_a: str,
    name_b: str,
) -> EvalResult:
    cmd: List[str] = [
        python_bin,
        "-u",
        str(eval_script),
        "--response-file-a",
        str(task.file_a),
        "--response-file-b",
        str(task.file_b),
        "--name-a",
        str(name_a),
        "--name-b",
        str(name_b),
        "--preference-profile-file",
        str(preference_profile_file),
        "--runs",
        str(runs),
        "--judge-timeout-seconds",
        str(judge_timeout_seconds),
        "--output-dir",
        str(task.output_dir),
    ]
    if score_error_rows:
        cmd.append("--score-error-rows")
    if strict_pairing:
        cmd.append("--strict-pairing")

    env = os.environ.copy()
    env["SPMEM_JUDGE_NAME"] = judge_name
    env["SPMEM_JUDGE_MODEL_ID"] = judge_model
    env["SPMEM_JUDGE_BASE_URL"] = judge_base_url
    env["SPMEM_JUDGE_API_KEY"] = judge_api_key
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    task.output_dir.mkdir(parents=True, exist_ok=True)
    task.out_log.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with task.out_log.open("w", encoding="utf-8", errors="replace") as out_fp, task.err_log.open(
        "w", encoding="utf-8", errors="replace"
    ) as err_fp:
        proc = subprocess.run(cmd, stdout=out_fp, stderr=err_fp, env=env, check=False)
    elapsed = time.perf_counter() - started

    if proc.returncode != 0:
        return EvalResult(
            file_a=task.file_a,
            file_b=task.file_b,
            exit_code=int(proc.returncode),
            elapsed_seconds=float(elapsed),
            output_dir=task.output_dir,
            out_log=task.out_log,
            err_log=task.err_log,
            status="fail",
            message="evaluate_pairwise_tc_pq.py failed",
        )

    compact_path = _postprocess_outputs(
        output_dir=task.output_dir,
        judge_slug=judge_slug,
        keep_run_summaries=keep_run_summaries,
    )
    if compact_path is None:
        return EvalResult(
            file_a=task.file_a,
            file_b=task.file_b,
            exit_code=int(proc.returncode),
            elapsed_seconds=float(elapsed),
            output_dir=task.output_dir,
            out_log=task.out_log,
            err_log=task.err_log,
            status="fail",
            message="summary_over_runs.json missing after pair-wise run",
        )

    return EvalResult(
        file_a=task.file_a,
        file_b=task.file_b,
        exit_code=int(proc.returncode),
        elapsed_seconds=float(elapsed),
        output_dir=task.output_dir,
        out_log=task.out_log,
        err_log=task.err_log,
        status="done",
        message=f"compact={compact_path}",
    )


def main() -> None:
    args = _build_parser().parse_args()
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")
    if not args.judge_base_url or not args.judge_api_key:
        raise ValueError("judge base_url/api_key are required. Set --judge-base-url/--judge-api-key or OPENAI env.")

    root = Path(args.root_path).resolve()
    judge_name = str(args.judge_name).strip() or str(args.judge_model)
    judge_slug = _safe_slug(judge_name)
    eval_script = root / "eval" / "evaluate_pairwise_tc_pq.py"
    input_dir_a = Path(args.input_dir_a).resolve()
    input_dir_b = Path(args.input_dir_b).resolve()
    output_root = Path(args.output_root).resolve()
    logs_root = output_root / "logs_pairwise"

    progress_log_file = (
        Path(args.progress_log_file).resolve()
        if str(args.progress_log_file).strip()
        else (output_root / "batch_pairwise_progress.txt").resolve()
    )

    if not eval_script.exists():
        raise FileNotFoundError(f"pair-wise evaluate script not found: {eval_script}")
    if not input_dir_a.exists():
        raise FileNotFoundError(f"input_dir_a not found: {input_dir_a}")
    if not input_dir_b.exists():
        raise FileNotFoundError(f"input_dir_b not found: {input_dir_b}")

    output_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    progress_log_file.parent.mkdir(parents=True, exist_ok=True)

    files_a: Dict[str, Path] = {}
    files_b: Dict[str, Path] = {}
    dup_a: List[str] = []
    dup_b: List[str] = []

    for p in sorted(input_dir_a.glob(args.glob)):
        if not p.is_file():
            continue
        k = _pair_key_from_stem(p.stem)
        if k in files_a:
            dup_a.append(k)
            continue
        files_a[k] = p

    for p in sorted(input_dir_b.glob(args.glob)):
        if not p.is_file():
            continue
        k = _pair_key_from_stem(p.stem)
        if k in files_b:
            dup_b.append(k)
            continue
        files_b[k] = p

    common_stems = sorted(set(files_a.keys()).intersection(files_b.keys()))

    tasks: List[EvalTask] = []
    skips = 0
    missing_in_b = sorted(set(files_a.keys()) - set(files_b.keys()))
    missing_in_a = sorted(set(files_b.keys()) - set(files_a.keys()))

    with progress_log_file.open("w", encoding="utf-8") as progress_fp:
        if missing_in_b:
            line = f"[WARN] missing in B ({len(missing_in_b)}): {missing_in_b[:20]}"
            print(line)
            progress_fp.write(line + "\n")
        if missing_in_a:
            line = f"[WARN] missing in A ({len(missing_in_a)}): {missing_in_a[:20]}"
            print(line)
            progress_fp.write(line + "\n")
        if dup_a:
            line = f"[WARN] duplicate pair keys in A ignored ({len(dup_a)}): {dup_a[:20]}"
            print(line)
            progress_fp.write(line + "\n")
        if dup_b:
            line = f"[WARN] duplicate pair keys in B ignored ({len(dup_b)}): {dup_b[:20]}"
            print(line)
            progress_fp.write(line + "\n")

        for stem in common_stems:
            file_a = files_a[stem]
            file_b = files_b[stem]
            output_dir = output_root / stem
            summary_path = output_dir / judge_slug / "summary_over_runs.json"
            if summary_path.exists() and not bool(args.overwrite):
                line = f"[SKIP] exists: {summary_path}"
                print(line)
                progress_fp.write(line + "\n")
                skips += 1
                continue

            tasks.append(
                EvalTask(
                    file_a=file_a,
                    file_b=file_b,
                    output_dir=output_dir,
                    out_log=logs_root / f"{stem}.out.log",
                    err_log=logs_root / f"{stem}.err.log",
                )
            )

        total = len(tasks)
        header = (
            f"START files_a={len(files_a)} files_b={len(files_b)} common={len(common_stems)} "
            f"scheduled={total} skipped={skips} parallel={args.max_parallel} judge_model={args.judge_model}"
        )
        print("=" * 72)
        print("Batch Pair-wise Evaluate (TC/PQ)")
        print(f"eval_script: {eval_script}")
        print(f"input_dir_a: {input_dir_a}")
        print(f"input_dir_b: {input_dir_b}")
        print(f"glob: {args.glob}")
        print(f"output_root: {output_root}")
        print(f"logs_root: {logs_root}")
        print(f"progress_log_file: {progress_log_file}")
        print(f"name_a: {args.name_a}")
        print(f"name_b: {args.name_b}")
        print(f"judge_name: {judge_name}")
        print(header)
        print("=" * 72)
        progress_fp.write(header + "\n")
        progress_fp.flush()

        if total == 0:
            print("[INFO] no file pairs scheduled.")
            progress_fp.write("FINISHED success=0 fail=0 total=0\n")
            progress_fp.flush()
            return

        done = 0
        fail = 0
        completed = 0

        with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
            future_map = {
                pool.submit(
                    _run_one_eval,
                    task=task,
                    python_bin=args.python_bin,
                    eval_script=eval_script,
                    preference_profile_file=args.preference_profile_file,
                    runs=int(args.runs),
                    judge_timeout_seconds=float(args.judge_timeout_seconds),
                    judge_name=judge_name,
                    judge_slug=judge_slug,
                    judge_model=args.judge_model,
                    judge_base_url=args.judge_base_url,
                    judge_api_key=args.judge_api_key,
                    keep_run_summaries=bool(args.keep_run_summaries),
                    score_error_rows=bool(args.score_error_rows),
                    strict_pairing=bool(args.strict_pairing),
                    name_a=str(args.name_a),
                    name_b=str(args.name_b),
                ): task
                for task in tasks
            }

            pbar = tqdm(total=total, desc="Pairwise Files", unit="file") if tqdm is not None else None
            for fut in as_completed(future_map):
                result = fut.result()
                completed += 1
                if result.status == "done":
                    done += 1
                    line = (
                        f"[DONE] {result.file_a.name} vs {result.file_b.name} "
                        f"elapsed={result.elapsed_seconds:.1f}s out_dir={result.output_dir} "
                        f"completed={completed}/{total} done={done} fail={fail}"
                    )
                else:
                    fail += 1
                    line = (
                        f"[FAIL] {result.file_a.name} vs {result.file_b.name} "
                        f"elapsed={result.elapsed_seconds:.1f}s msg={result.message} "
                        f"out={result.out_log} err={result.err_log} "
                        f"completed={completed}/{total} done={done} fail={fail}"
                    )

                print(line)
                progress_fp.write(line + "\n")
                progress_fp.flush()
                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

        tail = f"FINISHED success={done} fail={fail} total={total}"
        print("=" * 72)
        print(tail)
        print("=" * 72)
        progress_fp.write(tail + "\n")
        progress_fp.flush()


if __name__ == "__main__":
    main()
