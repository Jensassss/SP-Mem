from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from openai import OpenAI

ALL_EVAL_MODES = (
    "mixed_denied",
    "mixed_allowed",
    "privacy_denied",
    "privacy_allowed",
    "non_privacy_only",
)

MODE_GROUPS = {
    "allowed": ("mixed_allowed", "privacy_allowed", "non_privacy_only"),
    "denied": ("mixed_denied", "privacy_denied"),
    "all_domain": ALL_EVAL_MODES,
}

# Keep mode policy consistent with current single-response evaluator:
# - TC for all modes
# - PQ only for mixed_* and non_privacy_only
MODE_METRICS = {
    "mixed_denied": ("TC", "PQ"),
    "mixed_allowed": ("TC", "PQ"),
    "privacy_denied": ("TC",),
    "privacy_allowed": ("TC",),
    "non_privacy_only": ("TC", "PQ"),
}

PAIRWISE_KEYS = (
    "A_win_count",
    "B_win_count",
    "tie_count",
    "scored_count",
    "missing_count",
    "judge_error_count",
    "swap_check_used_count",
    "swap_agree_count",
    "swap_conflict_count",
    "swap_partial_count",
)

TC_PAIR_PROMPT = """\
You are an impartial judge for pairwise evaluation of conversational assistants.

Compare Assistant A and Assistant B for Task Completion (TC).

Judging principles:
- Evaluate whether the response completes the user task and required deliverables.
- Focus on instruction-following, completeness, relevance, and practical usefulness.
- Do not prefer an answer because it is longer, more formal, or more verbose.
- Do not let position (A vs B) influence your decision.
- Output winner = "A" if Assistant A is clearly better on TC.
- Output winner = "B" if Assistant B is clearly better on TC.
- If both are similarly good or similarly weak, output TIE.


## Task Description
{scenario}

## User Query
{query}

## Required Entities (for completion)
{required_entities}

## Assistant A Response
{response_a}

## Assistant B Response
{response_b}

Return ONLY JSON:
{{"winner":"A|B|TIE","reason":"one concise sentence"}}
"""

PQ_PAIR_PROMPT = """\
You are an impartial judge for pairwise evaluation of conversational assistants.

Compare Assistant A and Assistant B on Personalization Quality (PQ).

Judging principles:
- Evaluate whether the response uses user preferences correctly and meaningfully.
- "Meaningful" means preferences materially shape content, tone, examples, or recommendations.
- Penalize hallucinated, incorrect, or irrelevant preference usage.
- Do not prefer an answer because it is longer, more formal, or more verbose.
- Do not let response position (A vs B) influence your decision.
- Output winner = "A" if Assistant A is clearly better on PQ.
- Output winner = "B" if Assistant B is clearly better on PQ.
- Output winner = "TIE" if both are similarly good or similarly weak.


## Required Preference Types
{preference_entities}

## User Preference Values (Ground Truth)
{preference_values}

## User Query
{query}

## Assistant A Response
{response_a}

## Assistant B Response
{response_b}

Return ONLY JSON:
{{"winner":"A|B|TIE","reason":"one concise sentence"}}
"""


def _safe_slug(text: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    token = token.strip("._-")
    return token or "judge"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_json_loads(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fenced:
        try:
            value = json.loads(fenced.group(1))
            return value if isinstance(value, dict) else {}
        except Exception:
            pass

    brace = re.search(r"(\{[\s\S]*\})", text)
    if brace:
        try:
            value = json.loads(brace.group(1))
            return value if isinstance(value, dict) else {}
        except Exception:
            pass

    return {}


def _variance(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pvariance(values))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _sample_id(row: Dict[str, Any]) -> str:
    return str(row.get("sample_id", "")).strip()


def _mode(row: Dict[str, Any]) -> str:
    return str(row.get("evaluation_mode", row.get("mode", ""))).strip()


def _dialogue_index(row: Dict[str, Any]) -> str:
    for key in ("dialogue_index", "dialog_idx", "turn_index", "query_index"):
        if key in row:
            return str(row.get(key))
    return ""


def _composite_key(row: Dict[str, Any], fallback_idx: int) -> str:
    mode = _mode(row)
    user_id = str(row.get("user_id", "")).strip()
    user_index = str(row.get("user_index", ""))
    d_idx = _dialogue_index(row)
    query = _normalize_text(str(row.get("query", "")))
    identity = user_id or user_index or str(fallback_idx)
    return f"{identity}|{mode}|{d_idx}|{query}"


def _pop_unseen(bucket: Deque[int], used: set[int]) -> Optional[int]:
    while bucket and bucket[0] in used:
        bucket.popleft()
    if not bucket:
        return None
    return bucket.popleft()


def _pair_rows(
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
    *,
    allow_index_fallback: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_sid_b: Dict[str, Deque[int]] = defaultdict(deque)
    by_comp_b: Dict[str, Deque[int]] = defaultdict(deque)
    for idx_b, row_b in enumerate(rows_b):
        sid = _sample_id(row_b)
        if sid:
            by_sid_b[sid].append(idx_b)
        by_comp_b[_composite_key(row_b, idx_b)].append(idx_b)

    used_b: set[int] = set()
    pairs: List[Dict[str, Any]] = []
    unmatched_a: List[int] = []

    for idx_a, row_a in enumerate(rows_a):
        idx_b: Optional[int] = None
        matched_by = ""

        sid = _sample_id(row_a)
        if sid and sid in by_sid_b:
            idx_b = _pop_unseen(by_sid_b[sid], used_b)
            if idx_b is not None:
                matched_by = "sample_id"

        if idx_b is None:
            comp = _composite_key(row_a, idx_a)
            if comp in by_comp_b:
                idx_b = _pop_unseen(by_comp_b[comp], used_b)
                if idx_b is not None:
                    matched_by = "composite"

        if idx_b is None and allow_index_fallback:
            if idx_a < len(rows_b) and idx_a not in used_b:
                idx_b = idx_a
                matched_by = "index"

        if idx_b is None:
            unmatched_a.append(idx_a)
            continue

        used_b.add(idx_b)
        row_b = rows_b[idx_b]
        pairs.append(
            {
                "a": row_a,
                "b": row_b,
                "matched_by": matched_by,
                "pair_key": sid or _composite_key(row_a, idx_a),
            }
        )

    unmatched_b = [i for i in range(len(rows_b)) if i not in used_b]
    matched_by_counts: Dict[str, int] = {"sample_id": 0, "composite": 0, "index": 0}
    for p in pairs:
        k = str(p.get("matched_by", ""))
        if k in matched_by_counts:
            matched_by_counts[k] += 1

    pairing_stats = {
        "rows_a": len(rows_a),
        "rows_b": len(rows_b),
        "paired": len(pairs),
        "unmatched_a": len(unmatched_a),
        "unmatched_b": len(unmatched_b),
        "unmatched_a_indices": unmatched_a[:50],
        "unmatched_b_indices": unmatched_b[:50],
        "allow_index_fallback": bool(allow_index_fallback),
        "matched_by_counts": matched_by_counts,
    }
    return pairs, pairing_stats


def _flatten_profile_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    ignore = {"user_id", "user_index"}
    for key, value in row.items():
        if key in ignore:
            continue
        if isinstance(value, dict):
            out.update(value)
        else:
            out[key] = value
    return out


def _load_profile_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")
    return _read_jsonl(path)


def _build_user_id_row_lookup(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        user_id = str(row.get("user_id", "")).strip()
        if user_id and user_id not in lookup:
            lookup[user_id] = row
    return lookup


def _resolve_user_index(item: Dict[str, Any], fallback_user_index: int) -> int:
    value = item.get("user_index", fallback_user_index)
    try:
        idx = int(value)
    except Exception:
        idx = fallback_user_index
    return idx if idx >= 0 else fallback_user_index


def _build_profile_for_item(
    item: Dict[str, Any],
    *,
    preference_rows: List[Dict[str, Any]],
    preference_rows_by_user_id: Dict[str, Dict[str, Any]],
    fallback_user_index: int,
) -> Dict[str, Any]:
    user_index = _resolve_user_index(item, fallback_user_index)
    user_id = str(item.get("user_id", "")).strip()

    merged: Dict[str, Any] = {}
    if user_id and user_id in preference_rows_by_user_id:
        merged.update(_flatten_profile_row(preference_rows_by_user_id[user_id]))
    elif 0 <= user_index < len(preference_rows):
        merged.update(_flatten_profile_row(preference_rows[user_index]))
    return merged


def _extract_pref_values(item: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    pref_entities = item.get("pref_entities", [])
    if not isinstance(pref_entities, list):
        pref_entities = []
    return {k: profile.get(k, "N/A") for k in pref_entities}


def _normalize_winner(raw: Any) -> Optional[str]:
    token = str(raw or "").strip().upper()
    if token in {"A", "ASSISTANT_A", "RESPONSE_A", "1"}:
        return "A"
    if token in {"B", "ASSISTANT_B", "RESPONSE_B", "2"}:
        return "B"
    if token in {"TIE", "DRAW", "EQUAL", "0"}:
        return "TIE"
    return None


def _flip_winner_ab(winner: Optional[str]) -> Optional[str]:
    if winner == "A":
        return "B"
    if winner == "B":
        return "A"
    if winner == "TIE":
        return "TIE"
    return None


def call_judge(prompt: str, client: OpenAI, model: str, timeout_seconds: float) -> Dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=220,
            timeout=timeout_seconds,
        )
        text = resp.choices[0].message.content or ""
        parsed = _safe_json_loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:
        return {"winner": None, "reason": f"judge_error: {e}"}


def _extract_response_text(item: Dict[str, Any]) -> str:
    return str(item.get("response", item.get("answer", "")))


def _extract_required_entities_text(item_a: Dict[str, Any], item_b: Dict[str, Any]) -> str:
    # Prefer explicit required_entities when available.
    for item in (item_a, item_b):
        req = item.get("required_entities")
        if isinstance(req, list):
            vals = [str(x).strip() for x in req if str(x).strip()]
            if vals:
                return ", ".join(vals)

    # Fallback to privacy_entities + pref_entities for compatibility.
    merged: List[str] = []
    for key in ("privacy_entities", "pref_entities"):
        for item in (item_a, item_b):
            vals = item.get(key, [])
            if isinstance(vals, list):
                for x in vals:
                    token = str(x).strip()
                    if token and token not in merged:
                        merged.append(token)
                if merged:
                    # Keep preference from A when available; don't over-merge from both sides.
                    break
    return ", ".join(merged) if merged else "none"


def _judge_once(
    prompt: str,
    judge_client: OpenAI,
    judge_model: str,
    timeout_seconds: float,
) -> Tuple[Optional[str], str, str]:
    result = call_judge(prompt, judge_client, judge_model, timeout_seconds)
    winner = _normalize_winner(result.get("winner"))
    reason = str(result.get("reason", "")).strip()
    if winner in {"A", "B", "TIE"}:
        return winner, reason, "ok"
    if reason.startswith("judge_error:"):
        return None, reason, "judge_error"
    return None, reason, "parse_error"


def _resolve_with_swap_check(
    *,
    prompt_builder,
    item_a: Dict[str, Any],
    item_b: Dict[str, Any],
    judge_client: OpenAI,
    judge_model: str,
    timeout_seconds: float,
    use_swap_check: bool,
) -> Dict[str, Any]:
    # Forward: A vs B
    prompt_f = prompt_builder(item_a, item_b)
    w_f, r_f, s_f = _judge_once(prompt_f, judge_client, judge_model, timeout_seconds)

    if not use_swap_check:
        return {
            "winner": w_f,
            "reason": r_f,
            "status": s_f,
            "forward_winner": w_f,
            "reverse_winner_raw": None,
            "reverse_winner_canonical": None,
            "swap_check_used": False,
            "swap_agree": None,
        }

    # Reverse: B vs A
    prompt_r = prompt_builder(item_b, item_a)
    w_r_raw, r_r, s_r = _judge_once(prompt_r, judge_client, judge_model, timeout_seconds)
    w_r_canonical = _flip_winner_ab(w_r_raw)

    # Strict conservative swap rule (MT-Bench style):
    # - A wins only if both orders prefer A
    # - B wins only if both orders prefer B
    # - otherwise => TIE
    if w_f == "A" and w_r_canonical == "A":
        return {
            "winner": "A",
            "reason": r_f or r_r,
            "status": "ok_swap_consensus_A",
            "forward_winner": w_f,
            "reverse_winner_raw": w_r_raw,
            "reverse_winner_canonical": w_r_canonical,
            "swap_check_used": True,
            "swap_agree": True,
        }

    if w_f == "B" and w_r_canonical == "B":
        return {
            "winner": "B",
            "reason": r_f or r_r,
            "status": "ok_swap_consensus_B",
            "forward_winner": w_f,
            "reverse_winner_raw": w_r_raw,
            "reverse_winner_canonical": w_r_canonical,
            "swap_check_used": True,
            "swap_agree": True,
        }

    # Any inconsistency / tie / parse issue / one-side failure -> TIE
    if "judge_error" in (s_f, s_r):
        status = "swap_judge_error_tie"
    elif "parse_error" in (s_f, s_r):
        status = "swap_parse_error_tie"
    elif w_f == "TIE" and w_r_canonical == "TIE":
        status = "ok_swap_tie_consensus"
    else:
        status = "ok_swap_conflict_tie"

    return {
        "winner": "TIE",
        "reason": "strict_swap_rule_tie",
        "status": status,
        "forward_winner": w_f,
        "reverse_winner_raw": w_r_raw,
        "reverse_winner_canonical": w_r_canonical,
        "swap_check_used": True,
        "swap_agree": False,
    }


def _judge_tc_pair(
    item_a: Dict[str, Any],
    item_b: Dict[str, Any],
    judge_client: OpenAI,
    judge_model: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    def _build_prompt(x_a: Dict[str, Any], x_b: Dict[str, Any]) -> str:
        ra = _extract_response_text(x_a)
        rb = _extract_response_text(x_b)
        required_entities_text = _extract_required_entities_text(x_a, x_b)
        return TC_PAIR_PROMPT.format(
            scenario=x_a.get("scenario", x_b.get("scenario", "")),
            query=x_a.get("query", x_b.get("query", "")),
            required_entities=required_entities_text,
            response_a=ra,
            response_b=rb,
        )

    return _resolve_with_swap_check(
        prompt_builder=_build_prompt,
        item_a=item_a,
        item_b=item_b,
        judge_client=judge_client,
        judge_model=judge_model,
        timeout_seconds=timeout_seconds,
        use_swap_check=True,
    )


def _judge_pq_pair(
    item_a: Dict[str, Any],
    item_b: Dict[str, Any],
    profile: Dict[str, Any],
    judge_client: OpenAI,
    judge_model: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    pref_entities = item_a.get("pref_entities", item_b.get("pref_entities", []))
    if not isinstance(pref_entities, list):
        pref_entities = []
    pref_values = _extract_pref_values(item_a, profile)

    def _build_prompt(x_a: Dict[str, Any], x_b: Dict[str, Any]) -> str:
        ra = _extract_response_text(x_a)
        rb = _extract_response_text(x_b)
        return PQ_PAIR_PROMPT.format(
            preference_entities=", ".join(pref_entities) or "none",
            preference_values=json.dumps(pref_values, ensure_ascii=False),
            query=x_a.get("query", x_b.get("query", "")),
            response_a=ra,
            response_b=rb,
        )

    return _resolve_with_swap_check(
        prompt_builder=_build_prompt,
        item_a=item_a,
        item_b=item_b,
        judge_client=judge_client,
        judge_model=judge_model,
        timeout_seconds=timeout_seconds,
        use_swap_check=True,
    )


def _score_one_pair(
    pair: Dict[str, Any],
    profile: Dict[str, Any],
    judge_client: OpenAI,
    judge_model: str,
    *,
    score_error_rows: bool,
    timeout_seconds: float,
) -> Dict[str, Any]:
    item_a = pair["a"]
    item_b = pair["b"]
    mode = _mode(item_a) or _mode(item_b)
    metrics = MODE_METRICS.get(mode, ("TC",))

    status_a = str(item_a.get("status", "ok")).strip().lower()
    status_b = str(item_b.get("status", "ok")).strip().lower()
    if (status_a == "error" or status_b == "error") and not score_error_rows:
        out: Dict[str, Any] = {
            "pair_key": pair.get("pair_key", ""),
            "matched_by": pair.get("matched_by", ""),
            "evaluation_mode": mode,
            "sample_id_a": _sample_id(item_a),
            "sample_id_b": _sample_id(item_b),
            "status_a": status_a,
            "status_b": status_b,
            "skipped": True,
            "skip_reason": "status=error",
        }
        for m in metrics:
            out[f"{m}_winner"] = None
            out[f"{m}_reason"] = ""
            out[f"{m}_status"] = "skipped_error_row"
        return out

    out = {
        "pair_key": pair.get("pair_key", ""),
        "matched_by": pair.get("matched_by", ""),
        "evaluation_mode": mode,
        "sample_id_a": _sample_id(item_a),
        "sample_id_b": _sample_id(item_b),
        "status_a": status_a,
        "status_b": status_b,
        "skipped": False,
        "query": item_a.get("query", item_b.get("query", "")),
    }

    if "TC" in metrics:
        rs = _judge_tc_pair(item_a, item_b, judge_client, judge_model, timeout_seconds)
        out["TC_winner"] = rs.get("winner")
        out["TC_reason"] = rs.get("reason", "")
        out["TC_status"] = rs.get("status", "")
        out["TC_forward_winner"] = rs.get("forward_winner")
        out["TC_reverse_winner_raw"] = rs.get("reverse_winner_raw")
        out["TC_reverse_winner_canonical"] = rs.get("reverse_winner_canonical")
        out["TC_swap_check_used"] = rs.get("swap_check_used")
        out["TC_swap_agree"] = rs.get("swap_agree")

    if "PQ" in metrics:
        rs = _judge_pq_pair(item_a, item_b, profile, judge_client, judge_model, timeout_seconds)
        out["PQ_winner"] = rs.get("winner")
        out["PQ_reason"] = rs.get("reason", "")
        out["PQ_status"] = rs.get("status", "")
        out["PQ_forward_winner"] = rs.get("forward_winner")
        out["PQ_reverse_winner_raw"] = rs.get("reverse_winner_raw")
        out["PQ_reverse_winner_canonical"] = rs.get("reverse_winner_canonical")
        out["PQ_swap_check_used"] = rs.get("swap_check_used")
        out["PQ_swap_agree"] = rs.get("swap_agree")

    return out


def _aggregate_run(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode: Dict[str, Any] = {}
    for mode in ALL_EVAL_MODES:
        mode_rows = [r for r in rows if str(r.get("evaluation_mode", "")).strip() == mode]
        mode_out: Dict[str, Any] = {
            "count": len(mode_rows),
            "non_error_count": sum(1 for r in mode_rows if not bool(r.get("skipped", False))),
            "error_count": sum(1 for r in mode_rows if bool(r.get("skipped", False))),
        }

        metrics = MODE_METRICS.get(mode, ("TC",))
        for m in metrics:
            winners = [r.get(f"{m}_winner") for r in mode_rows]
            statuses = [str(r.get(f"{m}_status", "")) for r in mode_rows]
            a_win = sum(1 for w in winners if w == "A")
            b_win = sum(1 for w in winners if w == "B")
            tie = sum(1 for w in winners if w == "TIE")
            scored = a_win + b_win + tie
            missing = sum(1 for w in winners if w not in {"A", "B", "TIE"})
            judge_error = sum(1 for s in statuses if "judge_error" in s)
            swap_used = sum(1 for r in mode_rows if bool(r.get(f"{m}_swap_check_used", False)))
            swap_agree = sum(1 for r in mode_rows if r.get(f"{m}_swap_agree") is True)
            swap_conflict = sum(1 for s in statuses if s == "ok_swap_conflict_tie")
            swap_partial = sum(
                1
                for s in statuses
                if s.startswith("ok_forward_only_") or s.startswith("ok_reverse_only_")
            )

            mode_out[f"{m}_A_win_count"] = a_win
            mode_out[f"{m}_B_win_count"] = b_win
            mode_out[f"{m}_tie_count"] = tie
            mode_out[f"{m}_scored_count"] = scored
            mode_out[f"{m}_missing_count"] = missing
            mode_out[f"{m}_judge_error_count"] = judge_error
            mode_out[f"{m}_swap_check_used_count"] = swap_used
            mode_out[f"{m}_swap_agree_count"] = swap_agree
            mode_out[f"{m}_swap_conflict_count"] = swap_conflict
            mode_out[f"{m}_swap_partial_count"] = swap_partial

            if scored > 0:
                mode_out[f"{m}_A_win_rate"] = a_win / scored
                mode_out[f"{m}_B_win_rate"] = b_win / scored
                mode_out[f"{m}_tie_rate"] = tie / scored
                mode_out[f"{m}_A_pair_score"] = (a_win + 0.5 * tie) / scored
                mode_out[f"{m}_B_pair_score"] = (b_win + 0.5 * tie) / scored
                mode_out[f"{m}_swap_agree_rate"] = swap_agree / scored
            else:
                mode_out[f"{m}_A_win_rate"] = None
                mode_out[f"{m}_B_win_rate"] = None
                mode_out[f"{m}_tie_rate"] = None
                mode_out[f"{m}_A_pair_score"] = None
                mode_out[f"{m}_B_pair_score"] = None
                mode_out[f"{m}_swap_agree_rate"] = None

        by_mode[mode] = mode_out
    return by_mode


def _aggregate_mode_metric(
    run_summaries: List[Dict[str, Any]],
    *,
    mode: str,
    metric_key: str,
) -> Tuple[Optional[float], Optional[float]]:
    values: List[float] = []
    for rs in run_summaries:
        mode_obj = rs.get(mode, {})
        value = mode_obj.get(metric_key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None, None
    return float(sum(values) / len(values)), _variance(values)


def _sum_numeric_present(dicts: List[Dict[str, Any]], key: str) -> Optional[float]:
    total = 0.0
    seen = False
    for d in dicts:
        v = d.get(key)
        if isinstance(v, (int, float)):
            total += float(v)
            seen = True
    return total if seen else None


def _build_group_run_summary(
    run_summary: Dict[str, Any],
    modes: Tuple[str, ...],
) -> Dict[str, Any]:
    mode_objs = [run_summary.get(m, {}) for m in modes]
    out: Dict[str, Any] = {}

    for key in ("count", "non_error_count", "error_count"):
        out[key] = _sum_numeric_present(mode_objs, key) or 0.0

    for metric in ("TC", "PQ"):
        metric_seen = False
        for key in PAIRWISE_KEYS:
            agg_v = _sum_numeric_present(mode_objs, f"{metric}_{key}")
            if agg_v is not None:
                out[f"{metric}_{key}"] = agg_v
                metric_seen = True

        if not metric_seen:
            continue

        scored = float(out.get(f"{metric}_scored_count", 0.0) or 0.0)
        a_win = float(out.get(f"{metric}_A_win_count", 0.0) or 0.0)
        b_win = float(out.get(f"{metric}_B_win_count", 0.0) or 0.0)
        tie = float(out.get(f"{metric}_tie_count", 0.0) or 0.0)
        swap_agree = float(out.get(f"{metric}_swap_agree_count", 0.0) or 0.0)

        if scored > 0:
            out[f"{metric}_A_win_rate"] = a_win / scored
            out[f"{metric}_B_win_rate"] = b_win / scored
            out[f"{metric}_tie_rate"] = tie / scored
            out[f"{metric}_A_pair_score"] = (a_win + 0.5 * tie) / scored
            out[f"{metric}_B_pair_score"] = (b_win + 0.5 * tie) / scored
            out[f"{metric}_swap_agree_rate"] = swap_agree / scored
        else:
            out[f"{metric}_A_win_rate"] = None
            out[f"{metric}_B_win_rate"] = None
            out[f"{metric}_tie_rate"] = None
            out[f"{metric}_A_pair_score"] = None
            out[f"{metric}_B_pair_score"] = None
            out[f"{metric}_swap_agree_rate"] = None

    return out


def _aggregate_group_metric(
    group_run_summaries: List[Dict[str, Any]],
    metric_key: str,
) -> Tuple[Optional[float], Optional[float]]:
    values: List[float] = []
    for rs in group_run_summaries:
        value = rs.get(metric_key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None, None
    return float(sum(values) / len(values)), _variance(values)


def _aggregate_runs(run_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for mode in ALL_EVAL_MODES:
        result[mode] = {}
        base_keys = ("count", "non_error_count", "error_count")
        for key in base_keys:
            mean_v, var_v = _aggregate_mode_metric(run_summaries, mode=mode, metric_key=key)
            result[mode][f"{key}_mean_over_runs"] = mean_v
            result[mode][f"{key}_var_over_runs"] = var_v

        metrics = MODE_METRICS.get(mode, ("TC",))
        for m in metrics:
            for key in (
                "A_win_count",
                "B_win_count",
                "tie_count",
                "scored_count",
                "missing_count",
                "judge_error_count",
                "swap_check_used_count",
                "swap_agree_count",
                "swap_conflict_count",
                "swap_partial_count",
                "A_win_rate",
                "B_win_rate",
                "tie_rate",
                "A_pair_score",
                "B_pair_score",
                "swap_agree_rate",
            ):
                mean_v, var_v = _aggregate_mode_metric(
                    run_summaries,
                    mode=mode,
                    metric_key=f"{m}_{key}",
                )
                result[mode][f"{m}_{key}_mean_over_runs"] = mean_v
                result[mode][f"{m}_{key}_var_over_runs"] = var_v

    # Group-level rollups: allowed / denied / all_domain
    for group_name, group_modes in MODE_GROUPS.items():
        group_run_summaries = [_build_group_run_summary(rs, group_modes) for rs in run_summaries]
        result[group_name] = {}

        for key in ("count", "non_error_count", "error_count"):
            mean_v, var_v = _aggregate_group_metric(group_run_summaries, key)
            result[group_name][f"{key}_mean_over_runs"] = mean_v
            result[group_name][f"{key}_var_over_runs"] = var_v

        for m in ("TC", "PQ"):
            for key in (
                *PAIRWISE_KEYS,
                "A_win_rate",
                "B_win_rate",
                "tie_rate",
                "A_pair_score",
                "B_pair_score",
                "swap_agree_rate",
            ):
                mean_v, var_v = _aggregate_group_metric(group_run_summaries, f"{m}_{key}")
                result[group_name][f"{m}_{key}_mean_over_runs"] = mean_v
                result[group_name][f"{m}_{key}_var_over_runs"] = var_v
    return result


def _judge_config_from_env() -> Dict[str, str]:
    return {
        "name": "gpt-4.1",
        "model": os.getenv("JUDGE_GPT41_MODEL", "gpt-4.1"),
        "base_url": os.getenv("JUDGE_GPT41_BASE_URL") or os.getenv("OPENAI_BASE_URL", ""),
        "api_key": os.getenv("JUDGE_GPT41_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pair-wise evaluate two response files on TC/PQ.")
    parser.add_argument("--response-file-a", required=True, help="Response JSONL file A.")
    parser.add_argument("--response-file-b", required=True, help="Response JSONL file B.")
    parser.add_argument(
        "--name-a",
        default="A",
        help="Display name for side A.",
    )
    parser.add_argument(
        "--name-b",
        default="B",
        help="Display name for side B.",
    )
    parser.add_argument(
        "--preference-profile-file",
        default=None,
        help="Ground-truth preference jsonl (line-index aligned by user_index).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "eval_outputs" / "scores_pairwise"),
        help="Output directory.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated judge runs.")
    parser.add_argument(
        "--fallback-user-index",
        type=int,
        default=0,
        help="Fallback user index when one row has invalid user_index.",
    )
    parser.add_argument(
        "--score-error-rows",
        action="store_true",
        help="If set, rows with status=error are still sent to judge.",
    )
    parser.add_argument(
        "--strict-pairing",
        action="store_true",
        help="Disable index fallback pairing. Only sample_id/composite(query+mode+user+turn) matches are allowed.",
    )
    parser.add_argument(
        "--judge-timeout-seconds",
        type=float,
        default=90.0,
        help="Timeout (seconds) for each judge API call.",
    )
    args = parser.parse_args()

    response_a = Path(args.response_file_a).resolve()
    response_b = Path(args.response_file_b).resolve()
    output_dir = Path(args.output_dir).resolve()
    preference_profile_path = Path(args.preference_profile_file).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_a = _read_jsonl(response_a)
    rows_b = _read_jsonl(response_b)
    if not rows_a:
        raise ValueError(f"Empty response file A: {response_a}")
    if not rows_b:
        raise ValueError(f"Empty response file B: {response_b}")

    pairs, pairing_stats = _pair_rows(
        rows_a,
        rows_b,
        allow_index_fallback=not bool(args.strict_pairing),
    )
    if not pairs:
        raise ValueError(f"No matched pairs between files:\nA={response_a}\nB={response_b}")

    preference_rows = _load_profile_rows(preference_profile_path)
    preference_rows_by_user_id = _build_user_id_row_lookup(preference_rows)

    judge = _judge_config_from_env()
    if not judge["base_url"] or not judge["api_key"]:
        raise ValueError(
            "Judge base_url/api_key missing. Set JUDGE_GPT41_BASE_URL/JUDGE_GPT41_API_KEY "
            "or OPENAI_BASE_URL/OPENAI_API_KEY."
        )

    judge_name = _safe_slug(judge["name"])
    judge_dir = output_dir / judge_name
    judge_dir.mkdir(parents=True, exist_ok=True)
    judge_client = OpenAI(base_url=judge["base_url"], api_key=judge["api_key"])

    print("=" * 80)
    print("Pair-wise Evaluate TC/PQ")
    print("response_file_a:", response_a)
    print("response_file_b:", response_b)
    print("name_a:", args.name_a)
    print("name_b:", args.name_b)
    print("paired:", pairing_stats["paired"])
    print("pairing_stats:", pairing_stats)
    print("preference_profile_file:", preference_profile_path)
    print("judge_name:", judge["name"])
    print("judge_model:", judge["model"])
    print("runs:", args.runs)
    print("score_error_rows:", args.score_error_rows)
    print("judge_timeout_seconds:", args.judge_timeout_seconds)
    print("=" * 80)

    run_summaries: List[Dict[str, Any]] = []
    run_files: List[str] = []

    for run_idx in range(1, int(args.runs) + 1):
        scored_rows: List[Dict[str, Any]] = []
        run_path = judge_dir / f"run_{run_idx}_pairwise.jsonl"
        run_summary_path = judge_dir / f"run_{run_idx}_summary.json"
        with run_path.open("w", encoding="utf-8") as run_file:
            total_pairs = len(pairs)
            for idx, pair in enumerate(pairs, start=1):
                profile = _build_profile_for_item(
                    pair["a"],
                    preference_rows=preference_rows,
                    preference_rows_by_user_id=preference_rows_by_user_id,
                    fallback_user_index=int(args.fallback_user_index),
                )
                scored = _score_one_pair(
                    pair,
                    profile,
                    judge_client,
                    judge["model"],
                    score_error_rows=bool(args.score_error_rows),
                    timeout_seconds=float(args.judge_timeout_seconds),
                )
                scored_rows.append(scored)
                run_file.write(json.dumps(scored, ensure_ascii=False) + "\n")
                run_file.flush()
                if idx % 10 == 0 or idx == total_pairs:
                    print(f"[RUN {run_idx}] progress={idx}/{total_pairs}")

        run_summary = _aggregate_run(scored_rows)
        _write_json(run_summary_path, run_summary)
        run_summaries.append(run_summary)
        run_files.append(str(run_path))
        print(f"[RUN {run_idx}] rows={len(scored_rows)} summary={run_summary_path}")

    aggregate = _aggregate_runs(run_summaries)
    summary_over_runs = {
        "judge_name": judge["name"],
        "judge_model": judge["model"],
        "side_a": str(args.name_a),
        "side_b": str(args.name_b),
        "response_file_a": str(response_a),
        "response_file_b": str(response_b),
        "preference_profile_file": str(preference_profile_path),
        "runs": int(args.runs),
        "score_error_rows": bool(args.score_error_rows),
        "pairing": pairing_stats,
        "run_files": run_files,
        "aggregate": aggregate,
    }

    summary_path = judge_dir / "summary_over_runs.json"
    _write_json(summary_path, summary_over_runs)
    print("=" * 80)
    print("summary_over_runs:", summary_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
