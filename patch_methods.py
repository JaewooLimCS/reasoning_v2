"""
patch_methods.py — 기존 수집 데이터에서 특정 method만 재실행
================================================================
프롬프트/로직 수정 후 전체 재수집 없이 문제가 있던 method만 교체.

Usage:
    python patch_methods.py                              # GSM8K (기본)
    python patch_methods.py --benchmark hotpotqa          # HotPotQA
    python patch_methods.py --benchmark bbh               # BBH
    python patch_methods.py --dry-run                     # API 호출 없이 테스트
    python patch_methods.py --ids 0,1,2                   # 특정 collected_id만
    python patch_methods.py --methods self_discover       # 특정 method만
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from models.gpt import call as model_call
from collect import (
    run_self_refine,
    run_self_discover,
    run_least_to_most,
    is_correct,
)

# ── 설정 ──────────────────────────────────────────────────────────────────────

ALL_METHODS = ["self_refine", "self_discover", "least_to_most"]


def get_config(benchmark: str):
    """벤치마크별 경로와 runner 반환."""
    collected_path = os.path.join(BASE_DIR, "collected", f"{benchmark}_collected.json")

    runners = {
        "self_refine":   lambda prob: run_self_refine(prob, model_call, benchmark),
        "self_discover": lambda prob: run_self_discover(prob, model_call, benchmark),
        "least_to_most": lambda prob: run_least_to_most(prob, model_call, benchmark),
    }

    return collected_path, runners


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def make_problem_dict(item: dict) -> dict:
    """collected item에서 runner에 넘길 problem dict 구성."""
    return {
        "question": item["question"],
        "answer":   item["answer"],
        "subtask":  item.get("subtask", ""),
        "context":  item.get("context", ""),
    }


def summarize_result(method: str, result: dict, item: dict) -> str:
    """한 줄 요약 문자열 생성."""
    correct = result.get("correct", False)
    elapsed = result.get("elapsed_seconds", 0)
    mark = "✓" if correct else "✗"

    detail = ""
    if method == "self_refine":
        iters = result.get("iterations", [])
        n_iter = result.get("total_iterations", len(iters))
        early = any(it.get("stopped_early") for it in iters)
        detail = f"iter={n_iter}" + (" early_stop" if early else "")
    elif method == "self_discover":
        struct = result.get("stage1", {}).get("reasoning_structure", "")
        detail = f"structure={len(struct)}chars"
    elif method == "least_to_most":
        n_sq = len(result.get("sub_questions", []))
        detail = f"sub_q={n_sq}"

    return f"  #{item['collected_id']:>2d}  {mark} pred={result.get('predicted_answer')} gold={item['answer']}  ({detail})  {elapsed:.1f}s"


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Patch specific methods in collected data")
    parser.add_argument("--benchmark", choices=["gsm8k", "hotpotqa", "bbh"],
                        default="gsm8k", help="벤치마크 선택")
    parser.add_argument("--methods", type=str, default=None,
                        help="패치할 method (예: self_discover 또는 self_discover,self_refine)")
    parser.add_argument("--dry-run", action="store_true",
                        help="API 호출 없이 파이프라인 테스트")
    parser.add_argument("--ids", type=str, default=None,
                        help="특정 collected_id만 (예: 0,1,2)")
    args = parser.parse_args()

    benchmark = args.benchmark
    COLLECTED_PATH, RUNNERS = get_config(benchmark)

    # method 필터
    if args.methods:
        patch_methods = [m.strip() for m in args.methods.split(",")]
    else:
        patch_methods = ALL_METHODS

    # 1. 로드
    with open(COLLECTED_PATH, "r") as f:
        data = json.load(f)
    print(f"=== patch_methods.py — {benchmark.upper()} method re-collection ===")
    print(f"Loaded {len(data)} problems from {COLLECTED_PATH}")
    print(f"Methods to patch: {patch_methods}")

    # ID 필터
    all_ids = [item["collected_id"] for item in data]
    if args.ids:
        id_filter = set(int(x) for x in args.ids.split(","))
        print(f"ID filter: {sorted(id_filter)}")
    else:
        id_filter = None

    # 2. 백업
    backup_path = COLLECTED_PATH.replace(".json", ".backup.json")
    shutil.copy2(COLLECTED_PATH, backup_path)
    print(f"Backup: {backup_path}")
    print()

    # 3. method별 재실행
    total_stats = {}

    for method in patch_methods:
        if method not in RUNNERS:
            print(f"[{method}] unknown method, skipping")
            continue

        target_ids = list(id_filter) if id_filter else all_ids
        target_ids = sorted(target_ids)
        if not target_ids:
            continue

        print(f"[{method}] {len(target_ids)} problems...")
        stats = {"total": len(target_ids), "correct": 0, "failed": 0,
                 "early_stop": 0, "elapsed": []}

        # Build id → index lookup (collected_id may not equal list index)
        id_to_idx = {item["collected_id"]: i for i, item in enumerate(data)}

        for cid in target_ids:
            if cid not in id_to_idx:
                print(f"  #{cid:>2d}  not found in data, skipping")
                continue
            item = data[id_to_idx[cid]]

            problem = make_problem_dict(item)
            gold = item["answer"]
            subtask = item.get("subtask", "")

            if args.dry_run:
                print(f"  #{cid:>2d}  [dry-run] skipped")
                continue

            try:
                result = RUNNERS[method](problem)

                # correct / gold_answer 추가
                pred = result.get("predicted_answer")
                correct = is_correct(pred, gold, benchmark, subtask)
                result["correct"] = correct
                result["gold_answer"] = gold

                # 기존 method_results에서 해당 method만 교체
                item["method_results"][method] = result

                # 통계
                if correct:
                    stats["correct"] += 1
                stats["elapsed"].append(result.get("elapsed_seconds", 0))

                if method == "self_refine":
                    iters = result.get("iterations", [])
                    if any(it.get("stopped_early") for it in iters):
                        stats["early_stop"] += 1

                print(summarize_result(method, result, item))

            except Exception as e:
                stats["failed"] += 1
                print(f"  #{cid:>2d}  ✗ ERROR: {e}")

        # method 요약
        if not args.dry_run and stats["elapsed"]:
            avg_t = sum(stats["elapsed"]) / len(stats["elapsed"])
            summary = f"  Summary: {stats['correct']}/{stats['total']} correct"
            summary += f", avg {avg_t:.1f}s"
            if stats["failed"]:
                summary += f", {stats['failed']} failed"
            if method == "self_refine":
                summary += f", early_stop {stats['early_stop']}/{stats['total']}"
            print(summary)

        total_stats[method] = stats
        print()

    # 4. 저장
    if not args.dry_run:
        # 타임스탬프 업데이트
        for item in data:
            if "patch_timestamp" not in item:
                pass  # 원본 유지
        with open(COLLECTED_PATH, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {COLLECTED_PATH}")

        # 전체 요약
        print()
        print("=== Final Summary ===")
        for method, stats in total_stats.items():
            if stats["elapsed"]:
                print(f"  {method}: {stats['correct']}/{stats['total']} correct"
                      + (f", {stats['failed']} failed" if stats['failed'] else ""))
    else:
        print("[dry-run] No changes saved.")


if __name__ == "__main__":
    main()
