"""
analyze.py — 수집 결과 분석
"""
import json
import os
from collections import defaultdict

COLLECTED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collected")
ALL_METHODS   = ["standard_io","zero_shot_cot","least_to_most",
                 "cot_sc","self_refine","self_discover"]


def load_collected(bm):
    path = os.path.join(COLLECTED_DIR, f"{bm}_collected.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []


def load_progress():
    path = os.path.join(COLLECTED_DIR, "progress.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def analyze(benchmark):
    collected = load_collected(benchmark)
    bp        = load_progress().get(benchmark, {})
    n         = len(collected)
    cursor    = bp.get("pool_cursor", 0)

    print(f"\n{'='*60}")
    print(f"  [{benchmark.upper()}]")
    print(f"{'='*60}")
    print(f"  수집: {n}개  |  Pool 소비: {cursor}개  |  "
          f"효율: {n/cursor*100:.1f}%" if cursor else f"  수집: {n}개")

    if not collected:
        print("  (데이터 없음)")
        return

    print(f"\n  Method별 정답률:")
    for method in ALL_METHODS:
        ok = sum(1 for item in collected
                 if item.get("method_results",{}).get(method,{}).get("correct", False))
        bar = "█" * int(ok/n*20)
        print(f"    {method:20s}: {ok}/{n} ({ok/n*100:5.1f}%)  {bar}")

    if benchmark == "bbh":
        counts = defaultdict(int)
        for item in collected:
            counts[item.get("subtask","?")] += 1
        print(f"\n  BBH subtask 분포:")
        for st, cnt in sorted(counts.items()):
            print(f"    {st:42s}: {cnt}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["gsm8k","hotpotqa","bbh","all"], default="all")
    args = p.parse_args()
    benchmarks = ["gsm8k","hotpotqa","bbh"] if args.benchmark == "all" else [args.benchmark]

    total = 0
    for bm in benchmarks:
        analyze(bm)
        total += len(load_collected(bm))

    print(f"\n{'='*60}")
    print(f"  총 수집: {total}문제  |  총 응답: {total*6}개")
    print(f"  목표 달성: {'✅ 완료' if total >= 60 else f'⏳ {total}/60'}")
    print()
