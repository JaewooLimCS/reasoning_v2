#!/usr/bin/env python3
"""
quality_check.py — 최종 60문제 품질 검증
==========================================
대상: gsm8k_final_20.json, hotpotqa_final_20.json, bbh_final_20.json

체크 항목:
1. 모든 question × 6 method에 response 존재 (빈값/None)
2. response 길이 통계 (min/max/avg per method)
3. self_discover: reasoning_structure 존재 & 100chars 이상
4. cot_sc: paths 5개, vote_distribution 존재
5. least_to_most: sub_questions, solving_steps 존재
6. self_refine: iterations 배열 존재
7. predicted_answer 이상값 (잔여 텍스트)
8. correct=true 6/6 확인

Usage:
    python quality_check.py
"""

import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

METHODS = ['standard_io', 'zero_shot_cot', 'least_to_most', 'cot_sc', 'self_refine', 'self_discover']

FILES = [
    ("gsm8k", os.path.join(BASE_DIR, "collected", "gsm8k_final_20.json")),
    ("hotpotqa", os.path.join(BASE_DIR, "collected", "hotpotqa_final_20.json")),
    ("bbh", os.path.join(BASE_DIR, "collected", "bbh_final_20.json")),
]

# 잔여 텍스트 패턴 — predicted_answer에 있으면 안 되는 값
JUNK_PATTERNS = [
    re.compile(r'^[\}\{\]\[]$'),                    # lone brackets
    re.compile(r'^["\',\.\s]+$'),                   # only punctuation
    re.compile(r'format|structure|step|module', re.I),  # meta text
    re.compile(r'^\s*$'),                           # whitespace only
    re.compile(r'^(None|null|N/A)$', re.I),         # null-like
]


def get_response_text(mr: dict, method: str) -> str:
    """method별 주요 response 텍스트 반환."""
    if method == 'standard_io':
        return mr.get('response', '') or ''
    elif method == 'zero_shot_cot':
        return mr.get('reasoning', '') or ''
    elif method == 'least_to_most':
        return mr.get('final_response', '') or ''
    elif method == 'cot_sc':
        paths = mr.get('paths', [])
        return ' '.join(p.get('reasoning', '') or '' for p in paths)
    elif method == 'self_refine':
        return mr.get('final_solution', '') or ''
    elif method == 'self_discover':
        return mr.get('response', '') or ''
    return ''


def check_item(item: dict, benchmark: str) -> list:
    """문제 하나에 대한 모든 체크 수행. 실패 목록 반환."""
    failures = []
    qid = item.get('collected_id', item.get('final_id', '?'))

    for method in METHODS:
        mr = item.get('method_results', {}).get(method)
        prefix = f"{method}"

        # 1. response 존재 체크
        if not mr:
            failures.append(f"{prefix}: method_results 자체가 없음")
            continue

        resp = get_response_text(mr, method)
        if not resp or not resp.strip():
            failures.append(f"{prefix}: response 비어있음")

        # 3. self_discover: reasoning_structure
        if method == 'self_discover':
            stage1 = mr.get('stage1', {})
            struct = str(stage1.get('reasoning_structure', '') or '').strip()
            if not struct:
                failures.append(f"{prefix}: reasoning_structure 없음 (0 chars)")
            elif len(struct) < 100:
                failures.append(f"{prefix}: reasoning_structure 너무 짧음 ({len(struct)} chars)")

        # 4. cot_sc: paths 5개, vote_distribution
        if method == 'cot_sc':
            paths = mr.get('paths', [])
            if len(paths) != 5:
                failures.append(f"{prefix}: paths={len(paths)}개 (expected 5)")
            vote = mr.get('vote_distribution')
            if not vote:
                failures.append(f"{prefix}: vote_distribution 없음")

        # 5. least_to_most: sub_questions, solving_steps
        if method == 'least_to_most':
            subs = mr.get('sub_questions', [])
            steps = mr.get('solving_steps', [])
            if not subs:
                failures.append(f"{prefix}: sub_questions 없음")
            if not steps:
                failures.append(f"{prefix}: solving_steps 없음")

        # 6. self_refine: iterations
        if method == 'self_refine':
            iters = mr.get('iterations')
            if iters is None:
                failures.append(f"{prefix}: iterations 없음")

        # 7. predicted_answer 이상값
        pred = mr.get('predicted_answer')
        if pred is None:
            failures.append(f"{prefix}: predicted_answer=None")
        else:
            pred_str = str(pred).strip()
            for pat in JUNK_PATTERNS:
                if pat.search(pred_str):
                    failures.append(f"{prefix}: predicted_answer 이상값 {pred_str!r}")
                    break

        # 8. correct=true 체크
        if not mr.get('correct', False):
            failures.append(f"{prefix}: correct=False")

    return failures


def compute_response_stats(all_data: list) -> dict:
    """method별 response 길이 통계."""
    stats = {m: [] for m in METHODS}
    for benchmark, data in all_data:
        for item in data:
            for method in METHODS:
                mr = item.get('method_results', {}).get(method, {})
                resp = get_response_text(mr, method)
                stats[method].append(len(resp))
    return stats


def main():
    all_data = []
    total_items = 0

    for benchmark, path in FILES:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found")
            sys.exit(1)
        with open(path) as f:
            data = json.load(f)
        all_data.append((benchmark, data))
        total_items += len(data)

    print(f"{'='*80}")
    print(f"  Quality Check — {total_items}문제 × 6 methods = {total_items * 6} judgments")
    print(f"{'='*80}")

    # ── 2. Response 길이 통계 ──
    stats = compute_response_stats(all_data)
    print(f"\n{'─'*60}")
    print(f"  Response 길이 통계 (chars)")
    print(f"{'─'*60}")
    header = f"  {'Method':<20} {'Min':>6} {'Max':>6} {'Avg':>8} {'Zero':>6}"
    print(header)
    for m in METHODS:
        lengths = stats[m]
        zeros = sum(1 for l in lengths if l == 0)
        if lengths:
            print(f"  {m:<20} {min(lengths):>6} {max(lengths):>6} "
                  f"{sum(lengths)//len(lengths):>8} {zeros:>6}")

    # ── 문제별 체크 ──
    total_pass = 0
    total_fail = 0
    all_failures = []

    for benchmark, data in all_data:
        bm_pass = 0
        bm_fail = 0
        bm_failures = []

        for item in data:
            qid = item.get('collected_id', '?')
            failures = check_item(item, benchmark)

            if failures:
                bm_fail += 1
                bm_failures.append((qid, failures))
            else:
                bm_pass += 1

        total_pass += bm_pass
        total_fail += bm_fail

        print(f"\n{'='*80}")
        print(f"  [{benchmark.upper()}] PASS={bm_pass} / FAIL={bm_fail} / Total={len(data)}")
        print(f"{'='*80}")

        if bm_failures:
            for qid, failures in bm_failures:
                # correct=False만 있는 건 별도 표시
                non_correct = [f for f in failures if 'correct=False' not in f]
                correct_fails = [f for f in failures if 'correct=False' in f]

                if non_correct:
                    print(f"\n  Q{qid} — FAIL ({len(failures)} issues)")
                    for f in non_correct:
                        print(f"    ✗ {f}")
                    if correct_fails:
                        print(f"    + correct=False: {len(correct_fails)} methods")
                elif correct_fails:
                    # 6/6 아닌 경우
                    print(f"\n  Q{qid} — NOT 6/6 ({len(correct_fails)} methods incorrect)")
        else:
            print("  모든 문제 PASS ✓")

        all_failures.extend([(benchmark, qid, f) for qid, fs in bm_failures for f in fs])

    # ── 최종 요약 ──
    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"  Total: {total_pass} PASS / {total_fail} FAIL / {total_items} Total")

    # 이슈 유형별 집계
    from collections import Counter
    issue_types = Counter()
    for bm, qid, f in all_failures:
        # 카테고리 추출
        if 'correct=False' in f:
            issue_types['correct=False'] += 1
        elif 'reasoning_structure' in f:
            issue_types['reasoning_structure 문제'] += 1
        elif 'predicted_answer' in f:
            issue_types['predicted_answer 이상'] += 1
        elif 'response 비어있음' in f:
            issue_types['빈 response'] += 1
        elif 'paths' in f or 'vote' in f:
            issue_types['cot_sc 구조 문제'] += 1
        elif 'sub_questions' in f or 'solving_steps' in f:
            issue_types['least_to_most 구조 문제'] += 1
        elif 'iterations' in f:
            issue_types['self_refine 구조 문제'] += 1
        elif 'method_results' in f:
            issue_types['method 누락'] += 1
        else:
            issue_types['기타'] += 1

    if issue_types:
        print(f"\n  이슈 유형별:")
        for issue, cnt in issue_types.most_common():
            print(f"    {issue:30s} {cnt}건")

    # 6/6 아닌 문제 목록
    not_66 = []
    for benchmark, data in all_data:
        for item in data:
            qid = item.get('collected_id', '?')
            n = sum(1 for m in METHODS
                    if item.get('method_results', {}).get(m, {}).get('correct', False))
            if n < 6:
                not_66.append((benchmark, qid, n))

    if not_66:
        print(f"\n  ⚠️  6/6 아닌 문제:")
        for bm, qid, n in not_66:
            print(f"    {bm} Q{qid}: {n}/6")
    else:
        print(f"\n  ✅ 모든 60문제 6/6 correct")


if __name__ == "__main__":
    main()
