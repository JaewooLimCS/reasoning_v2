#!/usr/bin/env python3
"""
Manual verification script for BBH collected results.

Judgment criteria:
- PASS: gold answer is contained in prediction (case-insensitive)
- FAIL: prediction is empty or gold is completely absent
- REVIEW: ambiguous cases (partial match, synonyms, etc.)

BBH answers are typically exact (numbers, dates, option labels),
so most cases should auto-resolve as PASS or FAIL.

Interactive mode for REVIEW cases, then saves bbh_verified.json.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

INPUT_PATH = Path("collected/bbh_collected.json")
OUTPUT_PATH = Path("collected/bbh_verified.json")

METHODS = [
    "standard_io", "zero_shot_cot", "least_to_most",
    "cot_sc", "self_refine", "self_discover",
]


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/whitespace for comparison."""
    text = text.lower().strip()
    # Remove trailing periods and common punctuation
    text = re.sub(r'[.!?,;:]+$', '', text).strip()
    return text


def auto_judge(gold: str, pred: str) -> str:
    """
    Returns 'PASS', 'FAIL', or 'REVIEW'.
    """
    if not pred or not pred.strip():
        return "FAIL"

    gold_norm = normalize(gold)
    pred_norm = normalize(pred)

    # Exact containment (case-insensitive)
    if gold_norm in pred_norm:
        return "PASS"

    # BBH-specific: strip parentheses for option matching e.g. "(A)" vs "A"
    gold_clean = re.sub(r'[\(\)]', '', gold_norm).strip()
    pred_clean = re.sub(r'[\(\)]', '', pred_norm).strip()
    if gold_clean and gold_clean in pred_clean:
        return "PASS"

    # Check if all significant words of gold appear in pred
    gold_words = set(re.findall(r'\w+', gold_norm))
    pred_words = set(re.findall(r'\w+', pred_norm))

    # Remove very short stopwords for word-level check
    stopwords = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'is', 'and', 'or', 'for'}
    gold_content = gold_words - stopwords
    pred_content = pred_words - stopwords

    if not gold_content:
        return "REVIEW"

    # All content words present → likely correct but not substring match → REVIEW
    if gold_content.issubset(pred_content):
        return "REVIEW"

    # No overlap at all → FAIL
    if gold_content.isdisjoint(pred_content):
        return "FAIL"

    # Partial overlap → REVIEW
    return "REVIEW"


def run_auto_judge(data):
    """Run auto-judge on all questions × methods. Returns judgments dict and review list."""
    judgments = {}
    reviews = []

    for item in data:
        qid = item["collected_id"]
        judgments[qid] = {}
        gold = item["answer"]

        for method in METHODS:
            mr = item["method_results"].get(method)
            if not mr:
                judgments[qid][method] = "FAIL"
                continue

            pred = str(mr.get("predicted_answer", "") or "")
            verdict = auto_judge(gold, pred)
            judgments[qid][method] = verdict

            if verdict == "REVIEW":
                reviews.append({
                    "qid": qid,
                    "method": method,
                    "question": item["question"][:100],
                    "subtask": item.get("subtask", ""),
                    "gold": gold,
                    "pred": pred,
                })

    return judgments, reviews


def print_summary(judgments, data):
    """Print summary table of auto-judge results."""
    print("\n" + "=" * 80)
    print("AUTO-JUDGE SUMMARY")
    print("=" * 80)

    # Per-method counts
    method_counts = {m: {"PASS": 0, "FAIL": 0, "REVIEW": 0} for m in METHODS}
    for qid, mj in judgments.items():
        for method, verdict in mj.items():
            method_counts[method][verdict] += 1

    header = f"{'Method':<16} {'PASS':>6} {'FAIL':>6} {'REVIEW':>8} {'Auto-Acc':>10}"
    print(header)
    print("-" * len(header))
    for m in METHODS:
        c = method_counts[m]
        total = c["PASS"] + c["FAIL"] + c["REVIEW"]
        auto_acc = c["PASS"] / total * 100 if total else 0
        print(f"{m:<16} {c['PASS']:>6} {c['FAIL']:>6} {c['REVIEW']:>8} {auto_acc:>9.1f}%")

    total_review = sum(c["REVIEW"] for c in method_counts.values())
    print(f"\nTotal REVIEW cases requiring manual check: {total_review}")

    # Per-subtask summary
    subtask_counts = {}
    for item in data:
        st = item.get("subtask", "unknown")
        if st not in subtask_counts:
            subtask_counts[st] = 0
        subtask_counts[st] += 1
    print(f"\nSubtask distribution:")
    for st, cnt in sorted(subtask_counts.items()):
        print(f"  {st}: {cnt}")

    return total_review


def interactive_review(reviews, judgments):
    """Show REVIEW cases one by one, ask y/n."""
    if not reviews:
        print("\nNo REVIEW cases — all judgments are auto-determined.")
        return

    print(f"\n{'=' * 80}")
    print(f"INTERACTIVE REVIEW — {len(reviews)} cases")
    print(f"For each case: [y] correct, [n] incorrect, [s] skip (keep as REVIEW)")
    print(f"{'=' * 80}\n")

    for i, r in enumerate(reviews, 1):
        print(f"--- [{i}/{len(reviews)}] Q{r['qid']} / {r['method']} [{r['subtask']}] ---")
        print(f"  Question: {r['question']}...")
        print(f"  Gold:     {r['gold']}")
        print(f"  Pred:     {r['pred']}")

        while True:
            try:
                choice = input("  Correct? [y/n/s]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted — keeping remaining REVIEW cases as FAIL.")
                for remaining in reviews[i - 1:]:
                    judgments[remaining["qid"]][remaining["method"]] = "FAIL"
                return
            if choice in ('y', 'n', 's'):
                break
            print("  Please enter y, n, or s.")

        if choice == 'y':
            judgments[r["qid"]][r["method"]] = "PASS"
        elif choice == 'n':
            judgments[r["qid"]][r["method"]] = "FAIL"
        print()


def build_verified(data, judgments):
    """Build verified JSON with correct fields added."""
    verified = json.loads(json.dumps(data))  # deep copy

    for item in verified:
        qid = item["collected_id"]
        any_correct = False

        for method in METHODS:
            mr = item["method_results"].get(method)
            if mr:
                verdict = judgments[qid].get(method, "FAIL")
                is_correct = (verdict == "PASS")
                mr["correct"] = is_correct
                if is_correct:
                    any_correct = True

        item["verified"] = any_correct

    return verified


def print_final_summary(verified):
    """Print final accuracy per method after verification."""
    print("\n" + "=" * 80)
    print("FINAL VERIFIED RESULTS")
    print("=" * 80)

    method_correct = {m: 0 for m in METHODS}
    total = len(verified)

    for item in verified:
        for method in METHODS:
            mr = item["method_results"].get(method)
            if mr and mr.get("correct"):
                method_correct[method] += 1

    header = f"{'Method':<16} {'Correct':>8} {'Total':>6} {'Accuracy':>10}"
    print(header)
    print("-" * len(header))
    for m in METHODS:
        acc = method_correct[m] / total * 100 if total else 0
        print(f"{m:<16} {method_correct[m]:>8} {total:>6} {acc:>9.1f}%")

    q_correct = sum(1 for item in verified if item["verified"])
    print(f"\nQuestions with at least 1 correct method: {q_correct}/{total}")

    # 6/6 distribution
    dist = Counter()
    for item in verified:
        n = sum(1 for m in METHODS if item["method_results"].get(m, {}).get("correct", False))
        dist[n] += 1
    print(f"\n{'=' * 80}")
    print("CORRECT METHOD DISTRIBUTION")
    print("=" * 80)
    for k in range(6, -1, -1):
        print(f"  {k}/6 correct: {dist[k]}개")


def load_existing_judgments() -> dict:
    """Load already-verified judgments from bbh_verified.json.
    Returns {collected_id: {method: bool}} for previously verified items."""
    if not OUTPUT_PATH.exists():
        return {}
    with open(OUTPUT_PATH) as f:
        verified = json.load(f)
    existing = {}
    for item in verified:
        qid = item["collected_id"]
        existing[qid] = {}
        for method in METHODS:
            mr = item["method_results"].get(method)
            if mr and "correct" in mr:
                existing[qid][method] = mr["correct"]
    return existing


def main():
    # Load data
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found")
        sys.exit(1)

    with open(INPUT_PATH) as f:
        data = json.load(f)

    # Load existing verified judgments (preserve previous manual reviews)
    existing = load_existing_judgments()
    existing_ids = set(existing.keys())
    new_ids = {item["collected_id"] for item in data} - existing_ids

    print(f"Loaded {len(data)} questions × {len(METHODS)} methods = {len(data) * len(METHODS)} judgments")
    print(f"  Previously verified: {len(existing_ids)} questions (preserved)")
    print(f"  New to judge: {len(new_ids)} questions")

    # Step 1: Build judgments — reuse existing, auto-judge only new
    judgments = {}
    reviews = []

    for item in data:
        qid = item["collected_id"]
        gold = item["answer"]

        if qid in existing:
            judgments[qid] = {}
            for method in METHODS:
                if method in existing[qid]:
                    judgments[qid][method] = "PASS" if existing[qid][method] else "FAIL"
                else:
                    judgments[qid][method] = "FAIL"
        else:
            judgments[qid] = {}
            for method in METHODS:
                mr = item["method_results"].get(method)
                if not mr:
                    judgments[qid][method] = "FAIL"
                    continue
                pred = str(mr.get("predicted_answer", "") or "")
                verdict = auto_judge(gold, pred)
                judgments[qid][method] = verdict
                if verdict == "REVIEW":
                    reviews.append({
                        "qid": qid,
                        "method": method,
                        "question": item["question"][:100],
                        "subtask": item.get("subtask", ""),
                        "gold": gold,
                        "pred": pred,
                    })

    # Step 2: Print summary (new questions only)
    if new_ids:
        print(f"\n{'=' * 80}")
        print(f"AUTO-JUDGE SUMMARY (new questions only: Q{min(new_ids)}–Q{max(new_ids)})")
        print(f"{'=' * 80}")
        new_counts = {m: {"PASS": 0, "FAIL": 0, "REVIEW": 0} for m in METHODS}
        for qid in new_ids:
            for method, verdict in judgments.get(qid, {}).items():
                new_counts[method][verdict] += 1
        header = f"{'Method':<16} {'PASS':>6} {'FAIL':>6} {'REVIEW':>8}"
        print(header)
        print("-" * len(header))
        for m in METHODS:
            c = new_counts[m]
            print(f"{m:<16} {c['PASS']:>6} {c['FAIL']:>6} {c['REVIEW']:>8}")
        total_review = sum(c["REVIEW"] for c in new_counts.values())
        print(f"\nNew REVIEW cases requiring manual check: {total_review}")
    else:
        total_review = 0

    # Step 3: Interactive review (new REVIEW cases only)
    if total_review > 0:
        interactive_review(reviews, judgments)

    # Convert any remaining REVIEW to FAIL for final scoring
    for qid, mj in judgments.items():
        for method in list(mj.keys()):
            if mj[method] == "REVIEW":
                mj[method] = "FAIL"

    # Step 4: Build and save verified JSON
    verified = build_verified(data, judgments)
    print_final_summary(verified)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(verified, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
