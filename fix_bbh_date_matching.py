#!/usr/bin/env python3
"""
Fix BBH date_understanding correctness by matching predicted dates to option labels.

Problem: model predicts "06/08/1972" but gold is "(E)" which maps to 06/08/1972.
The is_correct check fails because it compares "(E)" vs "06/08/1972" literally.

This script:
1. Parses option labels from question text: (A)=06/19/2017, (B)=07/17/2017, etc.
2. If predicted_answer is a date that matches the gold option's date → correct=True
3. Updates both bbh_collected.json and bbh_verified.json (if exists)

Usage:
    python fix_bbh_date_matching.py              # apply
    python fix_bbh_date_matching.py --dry-run    # preview
"""

import argparse
import json
import os
import re
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTED_PATH = os.path.join(BASE_DIR, "collected", "bbh_collected.json")
VERIFIED_PATH = os.path.join(BASE_DIR, "collected", "bbh_verified.json")

METHODS = ['standard_io', 'zero_shot_cot', 'least_to_most', 'cot_sc', 'self_refine', 'self_discover']


def parse_options(question: str) -> dict:
    """Parse (A) 06/19/2017 style options from question text.
    Returns {label: date_str}, e.g. {'(A)': '06/19/2017', ...}"""
    options = {}
    for m in re.finditer(r'\(([A-F])\)\s+(\d{2}/\d{2}/\d{4})', question):
        options[f"({m.group(1)})"] = m.group(2)
    return options


def pred_matches_gold_date(pred: str, gold_label: str, options: dict) -> bool:
    """Check if prediction contains the date corresponding to the gold option label."""
    gold_date = options.get(gold_label)
    if not gold_date or not pred:
        return False

    pred = str(pred).strip()

    # Direct option label match
    gold_letter = gold_label.strip("()")
    if re.search(rf'\(?{gold_letter}\)?', pred):
        return True

    # Date match: pred contains gold date
    if gold_date in pred:
        return True

    return False


def fix_file(path: str, dry_run: bool) -> int:
    if not os.path.exists(path):
        print(f"  {path} not found, skipping")
        return 0

    with open(path) as f:
        data = json.load(f)

    fixes = 0

    for item in data:
        if item.get("subtask") != "date_understanding":
            continue

        qid = item["collected_id"]
        gold = item["answer"]
        question = item["question"]
        options = parse_options(question)

        if not options:
            continue

        gold_date = options.get(gold, "?")

        for method in METHODS:
            mr = item["method_results"].get(method)
            if not mr:
                continue

            pred = str(mr.get("predicted_answer", "") or "")
            old_correct = mr.get("correct", False)

            if not old_correct and pred_matches_gold_date(pred, gold, options):
                fixes += 1
                print(f"  Q{qid:>2} {method:20s}: pred={pred!r:30s} matches gold {gold}={gold_date} → correct=True")
                if not dry_run:
                    mr["correct"] = True

    # Update verified flag
    if not dry_run:
        for item in data:
            if item.get("subtask") != "date_understanding":
                continue
            any_correct = any(
                item["method_results"].get(m, {}).get("correct", False)
                for m in METHODS
            )
            item["verified"] = any_correct

    if not dry_run and fixes > 0:
        backup = path.replace(".json", ".pre_datefix.json")
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
            print(f"  Backup: {backup}")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {path}")

    return fixes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total = 0
    for label, path in [("bbh_collected.json", COLLECTED_PATH),
                         ("bbh_verified.json", VERIFIED_PATH)]:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        n = fix_file(path, args.dry_run)
        total += n
        print(f"  Fixes: {n}")

    print(f"\nTotal fixes: {total}")
    if args.dry_run:
        print("[dry-run] No changes saved.")


if __name__ == "__main__":
    main()
