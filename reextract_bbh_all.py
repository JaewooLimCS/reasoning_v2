#!/usr/bin/env python3
"""
Re-extract ALL method predicted_answers for BBH from existing responses.
No API calls — re-parses stored response text with updated extraction logic.

Key fixes:
- BBH option labels (A)~(F) extracted before date/text patterns
- self_discover JSON parsing with final_answer key search

Usage:
    python reextract_bbh_all.py               # apply changes
    python reextract_bbh_all.py --dry-run     # preview only
"""

import argparse
import json
import os
import re
import shutil
import string
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── BBH option extraction ────────────────────────────────────────────────────

def extract_bbh_option(text: str):
    """Extract BBH multiple-choice option label (A)~(F) from text."""
    m = re.search(r'[Tt]he answer is[:\s]*\(?([A-F])\)?', text)
    if m:
        return f"({m.group(1)})"
    m = re.search(r'[Aa]nswer[:\s]*\(?([A-F])\)?', text)
    if m:
        return f"({m.group(1)})"
    m = re.search(r'(?:^|\n)\s*\(?([A-F])\)?[\s\.\)]', text)
    if m:
        return f"({m.group(1)})"
    m = re.search(r'\(([A-F])\)', text)
    if m:
        return f"({m.group(1)})"
    return None


# ── Generic extraction with BBH option priority ──────────────────────────────

def extract_answer_generic(text: str, bbh_options: bool = False) -> str:
    text = text.strip()
    if bbh_options:
        opt = extract_bbh_option(text)
        if opt:
            return opt
    for pat in [
        r"[Tt]he answer is[:\s]+(.+?)(?:\n|$)",
        r"[Ff]inal [Aa]nswer[:\s]+(.+?)(?:\n|$)",
        r"[Aa]nswer[:\s]+(.+?)(?:\n|$)",
    ]:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            return m.group(1).strip().rstrip(".")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[-1] if lines else text[:100]


# ── Number extraction ────────────────────────────────────────────────────────

def extract_number(text: str):
    text = text.strip()
    for pat in [
        r"[Tt]he answer is \$?(-?[\d,]+\.?\d*)",
        r"####\s*\$?(-?[\d,]+\.?\d*)",
        r"answer[:\s]+\$?(-?[\d,]+\.?\d*)",
        r"=\s*\$?(-?[\d,]+\.?\d*)\s*$",
        r"\$?(-?[\d,]+\.?\d*)\s*$",
    ]:
        m = re.search(pat, text)
        if m:
            val = m.group(1).replace(",", "")
            try:
                f = float(val)
                return int(f) if f == int(f) else f
            except ValueError:
                pass
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        val = numbers[-1].replace(",", "").lstrip("$")
        try:
            f = float(val)
            return int(f) if f == int(f) else f
        except ValueError:
            pass
    return None


# ── JSON extraction for self_discover ─────────────────────────────────────────

def extract_number_from_json(text: str):
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        m = re.search(r'\{[\s\S]+\}', text)
        if not m:
            return None
        try:
            obj = json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            return None

    def _to_number(node):
        if isinstance(node, bool):
            return None
        if isinstance(node, (int, float)):
            return int(node) if float(node) == int(float(node)) else node
        if isinstance(node, str):
            cleaned = node.strip().replace(',', '').lstrip('$')
            try:
                f = float(cleaned)
                return int(f) if f == int(f) else f
            except ValueError:
                return None
        return None

    def _extract_from_subtree(node, depth=0):
        if depth > 5:
            return None
        n = _to_number(node)
        if n is not None:
            return n
        if isinstance(node, dict):
            if 'value' in node:
                n = _to_number(node['value'])
                if n is not None:
                    return n
            for v in node.values():
                n = _to_number(v)
                if n is not None:
                    return n
            for v in node.values():
                if isinstance(v, dict):
                    n = _extract_from_subtree(v, depth + 1)
                    if n is not None:
                        return n
        return None

    final_pat = re.compile(r'final', re.IGNORECASE)
    candidates = []

    def _collect_final(node, depth=0, path=""):
        if depth > 15:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                p = f"{path}.{k}"
                if final_pat.search(k):
                    candidates.append((depth, p, v))
                _collect_final(v, depth + 1, p)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _collect_final(v, depth + 1, f"{path}[{i}]")

    _collect_final(obj)
    candidates.sort(key=lambda x: x[0], reverse=True)
    for _, path, node in candidates:
        n = _extract_from_subtree(node)
        if n is not None:
            return n

    fallback_pat = re.compile(r'^(answer|result)$', re.IGNORECASE)
    fallbacks = []

    def _collect_fb(node, depth=0):
        if depth > 15:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                if fallback_pat.match(k):
                    fallbacks.append((depth, v))
                _collect_fb(v, depth + 1)
        elif isinstance(node, list):
            for v in node:
                _collect_fb(v, depth + 1)

    _collect_fb(obj)
    fallbacks.sort(key=lambda x: x[0], reverse=True)
    for _, node in fallbacks:
        n = _extract_from_subtree(node)
        if n is not None:
            return n

    return None


# ── JSON option extraction for self_discover ─────────────────────────────────

def extract_option_from_json(text: str):
    """Extract option label from JSON self_discover response.
    Searches for keys like final_answer, final_choice, option, multiple_choice
    and converts single-letter values to (X) format."""
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        m = re.search(r'\{[\s\S]+\}', text)
        if not m:
            return None
        try:
            obj = json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            return None

    option_keys = re.compile(
        r'(final_answer|final_choice|answer|option|multiple_choice|choice)',
        re.IGNORECASE
    )
    candidates = []  # (depth, key, value)

    def _collect(node, depth=0):
        if depth > 15:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                if option_keys.search(k):
                    candidates.append((depth, k, v))
                _collect(v, depth + 1)
        elif isinstance(node, list):
            for item in node:
                _collect(item, depth + 1)

    _collect(obj)
    # Deepest first, prefer keys with 'final'
    candidates.sort(key=lambda x: (x[0], 'final' in x[1].lower()), reverse=True)

    for _, key, val in candidates:
        s = str(val).strip().strip('"').strip("'").strip("()")
        # Single letter A-F
        if re.match(r'^[A-F]$', s, re.IGNORECASE):
            return f"({s.upper()})"
        # "(X)" format already
        m = re.match(r'^\(?([A-F])\)?', s, re.IGNORECASE)
        if m:
            return f"({m.group(1).upper()})"
        # "X) description" format
        m = re.match(r'^([A-F])\)', s, re.IGNORECASE)
        if m:
            return f"({m.group(1).upper()})"
        # Contains option in longer text like "The answer is (C)"
        m = re.search(r'\(([A-F])\)', s)
        if m:
            return f"({m.group(1)})"

    return None


# ── Correctness check ────────────────────────────────────────────────────────

def is_correct_bbh(predicted, gold: str, subtask: str = "") -> bool:
    predicted = str(predicted).strip() if predicted is not None else ""
    gold = str(gold).strip()
    p = re.sub(r'[\(\)]', '', predicted.lower().rstrip(".")).strip()
    g = re.sub(r'[\(\)]', '', gold.lower().rstrip(".")).strip()
    return p == g or p.startswith(g) or g.startswith(p)


# ── Per-method extraction ────────────────────────────────────────────────────

def reextract_method(method: str, mr: dict, subtask: str) -> tuple:
    """Re-extract predicted_answer for a method. Returns (new_pred, source_text)."""
    is_arith = subtask == "multistep_arithmetic_two"
    needs_option = not is_arith  # all BBH except arithmetic have (A)~(F) options

    if method == "standard_io":
        resp = mr.get("response", "")
        if is_arith:
            return extract_number(resp), resp
        return extract_answer_generic(resp, bbh_options=needs_option), resp

    elif method == "zero_shot_cot":
        ans = mr.get("answer_text", "")
        reasoning = mr.get("reasoning", "")
        if is_arith:
            return extract_number(ans) or extract_number(reasoning), ans
        return extract_answer_generic(ans, bbh_options=needs_option), ans

    elif method == "least_to_most":
        fr = mr.get("final_response", "")
        if is_arith:
            return extract_number(fr), fr
        return extract_answer_generic(fr, bbh_options=needs_option), fr

    elif method == "cot_sc":
        # Re-extract each path, then majority vote
        paths = mr.get("paths", [])
        answers = []
        for p in paths:
            raw = p.get("reasoning", "")
            if is_arith:
                ans = extract_number(raw)
            else:
                ans = extract_answer_generic(raw, bbh_options=needs_option)
            if ans is not None:
                answers.append(str(ans))
        if answers:
            counter = Counter(answers)
            return counter.most_common(1)[0][0], "paths"
        return None, "paths"

    elif method == "self_refine":
        sol = mr.get("final_solution", "")
        if is_arith:
            return extract_number(sol), sol
        return extract_answer_generic(sol, bbh_options=needs_option), sol

    elif method == "self_discover":
        resp = mr.get("response", "")
        if is_arith:
            return extract_number_from_json(resp) or extract_number(resp), resp
        # For option-based tasks, parse JSON for answer keys first
        if needs_option:
            opt = extract_option_from_json(resp)
            if opt:
                return opt, resp
            # Fallback to regex on raw text
            opt = extract_bbh_option(resp)
            if opt:
                return opt, resp
        return extract_answer_generic(resp, bbh_options=needs_option), resp

    return None, ""


# ── Main ─────────────────────────────────────────────────────────────────────

METHODS = ['standard_io', 'zero_shot_cot', 'least_to_most', 'cot_sc', 'self_refine', 'self_discover']


def main():
    parser = argparse.ArgumentParser(description="Re-extract all BBH answers")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = os.path.join(BASE_DIR, "collected", "bbh_collected.json")
    with open(path) as f:
        data = json.load(f)

    total_changes = 0
    method_changes = Counter()
    correctness_changes = {"gained": 0, "lost": 0}

    for item in data:
        qid = item["collected_id"]
        gold = item["answer"]
        subtask = item.get("subtask", "")

        for method in METHODS:
            mr = item["method_results"].get(method)
            if not mr:
                continue

            old_pred = mr.get("predicted_answer")
            old_correct = mr.get("correct", False)

            # Skip if already correct with pred=None (set by patch_methods.py)
            if old_correct and old_pred is None:
                continue

            new_pred, _ = reextract_method(method, mr, subtask)
            new_correct = is_correct_bbh(new_pred, gold, subtask) if new_pred is not None else False

            if str(old_pred) != str(new_pred):
                total_changes += 1
                method_changes[method] += 1

                if new_correct and not old_correct:
                    correctness_changes["gained"] += 1
                    marker = "✅"
                elif old_correct and not new_correct:
                    correctness_changes["lost"] += 1
                    marker = "❌ REGRESSION"
                else:
                    marker = "—"

                print(f"  Q{qid:>2} [{subtask:30s}] {method:20s}: "
                      f"{old_pred!r} → {new_pred!r}  "
                      f"correct {old_correct}→{new_correct} {marker}")

                if not args.dry_run:
                    mr["predicted_answer"] = new_pred
                    mr["correct"] = new_correct
                    mr["gold_answer"] = gold

    print(f"\n{'='*60}")
    print(f"Summary: {total_changes} changes")
    for m in METHODS:
        if method_changes[m]:
            print(f"  {m:20s}: {method_changes[m]} changes")
    print(f"  Correctness gained: {correctness_changes['gained']}")
    print(f"  Correctness lost:   {correctness_changes['lost']}")

    if not args.dry_run and total_changes > 0:
        backup = path.replace(".json", ".pre_reextract_all.json")
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
            print(f"\n  Backup: {backup}")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {path}")
    elif args.dry_run:
        print("\n  [dry-run] No changes saved.")


if __name__ == "__main__":
    main()
