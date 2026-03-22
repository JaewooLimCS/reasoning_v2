#!/usr/bin/env python3
"""
Re-extract self_discover predicted_answer for HotPotQA from existing responses.
No API calls — re-parses stored response text with text-aware extraction.

HotPotQA answers are text (names, dates, yes/no), so extraction strategy:
1. JSON key search: "final_answer", "answer", "conclusion" → extract string value
2. Fallback: extract_answer_generic on raw response text

Usage:
    python reextract_hotpotqa_self_discover.py              # apply
    python reextract_hotpotqa_self_discover.py --dry-run    # preview
"""

import argparse
import json
import os
import re
import shutil
import string
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Text extraction from JSON ────────────────────────────────────────────────

def extract_text_from_json(text: str):
    """Extract text answer from JSON-structured Self-Discover response.

    Strategy:
      Pass 1 — find keys containing 'final' or 'conclusion', extract string value.
      Pass 2 — fall back to 'answer' keys at deepest level.
    Returns a string or None."""
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

    def _to_text(node):
        """Try to get a meaningful text string from a node."""
        if isinstance(node, bool):
            return "yes" if node else "no"
        if isinstance(node, str):
            s = node.strip().strip('"').strip("'").rstrip(".")
            if s and len(s) < 500:  # skip very long strings (reasoning, not answers)
                return s
            return None
        if isinstance(node, (int, float)):
            return str(node)
        return None

    def _extract_from_subtree(node, depth=0):
        """Extract answer text from a subtree, preferring 'value' key."""
        if depth > 5:
            return None
        t = _to_text(node)
        if t is not None:
            return t
        if isinstance(node, dict):
            # Prefer 'value' or 'text' key
            for key in ('value', 'text', 'answer'):
                if key in node:
                    t = _to_text(node[key])
                    if t is not None:
                        return t
            # Try 'statement', 'one_line_statement', etc.
            for key in ('statement', 'one_line_statement', 'answer_sentence'):
                if key in node:
                    t = _to_text(node[key])
                    if t is not None:
                        # Extract just the key part from a sentence
                        return t
            # Last resort: any short string value
            for v in node.values():
                t = _to_text(v)
                if t is not None and len(t) < 200:
                    return t
        return None

    # Pass 1: keys containing 'final' or 'conclusion'
    final_pat = re.compile(r'(final|conclusion)', re.IGNORECASE)
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
    candidates.sort(key=lambda x: x[0], reverse=True)  # deepest first
    for _, path, node in candidates:
        t = _extract_from_subtree(node)
        if t is not None:
            return t

    # Pass 2: 'answer' keys at any level, deepest first
    fallback_pat = re.compile(r'^answer$', re.IGNORECASE)
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
        t = _extract_from_subtree(node)
        if t is not None:
            return t

    return None


# ── Generic text extraction (fallback) ───────────────────────────────────────

def extract_answer_generic(text: str) -> str:
    text = text.strip()
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


# ── Correctness check ────────────────────────────────────────────────────────

def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.split())


def _hotpotqa_f1(pred: str, gold: str) -> float:
    p_tokens = _normalize_answer(pred).split()
    g_tokens = _normalize_answer(gold).split()
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return (2 * precision * recall) / (precision + recall)


def is_correct(predicted, gold: str) -> bool:
    predicted = str(predicted).strip() if predicted is not None else ""
    gold = str(gold).strip()
    if _normalize_answer(predicted) == _normalize_answer(gold):
        return True
    return _hotpotqa_f1(predicted, gold) >= 0.5


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Re-extract HotPotQA self_discover answers")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = os.path.join(BASE_DIR, "collected", "hotpotqa_collected.json")
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} questions from hotpotqa_collected.json")

    fixed = 0
    already_ok = 0
    still_bad = 0

    for item in data:
        qid = item["collected_id"]
        gold = item["answer"]
        sd = item["method_results"].get("self_discover")
        if not sd:
            continue

        response = sd.get("response", "")
        old_pred = sd.get("predicted_answer")
        old_correct = sd.get("correct", False)

        # Re-extract: JSON first, then generic fallback
        new_pred = extract_text_from_json(response)
        if new_pred is None:
            new_pred = extract_answer_generic(response)

        new_correct = is_correct(new_pred, gold)

        # Skip if no change
        if str(old_pred) == str(new_pred):
            already_ok += 1
            continue

        if new_correct and not old_correct:
            marker = "✅ GAINED"
        elif old_correct and not new_correct:
            marker = "❌ LOST"
        elif new_correct == old_correct:
            marker = "—"
        else:
            marker = "?"

        fixed += 1
        print(f"  Q{qid:>2}: {old_pred!r} → {new_pred!r}  "
              f"correct {old_correct}→{new_correct}  gold={gold!r}  {marker}")

        if not args.dry_run:
            sd["predicted_answer"] = new_pred
            sd["correct"] = new_correct
            sd["gold_answer"] = gold

    print(f"\nSummary: {fixed} changed, {already_ok} unchanged, {still_bad} still bad")

    if not args.dry_run and fixed > 0:
        backup = path.replace(".json", ".pre_reextract_sd.json")
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
            print(f"Backup: {backup}")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {path}")
    elif args.dry_run:
        print("[dry-run] No changes saved.")


if __name__ == "__main__":
    main()
