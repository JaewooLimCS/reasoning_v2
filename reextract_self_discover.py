#!/usr/bin/env python3
"""
Re-extract self_discover predicted_answer from existing collected JSON responses.
No API calls needed — just re-parses the stored response text.
Standalone script — no heavy imports (datasets, openai, etc.).

Usage:
    python reextract_self_discover.py                          # GSM8K only
    python reextract_self_discover.py --benchmark bbh          # BBH only
    python reextract_self_discover.py --benchmark all          # both
    python reextract_self_discover.py --dry-run                # preview without saving
"""

import argparse
import json
import os
import re
import shutil
import string


# ── Extraction functions (copied from collect.py to avoid import chain) ──────

def extract_number_from_json(text: str):
    """Extract numeric answer from JSON-structured response (Self-Discover).

    Strategy: two-pass search.
      Pass 1 — find keys whose name contains 'final' and extract number from subtree.
      Pass 2 — fall back to 'answer'/'result' keys at deepest level.
    Avoids grabbing step numbers or intermediate 'value' fields."""
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

    # Pass 1: keys containing 'final'
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

    # Pass 2: 'answer'/'result' keys deepest first
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

    return _search(obj)


def extract_number(text: str):
    text = text.strip()
    for pat in [
        r"[Tt]he answer is \$?([\d,]+\.?\d*)",
        r"####\s*\$?([\d,]+\.?\d*)",
        r"answer[:\s]+\$?([\d,]+\.?\d*)",
        r"=\s*\$?([\d,]+\.?\d*)\s*$",
        r"\$?([\d,]+\.?\d*)\s*$",
    ]:
        m = re.search(pat, text)
        if m:
            val = m.group(1).replace(",", "")
            try:
                f = float(val)
                return int(f) if f == int(f) else f
            except ValueError:
                pass
    numbers = re.findall(r"\$?([\d,]+\.?\d*)", text)
    if numbers:
        val = numbers[-1].replace(",", "")
        try:
            f = float(val)
            return int(f) if f == int(f) else f
        except ValueError:
            pass
    return None


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


# ── Correctness check (standalone) ──────────────────────────────────────────

from collections import Counter

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

def is_correct(predicted, gold: str, benchmark: str, subtask: str = "") -> bool:
    predicted = str(predicted).strip() if predicted is not None else ""
    gold = str(gold).strip()
    if benchmark == "gsm8k":
        try:
            return int(float(predicted.replace(",", ""))) == int(float(gold.replace(",", "")))
        except Exception:
            return False
    elif benchmark == "hotpotqa":
        if _normalize_answer(predicted) == _normalize_answer(gold):
            return True
        return _hotpotqa_f1(predicted, gold) >= 0.5
    elif benchmark == "bbh":
        p = re.sub(r'[\(\)]', '', predicted.lower().rstrip(".")).strip()
        g = re.sub(r'[\(\)]', '', gold.lower().rstrip(".")).strip()
        return p == g or p.startswith(g) or g.startswith(p)
    return False


# ── Main ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def reextract(benchmark: str, dry_run: bool = False):
    path = os.path.join(BASE_DIR, "collected", f"{benchmark}_collected.json")
    if not os.path.exists(path):
        print(f"  {path} not found, skipping")
        return

    with open(path) as f:
        data = json.load(f)

    fixed = 0
    already_ok = 0
    still_none = 0

    for item in data:
        qid = item["collected_id"]
        gold = item["answer"]
        subtask = item.get("subtask", "")
        sd = item["method_results"].get("self_discover")
        if not sd:
            continue

        response = sd.get("response", "")
        old_pred = sd.get("predicted_answer")

        if benchmark in ("gsm8k", "bbh"):
            new_pred = extract_number_from_json(response) or extract_number(response)
        else:
            new_pred = extract_answer_generic(response)

        new_correct = is_correct(new_pred, gold, benchmark, subtask)
        old_correct = sd.get("correct", False)

        if str(old_pred) == str(new_pred):
            already_ok += 1
            continue

        if new_pred is not None:
            fixed += 1
            print(f"  Q{qid:>2}: pred {old_pred!r} → {new_pred!r}  "
                  f"correct {old_correct} → {new_correct}  gold={gold}")
            if not dry_run:
                sd["predicted_answer"] = new_pred
                sd["correct"] = new_correct
                sd["gold_answer"] = gold
        else:
            still_none += 1
            print(f"  Q{qid:>2}: pred {old_pred!r} → still None  gold={gold}")

    print(f"\n  Summary: {fixed} fixed, {already_ok} already ok, {still_none} still None")

    if not dry_run and fixed > 0:
        backup = path.replace(".json", ".pre_reextract.json")
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
            print(f"  Backup: {backup}")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Re-extract self_discover answers")
    parser.add_argument("--benchmark", choices=["gsm8k", "bbh", "all"], default="gsm8k")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    benchmarks = ["gsm8k", "bbh"] if args.benchmark == "all" else [args.benchmark]

    for bm in benchmarks:
        print(f"\n{'='*60}")
        print(f"  Re-extracting self_discover — {bm.upper()}")
        print(f"{'='*60}")
        reextract(bm, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
