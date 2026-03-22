"""
collect.py — Pool 기반 데이터 수집 메인 실행
==============================================
목표: 각 벤치마크에서 "6 methods 모두 정답" 문제 20개 수집
     총 60문제 × 6 methods = 360개 정답 데이터

Usage:
    python collect.py                    # 전체 실행
    python collect.py --benchmark gsm8k  # 특정 벤치마크만
    python collect.py --dry-run          # API 호출 없이 파이프라인 테스트
    python collect.py --resume           # 이전 진행상황 이어서

Output:
    collected/gsm8k_collected.json
    collected/hotpotqa_collected.json
    collected/bbh_collected.json
    collected/progress.json
    collected/final_dataset.json
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data.loader import load_all_pools
import prompts.standard_io   as p_standard_io
import prompts.zero_shot_cot as p_zscot
import prompts.least_to_most as p_l2m
import prompts.cot_sc        as p_cotsc
import prompts.self_refine   as p_selfrefine
import prompts.self_discover as p_selfdiscover

COLLECTED_DIR = os.path.join(BASE_DIR, "collected")
os.makedirs(COLLECTED_DIR, exist_ok=True)

TARGET_PER_BENCHMARK = 20
TARGET_PER_BENCHMARK_LENIENT = 30  # HotPotQA, BBH (manual verification)
COT_SC_N_PATHS = 5
ALL_METHODS = [
    "standard_io", "zero_shot_cot", "least_to_most",
    "cot_sc", "self_refine", "self_discover"
]

BBH_TASK_DESCRIPTIONS = {
    "multistep_arithmetic_two":       "Evaluate a mathematical expression following order of operations.",
    "disambiguation_qa":              "Identify what the pronoun refers to in the sentence, or state if it is ambiguous. Choose the correct option.",
    "date_understanding":             "Determine the date based on the given information. Answer in MM/DD/YYYY format.",
    "logical_deduction_five_objects": "Given logical constraints about five objects, deduce the correct answer."
}


# ── 정답 추출 및 판별 ─────────────────────────────────────────────────────────

def extract_number_from_json(text: str):
    """Extract numeric answer from JSON-structured response (Self-Discover).

    Strategy: two-pass search.
      Pass 1 — find keys whose name contains 'final' (final_answer, final_numeric_answer,
               conclusion_final_answer, etc.) and extract a number from their subtree.
      Pass 2 — fall back to 'answer', 'result', 'value' at the deepest level.
    This avoids grabbing step numbers or intermediate 'value' fields."""
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
        """Try to coerce a scalar node to a number."""
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
        """Extract a number from a subtree: check 'value' key first, then scalars."""
        if depth > 5:
            return None
        n = _to_number(node)
        if n is not None:
            return n
        if isinstance(node, dict):
            # Prefer 'value' key inside this subtree
            if 'value' in node:
                n = _to_number(node['value'])
                if n is not None:
                    return n
            # Try other scalar values
            for v in node.values():
                n = _to_number(v)
                if n is not None:
                    return n
            # Recurse one more level
            for v in node.values():
                if isinstance(v, dict):
                    n = _extract_from_subtree(v, depth + 1)
                    if n is not None:
                        return n
        return None

    # Pass 1: find keys containing 'final' — these are the conclusion fields
    final_keys_pattern = re.compile(r'final', re.IGNORECASE)
    candidates = []  # (depth, key_path, node)

    def _collect_final_keys(node, depth=0, path=""):
        if depth > 15:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                p = f"{path}.{k}"
                if final_keys_pattern.search(k):
                    candidates.append((depth, p, v))
                _collect_final_keys(v, depth + 1, p)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _collect_final_keys(v, depth + 1, f"{path}[{i}]")

    _collect_final_keys(obj)

    # Sort by depth descending — deepest 'final' key is most likely the actual answer
    candidates.sort(key=lambda x: x[0], reverse=True)
    for _, path, node in candidates:
        n = _extract_from_subtree(node)
        if n is not None:
            return n

    # Pass 2: fallback — look for 'answer'/'result' keys at any level (deepest first)
    fallback_keys = re.compile(r'^(answer|result)$', re.IGNORECASE)
    fallbacks = []

    def _collect_fallbacks(node, depth=0):
        if depth > 15:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                if fallback_keys.match(k):
                    fallbacks.append((depth, v))
                _collect_fallbacks(v, depth + 1)
        elif isinstance(node, list):
            for v in node:
                _collect_fallbacks(v, depth + 1)

    _collect_fallbacks(obj)
    fallbacks.sort(key=lambda x: x[0], reverse=True)
    for _, node in fallbacks:
        n = _extract_from_subtree(node)
        if n is not None:
            return n

    return None


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


def _extract_pred(text: str, benchmark: str):
    """Dispatch to the right extractor based on benchmark."""
    if benchmark == "gsm8k":
        return extract_number(text)
    return extract_answer_generic(text, bbh_options=(benchmark == "bbh"))


def extract_bbh_option(text: str):
    """Extract BBH multiple-choice option label (A)~(F) from text.
    Returns the option string like '(A)' or None if not found."""
    # Priority 1: "the answer is (X)" pattern
    m = re.search(r'[Tt]he answer is[:\s]*\(?([A-F])\)?', text)
    if m:
        return f"({m.group(1)})"
    # Priority 2: "Answer: (X)" or "answer: X"
    m = re.search(r'[Aa]nswer[:\s]*\(?([A-F])\)?', text)
    if m:
        return f"({m.group(1)})"
    # Priority 3: option label at start of text or after newline
    m = re.search(r'(?:^|\n)\s*\(?([A-F])\)?[\s\.\)]', text)
    if m:
        return f"({m.group(1)})"
    # Priority 4: any standalone (X) pattern
    m = re.search(r'\(([A-F])\)', text)
    if m:
        return f"({m.group(1)})"
    return None


def extract_answer_generic(text: str, bbh_options: bool = False) -> str:
    text = text.strip()

    # BBH option extraction — try first if requested
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


def _normalize_answer(s: str) -> str:
    import string as _str
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in set(_str.punctuation))
    return ' '.join(s.split())


def _hotpotqa_f1(pred: str, gold: str) -> float:
    p_tokens = _normalize_answer(pred).split()
    g_tokens = _normalize_answer(gold).split()
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall    = num_same / len(g_tokens)
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


def _exec_code_answer(code: str):
    code_clean = re.sub(r"```python\n?|```\n?", "", code).strip()
    if "def solution" not in code_clean:
        return None
    try:
        ns = {}
        exec(code_clean, ns)
        result = ns["solution"]()
        return int(result) if float(result) == int(result) else float(result)
    except Exception:
        return None


# ── Method 실행기 ─────────────────────────────────────────────────────────────

def run_standard_io(problem, model_fn, benchmark):
    subtask = problem.get("subtask", "")
    context = problem.get("context", "")
    td = BBH_TASK_DESCRIPTIONS.get(subtask, "") if benchmark == "bbh" else ""
    prompt = p_standard_io.build_prompt(benchmark, problem["question"], td, context=context)
    t0 = time.time()
    response = model_fn(prompt)
    elapsed = round(time.time() - t0, 2)
    pred = _extract_pred(response, benchmark)
    return {"method": "standard_io", "prompt": prompt, "response": response,
            "predicted_answer": pred, "elapsed_seconds": elapsed}


def run_zero_shot_cot(problem, model_fn, benchmark):
    q = problem["question"]
    context = problem.get("context", "")
    p1 = p_zscot.build_reasoning_prompt(q, context=context)
    t0 = time.time()
    reasoning = model_fn(p1)
    p2 = p_zscot.build_extract_prompt(q, reasoning, context=context)
    answer_text = model_fn(p2)
    elapsed = round(time.time() - t0, 2)
    if benchmark == "gsm8k":
        pred = extract_number(answer_text) or extract_number(reasoning)
    else:
        pred = extract_answer_generic(answer_text, bbh_options=(benchmark == "bbh"))
    return {"method": "zero_shot_cot", "prompt_step1": p1, "reasoning": reasoning,
            "prompt_step2": p2, "answer_text": answer_text,
            "predicted_answer": pred, "elapsed_seconds": elapsed}


def run_least_to_most(problem, model_fn, benchmark):
    subtask = problem.get("subtask", "")
    context = problem.get("context", "")
    td = BBH_TASK_DESCRIPTIONS.get(subtask, "") if benchmark == "bbh" else ""
    q = problem["question"]
    t0 = time.time()

    # Pass 1: Decomposition
    decompose_prompt = p_l2m.build_decompose_prompt(
        benchmark, q, td, subtask, context=context)
    decompose_response = model_fn(decompose_prompt)
    sub_questions = p_l2m.parse_subquestions(decompose_response, q)

    # Pass 2: Sequential Solving
    solved = []  # list of (sub_question, answer_text)
    for sub_q in sub_questions:
        solve_prompt = p_l2m.build_solve_prompt(
            benchmark, q, sub_q, solved, td, context=context)
        answer = model_fn(solve_prompt)
        solved.append((sub_q, answer))

    # Final answer from last solving step
    final_response = solved[-1][1] if solved else ""
    elapsed = round(time.time() - t0, 2)
    pred = _extract_pred(final_response, benchmark)

    return {"method": "least_to_most",
            "decompose_prompt": decompose_prompt,
            "decomposition": decompose_response,
            "sub_questions": sub_questions,
            "solving_steps": [{"sub_question": sq, "answer": ans} for sq, ans in solved],
            "final_response": final_response,
            "predicted_answer": pred, "elapsed_seconds": elapsed}


def run_cot_sc(problem, model_fn_n, benchmark):
    subtask = problem.get("subtask", "")
    context = problem.get("context", "")
    prompt = p_cotsc.build_prompt(benchmark, problem["question"],
                                  task_description="", subtask=subtask,
                                  context=context)
    t0 = time.time()
    paths_raw = model_fn_n(prompt, COT_SC_N_PATHS)
    elapsed = round(time.time() - t0, 2)
    paths, answers = [], []
    for i, raw in enumerate(paths_raw):
        ans = _extract_pred(raw, benchmark)
        paths.append({"path_id": i+1, "reasoning": raw, "extracted_answer": ans})
        if ans is not None:
            answers.append(str(ans))
    if answers:
        counter = Counter(answers)
        final_answer = counter.most_common(1)[0][0]
        vote_dist = dict(counter)
    else:
        final_answer, vote_dist = None, {}
    return {"method": "cot_sc", "prompt": prompt, "paths": paths,
            "vote_distribution": vote_dist, "predicted_answer": final_answer,
            "elapsed_seconds": elapsed}


def run_self_refine(problem, model_fn, benchmark):
    subtask = problem.get("subtask", "")
    context = problem.get("context", "")
    td = BBH_TASK_DESCRIPTIONS.get(subtask, "") if benchmark == "bbh" else ""
    q = problem["question"]
    t0 = time.time()

    # Step 1: Initial generation (Equation 1)
    pgen = p_selfrefine.build_pgen(benchmark, q, td, context=context)
    solution = model_fn(pgen)

    iterations = []
    history = []  # ✅ 추가: Equation 4 히스토리 누적
    gold = problem.get("answer")  # GSM8K oracle stopping용

    for i in range(p_selfrefine.MAX_ITERATIONS):
        # GSM8K oracle stopping (Appendix O): exec하여 정답이면 중단
        if benchmark == "gsm8k" and gold is not None:
            if p_selfrefine.is_stop_gsm8k(solution, float(gold)):
                break

        # Step 2: Feedback (Equation 2)
        pfb = p_selfrefine.build_pfb(benchmark, q, solution, td, context=context)
        feedback = model_fn(pfb)

        if p_selfrefine.is_stop(feedback):
            iterations.append({"iteration": i+1, "feedback": feedback,
                                "stopped_early": True, "refined_solution": None})
            break

        # ✅ 수정: 히스토리에 (solution, feedback) 쌍 추가
        history.append((solution, feedback))

        # ✅ 수정: history 리스트 전달 (기존: solution, feedback 개별 전달)
        prefine = p_selfrefine.build_prefine(benchmark, q, history, td, context=context)
        refined = model_fn(prefine)

        iterations.append({"iteration": i+1, "feedback": feedback,
                            "stopped_early": False, "refined_solution": refined})
        solution = refined

    elapsed = round(time.time() - t0, 2)

    if benchmark == "gsm8k":
        pred = _exec_code_answer(solution) or extract_number(solution)
    else:
        pred = extract_answer_generic(solution, bbh_options=(benchmark == "bbh"))

    return {"method": "self_refine", "initial_pgen": pgen, "iterations": iterations,
            "final_solution": solution, "predicted_answer": pred,
            "total_iterations": len(iterations), "elapsed_seconds": elapsed}


def run_self_discover(problem, model_fn, benchmark):
    subtask = problem.get("subtask", "")
    context = problem.get("context", "")
    td = p_selfdiscover.get_task_description(benchmark, subtask)
    q = problem["question"]
    t0 = time.time()
    # Stage 1: SELECT → resolve module numbers to full descriptions
    selected_raw = model_fn(p_selfdiscover.build_select_prompt(td))
    selected = p_selfdiscover.resolve_selected_modules(selected_raw)

    # Stage 1: ADAPT — detect failure (returned selected verbatim) and retry
    adapt_prompt = p_selfdiscover.build_adapt_prompt(td, selected)
    adapted = model_fn(adapt_prompt)
    if not adapted.strip() or adapted.strip() == selected.strip() or len(adapted.strip()) < len(selected.strip()) * 1.3:
        print(f"      [self_discover] ADAPT retry — output too similar to selected ({len(adapted)} vs {len(selected)} chars)")
        adapted = model_fn(adapt_prompt)
    if not adapted.strip():
        adapted = selected

    # Stage 1: IMPLEMENT — retry up to 3 times if empty
    structure = ""
    impl_prompt = p_selfdiscover.build_implement_prompt(td, adapted)
    for impl_attempt in range(3):
        structure = model_fn(impl_prompt)
        if structure.strip():
            break
        print(f"      [self_discover] IMPLEMENT retry {impl_attempt + 1}/3 — empty response")
    apply_prompt = p_selfdiscover.build_apply_prompt(td, structure, q, context=context)
    response  = model_fn(apply_prompt)
    elapsed = round(time.time() - t0, 2)
    if benchmark in ("gsm8k", "bbh"):
        # Priority: JSON keys → "the answer is" pattern → last number
        pred = extract_number_from_json(response) or extract_number(response)
    else:
        pred = extract_answer_generic(response, bbh_options=(benchmark == "bbh"))
    return {"method": "self_discover", "task_description": td,
            "stage1": {"selected_modules_raw": selected_raw,
                       "selected_modules": selected, "adapted_modules": adapted,
                       "reasoning_structure": structure},
            "apply_prompt": apply_prompt, "response": response,
            "predicted_answer": pred, "elapsed_seconds": elapsed}


# ── 문제 단위 6 Methods 동시 실행 ─────────────────────────────────────────────

def run_all_methods(problem, benchmark, model_fn, model_fn_n, dry_run=False):
    """
    문제 하나에 6 methods 모두 실행 후 결과 반환.
    dry_run=True 이면 API 호출 없이 항상 정답 반환.
    """
    runners = {
        "standard_io":   lambda: run_standard_io(problem, model_fn, benchmark),
        "zero_shot_cot": lambda: run_zero_shot_cot(problem, model_fn, benchmark),
        "least_to_most": lambda: run_least_to_most(problem, model_fn, benchmark),
        "cot_sc":        lambda: run_cot_sc(problem, model_fn_n, benchmark),
        "self_refine":   lambda: run_self_refine(problem, model_fn, benchmark),
        "self_discover": lambda: run_self_discover(problem, model_fn, benchmark),
    }

    method_results = {}
    correct_methods = []
    failed_methods  = []
    subtask = problem.get("subtask", "")

    for method_name in ALL_METHODS:
        try:
            if dry_run:
                result = {"method": method_name,
                          "predicted_answer": problem["answer"],
                          "elapsed_seconds": 0.0, "dry_run": True}
            else:
                result = runners[method_name]()

            pred    = result.get("predicted_answer")
            correct = is_correct(pred, problem["answer"], benchmark, subtask)
            result["correct"]     = correct
            result["gold_answer"] = problem["answer"]
            method_results[method_name] = result

            if correct:
                correct_methods.append(method_name)
                print(f"      ✓ {method_name}: {pred}")
            else:
                failed_methods.append(method_name)
                print(f"      ✗ {method_name}: {pred}  (gold: {problem['answer']})")

        except Exception as e:
            print(f"      ERROR [{method_name}]: {e}")
            traceback.print_exc()
            method_results[method_name] = {
                "method": method_name, "error": str(e),
                "correct": False, "gold_answer": problem["answer"]
            }
            failed_methods.append(method_name)

    return {
        "method_results":  method_results,
        "all_correct":     len(failed_methods) == 0,
        "correct_methods": correct_methods,
        "failed_methods":  failed_methods
    }


# ── 진행상황 저장/로드 ────────────────────────────────────────────────────────

def load_progress() -> dict:
    path = os.path.join(COLLECTED_DIR, "progress.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "gsm8k":    {"pool_cursor": 0, "collected_count": 0, "done": False},
        "hotpotqa": {"pool_cursor": 0, "collected_count": 0, "done": False},
        "bbh":      {"pool_cursor": 0, "collected_count": 0, "done": False},
        "started_at": datetime.now().isoformat(),
        "last_updated": None
    }


def save_progress(progress: dict):
    progress["last_updated"] = datetime.now().isoformat()
    with open(os.path.join(COLLECTED_DIR, "progress.json"), "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def load_collected(benchmark: str) -> list:
    path = os.path.join(COLLECTED_DIR, f"{benchmark}_collected.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_collected(benchmark: str, collected: list):
    path = os.path.join(COLLECTED_DIR, f"{benchmark}_collected.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(collected, f, ensure_ascii=False, indent=2)


# ── 핵심: Pool 기반 동적 수집 ─────────────────────────────────────────────────

def collect_benchmark(benchmark, pool, model_fn, model_fn_n,
                      progress, target=TARGET_PER_BENCHMARK, dry_run=False):
    """
    Pool에서 cursor 위치부터 순차 처리.
    "6 methods 모두 정답"인 문제만 collected에 추가.
    목표 달성 또는 Pool 소진 시 종료.
    """
    bp = progress[benchmark]
    if bp["done"]:
        print(f"\n  [{benchmark.upper()}] 이미 완료 ({bp['collected_count']}/{target})")
        return

    collected = load_collected(benchmark)
    cursor    = bp["pool_cursor"]

    print(f"\n{'='*65}")
    print(f"  [{benchmark.upper()}]  pool={len(pool)}개  cursor={cursor}  수집={len(collected)}/{target}")
    print(f"{'='*65}")

    while len(collected) < target and cursor < len(pool):
        problem = pool[cursor]
        subtask = problem.get("subtask", "")
        label   = f"[{subtask}] " if subtask else ""
        q_prev  = problem["question"][:60].replace("\n", " ")

        print(f"\n  문제 {cursor+1}/{len(pool)} | 수집 {len(collected)}/{target}")
        print(f"  {label}Q: {q_prev}...")
        print(f"  Gold: {problem['answer']}")
        print(f"  {'─'*55}")

        result = run_all_methods(problem, benchmark, model_fn, model_fn_n, dry_run)
        cursor += 1

        # GSM8K: 6/6 정답만 수집 (자동 채점 정확)
        # HotPotQA/BBH: 무조건 수집 (자동 채점 부정확, manual verification 예정)
        if benchmark == "gsm8k":
            should_collect = result["all_correct"]
        else:
            should_collect = True

        n_ok = len(result["correct_methods"])

        if should_collect:
            entry = {
                "collected_id":              len(collected),
                "benchmark":                 benchmark,
                "question":                  problem["question"],
                "answer":                    problem["answer"],
                "subtask":                   subtask,
                "source_idx":                problem.get("source_idx"),
                "pool_cursor_at_collection": cursor - 1,
                "timestamp":                 datetime.now().isoformat(),
                "method_results":            result["method_results"]
            }
            collected.append(entry)
            save_collected(benchmark, collected)
            if benchmark == "gsm8k":
                print(f"\n  ✅ COLLECTED! ({len(collected)}/{target}) — 6/6 correct")
            else:
                print(f"\n  ✅ COLLECTED! ({len(collected)}/{target}) — {n_ok}/6 auto-correct, manual verification needed")
        else:
            print(f"\n  ❌ SKIP — {n_ok}/6  실패: {result['failed_methods']}")

        bp["pool_cursor"]    = cursor
        bp["collected_count"] = len(collected)
        save_progress(progress)

    if len(collected) >= target:
        bp["done"] = True
        save_progress(progress)
        print(f"\n  🎉 [{benchmark.upper()}] 목표 달성! {len(collected)}/{target}")
    else:
        print(f"\n  ⚠️  [{benchmark.upper()}] Pool 소진. {len(collected)}/{target}")
        print(f"     data/loader.py의 POOL_SIZE를 늘린 후 --resume으로 재실행하세요.")


# ── 최종 데이터셋 통합 저장 ──────────────────────────────────────────────────

def save_final_dataset(benchmarks):
    all_entries = []
    for benchmark in benchmarks:
        for item in load_collected(benchmark):
            representations = {}
            for method, res in item.get("method_results", {}).items():
                rep = {k: res.get(k) for k in
                       ["predicted_answer", "correct", "elapsed_seconds"]}
                # method별 핵심 출력
                if method == "standard_io":
                    rep["response"] = res.get("response")
                elif method == "zero_shot_cot":
                    rep["reasoning"]    = res.get("reasoning")
                    rep["answer_text"]  = res.get("answer_text")
                elif method == "least_to_most":
                    rep["decomposition"] = res.get("decomposition")
                    rep["sub_questions"] = res.get("sub_questions")
                    rep["solving_steps"] = res.get("solving_steps")
                    rep["final_response"] = res.get("final_response")
                elif method == "cot_sc":
                    rep["paths"]           = res.get("paths")
                    rep["vote_distribution"] = res.get("vote_distribution")
                elif method == "self_refine":
                    rep["final_solution"]  = res.get("final_solution")
                    rep["total_iterations"] = res.get("total_iterations")
                elif method == "self_discover":
                    rep["reasoning_structure"] = res.get("stage1", {}).get("reasoning_structure")
                    rep["response"]            = res.get("response")
                representations[method] = rep

            all_entries.append({
                "id":                      len(all_entries),
                "benchmark":               benchmark,
                "subtask":                 item.get("subtask", ""),
                "question":                item["question"],
                "answer":                  item["answer"],
                "reasoning_representations": representations
            })

    final_path = os.path.join(COLLECTED_DIR, "final_dataset.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*65}")
    print(f"  📦 final_dataset.json 저장 완료")
    print(f"  총 문제: {len(all_entries)}개  |  총 응답: {len(all_entries)*6}개")
    print(f"  경로: {final_path}")
    print(f"{'='*65}")
    return final_path


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pool-Based Reasoning Data Collector")
    parser.add_argument("--benchmark", choices=["gsm8k","hotpotqa","bbh","all"],
                        default="all")
    parser.add_argument("--target",   type=int, default=TARGET_PER_BENCHMARK,
                        help=f"벤치마크당 수집 목표 (기본: {TARGET_PER_BENCHMARK})")
    parser.add_argument("--dry-run",  action="store_true",
                        help="API 없이 파이프라인 테스트")
    parser.add_argument("--resume",   action="store_true",
                        help="이전 진행상황 이어서 실행")
    parser.add_argument("--finalize", action="store_true",
                        help="final_dataset.json만 생성")
    args = parser.parse_args()

    benchmarks = ["gsm8k","hotpotqa","bbh"] if args.benchmark == "all" else [args.benchmark]

    if args.finalize:
        save_final_dataset(benchmarks)
        return

    # 모델 로드
    if not args.dry_run:
        import models.gpt as gpt
        model_fn   = gpt.call
        model_fn_n = gpt.call_n
    else:
        print("\n⚠️  DRY-RUN 모드 (API 호출 없음, 항상 정답 반환)")
        model_fn   = lambda p: "[DRY RUN]"
        model_fn_n = lambda p, n: ["[DRY RUN]"] * n

    progress = load_progress()

    if args.resume:
        print("\n📂 Resume 모드")
        for bm in benchmarks:
            bp = progress.get(bm, {})
            print(f"  {bm}: cursor={bp.get('pool_cursor',0)}, "
                  f"collected={bp.get('collected_count',0)}, done={bp.get('done',False)}")

    print(f"\n{'='*65}")
    print(f"  Reasoning Data Collector")
    print(f"  목표: 벤치마크당 {args.target}문제 × 6 methods = {args.target*6}개")
    print(f"  벤치마크: {benchmarks}")
    print(f"{'='*65}")

    pools = load_all_pools()
    start = time.time()

    for benchmark in benchmarks:
        if args.target != TARGET_PER_BENCHMARK:
            bm_target = args.target  # --target 명시 시 그대로 사용
        elif benchmark == "gsm8k":
            bm_target = TARGET_PER_BENCHMARK
        else:
            bm_target = TARGET_PER_BENCHMARK_LENIENT

        collect_benchmark(
            benchmark=benchmark,
            pool=pools[benchmark],
            model_fn=model_fn,
            model_fn_n=model_fn_n,
            progress=progress,
            target=bm_target,
            dry_run=args.dry_run
        )

    elapsed = round(time.time() - start, 1)
    all_done = all(progress.get(bm, {}).get("done", False) for bm in benchmarks)

    print(f"\n\n{'='*65}")
    print(f"  수집 결과  (소요: {elapsed}초)")
    print(f"{'='*65}")
    for bm in benchmarks:
        bp = progress.get(bm, {})
        status = "✅" if bp.get("done") else "⚠️ "
        print(f"  {bm:12s}: {bp.get('collected_count',0):3d}/{args.target}  "
              f"{status}  (pool 소비: {bp.get('pool_cursor',0)}개)")

    if all_done:
        print(f"\n  🎉 전체 완료!")
        save_final_dataset(benchmarks)
    else:
        print(f"\n  ℹ️  미완료 항목은 --resume으로 이어서 실행하세요.")


if __name__ == "__main__":
    main()
