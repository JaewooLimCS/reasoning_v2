"""
data/loader.py — Pool 기반 데이터 로더
========================================
각 벤치마크에서 대형 후보 Pool을 로드.
collect.py가 Pool에서 순차적으로 문제를 꺼내 처리.

Pool 크기:
  GSM8K    : 300개 (test split: 1319개)
  HotPotQA : 300개 (validation split: 7405개)
  BBH      : subtask당 50개 × 4 = 200개
"""

import json
import os
import random
from datasets import load_dataset

SEED = 42

GSM8K_POOL_SIZE           = 50
HOTPOTQA_POOL_SIZE        = 300
BBH_POOL_SIZE_PER_SUBTASK = 50   # × 4 subtasks = 200개

BBH_SUBTASKS = [
    "multistep_arithmetic_two",
    "disambiguation_qa",
    "date_understanding",
    "logical_deduction_five_objects"
]

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_gsm8k_pool(seed: int = SEED) -> list:
    print("  Loading GSM8K pool...")
    ds = load_dataset("gsm8k", "main", split="test")
    pool_size = min(GSM8K_POOL_SIZE, len(ds))
    random.seed(seed)
    indices = random.sample(range(len(ds)), pool_size)
    problems = []
    for i in sorted(indices):
        item = ds[i]
        answer_num = item["answer"].split("####")[-1].strip().replace(",", "")
        problems.append({
            "question":   item["question"],
            "answer":     answer_num,
            "type":       "math",
            "source_idx": i
        })
    print(f"    GSM8K pool: {len(problems)}개")
    return problems


def load_hotpotqa_pool(seed: int = SEED) -> list:
    print("  Loading HotPotQA pool...")
    ds = load_dataset("hotpot_qa", "fullwiki", split="validation")
    pool_size = min(HOTPOTQA_POOL_SIZE, len(ds))
    random.seed(seed)
    indices = random.sample(range(len(ds)), pool_size)
    problems = []
    for i in sorted(indices):
        item = ds[i]
        context_dict = {}
        for title, sentences in zip(item["context"]["title"],
                                    item["context"]["sentences"]):
            context_dict[title] = " ".join(sentences)
        supporting_titles = list(set(
            f[0] for f in zip(item["supporting_facts"]["title"],
                              item["supporting_facts"]["sent_id"])
        ))
        context = "\n\n".join([
            f"[{t}]: {context_dict[t]}"
            for t in supporting_titles if t in context_dict
        ])
        problems.append({
            "question":   item["question"],
            "answer":     item["answer"],
            "context":    context,
            "type":       "multihop_qa",
            "source_idx": i
        })
    print(f"    HotPotQA pool: {len(problems)}개")
    return problems


def load_bbh_pool(seed: int = SEED) -> list:
    print("  Loading BBH pool...")
    all_problems = []
    for subtask in BBH_SUBTASKS:
        print(f"    subtask: {subtask}")
        ds = load_dataset("lukaemon/bbh", subtask, split="test")
        pool_size = min(BBH_POOL_SIZE_PER_SUBTASK, len(ds))
        random.seed(seed)
        indices = random.sample(range(len(ds)), pool_size)
        for i in sorted(indices):
            item = ds[i]
            all_problems.append({
                "question":   item["input"],
                "answer":     item["target"],
                "subtask":    subtask,
                "type":       "bbh",
                "source_idx": i
            })
    print(f"    BBH pool: {len(all_problems)}개 (subtask당 {BBH_POOL_SIZE_PER_SUBTASK}개)")
    return all_problems


def load_all_pools(seed: int = SEED, use_cache: bool = True) -> dict:
    """
    모든 벤치마크 Pool 로드. 캐시 있으면 캐시 사용.

    Returns:
        {"gsm8k": [...], "hotpotqa": [...], "bbh": [...]}
    """
    cache_path = os.path.join(DATA_DIR, f"pool_seed{seed}.json")

    if use_cache and os.path.exists(cache_path):
        print(f"  캐시 로드: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            pool = json.load(f)
        for k, v in pool.items():
            print(f"    {k}: {len(v)}개 후보")
        return pool

    print(f"\nPool 생성 중 (seed={seed})...")
    pool = {
        "gsm8k":    load_gsm8k_pool(seed),
        "hotpotqa": load_hotpotqa_pool(seed),
        "bbh":      load_bbh_pool(seed)
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)
    print(f"  Pool 캐시 저장: {cache_path}")
    return pool
