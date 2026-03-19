"""
Least-to-Most Prompting (2-Pass)
=================================
Method: B1 in taxonomy
Paper: Zhou et al. 2022, "Least-to-Most Prompting Enables Complex Reasoning in LLMs"
       arXiv:2205.10625, ICLR 2023

Key idea: Two-pass pipeline (paper Section 3, Table 9)
  Pass 1 — Decomposition: Break complex problem into sub-questions
  Pass 2 — Sequential Solving: Solve sub-questions one by one,
           accumulating previous answers as context

Exemplars based on paper Table 9 (GSM8K) and adapted for HotPotQA/BBH.
"""

import re
from typing import List, Tuple

MAX_SUBQUESTIONS = 6

# ============================================================
# PASS 1: DECOMPOSITION PROMPTS
# ============================================================

GSM8K_DECOMPOSE = """Q: Elsa has 5 apples. Anna has 2 more apples than Elsa. How many apples do they have together?
A: To solve "How many apples do they have together?", I need to first answer:
1. How many apples does Anna have?
2. How many apples do they have together?

Q: {question}
A: To solve this, I need to first answer:"""

HOTPOTQA_DECOMPOSE = """Q: Were Scott Derrickson and Ed Wood of the same nationality?
A: To answer this, I need to first answer:
1. What is Scott Derrickson's nationality?
2. What is Ed Wood's nationality?
3. Are they of the same nationality?

Context:
{context}

Q: {question}
A: To answer this, I need to first answer:"""

BBH_DECOMPOSE = """{task_description}

Q: {example_question}
A: To solve this, I need to first answer:
{example_subquestions}

Q: {question}
A: To solve this, I need to first answer:"""

BBH_DECOMPOSE_EXAMPLES = {
    "multistep_arithmetic_two": {
        "example_question": "((2 + 3) * 4) - 1 =",
        "example_subquestions": "1. What is 2 + 3?\n2. What is the result multiplied by 4?\n3. What is the result minus 1?",
    },
    "disambiguation_qa": {
        "example_question": "In the sentence 'The trophy doesn't fit in the brown suitcase because it's too big.' What is 'it'?\n(A) The trophy (B) The suitcase (C) Ambiguous",
        "example_subquestions": "1. What are the two objects mentioned?\n2. What is said to be too big?\n3. Which object does 'it' refer to?",
    },
    "date_understanding": {
        "example_question": "Yesterday was April 30, 2021. What is the date today in MM/DD/YYYY?",
        "example_subquestions": "1. What was the date yesterday?\n2. What is today's date?",
    },
    "logical_deduction_five_objects": {
        "example_question": "Alice is taller than Bob. Bob is taller than Carol. Is Alice taller than Carol?",
        "example_subquestions": "1. What do we know about Alice and Bob?\n2. What do we know about Bob and Carol?\n3. What can we conclude about Alice and Carol?",
    },
}

# ============================================================
# PASS 2: SOLVING PROMPTS
# ============================================================

GSM8K_SOLVE = """Q: Elsa has 5 apples. Anna has 2 more apples than Elsa. How many apples do they have together?

Q: How many apples does Anna have?
A: Anna has 2 more apples than Elsa. Elsa has 5 apples. So Anna has 2 + 5 = 7 apples. The answer is: 7.

Q: How many apples do they have together?
(We already know: Anna has 7 apples.)
A: Elsa has 5 apples. Anna has 7 apples. 5 + 7 = 12. The answer is: 12.

Q: {question}

{solved_context}Q: {sub_question}
{prior_knowledge}A:"""

HOTPOTQA_SOLVE = """Context:
{context}

Q: {question}

{solved_context}Q: {sub_question}
{prior_knowledge}A:"""

BBH_SOLVE = """{task_description}

Q: {question}

{solved_context}Q: {sub_question}
{prior_knowledge}A:"""


# ============================================================
# Decomposition prompt builders
# ============================================================

def build_decompose_prompt(benchmark: str, question: str,
                           task_description: str = "", subtask: str = "",
                           context: str = "") -> str:
    if benchmark == "gsm8k":
        return GSM8K_DECOMPOSE.format(question=question)
    elif benchmark == "hotpotqa":
        return HOTPOTQA_DECOMPOSE.format(context=context, question=question)
    elif benchmark == "bbh":
        ex = BBH_DECOMPOSE_EXAMPLES.get(subtask,
             BBH_DECOMPOSE_EXAMPLES["multistep_arithmetic_two"])
        return BBH_DECOMPOSE.format(
            task_description=task_description,
            example_question=ex["example_question"],
            example_subquestions=ex["example_subquestions"],
            question=question,
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


# ============================================================
# Solving prompt builders
# ============================================================

def _format_prior_knowledge(solved: List[Tuple[str, str]]) -> str:
    """Format accumulated (sub_question, answer) pairs as prior knowledge."""
    if not solved:
        return ""
    parts = [ans for _, ans in solved]
    return "(We already know: " + " ".join(parts) + ")\n"


def _format_solved_context(solved: List[Tuple[str, str]]) -> str:
    """Format previously solved sub-questions as Q/A pairs."""
    if not solved:
        return ""
    blocks = []
    for sq, ans in solved:
        blocks.append(f"Q: {sq}\nA: {ans}")
    return "\n\n".join(blocks) + "\n\n"


def build_solve_prompt(benchmark: str, question: str, sub_question: str,
                       solved: List[Tuple[str, str]],
                       task_description: str = "", context: str = "") -> str:
    solved_context = _format_solved_context(solved)
    prior_knowledge = _format_prior_knowledge(solved)

    if benchmark == "gsm8k":
        return GSM8K_SOLVE.format(
            question=question,
            solved_context=solved_context,
            sub_question=sub_question,
            prior_knowledge=prior_knowledge,
        )
    elif benchmark == "hotpotqa":
        return HOTPOTQA_SOLVE.format(
            context=context,
            question=question,
            solved_context=solved_context,
            sub_question=sub_question,
            prior_knowledge=prior_knowledge,
        )
    elif benchmark == "bbh":
        return BBH_SOLVE.format(
            task_description=task_description,
            question=question,
            solved_context=solved_context,
            sub_question=sub_question,
            prior_knowledge=prior_knowledge,
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


# ============================================================
# Sub-question parser
# ============================================================

def parse_subquestions(response: str, original_question: str = "") -> List[str]:
    """Parse numbered sub-questions from decomposition response.

    Looks for patterns like "1. ...", "2. ...", etc.
    Falls back to original question as single sub-question if parsing fails.
    """
    lines = response.strip().split("\n")
    subqs = []
    for line in lines:
        m = re.match(r'\s*\d+\.\s*(.+)', line.strip())
        if m:
            q = m.group(1).strip()
            if q:
                subqs.append(q)

    if not subqs:
        # Fallback: use original question
        return [original_question] if original_question else []

    return subqs[:MAX_SUBQUESTIONS]
