"""
Standard IO (Baseline)
======================
Method: A1 in taxonomy
No special prompting — question given directly, answer expected immediately.
Used as baseline across all benchmarks.
"""

# ============================================================
# GSM8K — direct Q→A
# ============================================================
GSM8K_PROMPT = """Q: {question}
A:"""

# ============================================================
# HotPotQA — direct Q→A
# ============================================================
HOTPOTQA_PROMPT = """Context:
{context}

Q: {question}
A:"""

# ============================================================
# BBH — task description + Q→A
# ============================================================
BBH_PROMPT = """{task_description}

Q: {question}
A:"""


def build_prompt(benchmark: str, question: str, task_description: str = "",
                 context: str = "") -> str:
    if benchmark == "gsm8k":
        return GSM8K_PROMPT.format(question=question)
    elif benchmark == "hotpotqa":
        return HOTPOTQA_PROMPT.format(context=context, question=question)
    elif benchmark == "bbh":
        return BBH_PROMPT.format(task_description=task_description, question=question)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")