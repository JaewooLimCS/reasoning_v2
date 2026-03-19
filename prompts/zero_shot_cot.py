"""
Zero-shot Chain-of-Thought
===========================
Method: A2 in taxonomy
Paper: Kojima et al. 2022, "Large Language Models are Zero-Shot Reasoners"
       arXiv:2205.11916

Key idea: Append "Let's think step by step." to trigger reasoning.
Two-step process:
  Step 1 — Reasoning: append trigger phrase → get reasoning chain
  Step 2 — Extract:   append "Therefore, the answer is" → get final answer

Prompt confirmed from paper abstract + Figure 1.
"""

# ============================================================
# STEP 1: Reasoning prompt
# Appends "Let's think step by step." to trigger CoT
# Identical across all benchmarks (paper uses single template)
# ============================================================
REASONING_PROMPT = """Q: {question}
A: Let's think step by step."""

HOTPOTQA_REASONING_PROMPT = """Context:
{context}

Q: {question}
A: Let's think step by step."""

# ============================================================
# STEP 2: Answer extraction prompt
# Appends the reasoning chain and extracts final answer
# ============================================================
EXTRACT_PROMPT = """Q: {question}
A: Let's think step by step.
{reasoning}

Therefore, the answer is"""

HOTPOTQA_EXTRACT_PROMPT = """Context:
{context}

Q: {question}
A: Let's think step by step.
{reasoning}

Therefore, the answer is"""


def build_reasoning_prompt(question: str, context: str = "") -> str:
    """Step 1: trigger CoT reasoning."""
    if context:
        return HOTPOTQA_REASONING_PROMPT.format(question=question, context=context)
    return REASONING_PROMPT.format(question=question)


def build_extract_prompt(question: str, reasoning: str, context: str = "") -> str:
    """Step 2: extract final answer from reasoning chain."""
    if context:
        return HOTPOTQA_EXTRACT_PROMPT.format(question=question, reasoning=reasoning, context=context)
    return EXTRACT_PROMPT.format(question=question, reasoning=reasoning)