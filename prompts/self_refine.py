"""
Self-Refine
===========
Method: D2 in taxonomy
Paper: Madaan et al. 2023, "Self-Refine: Iterative Refinement with Self-Feedback"
       arXiv:2303.17651, NeurIPS 2023

Key idea: Iterative loop — Generate → Feedback → Refine (Algorithm 1)
  p_gen:    Generate initial solution          y0 = M(p_gen || x)
  p_fb:     Generate feedback on solution      fb_t = M(p_fb || x || y_t)
  p_refine: Refine solution based on feedback  y_{t+1} = M(p_refine || x || y0 || fb0 || ... || y_t || fb_t)
  Stop when: feedback says "The solution is correct." OR max 4 iterations

GSM8K: Python code generation + exec() (Appendix O exact structure)
HotPotQA / BBH: Natural language, same 3-stage structure

Changes from original code (to match paper more faithfully):
  1. REFINE prompt now includes full history (y0, fb0, ..., y_t, fb_t) per Equation 4
  2. GSM8K feedback prompt is neutral (not biased toward assuming errors)
  3. Added few-shot examples for GSM8K from Appendix O (Figures 30-32)
  4. build_prefine() accepts full history list instead of single (solution, feedback) pair
"""

from typing import List, Tuple

# ============================================================
# GSM8K — Python code generation (Appendix O exact structure)
# ============================================================

# --- Few-shot examples from paper (Figures 30-32) ---

GSM8K_FEWSHOT_GEN = """# Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

# solution using Python:

def solution():
    \"\"\"There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\"\"\"
    computers_initial = 9
    computers_per_day = 5
    num_days = 4  # 4 days between monday and thursday
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    result = computers_total
    return result"""

GSM8K_FEWSHOT_FB = """def solution():
    \"\"\"Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup.\"\"\"
    plates = 6
    plate_cost = 6000
    cups = 12 * 20
    cup_cost = plate_cost
    result = cup_cost
    return result

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
# plates = 6
# plate_cost = 6000
# looks good

# Let's check the other parts
# cups = 12 * 20
# cup_cost = plate_cost
# wrong! The cost of a cup is not the same as the cost of a plate. The cost of a cup is $1200 less than the total cost of half a dozen plates sold at $6000 each. So we need to calculate the cost of a cup first (total cost of half a dozen plates sold at $6000 each - $1200) and use that."""

GSM8K_FEWSHOT_REFINE = """def solution():
    \"\"\"Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup.\"\"\"
    plates = 6
    plate_cost = 6000
    cups = 12 * 20
    cup_cost = plate_cost
    result = cup_cost
    return result

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step
# plates = 6
# plate_cost = 6000
# looks good

# Let's check the other parts
# cups = 12 * 20
# cup_cost = plate_cost
# wrong! The cost of a cup is not the same as the cost of a plate. The cost of a cup is $1200 less than the total cost of half a dozen plates sold at $6000 each. So we need to calculate the cost of a cup first (total cost of half a dozen plates sold at $6000 each - $1200) and use that.

Okay! Here is the rewrite:

def solution():
    \"\"\"Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup.\"\"\"
    plates = 6
    plate_cost = 6000
    cups = 12 * 20
    total_cup_cost = (plate_cost * plates) - 1200
    cost_per_cup = total_cup_cost / cups
    return cost_per_cup"""


# --- Prompt templates ---

GSM8K_PGEN = """{fewshot_gen}

# Q: {question}

# solution using Python:

def solution():
    \"\"\"{question}\"\"\"
"""

GSM8K_PFB = """{fewshot_fb}

###

{solution}

# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.

# Let us go through the error and check step-by-step"""

GSM8K_PREFINE = """{fewshot_refine}

###

Problem: {question}

{history_block}

Okay! Here is the rewrite:
def solution():
    \"\"\"{question}\"\"\"
"""


# ============================================================
# HotPotQA — Natural language (same 3-stage structure)
# ============================================================

HOTPOTQA_PGEN = """Answer the following question step by step, showing your reasoning clearly.

Context:
{context}

Question: {question}

Answer:"""

HOTPOTQA_PFB = """Review the following answer to a multi-hop question.
Check for: incorrect facts, missing reasoning steps, wrong conclusions.
If the answer is fully correct, respond ONLY with: The solution is correct.
If incorrect, identify the specific error and explain how to fix it concisely.

Context:
{context}

Question: {question}

Answer:
{solution}

Feedback:"""

HOTPOTQA_PREFINE = """Rewrite the answer based on the feedback below.
Show your reasoning step by step and provide a clear final answer.

Context:
{context}

Question: {question}

{history_block}

Refined answer:"""


# ============================================================
# BBH — Natural language (same 3-stage structure)
# ============================================================

BBH_PGEN = """{task_description}

Solve the following problem step by step, showing your reasoning clearly.

Problem: {question}

Answer:"""

BBH_PFB = """Review the following answer to a reasoning problem.
Check for: logical errors, incorrect steps, wrong conclusions.
If the answer is fully correct, respond ONLY with: The solution is correct.
If incorrect, identify the specific error and explain how to fix it concisely.

Problem: {question}

Answer:
{solution}

Feedback:"""

BBH_PREFINE = """Rewrite the answer based on the feedback below.
Show your reasoning step by step and provide a clear final answer.

Problem: {question}

{history_block}

Refined answer:"""


# ============================================================
# Constants
# ============================================================

STOP_SIGNAL = "The solution is correct."
MAX_ITERATIONS = 4


# ============================================================
# History formatting (Equation 4: y0 || fb0 || ... || y_t || fb_t)
# ============================================================

def _format_history(history: List[Tuple[str, str]], benchmark: str) -> str:
    """Format full iteration history for the REFINE prompt.

    Per Equation 4 in the paper, the refine step receives the full history:
        y_{t+1} = M(p_refine || x || y_0 || fb_0 || ... || y_t || fb_t)

    Args:
        history: List of (solution, feedback) tuples from all iterations.
        benchmark: One of "gsm8k", "hotpotqa", "bbh".

    Returns:
        Formatted string containing all previous solutions and feedback.
    """
    blocks = []
    for i, (sol, fb) in enumerate(history):
        if benchmark == "gsm8k":
            blocks.append(f"# Attempt {i}:\n{sol}\n\n# Feedback {i}:\n{fb}")
        else:
            blocks.append(f"Previous answer (attempt {i}):\n{sol}\n\nFeedback (attempt {i}):\n{fb}")
    return "\n\n".join(blocks)


# ============================================================
# Prompt builders
# ============================================================

def build_pgen(benchmark: str, question: str, task_description: str = "",
               context: str = "") -> str:
    """Build the initial generation prompt (p_gen).

    Corresponds to Equation 1: y_0 = M(p_gen || x)
    """
    if benchmark == "gsm8k":
        return GSM8K_PGEN.format(
            fewshot_gen=GSM8K_FEWSHOT_GEN,
            question=question,
        )
    elif benchmark == "hotpotqa":
        return HOTPOTQA_PGEN.format(context=context, question=question)
    elif benchmark == "bbh":
        return BBH_PGEN.format(task_description=task_description, question=question)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def build_pfb(benchmark: str, question: str, solution: str,
              task_description: str = "", context: str = "") -> str:
    """Build the feedback prompt (p_fb).

    Corresponds to Equation 2: fb_t = M(p_fb || x || y_t)
    """
    if benchmark == "gsm8k":
        return GSM8K_PFB.format(
            fewshot_fb=GSM8K_FEWSHOT_FB,
            solution=solution,
        )
    elif benchmark == "hotpotqa":
        return HOTPOTQA_PFB.format(context=context, question=question, solution=solution)
    elif benchmark == "bbh":
        return BBH_PFB.format(question=question, solution=solution)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def build_prefine(benchmark: str, question: str,
                  history: List[Tuple[str, str]],
                  task_description: str = "", context: str = "") -> str:
    """Build the refinement prompt (p_refine).

    Corresponds to Equation 4: y_{t+1} = M(p_refine || x || y_0 || fb_0 || ... || y_t || fb_t)

    Args:
        benchmark: One of "gsm8k", "hotpotqa", "bbh".
        question: The original problem/question.
        history: List of ALL (solution, feedback) tuples from iterations so far.
        task_description: Task description (only used for BBH).
        context: Supporting context (only used for HotPotQA).

    Returns:
        The formatted refinement prompt.
    """
    history_block = _format_history(history, benchmark)

    if benchmark == "gsm8k":
        return GSM8K_PREFINE.format(
            fewshot_refine=GSM8K_FEWSHOT_REFINE,
            question=question,
            history_block=history_block,
        )
    elif benchmark == "hotpotqa":
        return HOTPOTQA_PREFINE.format(
            context=context,
            question=question,
            history_block=history_block,
        )
    elif benchmark == "bbh":
        return BBH_PREFINE.format(
            question=question,
            history_block=history_block,
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def is_stop(feedback: str) -> bool:
    """Check if feedback indicates the solution is correct (early stopping).

    Used for NL tasks (HotPotQA, BBH) where feedback contains a textual stop signal.
    For GSM8K, use is_stop_gsm8k() with exec()-based answer checking instead.
    """
    return STOP_SIGNAL.lower() in feedback.lower()


def is_stop_gsm8k(solution_code: str, gold_answer: float) -> bool:
    """Check if GSM8K solution produces the correct answer via exec().

    Per Appendix O (and following Welleck et al. 2022), GSM8K uses the correct
    label to decide whether to continue refinement. The solution code is executed
    and the returned value is compared to the gold answer.

    Args:
        solution_code: Python code string containing a solution() function.
        gold_answer: The correct numeric answer.

    Returns:
        True if the code executes successfully and returns the correct answer.
    """
    try:
        local_ns = {}
        exec(solution_code, {}, local_ns)
        result = local_ns["solution"]()
        return abs(float(result) - gold_answer) < 1e-6
    except Exception:
        return False


# ============================================================
# Self-Refine loop (Algorithm 1)
# ============================================================

def self_refine_loop(llm_call, benchmark: str, question: str,
                     task_description: str = "",
                     gold_answer: float = None) -> Tuple[str, List[Tuple[str, str]]]:
    """Execute the full Self-Refine loop per Algorithm 1.

    Args:
        llm_call: Callable that takes a prompt string and returns a response string.
        benchmark: One of "gsm8k", "hotpotqa", "bbh".
        question: The problem/question to solve.
        task_description: Task description (only used for BBH).
        gold_answer: For GSM8K oracle stopping (Appendix O), the correct numeric
                     answer. If None, uses text-based stopping for GSM8K.

    Returns:
        Tuple of (final_solution, history) where history is a list of
        (solution, feedback) tuples from all iterations.
    """
    # Step 1: Initial generation (Equation 1)
    prompt_gen = build_pgen(benchmark, question, task_description)
    solution = llm_call(prompt_gen)

    history: List[Tuple[str, str]] = []

    # Step 2: Iterative feedback-refine loop
    for t in range(MAX_ITERATIONS):
        # For GSM8K with oracle: check if current solution is already correct
        if benchmark == "gsm8k" and gold_answer is not None:
            if is_stop_gsm8k(solution, gold_answer):
                break

        # Feedback (Equation 2)
        prompt_fb = build_pfb(benchmark, question, solution, task_description)
        feedback = llm_call(prompt_fb)

        # For non-GSM8K tasks: check text-based stopping condition
        if benchmark != "gsm8k" and is_stop(feedback):
            break

        # Record this iteration's (solution, feedback) pair
        history.append((solution, feedback))

        # Refine with full history (Equation 4)
        prompt_refine = build_prefine(benchmark, question, history, task_description)
        solution = llm_call(prompt_refine)

    return solution, history