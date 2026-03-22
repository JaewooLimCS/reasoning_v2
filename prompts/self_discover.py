"""
SELF-DISCOVER
=============
Method: E1 in taxonomy
Paper: Zhou et al. 2024, "SELF-DISCOVER: Large Language Models Self-Compose
       Reasoning Structures"
       arXiv:2402.03620, NeurIPS 2024

Key idea: Two-stage pipeline
  Stage 1 (Self-Discover): SELECT → ADAPT → IMPLEMENT reasoning structure
  Stage 2 (Solve): Apply discovered structure to solve the instance

Meta-prompts: Paper Figure 10 (Appendix A, exact)
39 Reasoning Modules: Paper Table 2 (Appendix A, exact)
"""

import re as _re

# ============================================================
# 39 REASONING MODULES — Paper Table 2 (exact)
# ============================================================
REASONING_MODULES = """
1 How could I devise an experiment to help solve that problem?
2 Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.
3 How could I measure progress on this problem?
4 How can I simplify the problem so that it is easier to solve?
5 What are the key assumptions underlying this problem?
6 What are the potential risks and drawbacks of each solution?
7 What are the alternative perspectives or viewpoints on this problem?
8 What are the long-term implications of this problem and its solutions?
9 How can I break down this problem into smaller, more manageable parts?
10 Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating
the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying
potential biases or flaws in thinking.
11 Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions,
thinking beyond traditional boundaries, and encouraging imagination and originality.
12 Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the
diverse perspectives and expertise of a group to come up with effective solutions.
13 Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements.
Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic
solutions that address the system as a whole.
14 Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a
problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based
on a balanced analysis of risks and benefits.
15 Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases,
assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve
future approaches.
16 What is the core issue or problem that needs to be addressed?
17 What are the underlying causes or factors contributing to the problem?
18 Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?
19 What are the potential obstacles or challenges that might arise in solving this problem?
20 Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available,
and how can they be analyzed?
21 Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?
22 What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?
23 How can progress or success in solving the problem be measured or evaluated?
24 What indicators or metrics can be used?
25 Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or
theoretical problem?
26 Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?
27 Is the problem related to human behavior, such as a social, cultural, or psychological issue?
28 Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing
objectives?
29 Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?
30 Is the problem a design challenge that requires creative solutions and innovation?
31 Does the problem require addressing systemic or structural issues rather than just individual instances?
32 Is the problem time-sensitive or urgent, requiring immediate attention and action?
33 What kinds of solution typically are produced for this kind of problem specification?
34 Given the problem specification and the current best solution, have a guess about other possible solutions.
35 Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?
36 What is the best way to modify this current best solution, given what you know about these kinds of problem specification?
37 Ignoring the current best solution, create an entirely new solution to the problem.
38 Let’s think step by step.
39 Let’s make a step by step plan and implement it with good notion and explanation."""

# ============================================================
# STAGE 1: SELECT meta-prompt — Paper Figure 10 
# ============================================================
SELECT_PROMPT = """Select several reasoning modules that are crucial to utilize in order to solve the given task:

All reasoning module descriptions:
{reasoning_modules}

Task: {task_description}

Select several modules are crucial for solving the tasks above:"""

# ============================================================
# STAGE 1: ADAPT meta-prompt — Paper Figure 10
# ============================================================
ADAPT_PROMPT = """Rephrase and specify each reasoning module so that it better helps solving the task:

SELECTED module descriptions:
{selected_modules}

Task: {task_description}

Adapt each reasoning module description to better solve the tasks:"""

# ============================================================
# STAGE 1: IMPLEMENT meta-prompt — Paper Figure 10
# ============================================================
IMPLEMENT_PROMPT = """Operationalize the reasoning modules into a step-by-step reasoning plan in JSON format:

Adapted module descriptions:
{adapted_modules}

Task: {task_description}

Implement a reasoning structure (in JSON) for solvers to follow step-by-step and arrive at correct answers:"""

# ============================================================
# STAGE 2: APPLY (Solve) — Paper Figure 10 
# ============================================================
APPLY_PROMPT = """Follow the step-by-step reasoning plan in JSON to correctly solve the task. Fill in the values following the keys by reasoning specifically about the task given. Do not simply rephrase the keys.

Reasoning Structure:
{reasoning_structure}

Task: {task_description}

{context_block}Question: {question}

Thus, the final answer is"""


# ============================================================
# Module lookup: parse REASONING_MODULES into {number: full_text}
# ============================================================

_MODULE_LOOKUP: dict = {}

def _build_module_lookup():
    """Parse REASONING_MODULES into {int: str} mapping once."""
    if _MODULE_LOOKUP:
        return
    lines = REASONING_MODULES.strip().splitlines()
    current_num = None
    current_text = []
    for line in lines:
        m = _re.match(r'^(\d{1,2})\s+', line)
        if m:
            if current_num is not None:
                _MODULE_LOOKUP[current_num] = ' '.join(current_text).strip()
            current_num = int(m.group(1))
            current_text = [line]
        elif current_num is not None:
            current_text.append(line)
    if current_num is not None:
        _MODULE_LOOKUP[current_num] = ' '.join(current_text).strip()

_build_module_lookup()


def resolve_selected_modules(selected_raw: str) -> str:
    """Normalize SELECT output: if only numbers, expand to full descriptions.

    SELECT sometimes returns just "4, 9, 10, 16, 38, 39" without descriptions.
    This function detects that and looks up the full module text, ensuring
    ADAPT always receives complete descriptions.
    """
    stripped = selected_raw.strip()
    if not stripped:
        return stripped

    # Heuristic: if the response is short and looks like just numbers/commas,
    # expand them. If it already contains descriptions (long text), pass through.
    nums = _re.findall(r'\b(\d{1,2})\b', stripped)
    non_digit_chars = _re.sub(r'[\d,\s\-\.]', '', stripped)

    # If almost all content is numbers/punctuation → needs expansion
    if nums and len(non_digit_chars) < 30:
        expanded = []
        for n in nums:
            n_int = int(n)
            if n_int in _MODULE_LOOKUP:
                expanded.append(f"- {_MODULE_LOOKUP[n_int]}")
        if expanded:
            return '\n'.join(expanded)

    return stripped


def build_select_prompt(task_description: str) -> str:
    return SELECT_PROMPT.format(
        reasoning_modules=REASONING_MODULES,
        task_description=task_description
    )


def build_adapt_prompt(task_description: str, selected_modules: str) -> str:
    return ADAPT_PROMPT.format(
        selected_modules=selected_modules,
        task_description=task_description
    )


def build_implement_prompt(task_description: str, adapted_modules: str) -> str:
    return IMPLEMENT_PROMPT.format(
        adapted_modules=adapted_modules,
        task_description=task_description
    )


DEFAULT_STRUCTURE = """{
    "Understand": {
        "Step 1": "Identify the key information and what the question asks for",
        "Step 2": "List the given values and relationships"
    },
    "Solve": {
        "Step 3": "Break the problem into sub-steps",
        "Step 4": "Solve each sub-step showing your work"
    },
    "Verify": {
        "Step 5": "Check the answer for correctness and state the final answer"
    }
}"""


def build_apply_prompt(task_description: str, reasoning_structure: str,
                       question: str, context: str = "") -> str:
    if not reasoning_structure.strip():
        reasoning_structure = DEFAULT_STRUCTURE
    context_block = f"Context:\n{context}\n\n" if context else ""
    return APPLY_PROMPT.format(
        reasoning_structure=reasoning_structure,
        task_description=task_description,
        context_block=context_block,
        question=question
    )


# ============================================================
# Task descriptions per benchmark
# ============================================================
TASK_DESCRIPTIONS = {
    "gsm8k": "Solve a multi-step grade school math word problem. Compute the final numeric answer.",
    "hotpotqa": "Answer a multi-hop question that requires reasoning across multiple facts. Provide a concise factual answer.",
    "bbh": {
        "multistep_arithmetic_two": "Evaluate a mathematical expression with nested arithmetic operations following order of operations.",
        "disambiguation_qa": "Identify the antecedent of a pronoun in a sentence, or determine if it is ambiguous. Choose from the given options.",
        "date_understanding": "Determine a specific date based on given information. Answer in MM/DD/YYYY format.",
        "logical_deduction_five_objects": "Given logical constraints about the ordering of five objects, deduce the correct ordering or answer the question about their arrangement."
    }
}


def get_task_description(benchmark: str, subtask: str = "") -> str:
    if benchmark == "bbh":
        return TASK_DESCRIPTIONS["bbh"].get(subtask, "Solve the reasoning task.")
    return TASK_DESCRIPTIONS.get(benchmark, "Solve the given task.")