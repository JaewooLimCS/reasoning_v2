"""
Chain-of-Thought Self-Consistency (CoT-SC)
==========================================
Method: C1 in taxonomy
Paper: Wang et al. 2023, "Self-Consistency Improves Chain of Thought Reasoning"
       arXiv:2203.11171, ICLR 2023

Key idea: Sample N diverse CoT paths via temperature sampling (T=0.7),
          then majority vote for final answer.

8-shot CoT prompt: Wei et al. 2022 Figure 1 / Appendix A
(Used identically in Wang et al. 2023 for self-consistency)

Gemini only (GPT-5 does not support temperature != 1.0)
Temperature: 0.7, N=5 paths
"""

# ============================================================
# GSM8K
# Exact 8-shot CoT prompt from Wei et al. 2022 / Wang et al. 2023
# Source: Paper Figure 1, Appendix A
# ============================================================
GSM8K_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 4 * 5 = 20 computers were added. 9 + 20 = 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8.

Q: {question}
A:"""

# ============================================================
# HotPotQA
# 8-shot CoT prompt for multi-hop QA, same structure
# ============================================================
HOTPOTQA_PROMPT = """Q: Were Scott Derrickson and Ed Wood of the same nationality?
A: Scott Derrickson is an American director. Ed Wood was an American director. So they are of the same nationality. The answer is: Yes.

Q: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
A: Shirley Temple portrayed Corliss Archer in Kiss and Tell. Shirley Temple was the US Ambassador to Ghana and to Czechoslovakia. The answer is: Ambassador.

Q: Are director of film Junglee and director of film Baghban both from the same country?
A: Junglee was directed by Subodh Mukerji. Baghban was directed by Ravi Chopra. Both are from India. The answer is: Yes.

Q: The Oberoi family is part of a hotel company that has a head office in what city?
A: The Oberoi family is part of The Oberoi Group. The Oberoi Group has its head office in Delhi, India. The answer is: Delhi.

Q: What nationality was James Henry Miller's wife?
A: James Henry Miller was the American journalist and social activist known as Henry Miller. His wife was Lepska, a Polish-born American. The answer is: Polish-American.

Q: Which magazine was started first, Arthur's Magazine or First for Women?
A: Arthur's Magazine was started in 1844. First for Women was started in 1989. So Arthur's Magazine was started first. The answer is: Arthur's Magazine.

Q: Were Pavel Urysohn and Leonid Levin born in the same country?
A: Pavel Urysohn was born in Odessa, Russian Empire. Leonid Levin was born in Dnepropetrovsk, Soviet Union. Both were born in what was the Russian Empire / Soviet Union. The answer is: Yes.

Q: Are both Canggu and Seminyak located in Bali?
A: Canggu is a village in Bali. Seminyak is a district in Bali. Both are located in Bali. The answer is: Yes.

Context:
{context}

Q: {question}
A:"""

# ============================================================
# BBH
# 3-shot CoT prompt per subtask
# ============================================================
BBH_PROMPT = """{task_description}

{few_shot_examples}
Q: {question}
A:"""

BBH_FEW_SHOTS = {
    "multistep_arithmetic_two": """Q: ((-5 + 9 * -4 - 0) * (4 + -7 + 0 * -5)) =
A: Let's compute step by step. 9 * -4 = -36. -5 + -36 - 0 = -41. -7 + 0 * -5 = -7. 4 + -7 = -3. -41 * -3 = 123. The answer is 123.

Q: ((-9 * 7 + -5 + 0) * (-3 + -2 * -4 - -8)) =
A: Let's compute step by step. -9 * 7 = -63. -63 + -5 + 0 = -68. -2 * -4 = 8. -3 + 8 - -8 = -3 + 8 + 8 = 13. -68 * 13 = -884. The answer is -884.

Q: ((3 + 7 * 9 * -3) * (8 + -6 - 2 * -1)) =
A: Let's compute step by step. 7 * 9 = 63. 63 * -3 = -189. 3 + -189 = -186. 2 * -1 = -2. 8 + -6 - -2 = 8 + -6 + 2 = 4. -186 * 4 = -744. The answer is -744.

""",
    "disambiguation_qa": """Q: In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.
Sentence: The trophy doesn't fit into the brown suitcase because it's too large.
(A) The trophy is too large. (B) The suitcase is too large. (C) Ambiguous.
A: The trophy is the thing that doesn't fit. If it is too large, the trophy is too large. The answer is (A).

Q: In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.
Sentence: The lawyer asked the witness a question, but he was reluctant to answer.
(A) The lawyer was reluctant to answer. (B) The witness was reluctant to answer. (C) Ambiguous.
A: There is no clear indication of who was reluctant to answer. The answer is (C).

Q: In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.
Sentence: Sam tried to paint a picture of John, but he couldn't get the nose right.
(A) Sam couldn't get the nose right. (B) John couldn't get the nose right. (C) Ambiguous.
A: Sam is the one trying to paint, so he is the one who couldn't get the nose right. The answer is (A).

""",
    "date_understanding": """Q: 2015 is coming in 36 hours. What is the date one week from today in MM/DD/YYYY?
A: If 2015 is coming in 36 hours, then today is 12/30/2014. One week from today is 01/06/2015. The answer is 01/06/2015.

Q: The first day of 2019 is a Tuesday, and today is the first Monday of 2019. What is the date today in MM/DD/YYYY?
A: The first day of 2019 is a Tuesday. The first Monday of 2019 is January 7th, 2019. The answer is 01/07/2019.

Q: The concert was scheduled to be on 06/01/1943, but was delayed by one day to today. What is the date 10 days ago in MM/DD/YYYY?
A: The concert was scheduled to be on 06/01/1943, but was delayed by one day, so today is 06/02/1943. Ten days ago is 05/23/1943. The answer is 05/23/1943.

""",
    "logical_deduction_five_objects": """Q: The following paragraphs each describe a set of five objects arranged in a fixed order. The statements are logically consistent within each paragraph. In an antique car show, there are five vehicles: a motorcyle, a limousine, a tractor, a bus, and a convertible. The motorcyle is the oldest. The tractor is newer than the limousine. The bus is newer than the convertible. The convertible is newer than the tractor.
A: The motorcycle is oldest. The order from oldest to newest is: motorcycle, limousine, tractor, convertible, bus. The answer is: motorcycle.

Q: The following paragraphs each describe a set of five objects arranged in a fixed order. A fruit stand sells five fruits: peaches, limes, plums, apples, and mangoes. The plums are less expensive than the mangoes. The apples are less expensive than the limes. The mangoes are less expensive than the apples. The peaches are the most expensive.
A: Plums < Mangoes < Apples < Limes < Peaches. The cheapest fruit is plums. The answer is: plums.

Q: The following paragraphs each describe a set of five objects arranged in a fixed order. On a shelf, there are five books: a red book, a green book, a blue book, an orange book, and a purple book. The green book is to the right of the red book. The blue book is to the right of the green book. The orange book is to the left of the red book. The purple book is to the left of the orange book.
A: Purple, Orange, Red, Green, Blue from left to right. The leftmost book is purple. The answer is: purple.

"""
}


def build_prompt(benchmark: str, question: str,
                 task_description: str = "", subtask: str = "",
                 context: str = "") -> str:
    if benchmark == "gsm8k":
        return GSM8K_PROMPT.format(question=question)
    elif benchmark == "hotpotqa":
        return HOTPOTQA_PROMPT.format(context=context, question=question)
    elif benchmark == "bbh":
        few_shots = BBH_FEW_SHOTS.get(subtask, "")
        return BBH_PROMPT.format(
            task_description=task_description,
            few_shot_examples=few_shots,
            question=question
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")