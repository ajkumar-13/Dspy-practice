"""
Blog 4.2: MIPROv2 Optimization
Run: python 01_miprov2.py
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Training data
trainset = [
    dspy.Example(question="What is the largest ocean?", answer="Pacific Ocean").with_inputs("question"),
    dspy.Example(question="Who discovered penicillin?", answer="Alexander Fleming").with_inputs("question"),
    dspy.Example(question="What is the hardest natural substance?", answer="Diamond").with_inputs("question"),
    dspy.Example(question="How many bones in the human body?", answer="206").with_inputs("question"),
    dspy.Example(question="What planet is known as the Red Planet?", answer="Mars").with_inputs("question"),
    # ... ideally 50+ examples for MIPROv2 to shine
]


# Metric
def answer_match(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()


# Program
program = dspy.ChainOfThought("question -> answer")

# -------------------------------------------------------------------------
# Option 1: Standard MIPROv2 optimization
# -------------------------------------------------------------------------
tp = dspy.MIPROv2(metric=answer_match, auto="medium", num_threads=8)

optimized = tp.compile(
    program,
    trainset=trainset,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
)

# Test
result = optimized(question="What is the smallest country in the world?")
print(f"Answer: {result.answer}")

# Save the optimized program
optimized.save("optimized_miprov2.json")
print("Optimized program saved to optimized_miprov2.json")

# -------------------------------------------------------------------------
# Option 2: Using a teacher model for higher-quality proposals
# -------------------------------------------------------------------------
# teacher_lm = dspy.LM("openai/gpt-4o")
#
# tp_teacher = dspy.MIPROv2(metric=answer_match, auto="medium", num_threads=8)
#
# optimized_teacher = tp_teacher.compile(
#     program,
#     trainset=trainset,
#     max_bootstrapped_demos=2,
#     max_labeled_demos=2,
#     teacher_settings=dict(lm=teacher_lm),
# )

# -------------------------------------------------------------------------
# Option 3: Zero-shot optimization (instructions only, no demos)
# -------------------------------------------------------------------------
# tp_zero = dspy.MIPROv2(metric=answer_match, auto="medium", num_threads=8)
#
# optimized_zero = tp_zero.compile(
#     program,
#     trainset=trainset,
#     max_bootstrapped_demos=0,
#     max_labeled_demos=0,
# )
