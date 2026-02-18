"""
Blog 4.3: GEPA Optimization
Run: python 01_gepa.py
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Training data (math/reasoning examples where GEPA excels)
trainset = [
    dspy.Example(
        question="If a shirt costs $25 and is 20% off, what do you pay?",
        answer="$20"
    ).with_inputs("question"),
    dspy.Example(
        question="A train travels 120 miles in 2 hours. What is its speed in mph?",
        answer="60 mph"
    ).with_inputs("question"),
    # ... more examples, ideally 50+
]


# Metric (with normalization for currency/number formatting)
def answer_match(example, prediction, trace=None):
    pred = prediction.answer.strip().lower().replace("$", "").replace(",", "")
    gold = example.answer.strip().lower().replace("$", "").replace(",", "")
    return pred == gold


# Program
program = dspy.ChainOfThought("question -> answer")

# Optimize with GEPA
tp = dspy.GEPA(
    metric=answer_match,
    num_threads=8,
)

optimized = tp.compile(program, trainset=trainset)

# Test
result = optimized(question="A book costs $30 and is 15% off. What do you pay?")
print(f"Answer: {result.answer}")

# Save the optimized program
optimized.save("optimized_gepa.json")
print("Optimized program saved to optimized_gepa.json")
