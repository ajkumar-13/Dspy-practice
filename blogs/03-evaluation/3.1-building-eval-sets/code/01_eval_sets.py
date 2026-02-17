"""
Blog 3.1: Building Evaluation Sets
Creating and managing DSPy examples and datasets.
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# --- Creating examples manually ---
examples = [
    dspy.Example(
        question="What is the capital of France?",
        answer="Paris",
    ).with_inputs("question"),
    dspy.Example(
        question="What is 2 + 2?",
        answer="4",
    ).with_inputs("question"),
    dspy.Example(
        question="Who wrote Romeo and Juliet?",
        answer="William Shakespeare",
    ).with_inputs("question"),
]

# --- Split into train/dev ---
trainset = examples[:2]
devset = examples[2:]

print(f"Train size: {len(trainset)}")
print(f"Dev size: {len(devset)}")

# --- Inspect an example ---
ex = trainset[0]
print(f"\nExample inputs: {ex.inputs()}")
print(f"Example labels: {ex.labels()}")
