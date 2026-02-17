"""
Blog 1.3: Predict vs ChainOfThought vs ProgramOfThought
Compare the same signature across different modules.
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

QUESTION = "If a train travels 120 km in 2 hours, and then 180 km in 3 hours, what is its average speed for the entire journey?"

# --- Predict: Direct answer ---
print("=" * 60)
print("PREDICT (Direct)")
print("=" * 60)
predict = dspy.Predict("question -> answer")
result = predict(question=QUESTION)
print(f"Answer: {result.answer}\n")

# --- ChainOfThought: Step-by-step reasoning ---
print("=" * 60)
print("CHAIN OF THOUGHT")
print("=" * 60)
cot = dspy.ChainOfThought("question -> answer")
result = cot(question=QUESTION)
print(f"Reasoning: {result.reasoning}")
print(f"Answer: {result.answer}\n")

# --- ProgramOfThought: Code execution ---
print("=" * 60)
print("PROGRAM OF THOUGHT")
print("=" * 60)
pot = dspy.ProgramOfThought("question -> answer")
result = pot(question=QUESTION)
print(f"Answer: {result.answer}\n")
