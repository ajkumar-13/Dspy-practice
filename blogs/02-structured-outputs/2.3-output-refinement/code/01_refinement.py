"""
Blog 2.3: Output Refinement Strategies
Best-of-N sampling and iterative refinement.
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# --- Best-of-N example ---
# Generate N completions and pick the best one using a reward function


class CreativeWriter(dspy.Signature):
    """Write a creative one-paragraph story based on the prompt."""

    prompt: str = dspy.InputField()
    story: str = dspy.OutputField(desc="A creative, engaging one-paragraph story")


writer = dspy.Predict(CreativeWriter)
result = writer(prompt="A robot discovers it can dream")
print(f"Story: {result.story}")
