"""
Blog 1.2: Inline Signatures
Demonstrate different inline signature patterns.
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# --- Basic signature ---
qa = dspy.Predict("question -> answer")
result = qa(question="What is the capital of France?")
print(f"Basic: {result.answer}")

# --- Typed output ---
scorer = dspy.Predict("review -> rating: float")
result = scorer(review="This product is amazing! Best purchase ever.")
print(f"Rating: {result.rating}")

# --- Multi-input ---
translator = dspy.Predict("text, target_language -> translation")
result = translator(text="Hello, world!", target_language="Spanish")
print(f"Translation: {result.translation}")

# --- Multi-output ---
analyzer = dspy.Predict("text -> sentiment, confidence: float, keywords: list[str]")
result = analyzer(text="DSPy is a revolutionary framework for building LLM applications.")
print(f"Sentiment: {result.sentiment}")
print(f"Confidence: {result.confidence}")
print(f"Keywords: {result.keywords}")
