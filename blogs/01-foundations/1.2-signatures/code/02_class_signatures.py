"""
Blog 1.2: Class-Based Signatures
Demonstrate class-based signatures with fields and descriptions.
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


# --- Class-based signature with descriptions ---
class SentimentAnalysis(dspy.Signature):
    """Analyze the sentiment of a product review."""

    review: str = dspy.InputField(desc="A product review from a customer")
    sentiment: str = dspy.OutputField(desc="One of: positive, negative, neutral")
    confidence: float = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")
    reasoning: str = dspy.OutputField(desc="Brief explanation for the sentiment classification")


predictor = dspy.Predict(SentimentAnalysis)
result = predictor(review="The battery life is terrible but the screen quality is outstanding.")
print(f"Sentiment: {result.sentiment}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")


# --- Multi-input class signature ---
class Translation(dspy.Signature):
    """Translate text from one language to another while preserving tone and meaning."""

    source_text: str = dspy.InputField(desc="The text to translate")
    source_language: str = dspy.InputField(desc="The source language")
    target_language: str = dspy.InputField(desc="The target language")
    translation: str = dspy.OutputField(desc="The translated text")
    notes: str = dspy.OutputField(desc="Any translation notes about idioms or cultural context")


translator = dspy.Predict(Translation)
result = translator(
    source_text="It's raining cats and dogs",
    source_language="English",
    target_language="French",
)
print(f"\nTranslation: {result.translation}")
print(f"Notes: {result.notes}")
