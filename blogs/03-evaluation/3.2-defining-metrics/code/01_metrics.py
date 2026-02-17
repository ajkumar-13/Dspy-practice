"""
Blog 3.2: Defining Metrics
Built-in and custom metrics for DSPy programs.
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


# --- Custom metric function ---
def answer_quality_metric(example, prediction, trace=None):
    """Custom metric that checks answer correctness and quality."""
    # Check if the predicted answer matches the expected answer
    correct = example.answer.lower() in prediction.answer.lower()

    if trace is not None:
        # During optimization, return True/False for the optimizer
        return correct

    # During evaluation, return a score
    return float(correct)


# --- LLM-as-Judge metric ---
class JudgeAnswer(dspy.Signature):
    """Judge if the predicted answer is semantically equivalent to the gold answer."""

    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField(desc="The correct answer")
    predicted_answer: str = dspy.InputField(desc="The model's answer")
    is_correct: bool = dspy.OutputField(desc="True if semantically equivalent")


def llm_judge_metric(example, prediction, trace=None):
    judge = dspy.Predict(JudgeAnswer)
    result = judge(
        question=example.question,
        gold_answer=example.answer,
        predicted_answer=prediction.answer,
    )
    return float(result.is_correct)


# --- Test metrics ---
if __name__ == "__main__":
    example = dspy.Example(question="Capital of France?", answer="Paris").with_inputs("question")
    prediction = dspy.Prediction(answer="The capital of France is Paris.")

    score = answer_quality_metric(example, prediction)
    print(f"Custom metric score: {score}")

    judge_score = llm_judge_metric(example, prediction)
    print(f"LLM Judge score: {judge_score}")
