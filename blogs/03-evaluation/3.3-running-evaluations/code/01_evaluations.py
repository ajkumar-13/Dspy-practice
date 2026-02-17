"""
Blog 3.3: Running Evaluations
Systematically benchmark DSPy programs.
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# --- Setup ---
qa = dspy.Predict("question -> answer")

devset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
    dspy.Example(question="Who painted the Mona Lisa?", answer="Leonardo da Vinci").with_inputs("question"),
]


def simple_metric(example, prediction, trace=None):
    return float(example.answer.lower() in prediction.answer.lower())


# --- Run evaluation ---
evaluator = dspy.Evaluate(
    devset=devset,
    metric=simple_metric,
    num_threads=2,
    display_progress=True,
    display_table=True,
)

score = evaluator(qa)
print(f"\nOverall Score: {score}")
