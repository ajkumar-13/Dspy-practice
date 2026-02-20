"""
Blog 2.2: DSPy Assertions
Using runtime validation for programmatic constraints.
Note: dspy.Assert/dspy.Suggest were removed in DSPy 3.x.
Use standard Python validation instead.
"""

import logging

import dspy
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

logger = logging.getLogger(__name__)


class FactualQA(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer, source")

    def forward(self, question: str):
        result = self.generate(question=question)

        # Hard constraint: answer must not be empty
        assert len(result.answer) > 10, (
            "Answer must be at least 10 characters long and substantive."
        )

        # Soft constraint: suggest a source be provided
        if not result.source or len(result.source) == 0:
            logger.warning("Please provide a source or reference for your answer.")

        return result


if __name__ == "__main__":
    qa = FactualQA()
    result = qa(question="What is the significance of the Turing Test?")
    print(f"Answer: {result.answer}")
    print(f"Source: {result.source}")
