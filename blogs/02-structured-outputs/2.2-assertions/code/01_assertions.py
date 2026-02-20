"""
Blog 2.2: DSPy Assertions
Using Assert and Suggest for programmatic constraints.
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


class FactualQA(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer, source")

    def forward(self, question: str):
        result = self.generate(question=question)

        # Hard constraint: answer must not be empty
        dspy.Assert(  # type: ignore[attr-defined]
            len(result.answer) > 10,
            "Answer must be at least 10 characters long and substantive.",
        )

        # Soft constraint: suggest a source be provided
        dspy.Suggest(  # type: ignore[attr-defined]
            len(result.source) > 0,
            "Please provide a source or reference for your answer.",
        )

        return result


if __name__ == "__main__":
    qa = FactualQA()
    result = qa(question="What is the significance of the Turing Test?")
    print(f"Answer: {result.answer}")
    print(f"Source: {result.source}")
