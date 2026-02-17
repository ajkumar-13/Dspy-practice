"""
Blog 1.4: Building Custom Modules
Composing DSPy modules into multi-step programs.
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


# --- Simple Custom Module ---
class SimpleQA(dspy.Module):
    def __init__(self):
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str):
        return self.generate_answer(question=question)


# --- Multi-Step Custom Module ---
class ResearchSummarizer(dspy.Module):
    """Takes a topic, generates key questions, then answers them."""

    def __init__(self):
        self.generate_questions = dspy.Predict("topic -> questions: list[str]")
        self.answer_question = dspy.ChainOfThought("question, context -> answer")
        self.summarize = dspy.Predict("topic, qa_pairs -> summary")

    def forward(self, topic: str):
        # Step 1: Generate research questions
        questions_result = self.generate_questions(topic=topic)
        questions = questions_result.questions[:3]  # Limit to 3

        # Step 2: Answer each question
        qa_pairs = []
        for q in questions:
            answer = self.answer_question(question=q, context=topic)
            qa_pairs.append(f"Q: {q}\nA: {answer.answer}")

        # Step 3: Summarize all Q&A pairs
        qa_text = "\n\n".join(qa_pairs)
        summary = self.summarize(topic=topic, qa_pairs=qa_text)

        return dspy.Prediction(
            questions=questions,
            qa_pairs=qa_pairs,
            summary=summary.summary,
        )


# --- Test it ---
if __name__ == "__main__":
    researcher = ResearchSummarizer()
    result = researcher(topic="The impact of DSPy on LLM application development")

    print("Generated Questions:")
    for i, q in enumerate(result.questions, 1):
        print(f"  {i}. {q}")

    print("\nQ&A Pairs:")
    for pair in result.qa_pairs:
        print(f"  {pair}\n")

    print(f"Summary:\n  {result.summary}")
