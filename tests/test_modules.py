"""
Tests for DSPy module instantiation (Blogs 1.3, 1.4).

Validates that modules can be constructed and have the expected
sub-predictors without calling any LLM.
"""

import dspy

# ── Module definitions (from blogs 1.3, 1.4) ─────────────────────────


class SimpleQA(dspy.Module):
    def __init__(self):
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str):
        return self.generate_answer(question=question)


class ResearchSummarizer(dspy.Module):
    """Takes a topic, generates key questions, then answers them."""

    def __init__(self):
        self.generate_questions = dspy.Predict("topic -> questions: list[str]")
        self.answer_question = dspy.ChainOfThought("question, context -> answer")
        self.summarize = dspy.Predict("topic, qa_pairs -> summary")

    def forward(self, topic: str):
        questions_result = self.generate_questions(topic=topic)
        questions = questions_result.questions[:3]

        qa_pairs = []
        for q in questions:
            answer = self.answer_question(question=q, context=topic)
            qa_pairs.append(f"Q: {q}\nA: {answer.answer}")

        qa_text = "\n\n".join(qa_pairs)
        summary = self.summarize(topic=topic, qa_pairs=qa_text)

        return dspy.Prediction(
            questions=questions,
            qa_pairs=qa_pairs,
            summary=summary.summary,
        )


# ── Tests ─────────────────────────────────────────────────────────────


class TestBuiltinModules:
    def test_predict_instantiation(self):
        module = dspy.Predict("question -> answer")
        assert module is not None

    def test_chain_of_thought_instantiation(self):
        module = dspy.ChainOfThought("question -> answer")
        assert module is not None

    def test_predict_with_class_signature(self):
        class QA(dspy.Signature):
            """Answer the question."""

            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        module = dspy.Predict(QA)
        assert module is not None


class TestSimpleQA:
    def test_instantiation(self):
        module = SimpleQA()
        assert module is not None

    def test_has_generate_answer(self):
        module = SimpleQA()
        assert hasattr(module, "generate_answer")

    def test_generate_answer_is_cot(self):
        module = SimpleQA()
        assert isinstance(module.generate_answer, dspy.ChainOfThought)


class TestResearchSummarizer:
    def test_instantiation(self):
        module = ResearchSummarizer()
        assert module is not None

    def test_has_generate_questions(self):
        module = ResearchSummarizer()
        assert hasattr(module, "generate_questions")

    def test_has_answer_question(self):
        module = ResearchSummarizer()
        assert hasattr(module, "answer_question")

    def test_has_summarize(self):
        module = ResearchSummarizer()
        assert hasattr(module, "summarize")

    def test_generate_questions_is_predict(self):
        module = ResearchSummarizer()
        assert isinstance(module.generate_questions, dspy.Predict)

    def test_answer_question_is_cot(self):
        module = ResearchSummarizer()
        assert isinstance(module.answer_question, dspy.ChainOfThought)

    def test_summarize_is_predict(self):
        module = ResearchSummarizer()
        assert isinstance(module.summarize, dspy.Predict)
