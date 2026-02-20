"""
Tests for custom metric functions (Blog 3.2).

Validates pure-Python metric logic without calling any LLM.
"""

import dspy

# ── Metric definition (from blog 3.2) ────────────────────────────────


def answer_quality_metric(example, prediction, trace=None):
    """Custom metric that checks answer correctness and quality."""
    correct = example.answer.lower() in prediction.answer.lower()

    if trace is not None:
        return correct

    return float(correct)


# ── Tests ─────────────────────────────────────────────────────────────


class TestAnswerQualityMetric:
    def test_correct_answer_returns_1(self, qa_example, correct_prediction):
        score = answer_quality_metric(qa_example, correct_prediction)
        assert score == 1.0

    def test_incorrect_answer_returns_0(self, qa_example, incorrect_prediction):
        score = answer_quality_metric(qa_example, incorrect_prediction)
        assert score == 0.0

    def test_case_insensitive(self):
        ex = dspy.Example(question="Q", answer="PARIS").with_inputs("question")
        pred = dspy.Prediction(answer="paris is the answer")
        assert answer_quality_metric(ex, pred) == 1.0

    def test_partial_match(self):
        ex = dspy.Example(question="Q", answer="paris").with_inputs("question")
        pred = dspy.Prediction(answer="The capital of France is Paris, a beautiful city.")
        assert answer_quality_metric(ex, pred) == 1.0

    def test_no_match(self):
        ex = dspy.Example(question="Q", answer="paris").with_inputs("question")
        pred = dspy.Prediction(answer="London is the capital of England.")
        assert answer_quality_metric(ex, pred) == 0.0

    def test_trace_returns_bool_true(self, qa_example, correct_prediction):
        result = answer_quality_metric(qa_example, correct_prediction, trace="optimization")
        assert result is True

    def test_trace_returns_bool_false(self, qa_example, incorrect_prediction):
        result = answer_quality_metric(qa_example, incorrect_prediction, trace="optimization")
        assert result is False

    def test_no_trace_returns_float(self, qa_example, correct_prediction):
        result = answer_quality_metric(qa_example, correct_prediction, trace=None)
        assert isinstance(result, float)
