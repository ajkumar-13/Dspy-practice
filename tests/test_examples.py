"""
Tests for dspy.Example and dataset operations (Blog 3.1).

Validates example creation, input/label partitioning, and dataset
splitting without calling any LLM.
"""

import dspy


class TestExampleCreation:
    def test_create_example_with_fields(self):
        ex = dspy.Example(question="What is 2+2?", answer="4")
        assert ex.question == "What is 2+2?"
        assert ex.answer == "4"

    def test_with_inputs_returns_example(self):
        ex = dspy.Example(question="Q", answer="A").with_inputs("question")
        assert ex is not None

    def test_inputs_returns_only_input_fields(self, qa_example):
        inputs = qa_example.inputs()
        assert "question" in inputs
        assert "answer" not in inputs

    def test_labels_returns_only_label_fields(self, qa_example):
        labels = qa_example.labels()
        assert "answer" in labels
        assert "question" not in labels

    def test_field_access(self, qa_example):
        assert qa_example.question == "What is the capital of France?"
        assert qa_example.answer == "Paris"


class TestDatasetSplit:
    def test_split_sizes(self, qa_examples):
        trainset = qa_examples[:2]
        devset = qa_examples[2:]
        assert len(trainset) == 2
        assert len(devset) == 1

    def test_split_preserves_content(self, qa_examples):
        trainset = qa_examples[:2]
        assert trainset[0].question == "What is the capital of France?"
        assert trainset[1].question == "What is 2 + 2?"

    def test_all_examples_have_inputs(self, qa_examples):
        for ex in qa_examples:
            inputs = ex.inputs()
            assert "question" in inputs


class TestPrediction:
    def test_create_prediction(self):
        pred = dspy.Prediction(answer="Paris", confidence=0.95)
        assert pred.answer == "Paris"
        assert pred.confidence == 0.95

    def test_prediction_with_single_field(self):
        pred = dspy.Prediction(answer="42")
        assert pred.answer == "42"

    def test_prediction_multiple_fields(self):
        pred = dspy.Prediction(
            answer="Paris",
            reasoning="France's capital city",
            confidence=0.99,
        )
        assert pred.answer == "Paris"
        assert pred.reasoning == "France's capital city"
        assert pred.confidence == 0.99
