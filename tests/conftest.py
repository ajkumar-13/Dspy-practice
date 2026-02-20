"""
conftest.py - Shared fixtures for DSPy blog tests.

These tests validate code patterns WITHOUT calling any LLM API.
"""

import pytest
import dspy


@pytest.fixture
def qa_example():
    """A basic question-answer example."""
    return dspy.Example(
        question="What is the capital of France?",
        answer="Paris",
    ).with_inputs("question")


@pytest.fixture
def qa_examples():
    """A set of question-answer examples for dataset tests."""
    return [
        dspy.Example(
            question="What is the capital of France?",
            answer="Paris",
        ).with_inputs("question"),
        dspy.Example(
            question="What is 2 + 2?",
            answer="4",
        ).with_inputs("question"),
        dspy.Example(
            question="Who wrote Romeo and Juliet?",
            answer="William Shakespeare",
        ).with_inputs("question"),
    ]


@pytest.fixture
def correct_prediction():
    """A prediction that contains the correct answer."""
    return dspy.Prediction(answer="The capital of France is Paris.")


@pytest.fixture
def incorrect_prediction():
    """A prediction with a wrong answer."""
    return dspy.Prediction(answer="The capital of France is Berlin.")
