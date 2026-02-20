"""
Tests for DSPy Signature definitions (Blog 1.2).

Validates that class-based signatures have the correct fields,
types, and descriptions without calling any LLM.
"""

import dspy

# ── Signature definitions (from blog 1.2) ────────────────────────────


class SentimentAnalysis(dspy.Signature):
    """Analyze the sentiment of a product review."""

    review: str = dspy.InputField(desc="A product review from a customer")
    sentiment: str = dspy.OutputField(desc="One of: positive, negative, neutral")
    confidence: float = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")
    reasoning: str = dspy.OutputField(desc="Brief explanation for the sentiment classification")


class Translation(dspy.Signature):
    """Translate text from one language to another while preserving tone and meaning."""

    source_text: str = dspy.InputField(desc="The text to translate")
    source_language: str = dspy.InputField(desc="The source language")
    target_language: str = dspy.InputField(desc="The target language")
    translation: str = dspy.OutputField(desc="The translated text")
    notes: str = dspy.OutputField(desc="Any translation notes about idioms or cultural context")


# ── Tests ─────────────────────────────────────────────────────────────


class TestSentimentAnalysisSignature:
    def test_has_correct_input_fields(self):
        fields = SentimentAnalysis.input_fields
        assert "review" in fields

    def test_has_correct_output_fields(self):
        fields = SentimentAnalysis.output_fields
        assert "sentiment" in fields
        assert "confidence" in fields
        assert "reasoning" in fields

    def test_input_field_count(self):
        assert len(SentimentAnalysis.input_fields) == 1

    def test_output_field_count(self):
        assert len(SentimentAnalysis.output_fields) == 3

    def test_docstring(self):
        assert SentimentAnalysis.__doc__ is not None
        assert "sentiment" in SentimentAnalysis.__doc__.lower()

    def test_can_instantiate_predict(self):
        predictor = dspy.Predict(SentimentAnalysis)
        assert predictor is not None


class TestTranslationSignature:
    def test_has_correct_input_fields(self):
        fields = Translation.input_fields
        assert "source_text" in fields
        assert "source_language" in fields
        assert "target_language" in fields

    def test_has_correct_output_fields(self):
        fields = Translation.output_fields
        assert "translation" in fields
        assert "notes" in fields

    def test_input_field_count(self):
        assert len(Translation.input_fields) == 3

    def test_output_field_count(self):
        assert len(Translation.output_fields) == 2

    def test_docstring(self):
        assert Translation.__doc__ is not None
        assert "translate" in Translation.__doc__.lower()

    def test_can_instantiate_predict(self):
        predictor = dspy.Predict(Translation)
        assert predictor is not None
