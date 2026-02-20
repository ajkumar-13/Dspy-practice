"""
Tests for Pydantic models used with DSPy typed predictors (Blog 2.1).

Validates Pydantic field constraints, serialization, and rejection
without calling any LLM.
"""

from typing import Literal

import dspy
import pytest
from pydantic import BaseModel, Field, ValidationError

# ── Model definitions (from blog 2.1) ────────────────────────────────


class ExtractedEntity(BaseModel):
    name: str = Field(description="The entity name")
    entity_type: Literal["person", "organization", "location", "date", "amount"]
    confidence: float = Field(description="Confidence score 0.0-1.0", ge=0.0, le=1.0)


class EntityExtraction(dspy.Signature):
    """Extract all named entities from the given text."""

    text: str = dspy.InputField(desc="Text to extract entities from")
    entities: list[ExtractedEntity] = dspy.OutputField(desc="List of extracted entities")


# ── Tests ─────────────────────────────────────────────────────────────


class TestExtractedEntity:
    def test_valid_person(self):
        entity = ExtractedEntity(name="Tim Cook", entity_type="person", confidence=0.95)
        assert entity.name == "Tim Cook"
        assert entity.entity_type == "person"
        assert entity.confidence == 0.95

    def test_valid_organization(self):
        entity = ExtractedEntity(name="Apple Inc.", entity_type="organization", confidence=0.99)
        assert entity.entity_type == "organization"

    def test_valid_location(self):
        entity = ExtractedEntity(name="Cupertino", entity_type="location", confidence=0.85)
        assert entity.entity_type == "location"

    def test_valid_date(self):
        entity = ExtractedEntity(name="January 25, 2025", entity_type="date", confidence=0.92)
        assert entity.entity_type == "date"

    def test_valid_amount(self):
        entity = ExtractedEntity(name="$94.8 billion", entity_type="amount", confidence=0.88)
        assert entity.entity_type == "amount"

    def test_confidence_at_boundaries(self):
        low = ExtractedEntity(name="X", entity_type="person", confidence=0.0)
        high = ExtractedEntity(name="X", entity_type="person", confidence=1.0)
        assert low.confidence == 0.0
        assert high.confidence == 1.0

    def test_reject_confidence_above_1(self):
        with pytest.raises(ValidationError):
            ExtractedEntity(name="X", entity_type="person", confidence=1.5)

    def test_reject_confidence_below_0(self):
        with pytest.raises(ValidationError):
            ExtractedEntity(name="X", entity_type="person", confidence=-0.1)

    def test_reject_invalid_entity_type(self):
        with pytest.raises(ValidationError):
            ExtractedEntity(name="X", entity_type="event", confidence=0.5)  # type: ignore[arg-type]

    def test_serialization_roundtrip(self):
        entity = ExtractedEntity(name="Apple Inc.", entity_type="organization", confidence=0.99)
        data = entity.model_dump()
        restored = ExtractedEntity(**data)
        assert restored == entity

    def test_json_roundtrip(self):
        entity = ExtractedEntity(name="Paris", entity_type="location", confidence=0.9)
        json_str = entity.model_dump_json()
        restored = ExtractedEntity.model_validate_json(json_str)
        assert restored == entity


class TestEntityExtractionSignature:
    def test_has_text_input(self):
        assert "text" in EntityExtraction.input_fields

    def test_has_entities_output(self):
        assert "entities" in EntityExtraction.output_fields

    def test_can_instantiate_predict(self):
        predictor = dspy.Predict(EntityExtraction)
        assert predictor is not None
