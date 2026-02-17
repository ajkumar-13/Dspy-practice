"""
Blog 2.1: Typed Predictors & Pydantic Integration
Using Pydantic models for structured LM outputs.
"""

import dspy
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


# --- Pydantic model as output ---
class ExtractedEntity(BaseModel):
    name: str = Field(description="The entity name")
    entity_type: Literal["person", "organization", "location", "date", "amount"]
    confidence: float = Field(description="Confidence score 0.0-1.0", ge=0.0, le=1.0)


class EntityExtraction(dspy.Signature):
    """Extract all named entities from the given text."""

    text: str = dspy.InputField(desc="Text to extract entities from")
    entities: list[ExtractedEntity] = dspy.OutputField(desc="List of extracted entities")


extractor = dspy.Predict(EntityExtraction)
result = extractor(
    text="Apple Inc. reported $94.8 billion in revenue on January 25, 2025. CEO Tim Cook announced the results from Cupertino, California."
)

for entity in result.entities:
    print(f"  {entity.name} ({entity.entity_type}), confidence: {entity.confidence}")
