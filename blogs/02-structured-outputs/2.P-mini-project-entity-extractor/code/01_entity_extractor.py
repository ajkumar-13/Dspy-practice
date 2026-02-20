"""
Phase 2 Mini-Project: Structured Entity Extractor
Extract structured data from legal documents with validation.
"""

from typing import Literal

import dspy
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


class LegalEntity(BaseModel):
    name: str = Field(description="Full name of the entity")
    role: str = Field(
        description="Role in the document: plaintiff, defendant, witness, judge, attorney"
    )


class MonetaryAmount(BaseModel):
    amount: float = Field(description="The monetary amount")
    currency: str = Field(default="USD", description="Currency code")
    context: str = Field(description="What this amount refers to")


class ExtractedLegalInfo(BaseModel):
    entities: list[LegalEntity] = Field(description="All named entities with their roles")
    dates: list[str] = Field(description="All dates mentioned in ISO format")
    amounts: list[MonetaryAmount] = Field(description="All monetary amounts mentioned")
    case_type: Literal["civil", "criminal", "contract", "tort", "other"]
    summary: str = Field(description="Brief summary of the document")


class LegalDocExtractor(dspy.Module):
    def __init__(self):
        self.extract = dspy.ChainOfThought("document: str -> extracted_info: ExtractedLegalInfo")

    def forward(self, document: str):
        result = self.extract(document=document)

        dspy.Assert(  # type: ignore[attr-defined]
            len(result.extracted_info.entities) > 0,
            "Must extract at least one entity from the document.",
        )

        return result


if __name__ == "__main__":
    extractor = LegalDocExtractor()
    sample_doc = """
    On March 15, 2025, John Smith (plaintiff) filed a civil lawsuit against
    Acme Corporation (defendant) in the Superior Court of California.
    The plaintiff seeks damages of $2,500,000 for breach of contract.
    Attorney Sarah Johnson represents the plaintiff. The case was assigned
    to Judge Michael Chen. An initial hearing is scheduled for April 20, 2025.
    """
    result = extractor(document=sample_doc)
    print(result)
