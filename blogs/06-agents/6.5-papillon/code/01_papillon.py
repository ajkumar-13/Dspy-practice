"""
Blog 6.5 - Papillon: Privacy-Conscious Delegation
Run: python 01_papillon.py

Requires a local model via Ollama (or swap local_lm for another provider).
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# Cloud model (powerful but data leaves your network)
cloud_lm = dspy.LM("openai/gpt-4o-mini")

# Local model (less capable but data stays on-premises)
# Change this to your local model endpoint
local_lm = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434")

# Set the default to the cloud model
dspy.configure(lm=cloud_lm)


# ============================================================
# Part 1: Basic Model Switching
# ============================================================

print("=" * 60)
print("Part 1: Basic Model Switching")
print("=" * 60 + "\n")

summarizer = dspy.ChainOfThought("facts -> summary")
extractor = dspy.ChainOfThought("document -> extracted_facts")

# Use the local model for sensitive extraction
with dspy.context(lm=local_lm):
    facts = extractor(
        document="Patient John Doe, DOB 1985-03-15, diagnosed with Type 2 diabetes..."
    )
    print(f"Extracted (locally): {facts.extracted_facts}")

# Use the cloud model for the writing task (only de-identified facts)
summary = summarizer(facts=facts.extracted_facts)
print(f"Summary (cloud): {summary.summary}")


# ============================================================
# Part 2: Full Papillon Pipeline
# ============================================================

print("\n" + "=" * 60)
print("Part 2: Full Papillon Pipeline")
print("=" * 60 + "\n")


class SensitivityClassifier(dspy.Signature):
    """Classify whether a text contains sensitive personal information."""
    text: str = dspy.InputField()
    contains_pii: bool = dspy.OutputField(
        desc="True if text contains PII/PHI/sensitive data"
    )
    sensitivity_reason: str = dspy.OutputField(
        desc="Brief explanation of what sensitive data was found"
    )


class PrivateExtractor(dspy.Signature):
    """Extract key facts from a document, removing all personally identifiable information."""
    document: str = dspy.InputField()
    deidentified_facts: str = dspy.OutputField(
        desc="Key facts with all PII removed or replaced with placeholders"
    )


class PublicSummarizer(dspy.Signature):
    """Write a clear, professional summary from the provided facts."""
    facts: str = dspy.InputField()
    task_description: str = dspy.InputField()
    summary: str = dspy.OutputField()


class PapillonPipeline(dspy.Module):
    """Privacy-conscious pipeline that routes sensitive data to local models."""

    def __init__(self):
        self.classifier = dspy.Predict(SensitivityClassifier)
        self.extractor = dspy.ChainOfThought(PrivateExtractor)
        self.summarizer = dspy.ChainOfThought(PublicSummarizer)

    def forward(self, document: str, task_description: str) -> dspy.Prediction:
        # Step 1: Classify sensitivity (can run on either model)
        classification = self.classifier(text=document)

        if classification.contains_pii:
            # Step 2a: Extract and de-identify LOCALLY
            with dspy.context(lm=local_lm):
                extraction = self.extractor(document=document)
            facts = extraction.deidentified_facts
        else:
            # Step 2b: No sensitive data, use directly
            facts = document

        # Step 3: Summarize with the powerful cloud model
        result = self.summarizer(facts=facts, task_description=task_description)

        return dspy.Prediction(
            summary=result.summary,
            contained_pii=classification.contains_pii,
            sensitivity_reason=classification.sensitivity_reason,
        )


# Use the pipeline
pipeline = PapillonPipeline()

result = pipeline(
    document=(
        "Patient Jane Smith (SSN: 123-45-6789) presented with chronic migraine. "
        "Prescribed sumatriptan 50mg. Follow-up scheduled for March 2026."
    ),
    task_description="Write a clinical summary for the referring physician.",
)

print(f"PII Detected: {result.contained_pii}")
print(f"Reason: {result.sensitivity_reason}")
print(f"Summary: {result.summary}")
