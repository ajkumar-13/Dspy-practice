# 6.5 Papillon: Privacy-Conscious Delegation


## Introduction

Not all data should leave your network. Medical records, financial details, personal identifiers, proprietary code: some information is too sensitive to send to a cloud LLM. But cloud models are often more capable than local ones. How do you get the best of both worlds?

**Papillon** is a privacy-conscious delegation pattern for multi-model architectures. The core idea: **route sensitive data to local/private models** and **delegate non-sensitive reasoning to powerful cloud models**. DSPy makes this practical with its `dspy.context(lm=...)` mechanism, which lets you switch models at any point in your program, even mid-pipeline.

---

## What You'll Learn

- The Papillon pattern: privacy-conscious delegation between LMs
- Why multi-model architectures matter for production AI
- Using `dspy.context(lm=...)` for selective model switching
- Building a pipeline that routes tasks based on data sensitivity
- Privacy-preserving patterns for real-world applications

---

## Prerequisites

- Completed [6.4 Memory and Conversation History](../6.4-memory-agents/blog.md)
- DSPy installed (`uv add dspy python-dotenv`)
- Access to both a cloud LM (e.g., OpenAI) and a local LM (e.g., Ollama)

---

## The Privacy Problem

Consider a medical assistant that helps doctors write patient summaries. The workflow has two parts:

1. **Extract key facts** from patient records (names, diagnoses, medications). This involves PII and PHI.
2. **Generate a well-written summary** from those facts. This is a general writing task.

Sending patient records to GPT-4o violates HIPAA. Running everything on a local 7B model produces mediocre summaries. Papillon solves this by splitting the work:

- A **local model** handles step 1 (sensitive data never leaves your network).
- A **cloud model** handles step 2 (it only sees de-identified facts, not raw records).

---

## Model Switching with `dspy.context`

DSPy's `dspy.context(lm=...)` lets you temporarily switch which language model a block of code uses, without reconfiguring the global default:

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

# Cloud model — powerful but data leaves your network
cloud_lm = dspy.LM("openai/gpt-4o-mini")

# Local model — less capable but data stays on-premises
local_lm = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434")

# Set the default to the cloud model
dspy.configure(lm=cloud_lm)


# This module runs on the default (cloud) LM
summarizer = dspy.ChainOfThought("facts -> summary")

# This module runs on the local LM via context override
extractor = dspy.ChainOfThought("document -> extracted_facts")

# Use the local model for sensitive extraction
with dspy.context(lm=local_lm):
    facts = extractor(document="Patient John Doe, DOB 1985-03-15, diagnosed with Type 2 diabetes...")
    print(f"Extracted (locally): {facts.extracted_facts}")

# Use the cloud model for the writing task (only de-identified facts)
summary = summarizer(facts=facts.extracted_facts)
print(f"Summary (cloud): {summary.summary}")
```

The key insight: **the sensitive document never leaves your network.** Only the extracted facts, which can be de-identified, are sent to the cloud model.

---

## Building a Papillon Pipeline

Here's a more complete example with explicit sensitivity routing:

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure models
cloud_lm = dspy.LM("openai/gpt-4o-mini")
local_lm = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434")
dspy.configure(lm=cloud_lm)


class SensitivityClassifier(dspy.Signature):
    """Classify whether a text contains sensitive personal information."""
    text: str = dspy.InputField()
    contains_pii: bool = dspy.OutputField(desc="True if text contains PII/PHI/sensitive data")
    sensitivity_reason: str = dspy.OutputField(desc="Brief explanation of what sensitive data was found")


class PrivateExtractor(dspy.Signature):
    """Extract key facts from a document, removing all personally identifiable information."""
    document: str = dspy.InputField()
    deidentified_facts: str = dspy.OutputField(desc="Key facts with all PII removed or replaced with placeholders")


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
            # Step 2b: No sensitive data — use directly
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
    document="Patient Jane Smith (SSN: 123-45-6789) presented with chronic migraine. "
             "Prescribed sumatriptan 50mg. Follow-up scheduled for March 2026.",
    task_description="Write a clinical summary for the referring physician.",
)

print(f"PII Detected: {result.contained_pii}")
print(f"Reason: {result.sensitivity_reason}")
print(f"Summary: {result.summary}")
```

---

## When to Use Papillon

| Scenario | Route to Local | Route to Cloud |
|---|---|---|
| PII/PHI extraction | ✅ | ❌ |
| Code with trade secrets | ✅ | ❌ |
| General summarization | ❌ | ✅ |
| Creative writing | ❌ | ✅ |
| Financial record parsing | ✅ | ❌ |
| Translation of public content | ❌ | ✅ |

The Papillon pattern isn't limited to two models. You can route to **any number of models** based on sensitivity level, cost constraints, latency requirements, or capability needs. `dspy.context(lm=...)` makes the switching trivial.

---

## Key Takeaways

- **Papillon routes sensitive data to local models** and delegates non-sensitive work to powerful cloud models, achieving privacy without sacrificing capability.
- **`dspy.context(lm=local_lm)`** temporarily switches the active model for any block of code, including inside `dspy.Module.forward()`.
- **Sensitivity classification** can be automated as a first step in the pipeline.
- **De-identification** on the local model ensures that only safe, stripped-down facts reach cloud APIs.
- **The pattern generalizes** beyond privacy. Use it for cost optimization (cheap model for easy tasks), latency optimization (fast model for real-time, slow model for background), or capability routing.

---

## Next Up

You've learned every agent pattern in DSPy: ReAct, advanced tools, MCP, memory, and privacy-conscious delegation. Now it's time to combine them into a real project: a **financial analyst agent** that uses multiple tools, handles errors, and gets optimized for accuracy.

**[6.P Project: Financial Analyst Agent →](../6.P-project-financial-analyst/blog.md)**

---

## Resources

- 📖 [Papillon Tutorial](https://dspy.ai/tutorials/papillon/)
- 📖 [DSPy Context Management](https://dspy.ai/learn/programming/language_models/)
- 📖 [Local Models with Ollama](https://dspy.ai/learn/programming/language_models/#local-models)
- 💻 [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
