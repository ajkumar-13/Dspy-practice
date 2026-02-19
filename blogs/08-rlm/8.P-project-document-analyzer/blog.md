# 8.P: Project: Document Analyzer

## Introduction

You have learned what Recursive Language Models are and how to use `dspy.RLM` for large-context exploration through a sandboxed REPL. Now let's build something real: a **Document Analyzer** that extracts themes, arguments, and counterarguments from long documents, using structured Pydantic output models and a chunked processing pipeline. We will compare RLM against CoT to see exactly where REPL-based exploration earns its keep.

---

## Project Overview

Here is the full pipeline:

1. **Define an analysis schema** using Pydantic models for structured output
2. **Build analysis modules**: one with RLM, one with CoT for comparison
3. **Chunk and process documents**: handle documents longer than context windows
4. **Synthesize insights** across chunks into a unified analysis
5. **Compare RLM vs. CoT** quality on the same documents
6. **Evaluate with custom metrics** to quantify analysis quality

By the end, you will have a reusable document analysis pipeline that leverages RLM's programmatic exploration where it matters most.

---

## Prerequisites

- Completed [8.1: Understanding RLM](../8.1-understanding-rlm/blog.md) and [8.2: Building with RLM](../8.2-building-with-rlm/blog.md)
- `pydantic` installed (bundled with DSPy)
- Deno installed for the sandboxed interpreter ([installation guide](https://docs.deno.com/runtime/getting_started/installation/))

---

## Step 1: Define Analysis Schema

First, we define structured output models for our analysis results. This ensures every analysis, whether from RLM or CoT, produces consistent, machine-readable output.

```python
import dspy
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# Structured output models
class Argument(BaseModel):
    """A single argument or claim found in the document."""
    claim: str = Field(description="The core claim being made")
    evidence: str = Field(description="Evidence or reasoning supporting the claim")
    strength: str = Field(description="How strong the argument is: strong, moderate, or weak")


class Theme(BaseModel):
    """A major theme identified in the document."""
    name: str = Field(description="Short name for the theme")
    description: str = Field(description="What this theme covers")
    relevance: str = Field(description="Why this theme matters in the document's context")


class ChunkAnalysis(BaseModel):
    """Analysis of a single document chunk."""
    themes: list[Theme] = Field(description="Major themes in this chunk")
    arguments: list[Argument] = Field(description="Key arguments made in this chunk")
    counterarguments: list[Argument] = Field(description="Counterarguments or opposing views")
    key_quotes: list[str] = Field(description="Important direct quotes from the chunk")
    sentiment: str = Field(description="Overall sentiment: positive, negative, neutral, or mixed")


class DocumentSynthesis(BaseModel):
    """Synthesized analysis across all chunks of a document."""
    main_themes: list[Theme] = Field(description="Overarching themes across the full document")
    strongest_arguments: list[Argument] = Field(description="The most compelling arguments")
    key_tensions: list[str] = Field(description="Tensions or contradictions in the document")
    overall_assessment: str = Field(description="High-level assessment of the document")
    confidence: str = Field(description="Confidence in the analysis: high, medium, or low")
```

These Pydantic models give us type safety and validation. Every analysis result will conform to this schema, making downstream processing straightforward.

---

## Step 2: Build Analysis Modules

We build two versions of the analyzer, one using `dspy.RLM` and one using `dspy.ChainOfThought`, so we can compare them head-to-head.

```python
# Signatures
class AnalyzeChunk(dspy.Signature):
    """Analyze a document chunk, extracting themes, arguments, and counterarguments."""
    chunk: str = dspy.InputField(desc="A section of the document to analyze")
    document_title: str = dspy.InputField(desc="Title of the source document for context")
    analysis: ChunkAnalysis = dspy.OutputField(desc="Structured analysis of this chunk")


class SynthesizeAnalyses(dspy.Signature):
    """Synthesize multiple chunk analyses into a unified document analysis."""
    chunk_analyses: str = dspy.InputField(desc="JSON array of chunk analysis results")
    document_title: str = dspy.InputField(desc="Title of the source document")
    synthesis: DocumentSynthesis = dspy.OutputField(desc="Unified analysis across all chunks")


# RLM-based analyzer (explores each chunk via REPL)
class RLMDocumentAnalyzer(dspy.Module):
    """Document analyzer using RLM for deep REPL-based exploration."""

    def __init__(self):
        self.analyze_chunk = dspy.RLM(AnalyzeChunk, max_iterations=15)
        self.synthesize = dspy.RLM(SynthesizeAnalyses, max_iterations=10)

    def forward(self, chunks: list[str], document_title: str) -> DocumentSynthesis:
        # Analyze each chunk via REPL exploration
        chunk_results = []
        for chunk in chunks:
            result = self.analyze_chunk(chunk=chunk, document_title=document_title)
            chunk_results.append(result.analysis)

        # Synthesize across chunks
        import json
        analyses_json = json.dumps(
            [a.model_dump() for a in chunk_results], indent=2
        )
        synthesis = self.synthesize(
            chunk_analyses=analyses_json,
            document_title=document_title,
        )
        return synthesis.synthesis


# CoT-based analyzer (for comparison)
class CoTDocumentAnalyzer(dspy.Module):
    """Document analyzer using ChainOfThought for comparison."""

    def __init__(self):
        self.analyze_chunk = dspy.ChainOfThought(AnalyzeChunk)
        self.synthesize = dspy.ChainOfThought(SynthesizeAnalyses)

    def forward(self, chunks: list[str], document_title: str) -> DocumentSynthesis:
        chunk_results = []
        for chunk in chunks:
            result = self.analyze_chunk(chunk=chunk, document_title=document_title)
            chunk_results.append(result.analysis)

        import json
        analyses_json = json.dumps(
            [a.model_dump() for a in chunk_results], indent=2
        )
        synthesis = self.synthesize(
            chunk_analyses=analyses_json,
            document_title=document_title,
        )
        return synthesis.synthesis
```

The two analyzers are structurally identical. The only difference is which DSPy module processes the signatures. This isolates the effect of REPL-based exploration (RLM) vs. prompted reasoning (CoT).

---

## Step 3: Process Documents

Real documents are often longer than what we want to send in a single LM call. We chunk them and analyze each chunk independently before synthesizing.

```python
# Document chunking
def chunk_document(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    """Split a document into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


# Sample document
SAMPLE_DOCUMENT = """
The Rise and Fall of Remote Work: A Critical Analysis

The COVID-19 pandemic forced the largest work-from-home experiment in history.
By April 2020, over 70% of knowledge workers were fully remote. Companies that
had resisted remote work for decades made the transition in weeks. And for many,
it worked surprisingly well.

Productivity studies during 2020-2021 showed mixed but generally positive results.
Stanford economist Nick Bloom found that remote workers were 13% more productive
in a controlled experiment. Microsoft's analysis of its own workforce showed that
individual productivity remained stable, though cross-team collaboration declined.

The arguments for remote work are compelling. Workers save an average of 72 minutes
per day in commute time. Companies can recruit from a global talent pool. Office
real estate costs drop dramatically. Employee satisfaction surveys consistently
show that 80%+ of workers prefer at least some remote work.

However, the counterarguments are equally substantial. Studies from MIT and Harvard
have shown that spontaneous innovation, the kind that happens at coffee machines
and in hallway conversations, declined by 25% in fully remote organizations.
New employee onboarding takes 30% longer. Junior employees report feeling isolated
and missing out on mentorship opportunities.

The hybrid model has emerged as the dominant compromise, but it brings its own
challenges. Coordination costs increase when teams split between office and home.
"Proximity bias" means that in-office workers get promoted faster. And maintaining
company culture becomes harder when half the team is a grid of video thumbnails.

The debate is far from settled. Companies like Shopify and Spotify have gone
fully remote. Others like Goldman Sachs and JPMorgan have mandated full return
to office. Most fall somewhere in between, still experimenting with policies
that try to capture the benefits of both worlds.

What's clear is that the pre-pandemic default, five days in the office with no
exceptions, is gone for good. The question is no longer whether remote work is
viable, but how to make it work well. And that requires honest engagement with
both the data and the human factors that no study can fully capture.
"""

# Chunk and preview
chunks = chunk_document(SAMPLE_DOCUMENT, chunk_size=800, overlap=100)
print(f"Document split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i+1}: {len(chunk)} chars | {chunk[:60].strip()}...")
```

---

## Step 4: Synthesize Insights

Now we run our two analyzers on the same document and collect their output.

```python
# Run both analyzers
lm = dspy.LM("openai/gpt-4o")
standard_lm = dspy.LM("openai/gpt-4o-mini")

title = "The Rise and Fall of Remote Work: A Critical Analysis"

# RLM analysis (REPL-based exploration of each chunk)
print("\n=== RLM Analysis ===")
with dspy.context(lm=lm):
    rlm_analyzer = RLMDocumentAnalyzer()
    rlm_result = rlm_analyzer(chunks=chunks, document_title=title)

print(f"Main themes: {len(rlm_result.main_themes)}")
for theme in rlm_result.main_themes:
    print(f"  - {theme.name}: {theme.description}")

print(f"\nStrongest arguments: {len(rlm_result.strongest_arguments)}")
for arg in rlm_result.strongest_arguments:
    print(f"  - [{arg.strength}] {arg.claim}")

print(f"\nKey tensions:")
for tension in rlm_result.key_tensions:
    print(f"  - {tension}")

print(f"\nOverall: {rlm_result.overall_assessment}")
print(f"Confidence: {rlm_result.confidence}")

# CoT analysis
print("\n=== CoT Analysis ===")
with dspy.context(lm=standard_lm):
    cot_analyzer = CoTDocumentAnalyzer()
    cot_result = cot_analyzer(chunks=chunks, document_title=title)

print(f"Main themes: {len(cot_result.main_themes)}")
for theme in cot_result.main_themes:
    print(f"  - {theme.name}: {theme.description}")

print(f"\nStrongest arguments: {len(cot_result.strongest_arguments)}")
for arg in cot_result.strongest_arguments:
    print(f"  - [{arg.strength}] {arg.claim}")

print(f"\nOverall: {cot_result.overall_assessment}")
print(f"Confidence: {cot_result.confidence}")
```

---

## Step 5: Compare Approaches

Let's build a structured comparison to see where RLM excels.

```python
# Quality comparison
def compare_analyses(rlm_result: DocumentSynthesis, cot_result: DocumentSynthesis):
    """Print a side-by-side comparison of two analysis results."""
    print("\n" + "=" * 70)
    print("COMPARISON: RLM vs CoT Document Analysis")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'RLM':>15} {'CoT':>15}")
    print("-" * 65)
    print(f"{'Themes identified':<35} {len(rlm_result.main_themes):>15} {len(cot_result.main_themes):>15}")
    print(f"{'Arguments found':<35} {len(rlm_result.strongest_arguments):>15} {len(cot_result.strongest_arguments):>15}")
    print(f"{'Tensions identified':<35} {len(rlm_result.key_tensions):>15} {len(cot_result.key_tensions):>15}")
    print(f"{'Confidence':<35} {rlm_result.confidence:>15} {cot_result.confidence:>15}")

    # Check for nuance in assessments
    rlm_words = len(rlm_result.overall_assessment.split())
    cot_words = len(cot_result.overall_assessment.split())
    print(f"{'Assessment length (words)':<35} {rlm_words:>15} {cot_words:>15}")

    print(f"\nRLM Assessment:\n  {rlm_result.overall_assessment}")
    print(f"\nCoT Assessment:\n  {cot_result.overall_assessment}")


compare_analyses(rlm_result, cot_result)
```

Typical findings:

| Dimension | RLM | CoT |
|-----------|-----|-----|
| Themes | More nuanced, identifies subtler themes | Captures obvious themes |
| Arguments | Evaluates evidence quality | Lists arguments without evaluation |
| Counterarguments | Identifies implicit counterarguments | Only catches explicit ones |
| Tensions | Finds contradictions between sections | Misses cross-section tensions |
| Assessment | Balanced, acknowledges complexity | Tends toward simpler conclusions |

The difference emerges most clearly on longer documents where RLM's ability to programmatically search and cross-reference different sections gives it an advantage over CoT's single-pass approach.

---

## Step 6: Evaluate Quality

Finally, let's build custom metrics to quantify analysis quality programmatically.

```python
# Evaluation metrics
class AnalysisQualityJudge(dspy.Signature):
    """Judge the quality of a document analysis against the source text."""
    document: str = dspy.InputField(desc="The original document")
    analysis_json: str = dspy.InputField(desc="The analysis result as JSON")
    completeness: float = dspy.OutputField(desc="Score 0-1: does the analysis cover all major points?")
    accuracy: float = dspy.OutputField(desc="Score 0-1: are the claims factually grounded?")
    nuance: float = dspy.OutputField(desc="Score 0-1: does the analysis capture subtlety?")
    reasoning: str = dspy.OutputField(desc="Brief explanation of the scores")


def evaluate_analysis(document: str, analysis: DocumentSynthesis, label: str):
    """Score an analysis using an LLM judge."""
    import json

    # Use a strong model as judge (not the same model that produced the analysis)
    judge_lm = dspy.LM("openai/gpt-4o")

    with dspy.context(lm=judge_lm):
        judge = dspy.ChainOfThought(AnalysisQualityJudge)
        result = judge(
            document=document,
            analysis_json=json.dumps(analysis.model_dump(), indent=2),
        )

    print(f"\n--- {label} Quality Scores ---")
    print(f"  Completeness: {result.completeness}")
    print(f"  Accuracy:     {result.accuracy}")
    print(f"  Nuance:       {result.nuance}")
    print(f"  Reasoning:    {result.reasoning}")
    return {
        "completeness": float(result.completeness),
        "accuracy": float(result.accuracy),
        "nuance": float(result.nuance),
    }


# Score both analyses
rlm_scores = evaluate_analysis(SAMPLE_DOCUMENT, rlm_result, "RLM")
cot_scores = evaluate_analysis(SAMPLE_DOCUMENT, cot_result, "CoT")

# Summary
print("\n" + "=" * 50)
print("QUALITY SUMMARY")
print("=" * 50)
for metric in ["completeness", "accuracy", "nuance"]:
    rlm_val = rlm_scores[metric]
    cot_val = cot_scores[metric]
    winner = "RLM" if rlm_val > cot_val else "CoT" if cot_val > rlm_val else "Tie"
    print(f"  {metric:<20} RLM: {rlm_val:.2f}  CoT: {cot_val:.2f}  > {winner}")
```

The LLM-as-judge approach lets us evaluate subjective quality dimensions like "nuance" that are impossible to capture with exact-match metrics. Using GPT-4o as the judge (a different model than either analyzer) reduces bias.

---

## What We Learned

1. **Pydantic models + DSPy = structured analysis.** Defining your output schema with Pydantic ensures consistent, machine-readable results regardless of which module you use.

2. **Chunk, Analyze, Synthesize** is a robust pattern for long documents. Analyze chunks independently, then let a second LM call synthesize across chunks.

3. **RLM excels at exploration.** On analytical tasks with large inputs, RLM's ability to programmatically search and cross-reference different sections produces more thorough analysis than CoT's single-pass approach.

4. **CoT is often "good enough."** For straightforward documents with clear arguments, `ChainOfThought` with a strong standard model produces solid results at lower cost and latency.

5. **LLM-as-judge evaluation** fills the gap when exact-match metrics do not apply. It is not perfect, but it provides actionable signal for comparing approaches.

6. **The right tool for the right job.** Use `Predict` for extraction, `ChainOfThought` for moderate analysis, and `RLM` for tasks that need programmatic exploration of large contexts. Your pipeline can (and should) mix all three.

---

## Next Up

You have completed Phase 8! You now know how to leverage Recursive Language Models in DSPy, from understanding the theory to building a production-quality document analysis pipeline. In Phase 9, we will explore **RL Optimization**, using reinforcement learning to optimize DSPy programs for complex tasks.

[9.1: RL for DSPy](../../09-rl-optimization/9.1-rl-for-dspy/blog.md)

---

## Resources

- [DSPy RLM API Docs](https://dspy.ai/api/modules/RLM/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [DSPy Module Documentation](https://dspy.ai/learn/programming/modules/)
- [Blog code samples](code/)
