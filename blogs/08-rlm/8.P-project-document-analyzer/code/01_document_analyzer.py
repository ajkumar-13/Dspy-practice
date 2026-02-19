"""
Blog 8.P: Project - Document Analyzer
Run: python 01_document_analyzer.py

Builds a complete document analysis pipeline that:
1. Defines structured output schemas with Pydantic
2. Chunks large documents for processing
3. Analyzes chunks using both RLM and CoT
4. Synthesizes insights across chunks
5. Compares RLM vs CoT quality using LLM-as-judge

Prerequisites:
- Deno installed (https://docs.deno.com/runtime/getting_started/installation/)
- OpenAI API key in .env
"""

import json
import dspy
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# =====================================================
# Step 1: Structured Output Models
# =====================================================

class Argument(BaseModel):
    """A single argument or claim found in the document."""
    claim: str = Field(description="The core claim being made")
    evidence: str = Field(description="Evidence or reasoning supporting the claim")
    strength: str = Field(description="How strong: strong, moderate, or weak")


class Theme(BaseModel):
    """A major theme identified in the document."""
    name: str = Field(description="Short name for the theme")
    description: str = Field(description="What this theme covers")
    relevance: str = Field(description="Why this theme matters")


class ChunkAnalysis(BaseModel):
    """Analysis of a single document chunk."""
    themes: list[Theme] = Field(description="Major themes in this chunk")
    arguments: list[Argument] = Field(description="Key arguments made")
    counterarguments: list[Argument] = Field(description="Counterarguments or opposing views")
    key_quotes: list[str] = Field(description="Important direct quotes")
    sentiment: str = Field(description="Overall sentiment: positive, negative, neutral, or mixed")


class DocumentSynthesis(BaseModel):
    """Synthesized analysis across all chunks of a document."""
    main_themes: list[Theme] = Field(description="Overarching themes")
    strongest_arguments: list[Argument] = Field(description="Most compelling arguments")
    key_tensions: list[str] = Field(description="Tensions or contradictions")
    overall_assessment: str = Field(description="High-level assessment")
    confidence: str = Field(description="Confidence: high, medium, or low")


# =====================================================
# Step 2: DSPy Signatures and Modules
# =====================================================

class AnalyzeChunk(dspy.Signature):
    """Analyze a document chunk, extracting themes, arguments, and counterarguments."""
    chunk: str = dspy.InputField(desc="A section of the document to analyze")
    document_title: str = dspy.InputField(desc="Title of the source document")
    analysis: ChunkAnalysis = dspy.OutputField(desc="Structured analysis of this chunk")


class SynthesizeAnalyses(dspy.Signature):
    """Synthesize multiple chunk analyses into a unified document analysis."""
    chunk_analyses: str = dspy.InputField(desc="JSON array of chunk analysis results")
    document_title: str = dspy.InputField(desc="Title of the source document")
    synthesis: DocumentSynthesis = dspy.OutputField(desc="Unified analysis")


class RLMDocumentAnalyzer(dspy.Module):
    """Document analyzer using RLM for deep REPL-based exploration."""

    def __init__(self):
        self.analyze_chunk = dspy.RLM(AnalyzeChunk, max_iterations=15)
        self.synthesize = dspy.RLM(SynthesizeAnalyses, max_iterations=10)

    def forward(self, chunks: list[str], document_title: str) -> DocumentSynthesis:
        chunk_results = []
        for chunk in chunks:
            result = self.analyze_chunk(chunk=chunk, document_title=document_title)
            chunk_results.append(result.analysis)

        analyses_json = json.dumps(
            [a.model_dump() for a in chunk_results], indent=2
        )
        synthesis = self.synthesize(
            chunk_analyses=analyses_json,
            document_title=document_title,
        )
        return synthesis.synthesis


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

        analyses_json = json.dumps(
            [a.model_dump() for a in chunk_results], indent=2
        )
        synthesis = self.synthesize(
            chunk_analyses=analyses_json,
            document_title=document_title,
        )
        return synthesis.synthesis


# =====================================================
# Step 3: Document Chunking
# =====================================================

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


# =====================================================
# Step 4: Run Analysis Pipeline
# =====================================================

# Configure LM
lm = dspy.LM("openai/gpt-4o")
standard_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

title = "The Rise and Fall of Remote Work: A Critical Analysis"
chunks = chunk_document(SAMPLE_DOCUMENT, chunk_size=800, overlap=100)
print(f"Document split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i + 1}: {len(chunk)} chars | {chunk[:60].strip()}...")

# RLM analysis
print("\n=== RLM Analysis ===")
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


# =====================================================
# Step 5: Compare Approaches
# =====================================================

def compare_analyses(rlm_res: DocumentSynthesis, cot_res: DocumentSynthesis):
    """Print a side-by-side comparison of two analysis results."""
    print("\n" + "=" * 70)
    print("COMPARISON: RLM vs CoT Document Analysis")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'RLM':>15} {'CoT':>15}")
    print("-" * 65)
    print(f"{'Themes identified':<35} {len(rlm_res.main_themes):>15} {len(cot_res.main_themes):>15}")
    print(f"{'Arguments found':<35} {len(rlm_res.strongest_arguments):>15} {len(cot_res.strongest_arguments):>15}")
    print(f"{'Tensions identified':<35} {len(rlm_res.key_tensions):>15} {len(cot_res.key_tensions):>15}")
    print(f"{'Confidence':<35} {rlm_res.confidence:>15} {cot_res.confidence:>15}")

    rlm_words = len(rlm_res.overall_assessment.split())
    cot_words = len(cot_res.overall_assessment.split())
    print(f"{'Assessment length (words)':<35} {rlm_words:>15} {cot_words:>15}")

    print(f"\nRLM Assessment:\n  {rlm_res.overall_assessment}")
    print(f"\nCoT Assessment:\n  {cot_res.overall_assessment}")


compare_analyses(rlm_result, cot_result)


# =====================================================
# Step 6: LLM-as-Judge Evaluation
# =====================================================

class AnalysisQualityJudge(dspy.Signature):
    """Judge the quality of a document analysis against the source text."""
    document: str = dspy.InputField(desc="The original document")
    analysis_json: str = dspy.InputField(desc="The analysis result as JSON")
    completeness: float = dspy.OutputField(desc="Score 0-1: covers all major points?")
    accuracy: float = dspy.OutputField(desc="Score 0-1: claims factually grounded?")
    nuance: float = dspy.OutputField(desc="Score 0-1: captures subtlety?")
    reasoning: str = dspy.OutputField(desc="Brief explanation of the scores")


def evaluate_analysis(document: str, analysis: DocumentSynthesis, label: str):
    """Score an analysis using an LLM judge."""
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


rlm_scores = evaluate_analysis(SAMPLE_DOCUMENT, rlm_result, "RLM")
cot_scores = evaluate_analysis(SAMPLE_DOCUMENT, cot_result, "CoT")

print("\n" + "=" * 50)
print("QUALITY SUMMARY")
print("=" * 50)
for metric in ["completeness", "accuracy", "nuance"]:
    rlm_val = rlm_scores[metric]
    cot_val = cot_scores[metric]
    winner = "RLM" if rlm_val > cot_val else "CoT" if cot_val > rlm_val else "Tie"
    print(f"  {metric:<20} RLM: {rlm_val:.2f}  CoT: {cot_val:.2f}  > {winner}")
