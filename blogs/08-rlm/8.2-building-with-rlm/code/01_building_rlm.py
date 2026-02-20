"""
Blog 8.2: Building with RLM
Run: python 01_building_rlm.py

Demonstrates practical patterns for using dspy.RLM:
1. Large document analysis with iterative REPL exploration
2. Multi-source cross-referencing
3. RLM combined with retrieval (RAG)
4. Custom tools in the REPL
5. Comparing Predict vs. CoT vs. RLM on large inputs

Prerequisites:
- Deno installed (https://docs.deno.com/runtime/getting_started/installation/)
- OpenAI API key in .env
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)


# =====================================================
# Pattern 1: Large Document Analysis
# =====================================================


class FinancialAnalysis(dspy.Signature):
    """Analyze a financial report to answer a specific question."""

    report: str = dspy.InputField(desc="Full financial report text")
    question: str = dspy.InputField(desc="Analysis question")
    analysis: str = dspy.OutputField(desc="Detailed analysis with evidence")
    final_answer: str = dspy.OutputField(desc="Concise answer")


class FinancialAnalyzer(dspy.Module):
    def __init__(self):
        self.analyze = dspy.RLM(
            FinancialAnalysis,
            max_iterations=15,
        )

    def forward(self, report: str, question: str):
        return self.analyze(report=report, question=question)


# Simulated large financial report
report_text = """
ACME Corp Annual Report FY2024

Executive Summary:
ACME Corp delivered strong results in FY2024, with total revenue reaching $59.6M,
up 15% year-over-year. Key growth drivers included cloud services expansion in Q1,
enterprise contract wins in Q2, AI product launches in Q3, and international
expansion in Q4.

Q1 Results:
- Revenue: $12.5M (up 8% YoY)
- Operating expenses: $9.2M
- Net income: $3.3M
- Gross margin: 72.1%

Q2 Results:
- Revenue: $14.1M (up 12% YoY)
- Operating expenses: $10.1M
- Net income: $4.0M
- Gross margin: 73.5%

Q3 Results:
- Revenue: $15.8M (up 18% YoY)
- Operating expenses: $10.9M
- Net income: $4.9M
- Gross margin: 74.2%

Q4 Results:
- Revenue: $17.2M (up 22% YoY)
- Operating expenses: $11.5M
- Net income: $5.7M
- Gross margin: 75.0%

Risk Factors:
- Increasing competition in the AI market segment
- Dependency on key enterprise clients (top 10 = 45% of revenue)
- Foreign exchange exposure due to international expansion
- Talent retention challenges in engineering (15% turnover)
"""

print("=== Pattern 1: Large Document Analysis ===")
analyzer = FinancialAnalyzer()
result = analyzer(
    report=report_text,
    question="How did gross margins trend through the year, and what drove the improvements?",
)
print(f"Analysis: {result.analysis}")
print(f"Answer: {result.final_answer}")

# Inspect what the LLM did
if hasattr(result, "trajectory") and result.trajectory:
    print(f"\nRLM used {len(result.trajectory)} exploration steps")


# =====================================================
# Pattern 2: Multi-Source Cross-Referencing
# =====================================================


class CrossReference(dspy.Signature):
    """Compare multiple sources to answer a question."""

    sources: str = dspy.InputField(desc="Multiple data sources, separated by markers")
    question: str = dspy.InputField(desc="Question requiring cross-source reasoning")
    findings: str = dspy.OutputField(desc="Key findings from each source")
    conclusion: str = dspy.OutputField(desc="Synthesized conclusion")


print("\n=== Pattern 2: Multi-Source Cross-Referencing ===")
multi_rlm = dspy.RLM(CrossReference, max_iterations=15)

sources_data = """
=== SOURCE A: Internal Sales Data ===
Q3 enterprise deals closed: 12 (up from 8 in Q2)
Average deal size: $450K (up from $380K)
Sales cycle: 45 days average
Top sectors: Finance (40%), Healthcare (30%), Tech (30%)

=== SOURCE B: Customer Satisfaction Survey ===
NPS score: 72 (up from 65)
Top complaint: onboarding complexity (mentioned by 34% of respondents)
Feature request #1: API integrations (52% of respondents)
Renewal intent: 91% likely to renew

=== SOURCE C: Competitive Intelligence ===
Main competitor launched similar product at 20% lower price
Competitor lacks enterprise features (SSO, audit trails)
Market analyst report rates our product #1 for enterprise, #3 for SMB
"""

result2 = multi_rlm(
    sources=sources_data,
    question="What's our competitive position, and what should we prioritize?",
)
print(f"Findings: {result2.findings}")
print(f"Conclusion: {result2.conclusion}")


# =====================================================
# Pattern 3: RLM + Retrieval (RAG)
# =====================================================


class AnalyzeEvidence(dspy.Signature):
    """Analyze retrieved evidence to answer a complex question."""

    context: str = dspy.InputField(desc="Retrieved evidence passages")
    question: str = dspy.InputField(desc="Question to analyze")
    analysis: str = dspy.OutputField(desc="Detailed analysis based on evidence")
    conclusion: str = dspy.OutputField(desc="Final conclusion with confidence")


class ReasoningRAG(dspy.Module):
    """RAG pipeline that uses RLM for the analysis step."""

    def __init__(self):
        self.generate_queries = dspy.Predict("question -> search_queries: list[str]")
        self.analyze = dspy.RLM(AnalyzeEvidence, max_iterations=10)

    def forward(self, question: str, documents: str):
        return self.analyze(context=documents, question=question)


print("\n=== Pattern 3: RLM + Retrieval ===")
rag = ReasoningRAG()
docs = """
Document 1: Global semiconductor revenue reached $527B in 2024, up 19% YoY.
Document 2: AI chip demand grew 40% in 2024, driven by data center buildouts.
Document 3: TSMC's 3nm process now accounts for 15% of revenue, up from 6%.
Document 4: Intel foundry services reported $200M in external revenue in Q4 2024.
Document 5: Memory chip prices stabilized after a 2-year downturn, DRAM up 12%.
"""

result3 = rag(
    question="What are the key trends shaping the semiconductor industry?",
    documents=docs,
)
print(f"Analysis: {result3.analysis}")
print(f"Conclusion: {result3.conclusion}")


# =====================================================
# Pattern 4: Custom Tools
# =====================================================


def word_count(text: str) -> int:
    """Count the number of words in a text."""
    return len(text.split())


def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from text."""
    import re

    return [float(x) for x in re.findall(r"[\d]+\.?\d*", text.replace(",", ""))]


print("\n=== Pattern 4: Custom Tools ===")
rlm_with_tools = dspy.RLM(
    "document, question -> answer",
    tools=[word_count, extract_numbers],
    max_iterations=10,
)

result4 = rlm_with_tools(
    document=report_text,
    question="What is the total net income across all quarters?",
)
print(f"Answer: {result4.answer}")


# =====================================================
# Pattern 5: Comparing Approaches
# =====================================================

print("\n=== Pattern 5: Comparing Predict vs CoT vs RLM ===")

question = "What was the Q3 net income, and how did it compare to Q1?"

# Predict
predict_result = dspy.Predict("document, question -> answer")(
    document=report_text, question=question
)
print(f"Predict: {predict_result.answer}")

# ChainOfThought
cot_result = dspy.ChainOfThought("document, question -> answer")(
    document=report_text, question=question
)
print(f"CoT: {cot_result.answer}")

# RLM
rlm_result = dspy.RLM("document, question -> answer", max_iterations=10)(
    document=report_text, question=question
)
print(f"RLM: {rlm_result.answer}")
