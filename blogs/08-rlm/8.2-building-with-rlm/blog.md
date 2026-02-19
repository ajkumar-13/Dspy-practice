# 8.2: Building with RLM

## Introduction

In the previous blog, you learned *what* Recursive Language Models are and how `dspy.RLM` processes large contexts through an iterative REPL loop. Now it is time to put them to work. In this blog, we will build practical programs that leverage RLM's programmatic exploration for tasks where standard prompting struggles: large document analysis, complex data processing, and multi-source reasoning. We will also measure the cost-performance tradeoff so you can make informed decisions about when RLM earns its keep.

---

## What You'll Learn

- Practical patterns for using `dspy.RLM` in real programs
- How to combine RLM with retrieval for large-context analysis
- How to use custom tools to extend RLM's capabilities
- How to compare RLM vs. Predict vs. CoT on large input tasks
- Cost-performance tradeoffs with iterative REPL exploration
- Tips for getting the most out of RLM

---

## Prerequisites

- Completed [8.1: Understanding RLM](../8.1-understanding-rlm/blog.md)
- Deno installed for the sandboxed interpreter ([installation guide](https://docs.deno.com/runtime/getting_started/installation/))

---

## Pattern 1: Large Document Analysis

RLM excels when processing documents too large or complex for a single prompt. Instead of truncating the input, RLM lets the LLM programmatically search, filter, and aggregate the data it needs.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)


class FinancialAnalysis(dspy.Signature):
    """Analyze a financial report to answer a specific question."""
    report: str = dspy.InputField(desc="Full financial report text")
    question: str = dspy.InputField(desc="Analysis question")
    analysis: str = dspy.OutputField(desc="Detailed analysis with evidence")
    final_answer: str = dspy.OutputField(desc="Concise answer")


class FinancialAnalyzer(dspy.Module):
    def __init__(self):
        # RLM explores the report programmatically
        self.analyze = dspy.RLM(
            FinancialAnalysis,
            max_iterations=15,
        )

    def forward(self, report: str, question: str):
        return self.analyze(report=report, question=question)


analyzer = FinancialAnalyzer()
result = analyzer(
    report=open("annual_report.txt").read(),  # Large financial document
    question="What were the key revenue drivers, and how did margins change?",
)
print(f"Analysis: {result.analysis}")
print(f"Answer: {result.final_answer}")
```

Under the hood, the LLM might:

1. Print the first 2000 characters to understand the document structure
2. Search for revenue-related sections using `re.findall()`
3. Extract margin data with targeted string searches
4. Use `llm_query()` to semantically analyze the extracted snippets
5. Call `SUBMIT()` with the combined findings

With `dspy.Predict`, the same document would be truncated or overwhelm the model's attention. RLM handles this gracefully because it processes the data programmatically.

---

## Pattern 2: Multi-Source Reasoning

When you need to reason across multiple sources, RLM can systematically compare and cross-reference information.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)


class CrossReference(dspy.Signature):
    """Compare multiple sources to answer a question."""
    sources: str = dspy.InputField(desc="Multiple data sources, separated by markers")
    question: str = dspy.InputField(desc="Question requiring cross-source reasoning")
    findings: str = dspy.OutputField(desc="Findings from each source")
    conclusion: str = dspy.OutputField(desc="Synthesized conclusion")


class MultiSourceAnalyzer(dspy.Module):
    def __init__(self):
        self.compare = dspy.RLM(
            CrossReference,
            max_iterations=20,
        )

    def forward(self, sources: str, question: str):
        return self.compare(sources=sources, question=question)


analyzer = MultiSourceAnalyzer()

sources = """
=== SOURCE A: Market Research Report ===
Global semiconductor revenue reached $527B in 2024, up 19% YoY...
AI chip demand grew 40% in 2024, driven by data center buildouts...
Memory chip prices stabilized after a 2-year downturn, with DRAM up 12%...

=== SOURCE B: Company Earnings Call ===
Our AI accelerator segment grew 55% and now represents 30% of revenue...
We expect supply constraints to ease in Q2 2025...
Data center orders backlog stands at $8.2B...

=== SOURCE C: Industry Analyst Note ===
Top 3 semiconductor companies now control 65% of advanced node capacity...
We project the AI chip market will reach $200B by 2027...
Geopolitical risks remain elevated due to trade restrictions...
"""

result = analyzer(
    sources=sources,
    question="What's the AI chip market outlook, and what risks should investors watch?",
)
print(f"Findings: {result.findings}")
print(f"Conclusion: {result.conclusion}")
```

---

## Pattern 3: Combining RLM with Retrieval

RLM pairs naturally with retrieval. Use cheap, fast retrieval to gather relevant context, then use RLM to deeply analyze the retrieved information.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()


class AnalyzeWithContext(dspy.Signature):
    """Analyze retrieved evidence to answer a complex question."""
    context: str = dspy.InputField(desc="Retrieved evidence passages")
    question: str = dspy.InputField(desc="The question to analyze")
    analysis: str = dspy.OutputField(desc="Detailed analysis based on evidence")
    conclusion: str = dspy.OutputField(desc="Final conclusion with confidence level")


class ReasoningRAG(dspy.Module):
    """RAG pipeline that uses RLM for the analysis step."""

    def __init__(self):
        # Simple retrieval: use Predict (fast, cheap)
        self.generate_queries = dspy.Predict("question -> search_queries: list[str]")
        # Deep analysis: use RLM (thorough exploration)
        self.analyze = dspy.RLM(
            AnalyzeWithContext,
            max_iterations=10,
        )

    def forward(self, question: str, documents: str):
        return self.analyze(context=documents, question=question)


lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

rag = ReasoningRAG()

docs = """
Document 1: Global semiconductor revenue reached $527B in 2024, up 19% YoY.
Document 2: AI chip demand grew 40% in 2024, driven by data center buildouts.
Document 3: TSMC's 3nm process now accounts for 15% of revenue, up from 6%.
Document 4: Intel's foundry services reported $200M in external revenue in Q4 2024.
Document 5: Memory chip prices stabilized after a 2-year downturn, with DRAM up 12%.
"""

result = rag(
    question="What are the key trends shaping the semiconductor industry?",
    documents=docs,
)
print(f"Analysis: {result.analysis}")
print(f"Conclusion: {result.conclusion}")
```

The key pattern here is **selective use of RLM**: use cheap, fast `Predict` for query generation, and reserve the more expensive RLM for the step that actually needs deep exploration.

---

## Pattern 4: Custom Tools

RLM supports custom tools that the LLM can call during its REPL exploration. This extends RLM beyond text analysis to interact with external systems.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)


# Define custom tools the LLM can use in the REPL
def word_count(text: str) -> int:
    """Count the number of words in a text."""
    return len(text.split())

def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from text."""
    import re
    return [float(x) for x in re.findall(r"[\d,]+\.?\d*", text.replace(",", ""))]


rlm = dspy.RLM(
    "document, question -> answer",
    tools=[word_count, extract_numbers],
    max_iterations=10,
)

result = rlm(
    document="The revenue was $12.5M in Q1 and $14.8M in Q2...",
    question="What was the total revenue across all quarters?",
)
print(result.answer)
```

---

## Comparing Approaches: Predict vs. CoT vs. RLM

The key decision is **context size**. For small inputs, all three modules produce similar results. The differences emerge with large, complex inputs.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

# Simulate a large document with scattered data
large_doc = "Financial data section...\n" * 100  # Imagine 100K+ chars

# Predict: sends everything in one prompt (may miss details)
predict = dspy.Predict("document, question -> answer")
p_result = predict(document=large_doc, question="What was Q3 revenue?")

# ChainOfThought: reasons over full context (still may miss details)
cot = dspy.ChainOfThought("document, question -> answer")
c_result = cot(document=large_doc, question="What was Q3 revenue?")

# RLM: programmatically searches for the answer (more reliable)
rlm = dspy.RLM("document, question -> answer", max_iterations=10)
r_result = rlm(document=large_doc, question="What was Q3 revenue?")
```

### Typical Results on Large Contexts

| Approach | Large Context Accuracy | Relative Cost | Latency |
|----------|----------------------|---------------|---------|
| `dspy.Predict` | Moderate (attention diluted) | Low | Fast |
| `dspy.ChainOfThought` | Better (structured reasoning) | Medium | Moderate |
| `dspy.RLM` | Best (targeted exploration) | Higher | Slower |

### When Each Module Wins

- **Predict wins** on simple tasks with small context (fastest, cheapest)
- **ChainOfThought wins** on reasoning tasks that fit in the context window
- **RLM wins** on large-context tasks where targeted exploration matters

---

## Cost-Performance Tradeoffs

RLM's cost comes from its **iterative REPL loop**: each iteration makes an LLM call, and the LLM may also call `llm_query()` for sub-LLM queries. Here is how to think about the tradeoff:

### Cost Scaling

```
Cost per RLM call ~= (iterations * main_lm_cost) + (sub_queries * sub_lm_cost)
```

A typical RLM call might use 5-15 iterations, each with a moderate-sized prompt (context metadata + REPL history). This is more expensive than a single `Predict` call, but often cheaper than passing a 100K-character document to a standard model because:

1. Each iteration processes only a small amount of context
2. Using `sub_lm` delegates extraction to a cheaper model
3. The main LM only handles strategy and the final synthesis

### Cost Guidelines

| Task Type | Recommended Module | Why |
|-----------|-------------------|-----|
| Simple extraction | `dspy.Predict` | No exploration needed |
| Classification | `dspy.Predict` or `dspy.ChainOfThought` | Small context |
| Summarization (short) | `dspy.ChainOfThought` | Full context fits |
| Large document Q&A | `dspy.RLM` | Too large for single prompt |
| Data exploration | `dspy.RLM` | Needs programmatic access |
| Multi-source analysis | `dspy.RLM` | Systematic cross-referencing |

### Reducing Costs with Sub-LM

```python
# Main LM handles strategy, cheap sub-LM handles extraction
main_lm = dspy.LM("openai/gpt-4o")
cheap_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=main_lm)

rlm = dspy.RLM(
    "data, query -> summary",
    sub_lm=cheap_lm,          # Cheaper model for llm_query() calls
    max_iterations=10,         # Limit iterations to control cost
    max_llm_calls=30,          # Cap total LLM calls
)
```

---

## Tips for Getting the Most Out of RLM

1. **Be specific in your signatures.** RLM works best when the LLM knows exactly what it is looking for. Use descriptive field names and `desc` parameters.

2. **Set `verbose=True` during development.** This shows you exactly what code the LLM is writing and executing, helping you debug and understand the exploration strategy.

3. **Use RLM selectively in pipelines.** Not every step needs REPL exploration. Use `Predict` for simple steps and `RLM` only where large context handling matters.

4. **Tune `max_iterations` and `max_llm_calls`.** Start with lower values (5-10 iterations) and increase if the LLM consistently runs out of steps. Higher values cost more but allow deeper exploration.

5. **Use `sub_lm` for cost efficiency.** The main LM decides strategy while a cheaper sub-LM handles `llm_query()` extraction calls.

6. **Inspect the trajectory.** After each call, check `result.trajectory` to understand what the LLM did. This helps you optimize signatures and iteration limits.

---

## Key Takeaways

1. **Large document analysis** is the primary use case for RLM. Problems where context rot would degrade a standard prompt benefit most from RLM's programmatic exploration.

2. **Combine modules strategically**: use `Predict` for simple steps, `ChainOfThought` for moderate reasoning, and `RLM` for steps that need to explore large or complex inputs.

3. **RLM + Retrieval** is a powerful pattern: retrieve cheaply, explore deeply.

4. **Custom tools** extend RLM beyond text analysis, allowing the REPL to interact with databases, APIs, and computation.

5. **Always benchmark**: run the same evaluation with Predict, CoT, and RLM to see if the quality improvement justifies the cost for your specific task.

---

## Next Up

Time to put everything together. In the Phase 8 project, you will build a **Document Analyzer** that uses RLM for deep analysis of large documents, combining structured outputs, multi-section processing, and quality evaluation.

[8.P: Project: Document Analyzer](../8.P-project-document-analyzer/blog.md)

---

## Resources

- [DSPy RLM API Docs](https://dspy.ai/api/modules/RLM/)
- [DSPy Module Documentation](https://dspy.ai/learn/programming/modules/)
- [Deno Installation Guide](https://docs.deno.com/runtime/getting_started/installation/)
- [Blog code samples](code/)
