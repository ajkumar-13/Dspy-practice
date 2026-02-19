# 12.1: Real-World DSPy Architectures

## Introduction

You've built modules, optimized them, deployed them, and monitored them. Now let's zoom out and study the **architecture patterns** that power real-world DSPy systems. These aren't theoretical exercises - they're battle-tested patterns drawn from production deployments, DSPy tutorials, and Stanford NLP research.

The beauty of DSPy is that every module, no matter how complex the architecture, remains **optimizable**. A five-stage pipeline with agent orchestration and retrieval can be optimized end-to-end just like a single `dspy.Predict` call. The patterns in this post teach you how to structure systems so that optimization has the maximum impact.

---

## What You'll Learn

- Five production-tested architecture patterns for DSPy systems
- Multi-stage pipelines with `dspy.Parallel` for throughput
- Agent orchestration using ReAct with specialized sub-modules
- Optimization cascades: from MIPROv2 to BootstrapFinetune
- Evaluation-driven development as a design philosophy
- Privacy-preserving architectures with the PAPILLON pattern
- Real-world examples: llms.txt generation, Yahoo Finance agents, and Mem0 memory agents
- A decision framework for choosing the right architecture

---

## Prerequisites

- Completed [Phase 11: Production DSPy](../../11-production/11.1-caching-performance/blog.md)
- Familiarity with DSPy modules, optimization, retrieval, and agents (Phases 1-11)

---

## Pattern 1: Multi-Stage Pipelines

The most common production pattern is a **multi-stage pipeline** where each stage is a separate `dspy.Module` with its own signature. Data flows through stages sequentially, with each stage refining the output of the previous one.

### The Retrieve -> Analyze -> Synthesize -> Format Pattern

```python
import dspy
from typing import List

class Retrieve(dspy.Module):
    """Stage 1: Gather relevant information."""
    def __init__(self):
        self.retriever = dspy.Retrieve(k=5)

    def forward(self, question: str) -> dspy.Prediction:
        results = self.retriever(question)
        return dspy.Prediction(passages=results.passages)

class Analyze(dspy.Module):
    """Stage 2: Extract key insights from retrieved passages."""
    def __init__(self):
        self.extract = dspy.ChainOfThought(
            "passages: list[str], question: str -> key_findings: list[str], confidence: float"
        )

    def forward(self, question: str, passages: list[str]) -> dspy.Prediction:
        return self.extract(question=question, passages=passages)

class Synthesize(dspy.Module):
    """Stage 3: Combine findings into a coherent answer."""
    def __init__(self):
        self.synthesize = dspy.ChainOfThought(
            "question: str, key_findings: list[str] -> answer: str, citations: list[str]"
        )

    def forward(self, question: str, key_findings: list[str]) -> dspy.Prediction:
        return self.synthesize(question=question, key_findings=key_findings)

class Format(dspy.Module):
    """Stage 4: Format the answer for the target audience."""
    def __init__(self):
        self.format = dspy.Predict(
            "answer: str, citations: list[str], audience: str -> formatted_response: str"
        )

    def forward(self, answer: str, citations: list[str], audience: str = "general") -> dspy.Prediction:
        return self.format(answer=answer, citations=citations, audience=audience)

class ResearchPipeline(dspy.Module):
    """Full pipeline: Retrieve -> Analyze -> Synthesize -> Format."""
    def __init__(self):
        self.retrieve = Retrieve()
        self.analyze = Analyze()
        self.synthesize = Synthesize()
        self.format = Format()

    def forward(self, question: str) -> dspy.Prediction:
        retrieved = self.retrieve(question=question)
        analyzed = self.analyze(question=question, passages=retrieved.passages)
        synthesized = self.synthesize(question=question, key_findings=analyzed.key_findings)
        formatted = self.format(answer=synthesized.answer, citations=synthesized.citations)
        return dspy.Prediction(
            response=formatted.formatted_response,
            citations=synthesized.citations
        )
```

### Parallelizing Independent Stages

When stages don't depend on each other, use `dspy.Parallel` to run them concurrently. This cuts latency dramatically in fan-out architectures:

```python
class MultiSourceResearch(dspy.Module):
    """Research from multiple sources in parallel, then synthesize."""
    def __init__(self):
        self.web_search = dspy.ChainOfThought("question -> web_findings: str")
        self.doc_search = dspy.ChainOfThought("question -> doc_findings: str")
        self.db_search = dspy.ChainOfThought("question -> db_findings: str")
        self.synthesize = dspy.ChainOfThought(
            "question, web_findings, doc_findings, db_findings -> answer: str"
        )

    def forward(self, question: str):
        # Run all three searches in parallel using dspy.Parallel
        # Parallel.forward() expects exec_pairs: list[tuple[module, dspy.Example]]
        parallel = dspy.Parallel(num_threads=3)
        exec_pairs = [
            (self.web_search, dspy.Example(question=question)),
            (self.doc_search, dspy.Example(question=question)),
            (self.db_search, dspy.Example(question=question)),
        ]
        results = parallel(exec_pairs)

        # Synthesize parallel results
        return self.synthesize(
            question=question,
            web_findings=results[0].web_findings,
            doc_findings=results[1].doc_findings,
            db_findings=results[2].db_findings,
        )
```

**When to use this pattern:** Whenever your problem naturally decomposes into sequential processing stages or when multiple independent data sources need to be queried and merged.

---

## Pattern 2: Agent Orchestration

For open-ended tasks where the number of steps isn't known in advance, an **agent orchestration** pattern uses `dspy.ReAct` as a coordination layer with specialized sub-modules handling different capabilities.

```python
import dspy

# --- Specialized sub-modules as tools ---

class FinancialAnalyzer(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought(
            "financial_data: str -> analysis: str, risk_level: str, recommendation: str"
        )

    def forward(self, financial_data: str) -> dspy.Prediction:
        return self.analyze(financial_data=financial_data)

class SentimentScorer(dspy.Module):
    def __init__(self):
        self.score = dspy.Predict(
            "text: str -> sentiment: float, reasoning: str"
        )

    def forward(self, text: str) -> dspy.Prediction:
        return self.score(text=text)

# --- Tool functions wrapping sub-modules ---

analyzer = FinancialAnalyzer()
scorer = SentimentScorer()

def analyze_financials(data: str) -> str:
    """Analyze financial data and return risk assessment and recommendations."""
    result = analyzer(financial_data=data)
    return f"Analysis: {result.analysis}\nRisk: {result.risk_level}\nRecommendation: {result.recommendation}"

def score_sentiment(text: str) -> str:
    """Score the sentiment of text on a scale from -1.0 (negative) to 1.0 (positive)."""
    result = scorer(text=text)
    return f"Sentiment: {result.sentiment}, Reasoning: {result.reasoning}"

def fetch_stock_price(ticker: str) -> str:
    """Fetch the current stock price for a given ticker symbol."""
    # In production, call a real API (Yahoo Finance, Alpha Vantage, etc.)
    import random
    price = round(random.uniform(50, 500), 2)
    return f"{ticker}: ${price}"

# --- ReAct agent as orchestration layer ---

agent = dspy.ReAct(
    "question -> investment_advice: str",
    tools=[analyze_financials, score_sentiment, fetch_stock_price],
    max_iters=8,
)

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

result = agent(question="Should I invest in AAPL given recent earnings reports?")
print(result.investment_advice)
```

The key insight is that **each tool can itself be an optimizable DSPy module**. When you optimize the outer ReAct agent, the sub-modules benefit too, because their prompts participate in the optimization trace.

---

## Pattern 3: Optimization Cascade

Production systems rarely use a single optimizer. Instead, they follow an **optimization cascade** - starting cheap and escalating to more powerful (and expensive) optimizers as the system matures.

### The Four-Stage Cascade

```
Stage 1: MIPROv2          -> Optimize instructions (cheap, fast)
Stage 2: BootstrapFewShot -> Add few-shot demonstrations
Stage 3: BetterTogether   -> Joint prompt + weight optimization
Stage 4: BootstrapFinetune -> Distill to a smaller, fine-tuned model
```

```python
import dspy
from dspy.evaluate import Evaluate

# --- Stage 1: Instruction optimization with MIPROv2 ---
optimizer_v1 = dspy.MIPROv2(
    metric=my_metric,
    auto="light",   # Fast exploration
)
optimized_v1 = optimizer_v1.compile(
    my_program,
    trainset=train_data,
)

# Evaluate
evaluator = Evaluate(devset=dev_data, metric=my_metric, num_threads=8)
score_v1 = evaluator(optimized_v1)
print(f"MIPROv2 score: {score_v1}")

# --- Stage 2: Add few-shot demos with BootstrapFewShot ---
optimizer_v2 = dspy.BootstrapFewShot(
    metric=my_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)
optimized_v2 = optimizer_v2.compile(
    optimized_v1,       # Build on top of MIPROv2 result
    trainset=train_data,
)

score_v2 = evaluator(optimized_v2)
print(f"+ BootstrapFewShot score: {score_v2}")

# --- Stage 3: Joint optimization with BetterTogether ---
# BetterTogether takes prompt_optimizer and weight_optimizer (Teleprompter instances)
optimizer_v3 = dspy.BetterTogether(
    metric=my_metric,
    prompt_optimizer=dspy.MIPROv2(metric=my_metric, auto="light"),
    weight_optimizer=dspy.BootstrapFinetune(metric=my_metric),
)
optimized_v3 = optimizer_v3.compile(
    my_program,
    trainset=train_data,
)

score_v3 = evaluator(optimized_v3)
print(f"+ BetterTogether score: {score_v3}")

# --- Stage 4: Distill to fine-tuned model ---
optimizer_v4 = dspy.BootstrapFinetune(
    metric=my_metric,
)
optimized_v4 = optimizer_v4.compile(
    optimized_v3,
    trainset=train_data,
)

score_v4 = evaluator(optimized_v4)
print(f"+ BootstrapFinetune score: {score_v4}")
```

Each stage builds on the previous one's gains. You stop when the score plateaus or the cost exceeds the value. In practice, many teams find that MIPROv2 + BootstrapFewShot alone gets them 80-90% of the way there.

---

## Pattern 4: Evaluation-Driven Development

This is less a code pattern and more a **development philosophy**: define metrics first, build the program second, and let evaluations guide every decision.

### The EDD Workflow

```
1. Define evaluation metric  ->  "What does success look like?"
2. Build evaluation dataset  ->  "What are 50-100 representative examples?"
3. Build baseline program    ->  "Simplest thing that could work"
4. Evaluate baseline         ->  "Where do we stand?"
5. Optimize                  ->  "Can the optimizer do better?"
6. Iterate on program design ->  "Do I need more stages, better tools?"
7. Re-evaluate               ->  "Did the change help?"
```

```python
import dspy
from dspy.evaluate import Evaluate

# Step 1: Define metric FIRST
def answer_quality(example, prediction, trace=None):
    """Composite metric: correctness + completeness + conciseness."""
    correct = dspy.evaluate.SemanticF1()(example, prediction, trace)
    return correct

# Step 2: Build eval dataset
eval_set = [
    dspy.Example(question="What causes rain?", answer="Evaporation and condensation...").with_inputs("question"),
    # ... 50-100 more examples
]

# Step 3: Simplest baseline
baseline = dspy.Predict("question -> answer")

# Step 4: Evaluate
evaluator = Evaluate(devset=eval_set, metric=answer_quality, num_threads=8)
baseline_score = evaluator(baseline)
print(f"Baseline: {baseline_score}")

# Step 5: Optimize
optimized = dspy.MIPROv2(metric=answer_quality, auto="medium").compile(
    baseline, trainset=eval_set[:40]
)
optimized_score = evaluator(optimized)
print(f"Optimized: {optimized_score}")

# Step 6: If not good enough, iterate on program design
class BetterProgram(dspy.Module):
    def __init__(self):
        self.think = dspy.ChainOfThought("question -> answer")
    def forward(self, question):
        return self.think(question=question)

# Step 7: Re-evaluate
better_score = evaluator(BetterProgram())
print(f"ChainOfThought baseline: {better_score}")
```

**A/B testing in production:** Once you have an optimized program, serve both the baseline and optimized versions, route a percentage of traffic to each, and compare real-world metrics. This is trivial because DSPy programs are just Python objects - swap one for the other with no infrastructure changes.

---

## Pattern 5: Privacy-Preserving Architectures (PAPILLON)

PAPILLON (Privacy-Aware delegation of LLM Inference to Optimize Nimble networks) solves a critical enterprise problem: you want the quality of powerful cloud models (GPT-4o, Claude) but can't send sensitive data to external APIs.

### The Architecture

```
┌──────────────────────────────────────┐
│  User Request (contains PII/secrets) │
└──────────────┬───────────────────────┘
               │
       ┌───────▼──────────┐
       │  Small Local LM  │  ← Runs on your infrastructure
       │  (privacy-aware) │
       └───────┬──────────┘
               │  Sanitized/abstracted request
       ┌───────▼─────────┐
       │  Powerful Cloud │  ← External API (GPT-4o, etc.)
       │     LM          │
       └───────┬─────────┘
               │  High-quality response
       ┌───────▼─────────┐
       │  Small Local LM │  ← Re-contextualizes with private data
       └───────┬─────────┘
               │
       ┌───────▼────────┐
       │  Final Answer  │
       └────────────────┘
```

```python
import dspy

class PrivacyAwarePipeline(dspy.Module):
    """PAPILLON-inspired privacy-preserving pipeline."""
    def __init__(self, local_lm, cloud_lm):
        self.local_lm = local_lm
        self.cloud_lm = cloud_lm

        # Stage 1: Local model sanitizes the request
        self.sanitize = dspy.ChainOfThought(
            "sensitive_query: str -> sanitized_query: str, redacted_entities: str"
        )
        # Stage 2: Cloud model generates high-quality response
        self.generate = dspy.ChainOfThought(
            "sanitized_query: str -> response: str"
        )
        # Stage 3: Local model re-contextualizes
        self.recontextualize = dspy.ChainOfThought(
            "response: str, redacted_entities: str, original_query: str -> final_answer: str"
        )

    def forward(self, query: str):
        # Use local LM for privacy-sensitive stages
        with dspy.context(lm=self.local_lm):
            sanitized = self.sanitize(sensitive_query=query)

        # Use cloud LM for quality-critical generation
        with dspy.context(lm=self.cloud_lm):
            generated = self.generate(sanitized_query=sanitized.sanitized_query)

        # Back to local LM for re-contextualization
        with dspy.context(lm=self.local_lm):
            final = self.recontextualize(
                response=generated.response,
                redacted_entities=sanitized.redacted_entities,
                original_query=query,
            )

        return dspy.Prediction(answer=final.final_answer)

# Usage
local_lm = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434")
cloud_lm = dspy.LM("openai/gpt-4o-mini")

pipeline = PrivacyAwarePipeline(local_lm=local_lm, cloud_lm=cloud_lm)
```

This pattern ensures that **no raw sensitive data ever leaves your infrastructure** while still benefiting from powerful cloud models for reasoning and generation.

---

## Real-World Example: llms.txt Generation

The DSPy tutorials showcase building a system that generates `llms.txt` files - structured documentation designed for LLM consumption. This is a multi-stage pipeline that crawls content, extracts structure, and formats output:

```python
class LLMsTxtGenerator(dspy.Module):
    """Generate llms.txt files from website content."""
    def __init__(self):
        self.extract = dspy.ChainOfThought(
            "url: str, page_content: str -> title: str, summary: str, sections: list[str]"
        )
        self.format = dspy.Predict(
            "title: str, summary: str, sections: list[str] -> llms_txt: str"
        )

    def forward(self, url: str, page_content: str):
        extracted = self.extract(url=url, page_content=page_content)
        formatted = self.format(
            title=extracted.title,
            summary=extracted.summary,
            sections=extracted.sections,
        )
        return dspy.Prediction(llms_txt=formatted.llms_txt)
```

This is Pattern 1 (multi-stage pipeline) applied to documentation generation - a practical use case where each stage is independently testable and optimizable.

---

## Real-World Example: Yahoo Finance Agent

A financial analysis agent combines Pattern 2 (agent orchestration) with real API integrations. The DSPy tutorials demonstrate building a ReAct agent that queries Yahoo Finance for stock data:

```python
import dspy

def get_stock_info(ticker: str) -> str:
    """Get current stock price and basic info for a ticker symbol."""
    # In production: call yfinance or Yahoo Finance API
    return "AAPL: $185.50, PE: 28.5, Market Cap: $2.9T"

def get_financial_news(topic: str) -> str:
    """Search for recent financial news about a topic."""
    return "Recent: AAPL reports strong Q4 earnings, beating estimates by 5%."

def calculate_metric(expression: str) -> str:
    """Calculate a financial metric given a mathematical expression."""
    try:
        result = eval(expression)  # In production, use a safe math parser
        return str(result)
    except Exception as e:
        return f"Error: {e}"

finance_agent = dspy.ReAct(
    "question -> analysis: str, recommendation: str",
    tools=[get_stock_info, get_financial_news, calculate_metric],
    max_iters=6,
)
```

The agent autonomously decides which tools to call, in what order, and when it has enough information to synthesize a recommendation.

---

## Real-World Example: Mem0 Memory Agents

Memory-enabled agents use persistent storage to maintain context across conversations. The Mem0 integration with DSPy creates agents that **remember past interactions**:

```python
import dspy

def store_memory(key: str, value: str) -> str:
    """Store a piece of information for later retrieval across conversations."""
    # In production: use Mem0, Redis, or a vector store
    return f"Stored: {key} = {value}"

def recall_memory(query: str) -> str:
    """Recall previously stored information relevant to the query."""
    # In production: semantic search over stored memories
    return f"Recalled: User prefers concise answers and works in fintech."

memory_agent = dspy.ReAct(
    "user_message, conversation_history: str -> response: str",
    tools=[store_memory, recall_memory],
    max_iters=5,
)
```

This combines Pattern 2 (agent orchestration) with persistent state - the agent can store user preferences, past decisions, and contextual information that persists across sessions.

---

## Architecture Decision Framework

Use this framework to choose your architecture pattern:

| Factor | Multi-Stage Pipeline | Agent Orchestration | Optimization Cascade | EDD | PAPILLON |
|--------|---------------------|--------------------|--------------------|-----|----------|
| **Task predictability** | High - steps are known | Low - steps vary per input | Any | Any | Any |
| **Latency sensitivity** | Medium - parallelizable | High - sequential iterations | N/A (offline) | N/A | Higher - two LMs |
| **Data sensitivity** | Low | Low | Low | Low | **High** |
| **Optimization potential** | High - each stage tunable | Medium - trajectory-based | **Maximum** | High - metric-driven | Medium |
| **Complexity** | Low to Medium | Medium to High | Low (process, not code) | Low | Medium |

**Rules of thumb:**

1. **Start with a pipeline.** If you can define the stages upfront, do it. Pipelines are easier to debug, test, and optimize.
2. **Add agents when you can't predict the steps.** If the task requires dynamic tool selection or variable-length reasoning, use ReAct.
3. **Always use EDD.** Define metrics first regardless of architecture. It costs nothing and pays compound dividends.
4. **Use PAPILLON when privacy matters.** Enterprise systems with PII, medical data, or financial records need privacy-preserving architectures.
5. **Layer optimization cascades on top.** Once the architecture is stable, run through the cascade to squeeze out maximum performance.

---

## Key Takeaways

- **Multi-stage pipelines** are the workhorse pattern - decompose complex tasks into optimizable stages
- **Agent orchestration** handles open-ended tasks where sub-modules serve as specialized tools
- **Optimization cascades** (MIPROv2 -> BootstrapFewShot -> BetterTogether -> BootstrapFinetune) extract maximum performance
- **Evaluation-driven development** means metrics come first, code comes second
- **PAPILLON** solves the privacy vs quality tradeoff with a local/cloud split
- Real-world examples like llms.txt generation, Yahoo Finance agents, and Mem0 memory agents demonstrate these patterns in practice
- Choose your architecture based on task predictability, latency, privacy needs, and optimization goals

---

## Next Up

[12.2: Real-World DSPy Applications](../12.2-real-world-applications/blog.md) explores concrete application case studies across industries, from customer service to code generation.

---

## Resources

- [DSPy Tutorials](https://dspy.ai/tutorials/)
- [DSPy API Reference - Modules](https://dspy.ai/api/modules/)
- [DSPy API Reference - Optimizers](https://dspy.ai/api/optimizers/)
- [PAPILLON Paper](https://arxiv.org/abs/2501.02649)
- [DSPy Real-World Examples](https://dspy.ai/tutorials/real_world_examples/)
- [ReAct Agents in DSPy](https://dspy.ai/tutorials/agents/)
