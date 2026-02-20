# 5.P: Project: Research Assistant

## Introduction

You've learned four retrieval patterns: local embeddings, single-hop RAG, multi-hop RAG, and agentic RAG. Now it is time to put them all together into a real, end-to-end project: a **research assistant** that answers complex questions by dynamically searching for evidence, reasoning over multiple sources, and producing well-grounded responses.

This project demonstrates the full DSPy workflow: **build, evaluate, optimize, and compare**. You'll start with a baseline agent, measure its performance, optimize it with MIPROv2, and quantify the improvement (all in about 120 lines of code).

---

## Project Overview

Here's what we'll build:

1.  **A retrieval backend**: ColBERTv2 over Wikipedia abstracts
2.  **Search and lookup tools**: for the agent to use dynamically
3.  **A ReAct-based research agent**: decides what to search and when to answer
4.  **An evaluation dataset**: complex, multi-hop questions with gold answers
5.  **A metric**: SemanticF1 for measuring answer quality
6.  **Baseline evaluation**: how the unoptimized agent performs
7.  **MIPROv2 optimization**: automatic prompt and demo tuning
8.  **Before/after comparison**: quantifying the improvement

---

## Step 1: Set Up Retrieval

We'll use `dspy.ColBERTv2` with the publicly available Wikipedia abstracts endpoint. This gives us fast, high-quality retrieval over a massive corpus without the need to manage any infrastructure.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Set up retrieval: ColBERTv2 over Wikipedia abstracts
search = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
```

---

## Step 2: Build the Agent

We'll create two tools for the agent and wire them into a `dspy.ReAct` module.

**`search_wikipedia`** performs a broad search and returns top passages. **`lookup_details`** extracts specific information from a passage by keyword, which is useful when the agent has a broad passage but needs a particular fact.

```python
def search_wikipedia(query: str) -> list[str]:
    """Search Wikipedia for relevant passages. Returns top-5 results for the given query."""
    results = search(query, k=30)
    return [r.long_text for r in results[:5]]


def lookup_details(text: str, keyword: str) -> str:
    """Find the sentence containing a specific keyword in a text passage."""
    sentences = text.split(". ")
    matches = [s.strip() for s in sentences if keyword.lower() in s.lower()]
    if matches:
        return ". ".join(matches)
    return f"Keyword '{keyword}' not found in the provided text."


# Build the research agent
agent = dspy.ReAct(
    "question -> answer",
    tools=[search_wikipedia, lookup_details],
)
```

The agent can now autonomously:

-   **Search** for broad information on any topic
-   **Look up** specific details within retrieved passages
-   **Reason** about whether it has enough information to answer
-   **Search again** with a refined query if needed

---

## Step 3: Prepare Evaluation Data

We need multi-hop questions that genuinely require multiple retrieval steps. These are inspired by the HotPotQA dataset pattern: each question requires combining facts from at least two different sources.

```python
# Training set: used for optimization
trainset = [
    dspy.Example(
        question="What is the capital of the country where the Taj Mahal is located?",
        answer="New Delhi",
    ).with_inputs("question"),
    dspy.Example(
        question="Who was the president of the United States when the Berlin Wall fell in 1989?",
        answer="George H. W. Bush",
    ).with_inputs("question"),
    dspy.Example(
        question="What language is primarily spoken in the country where Machu Picchu is located?",
        answer="Spanish",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the tallest mountain in the country where Mount Fuji is located?",
        answer="Mount Fuji",
    ).with_inputs("question"),
    dspy.Example(
        question="Who founded the company that created the iPhone?",
        answer="Steve Jobs",
    ).with_inputs("question"),
    dspy.Example(
        question="What ocean borders the western coast of the country where the Amazon rainforest is mostly located?",
        answer="Atlantic Ocean",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the currency used in the country where the Great Wall is located?",
        answer="Renminbi (Yuan)",
    ).with_inputs("question"),
    dspy.Example(
        question="Who wrote the novel that the movie The Shining is based on?",
        answer="Stephen King",
    ).with_inputs("question"),
    dspy.Example(
        question="What river flows through the city where the Louvre Museum is located?",
        answer="Seine",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the population of the city where the Colosseum is located?",
        answer="About 2.8 million",
    ).with_inputs("question"),
]

# Dev set: used for evaluation (held out from training)
devset = [
    dspy.Example(
        question="What continent is the country where the Pyramids of Giza are located on?",
        answer="Africa",
    ).with_inputs("question"),
    dspy.Example(
        question="Who is the architect of the building where the United Nations General Assembly meets?",
        answer="Oscar Niemeyer was part of the design team led by Wallace Harrison",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the official language of the country where Angkor Wat is located?",
        answer="Khmer",
    ).with_inputs("question"),
    dspy.Example(
        question="In what year was the university founded where Stephen Hawking was a professor?",
        answer="1209",
    ).with_inputs("question"),
    dspy.Example(
        question="What sea borders the country where Petra is located?",
        answer="Red Sea and Dead Sea",
    ).with_inputs("question"),
]
```

---

## Step 4: Define Metrics

We'll use `SemanticF1`, which measures the semantic overlap between the agent's answer and the gold answer. It handles paraphrasing and partial matches gracefully.

```python
# Define the evaluation metric
metric = dspy.SemanticF1()

# Create an evaluator
evaluate = dspy.Evaluate(
    devset=devset,
    metric=metric,
    num_threads=4,
    display_progress=True,
    display_table=5,  # Show a table of the first 5 examples
)
```

---

## Step 5: Baseline Evaluation

Before optimizing, let's measure how the unoptimized agent performs:

```python
# Evaluate the baseline agent
print("=" * 60)
print("BASELINE EVALUATION")
print("=" * 60)

baseline_score = evaluate(agent)
print(f"\nBaseline SemanticF1: {baseline_score:.1f}%")
```

With GPT-4o-mini and no optimization, you'll typically see a baseline SemanticF1 in the range of **20 to 40%** on multi-hop questions. The model often:

-   Searches once when it should search multiple times
-   Uses the raw question as a search query instead of decomposing it
-   Stops too early without enough evidence
-   Hallucinates answers instead of searching for them

---

## Step 6: Optimize

Now let's unleash MIPROv2 on the agent. The optimizer will discover better instructions, few-shot demonstrations, and reasoning patterns:

```python
print("\n" + "=" * 60)
print("OPTIMIZING WITH MIPROv2")
print("=" * 60)

optimizer = dspy.MIPROv2(
    metric=metric,
    auto="medium",
    num_threads=4,
)

optimized_agent = optimizer.compile(
    agent,
    trainset=trainset,
)

# Save the optimized agent for later use
optimized_agent.save("research_assistant_optimized.json")
print("\nOptimized agent saved to research_assistant_optimized.json")
```

`auto="medium"` runs a moderate optimization: enough to find good prompts without excessive cost. For production use, you can try `auto="heavy"` for a more thorough search.

---

## Step 7: Compare Results

The moment of truth: let's compare baseline vs. optimized performance:

```python
print("\n" + "=" * 60)
print("OPTIMIZED EVALUATION")
print("=" * 60)

optimized_score = evaluate(optimized_agent)

print("\n" + "=" * 60)
print("RESULTS COMPARISON")
print("=" * 60)
print(f"Baseline SemanticF1:  {baseline_score:.1f}%")
print(f"Optimized SemanticF1: {optimized_score:.1f}%")
print(f"Improvement:          {optimized_score - baseline_score:.1f}% absolute")
print("=" * 60)
```

In the official DSPy tutorial, this pattern yielded improvements from **8% to 42%**, and that was on a harder dataset. On our curated questions, you should see a meaningful jump.

### Inspecting What Changed

After optimization, inspect the agent's behavior to understand what the optimizer discovered:

```python
# Run a query and inspect the full reasoning trace
result = optimized_agent(
    question="What river flows through the city where the Louvre Museum is located?"
)
print(f"\nAnswer: {result.answer}")
print("\n--- Full Reasoning Trace ---")
dspy.inspect_history(n=1)
```

You'll often see that the optimized agent:

-   **Decomposes questions more effectively**: searching for components separately
-   **Writes better search queries**: using specific, keyword-rich queries instead of raw questions
-   **Reasons more carefully** about whether retrieved evidence is sufficient
-   **Terminates at the right time** instead of stopping too early or too late

---

## Complete Script

Here's the full project in one runnable file:

```python

import dspy
from dotenv import load_dotenv

load_dotenv()

# === Configuration ===
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

search = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")


# === Tools ===
def search_wikipedia(query: str) -> list[str]:
    """Search Wikipedia for relevant passages. Returns top-5 results for the given query."""
    results = search(query, k=30)
    return [r.long_text for r in results[:5]]


def lookup_details(text: str, keyword: str) -> str:
    """Find the sentence containing a specific keyword in a text passage."""
    sentences = text.split(". ")
    matches = [s.strip() for s in sentences if keyword.lower() in s.lower()]
    if matches:
        return ". ".join(matches)
    return f"Keyword '{keyword}' not found in the provided text."


# === Agent ===
agent = dspy.ReAct("question -> answer", tools=[search_wikipedia, lookup_details])

# === Data ===
trainset = [
    dspy.Example(question="What is the capital of the country where the Taj Mahal is located?", answer="New Delhi").with_inputs("question"),
    dspy.Example(question="Who was the president of the US when the Berlin Wall fell?", answer="George H. W. Bush").with_inputs("question"),
    dspy.Example(question="What language is primarily spoken where Machu Picchu is located?", answer="Spanish").with_inputs("question"),
    dspy.Example(question="Who founded the company that created the iPhone?", answer="Steve Jobs").with_inputs("question"),
    dspy.Example(question="What is the currency used where the Great Wall is located?", answer="Renminbi (Yuan)").with_inputs("question"),
    dspy.Example(question="Who wrote the novel The Shining is based on?", answer="Stephen King").with_inputs("question"),
    dspy.Example(question="What river flows through the city where the Louvre is located?", answer="Seine").with_inputs("question"),
    dspy.Example(question="What ocean borders the western coast of the country with the Amazon rainforest?", answer="Atlantic Ocean").with_inputs("question"),
]

devset = [
    dspy.Example(question="What continent is the country with the Pyramids of Giza on?", answer="Africa").with_inputs("question"),
    dspy.Example(question="What is the official language of the country where Angkor Wat is located?", answer="Khmer").with_inputs("question"),
    dspy.Example(question="In what year was the university founded where Stephen Hawking was a professor?", answer="1209").with_inputs("question"),
    dspy.Example(question="What sea borders the country where Petra is located?", answer="Red Sea and Dead Sea").with_inputs("question"),
]

# === Evaluate Baseline ===
metric = dspy.SemanticF1()
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True, display_table=5)

print("=" * 60)
print("BASELINE EVALUATION")
print("=" * 60)
baseline_score = evaluate(agent)
print(f"\nBaseline SemanticF1: {baseline_score:.1f}%")

# === Optimize ===
print("\n" + "=" * 60)
print("OPTIMIZING WITH MIPROv2")
print("=" * 60)

optimizer = dspy.MIPROv2(metric=metric, auto="medium", num_threads=4)
optimized_agent = optimizer.compile(agent, trainset=trainset)
optimized_agent.save("research_assistant.json")

# === Evaluate Optimized ===
print("\n" + "=" * 60)
print("OPTIMIZED EVALUATION")
print("=" * 60)
optimized_score = evaluate(optimized_agent)

# === Compare ===
print("\n" + "=" * 60)
print("RESULTS COMPARISON")
print("=" * 60)
print(f"Baseline SemanticF1:  {baseline_score:.1f}%")
print(f"Optimized SemanticF1: {optimized_score:.1f}%")
print(f"Improvement:          {optimized_score - baseline_score:.1f}% absolute")
print("=" * 60)
```

---

## What We Learned

This project ties together everything from Phase 5:

1.  **Retrieval is just a callable** (5.1): we wrapped ColBERTv2 in a simple function.
2.  **RAG is a module** (5.2): composable, evaluable, and optimizable.
3.  **Multi-hop reasoning** (5.3): the agent naturally performs multiple searches when needed.
4.  **Agentic patterns** (5.4): `dspy.ReAct` gives the model autonomy over its retrieval strategy.
5.  **The DSPy workflow**: build, evaluate, optimize, and compare. Every project follows this pattern.

The key insight: **you never wrote a prompt.** You defined tools, specified a signature, provided examples, chose a metric, and let the optimizer do the work. That is programming, not prompting.

---

## Next Up

You've mastered retrieval and RAG. In Phase 6, we will go beyond retrieval tools and into full **agent architectures**: ReAct agents with multiple tools, advanced tool design, MCP integration, and agents with persistent memory.

**[6.1: ReAct Agents →](../../06-agents/6.1-react-agents/blog.md)**

---

## Resources

- [DSPy RAG Tutorial](https://dspy.ai/tutorials/rag/)
- [DSPy Agents Tutorial](https://dspy.ai/tutorials/agents/)
- [DSPy ReAct Documentation](https://dspy.ai/api/modules/ReAct/)
- [MIPROv2 Optimizer](https://dspy.ai/api/optimizers/MIPROv2/)
- [SemanticF1 Metric](https://dspy.ai/api/metrics/SemanticF1/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
