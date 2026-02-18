# 5.4: RAG as Agent

## Introduction

In the previous blogs, our RAG pipelines followed a **fixed flow**: retrieve a set number of passages, then generate a response. The model had no choice about *when* to retrieve, *what* to search for, or *whether* the retrieved context was sufficient. It simply ran the pipeline mechanically.

**Agentic RAG** flips this around. Using `dspy.ReAct`, you turn retrieval into a **tool** that the model calls on demand, deciding autonomously when it needs more information, what queries to run, and when it has enough evidence to respond. This approach is more flexible, more powerful, and often dramatically more effective than fixed pipelines.

---

## What You'll Learn

- How `dspy.ReAct` turns retrieval into an agent-driven tool
- Converting retriever functions into ReAct-compatible tools
- Building an agentic RAG pipeline with dynamic retrieval
- Advantages over fixed RAG pipelines
- Optimizing agentic RAG with MIPROv2
- When to use agentic RAG vs. fixed pipelines

---

## Prerequisites

- Completed [5.3: Multi-Hop RAG](../5.3-multi-hop-rag/blog.md)
- Understanding of `dspy.ReAct` basics (we will cover everything you need here; Phase 6 goes deeper)

---

## From Fixed Pipelines to Agents

In a fixed RAG pipeline, the flow is predetermined:

```
question -> retrieve(k=3) -> respond
```

The model always retrieves exactly 3 passages, always uses the original question as the query, and always responds after one retrieval step. This works for simple questions but fails when:

- The first retrieval doesn't return useful results
- The question requires multiple searches with different queries
- The model needs to reason about *what* to search for before searching
- Some questions don't need retrieval at all

An agentic RAG pipeline gives the model a **search tool** and lets it decide:

```
question -> [think -> search("query1") -> think -> search("query2") -> think -> respond]
```

The model orchestrates its own retrieval strategy. It can search zero times, once, or many times, whatever the question demands.

---

## Building Agentic RAG with ReAct

Here's how to build an agentic RAG system using `dspy.ReAct`:

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Set up retrieval
search = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")


# Define retrieval as a tool function
def search_wikipedia(query: str) -> list[str]:
    """Search Wikipedia for relevant passages. Returns top-5 results."""
    results = search(query, k=30)
    return [r.long_text for r in results[:5]]


# Create the agentic RAG pipeline with ReAct
react = dspy.ReAct(
    "question -> answer",
    tools=[search_wikipedia],
)

# Test it
result = react(question="What year was the university founded where Tim Berners-Lee studied?")
print(f"Answer: {result.answer}")
```

That's it. Three key pieces:

1.  **A retriever function** with a docstring (ReAct uses the docstring to understand the tool)
2.  **`dspy.ReAct`** with the function passed as a tool
3.  **A signature** declaring the task (`"question -> answer"`)

### What Happens Under the Hood

When you call `react(question=...)`, the ReAct module runs a loop:

1.  **Thought:** The model reasons about what it needs to know
2.  **Action:** It decides to call `search_wikipedia` with a specific query
3.  **Observation:** It receives the search results
4.  **Repeat:** It may think again and decide to search again, or decide it has enough context
5.  **Finish:** It produces the final answer

This loop is fully autonomous: the model decides the number of iterations, the queries, and when to stop. You can inspect the trace:

```python
# See the full thought-action-observation trace
dspy.inspect_history(n=1)
```

---

## Multiple Tools

Agentic RAG becomes even more powerful when you give the model multiple tools:

```python
def search_wikipedia(query: str) -> list[str]:
    """Search Wikipedia for factual information. Returns top-5 results."""
    results = search(query, k=30)
    return [r.long_text for r in results[:5]]


def lookup_keyword(passage: str, keyword: str) -> str:
    """Look up a keyword in a passage and return the surrounding sentence."""
    sentences = passage.split(". ")
    for sentence in sentences:
        if keyword.lower() in sentence.lower():
            return sentence.strip()
    return f"Keyword '{keyword}' not found in the passage."


react = dspy.ReAct(
    "question -> answer",
    tools=[search_wikipedia, lookup_keyword],
)

result = react(
    question="What is the elevation of the city where the headquarters of UNESCO is located?"
)
print(f"Answer: {result.answer}")
```

The model can now **search** for broad information and then **look up** specific details within passages. This mimics how a human researcher works.

---

## Optimizing Agentic RAG

Just like fixed RAG pipelines, agentic RAG is fully optimizable with MIPROv2. The optimizer tunes the instructions and demonstrations for the ReAct module, improving how the model reasons, what queries it generates, and how it synthesizes answers.

```python
# Training data
trainset = [
    dspy.Example(
        question="What country is the Taj Mahal located in?",
        answer="India",
    ).with_inputs("question"),
    dspy.Example(
        question="Who painted the ceiling of the Sistine Chapel?",
        answer="Michelangelo",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the tallest mountain in Africa?",
        answer="Mount Kilimanjaro",
    ).with_inputs("question"),
    dspy.Example(
        question="In what year did the Berlin Wall fall?",
        answer="1989",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the chemical symbol for iron?",
        answer="Fe",
    ).with_inputs("question"),
]

devset = [
    dspy.Example(
        question="Which planet is known as the Red Planet?",
        answer="Mars",
    ).with_inputs("question"),
    dspy.Example(
        question="What river runs through London?",
        answer="The Thames",
    ).with_inputs("question"),
    dspy.Example(
        question="Who wrote the play Hamlet?",
        answer="William Shakespeare",
    ).with_inputs("question"),
]

# Evaluate baseline
metric = dspy.SemanticF1()
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)

baseline_score = evaluate(react)
print(f"Baseline: {baseline_score:.1f}%")

# Optimize
optimizer = dspy.MIPROv2(metric=metric, auto="medium", num_threads=4)
optimized_react = optimizer.compile(react, trainset=trainset)

optimized_score = evaluate(optimized_react)
print(f"Optimized: {optimized_score:.1f}%")
print(f"Improvement: {optimized_score - baseline_score:.1f}% absolute")
```

In the official DSPy tutorial, agentic RAG optimization produced improvements from **8% to 42%**, a massive jump. The optimizer learns:

-   **Better reasoning patterns**: how to decompose questions before searching
-   **Better query formulation**: what makes an effective search query
-   **Better termination logic**: when the model has enough evidence to answer confidently

---

## When to Use Agentic RAG vs. Fixed Pipelines

| Use Case | Recommended Approach |
|----------|---------------------|
| Simple factual Q&A | Fixed single-hop RAG |
| Questions with predictable structure | Fixed multi-hop RAG |
| Highly variable question complexity | Agentic RAG (ReAct) |
| Multiple retrieval sources available | Agentic RAG with multiple tools |
| Latency-critical applications | Fixed RAG (fewer LM calls) |
| Cost-sensitive applications | Fixed RAG (predictable cost per query) |
| Research / exploration tasks | Agentic RAG |

**Rules of thumb:**

-   **Start with fixed RAG.** It is simpler, faster, and cheaper. Many tasks do not need the flexibility of an agent.
-   **Upgrade to agentic RAG when** you see failures caused by rigid retrieval, such as questions that need dynamic query generation, multiple searches, or conditional retrieval logic.
-   **Cost tradeoff:** Agentic RAG makes more LM calls per query (thought-action-observation loops), so it costs more per query but may need fewer total queries to reach good accuracy.

---

## End-to-End Example

Here's a complete, self-contained agentic RAG pipeline you can run:

```python

import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

search = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")


def search_wikipedia(query: str) -> list[str]:
    """Search Wikipedia for relevant passages. Returns top-5 results."""
    results = search(query, k=30)
    return [r.long_text for r in results[:5]]


# Build the agent
agent = dspy.ReAct("question -> answer", tools=[search_wikipedia])

# Ask a multi-hop question
result = agent(
    question="What is the population of the country where Mount Kilimanjaro is located?"
)
print(f"Answer: {result.answer}")

# Inspect the reasoning trace
dspy.inspect_history(n=1)
```

---

## Key Takeaways

- **Agentic RAG uses `dspy.ReAct`** to let the model decide when, what, and how often to retrieve.
- **Any Python function with a docstring** can be a ReAct tool. Just pass it in the `tools` list.
- **Multiple tools** enable richer behavior: search, lookup, calculate, etc.
- **Optimization with MIPROv2** can produce dramatic improvements (8% to 42% in the official tutorial).
- **Fixed RAG is simpler and cheaper.** Use agentic RAG when you need flexible, multi-step retrieval.
- **The reasoning trace** (`dspy.inspect_history`) shows exactly how the agent decided to use its tools.

---

## Next Up

You've now learned four retrieval patterns: local embeddings, single-hop RAG, multi-hop RAG, and agentic RAG. In the capstone project, you'll combine everything to build a **research assistant** that answers complex questions using multi-step retrieval, evaluation, and optimization.

**[5.P: Project: Research Assistant →](../5.P-project-research-assistant/blog.md)**

---

## Resources

- 📖 [DSPy ReAct Documentation](https://dspy.ai/api/modules/ReAct/)
- 📖 [DSPy RAG Tutorial](https://dspy.ai/tutorials/rag/)
- 📖 [DSPy Agents Tutorial](https://dspy.ai/tutorials/agents/)
- 📖 [ReAct Paper (Yao et al.)](https://arxiv.org/abs/2210.03629)
- 💻 [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
