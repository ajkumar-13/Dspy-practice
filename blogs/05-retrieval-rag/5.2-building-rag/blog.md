# 5.2: Building RAG Pipelines

## Introduction

Retrieval-Augmented Generation is the most widely deployed LM pattern in production today. The idea is deceptively simple: retrieve relevant context, then have the language model reason over it. But in practice, RAG pipelines are fragile: they're sensitive to retrieval quality, chunk size, prompt wording, and the number of passages you include.

DSPy changes the game. Instead of hand-tuning every component, you define your RAG pipeline as a **composable module**, evaluate it with a principled metric, and let an optimizer improve it automatically. In this blog, you'll build a complete RAG system from scratch, evaluate it with `SemanticF1`, and optimize it with `MIPROv2`, going from a mediocre baseline to strong performance with zero manual prompt engineering.

---

## What You'll Learn

- The classic RAG pattern: retrieve → read → respond
- Building RAG as a `dspy.Module` with `ChainOfThought`
- Evaluating RAG with the `SemanticF1` metric
- Optimizing RAG with `MIPROv2` for automatic improvement
- Query generation: having the LM generate search queries
- Multi-retriever RAG: combining different retrieval sources
- Practical considerations: chunk size, top-k, context window limits

---

## Prerequisites

- Completed [5.1: Retrieval in DSPy](../5.1-retrieval-in-dspy/blog.md)
- `faiss-cpu` installed (`uv add faiss-cpu`)
- Familiarity with DSPy evaluation and optimization from Phases 3-4

---

## The Classic RAG Pattern

Every RAG pipeline follows the same three-step flow:

1. **Retrieve**: find relevant passages for the user's question
2. **Read**: reason over the retrieved context
3. **Respond**: generate an answer grounded in the evidence

```mermaid
flowchart LR
    Q[Question] --> R["Retrieve(search fn)"]
    R -->|"list[str] passages"| CoT["Read & Reason(ChainOfThought)"]
    CoT --> Resp[Response]
```

In DSPy, this becomes a compact module:

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Set up retrieval: FAISS-backed local search
embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)
corpus = [
    "The Amazon rainforest produces about 20% of the world's oxygen.",
    "The Sahara Desert is the largest hot desert, spanning 3.6 million square miles.",
    "The Pacific Ocean covers more area than all the Earth's land combined.",
    "Mount Everest grows approximately 4mm taller each year due to tectonic activity.",
    "The Mariana Trench is the deepest point in the ocean at about 36,000 feet.",
    "Antarctica contains about 70% of the world's fresh water in its ice sheet.",
    "The Dead Sea is the lowest point on land at 430 meters below sea level.",
    "Lake Baikal in Russia holds about 20% of the world's unfrozen fresh water.",
    "The Nile River, at 6,650 km, is the longest river in the world.",
    "The Great Barrier Reef is the largest living structure on Earth, visible from space.",
]

search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=3)


class RAG(dspy.Module):
    """A simple Retrieve-then-Read RAG pipeline."""

    def __init__(self):
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        context = search(question)
        return self.respond(context=context, question=question)


# Test it
rag = RAG()
result = rag(question="What is the deepest point in the ocean?")
print(f"Response: {result.response}")
```

This is the heart of RAG in DSPy. Notice what's *not* here: no system prompts, no instructions about "use the context to answer," no output parsing. The signature `"context, question -> response"` tells DSPy what the module should do. `ChainOfThought` adds step-by-step reasoning. The optimizer will handle the rest.

---

## Evaluating RAG with SemanticF1

Before optimizing, you need to know how your baseline performs. DSPy provides `SemanticF1`: a metric that measures the semantic overlap between the predicted response and the gold answer. Unlike exact match, it handles paraphrasing and partial correctness.

```python
# Build evaluation data
trainset = [
    dspy.Example(
        question="What percentage of the world's oxygen does the Amazon produce?",
        response="About 20%",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the largest hot desert in the world?",
        response="The Sahara Desert",
    ).with_inputs("question"),
    dspy.Example(
        question="How deep is the Mariana Trench?",
        response="About 36,000 feet",
    ).with_inputs("question"),
    dspy.Example(
        question="What percentage of the world's fresh water does Antarctica contain?",
        response="About 70%",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the longest river in the world?",
        response="The Nile River at 6,650 km",
    ).with_inputs("question"),
]

devset = [
    dspy.Example(
        question="What ocean covers more area than all land combined?",
        response="The Pacific Ocean",
    ).with_inputs("question"),
    dspy.Example(
        question="How much does Mount Everest grow each year?",
        response="About 4mm per year",
    ).with_inputs("question"),
    dspy.Example(
        question="Where is the lowest point on land?",
        response="The Dead Sea, at 430 meters below sea level",
    ).with_inputs("question"),
]

# Evaluate baseline
metric = dspy.SemanticF1()
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)

rag = RAG()
baseline_score = evaluate(rag)
print(f"Baseline SemanticF1: {baseline_score:.1f}%")
```

`SemanticF1` uses token-level overlap to compute precision, recall, and F1 between the prediction and the gold answer. It's more forgiving than exact match but still rewards precise, factual responses.

---

## Optimizing RAG with MIPROv2

Here's where DSPy shines. A single `MIPROv2` call can significantly boost your RAG pipeline by discovering better instructions and few-shot demonstrations:

```python
# Optimize with MIPROv2
optimizer = dspy.MIPROv2(
    metric=metric,
    auto="medium",  # Controls optimization intensity
    num_threads=4,
)

optimized_rag = optimizer.compile(rag, trainset=trainset)

# Evaluate the optimized pipeline
optimized_score = evaluate(optimized_rag)
print(f"Optimized SemanticF1: {optimized_score:.1f}%")
print(f"Improvement: {optimized_score - baseline_score:.1f}% absolute")
```

In the official DSPy RAG tutorial, this pattern produced improvements from **42% to 61%** SemanticF1: a meaningful jump with zero manual prompt engineering. The optimizer discovers:

- **Better instructions** for the `respond` predictor
- **Better few-shot demonstrations** that show the model how to reason over context
- **Better query patterns** implicitly, by rewarding outputs that use retrieved context well

### Saving the Optimized Pipeline

```python
# Save for later use or deployment
optimized_rag.save("optimized_rag.json")

# Load it back
loaded_rag = RAG()
loaded_rag.load("optimized_rag.json")
```

---

## Query Generation

In the basic RAG pipeline above, the user's question goes directly to the retriever. But user questions are often poor search queries: they're conversational, ambiguous, or multi-faceted. **Query generation** lets the LM rewrite the question into a better search query:

```python
class RAGWithQueryGen(dspy.Module):
    """RAG with LM-generated search queries."""

    def __init__(self):
        self.generate_query = dspy.ChainOfThought(
            "question -> search_query: str"
        )
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        # Step 1: Generate an optimized search query
        query = self.generate_query(question=question).search_query
        # Step 2: Retrieve with the generated query
        context = search(query)
        # Step 3: Respond using the original question + retrieved context
        return self.respond(context=context, question=question)


rag_qgen = RAGWithQueryGen()
result = rag_qgen(question="I heard some sea is really low, where is it?")
print(f"Response: {result.response}")
```

This is powerful because the optimizer can now tune the query generation step too. MIPROv2 will discover instructions and demonstrations that produce better search queries, improving retrieval quality without touching the retriever itself.

---

## Multi-Retriever RAG

Sometimes you need to combine results from multiple retrieval sources, for example, a local knowledge base *and* a web search API. DSPy makes this straightforward:

```python
class MultiRetrieverRAG(dspy.Module):
    """RAG combining multiple retrieval sources."""

    def __init__(self, retrievers: list, k_per_source: int = 3):
        self.retrievers = retrievers
        self.k_per_source = k_per_source
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        # Gather passages from all sources
        all_passages = []
        for retriever in self.retrievers:
            passages = retriever(question, k=self.k_per_source)
            all_passages.extend(passages)

        return self.respond(context=all_passages, question=question)


# Example usage with two retrievers
# multi_rag = MultiRetrieverRAG(retrievers=[local_search, web_search], k_per_source=3)
```

Each retriever is just a callable, so mixing FAISS, ColBERTv2, and custom API-backed search is trivial.

---

## Practical Considerations

### Chunk Size

How you split your documents matters enormously:

| Chunk Size | Pros | Cons |
|-----------|------|------|
| Small (50-100 words) | Precise retrieval, less noise | May miss context, more chunks to search |
| Medium (150-300 words) | Good balance of precision and context | Most common in practice |
| Large (500+ words) | Complete context per chunk | Dilutes relevant information, fills context window |

Start with ~200-word chunks and experiment. DSPy's optimization will adapt to whatever chunk size you choose.

### Top-k Selection

More passages means more context for the LM, but also more noise and higher token costs:

- **k=3**: Good default for focused, factual questions
- **k=5**: Better for complex questions needing multiple evidence pieces
- **k=10+**: Usually only needed for recall-critical tasks

### Context Window Limits

A common pitfall: retrieving too many passages and exceeding the model's context window. As a rule of thumb, keep retrieved context under 50% of the model's context window to leave room for instructions, demonstrations, and the response.

---

## Putting It All Together

Here's the full pattern that you'll use in almost every RAG project:

```python
# 1. Set up retrieval
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5)

# 2. Build the module
rag = RAG()

# 3. Prepare data
trainset = [...]  # 20-50 examples
devset = [...]    # 10-20 examples

# 4. Evaluate baseline
metric = dspy.SemanticF1()
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=4)
print(f"Baseline: {evaluate(rag):.1f}")

# 5. Optimize
optimizer = dspy.MIPROv2(metric=metric, auto="medium")
optimized = optimizer.compile(rag, trainset=trainset)
print(f"Optimized: {evaluate(optimized):.1f}")

# 6. Save
optimized.save("rag_optimized.json")
```

Build → evaluate → optimize → save. This is the DSPy workflow, and it applies to RAG just as well as it applies to any other LM task.

---

## Key Takeaways

- **RAG in DSPy is a module**, not a prompt chain. This makes it composable, testable, and optimizable.
- **`SemanticF1`** is the go-to metric for evaluating RAG responses: it handles paraphrasing gracefully.
- **`MIPROv2` can dramatically improve RAG** by discovering better instructions and demonstrations (42% -> 61% in the official tutorial).
- **Query generation** adds an LM-powered rewriting step that improves retrieval quality, and it's also optimizable.
- **Multi-retriever RAG** combines sources by treating each retriever as a callable: no special wiring needed.
- **Chunk size and top-k** are important hyperparameters; start with 200-word chunks and k=3-5.

---

## Next Up

Basic RAG handles single-step questions well, but what about questions that require *multiple* retrieval steps? In the next blog, we'll build **multi-hop RAG**: iteratively retrieving and reasoning until the model has enough evidence to answer complex, multi-part questions.

**[5.3: Multi-Hop RAG →](../5.3-multi-hop-rag/blog.md)**

---

## Resources

- [DSPy RAG Tutorial](https://dspy.ai/tutorials/rag/)
- [SemanticF1 Documentation](https://dspy.ai/api/metrics/SemanticF1/)
- [MIPROv2 Optimizer](https://dspy.ai/api/optimizers/MIPROv2/)
- [DSPy Retrieval Overview](https://dspy.ai/learn/retrieval_models_clients/overview/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
