# 5.1: Retrieval in DSPy

## Introduction

Retrieval is the foundation of every RAG system: before a language model can reason over external knowledge, it has to *find* the right knowledge. DSPy treats retrieval as a first-class primitive. Rather than acting as a brittle glue layer between a vector database and a prompt, it serves as a **composable, optimizable building block** that plugs into the same module-and-optimizer ecosystem you've been learning throughout this series.

In this post, you'll learn every retrieval abstraction DSPy offers: the `Embedder` class for embedding models (including HuggingFace and fine-tuned models), the `Embeddings` retriever for local FAISS-backed vector search, `ColBERTv2` for hosted ColBERT endpoints, and the pattern for writing custom retriever functions that wrap any external system.

---

## What You'll Learn

- How DSPy models retrieval and why it matters for optimization
- Using `dspy.Embedder` to wrap hosted and local embedding models
- Integrating HuggingFace sentence-transformers and fine-tuned embedding models
- Building a local vector retriever with `dspy.retrievers.Embeddings` and FAISS
- Saving and loading embedding indexes to avoid recomputation
- Using `dspy.ColBERTv2` for hosted ColBERT retrieval
- Writing custom retriever functions for any search backend
- Integration patterns with Pinecone, Weaviate, and ChromaDB

---

## Prerequisites

- Completed [Phase 4: Optimization](../../04-optimization/4.1-bootstrap-rs/blog.md)
- DSPy installed (`uv add dspy python-dotenv`)
- `faiss-cpu` installed for local vector search (`uv add faiss-cpu`)
- An OpenAI API key (for hosted embedding models)
- Optional: `sentence-transformers` for local HuggingFace models (`uv add sentence-transformers`)

---

## Retrieval Concepts in DSPy

In traditional RAG implementations, retrieval is usually a hand-wired step: you embed a query, call a vector database, parse the results, and paste them into a prompt. If you want to change the retrieval strategy, you rewrite plumbing code. If you want the optimizer to improve retrieval, you're on your own.

DSPy takes a different approach. Retrieval is modeled as a **callable**: any Python function or object that accepts a query (or queries) and returns a list of passages. This means:

1. **Retrieval is composable**: it slots into `dspy.Module` pipelines like any other component.
2. **Retrieval is swappable**: switch from a local FAISS index to Pinecone by changing one line.
3. **Retrieval participates in optimization**: optimizers can tune the queries that get sent to the retriever.

DSPy ships with two built-in retriever types and supports arbitrary custom retrievers.

---

## Built-in Retrievers

### The Embeddings Retriever

The most common local retriever in DSPy uses `dspy.Embedder` paired with `dspy.retrievers.Embeddings`. Together they give you a FAISS-backed vector search with zero external dependencies beyond `faiss-cpu`.

**`dspy.Embedder`** wraps any embedding model supported by LiteLLM. You specify the model using the same `provider/model` format as `dspy.LM`:

```python
embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)
```

**`dspy.retrievers.Embeddings`** takes an embedder and a corpus, builds a FAISS index, and provides a search interface:

```python
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5)
results = search("your query here")  # returns a dspy.Prediction with .passages
passages = results.passages  # list[str] of top-k passages
```

### ColBERTv2 Retriever

For hosted retrieval over large corpora, DSPy supports `dspy.ColBERTv2`, which connects to a ColBERTv2 server over HTTP. This is ideal for large-scale corpora like Wikipedia where you don't want to manage an index locally.

---

## Using Embeddings with FAISS

Let's build a complete local retriever from scratch. The workflow is: create an embedder, define a corpus, then build a retriever.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure the LM (for later use in RAG)
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Step 1: Create an embedder using OpenAI's text-embedding-3-small
embedder = dspy.Embedder(
    "openai/text-embedding-3-small",
    dimensions=512,
)

# Step 2: Define a corpus: a list of strings
corpus = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889.",
    "The Great Wall of China stretches over 13,000 miles and was built over many centuries.",
    "The Colosseum in Rome, Italy, is an ancient amphitheater that could hold up to 80,000 spectators.",
    "Machu Picchu is a 15th-century Inca citadel in the Andes Mountains of Peru.",
    "The Taj Mahal is an ivory-white marble mausoleum in Agra, India, built between 1632 and 1653.",
    "The Statue of Liberty was a gift from France to the United States, dedicated in 1886.",
    "Petra is a historical city in southern Jordan, famous for its rock-cut architecture.",
    "Christ the Redeemer is a 30-meter Art Deco statue of Jesus in Rio de Janeiro, Brazil.",
    "The Sydney Opera House opened in 1973 and is one of the most distinctive buildings of the 20th century.",
    "Angkor Wat in Cambodia is the largest religious monument in the world, spanning 400 acres.",
]

# Step 3: Build the retriever: this creates a FAISS index automatically
search = dspy.retrievers.Embeddings(
    embedder=embedder,
    corpus=corpus,
    k=3,  # Return top-3 passages by default
)

# Step 4: Use the retriever
results = search("Where can I find ancient Roman architecture?")
print("Query: Where can I find ancient Roman architecture?\n")
for i, passage in enumerate(results.passages, 1):
    print(f"  [{i}] {passage}")
```

A few things to note:

- **`dspy.Embedder`** wraps any embedding model supported by LiteLLM. The `"openai/text-embedding-3-small"` string follows the same provider/model format as `dspy.LM`.
- **`dimensions=512`** reduces the embedding dimensionality (OpenAI's text-embedding-3-small supports this natively). Smaller dimensions mean faster search with minimal quality loss.
- **`dspy.retrievers.Embeddings`** builds a FAISS index from your corpus at construction time. Subsequent calls run approximate nearest-neighbor search. Pass `brute_force_threshold=30_000` to avoid requiring `faiss-cpu` for smaller corpora.
- The retriever returns a **`dspy.Prediction`** object with a `.passages` attribute containing the top-k passages sorted by similarity.

### Loading Corpus from Files

For real projects, your corpus won't be a hardcoded list. Here's how to load documents from text files and chunk them:

```python
from pathlib import Path


def load_corpus(directory: str, chunk_size: int = 200) -> list[str]:
    """Load all .txt files from a directory as chunked corpus passages."""
    corpus = []
    for path in Path(directory).glob("*.txt"):
        text = path.read_text(encoding="utf-8").strip()
        # Split into chunks of ~chunk_size words
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                corpus.append(chunk)
    return corpus


corpus = load_corpus("./documents")
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5)
```

---

## Using Custom and HuggingFace Embedding Models

The examples above use OpenAI's hosted embedding model, but DSPy's `Embedder` class is designed to work with **any embedding model**, including HuggingFace sentence-transformers, your own fine-tuned models, or completely custom embedding functions.

The `dspy.Embedder` constructor accepts two types of `model` argument:

1. **A string**: the name of a hosted model (via LiteLLM), e.g., `"openai/text-embedding-3-small"`
2. **A callable**: any Python function that takes a list of strings and returns a 2D numpy array (or list of lists) of float32 values

This makes it trivially easy to swap in local or custom models.

### HuggingFace Sentence-Transformers

[Sentence-Transformers](https://huggingface.co/models?library=sentence-transformers) is the most popular library for local embedding models. DSPy integrates with it seamlessly by passing `model.encode` as the callable:

```python
# pip install sentence-transformers
import dspy
from sentence_transformers import SentenceTransformer

# Load any sentence-transformers model from HuggingFace
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Create an embedder by passing the model's encode method
embedder = dspy.Embedder(model.encode)

# Use it exactly like the hosted version
embeddings = embedder(["Hello, world!", "DSPy is great!"])
print(f"Embedding shape: {embeddings.shape}")  # (2, 384) for all-MiniLM-L6-v2

# Plug into the Embeddings retriever
corpus = [
    "DSPy is a framework for programming language models.",
    "Sentence transformers produce dense vector representations of text.",
    "FAISS enables efficient similarity search over dense vectors.",
    "RAG combines retrieval with language model generation.",
]
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=2)
results = search("What is DSPy?")
for passage in results.passages:
    print(passage)
```

Some popular HuggingFace sentence-transformer models to try:

| Model | Dimensions | Speed | Use Case |
|-------|-----------|-------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | General-purpose, balanced quality/speed |
| `all-mpnet-base-v2` | 768 | Medium | Higher quality, general-purpose |
| `static-retrieval-mrl-en-v1` | 1024 | Very Fast | CPU-optimized, extremely efficient |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | Fast | Optimized for question-answering retrieval |
| `bge-large-en-v1.5` | 1024 | Slow | State-of-the-art quality (BAAI) |

### Using a Fine-Tuned Embedding Model

If you've fine-tuned your own embedding model (e.g., on domain-specific data), the integration is identical. The only requirement is that your model provides an `encode` method or a callable that converts text to vectors:

```python
import dspy
from sentence_transformers import SentenceTransformer

# Load your fine-tuned model from a local directory
finetuned_model = SentenceTransformer("./my-finetuned-embedder")

# Or load from HuggingFace Hub (your own uploaded model)
# finetuned_model = SentenceTransformer("your-username/your-finetuned-model")

embedder = dspy.Embedder(finetuned_model.encode)

# Everything works the same
corpus = ["Your domain-specific documents..."]
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5)
results = search("domain-specific query")
```

### Using a Custom Embedding Function

For maximum flexibility, you can pass any callable that takes a list of strings and returns embeddings. This is useful when you have a custom model loaded via the `transformers` library directly, or when you need custom preprocessing:

```python
import dspy
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


def custom_embed(texts: list[str]) -> np.ndarray:
    """Custom embedding function using a HuggingFace transformers model."""
    tokenizer = AutoTokenizer.from_pretrained("your-model-name")
    model = AutoModel.from_pretrained("your-model-name")

    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**encoded)

    # Mean pooling over token embeddings
    attention_mask = encoded["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

    return embeddings.numpy()


# Use the custom function as an embedder
embedder = dspy.Embedder(custom_embed)

corpus = ["Your documents here..."]
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5)
```

> **Tip**: When using a custom callable with `dspy.Embedder`, caching of embedding responses is disabled (caching only works with hosted LiteLLM models). If you need caching for a local model, implement it in your custom function.

### Choosing Between Hosted and Local Models

| Factor | Hosted (e.g., OpenAI) | Local (e.g., Sentence-Transformers) |
|--------|----------------------|--------------------------------------|
| **Setup** | Just an API key | Install `sentence-transformers` + download model |
| **Latency** | Network round-trip | Local inference (faster for small batches) |
| **Cost** | Pay per token | Free after model download |
| **Privacy** | Data sent to API | Data stays local |
| **Quality** | Generally high | Varies by model; fine-tuned models can exceed hosted |
| **Offline support** | No | Yes |

For production systems, local embedding models are often preferred because they eliminate API costs and latency for the embedding step, while still using hosted LLMs for generation.

---

## ColBERTv2 Retriever

If you have a ColBERTv2 server running (for example, the publicly available Wikipedia abstracts endpoint), DSPy connects to it with a single line:

```python
import dspy

# Connect to a ColBERTv2 endpoint over HTTP
colbert = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")

# Search: returns objects with .long_text, .score, and .pid
results = colbert("What is the tallest mountain in the world?", k=5)
for r in results:
    print(r.long_text)
```

ColBERTv2 is particularly useful for:

- **Large-scale corpora** (Wikipedia, PubMed) where local FAISS would be impractical
- **Benchmarking** against standard retrieval baselines
- **Multi-hop RAG** where you need fast, high-quality retrieval at each hop

---

## Custom Retriever Functions

DSPy's most powerful retrieval pattern is also the simplest: **any Python callable that takes a query and returns passages works as a retriever.** This means you can wrap any search API, database, or custom logic.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


def my_custom_search(query: str, k: int = 5) -> list[str]:
    """
    A custom retriever function.
    DSPy just needs a callable that returns a list of strings.
    """
    corpus = [
        "Python was created by Guido van Rossum and first released in 1991.",
        "JavaScript is the most popular programming language for web development.",
        "Rust emphasizes memory safety without garbage collection.",
        "Go was designed at Google and is known for its simplicity and concurrency support.",
    ]
    # Simple keyword matching (in production, use proper vector similarity)
    scored = [
        (doc, sum(1 for w in query.lower().split() if w in doc.lower()))
        for doc in corpus
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored[:k]]


# Use it directly
results = my_custom_search("Who created Python?")
print(results[0])  # "Python was created by Guido van Rossum..."
```

This pattern is the universal bridge to external vector databases.

---

## Saving and Loading Embeddings

For large corpora, computing embeddings can be expensive and time-consuming. DSPy's `Embeddings` retriever supports **saving and loading** the index to disk so you only compute embeddings once:

```python
import dspy

embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)

# Build the index (expensive for large corpora)
corpus = ["doc1...", "doc2...", "doc3..."]
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5)

# Save the index to disk
search.save("./saved_embeddings")

# Load later without recomputing embeddings (recommended approach)
loaded_search = dspy.retrievers.Embeddings.from_saved("./saved_embeddings", embedder)
results = loaded_search("your query")
```

The `from_saved()` class method is the **recommended** way to load saved embeddings, as it creates a new instance without unnecessarily computing anything. You can also use `load()` on an existing instance for method chaining.

---

## External Vector DB Integration

Here's how you'd wrap popular vector databases as DSPy-compatible retriever functions. The pattern is always the same: **wrap your external search in a function that takes a query string and returns a list of passage strings.**

### ChromaDB

```python
import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection("my_docs")

# Add documents (do this once)
collection.add(
    documents=["Document 1 text...", "Document 2 text..."],
    ids=["doc1", "doc2"],
)


def search_chromadb(query: str, k: int = 5) -> list[str]:
    """ChromaDB retriever function."""
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]
```

### Pinecone

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.Index("my-index")

embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)


def search_pinecone(query: str, k: int = 5) -> list[str]:
    """Pinecone retriever function."""
    query_embedding = embedder([query])[0]
    results = index.query(vector=query_embedding, top_k=k, include_metadata=True)
    return [match.metadata["text"] for match in results.matches]
```

### Weaviate

```python
import weaviate

client = weaviate.connect_to_local()
collection = client.collections.get("Documents")


def search_weaviate(query: str, k: int = 5) -> list[str]:
    """Weaviate retriever function."""
    response = collection.query.near_text(query=query, limit=k)
    return [obj.properties["text"] for obj in response.objects]
```

DSPy does not care what happens inside your retriever. It just needs the interface. This means you can also build **hybrid retrievers** that combine keyword search (BM25) with vector similarity, or retrievers that query multiple sources and merge results.

---

## Key Takeaways

- **DSPy treats retrieval as a callable**: any function that returns passages works as a retriever, making it composable and swappable.
- **`dspy.Embedder`** supports both hosted models (via LiteLLM string format) and local models (via any callable, e.g., `SentenceTransformer.encode`).
- **HuggingFace and fine-tuned models** integrate natively: pass `model.encode` to `dspy.Embedder` and everything works out of the box.
- **`dspy.retrievers.Embeddings`** gives you local FAISS-backed vector search with zero infrastructure. Use `save()` and `from_saved()` to persist indexes.
- **`dspy.ColBERTv2`** connects to hosted ColBERT endpoints for large-scale retrieval.
- **Custom retriever functions** are the universal adapter: wrap any vector DB, search API, or hybrid system.
- **Retrieval participates in optimization**: when you optimize a RAG pipeline, DSPy tunes the queries and prompts around the retriever.

---

## Next Up

With retrieval wired up, it's time to build the most important LM pattern in production today: **Retrieval-Augmented Generation.** We'll build a complete RAG pipeline as a `dspy.Module`, evaluate it, and optimize it.

**[5.2: Building RAG Pipelines →](../5.2-building-rag/blog.md)**

---

## Resources

- 📖 [DSPy Embedder API](https://dspy.ai/api/models/Embedder/)
- 📖 [DSPy Embeddings Retriever API](https://dspy.ai/api/tools/Embeddings/)
- 📖 [DSPy RAG Tutorial](https://dspy.ai/tutorials/rag/)
- 📖 [Sentence-Transformers Models](https://huggingface.co/models?library=sentence-transformers)
- 📖 [FAISS Documentation](https://github.com/facebookresearch/faiss)
- 📖 [ColBERTv2 Paper](https://arxiv.org/abs/2112.01488)
- 💻 [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
