# 8.3: RLM Production Patterns

## Introduction

In [8.1](../8.1-understanding-rlm/blog.md), you learned what Recursive Language Models are. In [8.2](../8.2-building-with-rlm/blog.md), you built programs with `dspy.RLM`. In [8.P](../8.P-project-document-analyzer/blog.md), you built a document analyzer. But taking RLM to production introduces challenges that tutorials don't cover: **how do you chunk 500-page documents efficiently? How do you handle failures mid-recursion? How do you monitor token usage across recursive depths?**

This post bridges the gap between tutorial RLM and production RLM. You'll learn the architectural patterns, error handling strategies, and monitoring approaches that make recursive processing reliable at scale.

---

## What You'll Learn

- Production-grade chunking strategies: fixed-size, semantic, sliding window
- State management across recursive iterations
- Error handling and retry patterns for long-running RLM pipelines
- Memory-efficient processing for documents exceeding 500 pages
- Monitoring recursive depth, token usage, and processing time
- A real-world case study: processing a legal contract corpus

---

## Prerequisites

- Completed [8.1: Understanding RLM](../8.1-understanding-rlm/blog.md), [8.2: Building with RLM](../8.2-building-with-rlm/blog.md), and [8.P: Document Analyzer](../8.P-project-document-analyzer/blog.md)
- Deno installed for the sandboxed interpreter
- Familiarity with `dspy.RLM` and its REPL-based execution model

---

## Pattern 1: Chunking Strategies

The first production challenge is **how to break large documents into chunks** that an LM can process. The choice of strategy directly impacts output quality.

### Fixed-Size Chunking

The simplest approach. Split by token count with overlap:

```python
def fixed_size_chunks(text, chunk_size=3000, overlap=200):
    """Split text into fixed-size chunks with overlap.

    Args:
        text: The full document text
        chunk_size: Maximum tokens per chunk (approximate, using words as proxy)
        overlap: Number of words to overlap between chunks for context continuity
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append({
            "text": chunk,
            "start_idx": start,
            "end_idx": end,
            "chunk_id": len(chunks),
        })
        start = end - overlap  # Overlap for context continuity

    return chunks

# Usage
document = open("large_document.txt").read()
chunks = fixed_size_chunks(document, chunk_size=2000, overlap=150)
print(f"Document split into {len(chunks)} chunks")
```

**When to use:** Quick prototyping, uniformly structured documents (logs, transcripts), when you need predictable processing time per chunk.

### Semantic Chunking

Split at natural boundaries (paragraphs, sections, headings):

```python
import re


def semantic_chunks(text, max_chunk_size=3000):
    """Split text at semantic boundaries (headings, paragraphs).

    Respects document structure by splitting at headings and double newlines,
    then merging small sections until they approach max_chunk_size.
    """
    # Split at headings and double newlines
    sections = re.split(r"\n(?=#{1,4}\s)|(?:\n\n)", text)
    sections = [s.strip() for s in sections if s.strip()]

    chunks = []
    current_chunk = []
    current_size = 0

    for section in sections:
        section_size = len(section.split())

        if current_size + section_size > max_chunk_size and current_chunk:
            chunks.append({
                "text": "\n\n".join(current_chunk),
                "chunk_id": len(chunks),
                "sections": len(current_chunk),
            })
            current_chunk = [section]
            current_size = section_size
        else:
            current_chunk.append(section)
            current_size += section_size

    if current_chunk:
        chunks.append({
            "text": "\n\n".join(current_chunk),
            "chunk_id": len(chunks),
            "sections": len(current_chunk),
        })

    return chunks
```

**When to use:** Structured documents (contracts, papers, manuals), when semantic coherence within chunks matters more than uniform size.

### Sliding Window with Context

Each chunk includes context from the previous chunk's summary:

```python
import dspy


class SlidingWindowProcessor(dspy.Module):
    """Process chunks with a sliding context window.

    Each chunk receives a summary of all previous chunks as context,
    enabling the model to maintain awareness of the full document.
    """

    def __init__(self):
        self.summarize = dspy.ChainOfThought(
            "text, previous_context -> summary: str, key_findings: str"
        )
        self.process = dspy.ChainOfThought(
            "text, running_context -> analysis: str, entities: list[str]"
        )

    def forward(self, chunks: list[dict]):
        results = []
        running_context = ""

        for chunk in chunks:
            # Process current chunk with accumulated context
            result = self.process(
                text=chunk["text"],
                running_context=running_context,
            )
            results.append(result)

            # Update running context with a summary (keeps it bounded)
            summary = self.summarize(
                text=chunk["text"],
                previous_context=running_context,
            )
            running_context = summary.summary  # Compressed context

        return results
```

**When to use:** Documents where later content depends on earlier content (narratives, legal arguments, research papers with forward references).

---

## Pattern 2: Error Handling and Retry

Long-running RLM pipelines will encounter failures. Build resilience from the start:

```python
import time
import logging

logger = logging.getLogger(__name__)


class ResilientRLMPipeline(dspy.Module):
    """RLM pipeline with retry logic and checkpoint recovery."""

    def __init__(self, max_retries=3, retry_delay=2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.analyze = dspy.ChainOfThought(
            "text, context -> analysis: str, confidence: float"
        )
        self.checkpoints = {}  # chunk_id -> result

    def process_chunk(self, chunk, context=""):
        """Process a single chunk with retry logic."""
        chunk_id = chunk["chunk_id"]

        # Check for existing checkpoint
        if chunk_id in self.checkpoints:
            logger.info(f"Chunk {chunk_id}: restored from checkpoint")
            return self.checkpoints[chunk_id]

        for attempt in range(self.max_retries):
            try:
                result = self.analyze(
                    text=chunk["text"],
                    context=context,
                )
                # Save checkpoint
                self.checkpoints[chunk_id] = result
                return result

            except Exception as e:
                logger.warning(
                    f"Chunk {chunk_id}, attempt {attempt + 1}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Chunk {chunk_id}: all retries exhausted")
                    raise

    def forward(self, chunks: list[dict]):
        results = []
        context = ""

        for chunk in chunks:
            result = self.process_chunk(chunk, context=context)
            results.append(result)
            context = result.analysis[:500]  # Carry forward truncated context

        return results
```

Key patterns:
- **Checkpointing:** Save each chunk's result so you can resume after failure
- **Exponential backoff:** Wait longer between retries to avoid rate limits
- **Bounded context:** Truncate the running context to prevent token overflow

---

## Pattern 3: Memory-Efficient Processing

For very large documents (500+ pages), process chunks in a streaming fashion to limit memory usage:

```python
class StreamingDocumentProcessor(dspy.Module):
    """Process arbitrarily large documents without loading everything into memory."""

    def __init__(self, chunk_size=2000):
        self.chunk_size = chunk_size
        self.extract = dspy.ChainOfThought(
            "text -> themes: list[str], entities: list[str], summary: str"
        )
        self.merge = dspy.ChainOfThought(
            "summaries: str, all_themes: str -> final_summary: str, top_themes: list[str]"
        )

    def stream_chunks(self, filepath):
        """Yield chunks from a file without loading the entire file."""
        buffer = []
        buffer_size = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                words = line.split()
                buffer.extend(words)
                buffer_size += len(words)

                if buffer_size >= self.chunk_size:
                    yield {
                        "text": " ".join(buffer[:self.chunk_size]),
                        "chunk_id": hash(" ".join(buffer[:10])),
                    }
                    buffer = buffer[self.chunk_size:]
                    buffer_size = len(buffer)

        if buffer:
            yield {"text": " ".join(buffer), "chunk_id": hash(" ".join(buffer[:10]))}

    def forward(self, filepath: str):
        all_summaries = []
        all_themes = set()

        for chunk in self.stream_chunks(filepath):
            result = self.extract(text=chunk["text"])
            all_summaries.append(result.summary)
            all_themes.update(result.themes)

        # Final merge
        merged = self.merge(
            summaries="\n".join(all_summaries),
            all_themes=", ".join(all_themes),
        )

        return merged
```

---

## Pattern 4: Monitoring Recursive Processing

Track every aspect of RLM execution for production visibility:

```python
import time
from dataclasses import dataclass, field


@dataclass
class RLMMetrics:
    """Track metrics across recursive processing."""

    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    chunk_times: list = field(default_factory=list)
    depth_history: list = field(default_factory=list)

    @property
    def avg_chunk_time(self):
        return sum(self.chunk_times) / len(self.chunk_times) if self.chunk_times else 0

    @property
    def estimated_remaining(self):
        remaining = self.total_chunks - self.processed_chunks
        return remaining * self.avg_chunk_time

    def report(self):
        print(f"\n--- RLM Processing Report ---")
        print(f"Chunks: {self.processed_chunks}/{self.total_chunks} "
              f"({self.failed_chunks} failed)")
        print(f"Total time: {self.total_time_seconds:.1f}s")
        print(f"Avg chunk time: {self.avg_chunk_time:.1f}s")
        print(f"Est. remaining: {self.estimated_remaining:.0f}s")
        if self.total_tokens:
            print(f"Total tokens: {self.total_tokens:,}")


class MonitoredPipeline(dspy.Module):
    """RLM pipeline with built-in metrics collection."""

    def __init__(self):
        self.analyze = dspy.ChainOfThought(
            "text, context -> analysis: str"
        )
        self.metrics = RLMMetrics()

    def forward(self, chunks: list[dict]):
        self.metrics.total_chunks = len(chunks)
        start = time.perf_counter()
        results = []
        context = ""

        for chunk in chunks:
            chunk_start = time.perf_counter()
            try:
                result = self.analyze(text=chunk["text"], context=context)
                results.append(result)
                context = result.analysis[:500]
                self.metrics.processed_chunks += 1
            except Exception:
                self.metrics.failed_chunks += 1

            self.metrics.chunk_times.append(time.perf_counter() - chunk_start)

            # Progress update every 10 chunks
            if self.metrics.processed_chunks % 10 == 0:
                pct = self.metrics.processed_chunks / self.metrics.total_chunks * 100
                est = self.metrics.estimated_remaining
                print(f"  Progress: {pct:.0f}% | ETA: {est:.0f}s")

        self.metrics.total_time_seconds = time.perf_counter() - start

        # Usage tracking
        lm = dspy.settings.lm
        usage = lm.history if hasattr(lm, "history") else []
        self.metrics.total_tokens = sum(
            entry.get("usage", {}).get("total_tokens", 0)
            for entry in usage
        ) if usage else 0

        self.metrics.report()
        return results
```

---

## Case Study: Legal Contract Corpus

Here's how these patterns combine for a real-world task - processing a corpus of 50 legal contracts (averaging 30 pages each):

```python
import json
from pathlib import Path


class LegalContractAnalyzer(dspy.Module):
    """Production-grade legal contract analysis using RLM patterns."""

    def __init__(self):
        self.extract_clauses = dspy.ChainOfThought(
            "contract_text, context -> clauses: list[str], risks: list[str], obligations: list[str]"
        )
        self.summarize = dspy.ChainOfThought(
            "clauses: str, risks: str, obligations: str -> executive_summary: str, risk_score: float"
        )
        self.metrics = RLMMetrics()

    def analyze_contract(self, filepath: str):
        """Analyze a single contract using semantic chunking with sliding context."""
        text = Path(filepath).read_text(encoding="utf-8")
        chunks = semantic_chunks(text, max_chunk_size=2500)

        all_clauses = []
        all_risks = []
        all_obligations = []
        context = ""

        for chunk in chunks:
            chunk_start = time.perf_counter()
            result = self.extract_clauses(
                contract_text=chunk["text"],
                context=context,
            )
            all_clauses.extend(result.clauses)
            all_risks.extend(result.risks)
            all_obligations.extend(result.obligations)
            context = f"Clauses so far: {len(all_clauses)}, Risks: {len(all_risks)}"

            self.metrics.chunk_times.append(time.perf_counter() - chunk_start)
            self.metrics.processed_chunks += 1

        # Final synthesis
        summary = self.summarize(
            clauses=json.dumps(all_clauses),
            risks=json.dumps(all_risks),
            obligations=json.dumps(all_obligations),
        )

        return {
            "file": filepath,
            "clauses": all_clauses,
            "risks": all_risks,
            "obligations": all_obligations,
            "executive_summary": summary.executive_summary,
            "risk_score": summary.risk_score,
        }

    def forward(self, contract_dir: str):
        """Process all contracts in a directory."""
        contracts = list(Path(contract_dir).glob("*.txt"))
        self.metrics.total_chunks = len(contracts) * 15  # estimated ~15 chunks per contract

        results = []
        for contract in contracts:
            print(f"Processing: {contract.name}")
            result = self.analyze_contract(str(contract))
            results.append(result)

        self.metrics.report()
        return results


# Usage
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

analyzer = LegalContractAnalyzer()
results = analyzer(contract_dir="contracts/")

# Export results
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Key Takeaways

- **Choose chunking strategy based on document structure.** Fixed-size for uniform data, semantic for structured documents, sliding window when context carries across sections.
- **Build retry and checkpoint logic from day one.** Long-running RLM pipelines will fail. Checkpointing lets you resume instead of restart.
- **Stream large documents** instead of loading everything into memory. Process chunk-by-chunk and merge at the end.
- **Monitor everything.** Track chunk processing time, token usage, failure rates, and estimated completion time. Without monitoring, you're flying blind on a 2-hour processing job.
- **Combine patterns for production.** The legal contract case study uses semantic chunking + sliding context + monitoring + error handling together.

---

## Next Up

You've mastered RLM for large-context processing. Next, we'll explore a different kind of optimization: using reinforcement learning to train DSPy program weights.

**[Phase 9: RL Optimization →](../../09-rl-optimization/9.1-rl-for-dspy/blog.md)**

---

## Resources

- [DSPy RLM Documentation](https://dspy.ai/api/modules/RLM/)
- [Deno Installation Guide](https://docs.deno.com/runtime/getting_started/installation/)
- [Code examples for this post](code/)
