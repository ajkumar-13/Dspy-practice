"""
8.3: RLM Production Patterns
==============================
Production-grade patterns for recursive language model processing:
chunking, error handling, memory efficiency, and monitoring.
"""

import logging
import re
import time
from dataclasses import dataclass, field

import dspy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------


def fixed_size_chunks(text, chunk_size=3000, overlap=200):
    """Split text into fixed-size chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(
            {
                "text": " ".join(words[start:end]),
                "start_idx": start,
                "end_idx": end,
                "chunk_id": len(chunks),
            }
        )
        start = end - overlap
    return chunks


def semantic_chunks(text, max_chunk_size=3000):
    """Split text at semantic boundaries (headings, paragraphs)."""
    sections = re.split(r"\n(?=#{1,4}\s)|(?:\n\n)", text)
    sections = [s.strip() for s in sections if s.strip()]

    chunks = []
    current_chunk = []
    current_size = 0

    for section in sections:
        section_size = len(section.split())
        if current_size + section_size > max_chunk_size and current_chunk:
            chunks.append(
                {
                    "text": "\n\n".join(current_chunk),
                    "chunk_id": len(chunks),
                    "sections": len(current_chunk),
                }
            )
            current_chunk = [section]
            current_size = section_size
        else:
            current_chunk.append(section)
            current_size += section_size

    if current_chunk:
        chunks.append(
            {
                "text": "\n\n".join(current_chunk),
                "chunk_id": len(chunks),
                "sections": len(current_chunk),
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------


@dataclass
class RLMMetrics:
    """Track metrics across recursive processing."""

    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    chunk_times: list = field(default_factory=list)

    @property
    def avg_chunk_time(self):
        return sum(self.chunk_times) / len(self.chunk_times) if self.chunk_times else 0

    @property
    def estimated_remaining(self):
        remaining = self.total_chunks - self.processed_chunks
        return remaining * self.avg_chunk_time

    def report(self):
        print("\n--- RLM Processing Report ---")
        print(f"Chunks: {self.processed_chunks}/{self.total_chunks} ({self.failed_chunks} failed)")
        print(f"Total time: {self.total_time_seconds:.1f}s")
        print(f"Avg chunk time: {self.avg_chunk_time:.1f}s")
        print(f"Est. remaining: {self.estimated_remaining:.0f}s")
        if self.total_tokens:
            print(f"Total tokens: {self.total_tokens:,}")


# ---------------------------------------------------------------------------
# Resilient pipeline with retry + checkpointing
# ---------------------------------------------------------------------------


class ResilientRLMPipeline(dspy.Module):
    """RLM pipeline with retry logic and checkpoint recovery."""

    def __init__(self, max_retries=3, retry_delay=2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.analyze = dspy.ChainOfThought("text, context -> analysis: str, confidence: float")
        self.checkpoints = {}
        self.metrics = RLMMetrics()

    def process_chunk(self, chunk, context=""):
        """Process a single chunk with retry logic."""
        chunk_id = chunk["chunk_id"]

        if chunk_id in self.checkpoints:
            logger.info(f"Chunk {chunk_id}: restored from checkpoint")
            return self.checkpoints[chunk_id]

        for attempt in range(self.max_retries):
            try:
                result = self.analyze(text=chunk["text"], context=context)
                self.checkpoints[chunk_id] = result
                return result
            except Exception as e:
                logger.warning(f"Chunk {chunk_id}, attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Chunk {chunk_id}: all retries exhausted")
                    raise

    def forward(self, chunks: list[dict]):
        self.metrics.total_chunks = len(chunks)
        start = time.perf_counter()
        results = []
        context = ""

        for chunk in chunks:
            chunk_start = time.perf_counter()
            try:
                result = self.process_chunk(chunk, context=context)
                results.append(result)
                context = result.analysis[:500]
                self.metrics.processed_chunks += 1
            except Exception:
                self.metrics.failed_chunks += 1

            self.metrics.chunk_times.append(time.perf_counter() - chunk_start)

            if self.metrics.processed_chunks % 10 == 0:
                pct = self.metrics.processed_chunks / self.metrics.total_chunks * 100
                print(f"  Progress: {pct:.0f}% | ETA: {self.metrics.estimated_remaining:.0f}s")

        self.metrics.total_time_seconds = time.perf_counter() - start
        self.metrics.report()
        return results


# ---------------------------------------------------------------------------
# Streaming processor for very large documents
# ---------------------------------------------------------------------------


class StreamingDocumentProcessor(dspy.Module):
    """Process large documents without loading everything into memory."""

    def __init__(self, chunk_size=2000):
        self.chunk_size = chunk_size
        self.extract = dspy.ChainOfThought(
            "text -> themes: list[str], entities: list[str], summary: str"
        )
        self.merge = dspy.ChainOfThought(
            "summaries: str, all_themes: str -> final_summary: str, top_themes: list[str]"
        )

    def stream_chunks(self, filepath):
        """Yield chunks from a file without loading entirely."""
        buffer = []
        buffer_size = 0
        chunk_id = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                words = line.split()
                buffer.extend(words)
                buffer_size += len(words)

                if buffer_size >= self.chunk_size:
                    yield {"text": " ".join(buffer[: self.chunk_size]), "chunk_id": chunk_id}
                    buffer = buffer[self.chunk_size :]
                    buffer_size = len(buffer)
                    chunk_id += 1

        if buffer:
            yield {"text": " ".join(buffer), "chunk_id": chunk_id}

    def forward(self, filepath: str):
        all_summaries = []
        all_themes = set()

        for chunk in self.stream_chunks(filepath):
            result = self.extract(text=chunk["text"])
            all_summaries.append(result.summary)
            all_themes.update(result.themes)

        merged = self.merge(
            summaries="\n".join(all_summaries),
            all_themes=", ".join(all_themes),
        )
        return merged


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    # Demo: process a sample document using fixed-size chunks
    sample_text = " ".join(["This is a sample document."] * 500)
    chunks = fixed_size_chunks(sample_text, chunk_size=100, overlap=10)
    print(f"Created {len(chunks)} chunks from sample text")

    pipeline = ResilientRLMPipeline()
    results = pipeline(chunks[:5])  # Process first 5 for demo
    for i, r in enumerate(results):
        print(f"Chunk {i}: {r.analysis[:80]}...")
