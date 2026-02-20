"""
Blog 11.1: Caching and Performance Optimization
Requires: pip install -U dspy python-dotenv
Requires: OpenAI API key
"""

import hashlib
import json
import threading
import time

import dspy
from dotenv import load_dotenv
from dspy.clients import Cache

load_dotenv()


# =====================================================
# Section 1: Cache Demo (Zero Configuration)
# =====================================================

def demo_automatic_caching():
    """Demonstrate DSPy's automatic 3-layer caching."""
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)

    predict = dspy.Predict("question -> answer")

    # First call: hits the LM provider
    start = time.time()
    result1 = predict(question="Who is the GOAT?")
    elapsed1 = time.time() - start
    print(f"First call:  {elapsed1:.4f}s")
    print(f"Answer: {result1.answer}")
    print(f"Usage: {result1.get_lm_usage()}")

    # Second call: served from cache
    start = time.time()
    result2 = predict(question="Who is the GOAT?")
    elapsed2 = time.time() - start
    print(f"\nSecond call: {elapsed2:.6f}s")
    print(f"Answer: {result2.answer}")
    print(f"Usage: {result2.get_lm_usage()}")

    print(f"\nSpeedup: {elapsed1 / elapsed2:.0f}x faster")


# =====================================================
# Section 2: Configuring Cache
# =====================================================

def demo_cache_configuration():
    """Show different cache configurations for various environments."""

    # High-throughput API server
    dspy.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_size_limit_bytes=10_000_000_000,  # 10 GB
        memory_max_entries=5_000_000,
    )
    print("Configured: high-throughput (10GB disk, 5M memory entries)")

    # Debug mode: disable caching entirely
    dspy.configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=False,
    )
    print("Configured: debug mode (no caching)")

    # Memory-constrained: disk only
    dspy.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_size_limit_bytes=500_000_000,  # 500 MB
    )
    print("Configured: memory-constrained (500MB disk only)")

    # Reset to defaults
    dspy.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
    )
    print("Reset to defaults")


# =====================================================
# Section 3: Custom Cache Implementation
# =====================================================

class ContentOnlyCache(Cache):
    """Cache that keys on message content only, ignoring model name.

    Useful for testing and development where you want to reuse
    cached responses across different models.
    """

    def cache_key(self, request):
        messages = request.get("messages", [])
        content = json.dumps(
            [m.get("content", "") for m in messages],
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, key):
        return self._memory_cache.get(key)

    def put(self, key, value):
        self._memory_cache[key] = value


def demo_custom_cache():
    """Demonstrate custom cache implementation."""
    dspy.cache = ContentOnlyCache(
        enable_memory_cache=True,
        enable_disk_cache=True,
    )

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    predict = dspy.Predict("question -> answer")
    result = predict(question="What is 2 + 2?")
    print(f"Custom cache result: {result.answer}")


# =====================================================
# Section 4: Save/Load Programs
# =====================================================

def demo_save_load():
    """Demonstrate saving and loading optimized programs."""
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    # Create a simple program
    program = dspy.ChainOfThought("question -> answer")

    # Run it once to populate state
    result = program(question="What is gravity?")
    print(f"Answer: {result.answer}")

    # Save state-only (recommended for production)
    program.save("demo_program_state.json", save_program=False)
    print("Saved program state to demo_program_state.json")

    # Load into fresh instance
    fresh_program = dspy.ChainOfThought("question -> answer")
    fresh_program.load("demo_program_state.json")
    print("Loaded program state successfully")


# =====================================================
# Section 5: Performance Patterns
# =====================================================

def demo_usage_tracking():
    """Track token usage for cost monitoring."""
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)

    predict = dspy.Predict("question -> answer")
    result = predict(question="Explain quantum computing")

    usage = result.get_lm_usage()
    print(f"Usage: {usage}")


def demo_thread_safe_contexts():
    """Use dspy.context() for thread-safe LM routing."""
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    def handle_request(question, model):
        with dspy.context(lm=dspy.LM(model)):
            predict = dspy.Predict("question -> answer")
            result = predict(question=question)
            print(f"[{model}] {question}: {result.answer[:50]}...")

    t1 = threading.Thread(
        target=handle_request,
        args=("What is AI?", "openai/gpt-4o-mini"),
    )
    t2 = threading.Thread(
        target=handle_request,
        args=("What is ML?", "openai/gpt-4o-mini"),
    )
    t1.start()
    t2.start()
    t1.join()
    t2.join()


class SmartRouter(dspy.Module):
    """Route to appropriate model based on question complexity."""

    def __init__(self):
        self.classify = dspy.Predict("question -> complexity: Low or High")
        self.simple_answer = dspy.Predict("question -> answer")
        self.complex_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        complexity = self.classify(question=question).complexity
        if "Low" in complexity:
            with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
                return self.simple_answer(question=question)
        else:
            with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
                return self.complex_answer(question=question)


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Blog 11.1: Caching and Performance Optimization")
    print("=" * 60)

    print("\n--- 1. Automatic Caching Demo ---")
    demo_automatic_caching()

    print("\n--- 2. Cache Configuration ---")
    demo_cache_configuration()

    print("\n--- 3. Usage Tracking ---")
    demo_usage_tracking()

    print("\n--- 4. Save/Load ---")
    demo_save_load()

    print("\n--- 5. Smart Router ---")
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    router = SmartRouter()
    result = router(question="What is 2 + 2?")
    print(f"Router answer: {result.answer}")
