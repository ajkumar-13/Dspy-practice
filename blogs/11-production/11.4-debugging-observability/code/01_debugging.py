"""
Blog 11.4: Debugging and Observability
Requires: pip install -U dspy python-dotenv
Requires: OpenAI API key
"""

import json
import logging
import time

import dspy
from dotenv import load_dotenv

load_dotenv()


# =====================================================
# Section 1: Inspecting LM History
# =====================================================

def demo_inspect_history():
    """Use dspy.inspect_history() to see raw prompts and responses."""
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    predict = dspy.Predict("question -> answer")
    result = predict(question="What is gravity?")
    print(f"Answer: {result.answer}\n")

    # See the last LM call
    print("--- Last LM Call ---")
    dspy.inspect_history(n=1)


# =====================================================
# Section 2: Logging
# =====================================================

def demo_logging():
    """Enable DSPy and LiteLLM logging for deep debugging."""
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    predict = dspy.Predict("question -> answer")

    # DSPy internal logging
    print("--- DSPy Logging ---")
    dspy.enable_logging()
    result = predict(question="What is gravity?")
    dspy.disable_logging()
    print(f"Answer: {result.answer}\n")

    # LiteLLM logging (provider-level)
    print("--- LiteLLM Logging ---")
    dspy.enable_litellm_logging()
    result = predict(question="What is gravity again?")
    print(f"Answer: {result.answer}")


# =====================================================
# Section 3: Usage Tracking
# =====================================================

def demo_usage_tracking():
    """Track token usage for cost monitoring."""
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)

    predict = dspy.Predict("question -> answer")
    result = predict(question="Explain quantum entanglement")

    usage = result.get_lm_usage()
    print(f"Usage: {usage}")

    # Multi-step pipeline usage
    class TwoStepPipeline(dspy.Module):
        def __init__(self):
            self.step1 = dspy.Predict("question -> analysis")
            self.step2 = dspy.ChainOfThought("analysis -> answer")

        def forward(self, question, **kwargs):
            analysis = self.step1(question=question)
            return self.step2(analysis=analysis.analysis)

    pipeline = TwoStepPipeline()
    result = pipeline(question="What is dark matter?")
    print(f"Pipeline usage: {result.get_lm_usage()}")


# =====================================================
# Section 4: Cache Debugging
# =====================================================

def demo_cache_debugging():
    """Demonstrate disabling cache for debugging."""
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)
    predict = dspy.Predict("question -> answer")

    # Normal call (may hit cache)
    result1 = predict(question="What is caching?")
    print(f"Normal call usage: {result1.get_lm_usage()}")

    # Disable cache for fresh response
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    result2 = predict(question="What is caching?")
    print(f"No-cache call usage: {result2.get_lm_usage()}")

    # Re-enable cache
    dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True)
    print("Cache re-enabled")


# =====================================================
# Section 5: Assertion Debugging
# =====================================================

def demo_assertion_debugging():
    """Debug assertion failures."""
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    class ValidatedQA(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict("question -> answer")

        def forward(self, question, **kwargs):
            result = self.predict(question=question)
            dspy.Assert(
                len(result.answer) > 50,
                "Answer must be at least 50 characters",
            )
            return result

    try:
        qa = ValidatedQA()
        result = qa(question="Say hi")
        print(f"Answer: {result.answer}")
    except dspy.DSPyAssertionError as e:
        print(f"Assertion failed: {e}")
        dspy.inspect_history(n=1)


# =====================================================
# Section 6: Monitored Module
# =====================================================

class MonitoredModule(dspy.Module):
    """Wrapper that adds observability to any DSPy module."""

    def __init__(self, module: dspy.Module, name: str = "unnamed"):
        super().__init__()
        self.module = module
        self.name = name
        self.call_count = 0
        self.total_time = 0.0
        self.error_count = 0

    def forward(self, **kwargs):
        self.call_count += 1
        start = time.time()
        try:
            result = self.module(**kwargs)
            elapsed = time.time() - start
            self.total_time += elapsed
            print(
                f"[{self.name}] call={self.call_count} "
                f"latency={elapsed:.3f}s "
                f"avg={self.total_time / self.call_count:.3f}s"
            )
            return result
        except Exception as e:
            self.error_count += 1
            elapsed = time.time() - start
            print(
                f"[{self.name}] call={self.call_count} "
                f"ERROR: {type(e).__name__}: {e} "
                f"latency={elapsed:.3f}s"
            )
            raise

    def stats(self):
        return {
            "name": self.name,
            "calls": self.call_count,
            "errors": self.error_count,
            "total_time": round(self.total_time, 3),
            "avg_time": round(
                self.total_time / max(self.call_count, 1), 3
            ),
            "error_rate": round(
                self.error_count / max(self.call_count, 1), 4
            ),
        }


def demo_monitored_module():
    """Use MonitoredModule for observability."""
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    predict = dspy.Predict("question -> answer")
    monitored = MonitoredModule(predict, name="qa_predict")

    result = monitored(question="What is machine learning?")
    print(f"Answer: {result.answer[:50]}...")
    print(f"Stats: {monitored.stats()}")


# =====================================================
# Section 7: Structured Logging
# =====================================================

class JSONFormatter(logging.Formatter):
    """JSON log formatter for production observability."""

    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(log_entry)


def demo_structured_logging():
    """Set up JSON-structured logging."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    logger = logging.getLogger("dspy_structured")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("DSPy service started")
    logger.info("Model configured: openai/gpt-4o-mini")
    logger.warning("Cache miss rate high")


# =====================================================
# Section 8: Debugging Workflow
# =====================================================

def demo_debugging_workflow():
    """Complete 5-step debugging workflow."""
    print("Step 1: Check basics")
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)
    print(f"LM: {dspy.settings.lm}")

    predict = dspy.Predict("question -> answer")
    result = predict(question="Say hello")
    print(f"Basic test: {result.answer}")

    print("\nStep 2: Inspect history")
    dspy.inspect_history(n=1)

    print("\nStep 3: Enable logging")
    dspy.enable_logging()
    result = predict(question="Debug test")
    dspy.disable_logging()

    print("\nStep 4: Disable cache")
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    result = predict(question="Fresh response test")
    dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True)

    print("\nStep 5: Check usage")
    result = predict(question="Usage test")
    print(f"Usage: {result.get_lm_usage()}")


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Blog 11.4: Debugging and Observability")
    print("=" * 60)

    print("\n--- 1. Inspect History ---")
    demo_inspect_history()

    print("\n--- 2. Usage Tracking ---")
    demo_usage_tracking()

    print("\n--- 3. Monitored Module ---")
    demo_monitored_module()

    print("\n--- 4. Structured Logging ---")
    demo_structured_logging()
