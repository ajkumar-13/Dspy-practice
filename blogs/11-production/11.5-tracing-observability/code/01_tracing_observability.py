"""
11.5: Production Tracing with Langfuse, Phoenix, and OpenTelemetry
===================================================================
Instrument DSPy applications with production-grade observability.
"""

import time

import dspy

# ---------------------------------------------------------------------------
# Option 1: Langfuse Integration
# ---------------------------------------------------------------------------

def langfuse_example():
    """Demonstrate Langfuse tracing for DSPy."""
    import os

    from langfuse import Langfuse

    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-..."
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-..."
    os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

    langfuse = Langfuse()
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    qa = dspy.ChainOfThought("question -> answer")

    def traced_predict(question: str):
        trace = langfuse.trace(
            name="qa-prediction",
            metadata={"model": "gpt-4o-mini"},
        )
        span = trace.span(name="chain-of-thought", input={"question": question})

        try:
            result = qa(question=question)
            span.end(output={"answer": result.answer})

            history = lm.history
            if history:
                last_call = history[-1]
                trace.generation(
                    name="lm-call",
                    model="gpt-4o-mini",
                    input=last_call.get("prompt", ""),
                    output=last_call.get("response", ""),
                    usage={
                        "input": last_call.get("usage", {}).get("prompt_tokens", 0),
                        "output": last_call.get("usage", {}).get("completion_tokens", 0),
                    },
                )

            trace.update(output={"answer": result.answer})
            return result
        except Exception as e:
            span.end(level="ERROR", status_message=str(e))
            raise
        finally:
            langfuse.flush()

    result = traced_predict("What is the CAP theorem?")
    print(f"Answer: {result.answer}")


# ---------------------------------------------------------------------------
# Option 2: Arize Phoenix Integration
# ---------------------------------------------------------------------------

def phoenix_example():
    """Demonstrate Arize Phoenix local tracing for DSPy."""
    import phoenix as px
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    # Launch Phoenix dashboard
    session = px.launch_app()
    print(f"Dashboard: {session.url}")

    # Configure OpenTelemetry
    provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("dspy-app")

    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)
    qa = dspy.ChainOfThought("question -> answer")

    def phoenix_traced_predict(question: str):
        with tracer.start_as_current_span("qa-prediction") as span:
            span.set_attribute("input.question", question)
            result = qa(question=question)
            span.set_attribute("output.answer", result.answer[:200])

            usage = lm.history[-1].get("usage", {}) if lm.history else {}
            span.set_attribute("llm.token_count.prompt", usage.get("prompt_tokens", 0))
            span.set_attribute("llm.token_count.completion", usage.get("completion_tokens", 0))
            return result

    result = phoenix_traced_predict("What is eventual consistency?")
    print(f"Answer: {result.answer}")
    print(f"View traces at: {session.url}")


# ---------------------------------------------------------------------------
# Option 3: Generic OpenTelemetry Middleware
# ---------------------------------------------------------------------------

class InstrumentedModule:
    """Wrapper that adds OpenTelemetry tracing to any DSPy module."""

    def __init__(self, module, name=None):
        from opentelemetry import metrics, trace

        self.module = module
        self.name = name or module.__class__.__name__
        self.tracer = trace.get_tracer("dspy-app")

        meter = metrics.get_meter("dspy-app")
        self.counter = meter.create_counter("dspy.predictions.total")
        self.latency = meter.create_histogram("dspy.prediction.latency_ms")

    def __call__(self, **kwargs):
        with self.tracer.start_as_current_span(f"dspy.{self.name}") as span:
            for key, value in kwargs.items():
                span.set_attribute(f"input.{key}", str(value)[:200])

            start = time.perf_counter()
            try:
                result = self.module(**kwargs)
                latency_ms = (time.perf_counter() - start) * 1000

                span.set_attribute("latency_ms", latency_ms)
                self.counter.add(1, {"module": self.name})
                self.latency.record(latency_ms, {"module": self.name})
                return result
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise


# ---------------------------------------------------------------------------
# Cost estimation utility
# ---------------------------------------------------------------------------

PRICING = {
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
}


def estimate_cost(model, prompt_tokens, completion_tokens):
    """Estimate cost for a single LM call (per 1M tokens pricing)."""
    prices = PRICING.get(model, {"input": 0, "output": 0})
    return (
        prompt_tokens / 1_000_000 * prices["input"]
        + completion_tokens / 1_000_000 * prices["output"]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Run one of the example functions:")
    print("  langfuse_example()   - requires Langfuse API keys")
    print("  phoenix_example()    - requires arize-phoenix installed")
