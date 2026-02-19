# 11.4: Debugging and Observability

## Introduction

DSPy programs are powerful because they're abstract: you declare *what* you want, and DSPy figures out *how* to prompt the LM. But abstraction has a cost. When something goes wrong, you need to see through the abstraction to understand what actually happened. What prompt was sent? What did the LM return? Why did the adapter fail to parse the response? Why is the cache returning stale data?

This post covers every debugging and observability tool DSPy provides, from quick inspection to full custom observability systems.

---

## What You'll Learn

- Inspecting LM call history with `dspy.inspect_history()`
- Enabling DSPy and LiteLLM logging
- Tracking token usage and costs
- Debugging common issues: adapter failures, cache problems, assertion errors
- Building custom observability wrappers
- Production monitoring patterns

---

## Prerequisites

- Completed [11.3: Deploying DSPy Applications](../11.3-deployment/blog.md)
- DSPy installed (`uv add dspy python-dotenv`)
- An OpenAI API key (or any LiteLLM-supported provider)

---

## Inspecting LM History

Your first debugging tool is `dspy.inspect_history()`. It shows you the raw prompts and responses for recent LM calls, letting you see exactly what DSPy sent to the provider and what came back:

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question -> answer")
result = predict(question="What is gravity?")

# See the last LM call: full prompt and response
dspy.inspect_history(n=1)
```

This prints the complete prompt (system message, user message, all fields) and the raw response. It is invaluable for understanding:
- What DSPy actually sent to the LM
- How the adapter formatted the prompt
- What the LM returned before parsing
- Whether few-shot demos were included

For multi-step pipelines, use `n` to see more calls:

```python
# See the last 5 LM calls, useful for multi-step pipelines
dspy.inspect_history(n=5)
```

---

## Logging

### DSPy Internal Logging

Enable DSPy's internal logging to see module execution details, adapter behavior, and internal state changes:

```python
# Enable DSPy's internal logging: shows module execution, adapter behavior
dspy.enable_logging()

# Run your program and observe the detailed logs
result = predict(question="What is gravity?")

# Disable when done
dspy.disable_logging()
```

### LiteLLM Logging

For provider-level debugging, enable LiteLLM logging. This shows raw HTTP requests to LM providers:

```python
# Enable LiteLLM logging: shows raw HTTP requests to providers
dspy.enable_litellm_logging()

# Run your program and observe API-level details
result = predict(question="What is gravity?")
```

LiteLLM logging is more verbose. It shows the raw HTTP requests and responses to or from providers, including headers, retry attempts, and error details. This is useful when debugging provider-specific issues like rate limits or authentication errors.

### Combining Both

For maximum visibility during debugging:

```python
# Enable everything
dspy.enable_logging()
dspy.enable_litellm_logging()

# Run the problematic code
result = predict(question="Why is this failing?")

# Disable again for clean output
dspy.disable_logging()
```

---

## Usage Tracking

Token usage tracking lets you monitor exactly how many tokens each call consumes, which is essential for cost management:

```python
# Enable usage tracking globally
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)

predict = dspy.Predict("question -> answer")
result = predict(question="Explain quantum entanglement")

# Get per-model usage breakdown
usage = result.get_lm_usage()
print(usage)
# {'openai/gpt-4o-mini': {'prompt_tokens': 42, 'completion_tokens': 150}}
```

For multi-step pipelines, `get_lm_usage()` aggregates across all LM calls made during the prediction:

```python
class Pipeline(dspy.Module):
    def __init__(self):
        self.step1 = dspy.Predict("question -> analysis")
        self.step2 = dspy.ChainOfThought("analysis -> answer")

    def forward(self, question, **kwargs):
        analysis = self.step1(question=question)
        return self.step2(analysis=analysis.analysis)


pipeline = Pipeline()
result = pipeline(question="What is dark matter?")
usage = result.get_lm_usage()
print(f"Total usage across all steps: {usage}")
```

---

## Debugging Common Issues

### Adapter Parsing Failures

When the LM returns output that doesn't match the expected format:

```python
predict = dspy.Predict("question -> answer: int")
try:
    result = predict(question="What is the meaning of life?")
except Exception as e:
    print(f"Error: {e}")
    # Inspect what was actually returned
    dspy.inspect_history(n=1)
```

Common causes and fixes:
- **Model returning free-form text** instead of structured output: try `dspy.ChainOfThought` or add field descriptions
- **Field names in output don't match signature**: check your signature definition
- **Model struggles with complex output types**: consider using `dspy.TypedPredictor` for strict schema enforcement

### Cache Debugging

When cache returns unexpected results, perhaps from a previous version of your program or with different settings, disable it temporarily:

```python
# Disable cache entirely for debugging
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

# Now every call hits the LM provider fresh
result = predict(question="Fresh call, no cache")

# Re-enable when done
dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
)
```

### Assertion Debugging

When `dspy.Assert` or `dspy.Suggest` fail, use inspect_history to see the LM's output before validation:

```python
class ValidatedQA(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question, **kwargs):
        result = self.predict(question=question)
        dspy.Assert(
            len(result.answer) > 50,
            "Answer must be at least 50 characters long",
        )
        return result

try:
    qa = ValidatedQA()
    result = qa(question="What is AI?")
except dspy.DSPyAssertionError as e:
    print(f"Assertion failed: {e}")
    dspy.inspect_history(n=1)  # See what the LM produced
```

---

## Building a Custom Observability Wrapper

For production systems, you may want structured logging of every LM call with performance metrics. Here's a pattern for wrapping a DSPy module with observability:

```python
import time
import logging

logger = logging.getLogger("dspy_monitor")


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

            # Log structured data
            logger.info(
                f"[{self.name}] "
                f"call={self.call_count} "
                f"latency={elapsed:.3f}s "
                f"avg={self.total_time / self.call_count:.3f}s"
            )
            return result
        except Exception as e:
            self.error_count += 1
            elapsed = time.time() - start
            logger.error(
                f"[{self.name}] "
                f"call={self.call_count} "
                f"error={type(e).__name__}: {e} "
                f"latency={elapsed:.3f}s"
            )
            raise

    async def aforward(self, **kwargs):
        self.call_count += 1
        start = time.time()

        try:
            result = await self.module.acall(**kwargs)
            elapsed = time.time() - start
            self.total_time += elapsed
            logger.info(
                f"[{self.name}] "
                f"async call={self.call_count} "
                f"latency={elapsed:.3f}s"
            )
            return result
        except Exception as e:
            self.error_count += 1
            logger.error(
                f"[{self.name}] "
                f"async error={type(e).__name__}: {e}"
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
```

Usage:

```python
# Wrap any module with monitoring
predict = dspy.Predict("question -> answer")
monitored = MonitoredModule(predict, name="qa_predict")

# Use it normally
result = monitored(question="What is machine learning?")
print(result.answer)

# Check stats
print(monitored.stats())
# {'name': 'qa_predict', 'calls': 1, 'errors': 0, 'total_time': 2.451, ...}
```

---

## Debugging Workflow

Here's a systematic approach for debugging DSPy issues:

### Step 1: Check the Basics

```python
# Verify your LM is configured correctly
print(dspy.settings.lm)  # Should show your model

# Make a simple test call
predict = dspy.Predict("question -> answer")
result = predict(question="Say hello")
print(result.answer)
```

### Step 2: Inspect History

```python
# See what actually went to/from the LM
dspy.inspect_history(n=1)
```

### Step 3: Enable Logging

```python
# Get module-level execution details
dspy.enable_logging()
result = predict(question="Debug this")
dspy.disable_logging()
```

### Step 4: Disable Cache

```python
# Rule out stale cached responses
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
result = predict(question="Fresh response")
```

### Step 5: Check Usage

```python
# Verify tokens are being consumed
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)
result = predict(question="Check usage")
print(result.get_lm_usage())
```

---

## Production Monitoring Patterns

### Structured Logging

Use structured logging for production observability:

```python
import json
import logging

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_entry.update(record.extra)
        return json.dumps(log_entry)

# Apply to logger
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger("dspy_production")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### Health Check with Diagnostics

Extend your health check to include diagnostic information:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": str(dspy.settings.lm),
        "cache": {
            "disk_enabled": True,
            "memory_enabled": True,
        },
    }

@app.get("/diagnostics")
async def diagnostics():
    # Quick test call
    try:
        predict = dspy.Predict("question -> answer")
        result = predict(question="Diagnostic test")
        return {
            "status": "operational",
            "lm_responsive": True,
            "sample_answer": result.answer[:50],
        }
    except Exception as e:
        return {
            "status": "degraded",
            "lm_responsive": False,
            "error": str(e),
        }
```

---

## Key Takeaways

1. **`dspy.inspect_history(n=5)`**: see what was actually sent and received
2. **`dspy.enable_logging()`**: get module-level execution details
3. **Disable cache**: ensure you're getting fresh responses when debugging
4. **Check usage**: verify tokens are being consumed as expected
5. **Wrap modules with monitoring**: track latency, errors, and call counts
6. **Structured logging**: JSON-formatted logs for production systems
7. **Systematic debugging**: follow the 5-step workflow: basics, history, logging, cache, usage

---

**Next up:** [11.P: Project - Production-Ready DSPy API](../11.P-project-production-api/blog.md) ties everything together in a complete, deployable production project.

---

## Resources

- [DSPy Debugging Documentation](https://dspy.ai/learn/debugging/)
- [DSPy Assertions/Suggestions](https://dspy.ai/learn/assertions/)
- [Python Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html)
