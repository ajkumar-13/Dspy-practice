# 11.2: Async Programming and Streaming in DSPy

## Introduction

In development, synchronous code is fine: you make an LM call, wait for the result, and move on. But production is different. When your API handles hundreds of concurrent requests, synchronous calls create a bottleneck: each request blocks a thread while waiting for the LM provider to respond. And when users are watching a chat interface, they shouldn't have to wait 10 seconds staring at a blank screen before seeing any output.

DSPy solves both problems natively. **Async programming** lets your server handle many concurrent requests with fewer resources, and **streaming** gives users real-time token-by-token output plus intermediate status updates as your pipeline progresses.

This post covers everything: async modules, async tools, output token streaming, status message streaming, and the sync-vs-async decision framework.

---

## What You'll Learn

- Using `acall()` for async LM calls on all built-in modules
- Implementing `aforward()` for custom async modules
- Async tool usage with `dspy.Tool` and `acall()`
- Sync-to-async conversion for mixed codebases
- Output token streaming with `dspy.streamify()` and `StreamListener`
- Streaming multiple fields and handling ReAct loops
- Synchronous streaming with `async_streaming=False`
- Intermediate status streaming with `StatusMessageProvider`

---

## Prerequisites

- Completed [11.1: Caching and Performance Optimization](../11.1-caching-performance/blog.md)
- DSPy installed (`uv add dspy python-dotenv`)
- Basic familiarity with Python's `async`/`await` syntax
- An OpenAI API key (or any LiteLLM-supported provider)

---

## Async Programming in DSPy

### The Basics: `acall()`

Every built-in DSPy module (`Predict`, `ChainOfThought`, `ReAct`, and others) supports the `acall()` method. It's the async counterpart to calling the module directly:

```python
import asyncio
import dspy
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


async def main():
    predict = dspy.Predict("question -> answer")

    # Async call: non-blocking
    output = await predict.acall(question="What is the capital of France?")
    print(output.answer)


asyncio.run(main())
```

The result is identical to a synchronous call. The difference is that while waiting for the LM provider to respond, the event loop is free to handle other tasks: other API requests, database queries, or additional LM calls.

### Concurrent Async Calls

The real power of async is concurrency. You can fire multiple LM calls simultaneously:

```python
async def answer_questions(questions: list[str]):
    predict = dspy.Predict("question -> answer")

    # Fire all calls concurrently
    tasks = [predict.acall(question=q) for q in questions]
    results = await asyncio.gather(*tasks)

    for q, r in zip(questions, results):
        print(f"Q: {q}\nA: {r.answer}\n")


asyncio.run(answer_questions([
    "What is the speed of light?",
    "Who wrote Hamlet?",
    "What is the largest ocean?",
]))
```

Three LM calls run concurrently rather than sequentially, cutting total wall-clock time to roughly the time of the slowest single call.

### Custom Async Modules: `aforward()`

When building custom modules, implement `aforward()` instead of `forward()` to support async execution:

```python
class AsyncPipeline(dspy.Module):
    def __init__(self):
        self.analyze = dspy.Predict("text -> sentiment, topics")
        self.summarize = dspy.Predict("text, sentiment, topics -> summary")

    async def aforward(self, text, **kwargs):
        # First step: analyze the text
        analysis = await self.analyze.acall(text=text)

        # Second step: summarize with analysis context
        result = await self.summarize.acall(
            text=text,
            sentiment=analysis.sentiment,
            topics=analysis.topics,
        )
        return result


async def main():
    pipeline = AsyncPipeline()
    result = await pipeline.acall(text="DSPy is a framework for programming LMs...")
    print(f"Summary: {result.summary}")

asyncio.run(main())
```

If your custom module has both `forward()` and `aforward()`, DSPy uses the appropriate one depending on whether you call the module synchronously or via `acall()`.

### Async Tools

DSPy tools support async functions natively. Pass an async function to `dspy.Tool`, and use `acall()` to invoke it:

```python
import httpx

async def fetch_weather(city: str) -> str:
    """Fetch current weather for a city."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://wttr.in/{city}?format=3")
        return resp.text

# Wrap async function as a DSPy tool
weather_tool = dspy.Tool(fetch_weather)

# Use it asynchronously
result = await weather_tool.acall(city="Paris")
print(result)
```

### Sync-to-Async Conversion

When working with libraries that mix sync and async code, you may have synchronous tools that you want to use in async ReAct agents. DSPy provides a context flag to handle this automatically:

```python
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

# Allow DSPy to automatically convert sync tools for async execution
with dspy.context(allow_tool_async_sync_conversion=True):
    agent = dspy.ReAct("question -> answer", tools=[calculate])
    result = await agent.acall(question="What is 15 * 23 + 42?")
    print(result.answer)
```

With `allow_tool_async_sync_conversion=True`, DSPy wraps synchronous functions so they can be used in async contexts. The trade-off is that the sync function still blocks its thread, but the agent loop continues running asynchronously around it.

### ReAct with Async

When you call a ReAct agent with `acall()`, it automatically uses async tool execution:

```python
async def search_web(query: str) -> str:
    """Search the web for information."""
    async with httpx.AsyncClient() as client:
        # Simulated search
        return f"Results for: {query}"

agent = dspy.ReAct("question -> answer", tools=[search_web])
result = await agent.acall(question="What is the population of Tokyo?")
```

### When to Use Sync vs Async

| Scenario | Recommendation |
|---|---|
| Prototyping and research | **Sync**: simpler, easier to debug |
| Small scripts and notebooks | **Sync**: less boilerplate |
| High-throughput API servers | **Async**: handle many concurrent requests |
| Async tools (HTTP, databases) | **Async**: avoid blocking the event loop |
| Concurrent LM calls | **Async**: fire multiple calls simultaneously |
| Production web services | **Async**: better resource utilization |

---

## Output Token Streaming

Streaming lets users see LM output as it's generated, token by token, instead of waiting for the complete response. DSPy makes this work with `dspy.streamify()`.

### Basic Streaming Setup

```python
import asyncio
import dspy
from dspy.streaming import StreamListener
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


program = dspy.ChainOfThought("question -> answer")

# Create a stream listener for the "answer" field
listener = StreamListener(signature_field_name="answer")

# Wrap the program with streaming
streaming_program = dspy.streamify(program, stream_listeners=[listener])


async def main():
    output = streaming_program(question="Explain black holes in simple terms")

    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            # Token-by-token output
            print(chunk.chunk, end="", flush=True)
        elif isinstance(chunk, dspy.Prediction):
            # Final complete prediction
            print(f"\n\n--- Final answer ---")
            print(chunk.answer)


asyncio.run(main())
```

### Understanding the Stream Objects

When iterating over the stream, you receive two types of objects:

- **`StreamResponse`**: individual token chunks with metadata:
  - `chunk`: the text fragment
  - `predict_name`: which predictor generated this chunk
  - `signature_field_name`: which output field is being streamed

- **`dspy.Prediction`**: the final complete prediction, emitted after all streaming is done

### Streaming Multiple Fields

You can stream multiple output fields simultaneously by adding multiple listeners:

```python
program = dspy.Predict("article -> title, summary, keywords")

listeners = [
    StreamListener(signature_field_name="title"),
    StreamListener(signature_field_name="summary"),
    StreamListener(signature_field_name="keywords"),
]

streaming_program = dspy.streamify(program, stream_listeners=listeners)

async for chunk in streaming_program(article="..."):
    if isinstance(chunk, dspy.streaming.StreamResponse):
        print(f"[{chunk.signature_field_name}] {chunk.chunk}", end="")
```

### Streaming in ReAct Loops

ReAct agents iterate multiple times, and each iteration produces output for the same fields. Use `allow_reuse=True` on the listener to capture output from every iteration:

```python
agent = dspy.ReAct("question -> answer", tools=[search_tool])

listener = StreamListener(
    signature_field_name="answer",
    allow_reuse=True,  # Capture output from every ReAct iteration
)

streaming_agent = dspy.streamify(agent, stream_listeners=[listener])

async for chunk in streaming_agent(question="Research quantum computing"):
    if isinstance(chunk, dspy.streaming.StreamResponse):
        print(chunk.chunk, end="", flush=True)
```

### Handling Duplicate Field Names

When your pipeline has multiple predictors with the same output field name, use the `predict_name` parameter to disambiguate:

```python
class TwoStepPipeline(dspy.Module):
    def __init__(self):
        self.step1 = dspy.Predict("text -> answer")
        self.step2 = dspy.Predict("draft_answer -> answer")

    def forward(self, text, **kwargs):
        draft = self.step1(text=text)
        return self.step2(draft_answer=draft.answer)

# Target the specific predictor by name
listener = StreamListener(
    signature_field_name="answer",
    predict_name="step2",  # Only stream from the second predictor
)
```

### Synchronous Streaming

If you're not in an async context, use `async_streaming=False` to get a synchronous generator:

```python
streaming_program = dspy.streamify(
    program,
    stream_listeners=[listener],
    async_streaming=False,  # Returns sync generator
)

# Use regular for loop, no async needed
for chunk in streaming_program(question="Explain gravity"):
    if isinstance(chunk, dspy.streaming.StreamResponse):
        print(chunk.chunk, end="", flush=True)
```

---

## Intermediate Status Streaming

Beyond token streaming, DSPy supports **status messages** that tell the user what's happening inside a complex pipeline: "Searching the web...", "Analyzing results...", "Generating summary...". This is invaluable for multi-step agents where the user might otherwise wait minutes without feedback.

### Creating a Status Message Provider

Subclass `dspy.streaming.StatusMessageProvider` and implement the hooks you need:

```python
import asyncio
import dspy
from dspy.streaming import StatusMessageProvider, StreamListener
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


class MyStatusProvider(StatusMessageProvider):
    def lm_start_status_message(self, instance, inputs):
        return f"Thinking..."

    def lm_end_status_message(self, instance, outputs):
        return f"Got response"

    def module_start_status_message(self, instance, inputs):
        module_name = instance.__class__.__name__
        return f"Starting {module_name}..."

    def module_end_status_message(self, instance, outputs):
        module_name = instance.__class__.__name__
        return f"Finished {module_name}"

    def tool_start_status_message(self, instance, inputs):
        return f"Calling tool: {instance.name}..."

    def tool_end_status_message(self, instance, outputs):
        return f"Tool {instance.name} completed"


program = dspy.ChainOfThought("question -> answer")
listener = StreamListener(signature_field_name="answer")

streaming_program = dspy.streamify(
    program,
    stream_listeners=[listener],
    status_message_provider=MyStatusProvider(),
)


async def main():
    output = streaming_program(question="What causes tides?")

    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StatusMessage):
            print(f"\n[STATUS] {chunk.message}")
        elif isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk.chunk, end="", flush=True)
        elif isinstance(chunk, dspy.Prediction):
            print(f"\n\nFinal: {chunk.answer}")


asyncio.run(main())
```

### Available Status Hooks

| Hook | When it fires |
|---|---|
| `lm_start_status_message` | Before an LM call begins |
| `lm_end_status_message` | After an LM call completes |
| `module_start_status_message` | Before a module's `forward()`/`aforward()` |
| `module_end_status_message` | After a module's `forward()`/`aforward()` |
| `tool_start_status_message` | Before a tool is called |
| `tool_end_status_message` | After a tool completes |

Each hook receives the module/tool instance and the inputs or outputs, so you can create context-aware status messages.

---

## Putting It All Together

Here's a complete async streaming pipeline that combines everything:

```python
import asyncio
import dspy
from dspy.streaming import StreamListener, StatusMessageProvider
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


class ResearchPipeline(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought("question -> key_points")
        self.synthesize = dspy.ChainOfThought("question, key_points -> answer")

    async def aforward(self, question, **kwargs):
        analysis = await self.analyze.acall(question=question)
        return await self.synthesize.acall(
            question=question,
            key_points=analysis.key_points,
        )


class PipelineStatus(StatusMessageProvider):
    def module_start_status_message(self, instance, inputs):
        if hasattr(instance, '__class__'):
            return f"Starting {instance.__class__.__name__}..."
        return None

    def lm_start_status_message(self, instance, inputs):
        return "Generating response..."


pipeline = ResearchPipeline()
listener = StreamListener(signature_field_name="answer", predict_name="synthesize")

streaming_pipeline = dspy.streamify(
    pipeline,
    stream_listeners=[listener],
    status_message_provider=PipelineStatus(),
)


async def main():
    output = streaming_pipeline(question="What are the key challenges in quantum computing?")

    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StatusMessage):
            print(f"\n[{chunk.message}]")
        elif isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk.chunk, end="", flush=True)
        elif isinstance(chunk, dspy.Prediction):
            print(f"\n\nDone!")


asyncio.run(main())
```

---

## Key Takeaways

- **All built-in modules support `acall()`** for non-blocking async execution.
- **Custom modules implement `aforward()`** instead of `forward()` for async support.
- **Async tools work natively** by passing async functions to `dspy.Tool`.
- **`dspy.streamify()`** wraps any program for token-by-token streaming.
- **`StreamListener`** targets specific output fields and predictors.
- **`StatusMessageProvider`** gives users real-time pipeline progress updates.
- **Sync streaming** is available with `async_streaming=False`.
- **Use async for production**, sync for prototyping and research.

---

**Next up:** [11.3: Deploying DSPy Applications](../11.3-deployment/blog.md) covers packaging, containerizing, and deploying your DSPy programs as production APIs.

---

## Resources

- [DSPy Async Documentation](https://dspy.ai/learn/programming/async/)
- [DSPy Streaming Documentation](https://dspy.ai/learn/programming/streaming/)
- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)


