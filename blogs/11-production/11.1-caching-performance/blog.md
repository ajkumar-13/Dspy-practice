# 11.1: Caching and Performance Optimization

## Introduction

Every production LM application has two enemies: **latency** and **cost**. A single GPT-4o call takes 1 to 5 seconds and costs real money. Now multiply that by thousands of requests per day, and you're staring at slow response times and a terrifying cloud bill.

DSPy solves this with a **3-layer caching architecture** that is enabled by default. No configuration, no setup, no opt-in. The moment you make an LM call, DSPy automatically caches the result so that identical calls return instantly from memory or disk. You also get fine-grained control to tune, disable, or completely replace every layer when production demands it.

In this post, you'll learn how DSPy's caching works under the hood, how to configure it for production workloads, how to implement custom cache backends, and how to save and load optimized programs for deployment.

---

## What You'll Learn

- DSPy's 3-layer caching architecture: in-memory, on-disk, and provider-side
- How caching works automatically with zero configuration
- Measuring cache impact with timing and usage tracking
- Configuring cache layers with `dspy.configure_cache()`
- Provider-side prompt caching (Anthropic, OpenAI)
- Building custom cache backends by subclassing `dspy.clients.Cache`
- Saving and loading DSPy programs for deployment

---

## Prerequisites

- Completed [Phase 10: Multi-Modal](../../10-multi-modal/10.1-image-audio/blog.md)
- DSPy installed (`uv add dspy python-dotenv`)
- An OpenAI API key (or any LiteLLM-supported provider)

---

## DSPy's 3-Layer Caching Architecture

DSPy caches LM responses at three levels, each serving a distinct role:

### Layer 1: In-Memory Cache (`cachetools.LRUCache`)

The fastest layer. DSPy keeps a `cachetools.LRUCache` in process memory. When you make an LM call, DSPy first checks this in-memory store. If the exact same request was made during this session, the response comes back in microseconds, with no disk I/O and no network calls.

- **Speed:** Sub-millisecond lookups
- **Lifetime:** Current process only; lost when the program exits
- **Default size:** 1,000,000 entries

### Layer 2: On-Disk Cache (`diskcache.FanoutCache`)

If the in-memory cache misses, DSPy checks its on-disk cache, powered by `diskcache.FanoutCache`. This is a persistent, file-based cache that survives process restarts. It means you can stop your program, come back tomorrow, and repeat the same calls without hitting the LM provider again.

- **Speed:** Millisecond-range lookups (local disk I/O)
- **Lifetime:** Persistent across restarts
- **Default size limit:** 1 GB
- **Location:** Stored in DSPy's cache directory

### Layer 3: Provider-Side Prompt Cache

Some providers (Anthropic, OpenAI) support server-side prompt caching. When you send a long system prompt repeatedly, the provider caches the tokenized version on their servers and gives you a discount on subsequent calls. This layer is outside DSPy's direct control but can be triggered via configuration.

- **Speed:** Reduces provider-side processing time
- **Lifetime:** Provider-managed (typically minutes to hours)
- **Cost:** Reduced token costs on cache hits

The three layers work in sequence: **memory, then disk, then provider**. A cache hit at any layer short-circuits the rest.

---

## Cache in Action: Zero Configuration Required

The beauty of DSPy's caching is that it works automatically. Let's demonstrate with timing:

```python
import time
import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM with usage tracking
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)

predict = dspy.Predict("question -> answer")

# First call: hits the LM provider
start = time.time()
result1 = predict(question="Who is the GOAT?")
elapsed1 = time.time() - start
print(f"First call:  {elapsed1:.4f}s")
print(f"Answer: {result1.answer}")
print(f"Usage: {result1.get_lm_usage()}")

# Second call: same input, served from cache
start = time.time()
result2 = predict(question="Who is the GOAT?")
elapsed2 = time.time() - start
print(f"\nSecond call: {elapsed2:.6f}s")
print(f"Answer: {result2.answer}")
print(f"Usage: {result2.get_lm_usage()}")

print(f"\nSpeedup: {elapsed1 / elapsed2:.0f}x faster")
```

Typical output:

```
First call:  3.8421s
Answer: The term "GOAT"...
Usage: {'openai/gpt-4o-mini': {'prompt_tokens': 42, 'completion_tokens': 85}}

Second call: 0.000487s
Answer: The term "GOAT"...
Usage: {}

Speedup: 7889x faster
```

Notice two key things:
1. **The second call is ~8000x faster** because it's served from the in-memory cache.
2. **Usage is empty on the second call** because no tokens were consumed and no LM call was made.

This is automatic. You didn't configure anything. DSPy just does it.

---

## Configuring Cache with `dspy.configure_cache()`

While the defaults are excellent for development, production workloads often need tuning. Use `dspy.configure_cache()` to control each layer:

```python
# Fine-tune cache behavior
dspy.configure_cache(
    enable_disk_cache=True,               # Keep disk cache on (default: True)
    enable_memory_cache=True,             # Keep memory cache on (default: True)
    disk_size_limit_bytes=5_000_000_000,  # 5 GB disk cache
    memory_max_entries=2_000_000,         # 2M entries in memory
)
```

### Common Production Configurations

**High-throughput API server:** maximize memory cache, large disk cache:

```python
dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_size_limit_bytes=10_000_000_000,  # 10 GB
    memory_max_entries=5_000_000,
)
```

**Debug mode:** disable caching to always get fresh responses:

```python
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
```

**Memory-constrained environment:** disk only, no in-memory overhead:

```python
dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=False,
    disk_size_limit_bytes=500_000_000,  # 500 MB
)
```

---

## Provider-Side Prompt Caching

Anthropic and OpenAI support server-side caching of long prompts. This is especially useful when you have large system prompts or extensive few-shot examples that repeat across calls. DSPy integrates with this via the `cache_control_injection_points` parameter:

```python
# Enable Anthropic's prompt caching for system messages
lm = dspy.LM(
    "anthropic/claude-sonnet-4-5-20250929",
    cache_control_injection_points=[
        {"location": "message", "role": "system"}
    ],
)
dspy.configure(lm=lm)
```

This tells the provider to cache the system message tokens on their servers. Subsequent calls with the same system prompt get a **discount** and reduced latency on the provider side.

You can also cache user messages or other content:

```python
# Cache both system and the last user message
lm = dspy.LM(
    "anthropic/claude-sonnet-4-5-20250929",
    cache_control_injection_points=[
        {"location": "message", "role": "system"},
        {"location": "message", "role": "user"},
    ],
)
```

Provider-side caching is complementary to DSPy's local caching. It helps when the local cache misses but the provider has seen the same prompt prefix recently. For optimized programs with long few-shot demonstrations in the system prompt, this can significantly reduce per-token costs.

---

## Custom Cache Implementations

For advanced use cases like Redis-backed caching, shared caches across services, or custom key strategies, you can implement your own cache by subclassing `dspy.clients.Cache`:

```python
import hashlib
import json
import dspy
from dspy.clients import Cache
from dotenv import load_dotenv

load_dotenv()


class ContentOnlyCache(Cache):
    """
    A custom cache that generates keys based only on message content,
    ignoring the model name. This means cached responses can be reused
    across different models, useful for testing and development.
    """

    def cache_key(self, request):
        """Generate cache key from message content only."""
        messages = request.get("messages", [])
        content = json.dumps(
            [m.get("content", "") for m in messages],
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, key):
        """Retrieve cached response."""
        return self._memory_cache.get(key)

    def put(self, key, value):
        """Store response in cache."""
        self._memory_cache[key] = value


# Replace DSPy's default cache with your custom implementation
dspy.cache = ContentOnlyCache(
    enable_memory_cache=True,
    enable_disk_cache=True,
)

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question -> answer")
result = predict(question="What is 2 + 2?")
print(result.answer)
```

The three methods you override are:

- **`cache_key(request)`** generates a unique key from the request dict. Customize this to control what makes two requests "the same."
- **`get(key)`** retrieves a cached value by key. Return `None` on cache miss.
- **`put(key, value)`** stores a value in the cache.

### When to Use Custom Caches

| Use Case | Custom Cache Strategy |
|---|---|
| Shared cache across microservices | Redis-backed cache with `get()`/`put()` calling Redis |
| Model-agnostic caching | Key generation ignoring model name |
| TTL-based expiration | Add timestamp checks in `get()` |
| Cache analytics | Instrument `get()`/`put()` with hit/miss counters |
| Multi-region deployment | Cache backed by a distributed store |

---

## Saving and Loading Programs

Caching helps at runtime, but for deployment you need to **save optimized programs** and load them in production. DSPy provides two approaches:

### Approach 1: State-Only Saving

Saves just the learned parameters (optimized prompts, few-shot demos) as a JSON file:

```python
# After optimization
optimized_program = optimizer.compile(program, trainset=trainset)

# Save state only, lightweight and portable
optimized_program.save("optimized_program.json", save_program=False)

# Load state into a fresh program instance
fresh_program = MyModule()
fresh_program.load("optimized_program.json")
```

This is the **recommended approach** for most production deployments. The JSON file is small, human-readable, and version-controllable. You maintain the program class definition in your codebase and just load the optimized state at startup.

### Approach 2: Whole-Program Saving

Saves the entire program structure plus state using `cloudpickle`:

```python
# Save the complete program (structure + state)
optimized_program.save("./saved_program/", save_program=True)

# Load without needing the class definition
loaded_program = dspy.load("./saved_program/")
```

This approach is useful when:
- You don't want to maintain the program class definition in production
- The program structure is complex with dynamic nested modules
- You want a single deployable artifact that captures everything

### Advanced: Selective Serialization

The `modules_to_serialize` parameter lets you control which modules get serialized when using whole-program saving:

```python
# Only serialize specific modules
optimized_program.save(
    "./saved_program/",
    save_program=True,
    modules_to_serialize=["respond", "classifier"],
)
```

### Backward Compatibility

DSPy guarantees backward compatibility for saved programs from `dspy>=3.0.0`. Programs saved with DSPy 3.0 will load correctly in all future versions, so you can safely version and archive your optimized program artifacts.

---

## Performance Optimization Patterns

Beyond caching, here are key patterns for production performance:

### 1. Track Usage to Monitor Costs

```python
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)

predict = dspy.Predict("question -> answer")
result = predict(question="Explain quantum computing")

usage = result.get_lm_usage()
print(usage)
# {'openai/gpt-4o-mini': {'prompt_tokens': 42, 'completion_tokens': 180}}
```

### 2. Use Thread-Safe Contexts

In multi-threaded servers, use `dspy.context()` to isolate LM configurations per request:

```python
import threading

def handle_request(question, model):
    with dspy.context(lm=dspy.LM(model)):
        predict = dspy.Predict("question -> answer")
        return predict(question=question)

# Each thread can use a different model safely
t1 = threading.Thread(target=handle_request, args=("Q1", "openai/gpt-4o-mini"))
t2 = threading.Thread(target=handle_request, args=("Q2", "anthropic/claude-sonnet-4-5-20250929"))
t1.start(); t2.start()
t1.join(); t2.join()
```

### 3. Route to the Right Model per Task

Not every call needs your most expensive model. Use `dspy.context()` to route different tasks to appropriate models:

```python
class SmartRouter(dspy.Module):
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
            with dspy.context(lm=dspy.LM("openai/gpt-4o")):
                return self.complex_answer(question=question)
```

---

## Key Takeaways

- **DSPy caches everything by default** in-memory and on-disk, with zero configuration.
- **Cache hits are ~8000x faster** and consume zero tokens.
- **`dspy.configure_cache()`** gives you full control over cache layers and sizing.
- **Provider-side prompt caching** reduces costs for repetitive long prompts.
- **Custom cache backends** let you use Redis, add TTLs, or share caches across services.
- **Save programs with `.save()`** and load with `.load()` for production deployment.
- **Track usage** with `track_usage=True` to monitor token consumption and costs.

---

**Next up:** [11.2: Async Programming and Streaming in DSPy](../11.2-async-streaming/blog.md) covers high-throughput async pipelines with real-time streaming.

---

## Resources

- [DSPy Caching Documentation](https://dspy.ai/learn/caching/)
- [DSPy Saving/Loading Programs](https://dspy.ai/learn/saving_loading/)
- [LiteLLM Provider Documentation](https://docs.litellm.ai/)
- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)