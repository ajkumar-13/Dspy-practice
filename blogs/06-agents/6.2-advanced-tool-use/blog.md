# 6.2 Advanced Tool Use

## Introduction

In the previous post, you built a ReAct agent by passing Python functions directly to `dspy.ReAct`. That works great for straightforward use cases, but real-world agents often need more control: wrapping tools with custom schemas, handling tool calls manually, leveraging native function calling from providers like OpenAI, or running tools asynchronously. DSPy provides a layered toolkit for all of these patterns.

This post covers the `dspy.Tool` wrapper class, the `dspy.ToolCalls` type for manual tool orchestration, native function calling via adapters, and async tool execution, giving you the precision you need when `dspy.ReAct` alone isn't enough.

---

## What You'll Learn

- The `dspy.Tool` class and why you'd use it over raw functions
- Manual tool handling with `dspy.ToolCalls` and `ToolCall.execute()`
- Native function calling with `dspy.ChatAdapter` and `dspy.JSONAdapter`
- Async tools with `tool.acall()` and sync-async conversion
- When to use ReAct vs. manual tool handling

---

## Prerequisites

- Completed [6.1 ReAct Agents](../6.1-react-agents/blog.md)
- DSPy installed (`uv add dspy python-dotenv`)
- An OpenAI API key (or any LiteLLM-supported provider)

---

## The `dspy.Tool` Class

While `dspy.ReAct` accepts plain Python functions, DSPy also provides `dspy.Tool`, a wrapper that gives you explicit control over how a tool is presented to the LM.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


def search_web(query: str, max_results: int = 5) -> list[str]:
    """Search the web for information. Returns a list of relevant snippets."""
    # Simulated search results
    return [f"Result {i+1} for '{query}'" for i in range(max_results)]


# Wrap with dspy.Tool for explicit control
search_tool = dspy.Tool(
    func=search_web,
    name="search_web",           # Override the function name if desired
    desc="Search the web for current information on any topic.",
    args={                        # Override argument descriptions
        "query": "The search query string",
        "max_results": "Maximum number of results to return (default: 5)",
    },
)

# Use with ReAct — dspy.Tool objects work just like functions
agent = dspy.ReAct(
    "question -> answer",
    tools=[search_tool],
    max_iters=5,
)
```

When should you use `dspy.Tool` instead of a plain function?

- When you want to **override the name or description** without changing the function itself.
- When you're wrapping **third-party functions** whose docstrings aren't LM-friendly.
- When you need to **customize argument descriptions** for better tool selection.
- When converting tools from other frameworks (like MCP, covered in 6.3).

---

## Manual Tool Handling with `ToolCalls`

`dspy.ReAct` runs the full Thought, Tool, Observation loop for you. But sometimes you want **manual control**: deciding which tools to call, when to call them, or handling the results yourself. DSPy provides the `dspy.ToolCalls` type for this.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    prices = {"AAPL": "187.50", "GOOGL": "141.20", "MSFT": "378.90"}
    return prices.get(ticker.upper(), f"Unknown ticker: {ticker}")


def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Get the exchange rate between two currencies."""
    rates = {("USD", "EUR"): "0.92", ("USD", "GBP"): "0.79"}
    key = (from_currency.upper(), to_currency.upper())
    return rates.get(key, f"Rate not available for {from_currency}/{to_currency}")


# Define a signature that outputs ToolCalls
class ToolSelector(dspy.Signature):
    """Decide which tool to call based on the user's question."""
    question: str = dspy.InputField()
    tool_calls: dspy.ToolCalls = dspy.OutputField()


# Create the predictor
selector = dspy.Predict(ToolSelector)

# Call the predictor — it produces structured tool calls
result = selector(question="What is Apple's stock price?")

# Execute the tool calls — ToolCall.execute() auto-discovers functions
for tool_call in result.tool_calls:
    output = tool_call.execute()
    print(f"Tool: {tool_call.name}, Args: {tool_call.args}, Result: {output}")
```

The `ToolCall.execute()` method (available in DSPy v3.0.4b2+) automatically discovers the function by name from the calling scope and executes it. This is useful when you want to:

- **Build custom agent loops** with your own control flow logic.
- **Filter or modify tool calls** before execution (e.g., add rate limiting, logging).
- **Compose multiple tool-calling steps** in a pipeline.

---

## Native Function Calling

Modern LLM providers (OpenAI, Anthropic, Google) support **native function calling**, where the model generates structured JSON tool calls directly, rather than the adapter parsing them from text. DSPy supports this through its adapter system.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

# Option 1: ChatAdapter with native function calling enabled
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    adapter=dspy.ChatAdapter(use_native_function_calling=True),
)

# Option 2: JSONAdapter — uses native function calling by default
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    adapter=dspy.JSONAdapter(),  # Native function calling is the default
)
```

Native function calling has several advantages over text-based tool parsing:

- **Higher reliability**: the model's output is structured JSON, not free text that needs parsing.
- **Better tool selection**: providers optimize their models for function calling.
- **Lower error rates**: no parsing failures from malformed tool call syntax.

The choice between `ChatAdapter` and `JSONAdapter` depends on your use case. `JSONAdapter` is newer and uses native function calling by default, while `ChatAdapter` gives you the option to toggle it on or off.

---

## Async Tools

For I/O-bound tools (API calls, database queries, web scraping), async execution can dramatically improve performance. DSPy supports async tools natively:

```python
import dspy
import asyncio
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


async def fetch_url(url: str) -> str:
    """Fetch the content of a URL asynchronously."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            text = await response.text()
            return text[:500]  # Return first 500 chars


# Wrap the async function as a tool
url_tool = dspy.Tool(func=fetch_url)

# Call async tools directly
result = await url_tool.acall(url="https://example.com")
```

If your agent uses a mix of sync and async tools, DSPy can bridge the gap:

```python
# Allow DSPy to convert between sync and async automatically
with dspy.context(allow_tool_async_sync_conversion=True):
    agent = dspy.ReAct(
        "question -> answer",
        tools=[fetch_url, get_stock_price],  # Mix of async and sync tools
        max_iters=5,
    )
    # Use acall for async execution
    result = await agent.acall(question="What's on example.com?")
```

---

## When to Use ReAct vs. Manual Tool Handling

| Scenario | Approach |
|---|---|
| General Q&A with tools | `dspy.ReAct` |
| Single tool call, no reasoning loop | `dspy.Predict` with `ToolCalls` |
| Custom control flow (retry logic, branching) | Manual `ToolCalls` + `execute()` |
| High-reliability tool selection | Native function calling |
| Multiple async tools | `dspy.ReAct` with `acall()` |
| Pipeline of tool-augmented steps | Custom `dspy.Module` |

The rule of thumb: **start with `dspy.ReAct`**. If you find yourself needing more control over the tool-calling process (custom retry logic, filtering, or non-linear control flow), drop down to manual `ToolCalls` handling.

---

## Key Takeaways

- **`dspy.Tool`** wraps functions with custom names, descriptions, and argument schemas for LM-friendly presentation.
- **`dspy.ToolCalls`** and **`ToolCall.execute()`** enable manual tool handling when you need precise control over the agent loop.
- **Native function calling** via `dspy.ChatAdapter(use_native_function_calling=True)` or `dspy.JSONAdapter()` improves reliability by leveraging provider-native structured outputs.
- **Async tools** with `tool.acall()` and `dspy.context(allow_tool_async_sync_conversion=True)` enable efficient I/O-bound tool execution.
- **Start with ReAct, drop to manual when needed**. DSPy gives you the full spectrum of control.

---

## Next Up

Tools don't have to live in your codebase. The **Model Context Protocol (MCP)** is an open standard that lets your agents access a growing ecosystem of pre-built tools, from file systems to databases to web APIs. In the next post, we'll integrate MCP servers into DSPy agents.

**[6.3: MCP Integration →](../6.3-mcp-integration/blog.md)**

---

## Resources

- [DSPy Tool Use Guide](https://dspy.ai/tutorials/agents/)
- [DSPy Agents API Reference](https://dspy.ai/api/modules/ReAct/)
- [DSPy Tool Class](https://dspy.ai/api/tools/Tool/)
- [Native Function Calling](https://dspy.ai/learn/programming/language_models/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
