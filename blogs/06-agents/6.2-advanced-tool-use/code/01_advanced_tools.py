"""
Blog 6.2 - Advanced Tool Use: dspy.Tool, ToolCalls, Native Function Calling, Async
Run: python 01_advanced_tools.py
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# ============================================================
# Part 1: The dspy.Tool class
# ============================================================

def search_web(query: str, max_results: int = 5) -> list[str]:
    """Search the web for information. Returns a list of relevant snippets."""
    return [f"Result {i+1} for '{query}'" for i in range(max_results)]


# Wrap with dspy.Tool for explicit control
search_tool = dspy.Tool(
    func=search_web,
    name="search_web",
    desc="Search the web for current information on any topic.",
    args={
        "query": "The search query string",
        "max_results": "Maximum number of results to return (default: 5)",
    },
)

# Use with ReAct
agent = dspy.ReAct(
    "question -> answer",
    tools=[search_tool],
    max_iters=5,
)

result = agent(question="What are the latest trends in AI?")
print(f"Part 1 (dspy.Tool) Answer: {result.answer}")


# ============================================================
# Part 2: Manual tool handling with ToolCalls
# ============================================================

def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    prices = {"AAPL": "187.50", "GOOGL": "141.20", "MSFT": "378.90"}
    return prices.get(ticker.upper(), f"Unknown ticker: {ticker}")


def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Get the exchange rate between two currencies."""
    rates = {("USD", "EUR"): "0.92", ("USD", "GBP"): "0.79"}
    key = (from_currency.upper(), to_currency.upper())
    return rates.get(key, f"Rate not available for {from_currency}/{to_currency}")


class ToolSelector(dspy.Signature):
    """Decide which tool to call based on the user's question."""
    question: str = dspy.InputField()
    tool_calls: dspy.ToolCalls = dspy.OutputField()


selector = dspy.Predict(ToolSelector)
result = selector(question="What is Apple's stock price?")

print("\nPart 2 (ToolCalls) Results:")
for tool_call in result.tool_calls:
    output = tool_call.execute()
    print(f"  Tool: {tool_call.name}, Args: {tool_call.args}, Result: {output}")


# ============================================================
# Part 3: Native Function Calling
# ============================================================

# ChatAdapter with native function calling enabled
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    adapter=dspy.ChatAdapter(use_native_function_calling=True),
)

native_agent = dspy.ReAct(
    "question -> answer",
    tools=[search_tool],
    max_iters=5,
)

result = native_agent(question="Search for DSPy framework features")
print(f"\nPart 3 (Native FC) Answer: {result.answer}")


# ============================================================
# Part 4: Async Tools
# ============================================================

import asyncio


async def fetch_url(url: str) -> str:
    """Fetch the content of a URL asynchronously."""
    # Simulated async fetch
    return f"Content from {url} (simulated)"


url_tool = dspy.Tool(func=fetch_url)


async def run_async_demo():
    """Demonstrate async tool usage."""
    result = await url_tool.acall(url="https://example.com")
    print(f"\nPart 4 (Async) Result: {result}")

    # Mix sync and async tools with ReAct
    with dspy.context(allow_tool_async_sync_conversion=True):
        agent = dspy.ReAct(
            "question -> answer",
            tools=[fetch_url, get_stock_price],
            max_iters=5,
        )
        result = await agent.acall(question="What's on example.com?")
        print(f"Part 4 (Mixed Async Agent) Answer: {result.answer}")


asyncio.run(run_async_demo())
