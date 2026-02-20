# 6.P Project: Financial Analyst Agent


## Introduction

You've learned every agent pattern in DSPy: ReAct loops, advanced tool design, MCP integration, conversation memory, and privacy-conscious delegation. Now it's time to combine them into a real, end-to-end project.

In this project, you'll build a **financial analyst agent**, a ReAct agent equipped with tools for looking up stock prices, retrieving financial data, and performing calculations. You'll wire the tools together, test with real financial queries, add error handling, optimize with MIPROv2, and evaluate the results. The pattern follows the same workflow you've used throughout this series: **build, test, optimize, evaluate**.

---

## Project Overview

Here's what we'll build:

1. **Financial tools**: stock price lookup, financial data retrieval, and calculation
2. **A ReAct agent**: decides which tools to call and in what order
3. **Error handling**: graceful recovery from API failures and bad inputs
4. **Optimization**: MIPROv2 to improve analysis quality
5. **Evaluation**: quantitative measurement of agent performance

---

## Prerequisites

- Completed Phase 6 blogs (6.1 through 6.5)
- DSPy installed (`uv add dspy python-dotenv`)
- An OpenAI API key

---

## Step 1: Define Financial Tools

We'll create three tools that simulate financial data access. In production, you'd replace the mock data with real API calls (Yahoo Finance, Alpha Vantage, etc.).

```python

import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# --- Financial Tools ---

def get_stock_price(ticker: str) -> str:
    """Get the current stock price and daily change for a ticker symbol.
    Returns price in USD and percentage change from previous close."""
    stocks = {
        "AAPL": {"price": 187.50, "change": 1.2, "name": "Apple Inc."},
        "GOOGL": {"price": 141.20, "change": -0.8, "name": "Alphabet Inc."},
        "MSFT": {"price": 378.90, "change": 0.5, "name": "Microsoft Corp."},
        "AMZN": {"price": 178.25, "change": 2.1, "name": "Amazon.com Inc."},
        "TSLA": {"price": 248.50, "change": -1.5, "name": "Tesla Inc."},
        "NVDA": {"price": 721.30, "change": 3.4, "name": "NVIDIA Corp."},
        "META": {"price": 485.60, "change": 0.9, "name": "Meta Platforms Inc."},
    }
    ticker_upper = ticker.upper().strip()
    if ticker_upper in stocks:
        s = stocks[ticker_upper]
        direction = "up" if s["change"] > 0 else "down"
        return (
            f"{s['name']} ({ticker_upper}): ${s['price']:.2f}, "
            f"{direction} {abs(s['change'])}% today"
        )
    return f"Ticker '{ticker}' not found. Available: {', '.join(stocks.keys())}"


def get_financial_data(ticker: str, metric: str) -> str:
    """Get a specific financial metric for a company.
    Supported metrics: pe_ratio, market_cap, revenue, eps, dividend_yield."""
    data = {
        "AAPL": {
            "pe_ratio": "28.5",
            "market_cap": "$2.87T",
            "revenue": "$383.3B (TTM)",
            "eps": "$6.57",
            "dividend_yield": "0.56%",
        },
        "GOOGL": {
            "pe_ratio": "23.1",
            "market_cap": "$1.74T",
            "revenue": "$307.4B (TTM)",
            "eps": "$6.11",
            "dividend_yield": "0.50%",
        },
        "MSFT": {
            "pe_ratio": "35.2",
            "market_cap": "$2.81T",
            "revenue": "$227.6B (TTM)",
            "eps": "$10.76",
            "dividend_yield": "0.74%",
        },
        "NVDA": {
            "pe_ratio": "62.4",
            "market_cap": "$1.78T",
            "revenue": "$79.8B (TTM)",
            "eps": "$11.56",
            "dividend_yield": "0.03%",
        },
    }
    ticker_upper = ticker.upper().strip()
    metric_lower = metric.lower().strip()
    if ticker_upper not in data:
        return f"Financial data not available for '{ticker}'. Available: {', '.join(data.keys())}"
    if metric_lower not in data[ticker_upper]:
        available = ", ".join(data[ticker_upper].keys())
        return f"Metric '{metric}' not found. Available metrics: {available}"
    return f"{ticker_upper} {metric_lower}: {data[ticker_upper][metric_lower]}"


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression for financial calculations.
    Supports: +, -, *, /, **, (), and common operations.
    Example: '187.50 * 28.5' to compute price × PE ratio."""
    try:
        allowed_chars = set("0123456789.+-*/() eE")
        if not all(c in allowed_chars for c in expression):
            return "Error: Only numeric expressions with +, -, *, /, ** are supported."
        result = eval(expression)
        if isinstance(result, float):
            return f"{result:,.2f}"
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Calculation error: {e}"
```

Each tool follows the best practices from 6.1:

- **Descriptive names** that tell the LM what the tool does
- **Detailed docstrings** with supported parameters and examples
- **Type hints** on all parameters
- **Error messages** instead of exceptions for bad inputs

---

## Step 2: Build the Agent

Wire the tools into a `dspy.ReAct` agent:

```python
# Build the financial analyst agent
analyst = dspy.ReAct(
    "question -> analysis",
    tools=[get_stock_price, get_financial_data, calculate],
    max_iters=10,
)

# Test with a simple question
result = analyst(question="What is Apple's current stock price and PE ratio?")
print(f"Analysis: {result.analysis}")
```

The signature uses `"question -> analysis"` rather than `"question -> answer"` to signal that we expect an analytical response, not just a factual lookup.

---

## Step 3: Test with Financial Queries

Let's test the agent with increasingly complex queries that require multiple tool calls:

```python
# Test queries of varying complexity
test_queries = [
    # Simple: one tool call
    "What is NVIDIA's stock price?",

    # Medium: two tool calls
    "Compare the PE ratios of Apple and Microsoft.",

    # Complex: multiple tools + calculation
    "Which has a higher market cap, Apple or Microsoft? What's the difference?",

    # Multi-step reasoning
    "If I invested $10,000 in NVIDIA stock at the current price, "
    "how many shares could I buy and what dividend income would I earn annually?",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Q: {query}")
    result = analyst(question=query)
    print(f"A: {result.analysis}")

    # Show trajectory length
    num_steps = len([k for k in result.trajectory.keys() if k.startswith("thought")])
    print(f"   (Agent used {num_steps} reasoning steps)")
```

---

## Step 4: Add Error Handling

Real financial APIs fail. Let's add a wrapper pattern for robust error handling:

```python
import functools


def with_error_handling(func):
    """Decorator that catches exceptions and returns error messages."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConnectionError:
            return f"Error: Could not connect to data source for {func.__name__}. Please try again."
        except TimeoutError:
            return f"Error: Request timed out for {func.__name__}. Please try again."
        except Exception as e:
            return f"Error in {func.__name__}: {str(e)}"
    return wrapper


# Apply to all tools
get_stock_price_safe = with_error_handling(get_stock_price)
get_financial_data_safe = with_error_handling(get_financial_data)
calculate_safe = with_error_handling(calculate)

# Rebuild agent with safe tools
analyst_safe = dspy.ReAct(
    "question -> analysis",
    tools=[get_stock_price_safe, get_financial_data_safe, calculate_safe],
    max_iters=10,
)
```

The agent's built-in error recovery (from 6.1) works together with this: if a tool returns an error message, the model sees it as an observation and can decide to retry, try different parameters, or answer with what it has.

---

## Step 5: Optimize

Let's optimize the agent with MIPROv2 to improve the quality of its financial analysis:

```python
from dspy.evaluate import SemanticF1

# Build a training set of financial questions with expected analyses
trainset = [
    dspy.Example(
        question="What is Apple's stock price and PE ratio?",
        analysis="Apple Inc. (AAPL) is currently trading at $187.50, up 1.2% today. "
                 "Its PE ratio is 28.5, which is moderate for a large-cap tech company."
    ).with_inputs("question"),
    dspy.Example(
        question="Compare NVIDIA and Microsoft's PE ratios. Which is more expensive?",
        analysis="NVIDIA has a PE ratio of 62.4, while Microsoft's is 35.2. "
                 "NVIDIA is significantly more expensive on a PE basis, reflecting "
                 "higher growth expectations in the AI chip market."
    ).with_inputs("question"),
    dspy.Example(
        question="What is Google's dividend yield and EPS?",
        analysis="Alphabet Inc. (GOOGL) has a dividend yield of 0.50% and earnings per share "
                 "of $6.11. The dividend yield is modest, typical for a growth-oriented tech company."
    ).with_inputs("question"),
    dspy.Example(
        question="If I have $5000, how many shares of AMZN can I buy?",
        analysis="Amazon (AMZN) is currently at $178.25. With $5,000 you could buy "
                 "approximately 28 shares ($5,000 / $178.25 = 28.05), spending $4,991.00 "
                 "with $9.00 remaining."
    ).with_inputs("question"),
    dspy.Example(
        question="What is Microsoft's revenue and market cap?",
        analysis="Microsoft Corp. (MSFT) has trailing twelve-month revenue of $227.6B "
                 "and a market capitalization of $2.81T, making it one of the most valuable "
                 "companies in the world."
    ).with_inputs("question"),
]

# Build evaluation set
evalset = [
    dspy.Example(
        question="Which stock is up the most today: AAPL, GOOGL, or NVDA?",
        analysis="NVIDIA (NVDA) is up the most at 3.4%, compared to Apple at 1.2% "
                 "and Alphabet at -0.8% (which is actually down)."
    ).with_inputs("question"),
    dspy.Example(
        question="What is the PE ratio difference between Apple and Google?",
        analysis="Apple's PE ratio is 28.5 and Google's is 23.1. The difference is 5.4, "
                 "meaning Apple trades at a slight premium to Google on a PE basis."
    ).with_inputs("question"),
]

# Optimize
metric = SemanticF1()
optimizer = dspy.MIPROv2(
    metric=metric,
    auto="light",
    num_threads=4,
)

optimized_analyst = optimizer.compile(
    analyst,
    trainset=trainset,
    requires_permission_to_run=False,
)

# Save the optimized agent
optimized_analyst.save("optimized_financial_analyst.json")
```

---

## Step 6: Evaluate

Compare the baseline and optimized agents on the evaluation set:

```python
from dspy.evaluate import Evaluate

# Run evaluation
evaluator = Evaluate(
    devset=evalset,
    metric=metric,
    num_threads=4,
    display_progress=True,
    display_table=5,
)

# Baseline performance
print("=== Baseline Agent ===")
baseline_score = evaluator(analyst)

# Optimized performance
print("\n=== Optimized Agent ===")
optimized_score = evaluator(optimized_analyst)

print(f"\nBaseline SemanticF1:  {baseline_score:.1%}")
print(f"Optimized SemanticF1: {optimized_score:.1%}")
print(f"Improvement:          {optimized_score - baseline_score:+.1%}")
```

You can also inspect individual trajectories to qualitatively compare the agents:

```python
# Compare trajectories on a specific question
question = "If NVDA drops 10%, what would the new price be?"

baseline_result = analyst(question=question)
optimized_result = optimized_analyst(question=question)

print(f"Baseline:  {baseline_result.analysis}")
print(f"Optimized: {optimized_result.analysis}")

# Check how many steps each used
baseline_steps = len([k for k in baseline_result.trajectory if k.startswith("thought")])
optimized_steps = len([k for k in optimized_result.trajectory if k.startswith("thought")])
print(f"\nBaseline steps:  {baseline_steps}")
print(f"Optimized steps: {optimized_steps}")
```

---

## What We Learned

This project ties together everything from Phase 6:

1. **ReAct agents** (6.1): the financial analyst uses iterative reasoning with tool calls.
2. **Tool design** (6.2): clear docstrings, type hints, and error-returning patterns make tools LM-friendly.
3. **Error handling** (6.1, 6.4): the decorator pattern wraps tools for graceful failure, and ReAct's built-in recovery adapts to errors.
4. **Optimization** (6.1): MIPROv2 improves analysis quality without changing code.
5. **The DSPy workflow**: build, test, optimize, evaluate. Every project follows this pattern.

To extend this into a production system, you would:

- Replace mock data with real financial APIs (Yahoo Finance, Alpha Vantage)
- Add MCP tools (6.3) for standardized financial data access
- Add conversation memory (6.4) for multi-turn financial discussions
- Use Papillon (6.5) to keep portfolio data on-premises while using cloud models for analysis

---

## Next Up

You've completed Phase 6. You can now build, optimize, and evaluate agents with tools, memory, and multi-model routing. In Phase 7, we'll go deeper on optimization by **fine-tuning model weights**, moving beyond prompt optimization to actually change the model itself.

**[7.1 — Bootstrap Fine-Tuning →](../../07-finetuning/7.1-bootstrap-finetune/blog.md)**

---

## Resources

- [DSPy Yahoo Finance Agent Tutorial](https://dspy.ai/tutorials/yahoo_finance_react/)
- [DSPy Agents Tutorial](https://dspy.ai/tutorials/agents/)
- [DSPy ReAct API Reference](https://dspy.ai/api/modules/ReAct/)
- [MIPROv2 Optimizer](https://dspy.ai/api/optimizers/MIPROv2/)
- [SemanticF1 Metric](https://dspy.ai/api/metrics/SemanticF1/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
