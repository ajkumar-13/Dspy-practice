"""
Blog 6.P - Project: Financial Analyst Agent
Run: python 01_financial_analyst.py
"""

import functools

import dspy
from dotenv import load_dotenv
from dspy.evaluate import Evaluate, SemanticF1

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# ============================================================
# Step 1: Define Financial Tools
# ============================================================


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
    Example: '187.50 * 28.5' to compute price x PE ratio."""
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


# ============================================================
# Step 2: Build the Agent
# ============================================================

analyst = dspy.ReAct(
    "question -> analysis",
    tools=[get_stock_price, get_financial_data, calculate],
    max_iters=10,
)

# Quick test
result = analyst(question="What is Apple's current stock price and PE ratio?")
print(f"Analysis: {result.analysis}")


# ============================================================
# Step 3: Test with Financial Queries
# ============================================================

print("\n" + "=" * 60)
print("Testing with financial queries")
print("=" * 60)

test_queries = [
    "What is NVIDIA's stock price?",
    "Compare the PE ratios of Apple and Microsoft.",
    "Which has a higher market cap, Apple or Microsoft? What's the difference?",
    (
        "If I invested $10,000 in NVIDIA stock at the current price, "
        "how many shares could I buy and what dividend income would I earn annually?"
    ),
]

for query in test_queries:
    print(f"\n{'=' * 60}")
    print(f"Q: {query}")
    result = analyst(question=query)
    print(f"A: {result.analysis}")
    num_steps = len([k for k in result.trajectory.keys() if k.startswith("thought")])
    print(f"   (Agent used {num_steps} reasoning steps)")


# ============================================================
# Step 4: Error Handling Decorator
# ============================================================


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


get_stock_price_safe = with_error_handling(get_stock_price)
get_financial_data_safe = with_error_handling(get_financial_data)
calculate_safe = with_error_handling(calculate)

analyst_safe = dspy.ReAct(
    "question -> analysis",
    tools=[get_stock_price_safe, get_financial_data_safe, calculate_safe],
    max_iters=10,
)


# ============================================================
# Step 5: Optimize with MIPROv2
# ============================================================

print("\n" + "=" * 60)
print("Optimizing with MIPROv2")
print("=" * 60)

trainset = [
    dspy.Example(
        question="What is Apple's stock price and PE ratio?",
        analysis=(
            "Apple Inc. (AAPL) is currently trading at $187.50, up 1.2% today. "
            "Its PE ratio is 28.5, which is moderate for a large-cap tech company."
        ),
    ).with_inputs("question"),
    dspy.Example(
        question="Compare NVIDIA and Microsoft's PE ratios. Which is more expensive?",
        analysis=(
            "NVIDIA has a PE ratio of 62.4, while Microsoft's is 35.2. "
            "NVIDIA is significantly more expensive on a PE basis, reflecting "
            "higher growth expectations in the AI chip market."
        ),
    ).with_inputs("question"),
    dspy.Example(
        question="What is Google's dividend yield and EPS?",
        analysis=(
            "Alphabet Inc. (GOOGL) has a dividend yield of 0.50% and earnings per share "
            "of $6.11. The dividend yield is modest, typical for a growth-oriented tech company."
        ),
    ).with_inputs("question"),
    dspy.Example(
        question="If I have $5000, how many shares of AMZN can I buy?",
        analysis=(
            "Amazon (AMZN) is currently at $178.25. With $5,000 you could buy "
            "approximately 28 shares ($5,000 / $178.25 = 28.05), spending $4,991.00 "
            "with $9.00 remaining."
        ),
    ).with_inputs("question"),
    dspy.Example(
        question="What is Microsoft's revenue and market cap?",
        analysis=(
            "Microsoft Corp. (MSFT) has trailing twelve-month revenue of $227.6B "
            "and a market capitalization of $2.81T, making it one of the most valuable "
            "companies in the world."
        ),
    ).with_inputs("question"),
]

evalset = [
    dspy.Example(
        question="Which stock is up the most today: AAPL, GOOGL, or NVDA?",
        analysis=(
            "NVIDIA (NVDA) is up the most at 3.4%, compared to Apple at 1.2% "
            "and Alphabet at -0.8% (which is actually down)."
        ),
    ).with_inputs("question"),
    dspy.Example(
        question="What is the PE ratio difference between Apple and Google?",
        analysis=(
            "Apple's PE ratio is 28.5 and Google's is 23.1. The difference is 5.4, "
            "meaning Apple trades at a slight premium to Google on a PE basis."
        ),
    ).with_inputs("question"),
]

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

optimized_analyst.save("optimized_financial_analyst.json")
print("Optimized agent saved to optimized_financial_analyst.json")


# ============================================================
# Step 6: Evaluate
# ============================================================

print("\n" + "=" * 60)
print("Evaluating baseline vs. optimized")
print("=" * 60)

evaluator = Evaluate(
    devset=evalset,
    metric=metric,
    num_threads=4,
    display_progress=True,
    display_table=5,
)

print("\n=== Baseline Agent ===")
baseline_score = evaluator(analyst)

print("\n=== Optimized Agent ===")
optimized_score = evaluator(optimized_analyst)

print(f"\nBaseline SemanticF1:  {baseline_score:.1%}")
print(f"Optimized SemanticF1: {optimized_score:.1%}")
print(f"Improvement:          {optimized_score - baseline_score:+.1%}")
