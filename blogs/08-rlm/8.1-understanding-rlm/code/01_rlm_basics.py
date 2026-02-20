"""
Blog 8.1: Understanding RLM (Recursive Language Models)
Run: python 01_rlm_basics.py

Demonstrates the basics of dspy.RLM for exploring large contexts
through an iterative REPL loop with sandboxed Python code execution.

Prerequisites:
- Deno installed (https://docs.deno.com/runtime/getting_started/installation/)
- OpenAI API key in .env
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# Step 1: Configure a strong LM (any capable model works with RLM)
lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

# Step 2: Basic RLM usage for large context Q&A
# RLM takes a signature just like Predict or ChainOfThought
rlm = dspy.RLM("context, question -> answer")

# Simulate a large document
large_document = """
Quarterly Financial Report - FY2024

Q1 Results:
- Revenue: $12.5M (up 8% YoY)
- Operating expenses: $9.2M
- Net income: $3.3M
- Key growth driver: Cloud services expansion

Q2 Results:
- Revenue: $14.1M (up 12% YoY)
- Operating expenses: $10.1M
- Net income: $4.0M
- Key growth driver: Enterprise contract wins

Q3 Results:
- Revenue: $15.8M (up 18% YoY)
- Operating expenses: $10.9M
- Net income: $4.9M
- Key growth driver: AI product launches

Q4 Results:
- Revenue: $17.2M (up 22% YoY)
- Operating expenses: $11.5M
- Net income: $5.7M
- Key growth driver: International expansion

Annual Summary:
- Total revenue: $59.6M (up 15% YoY)
- Total net income: $17.9M
- Employee count: 450 (up from 380)
- Customer retention rate: 94%
"""

result = rlm(
    context=large_document,
    question="What was the total annual revenue and which quarter had the highest growth?",
)
print("=== Basic RLM Usage ===")
print(f"Answer: {result.answer}")

# Step 3: Inspect the trajectory (what code the LLM executed)
print("\n=== Trajectory ===")
if hasattr(result, "trajectory") and result.trajectory:
    for i, step in enumerate(result.trajectory):
        print(f"\nStep {i + 1}:")
        print(f"  Code: {step.get('code', 'N/A')[:200]}")
        print(f"  Output: {step.get('output', 'N/A')[:200]}")


# Step 4: RLM with max_iterations control
print("\n=== RLM with max_iterations ===")
rlm_controlled = dspy.RLM(
    "context, question -> answer",
    max_iterations=5,  # Limit exploration steps
    verbose=True,  # Print REPL steps in real-time
)

result2 = rlm_controlled(
    context=large_document,
    question="Calculate the average quarterly net income.",
)
print(f"Answer: {result2.answer}")


# Step 5: RLM with sub_lm for cost efficiency
print("\n=== RLM with Sub-LM ===")
cheap_lm = dspy.LM("openai/gpt-4o-mini")

rlm_with_sub = dspy.RLM(
    "context, question -> answer",
    sub_lm=cheap_lm,  # Cheaper LM handles llm_query() calls
)

result3 = rlm_with_sub(
    context=large_document,
    question="Which quarter showed the strongest net income growth?",
)
print(f"Answer: {result3.answer}")


# Step 6: RLM with typed outputs
print("\n=== RLM with Typed Outputs ===")
rlm_typed = dspy.RLM(
    "logs -> error_count: int, critical_errors: list[str]",
)

sample_logs = """
[2024-01-15 10:23:45] INFO: Server started on port 8080
[2024-01-15 10:24:01] ERROR: Database connection timeout after 30s
[2024-01-15 10:24:15] WARN: Retry attempt 1 for database connection
[2024-01-15 10:24:30] ERROR: Database connection failed - CRITICAL
[2024-01-15 10:25:00] INFO: Fallback to read replica activated
[2024-01-15 10:26:12] ERROR: Authentication service unreachable
[2024-01-15 10:27:00] INFO: Authentication service recovered
[2024-01-15 10:28:45] ERROR: Disk space below 5% - CRITICAL
"""

result4 = rlm_typed(logs=sample_logs)
print(f"Error count: {result4.error_count}")
print(f"Critical errors: {result4.critical_errors}")
