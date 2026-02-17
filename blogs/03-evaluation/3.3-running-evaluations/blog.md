# 3.3: Running Evaluations

## Introduction

You've built evaluation sets. You've defined metrics. Now it's time to actually *run* evaluations and make sense of the results. This is where the rubber meets the road: systematic evaluation is the foundation that everything in DSPy's optimization pipeline is built on.

DSPy provides `dspy.Evaluate`, a utility that handles parallel execution, progress tracking, result display, and score aggregation. But before we get to that, let's start with the basics, because understanding the simple loop helps you understand what the utility automates.

---

## What You'll Learn

- How to write a simple evaluation loop from scratch
- How to use `dspy.Evaluate` for parallel, production-grade evaluation
- How to interpret results and identify failure patterns
- How to track LM costs during evaluation
- The iterative development loop: evaluate → diagnose → improve → re-evaluate

---

## Prerequisites

- Completed [3.2: Defining Metrics](../3.2-defining-metrics/blog.md)
- A configured LM and a dataset with labeled examples

---

## Simple Evaluation Loop

Before using any utility, it's worth understanding what evaluation actually does. The loop is straightforward:

```python
import dspy
from dotenv import load_dotenv

load_dotenv()
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# A simple QA program
class QA(dspy.Signature):
    """Answer the question concisely."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


program = dspy.Predict(QA)

# A small evaluation set
devset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="Who painted the Mona Lisa?", answer="Leonardo da Vinci").with_inputs("question"),
    dspy.Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
    dspy.Example(question="What planet is closest to the Sun?", answer="Mercury").with_inputs("question"),
    dspy.Example(question="Who wrote Hamlet?", answer="William Shakespeare").with_inputs("question"),
]

# Simple metric
def exact_match(example, pred, trace=None):
    return example.answer.lower().strip() == pred.answer.lower().strip()

# Manual evaluation loop
scores = []
for example in devset:
    pred = program(**example.inputs())
    score = exact_match(example, pred)
    scores.append(score)
    print(f"Q: {example.question}")
    print(f"  Expected: {example.answer}")
    print(f"  Got:      {pred.answer}")
    print(f"  Score:    {score}")
    print()

avg_score = sum(scores) / len(scores)
print(f"Average score: {avg_score:.2%}")
```

This works fine for 5 examples. But with 200+ examples (and you should have 200+) you need parallel execution, progress tracking, and structured output. That's what `dspy.Evaluate` provides.

---

## Using dspy.Evaluate

`dspy.Evaluate` is the standard way to benchmark DSPy programs. It handles threading, progress display, and result aggregation:

```python
# Create the evaluator
evaluator = dspy.Evaluate(
    devset=devset,
    num_threads=24,
    display_progress=True,
    display_table=5,        # show first 5 results in a table
)

# Run evaluation: returns the average score
avg_score = evaluator(program, metric=exact_match)
print(f"\nAverage score: {avg_score:.2%}")
```

Let's break down the parameters:

- **`devset`**: your list of `dspy.Example` objects
- **`num_threads`**: how many examples to evaluate in parallel. Set this high (16-32) for API-based LMs to speed things up dramatically
- **`display_progress`**: shows a progress bar during evaluation
- **`display_table`**: after evaluation, displays a table of the first N results with inputs, expected outputs, predictions, and scores

The return value is the **average metric score** across all examples. If your metric returns `bool`, this is the accuracy (percentage of `True`). If it returns `float`, this is the mean score.

### Using with SemanticF1

For long-form QA, swap in a more forgiving metric:

```python
from dspy.evaluate import SemanticF1

semantic_f1 = SemanticF1()

evaluator = dspy.Evaluate(
    devset=devset,
    num_threads=24,
    display_progress=True,
    display_table=5,
)

avg_f1 = evaluator(program, metric=semantic_f1)
print(f"Average SemanticF1: {avg_f1:.2%}")
```

---

## Interpreting Results

Running the evaluator gives you a number. But a number alone doesn't tell you *why* your program is succeeding or failing. Here's how to dig deeper.

### Inspect Individual Failures

The display table shows you the first N results, but you can also collect results manually for deeper analysis:

```python
# Collect detailed results
results = []
for example in devset:
    pred = program(**example.inputs())
    score = exact_match(example, pred)
    results.append({
        "question": example.question,
        "expected": example.answer,
        "predicted": pred.answer,
        "score": score,
    })

# Filter to failures
failures = [r for r in results if not r["score"]]
print(f"\n{len(failures)} failures out of {len(results)} examples:\n")
for f in failures:
    print(f"  Q: {f['question']}")
    print(f"  Expected: {f['expected']}")
    print(f"  Got:      {f['predicted']}")
    print()
```

### Common Failure Patterns

Look for these recurring issues:

1. **Formatting mismatches**: the model says "The answer is Paris" instead of "Paris". Fix with a stricter signature description or a more forgiving metric.
2. **Ambiguous questions**: the question has multiple valid answers. Consider using `SemanticF1` or a passage-match metric.
3. **Knowledge gaps**: the model simply doesn't know the answer. Consider adding retrieval (RAG).
4. **Overly verbose responses**: the model gives a paragraph when you want a word. Add a constraint to your signature description.

---

## Tracking Costs

LM calls cost money. During evaluation, you're making potentially hundreds of API calls. Track the cost so there are no surprises:

```python
# After running evaluation, check the cost
total_cost = sum(
    x["cost"] for x in lm.history if x.get("cost") is not None
)
print(f"Total LM cost: ${total_cost:.4f}")
print(f"Number of LM calls: {len(lm.history)}")
print(f"Average cost per call: ${total_cost / max(len(lm.history), 1):.6f}")
```

For a rough estimate before running: if you're using `gpt-4o-mini` at ~\$0.15 per million input tokens, evaluating 200 short QA examples costs roughly \$0.01–0.05. Very affordable. But if you're using an AI-feedback metric that makes *additional* LM calls per example, costs multiply. Factor that in.

---

## Iterative Development

Evaluation isn't a one-time event. It's a continuous loop that drives improvement:

```
┌─────────────┐
│ 1. Evaluate │ ← Run dspy.Evaluate with your metric
└──────┬──────┘
       │
┌──────▼──────────┐
│ 2. Diagnose     │ ← Inspect failures, find patterns
└──────┬──────────┘
       │
┌──────▼──────────┐
│ 3. Improve      │ ← Change signature, module, or metric
└──────┬──────────┘
       │
┌──────▼──────────┐
│ 4. Re-evaluate  │ ← Confirm improvement, repeat
└──────┬──────────┘
       │
       └──→ back to step 1
```

Here's the workflow in practice:

```python
# Step 1: Baseline evaluation
program_v1 = dspy.Predict(QA)
score_v1 = evaluator(program_v1, metric=exact_match)
print(f"Predict baseline: {score_v1:.2%}")

# Step 2: Diagnose: maybe the model needs to reason more?

# Step 3: Improve: try ChainOfThought
program_v2 = dspy.ChainOfThought(QA)
score_v2 = evaluator(program_v2, metric=exact_match)
print(f"ChainOfThought:   {score_v2:.2%}")

# Step 4: Compare
print(f"\nImprovement: {score_v2 - score_v1:+.2%}")
```

This manual iteration is valuable, but DSPy's optimizers (Phase 4) automate much of this loop. The key insight: **you need reliable evaluation before optimization makes sense.** If your metric doesn't capture what "good" means, the optimizer will optimize for the wrong thing.

### MLflow Integration

For more sophisticated tracking across runs, DSPy integrates with MLflow for tracing and observability:

```python
import mlflow

mlflow.dspy.autolog()

with mlflow.start_run(run_name="qa_baseline"):
    score = evaluator(program, metric=exact_match)
    mlflow.log_metric("exact_match", score)
    mlflow.log_param("module_type", "Predict")
```

This persists your evaluation results, making it easy to compare experiments over time.

---

## Key Takeaways

- **Start with the simple loop** to understand what evaluation does, then graduate to `dspy.Evaluate` for parallel execution and structured output.
- **`dspy.Evaluate`** is the workhorse: pass it a devset, metric, and program. It returns the average score and optionally displays a results table.
- **Inspect failures, not just scores.** The average tells you *how much* is wrong; individual failures tell you *what* is wrong.
- **Track costs** with `lm.history`, especially when using AI-feedback metrics that make extra LM calls.
- **Evaluation is iterative.** Evaluate → diagnose → improve → re-evaluate. This loop continues even after you start using optimizers.

---

## Next Up

You've mastered the evaluation workflow. Now let's put it all together in a hands-on mini-project: building a reusable evaluation harness that compares multiple programs with multiple metrics across different configurations.

**[3.P: Mini-Project: Evaluation Harness →](../3.P-mini-project-eval-harness/blog.md)**

---

## Resources

- 📖 [DSPy Evaluation Documentation](https://dspy.ai/learn/evaluation/overview/)
- 📖 [DSPy Metrics Reference](https://dspy.ai/learn/evaluation/metrics/)
- 📖 [MLflow DSPy Integration](https://mlflow.org/docs/latest/llms/dspy/index.html)
- 💻 [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- 📁 [Code examples for this post](code/)
