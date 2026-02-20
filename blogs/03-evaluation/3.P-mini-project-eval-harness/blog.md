# 3.P: Mini-Project: Building an Evaluation Harness

## Introduction

You've learned to build evaluation sets, define metrics, and run evaluations. Now let's put it all together into something you'll actually reuse: a **modular evaluation harness** that loads a dataset, defines multiple metrics, benchmarks different program variants side by side, and saves results for tracking over time.

This isn't just an exercise, it's a template. Every serious DSPy project needs an evaluation harness, and the one we build here will serve as your starting point for every project in the rest of this series.

---

## Project Overview

We'll build a harness that:

1. Loads a QA dataset (a HotPotQA subset)
2. Defines three metrics: exact match, passage match, and SemanticF1
3. Builds three program variants: `Predict`, `ChainOfThought`, and a custom two-step module
4. Evaluates all combinations in a systematic grid
5. Displays a comparison table
6. Saves results to JSON for historical tracking

---

## Step 1: Load the Dataset

We'll use HotPotQA, a well-known multi-hop QA dataset, and take a small subset for fast iteration. In your own projects, swap this for your domain-specific data.

```python
import dspy
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


def load_hotpotqa(num_examples=50):
    """Load a subset of HotPotQA for evaluation."""
    from datasets import load_dataset

    raw = load_dataset("hotpot_qa", "fullwiki", split=f"validation[:{num_examples}]")

    examples = [
        dspy.Example(
            question=row["question"],
            answer=row["answer"],
        ).with_inputs("question")
        for row in raw
    ]

    print(f"Loaded {len(examples)} examples")
    print(f"Sample: {examples[0].question} → {examples[0].answer}")
    return examples


devset = load_hotpotqa(50)
```

For faster development, start with just 50 examples. Once your harness is working, scale up to 200+.

---

## Step 2: Define Multiple Metrics

A single metric tells one story. Multiple metrics reveal the full picture. We'll define three that capture different aspects of answer quality:

```python
from dspy.evaluate import answer_exact_match, answer_passage_match, SemanticF1

# 1. Exact match: strictest, good for factoid QA
def metric_exact_match(example, pred, trace=None):
    return answer_exact_match(example, pred)

# 2. Passage match: more forgiving, checks if answer appears in response
def metric_passage_match(example, pred, trace=None):
    return answer_passage_match(example, pred)

# 3. SemanticF1: best for open-ended QA, token-level overlap
semantic_f1 = SemanticF1()

def metric_semantic_f1(example, pred, trace=None):
    return semantic_f1(example, pred)

# Bundle them for easy iteration
METRICS = {
    "exact_match": metric_exact_match,
    "passage_match": metric_passage_match,
    "semantic_f1": metric_semantic_f1,
}
```

### Optional: Add an AI-Feedback Metric

If you want to include a quality judgment beyond token matching, add an AI judge. Note that this makes evaluation slower and more expensive:

```python
class AssessAnswer(dspy.Signature):
    """Assess whether the predicted answer is correct given the question and reference answer."""

    question: str = dspy.InputField()
    reference_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    is_correct: bool = dspy.OutputField(desc="Whether the predicted answer is factually correct")


assess = dspy.Predict(AssessAnswer)


def metric_ai_judge(example, pred, trace=None):
    result = assess(
        question=example.question,
        reference_answer=example.answer,
        predicted_answer=pred.answer,
    )
    return result.is_correct


# Uncomment to include AI judge in metrics:
# METRICS["ai_judge"] = metric_ai_judge
```

---

## Step 3: Build Program Variants

Now let's define three different program architectures to compare. Same signature, different execution strategies:

```python
# Shared signature
class AnswerQuestion(dspy.Signature):
    """Answer the question concisely and accurately."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="A concise, factual answer")


# Variant 1: Simple Predict
predict_program = dspy.Predict(AnswerQuestion)

# Variant 2: ChainOfThought
cot_program = dspy.ChainOfThought(AnswerQuestion)


# Variant 3: Custom two-step module (decompose then answer)
class DecomposeAndAnswer(dspy.Module):
    def __init__(self):
        self.decompose = dspy.ChainOfThought(
            "question -> sub_questions: list[str]"
        )
        self.answer = dspy.ChainOfThought(AnswerQuestion)

    def forward(self, question):
        # Step 1: Break down the question
        decomposed = self.decompose(question=question)
        sub_qs = ", ".join(decomposed.sub_questions)

        # Step 2: Answer with context of sub-questions
        enriched_question = f"{question} (Consider: {sub_qs})"
        result = self.answer(question=enriched_question)
        return dspy.Prediction(answer=result.answer)


decompose_program = DecomposeAndAnswer()

# Bundle them
PROGRAMS = {
    "Predict": predict_program,
    "ChainOfThought": cot_program,
    "DecomposeAndAnswer": decompose_program,
}
```

---

## Step 4: Run Evaluations

Here's the core of the harness, a function that evaluates every program against every metric and collects the results:

```python
def run_evaluation_grid(programs, metrics, devset, num_threads=16):
    """Evaluate all programs against all metrics. Returns a results dict."""
    results = {}

    evaluator = dspy.Evaluate(
        devset=devset,
        num_threads=num_threads,
        display_progress=True,
        display_table=0,  # suppress table per-run, we'll build our own
    )

    for prog_name, program in programs.items():
        results[prog_name] = {}
        for metric_name, metric_fn in metrics.items():
            print(f"\nEvaluating {prog_name} with {metric_name}...")
            start_time = time.time()

            score = evaluator(program, metric=metric_fn)

            elapsed = time.time() - start_time
            results[prog_name][metric_name] = {
                "score": round(score, 4),
                "time_seconds": round(elapsed, 2),
            }
            print(f"  Score: {score:.2%} ({elapsed:.1f}s)")

    return results
```

Run it:

```python
results = run_evaluation_grid(PROGRAMS, METRICS, devset)
```

---

## Step 5: Compare and Analyze

Now let's format the results into a readable comparison table:

```python
def print_comparison_table(results):
    """Print a formatted comparison table of evaluation results."""
    # Collect all metric names
    metric_names = list(next(iter(results.values())).keys())

    # Header
    header = f"{'Program':<22}"
    for m in metric_names:
        header += f"  {m:>15}"
    header += f"  {'Avg':>8}"

    print("\n" + "=" * len(header))
    print("EVALUATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Rows
    for prog_name, metrics in results.items():
        row = f"{prog_name:<22}"
        scores = []
        for m in metric_names:
            score = metrics[m]["score"]
            scores.append(score)
            row += f"  {score:>14.2%}"
        avg = sum(scores) / len(scores)
        row += f"  {avg:>7.2%}"
        print(row)

    print("=" * len(header))

    # Best program per metric
    print("\nBest per metric:")
    for m in metric_names:
        best_prog = max(results, key=lambda p: results[p][m]["score"])
        best_score = results[best_prog][m]["score"]
        print(f"  {m}: {best_prog} ({best_score:.2%})")


print_comparison_table(results)
```

Expected output (scores will vary):

```
==========================================================
EVALUATION RESULTS
==========================================================
Program                  exact_match   passage_match     semantic_f1       Avg
----------------------------------------------------------
Predict                       42.00%          56.00%          61.30%    53.10%
ChainOfThought                48.00%          64.00%          68.50%    60.17%
DecomposeAndAnswer            46.00%          62.00%          66.80%    58.27%
==========================================================

Best per metric:
  exact_match: ChainOfThought (48.00%)
  passage_match: ChainOfThought (64.00%)
  semantic_f1: ChainOfThought (68.50%)
```

---

## Step 6: Save Results

Save your results to JSON so you can track performance over time. Each run gets a timestamp, making it easy to see how your programs improve as you iterate:

```python
def save_results(results, filepath="eval_results.json"):
    """Append evaluation results to a JSON file for historical tracking."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "num_examples": len(devset),
        "model": "openai/gpt-4o-mini",
        "results": results,
    }

    # Load existing history or start fresh
    try:
        with open(filepath, "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.append(record)

    with open(filepath, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to {filepath}")
    print(f"Total runs tracked: {len(history)}")


save_results(results, "eval_results.json")
```

After several iterations, your `eval_results.json` becomes a log of progress:

```json
[
  {
    "timestamp": "2026-02-15T10:30:00",
    "num_examples": 50,
    "model": "openai/gpt-4o-mini",
    "results": {
      "Predict": {"exact_match": {"score": 0.42}, "...": "..."},
      "ChainOfThought": {"exact_match": {"score": 0.48}, "...": "..."}
    }
  }
]
```

---

## Reusable Harness Template

Here's the full harness condensed into a reusable function you can drop into any project:

```python
import dspy
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from dspy.evaluate import SemanticF1

load_dotenv()


def run_harness(programs, metrics, devset, save_path=None, num_threads=16):
    """
    Reusable evaluation harness.

    Args:
        programs: dict of {name: dspy.Module}
        metrics: dict of {name: metric_fn}
        devset: list of dspy.Example
        save_path: optional JSON path to save results
        num_threads: parallel evaluation threads
    """
    evaluator = dspy.Evaluate(
        devset=devset,
        num_threads=num_threads,
        display_progress=True,
        display_table=0,
    )

    results = {}
    for prog_name, program in programs.items():
        results[prog_name] = {}
        for metric_name, metric_fn in metrics.items():
            print(f"  {prog_name} × {metric_name}...", end=" ")
            start = time.time()
            score = evaluator(program, metric=metric_fn)
            elapsed = time.time() - start
            results[prog_name][metric_name] = {
                "score": round(score, 4),
                "time_seconds": round(elapsed, 2),
            }
            print(f"{score:.2%} ({elapsed:.1f}s)")

    # Print comparison
    metric_names = list(next(iter(results.values())).keys())
    print(f"\n{'Program':<22}" + "".join(f"  {m:>15}" for m in metric_names))
    print("-" * (22 + 17 * len(metric_names)))
    for prog, mets in results.items():
        row = f"{prog:<22}" + "".join(f"  {mets[m]['score']:>14.2%}" for m in metric_names)
        print(row)

    # Save if requested
    if save_path:
        record = {
            "timestamp": datetime.now().isoformat(),
            "num_examples": len(devset),
            "results": results,
        }
        try:
            with open(save_path, "r") as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []
        history.append(record)
        with open(save_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nSaved to {save_path}")

    return results
```

Usage in future projects:

```python
# Set up LM
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Define your programs and metrics
programs = {"baseline": my_program, "optimized": my_optimized_program}
metrics = {"semantic_f1": SemanticF1(), "my_custom": my_metric}

# Run
results = run_harness(programs, metrics, my_devset, save_path="results.json")
```

---

## What We Learned

This mini-project tied together everything from Phase 3:

- **Evaluation sets** (3.1): we loaded HotPotQA and created a devset of `dspy.Example` objects with `.with_inputs()`
- **Metrics** (3.2): we defined three metrics (exact match, passage match, SemanticF1) and optionally an AI judge
- **Running evaluations** (3.3): we used `dspy.Evaluate` with parallel threads and built a comparison table
- **Reusable patterns**: the harness template works for any task: swap the dataset, metrics, and programs

The key insight: **evaluation is infrastructure, not overhead.** Every minute you spend building a solid evaluation harness pays back tenfold when you start optimizing in Phase 4, because the optimizer is only as good as the evaluation that guides it.

---

## Next Up

Phase 3 is complete. You can build datasets, define metrics, run evaluations, and compare program variants systematically. In Phase 4, we'll use this evaluation infrastructure to automatically optimize your DSPy programs, starting with BootstrapFewShot, then moving to MIPROv2 and GEPA.

**[4.1: BootstrapFewShot & Random Search →](../../04-optimization/4.1-bootstrap-rs/blog.md)**

---

## Resources

- [DSPy Evaluation Documentation](https://dspy.ai/learn/evaluation/overview/)
- [DSPy Metrics Reference](https://dspy.ai/learn/evaluation/metrics/)
- [HotPotQA Dataset](https://hotpotqa.github.io/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [Code examples for this post](code/)
