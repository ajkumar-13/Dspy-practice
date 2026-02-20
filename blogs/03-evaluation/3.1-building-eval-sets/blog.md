# 3.1: Building Evaluation Sets

## Introduction

Here's a truth that separates hobby projects from production LM applications: **you can't improve what you can't measure.** And you can't measure without data.

DSPy's optimization engine, the thing that automatically improves your prompts, selects better demonstrations, and tunes instructions, runs on evaluation data. No data, no optimization. Bad data, bad optimization. The quality of your evaluation set directly determines the quality of your optimized program.

In this post, you'll learn how to build evaluation sets that power DSPy's self-improving capabilities. We'll cover the core data container (`dspy.Example`), loading data from various sources, splitting strategies that differ significantly from traditional ML, and practical guidelines for how much data you actually need.

---

## What You'll Learn

- How `dspy.Example` works as DSPy's core data container
- How to mark input vs. output fields with `.with_inputs()`
- Loading data from JSON files, CSVs, and HuggingFace datasets
- Data splitting strategies for different optimizers (and why they differ from traditional ML)
- Labeling guidelines: when you need labels, and when you don't
- Practical tips for dataset size and composition

---

## Prerequisites

- Completed [Phase 2: Structured Outputs](../../02-structured-outputs/2.1-typed-predictors/blog.md)
- DSPy installed (`uv add dspy python-dotenv datasets`)
- Basic familiarity with `dspy.Predict` and `dspy.ChainOfThought`

---

## Understanding dspy.Example

Everything in DSPy's evaluation and optimization pipeline revolves around `dspy.Example`. It's a lightweight, dictionary-like container that holds input-output pairs for your task. Think of it as a single row in your dataset.

```python
import dspy

# Create an example with keyword arguments
example = dspy.Example(
    question="What is the capital of France?",
    answer="Paris"
)

# Access fields like attributes
print(example.question)  # "What is the capital of France?"
print(example.answer)    # "Paris"
```

But here's the critical part: DSPy needs to know which fields are **inputs** (what gets passed to your program) and which are **labels** (what you expect as output, used for evaluation). You mark inputs with `.with_inputs()`:

```python
example = dspy.Example(
    question="What is the capital of France?",
    answer="Paris"
).with_inputs("question")
```

Now DSPy knows that `question` is an input field and `answer` is the expected output. This distinction matters because:

1. **During evaluation**, DSPy passes only the input fields to your program, then compares the program's output against the label fields using your metric.
2. **During optimization**, DSPy uses labeled examples as demonstrations or training signal. The optimizer needs to know what to feed vs. what to check.

You can mark multiple fields as inputs:

```python
example = dspy.Example(
    context="France is a country in Western Europe. Its capital is Paris.",
    question="What is the capital of France?",
    answer="Paris"
).with_inputs("context", "question")
```

### Building a Dataset by Hand

For quick experiments, just create a list of examples:

```python
devset = [
    dspy.Example(
        question="What is the capital of France?",
        answer="Paris"
    ).with_inputs("question"),
    dspy.Example(
        question="Who wrote Romeo and Juliet?",
        answer="William Shakespeare"
    ).with_inputs("question"),
    dspy.Example(
        question="What is the speed of light?",
        answer="299,792,458 meters per second"
    ).with_inputs("question"),
]
```

This works, but for serious evaluation you'll want to load data from files or existing datasets.

---

## Loading Data from Files

### From JSON

The most common format. Structure your JSON as a list of objects:

```python
import json

def load_from_json(filepath, input_fields):
    """Load a JSON file into a list of dspy.Examples."""
    with open(filepath, "r") as f:
        data = json.load(f)

    examples = []
    for item in data:
        example = dspy.Example(**item).with_inputs(*input_fields)
        examples.append(example)

    return examples

# Usage: assumes data.json contains:
# [{"question": "...", "answer": "..."}, ...]
devset = load_from_json("data.json", input_fields=["question"])
```

### From CSV

```python
import csv

def load_from_csv(filepath, input_fields):
    """Load a CSV file into a list of dspy.Examples."""
    examples = []
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            example = dspy.Example(**row).with_inputs(*input_fields)
            examples.append(example)
    return examples
```

### From Pandas

```python
import pandas as pd

def load_from_dataframe(df, input_fields):
    """Convert a pandas DataFrame to a list of dspy.Examples."""
    return [
        dspy.Example(**row.to_dict()).with_inputs(*input_fields)
        for _, row in df.iterrows()
    ]
```

---

## Loading from HuggingFace

DSPy provides a built-in `DataLoader` for HuggingFace datasets, which is the fastest way to get started with established benchmarks:

```python
from dspy.datasets import DataLoader

dl = DataLoader()

# Load a dataset from HuggingFace
dataset = dl.from_huggingface(
    "hotpot_qa",                    # dataset name on HuggingFace
    split="validation",             # which split to load
    input_keys=["question"],        # fields to mark as inputs
    fields=["question", "answer"],  # fields to keep (optional, filters columns)
)

print(f"Loaded {len(dataset)} examples")
print(dataset[0])
```

You can also load directly with the `datasets` library and convert manually, which gives you more control:

```python
from datasets import load_dataset

# Load HotPotQA
raw = load_dataset("hotpot_qa", "fullwiki", split="validation[:200]")

devset = [
    dspy.Example(
        question=row["question"],
        answer=row["answer"],
    ).with_inputs("question")
    for row in raw
]

print(f"Loaded {len(devset)} examples")
print(f"Sample: {devset[0].question}")
```

---

## Data Splitting Strategies

This is where DSPy diverges significantly from traditional machine learning. In ML, you typically use 80% of your data for training and 20% for validation. **In DSPy, the split ratios are often reversed**, and the reason is fundamental to how prompt optimization works.

### For Prompt Optimizers (BootstrapFewShot, MIPROv2)

Prompt optimizers use the **training set** to select few-shot demonstrations and the **validation set** to evaluate which demonstrations work best. Since you want robust evaluation, you need a larger validation set:

```python
import random

random.seed(42)
all_examples = load_from_json("data.json", input_fields=["question"])
random.shuffle(all_examples)

# 20% train, 80% validation: reversed from traditional ML!
split_point = int(len(all_examples) * 0.2)
trainset = all_examples[:split_point]
devset = all_examples[split_point:]

print(f"Train: {len(trainset)} examples (for demo selection)")
print(f"Dev:   {len(devset)} examples (for evaluation)")
```

Why? The training set provides *candidates* for few-shot demonstrations. You don't need thousands; even 10-20 good examples can yield excellent demonstrations. But the validation set determines whether those demonstrations actually help, so it needs to be large enough to give a reliable signal.

### For GEPA Optimizer

GEPA is a reflective prompt evolution optimizer that works differently from prompt optimizers. It maximizes the training set and only needs enough validation data to confirm improvements:

```python
# For GEPA: maximize training, keep validation just large enough
split_point = int(len(all_examples) * 0.8)
trainset = all_examples[:split_point]
devset = all_examples[split_point:]
```

### Quick Reference

| Optimizer | Train % | Dev % | Why |
|-----------|---------|-------|-----|
| BootstrapFewShot | 20% | 80% | Need large dev for reliable evaluation |
| MIPROv2 | 20% | 80% | Same: evaluates many candidate prompts |
| GEPA | 80% | 20% | Needs more training signal, less validation |

---

## Labeling Guidelines

Not every task needs fully labeled data. DSPy's flexibility here is one of its strengths.

### Input-Only (No Labels)

Some optimizers and evaluation approaches only need inputs. For example, if your metric uses an AI judge that doesn't compare against a gold answer:

```python
# Labeled examples not required for every use case
unlabeled = [
    dspy.Example(question="Explain quantum computing").with_inputs("question"),
    dspy.Example(question="What causes inflation?").with_inputs("question"),
]
```

### Input + Output (Fully Labeled)

For most evaluation and optimization, you want gold-standard outputs to compare against:

```python
labeled = [
    dspy.Example(
        question="What is photosynthesis?",
        answer="Photosynthesis is the process by which plants convert sunlight, "
               "water, and carbon dioxide into glucose and oxygen."
    ).with_inputs("question"),
]
```

### Practical Labeling Tips

- **Start small**: even 20 well-chosen examples are enough to begin iterating.
- **Aim for 200+**: this gives you statistical confidence in your evaluation results.
- **Use domain-specific data**: generic benchmarks are useful for sanity checks, but your *actual* task data will reveal the real failure modes.
- **Label iteratively**: run your program, find failures, add those as labeled examples. Your evaluation set grows organically from real errors.

---

## Key Takeaways

- **`dspy.Example`** is the core data container. Construct with keyword arguments, mark inputs with `.with_inputs("field1", "field2")`.
- **Load from anywhere**: JSON, CSV, pandas, HuggingFace. The `DataLoader` class or manual conversion both work well.
- **Split ratios depend on your optimizer.** For prompt optimizers like BootstrapFewShot and MIPROv2, use 20% train / 80% dev. For GEPA, reverse it.
- **Start small, grow iteratively.** 20 examples to start, 200+ for robust evaluation.
- **Domain-specific data matters.** Generic benchmarks are a starting point, not the finish line.

---

## Next Up

You have data. Now you need to define *what "good" looks like.* In the next post, we'll dive into DSPy's metric system, from simple exact match to sophisticated AI-feedback judges that can evaluate subjective quality.

**[3.2: Defining Metrics →](../3.2-defining-metrics/blog.md)**

---

## Resources

- [DSPy Data Documentation](https://dspy.ai/learn/evaluation/data/)
- [DSPy Evaluation Overview](https://dspy.ai/learn/evaluation/overview/)
- [HuggingFace Datasets Library](https://huggingface.co/docs/datasets/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [Code examples for this post](code/)
