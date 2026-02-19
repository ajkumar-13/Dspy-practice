# 7.3: Ensemble: Combining Programs

## Introduction

In machine learning, **ensemble methods** are one of the most reliable ways to squeeze extra performance out of a system. Train five models instead of one, aggregate their predictions, and you'll almost always do better. The same principle applies to DSPy programs.

**`dspy.Ensemble`** lets you combine multiple DSPy programs, each potentially optimized with different seeds, optimizers, or hyperparameters, into a single, more robust program. It's the DSPy equivalent of bagging, and it's remarkably simple to use.

---

## What You'll Learn

- How `dspy.Ensemble` combines multiple programs into one
- Strategies for generating diverse program variants
- How to configure full-set vs. sampled ensembles
- Voting and aggregation approaches
- Combining Ensemble with other optimizers for maximum performance

---

## Prerequisites

- Completed [7.2: BetterTogether](../7.2-better-together/blog.md)
- Familiarity with at least one optimizer (BootstrapRS, MIPROv2)

---

## Why Ensemble?

When you run MIPROv2 or BootstrapRS, the result depends on randomness: which training examples are sampled, which instructions are proposed, which search paths are explored. Run the same optimizer twice with different random seeds and you'll get different optimized programs, each with its own strengths and weaknesses.

An ensemble exploits this **diversity**: where one variant fails, others may succeed. The combined system is more robust than any individual variant.

---

## How dspy.Ensemble Works

`dspy.Ensemble` takes a list of programs and wraps them in a single program that runs all (or a subset) and aggregates results:

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Define the base program
class QA(dspy.Module):
    def __init__(self):
        self.answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.answer(question=question)

# Training data
trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="Who wrote Hamlet?", answer="Shakespeare").with_inputs("question"),
    dspy.Example(question="What is the speed of light in m/s?", answer="299792458").with_inputs("question"),
    # ... more examples
]

# Metric
def exact_match(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()

# Step 1: Optimize N variants with different seeds
programs = []
for seed in range(5):
    optimizer = dspy.BootstrapFewShotWithRandomSearch(
        metric=exact_match,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        num_candidate_programs=8,
        seed=seed,
    )
    optimized = optimizer.compile(QA(), trainset=trainset)
    programs.append(optimized)

# Step 2: Ensemble the variants
ensemble = dspy.Ensemble(reduce_fn=dspy.majority, size=5)
ensembled_program = ensemble.compile(programs)
```

The `ensembled_program` now runs all five variants on each input and uses **majority voting** to select the final answer.

---

## Ensemble Parameters

### `reduce_fn`: How to Aggregate

The `reduce_fn` parameter controls how outputs from multiple programs are combined:

- **`dspy.majority`**: Returns the most common answer (majority vote). Best for classification, short factual answers, and any task with discrete outputs.

```python
# Majority voting: great for classification
ensemble = dspy.Ensemble(reduce_fn=dspy.majority, size=5)
```

For tasks with open-ended text outputs where exact matching doesn't apply, you can write a custom reduce function:

```python
def pick_longest(predictions):
    """Select the longest response (proxy for most detailed)."""
    return max(predictions, key=lambda p: len(p.answer))

ensemble = dspy.Ensemble(reduce_fn=pick_longest, size=3)
```

### `size`: How Many Programs to Run

You don't have to run all programs for every input:

```python
# Run all 5 programs
ensemble_full = dspy.Ensemble(reduce_fn=dspy.majority, size=5)

# Randomly sample 3 of 5 for each input (cheaper, still robust)
ensemble_sampled = dspy.Ensemble(reduce_fn=dspy.majority, size=3)
```

Sampling a subset trades a small amount of robustness for proportional cost savings. In practice, sampling 3 of 5 retains most of the ensemble benefit at 60% of the cost.

---

## Generating Diverse Variants

The power of an ensemble depends on **diversity**. Here are strategies for creating meaningfully different program variants:

### Strategy 1: Different Random Seeds

The simplest approach. Each optimization run explores different parts of the search space:

```python
programs = []
for seed in range(5):
    opt = dspy.BootstrapFewShotWithRandomSearch(metric=metric, seed=seed)
    programs.append(opt.compile(MyProgram(), trainset=trainset))
```

### Strategy 2: Different Optimizers

Combine programs optimized with fundamentally different strategies:

```python
# Variant 1: BootstrapRS (demonstration-focused)
opt1 = dspy.BootstrapFewShotWithRandomSearch(metric=metric)
prog1 = opt1.compile(MyProgram(), trainset=trainset)

# Variant 2: MIPROv2 (instruction + demonstration)
opt2 = dspy.MIPROv2(metric=metric, auto="medium")
prog2 = opt2.compile(MyProgram(), trainset=trainset)

programs = [prog1, prog2]
```

### Strategy 3: Different Hyperparameters

Vary demo counts, search budgets, or other parameters:

```python
for max_demos in [2, 4, 8]:
    opt = dspy.BootstrapFewShotWithRandomSearch(
        metric=metric,
        max_bootstrapped_demos=max_demos,
    )
    programs.append(opt.compile(MyProgram(), trainset=trainset))
```

---

## Combining Ensemble with Fine-Tuning

A powerful pattern is to ensemble **both prompted and fine-tuned** variants:

```python
# Variant 1: Prompt-optimized on GPT-4o-mini
opt = dspy.MIPROv2(metric=metric, auto="medium")
prompted = opt.compile(MyProgram(), trainset=trainset)

# Variant 2: Fine-tuned GPT-4o-mini
student = MyProgram()
student.set_lm(dspy.LM("openai/gpt-4o-mini"))
teacher = MyProgram()
finetune = dspy.BootstrapFinetune(metric=metric)
finetuned = finetune.compile(student, trainset=trainset, teacher=teacher)

# Ensemble: prompted and fine-tuned complement each other
ensemble = dspy.Ensemble(reduce_fn=dspy.majority, size=2)
combined = ensemble.compile([prompted, finetuned])
```

Prompted models and fine-tuned models often make **different types of errors**, so ensembling them provides genuine complementary value.

---

## Key Takeaways

- **`dspy.Ensemble` combines multiple DSPy programs** into a single, more robust program using aggregation.
- **Majority voting** is the default and works well for classification and factual tasks. Custom reduce functions handle open-ended outputs.
- **Diversity is key**: ensemble programs optimized with different seeds, optimizers, or hyperparameters for maximum benefit.
- **Sampling subsets** (e.g., 3 of 5) offers a practical cost-robustness tradeoff.
- **Combining prompted and fine-tuned variants** is a powerful pattern; they make complementary errors.
- **Ensembling is cheap insurance**: if you've already optimized N variants, ensembling them costs nothing extra at optimization time.

---

## Next Up

You've now learned all three pillars of Phase 7: distillation with BootstrapFinetune, joint optimization with BetterTogether, and robustness through Ensemble. In the project blog, you'll put it all together to build a **complete model distillation pipeline** from scratch.

**[7.P: Project: Model Distillation Pipeline →](../7.P-project-distillation/blog.md)**

---

## Resources

- 📖 [Ensemble API Reference](https://dspy.ai/api/optimizers/Ensemble/)
- 📖 [DSPy Optimizers Overview](https://dspy.ai/learn/optimization/optimizers/)
- 💻 [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)

---
