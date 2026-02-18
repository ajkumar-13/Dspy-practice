# 4.1: BootstrapRS: Your First Optimizer

## Introduction

You've built modules, defined metrics, and run evaluations. Now it's time to unlock DSPy's superpower: **automatic optimization**. Instead of hand-tuning prompts, you'll let DSPy discover what works best, systematically, reproducibly, and often shockingly well.

In this blog, we start with the optimizer you should reach for first: **BootstrapFewShotWithRandomSearch** (commonly called **BootstrapRS**). It's fast, cheap, and remarkably effective. Think of it as the "Adam optimizer" of DSPy, a sensible default that works across a wide range of tasks.

---

## What You'll Learn

- What optimization means in DSPy: tuning prompts and/or LM weights automatically
- How `BootstrapFewShot` and `BootstrapFewShotWithRandomSearch` work
- The teacher concept and how demonstrations are generated
- Key parameters and how to configure them
- Running your first optimization end-to-end
- Inspecting, saving, and loading optimized programs

---

## Prerequisites

- Completed Phase 3 (Evaluation & Metrics)
- A working metric function and a small training set (10-50 examples)

---

## What is DSPy Optimization?

In traditional ML, an optimizer adjusts model weights to minimize a loss function. In DSPy, optimization means something different but equally powerful: **automatically discovering the best prompts, instructions, and few-shot demonstrations** that maximize your metric.

Here's the key insight: you never write these prompts yourself. You define:

1. **A program**: your DSPy module (e.g., a `ChainOfThought` pipeline)
2. **A metric**: a function that scores outputs (e.g., accuracy, F1)
3. **A training set**: examples to learn from

The optimizer then searches for the combination of demonstrations and instructions that makes your program perform best on those examples.

| What Gets Tuned | How |
|-----------------|-----|
| Few-shot demonstrations | Picked from successful runs of your program |
| Instructions | Generated and scored (MIPROv2, covered in [4.2](../4.2-miprov2/blog.md)) |
| LM weights | Fine-tuned on curated examples (covered in Phase 7) |

BootstrapRS focuses on the first: **finding the best few-shot demonstrations**.

---

## BootstrapFewShot Basics

`BootstrapFewShot` is the simplest optimizer. It works in three steps:

1. **Run your program** on each training example using a "teacher" module
2. **Validate** each output against your metric
3. **Collect passing examples** as demonstrations and inject them into future prompts

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# A simple QA program
program = dspy.ChainOfThought("question -> answer")

# Training data
trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is 7 * 8?", answer="56").with_inputs("question"),
    # ... more examples
]

# Define a metric
def exact_match(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()

# Compile with BootstrapFewShot
bootstrap = dspy.BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=4, max_labeled_demos=4)
optimized = bootstrap.compile(program, trainset=trainset)
```

After compilation, `optimized` is a copy of your program with carefully selected few-shot demonstrations baked into each predictor's prompt. When you call `optimized(question="What is 3 + 5?")`, those demonstrations appear in the prompt, guiding the model toward the right output format and reasoning style.

### The Teacher Concept

The "teacher" is the module that generates candidate demonstrations. By default, **the teacher is your program itself**. It runs on training examples, and successful outputs become demonstrations.

But you can use a **larger, more capable model** as teacher:

```python
teacher_lm = dspy.LM("openai/gpt-4o")

bootstrap = dspy.BootstrapFewShot(
    metric=exact_match,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    teacher_settings=dict(lm=teacher_lm),
)
optimized = bootstrap.compile(program, trainset=trainset)
```

This is powerful: GPT-4o generates the demonstrations, but your production program might run on GPT-4o-mini. The smaller model gets the "wisdom" of the larger one encoded in its few-shot examples.

---

## BootstrapFewShotWithRandomSearch

`BootstrapFewShot` runs once and gives you a single optimized program. But what if a different random subset of demonstrations would work better? That's where **BootstrapFewShotWithRandomSearch** (BootstrapRS) comes in.

BootstrapRS runs `BootstrapFewShot` **multiple times** with different random seeds, generating several candidate programs. It then evaluates each candidate on your training set and keeps the best one.

```python
teleprompter = dspy.BootstrapFewShotWithRandomSearch(
    metric=exact_match,
    max_bootstrapped_demos=4,     # Max auto-generated demos per predictor
    max_labeled_demos=4,          # Max labeled demos from your trainset
    num_candidate_programs=8,     # Number of random candidates to try
    num_threads=4,                # Parallel threads for evaluation
)

optimized = teleprompter.compile(program, trainset=trainset)
```

### Key Parameters

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `max_bootstrapped_demos` | 4 | Max teacher-generated demonstrations per predictor |
| `max_labeled_demos` | 16 | Max examples pulled directly from your labeled trainset |
| `num_candidate_programs` | 16 | How many random configurations to try |
| `num_threads` | None | Parallelism for running evaluations |

### Typical Cost and Time

For a task with ~50 training examples and 8 candidate programs using GPT-4o-mini:

- **Cost:** ~\$1-3
- **Time:** ~5-15 minutes
- **LM calls:** ~400-800 (cached after first run)

This is remarkably cheap for what you get, often a 10-30% jump in task performance with zero manual prompt engineering.

---

## Running Your First Optimization

Here's a complete, self-contained example:

```python

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Build training set
trainset = [
    dspy.Example(question="What is the largest planet?", answer="Jupiter").with_inputs("question"),
    dspy.Example(question="Who wrote Romeo and Juliet?", answer="Shakespeare").with_inputs("question"),
    dspy.Example(question="What is the boiling point of water in Celsius?", answer="100").with_inputs("question"),
    dspy.Example(question="What is the chemical symbol for gold?", answer="Au").with_inputs("question"),
    dspy.Example(question="How many continents are there?", answer="7").with_inputs("question"),
    dspy.Example(question="What is the square root of 144?", answer="12").with_inputs("question"),
    dspy.Example(question="Who painted the Mona Lisa?", answer="Leonardo da Vinci").with_inputs("question"),
    dspy.Example(question="What is the smallest prime number?", answer="2").with_inputs("question"),
    dspy.Example(question="What gas do plants absorb?", answer="Carbon dioxide").with_inputs("question"),
    dspy.Example(question="How many legs does a spider have?", answer="8").with_inputs("question"),
]

# Define metric
def answer_match(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()

# Create program
program = dspy.ChainOfThought("question -> answer")

# Optimize
teleprompter = dspy.BootstrapFewShotWithRandomSearch(
    metric=answer_match,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
    num_candidate_programs=5,
    num_threads=4,
)

optimized = teleprompter.compile(program, trainset=trainset)

# Test the optimized program
result = optimized(question="What is the capital of Japan?")
print(f"Answer: {result.answer}")
```

---

## Inspecting Optimized Prompts

After optimization, you'll want to see what DSPy actually learned. The best way is to run a prediction and inspect history:

```python
# Run the optimized program
result = optimized(question="What is the speed of light?")

# See the full prompt with injected demonstrations
dspy.inspect_history(n=1)
```

You'll see the few-shot demonstrations injected before your actual question, each one a successful example that passed your metric. This is what the model "learns" from during inference.

You can also inspect the demos programmatically:

```python
# Access the predictor's demos
for predictor in optimized.predictors():
    print(f"Demos count: {len(predictor.demos)}")
    for demo in predictor.demos:
        print(f"  Q: {demo.get('question', 'N/A')}")
        print(f"  A: {demo.get('answer', 'N/A')}")
        print()
```

---

## Saving and Loading

Once you've found an optimized program, save it so you don't have to re-run optimization:

```python
# Save the optimized program
optimized.save("optimized_qa.json")

# Later, load it back
loaded = dspy.ChainOfThought("question -> answer")
loaded.load("optimized_qa.json")

# Use it: the demonstrations are restored
result = loaded(question="What is the speed of light?")
print(result.answer)
```

The saved JSON contains all the optimized demonstrations and any learned instructions. You can version-control this file, deploy it, or share it with your team.

---

## Key Takeaways

- **DSPy optimization automatically discovers the best prompts**: you provide the program, metric, and data; the optimizer does the rest.
- **BootstrapFewShot** generates demonstrations by running your program and keeping successful outputs.
- **BootstrapRS** runs multiple random trials and picks the best candidate. It's the recommended starting point.
- **The teacher concept** lets you use a larger model to generate demonstrations for a smaller production model.
- **Typical cost is ~\$1-3** for a meaningful quality boost, making this one of the highest-ROI activities in LM development.
- **Always save your optimized programs** since re-optimization is wasteful if you can just load the result.

---

## Next Up

BootstrapRS optimizes *demonstrations*, but what about the **instructions** themselves? In the next blog, we'll explore **MIPROv2**, which jointly optimizes both instructions and few-shot examples using Bayesian optimization.

**[4.2: MIPROv2: Instruction and Demo Optimization →](../4.2-miprov2/blog.md)**

---

## Resources

- 📖 [DSPy Optimizers Overview](https://dspy.ai/learn/optimization/optimizers/)
- 📖 [BootstrapFewShot Docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)
- 📖 [BootstrapFewShotWithRandomSearch Docs](https://dspy.ai/api/optimizers/BootstrapFewShotWithRandomSearch/)
- 💻 [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
