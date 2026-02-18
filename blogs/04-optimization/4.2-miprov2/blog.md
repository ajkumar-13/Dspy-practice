# 4.2: MIPROv2: Instruction and Demo Optimization

## Introduction

BootstrapRS finds great few-shot demonstrations, but it never touches the **instructions**, the text that tells the model *what* to do. What if the instruction itself is limiting performance? What if phrasing "Answer the question concisely" differently could unlock better behavior?

**MIPROv2** (Multi-prompt Instruction PRoposal Optimizer v2) tackles this head-on. It jointly optimizes both **instructions** and **demonstrations** using Bayesian optimization, systematically searching through the combined space to find the best configuration for your task.

---

## What You'll Learn

- How MIPROv2 generates and scores both instructions and demonstrations
- The three auto modes: `"light"`, `"medium"`, and `"heavy"`
- How to configure teacher settings for higher-quality proposals
- Zero-shot optimization (instruction only, no demos)
- When to choose MIPROv2 over BootstrapRS

---

## Prerequisites

- Completed [4.1: BootstrapRS](../4.1-bootstrap-rs/blog.md)
- A training set with 50+ examples (more data = better results)

---

## How MIPROv2 Works

MIPROv2 operates in three stages:

### Stage 1: Proposal Generation

The optimizer first generates a **pool of candidate instructions** for each predictor in your program. These aren't random; they're **data-aware** and **demonstration-aware**:

- **Data-aware:** MIPROv2 examines your training examples to understand the task distribution
- **Demo-aware:** It looks at successful demonstrations to understand what good outputs look like

This means generated instructions like *"Given a factual question, provide a concise one-phrase answer based on established knowledge"* are informed by the actual data, not generic templates.

### Stage 2: Bayesian Optimization

With a pool of candidate instructions and demonstrations, MIPROv2 uses **Bayesian optimization** (specifically, a Tree-structured Parzen Estimator) to efficiently search the combinatorial space. Instead of trying every possible combination, which would be astronomically expensive, it learns which types of configurations tend to score well and focuses its search there.

### Stage 3: Selection

After the search budget is exhausted, MIPROv2 returns the best-performing combination of instructions and demonstrations.

---

## Auto Modes

MIPROv2 provides three convenience modes that automatically set the search budget:

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Define your metric
def answer_match(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()

# Light mode: quick exploration, ~\$0.50, ~5-10 minutes
tp_light = dspy.MIPROv2(metric=answer_match, auto="light", num_threads=8)

# Medium mode: balanced search, ~\$1.50, ~20-30 minutes
tp_medium = dspy.MIPROv2(metric=answer_match, auto="medium", num_threads=16)

# Heavy mode: thorough search, ~\$5+, ~1-2 hours
tp_heavy = dspy.MIPROv2(metric=answer_match, auto="heavy", num_threads=24)
```

| Mode | Trials | Cost (GPT-4o-mini) | Time | Best For |
|------|--------|---------------------|------|----------|
| `"light"` | ~7 | ~\$0.50 | 5-10 min | Quick iteration, prototyping |
| `"medium"` | ~25 | ~\$1.50 | 20-30 min | Balanced quality/cost trade-off |
| `"heavy"` | ~50+ | ~\$5+ | 1-2 hours | Production-grade optimization |

---

## Running MIPROv2

Here's a complete example:

```python

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Training data
trainset = [
    dspy.Example(question="What is the largest ocean?", answer="Pacific Ocean").with_inputs("question"),
    dspy.Example(question="Who discovered penicillin?", answer="Alexander Fleming").with_inputs("question"),
    dspy.Example(question="What is the hardest natural substance?", answer="Diamond").with_inputs("question"),
    dspy.Example(question="How many bones in the human body?", answer="206").with_inputs("question"),
    dspy.Example(question="What planet is known as the Red Planet?", answer="Mars").with_inputs("question"),
    # ... ideally 50+ examples for MIPROv2 to shine
]

# Metric
def answer_match(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()

# Program
program = dspy.ChainOfThought("question -> answer")

# Optimize with MIPROv2
tp = dspy.MIPROv2(metric=answer_match, auto="medium", num_threads=8)

optimized = tp.compile(
    program,
    trainset=trainset,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
)

# Test
result = optimized(question="What is the smallest country in the world?")
print(f"Answer: {result.answer}")

# Save the optimized program
optimized.save("optimized_miprov2.json")
```

---

## Using a Teacher Model

For best results, use a **larger model as the teacher** to generate instruction proposals and demonstrations, while your student program runs on a cheaper model:

```python
teacher_lm = dspy.LM("openai/gpt-4o")

tp = dspy.MIPROv2(metric=answer_match, auto="medium", num_threads=8)

optimized = tp.compile(
    program,
    trainset=trainset,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
    teacher_settings=dict(lm=teacher_lm),
)
```

The teacher model generates higher-quality instruction candidates and demonstrations. Your production program still uses GPT-4o-mini, just with better instructions and examples discovered by GPT-4o.

---

## Zero-Shot Optimization

Sometimes you want to optimize **instructions only**, with no few-shot demonstrations at all. This keeps prompts short and reduces token costs at inference time:

```python
tp = dspy.MIPROv2(metric=answer_match, auto="medium", num_threads=8)

optimized = tp.compile(
    program,
    trainset=trainset,
    max_bootstrapped_demos=0,   # No bootstrapped demos
    max_labeled_demos=0,        # No labeled demos
)
```

With zero demonstrations, the optimizer focuses entirely on finding the best instruction text. This is useful when you're latency-sensitive or when your task doesn't benefit much from few-shot examples.

---

## Inspecting Optimized Instructions

After optimization, check what MIPROv2 discovered:

```python
# Run and inspect
result = optimized(question="What is the speed of sound?")
dspy.inspect_history(n=1)
```

You'll see the optimized instruction text in the system message, often surprisingly specific and effective. MIPROv2 might discover instructions like *"Provide a precise, factual answer in the most commonly accepted unit or phrasing"* that you wouldn't have written yourself.

---

## Multi-Module Pipelines

MIPROv2 optimizes **every predictor** in a pipeline simultaneously. If you have a multi-step module with retrieval, reasoning, and answering stages, each gets its own optimized instruction and demonstrations:

```python
class MultiStepQA(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("question -> category")
        self.answer = dspy.ChainOfThought("question, category -> answer")

    def forward(self, question):
        category = self.classify(question=question).category
        return self.answer(question=question, category=category)

program = MultiStepQA()
tp = dspy.MIPROv2(metric=answer_match, auto="medium", num_threads=8)
optimized = tp.compile(program, trainset=trainset)
# Both classify and answer predictors get optimized instructions
```

---

## When to Choose MIPROv2

| Scenario | Recommended Optimizer |
|----------|----------------------|
| Quick baseline, small data (~10-50 examples) | BootstrapRS |
| Instruction matters, 50+ examples | **MIPROv2** |
| Zero-shot (no demos), any data size | **MIPROv2** |
| 40+ trials budget, 200+ examples | **MIPROv2 (heavy)** |
| Complex reasoning, needs trajectory reflection | GEPA (next blog) |

---

## Key Takeaways

- **MIPROv2 jointly optimizes instructions and demonstrations**, improving *what* your program says to the model, not just *which examples* it shows.
- **Auto modes** (`"light"`, `"medium"`, `"heavy"`) let you trade compute for quality with a single parameter.
- **Use a teacher model** for higher-quality proposals when budget allows.
- **Zero-shot optimization** is powerful for latency-sensitive applications. Great instructions can compensate for missing demonstrations.
- **Multi-module pipelines get end-to-end optimization** where every predictor gets its own tuned instruction.

---

## Next Up

MIPROv2 searches systematically, but what if the optimizer could **reflect** on *why* certain prompts work and propose entirely new strategies? That's exactly what GEPA does.

**[4.3: GEPA: Reflective Prompt Evolution →](../4.3-gepa/blog.md)**

---

## Resources

- 📖 [MIPROv2 Docs](https://dspy.ai/api/optimizers/MIPROv2/)
- 📖 [DSPy Optimizers Overview](https://dspy.ai/learn/optimization/optimizers/)
- 💻 [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
