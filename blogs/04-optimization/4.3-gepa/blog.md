# 4.3: GEPA: Reflective Prompt Evolution

## Introduction

BootstrapRS finds great demonstrations. MIPROv2 searches over instructions and demos using Bayesian optimization. But both treat optimization as a **search problem**, exploring configurations and scoring them. Neither asks *why* a configuration works or *what* went wrong when it doesn't.

**GEPA** takes a fundamentally different approach: it uses **LM reflection on program trajectories** to propose better prompts. Instead of blind search, GEPA analyzes successful and failed runs, identifies patterns, and evolves prompts that address specific weaknesses.

---

## What You'll Learn

- How GEPA uses reflection on program trajectories to evolve prompts
- The difference between search-based (MIPROv2) and reflection-based (GEPA) optimization
- How domain-specific textual feedback guides prompt evolution
- When GEPA outperforms other optimizers
- How to run GEPA on your own tasks

---

## Prerequisites

- Completed [4.2: MIPROv2](../4.2-miprov2/blog.md)
- Understanding of metrics and evaluation from Phase 3

---

## How GEPA Works

GEPA's optimization loop has four key phases:

### Phase 1: Trajectory Collection

GEPA runs your program on training examples and collects **full trajectories**, not just inputs and outputs, but the intermediate reasoning steps, demonstrations used, and the final metric score. This gives the optimizer rich context about *how* the program is failing, not just *that* it's failing.

### Phase 2: Reflection

Here's where GEPA diverges from other optimizers. It feeds the collected trajectories, both successes and failures, back to an LM and asks it to **reflect**:

- What patterns appear in successful runs?
- What goes wrong in failed runs?
- What prompt changes might address the failure modes?

This reflection step produces structured insights like *"The model fails on multi-step arithmetic because it doesn't show intermediate calculations"* or *"Answers to historical questions are too verbose when the metric expects concise responses."*

### Phase 3: Prompt Proposal

Based on the reflections, GEPA proposes new prompt variants that specifically address identified weaknesses. These aren't random mutations; they're **targeted improvements** informed by the actual failure modes.

### Phase 4: Evaluation and Selection

The proposed prompts are evaluated on the training data. The best-performing variants survive, and the cycle repeats. Over multiple generations, prompts evolve toward configurations that handle the task's specific challenges.

---

## GEPA vs. MIPROv2

| Aspect | MIPROv2 | GEPA |
|--------|---------|------|
| **Strategy** | Bayesian search over instruction/demo combos | Reflective evolution based on trajectory analysis |
| **Insight source** | Metric scores only | Full program trajectories + textual feedback |
| **Strength** | Broad, systematic exploration | Deep, targeted improvements |
| **Data split** | Uses validation for evaluation | Follows standard ML convention (maximize training set) |
| **Best for** | General-purpose optimization | Complex tasks where understanding *why* things fail matters |

A key difference: GEPA follows the **standard ML data split convention**, maximizing the size of the training set. Unlike MIPROv2, which reserves a validation set from your data for Bayesian optimization, GEPA uses training data more efficiently.

---

## Domain-Specific Textual Feedback

One of GEPA's most powerful features is its ability to leverage **domain-specific textual feedback**. If your metric can provide not just a score but a *reason* for the score, GEPA can incorporate that feedback into its reflection process:

```python
def detailed_metric(example, prediction, trace=None):
    """Metric that returns textual feedback for GEPA to reflect on."""
    score = 0.0
    feedback = []

    if prediction.answer.strip().lower() == example.answer.strip().lower():
        score = 1.0
    else:
        feedback.append(f"Expected '{example.answer}', got '{prediction.answer}'")

    if hasattr(prediction, 'reasoning') and len(prediction.reasoning) < 20:
        feedback.append("Reasoning was too brief to show work")

    return score
```

When GEPA analyzes why a particular trajectory failed, it can see these feedback signals and propose prompts that specifically address them. For instance, it might add instructions to "show all intermediate steps" if the feedback indicates reasoning was too brief.

---

## Running GEPA

Here's how to use GEPA on a task:

```python

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Training data
trainset = [
    dspy.Example(
        question="If a shirt costs $25 and is 20% off, what do you pay?",
        answer="$20"
    ).with_inputs("question"),
    dspy.Example(
        question="A train travels 120 miles in 2 hours. What is its speed in mph?",
        answer="60 mph"
    ).with_inputs("question"),
    # ... more examples, ideally 50+
]

# Metric
def answer_match(example, prediction, trace=None):
    pred = prediction.answer.strip().lower().replace("$", "").replace(",", "")
    gold = example.answer.strip().lower().replace("$", "").replace(",", "")
    return pred == gold

# Program
program = dspy.ChainOfThought("question -> answer")

# Optimize with GEPA
tp = dspy.GEPA(
    metric=answer_match,
    num_threads=8,
)

optimized = tp.compile(program, trainset=trainset)

# Test
result = optimized(question="A book costs $30 and is 15% off. What do you pay?")
print(f"Answer: {result.answer}")
```

---

## When GEPA Excels

GEPA tends to outperform search-based optimizers in specific scenarios:

1. **Complex reasoning tasks** such as math, logic, and multi-step inference, where understanding *why* the model fails is crucial for improvement
2. **Tasks with nuanced failure modes** where a simple metric score doesn't capture what's going wrong, but trajectory analysis reveals patterns
3. **Domain-specific tasks** where expert knowledge embedded in textual feedback can guide prompt evolution toward domain-appropriate behavior
4. **Tasks where human-designed prompts plateau** since GEPA's evolutionary approach can discover prompt strategies that humans wouldn't think of

### When to Use Something Else

- **Simple classification tasks**: BootstrapRS or MIPROv2 are typically sufficient and faster
- **Very small datasets (< 20 examples)**: not enough trajectories for meaningful reflection
- **Tight budget**: GEPA's reflection loop uses additional LM calls compared to BootstrapRS

---

## Key Takeaways

- **GEPA uses LM reflection on trajectories**, not just metric scores. It understands *why* things fail and proposes targeted improvements.
- **Reflection-based optimization** can find prompt strategies that search-based methods miss, especially for complex reasoning tasks.
- **Domain-specific textual feedback** supercharges GEPA by giving it richer signals to reflect on.
- **GEPA follows standard ML data splits**, maximizing training data efficiency.
- **Best for complex tasks** where understanding failure modes matters more than broad exploration.

---

## Next Up

You've now learned three different optimizers: BootstrapRS, MIPROv2, and GEPA. But DSPy has even more. In the next blog, we'll survey the **complete optimizer landscape**, covering every optimizer available, when to use each, and how to combine them.

**[4.4: The Optimizer Landscape →](../4.4-optimizer-landscape/blog.md)**

---

## Resources

- [GEPA Overview](https://dspy.ai/tutorials/gepa_ai_program/)
- [GEPA for AIME](https://dspy.ai/tutorials/gepa_aime/)
- [GEPA Paper](https://arxiv.org/abs/2507.19457)
- [DSPy Optimizers Overview](https://dspy.ai/learn/optimization/optimizers/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
