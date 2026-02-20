# 7.2: BetterTogether: Joint Optimization

## Introduction

In the previous blog, you learned how BootstrapFinetune distills a prompt-based program into fine-tuned weights. But here's a subtlety: in many real-world systems, you don't want to fine-tune **everything**. Some modules benefit from staying on a large, prompted model (complex reasoning, open-ended generation), while others are better served by a fast, fine-tuned small model (classification, extraction, formatting).

**`dspy.BetterTogether`** addresses exactly this scenario. It **jointly optimizes prompts and weights**, tuning instructions and demonstrations for the modules that stay on large models, while simultaneously fine-tuning the modules that move to small models. The result is a system where prompts and weights work **synergistically**, each compensating for the other's weaknesses.

---

## What You'll Learn

- What `dspy.BetterTogether` does and why joint optimization beats sequential optimization
- How it alternates between prompt optimization and weight optimization
- How to configure which modules get prompts vs. weights
- The research behind the approach (BetterTogether paper)
- When to choose BetterTogether over standalone BootstrapFinetune

---

## Prerequisites

- Completed [7.1: BootstrapFinetune](../7.1-bootstrap-finetune/blog.md)
- Familiarity with MIPROv2 from [4.2](../../04-optimization/4.2-miprov2/blog.md)

---

## Why Joint Optimization?

Consider a multi-step pipeline where Module A (complex reasoning) runs on GPT-4o and Module B (classification) runs on a fine-tuned GPT-4o-mini. If you optimize them independently:

1. Optimize prompts for Module A → works great in isolation
2. Fine-tune Module B on traces from the original Module A → decent results

But there's a problem: **Module A's optimized prompts might produce outputs that Module B wasn't fine-tuned on**, and Module B's fine-tuned behavior might not align with what Module A expects downstream.

BetterTogether solves this by alternating optimization rounds:

```
Round 1: Optimize prompts for large-model modules
Round 2: Fine-tune small-model modules on traces from Round 1
Round 3: Re-optimize prompts, accounting for fine-tuned modules
Round 4: Re-fine-tune, accounting for optimized prompts
...until convergence
```

Each round adapts to the other's changes, producing a **co-adapted** system.

---

## How BetterTogether Works

The optimizer takes three key inputs:

1. **A prompt optimizer**, typically MIPROv2, used for modules on large models
2. **A weight optimizer**, typically BootstrapFinetune, used for modules on small models
3. **Your program and training data**, as with any DSPy optimizer

It then runs alternating optimization cycles:

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure models
large_lm = dspy.LM("openai/gpt-4o")
small_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=large_lm)

# Define a multi-step pipeline
class AnalysisPipeline(dspy.Module):
    def __init__(self):
        # Complex reasoning stays on the large model
        self.analyze = dspy.ChainOfThought(
            "document -> analysis: str, key_findings: list[str]"
        )
        # Classification moves to the small model
        self.classify = dspy.Predict(
            "analysis, key_findings -> category: str, risk_level: str"
        )

    def forward(self, document):
        result = self.analyze(document=document)
        return self.classify(
            analysis=result.analysis,
            key_findings=result.key_findings,
        )

# Training data
trainset = [
    dspy.Example(
        document="Q3 revenue increased 15% YoY driven by cloud services...",
        category="growth",
        risk_level="low",
    ).with_inputs("document"),
    # ... 100+ examples
]

# Metric
def pipeline_metric(example, prediction, trace=None):
    category_match = prediction.category.strip().lower() == example.category.strip().lower()
    risk_match = prediction.risk_level.strip().lower() == example.risk_level.strip().lower()
    return category_match and risk_match

# Assign the small LM to the classify module
pipeline = AnalysisPipeline()
pipeline.classify.set_lm(small_lm)

# Configure BetterTogether with both optimizer types
better_together = dspy.BetterTogether(
    metric=pipeline_metric,
    prompt_optimizer=dspy.MIPROv2,
    weight_optimizer=dspy.BootstrapFinetune,
)

# Compile with the alternating strategy: prompt -> weight -> prompt
dspy.settings.experimental = True
optimized = better_together.compile(
    pipeline,
    trainset=trainset,
    strategy="p -> w -> p",  # Alternating prompt-weight-prompt rounds
)
```

After `compile()`, the `analyze` predictor has optimized prompts (instructions and demonstrations) for GPT-4o, while the `classify` predictor runs on a fine-tuned GPT-4o-mini, and both were optimized **aware of each other**.

---

## The Research Behind BetterTogether

BetterTogether is based on the paper **"Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together"** ([arXiv:2407.10930](https://arxiv.org/abs/2407.10930)). Key findings from the research:

- **Joint optimization outperforms sequential optimization** by 2-8% on average across multiple benchmarks.
- **Prompts and fine-tuning are complementary:** prompts provide task framing that fine-tuning alone misses, while fine-tuning internalizes patterns that prompts alone can't capture.
- **The effect is strongest in multi-step pipelines** where errors compound across modules.
- **Fewer training examples are needed** compared to fine-tuning alone, because the prompt optimization provides a stronger starting signal.

---

## BetterTogether vs. Standalone Approaches

| Approach | What It Does | Best For |
|----------|-------------|----------|
| MIPROv2 alone | Optimizes prompts only | When you can't fine-tune (budget, permissions) |
| BootstrapFinetune alone | Fine-tunes weights only | Single-model distillation |
| **BetterTogether** | **Jointly optimizes both** | **Multi-model pipelines, maximum quality** |

**Choose BetterTogether when:**

- Your pipeline uses both large and small models
- You've already tried prompt-only optimization and need more quality
- You have 100+ training examples and a reliable metric
- The runtime cost of alternating optimization rounds is acceptable (typically 2-4x the cost of a single optimization)

**Stick with simpler approaches when:**

- Your entire pipeline runs on a single model
- You're still iterating on the program architecture
- Fine-tuning isn't available for your target model

---

## Key Takeaways

- **BetterTogether jointly optimizes prompts and weights**, achieving results that beat either approach alone.
- **Alternating optimization rounds** ensure that prompt-optimized and fine-tuned modules co-adapt to each other.
- **The synergy is strongest in multi-step pipelines** where different modules benefit from different optimization strategies.
- **Based on peer-reviewed research** demonstrating 2-8% improvements over sequential optimization.
- **Use it when your system mixes large prompted models with small fine-tuned models**, which is increasingly the production norm.

---

## Next Up

You've seen how to optimize a single program and how to jointly optimize prompts and weights. But what if you have **multiple optimized variants** of the same program? In the next blog, we'll explore **Ensemble**, which combines multiple programs into one for greater robustness.

**[7.3: Ensemble: Combining Programs →](../7.3-ensemble/blog.md)**

---

## Resources

- [BetterTogether API Reference](https://dspy.ai/api/optimizers/BetterTogether/)
- [BetterTogether Paper (arXiv:2407.10930)](https://arxiv.org/abs/2407.10930)
- [DSPy Optimizers Overview](https://dspy.ai/learn/optimization/optimizers/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)

---
