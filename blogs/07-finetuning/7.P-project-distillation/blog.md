# 7.P: Project: Model Distillation Pipeline

## Introduction

You've learned the three core tools of Phase 7: **BootstrapFinetune** for distillation, **BetterTogether** for joint optimization, and **Ensemble** for combining variants. Now it's time to put them together in a real-world scenario.

In this project, you'll build a **complete model distillation pipeline**: start with a high-quality GPT-4o-powered classification system, optimize its prompts, evaluate it, distill it to GPT-4o-mini, and then compare cost, latency, and quality across every stage. By the end, you'll have a production-ready pattern for reducing LM costs by 10-20x while retaining most of the quality.

---

## Project Overview

Here's the full pipeline we'll build:

1. **Build a classification system** with GPT-4o
2. **Optimize prompts** with MIPROv2
3. **Evaluate baseline performance** on a held-out test set
4. **Distill to GPT-4o-mini** using BootstrapFinetune
5. **Evaluate the distilled model** on the same test set
6. **Compare cost, latency, and quality** across all stages

Each step builds on the previous one, mirroring the workflow you'd follow in a real production scenario.

---

## Step 1: Build the System

We'll build a news article classifier that categorizes text into one of six categories and extracts key entities. This is complex enough to benefit from distillation but concrete enough to evaluate precisely.

```python
import dspy
import time
from dspy.evaluate import Evaluate
from dotenv import load_dotenv

load_dotenv()

# ── Models ──────────────────────────────────────────────
teacher_lm = dspy.LM("openai/gpt-4o")
student_lm = dspy.LM("openai/gpt-4o-mini")

# Start with the teacher
dspy.configure(lm=teacher_lm)


# ── Program ─────────────────────────────────────────────
CATEGORIES = ["politics", "technology", "sports", "health", "finance", "entertainment"]


class NewsClassifier(dspy.Module):
    """Classifies news articles into categories and extracts key entities."""

    def __init__(self):
        self.classify = dspy.ChainOfThought(
            "article -> category: str, entities: list[str], summary: str"
        )

    def forward(self, article):
        result = self.classify(article=article)
        # Normalize category to match our expected labels
        result.category = result.category.strip().lower()
        return result


# ── Data ────────────────────────────────────────────────
# In production, you'd load from a file or database.
# Here's a representative sample.

all_examples = [
    dspy.Example(
        article="The Senate passed the infrastructure bill with bipartisan support, allocating $1.2 trillion for roads, bridges, and broadband.",
        category="politics",
    ).with_inputs("article"),
    dspy.Example(
        article="Apple unveiled the M3 chip at its latest event, promising 2x faster machine learning performance.",
        category="technology",
    ).with_inputs("article"),
    dspy.Example(
        article="LeBron James scored 40 points to lead the Lakers past the Celtics in overtime.",
        category="sports",
    ).with_inputs("article"),
    dspy.Example(
        article="A new FDA-approved drug shows 60% reduction in migraine frequency in clinical trials.",
        category="health",
    ).with_inputs("article"),
    dspy.Example(
        article="The Federal Reserve held interest rates steady, signaling potential cuts in Q2.",
        category="finance",
    ).with_inputs("article"),
    dspy.Example(
        article="The new Pixar film grossed $200M worldwide in its opening weekend.",
        category="entertainment",
    ).with_inputs("article"),
    dspy.Example(
        article="SpaceX successfully launched its latest Starship prototype from Boca Chica.",
        category="technology",
    ).with_inputs("article"),
    dspy.Example(
        article="The World Health Organization declared the end of the mpox global emergency.",
        category="health",
    ).with_inputs("article"),
    dspy.Example(
        article="Bitcoin surged past $100,000 for the first time amid institutional adoption.",
        category="finance",
    ).with_inputs("article"),
    dspy.Example(
        article="Serena Williams announced her return to competitive tennis at the Australian Open.",
        category="sports",
    ).with_inputs("article"),
    # ... add 100-200+ examples for serious distillation
]

# Split into train and test
trainset = all_examples[:8]
testset = all_examples[8:]
```

---

## Step 2: Optimize Prompts

Before distillation, we optimize the teacher program's prompts. This ensures the training traces for fine-tuning are as high quality as possible.

```python
# ── Metric ──────────────────────────────────────────────
def classification_metric(example, prediction, trace=None):
    """Check category match. Trace mode is stricter for optimization."""
    pred_cat = prediction.category.strip().lower()
    gold_cat = example.category.strip().lower()
    category_correct = pred_cat == gold_cat

    if trace is not None:
        # During optimization, require category match AND non-empty entities
        has_entities = hasattr(prediction, "entities") and len(prediction.entities) > 0
        return category_correct and has_entities

    return category_correct


# ── Prompt Optimization with MIPROv2 ───────────────────
print("=" * 60)
print("Step 2: Optimizing prompts with MIPROv2")
print("=" * 60)

optimizer = dspy.MIPROv2(
    metric=classification_metric,
    auto="light",
    num_threads=4,
)

optimized_teacher = optimizer.compile(
    NewsClassifier(),
    trainset=trainset,
)

# Save the optimized teacher
optimized_teacher.save("optimized_teacher.json")
print("Optimized teacher saved.")
```

---

## Step 3: Baseline Metrics

Now let's establish baseline performance numbers for the optimized teacher on our test set.

```python
# ── Evaluate the Optimized Teacher ──────────────────────
print("\n" + "=" * 60)
print("Step 3: Evaluating optimized teacher (GPT-4o)")
print("=" * 60)

evaluator = Evaluate(
    devset=testset,
    metric=classification_metric,
    num_threads=4,
    display_progress=True,
)

# Measure accuracy
teacher_score = evaluator(optimized_teacher)
print(f"Teacher accuracy: {teacher_score:.1f}%")

# Measure latency
start = time.time()
for ex in testset:
    optimized_teacher(article=ex.article)
teacher_latency = (time.time() - start) / len(testset)
print(f"Teacher avg latency: {teacher_latency:.2f}s per example")

# Track cost from LM history
teacher_cost_info = {
    "model": "gpt-4o",
    "score": teacher_score,
    "latency": teacher_latency,
}
```

---

## Step 4: Distill with BootstrapFinetune

This is the core step: using BootstrapFinetune to transfer the teacher's knowledge into a smaller model.

```python
# ── Distillation ────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Distilling to GPT-4o-mini with BootstrapFinetune")
print("=" * 60)

# Configure BootstrapFinetune
finetune_optimizer = dspy.BootstrapFinetune(
    metric=classification_metric,
    num_threads=4,
)

# Create student copy with the smaller LM
student = NewsClassifier()
student.set_lm(student_lm)

# Use the optimized teacher as the teacher program
# compile() takes: student, trainset, teacher
dspy.settings.experimental = True
distilled = finetune_optimizer.compile(
    student,
    trainset=trainset,
    teacher=optimized_teacher,
)

# Save the distilled program
distilled.save("distilled_classifier.json")
print("Distilled program saved.")
print("Fine-tuning job submitted. Waiting for completion...")
```

> **Note:** OpenAI fine-tuning jobs take 10-30 minutes to complete. The `compile()` call will block until the fine-tuning job finishes. For large datasets, this can take longer.

---

## Step 5: Evaluate Distilled Model

Once fine-tuning completes, we evaluate the distilled model on the exact same test set.

```python
# ── Evaluate the Distilled Student ──────────────────────
print("\n" + "=" * 60)
print("Step 5: Evaluating distilled student (fine-tuned GPT-4o-mini)")
print("=" * 60)

student_score = evaluator(distilled)
print(f"Student accuracy: {student_score:.1f}%")

# Measure latency
start = time.time()
for ex in testset:
    distilled(article=ex.article)
student_latency = (time.time() - start) / len(testset)
print(f"Student avg latency: {student_latency:.2f}s per example")

student_cost_info = {
    "model": "gpt-4o-mini (fine-tuned)",
    "score": student_score,
    "latency": student_latency,
}
```

---

## Step 6: Cost-Quality Analysis

Finally, let's put the numbers side by side and quantify the tradeoff.

```python
# ── Cost-Quality Comparison ─────────────────────────────
print("\n" + "=" * 60)
print("Step 6: Cost-Quality Analysis")
print("=" * 60)

# Approximate per-token costs (as of early 2025)
COST_PER_1M_INPUT = {
    "gpt-4o": 2.50,          # $2.50 per 1M input tokens
    "gpt-4o-mini": 0.15,      # $0.15 per 1M input tokens
    "gpt-4o-mini-ft": 0.30,   # Fine-tuned mini: ~2x base price
}
COST_PER_1M_OUTPUT = {
    "gpt-4o": 10.00,
    "gpt-4o-mini": 0.60,
    "gpt-4o-mini-ft": 1.20,
}

# Estimate cost per 1000 classifications (~500 tokens input, ~100 tokens output each)
teacher_cost_1k = (500 * COST_PER_1M_INPUT["gpt-4o"] + 100 * COST_PER_1M_OUTPUT["gpt-4o"]) / 1000
student_cost_1k = (500 * COST_PER_1M_INPUT["gpt-4o-mini-ft"] + 100 * COST_PER_1M_OUTPUT["gpt-4o-mini-ft"]) / 1000

print(f"\n{'Metric':<25} {'Teacher (GPT-4o)':<20} {'Student (4o-mini FT)':<20}")
print("-" * 65)
print(f"{'Accuracy':<25} {teacher_cost_info['score']:<20.1f} {student_cost_info['score']:<20.1f}")
print(f"{'Avg Latency (s)':<25} {teacher_cost_info['latency']:<20.2f} {student_cost_info['latency']:<20.2f}")
print(f"{'Cost per 1K calls':<25} ${teacher_cost_1k:<19.4f} ${student_cost_1k:<19.4f}")
print(f"{'Cost reduction':<25} {'1x (baseline)':<20} {f'{teacher_cost_1k/student_cost_1k:.1f}x cheaper':<20}")

quality_retention = (student_cost_info["score"] / teacher_cost_info["score"]) * 100
print(f"\nQuality retention: {quality_retention:.1f}%")
print(f"Cost reduction: {teacher_cost_1k / student_cost_1k:.1f}x")

if quality_retention >= 95:
    print("\n\u2705 Excellent distillation: student retains >95% of teacher quality!")
elif quality_retention >= 90:
    print("\n\u2705 Good distillation: student retains >90% of teacher quality.")
elif quality_retention >= 80:
    print("\n\u26a0\ufe0f Moderate distillation: consider more training data or a closer model gap.")
else:
    print("\n\u274c Poor distillation: try more training data, a bigger student model, or BetterTogether.")
```

### Typical Results

For a classification task like this, you can expect:

| Metric | Teacher (GPT-4o) | Student (GPT-4o-mini FT) |
|--------|:----------------:|:------------------------:|
| Accuracy | 95-98% | 92-96% |
| Latency | 0.8-1.2s | 0.3-0.5s |
| Cost per 1K calls | ~$2.25 | ~$0.27 |
| **Cost reduction** | 1x | **~8x cheaper** |

The exact numbers depend on your data, task complexity, and how much training data you provide.

---

## What We Learned

This project demonstrated the complete distillation lifecycle:

1. **Start with quality.** The teacher model must already work well; fine-tuning amplifies both good and bad patterns.
2. **Optimize prompts first.** MIPROv2 on the teacher produces higher-quality traces, which directly improves the fine-tuned student.
3. **Measure everything.** Without baseline metrics, you can't quantify the cost-quality tradeoff.
4. **Distillation is remarkably effective.** For well-defined tasks like classification, students typically retain 90-98% of teacher quality at a fraction of the cost.
5. **The savings compound at scale.** Going from \$2.25 to \$0.27 per 1,000 calls means saving \$1,980 per million calls.

### When to Go Further

- **If quality retention is below 90%:** Add more training data (aim for 500+ examples), or try BetterTogether for joint optimization.
- **If you need even lower cost:** Distill to an open-source model (Llama 3, Mistral) and self-host.
- **If you want maximum robustness:** Ensemble the fine-tuned model with a prompted variant (see Blog 7.3).

---

## Next Up

Congratulations, you've completed Phase 7! You now know how to distill DSPy programs from large models to small ones, jointly optimize prompts and weights, and ensemble multiple variants for robustness.

In **Phase 8**, we'll explore **Recursive Language Models (RLMs)**: DSPy's `dspy.RLM` module that handles large contexts by letting the LLM programmatically explore data through a sandboxed Python REPL. You'll learn when REPL-based exploration outperforms standard prompting and how to integrate RLM into your pipelines.

**[8.1: Understanding RLM →](../../08-rlm/8.1-understanding-rlm/blog.md)**

---

## Resources

- [BootstrapFinetune API Reference](https://dspy.ai/api/optimizers/BootstrapFinetune/)
- [Classification Fine-Tuning Tutorial](https://dspy.ai/tutorials/classification_finetuning/)
- [BetterTogether API Reference](https://dspy.ai/api/optimizers/BetterTogether/)
- [Ensemble API Reference](https://dspy.ai/api/optimizers/Ensemble/)
- [DSPy Optimizers Overview](https://dspy.ai/learn/optimization/optimizers/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)

---
