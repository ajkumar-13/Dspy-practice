# 7.1: BootstrapFinetune: Prompt to Weights

## Introduction

You've spent the last six phases building increasingly sophisticated DSPy programs (modules, pipelines, agents) all powered by large language models like GPT-4o. They work beautifully, but they come at a cost: every call to a frontier model adds latency and expense. What if you could **transfer that intelligence into a smaller, cheaper model** without losing quality?

That's exactly what **`dspy.BootstrapFinetune`** does. It takes a working DSPy program, uses it to generate high-quality training data, and then fine-tunes a smaller model on that data. The result is a new version of your program that runs on a fine-tuned model: faster, cheaper, and often surprisingly close in quality to the original.

This is **distillation** in the DSPy world: going from prompts to weights.

---

## What You'll Learn

- What `dspy.BootstrapFinetune` does and how it fits into the DSPy optimization landscape
- The teacher-student distillation workflow
- How to run BootstrapFinetune end-to-end
- Choosing source (teacher) and target (student) models
- Practical considerations: data requirements, training cost, and hosting
- When to use BootstrapFinetune vs. prompt-only optimization

---

## Prerequisites

- Completed Phase 6 (Agents & Tool Use)
- Familiarity with DSPy optimizers from Phase 4 (especially BootstrapRS or MIPROv2)
- An OpenAI API key with access to fine-tuning (or another provider that supports it)

---

## What is BootstrapFinetune?

`dspy.BootstrapFinetune` is a DSPy optimizer, but unlike BootstrapRS or MIPROv2 which optimize **prompts** (instructions and demonstrations), BootstrapFinetune optimizes **model weights**. It bridges the gap between prompt engineering and model training.

Here's the key idea:

| Optimizer | What It Tunes | Output |
|-----------|--------------|--------|
| BootstrapRS | Few-shot demonstrations | Better prompts, same model |
| MIPROv2 | Instructions + demonstrations | Better prompts, same model |
| **BootstrapFinetune** | **Model weights** | **New fine-tuned model** |

Think of it as a **compiler for DSPy programs**: your prompt-based program is the source code, and the fine-tuned model is the compiled binary: smaller, faster, and self-contained.

---

## The Distillation Workflow

The typical distillation workflow in DSPy follows a clear progression:

```
1. Develop with a large LM (GPT-4o, Claude 3.5 Sonnet)
2. Optimize prompts (BootstrapRS, MIPROv2)
3. Validate quality with evaluations
4. Distill to a smaller LM (GPT-4o-mini, Llama 3, Mistral)
5. Evaluate the distilled model
6. Deploy the cheaper, faster version
```

Under the hood, BootstrapFinetune works in three stages:

**Stage 1: Trace Collection.** The optimizer runs your program (the "teacher") on your training examples. Each successful execution produces a **trace**: a record of every LM call, including the prompt and completion.

**Stage 2: Trace Filtering.** Your metric function filters the traces, keeping only the ones that produced correct outputs. This ensures the fine-tuning data is high quality.

**Stage 3: Fine-Tuning.** The filtered traces are formatted as training examples and sent to a fine-tuning API (e.g., OpenAI's). Each predictor in your program gets its own fine-tuned model trained on its specific prompt-completion pairs.

---

## Running BootstrapFinetune

Here's a complete example that distills a GPT-4o classification pipeline into GPT-4o-mini:

```python
import dspy
from dspy.evaluate import Evaluate
from dotenv import load_dotenv

load_dotenv()

# Step 1: Configure the teacher (large model)
teacher_lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=teacher_lm)

# Step 2: Define your program
class Classifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(
            "text -> category: str, confidence: float"
        )

    def forward(self, text):
        return self.classify(text=text)

# Step 3: Prepare training data
trainset = [
    dspy.Example(text="The stock market rallied on strong earnings reports", category="finance").with_inputs("text"),
    dspy.Example(text="New study reveals high-protein diets improve recovery", category="health").with_inputs("text"),
    dspy.Example(text="SpaceX successfully launched 23 Starlink satellites", category="technology").with_inputs("text"),
    dspy.Example(text="Team wins championship after dramatic overtime victory", category="sports").with_inputs("text"),
    # ... include 50-200 examples for best results
]

# Step 4: Define a metric
def classification_metric(example, prediction, trace=None):
    return prediction.category.strip().lower() == example.category.strip().lower()

# Step 5: Configure BootstrapFinetune
student_lm = dspy.LM("openai/gpt-4o-mini")  # The student model to fine-tune

# Create a student copy and assign the smaller LM
student = Classifier()
student.set_lm(student_lm)

# Create a teacher copy (uses the configured large LM)
teacher = Classifier()

bootstrap_finetune = dspy.BootstrapFinetune(
    metric=classification_metric,
    num_threads=4,
)

# Step 6: Compile: this generates traces from the teacher and fine-tunes the student
distilled = bootstrap_finetune.compile(
    student,
    trainset=trainset,
    teacher=teacher,
)

# Step 7: The distilled program now uses the fine-tuned model
result = distilled(text="Researchers discover high-protein diets improve recovery times")
print(f"Category: {result.category}")
print(f"Confidence: {result.confidence}")
```

After `compile()` finishes, each predictor in `distilled` points to a newly fine-tuned model instead of the original GPT-4o. The prompts are simpler (fewer demonstrations needed) and the model itself has internalized the task.

---

## Choosing Source and Target Models

The source-target pairing is critical to success:

| Source (Teacher) | Target (Student) | Cost Reduction | Quality Retention |
|-----------------|-------------------|----------------|-------------------|
| GPT-4o | GPT-4o-mini | ~10-20x cheaper | Usually 90-98% |
| GPT-4o | GPT-3.5-turbo | ~20-30x cheaper | Usually 85-95% |
| Claude 3.5 Sonnet | Haiku-class models | ~10x cheaper | Usually 88-96% |
| Any large LM | Open-source (Llama, Mistral) | Near-zero marginal cost | Varies, 80-95% |

**General guidance:**

- Start with the **smallest gap** between teacher and student (e.g., GPT-4o to GPT-4o-mini). Larger gaps require more training data to compensate.
- Use **200+ training examples** for reliable distillation. More is better, up to about 1,000-2,000 examples.
- The teacher should already be optimized: run MIPROv2 or BootstrapRS first, then distill the optimized program.

---

## Practical Considerations

### Data Requirements

BootstrapFinetune needs enough successful traces to produce meaningful training data. If your metric is strict and only 30% of traces pass, you need at least 3× more training examples to compensate. Aim for **100+ passing traces per predictor**.

### Training Cost

Fine-tuning costs depend on your provider:
- **OpenAI:** ~\$8 per 1M training tokens for GPT-4o-mini. A typical DSPy distillation with 200 examples costs \$1-5.
- **Open-source:** Free to train, but you need GPU infrastructure (or use providers like Together, Anyscale, or Modal).

### Hosting Fine-Tuned Models

Fine-tuned OpenAI models are hosted by OpenAI with no infrastructure changes needed. Fine-tuned open-source models require your own serving infrastructure (vLLM, TGI, Ollama, etc.).

### When NOT to Use BootstrapFinetune

- Your program isn't already working well with a large LM (fix quality first)
- You have fewer than 50 training examples
- You need to change the task frequently (fine-tuned models are task-specific)
- The cost savings don't justify the complexity

---

## Key Takeaways

- **BootstrapFinetune distills prompt-based DSPy programs into fine-tuned model weights**, converting prompts into parameters.
- **The workflow is develop, optimize prompts, distill, deploy**. Fine-tuning is the last step, not the first.
- **Teacher-student distillation** uses a larger model's successful traces to train a smaller model.
- **Cost reductions of 10-30x** are typical, with quality retention of 85-98% depending on the model gap.
- **Data quality matters more than quantity**: your metric filters traces, so only correct outputs become training data.
- **Always evaluate the distilled model** against the original to quantify the quality-cost tradeoff.

---

## Next Up

BootstrapFinetune optimizes weights for a single model. But what if your system uses **both** large and small models? In the next blog, we'll explore **BetterTogether**, which jointly optimizes prompts for large models and weights for small models, achieving results that beat either approach alone.

**[7.2: BetterTogether: Joint Optimization →](../7.2-better-together/blog.md)**

---

## Resources

- [BootstrapFinetune API Reference](https://dspy.ai/api/optimizers/BootstrapFinetune/)
- [Classification Fine-Tuning Tutorial](https://dspy.ai/tutorials/classification_finetuning/)
- [DSPy Optimizers Overview](https://dspy.ai/learn/optimization/optimizers/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
