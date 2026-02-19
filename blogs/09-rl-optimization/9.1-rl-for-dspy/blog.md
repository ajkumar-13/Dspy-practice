# 9.1: RL Optimization for DSPy: Training LM Weights with GRPO

## Introduction

Throughout this series, you have optimized DSPy programs by tuning **prompts**: selecting better instructions, assembling few-shot demonstrations, refining output constraints. Optimizers like `dspy.MIPROv2` and `dspy.BootstrapRS` are powerful because they leave the underlying model untouched: you get better results without training a single weight.

But what if prompt optimization hits a ceiling? What if you're working with a small, local model that simply can't follow complex multi-module instructions no matter how good your prompts are? That's where **reinforcement learning (RL) optimization** enters the picture.

DSPy now supports **training actual LM weights**, not just tuning prompts, using a technique called **GRPO (Group Relative Policy Optimization)**, generalized to work across multi-module LM programs. This is made possible by the **Arbor RL framework**, an external library that integrates tightly with DSPy's compiler API.

> **EXPERIMENTAL WARNING:** RL optimization in DSPy is a **proof-of-concept**. It is not production-ready. The API may change, results can be inconsistent, and it requires significant GPU resources. The DSPy team describes it as "typically worse on a cost/quality basis than `dspy.MIPROv2` or `dspy.SIMBA`, but a solid start for online RL over arbitrary LM programs for small LMs." Treat this as a research tool, not a production optimizer.

---

## What You'll Learn

- What RL optimization means in the DSPy context: training weights, not prompts
- How Arbor and ArborGRPO generalize GRPO to multi-module DSPy programs
- How to set up the Arbor infrastructure for RL training
- The full training pipeline: model setup, config, compile, evaluate
- How RL differs from BootstrapFinetune and prompt-based optimization
- When RL makes sense, and when it does not

---

## Prerequisites

- Completed Phases 1-8 of the Learn DSPy series
- Familiarity with DSPy optimizers from [Phase 4](../../04-optimization/4.1-bootstrap-rs/blog.md)
- Understanding of BootstrapFinetune from [7.1: BootstrapFinetune](../../07-finetuning/7.1-bootstrap-finetune/blog.md)
- Access to multiple GPUs (4×H100 or similar) for practical RL training
- Basic understanding of RL concepts (reward signals, policy optimization)

---

## RL vs. Prompt Optimization: A Fundamental Shift

To understand why RL optimization matters, let's revisit what DSPy's existing optimizers do:

| Approach | What Changes | Training Signal | Model Weights |
|----------|-------------|-----------------|---------------|
| BootstrapRS | Few-shot demos | Metric on outputs | **Frozen** |
| MIPROv2 / SIMBA | Instructions + demos | Metric on outputs | **Frozen** |
| BootstrapFinetune | Model weights (offline) | Teacher traces | **Updated (offline)** |
| **ArborGRPO (RL)** | **Model weights (online)** | **Reward signal** | **Updated (online)** |

The key distinction: prompt optimization changes *what you say to the model*. RL optimization changes *the model itself*. BootstrapFinetune also changes model weights, but it does so **offline** (it collects traces from a teacher model, then fine-tunes). RL training is **online**: the model generates outputs, receives rewards, and updates its weights in a continuous loop.

This is closer to how reasoning models like DeepSeek-R1 were trained. The model learns from its own successes and failures, not from imitating a teacher.

---

## The Arbor RL Framework

DSPy's RL capabilities are powered by **Arbor**, an external library that handles the heavy lifting of RL training. Arbor provides:

- A local inference server for your model
- GRPO training infrastructure
- Integration with DSPy's compiler API via `ArborGRPO`

### Installation

```bash
pip install -U arbor-ai
```

> **EXPERIMENTAL:** Arbor is a separate project from DSPy. It is under active development and may have breaking changes. Always check the latest documentation.

### What is GRPO?

**Group Relative Policy Optimization (GRPO)** is a reinforcement learning algorithm designed for language models. Instead of using a separate reward model (as in RLHF), GRPO:

1. Generates a **group** of candidate outputs for each input
2. Scores each candidate using a reward function (your DSPy metric)
3. Computes **relative** advantages within the group
4. Updates model weights to favor higher-reward outputs

DSPy's `ArborGRPO` generalizes this from single-turn generation to **multi-module LM programs**. If your DSPy program has three `ChainOfThought` modules, ArborGRPO trains the underlying model to perform better at all three stages simultaneously.

---

## Setting Up Arbor

Arbor manages a local LM server and training loop. Here's the setup:

```python
# RL Optimization with ArborGRPO
# Requires: pip install -U dspy arbor-ai
# Requires: Multiple GPUs (4xH100 recommended)
# EXPERIMENTAL: proof of concept, not production-ready

import dspy
import arbor
from arbor import ArborProvider

# Step 1: Initialize Arbor
arbor.init()

# Step 2: Choose a local model
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Step 3: Start the Arbor server (returns connection info)
arbor_server_info = arbor.start_server(model_name)

# Step 4: Create a DSPy LM pointing to the Arbor server
lm = dspy.LM(
    model=f"openai/arbor:{model_name}",
    provider=ArborProvider(),
    api_base=arbor_server_info["base_url"],
)

# Step 5: Configure DSPy to use this model
dspy.configure(lm=lm)
```

Notice the LM configuration pattern: `openai/arbor:{model_name}`. This tells DSPy to route requests through Arbor's OpenAI-compatible API, which in turn manages the local model for both inference and training.

---

## The RL Training Pipeline

With Arbor running, you can set up an RL training loop. The process has three main components: your DSPy program, a metric function, and the `ArborGRPO` compiler.

### Step 1: Define Your Program

```python
class SimpleQA(dspy.Module):
    def __init__(self):
        self.answer = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.answer(question=question)

program = SimpleQA()
```

### Step 2: Define Your Metric

```python
def answer_metric(example, prediction, trace=None):
    """Simple exact-match metric."""
    gold = example.answer.lower().strip()
    pred = prediction.answer.lower().strip()
    return gold in pred or pred in gold
```

### Step 3: Prepare Your Data

```python
trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What year did WWII end?", answer="1945").with_inputs("question"),
    # ... hundreds more examples for meaningful RL training
]

devset = trainset[:50]  # Validation split
```

### Step 4: Configure Training Hyperparameters

This is where RL training gets GPU-intensive. The `train_kwargs` dictionary controls the training loop:

```python
from peft import LoraConfig

# LoRA configuration for parameter-efficient training
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training hyperparameters
train_kwargs = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "temperature": 1.0,
    "learning_rate": 5e-6,
    "lora_config": lora_config,
    "num_training_gpus": 2,
    "num_inference_gpus": 2,
    "loss_type": "dapo",            # DAPO loss for stable training
    "report_to": "wandb",           # Optional: log to Weights & Biases
}
```

A few notes on the configuration:

- **LoRA** keeps training efficient by only updating a small subset of parameters. Without it, you'd need far more GPU memory.
- **DAPO loss** (Dynamic Alignment Policy Optimization) is a variant that provides more stable training than standard GRPO. It was developed for the OpenRLHF project.
- **GPU split**: Arbor divides your GPUs between inference (generating rollouts) and training (updating weights). With 4×H100s, a typical split is 2 for inference and 2 for training.
- **Temperature**: Set to 1.0 for diverse rollout generation during training.

### Step 5: Configure and Run ArborGRPO

```python
from dspy.teleprompt import ArborGRPO

compiler = ArborGRPO(
    metric=answer_metric,
    num_dspy_examples_per_grpo_step=8,   # Batch of DSPy examples per step
    num_rollouts_per_grpo_step=4,         # Rollouts per example for GRPO
    exclude_demos=True,                    # Don't include few-shot demos
    num_train_steps=100,                   # Total training steps
    checkpoint="checkpoints/rl_qa",        # Save checkpoints
    train_kwargs=train_kwargs,
)

# Run RL training
optimized_program = compiler.compile(
    student=program,
    trainset=trainset,
    valset=devset,
)
```

> **EXPERIMENTAL:** Training will take hours, not minutes. Even on 4×H100 GPUs, expect 3-18 hours depending on task complexity and dataset size. Monitor with wandb if possible.

### Step 6: Evaluate

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(
    devset=devset,
    metric=answer_metric,
    num_threads=4,
    display_progress=True,
)

# Evaluate the RL-optimized program
score = evaluator(optimized_program)
print(f"RL-optimized score: {score}%")
```

---

## How ArborGRPO Differs from BootstrapFinetune

If you have read [7.1: BootstrapFinetune](../../07-finetuning/7.1-bootstrap-finetune/blog.md), you might wonder: both approaches update model weights, so what's different?

| Aspect | BootstrapFinetune | ArborGRPO |
|--------|------------------|-----------|
| Training type | **Offline** (supervised) | **Online** (reinforcement learning) |
| Data source | Teacher model traces | Model's own generated outputs |
| Signal | Imitation of teacher | Reward from metric function |
| Teacher required? | Yes (large model) | No, self-improvement |
| Multi-module | Trains each module separately | Trains all modules jointly |
| GPU requirements | Moderate (API-based fine-tuning possible) | Heavy (multiple GPUs, local training) |

The philosophical difference is profound. BootstrapFinetune says: *"Here is what a good answer looks like. Learn to imitate it."* ArborGRPO says: *"Try generating answers, see which ones score well, and learn to produce more of those."*

Online RL can discover solutions that a teacher model might not generate. It can also learn to coordinate across modules in ways that offline distillation cannot, because the reward signal flows through the entire program.

---

## When to Use RL Optimization

RL optimization is the **most expensive and least mature** optimization approach in DSPy. Use it only when:

1. **You're working with small local models** (e.g., Qwen2.5-1.5B, Llama-3.2-3B) that can't follow complex prompts well
2. **Prompt optimization has plateaued.** You have tried MIPROv2 and SIMBA and hit a ceiling.
3. **You need the model to run locally.** You cannot rely on API-based frontier models.
4. **You have GPU resources.** Multiple high-end GPUs and hours of training time are available.
5. **You are doing research.** You want to explore the frontier of LM program optimization.

For most practical applications, the optimization gradient looks like this:

```
Start here → MIPROv2/SIMBA → BootstrapFinetune → ArborGRPO (RL)
  (easy)        (moderate)       (advanced)       (experimental)
```

> **Honest assessment from the DSPy team:** ArborGRPO is "typically worse on a cost/quality basis than `dspy.MIPROv2` or `dspy.SIMBA`, but a solid start for online RL over arbitrary LM programs for small LMs." This is a research direction, not a production recommendation.

---

## Key Takeaways

1. **RL optimization trains model weights**, not prompts. It is fundamentally different from MIPROv2 or BootstrapRS.
2. **Arbor** is the external RL framework that powers DSPy's RL capabilities (`pip install -U arbor-ai`).
3. **ArborGRPO** generalizes GRPO to multi-module DSPy programs. All modules are trained jointly.
4. **This is experimental.** Expect rough edges, long training times, and high GPU requirements.
5. **Start with prompt optimization.** Only reach for RL when other approaches have plateaued.
6. **LoRA + DAPO** make training more efficient and stable, but you still need serious hardware

---

## What's Next

In [9.2: RL for Complex Multi-Module Programs](../9.2-rl-complex-tasks/blog.md), we will apply ArborGRPO to real multi-module scenarios: a privacy-preserving delegation system (PAPILLON) and a multi-hop research agent. You will see concrete results and learn how RL handles programs with multiple interacting modules.

---

## Resources

- [DSPy Documentation: ArborGRPO](https://dspy.ai)
- [Arbor AI GitHub Repository](https://github.com/arbor-ai/arbor)
- [GRPO Paper: DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [DAPO: An Open-Source LLM Reinforcement Learning System](https://arxiv.org/abs/2503.14476)
- [DSPy GitHub: RL Examples](https://github.com/stanfordnlp/dspy)
- [Phase 4: Optimizer Landscape](../../04-optimization/4.4-optimizer-landscape/blog.md)
- [Blog 7.1: BootstrapFinetune](../../07-finetuning/7.1-bootstrap-finetune/blog.md)
