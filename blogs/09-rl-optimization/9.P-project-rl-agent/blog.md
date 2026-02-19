# 9.P: Project: RL-Optimized Research Agent

## Introduction

You have learned the theory behind RL optimization in DSPy ([9.1](../9.1-rl-for-dspy/blog.md)) and seen how it applies to multi-module programs ([9.2](../9.2-rl-complex-tasks/blog.md)). Now it is time to build a complete RL-optimized research agent from the ground up.

In this project, you will construct a **multi-hop research system** that iteratively searches a corpus, gathers evidence, and synthesizes a final answer. You will train it with ArborGRPO, evaluate the results, compare against MIPROv2 prompt optimization, and draw conclusions about when RL training makes practical sense.

> **EXPERIMENTAL WARNING:** This entire project uses experimental DSPy features. The Arbor RL framework, ArborGRPO compiler, and associated APIs are proof-of-concept implementations. They require multiple GPUs (4xH100 recommended), hours of training time, and may produce inconsistent results across runs. Do not use this in production.

---

## What You'll Learn

- How to build a complete multi-hop research agent in DSPy
- Setting up BM25S retrieval over a Wikipedia corpus
- Defining evaluation metrics for multi-hop research
- Configuring Arbor infrastructure for RL training
- Running ArborGRPO and monitoring with wandb
- Comparing RL results against MIPROv2 prompt optimization
- Analyzing when RL provides meaningful improvements

---

## Prerequisites

- Completed [9.1](../9.1-rl-for-dspy/blog.md) and [9.2](../9.2-rl-complex-tasks/blog.md)
- Multi-GPU setup (4xH100 recommended, minimum 4xA100)
- Python environment with DSPy, Arbor, and BM25S installed
- ~50GB disk space for Wikipedia abstracts corpus
- wandb account (optional, for training monitoring)

---

## Project Overview

We are building a system that, given a factual claim, searches Wikipedia to verify it. The system performs multiple "hops". Each hop generates a search query, retrieves relevant documents, and updates its research notes. After all hops, it determines whether the claim is supported or refuted.

Here is the high-level architecture:

```
Claim → [Hop 1: Search + Notes] → [Hop 2: Search + Notes] → [Hop 3: Search + Notes] → Verdict
              ↕                          ↕                          ↕
           BM25 Retriever             BM25 Retriever             BM25 Retriever
              ↕                          ↕                          ↕
          Wikipedia Corpus           Wikipedia Corpus           Wikipedia Corpus
```

Each hop has two LM modules: one to generate the search query, one to update notes. That is 6 LM calls plus a final verdict, 7 total LM calls per claim, all running on the same local model.

---

## Step 1: Install Dependencies

```bash
# Core dependencies
pip install -U dspy arbor-ai bm25s python-dotenv wandb

# For LoRA training
pip install peft
```

> **EXPERIMENTAL:** Make sure you have the latest versions. Arbor's API may change between releases.

---

## Step 2: Define the DSPy Program

Let's build the research agent as a composable DSPy module:

```python
import dspy
from dotenv import load_dotenv

load_dotenv()


# Module Definitions

class ResearchHop(dspy.Module):
    """
    A single research hop: generates a search query based on the
    current claim and accumulated notes, retrieves passages,
    and updates the notes with relevant information.
    """

    def __init__(self):
        self.generate_query = dspy.ChainOfThought(
            "claim, notes -> search_query"
        )
        self.append_notes = dspy.ChainOfThought(
            "claim, notes, passages -> updated_notes: str"
        )

    def forward(self, claim, notes, retriever):
        # Generate a targeted search query
        query_pred = self.generate_query(claim=claim, notes=notes)

        # Retrieve relevant passages
        passages = retriever(query_pred.search_query)

        # Update research notes with new findings
        notes_pred = self.append_notes(
            claim=claim,
            notes=notes,
            passages=passages,
        )

        return notes_pred.updated_notes


class MultiHopResearchAgent(dspy.Module):
    """
    Complete multi-hop research agent.
    Performs N hops of search-and-note, then makes a final verdict.
    """

    def __init__(self, num_hops=3):
        self.num_hops = num_hops
        self.hops = [ResearchHop() for _ in range(num_hops)]
        self.verify = dspy.ChainOfThought(
            "claim, notes -> verdict: bool"
        )

    def forward(self, claim, retriever):
        notes = "No research notes yet."

        # Iteratively search and accumulate evidence
        for i, hop in enumerate(self.hops):
            notes = hop(claim=claim, notes=notes, retriever=retriever)

        # Final verdict based on accumulated evidence
        result = self.verify(claim=claim, notes=notes)

        return dspy.Prediction(
            verdict=result.verdict,
            notes=notes,
        )
```

Each `ResearchHop` is a self-contained unit with its own `generate_query` and `append_notes` modules. The `MultiHopResearchAgent` chains three hops together and adds a final verification step.

---

## Step 3: Set Up BM25 Retrieval

We will use **BM25S** for fast keyword-based retrieval over Wikipedia abstracts:

```python
import bm25s
import json


def load_corpus(path="data/wiki_abstracts.jsonl"):
    """
    Load Wikipedia abstracts corpus.
    Each line: {"title": "...", "text": "..."}
    """
    corpus = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus.append(doc)
    print(f"Loaded {len(corpus):,} documents")
    return corpus


class BM25Retriever:
    """BM25S-based retriever for Wikipedia abstracts."""

    def __init__(self, corpus, k=5):
        self.k = k
        self.corpus = corpus
        self.retriever = bm25s.BM25()

        # Build index
        texts = [doc["text"] for doc in corpus]
        tokenized = bm25s.tokenize(texts)
        self.retriever.index(tokenized)
        print(f"BM25 index built over {len(corpus):,} documents")

    def __call__(self, query):
        """Retrieve top-k passages for a query."""
        tokenized_query = bm25s.tokenize(query)
        results, scores = self.retriever.retrieve(
            tokenized_query, k=self.k
        )

        # Format retrieved passages with titles
        passages = []
        for idx in results[0]:
            doc = self.corpus[idx]
            passages.append(
                f"[{doc['title']}] {doc['text']}"
            )

        return "\n\n".join(passages)


# Initialize retrieval
corpus = load_corpus("data/wiki_abstracts.jsonl")
retriever = BM25Retriever(corpus, k=5)
```

BM25 is intentionally simple: it is a non-neural retrieval method. The RL training has no way to "cheat" by adjusting the retriever. Instead, the model must learn to **generate better queries** that surface the right documents through keyword matching.

---

## Step 4: Load the HoVer Dataset

The **HoVer dataset** provides multi-hop claims with gold-standard Wikipedia titles used as evidence:

```python
import random


def load_hover_data(path="data/hover_train.jsonl", max_examples=500):
    """
    Load HoVer dataset examples.
    Each example: claim, label, gold_titles (list of Wikipedia titles)
    """
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            example = dspy.Example(
                claim=item["claim"],
                gold_titles=item["supporting_facts_titles"],
                label=item["label"],
            ).with_inputs("claim")
            examples.append(example)

    random.seed(42)
    random.shuffle(examples)
    return examples[:max_examples]


# Load and split data
all_examples = load_hover_data()
trainset = all_examples[:400]
devset = all_examples[400:]

print(f"Training: {len(trainset)} examples")
print(f"Validation: {len(devset)} examples")
```

---

## Step 5: Define Evaluation Metrics

Our primary metric is **recall over gold Wikipedia titles**: did the research agent find the right source documents?

```python
import re


def extract_titles_from_notes(notes):
    """
    Extract Wikipedia titles mentioned in research notes.
    Looks for titles in [brackets] format.
    """
    titles = set(re.findall(r'\[([^\]]+)\]', notes))
    return titles


def recall_metric(example, prediction, trace=None):
    """
    Recall: what fraction of gold titles appear in the notes?
    This measures whether the research process found the
    right evidence sources.
    """
    gold_titles = set(example.gold_titles)

    if not gold_titles:
        return 1.0

    # Extract titles from the accumulated notes
    found_titles = extract_titles_from_notes(prediction.notes)

    # Calculate recall
    hits = sum(
        1 for title in gold_titles
        if any(title.lower() in found.lower() for found in found_titles)
    )

    return hits / len(gold_titles)
```

Let's also set up a comprehensive evaluation function:

```python
from dspy.evaluate import Evaluate


def run_evaluation(program, devset, retriever, metric, label=""):
    """Run evaluation with the retriever injected."""

    def eval_with_retriever(example, prediction, trace=None):
        return metric(example, prediction, trace)

    # Wrap program to include retriever
    class EvalWrapper(dspy.Module):
        def __init__(self, inner_program, retriever):
            super().__init__()
            self.inner = inner_program
            self.retriever = retriever

        def forward(self, claim):
            return self.inner(claim=claim, retriever=self.retriever)

    wrapped = EvalWrapper(program, retriever)

    evaluator = Evaluate(
        devset=devset,
        metric=eval_with_retriever,
        num_threads=4,
        display_progress=True,
    )

    score = evaluator(wrapped)
    print(f"\n{'='*50}")
    print(f"  {label} Score: {score:.1f}%")
    print(f"{'='*50}\n")
    return score
```

---

## Step 6: Establish a Baseline

Before any optimization, measure the unoptimized baseline:

```python
# Configure a local model for baseline evaluation
baseline_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=baseline_lm)

# Create baseline program
baseline_program = MultiHopResearchAgent(num_hops=3)

# Evaluate baseline
baseline_score = run_evaluation(
    baseline_program, devset, retriever, recall_metric,
    label="Baseline (GPT-4o-mini)"
)
```

---

## Step 7: Compare with MIPROv2 (Prompt Optimization)

Before investing in expensive RL training, let's see what prompt optimization achieves. This serves as a critical reference point:

```python
from dspy.teleprompt import MIPROv2


def mipro_optimization(program, trainset, devset, retriever, metric):
    """
    Optimize with MIPROv2 for comparison.
    This is the recommended approach for most use cases.
    """

    # Wrap metric to include retriever
    def wrapped_metric(example, prediction, trace=None):
        return metric(example, prediction, trace)

    optimizer = MIPROv2(
        metric=wrapped_metric,
        num_candidates=10,
        init_temperature=1.0,
    )

    optimized = optimizer.compile(
        program,
        trainset=trainset,
        num_batches=30,
        max_bootstrapped_demos=3,
        max_labeled_demos=5,
    )

    return optimized


# Run MIPROv2 optimization (much faster than RL)
print("Running MIPROv2 optimization...")
mipro_program = mipro_optimization(
    MultiHopResearchAgent(num_hops=3),
    trainset, devset, retriever, recall_metric,
)

mipro_score = run_evaluation(
    mipro_program, devset, retriever, recall_metric,
    label="MIPROv2 Optimized"
)
```

MIPROv2 typically completes in **30-60 minutes on a single GPU** and achieves significant improvements. Keep this result handy: it is the benchmark that RL needs to beat to justify its cost.

---

## Step 8: Set Up Arbor Infrastructure

Now for the RL path. This is where the GPU requirements become serious:

```python
import arbor
from arbor import ArborProvider
from peft import LoraConfig


def setup_arbor(model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    """
    Initialize Arbor RL infrastructure.
    EXPERIMENTAL: requires multi-GPU setup
    """
    # Initialize Arbor framework
    arbor.init()

    # Start the Arbor server with the local model
    server_info = arbor.start_server(model_name)
    print(f"Arbor server running at: {server_info['base_url']}")

    # Create DSPy LM pointing to Arbor
    lm = dspy.LM(
        model=f"openai/arbor:{model_name}",
        provider=ArborProvider(),
        api_base=server_info["base_url"],
    )

    return lm, server_info


# Set up Arbor
# This requires multiple GPUs!
arbor_lm, server_info = setup_arbor("Qwen/Qwen2.5-1.5B-Instruct")
dspy.configure(lm=arbor_lm)
```

---

## Step 9: Configure and Run ArborGRPO

```python
from dspy.teleprompt import ArborGRPO


def configure_rl_training():
    """
    Configure ArborGRPO for multi-hop research training.
    EXPERIMENTAL: training will take many hours
    """

    # LoRA config for parameter-efficient training
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
        "loss_type": "dapo",
        "report_to": "wandb",         # Monitor training curves
    }

    # ArborGRPO compiler configuration
    compiler = ArborGRPO(
        metric=recall_metric,
        num_dspy_examples_per_grpo_step=4,
        num_rollouts_per_grpo_step=4,
        exclude_demos=True,
        num_train_steps=300,
        checkpoint="checkpoints/rl_research_agent",
        train_kwargs=train_kwargs,
    )

    return compiler


# Configure the RL compiler
compiler = configure_rl_training()

# Create a fresh student program for RL training
rl_student = MultiHopResearchAgent(num_hops=3)

# Run RL training
# This will take approximately 18 hours on 4xH100!
print("Starting RL training...")
print("Monitor progress at: https://wandb.ai")
print("This will take many hours. Checkpoints saved periodically.")

rl_optimized = compiler.compile(
    student=rl_student,
    trainset=trainset,
    valset=devset,
)
```

### Monitoring with wandb

If you set `report_to="wandb"` in `train_kwargs`, you can monitor training in real time. Key metrics to watch:

- **reward/mean**: Should trend upward over training steps
- **loss/policy**: The GRPO policy loss, should generally decrease
- **reward/std**: Reward variance across rollouts, high variance is fine early but should stabilize
- **eval/recall**: Periodic evaluation on the validation set

```python
# Tip: Set up wandb before training
import wandb
wandb.init(
    project="dspy-rl-research-agent",
    name="arborgrpo-multihop-3hop",
    config={
        "model": "Qwen2.5-1.5B-Instruct",
        "num_hops": 3,
        "num_train_steps": 300,
        "learning_rate": 5e-6,
        "loss_type": "dapo",
    },
)
```

---

## Step 10: Evaluate and Compare

After training completes, evaluate the RL-optimized agent:

```python
# Evaluate the RL-optimized program
rl_score = run_evaluation(
    rl_optimized, devset, retriever, recall_metric,
    label="ArborGRPO (RL Optimized)"
)

# Print comparison
print("\n" + "=" * 60)
print("  FINAL COMPARISON")
print("=" * 60)
print(f"  Baseline (GPT-4o-mini):     {baseline_score:.1f}%")
print(f"  MIPROv2 (prompt opt):       {mipro_score:.1f}%")
print(f"  ArborGRPO (RL, Qwen 1.5B):  {rl_score:.1f}%")
print("=" * 60)
```

### Expected Results

Based on the research experiments documented in [9.2](../9.2-rl-complex-tasks/blog.md), approximate results:

| Approach | Recall Score | Training Time | Hardware | Approx. Cost |
|----------|-------------|---------------|----------|---------------|
| Baseline (no opt) | ~61.8% | n/a | 1 GPU | n/a |
| MIPROv2 | ~65% | 30-60 min | 1 GPU | ~\$2-5 |
| ArborGRPO | ~66.2% | ~18 hours | 4xH100 | ~\$270 |

RL training achieves a modest improvement over MIPROv2, but at dramatically higher cost. The value proposition depends entirely on your deployment scenario.

---

## Analysis and Lessons Learned

### When RL Makes the Difference

RL optimization works best when:

1. **You must deploy locally.** If you need a self-contained model running on your own hardware without API calls, RL lets you squeeze more performance out of a small model. MIPROv2 cannot help if the model fundamentally struggles to follow the optimized prompts.

2. **The task requires coordination across modules.** RL trains all modules jointly with end-to-end reward signals. Prompt optimization tunes each module's prompts somewhat independently. For tightly coupled multi-module programs, RL's joint optimization can find solutions that prompt optimization misses.

3. **You have reached the prompt optimization ceiling.** If you have run MIPROv2 and SIMBA with extensive tuning and the score will not budge, RL offers a different axis of improvement.

### When RL Does Not Make Sense

1. **You have access to larger models.** Using GPT-4o with MIPROv2 will almost certainly outperform a 1.5B model with RL, and at lower total cost.

2. **Training resources are limited.** If you do not have 4+ high-end GPUs available for 18+ hours, RL is not practical.

3. **Iteration speed matters.** RL takes hours to get feedback. MIPROv2 gives you results in under an hour. For rapid prototyping, prompt optimization wins hands down.

4. **Reproducibility is critical.** RL training has higher variance across runs. The same configuration can produce different results with different random seeds.

### The Optimization Gradient

For nearly any DSPy project, follow this progression:

```
1. Start with dspy.MIPROv2 or dspy.SIMBA         > 80% of possible gains
2. Try dspy.BootstrapFinetune for distillation   > Move to cheaper model
3. Consider ArborGRPO only if 1 & 2 plateau      > Squeeze out last points
```

This is directly from the DSPy team's guidance: ArborGRPO is "a solid start for online RL over arbitrary LM programs for small LMs," but MIPROv2 and SIMBA remain the recommended first-line optimizers.

---

## Complete Project Code

Here is the full script in one place for reference:

```python
# Project 9.P: RL-Optimized Research Agent (Complete)
# EXPERIMENTAL: requires Arbor RL framework and 4xH100 GPUs
#
# Usage:
#     python rl_research_agent.py --mode baseline
#     python rl_research_agent.py --mode mipro
#     python rl_research_agent.py --mode rl
#     python rl_research_agent.py --mode compare

import argparse
import json
import random
import re

import bm25s
import dspy
from dotenv import load_dotenv
from dspy.evaluate import Evaluate

load_dotenv()


# ===== Data Loading =====

def load_corpus(path="data/wiki_abstracts.jsonl"):
    corpus = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    return corpus


def load_hover_data(path="data/hover_train.jsonl", max_examples=500):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            examples.append(
                dspy.Example(
                    claim=item["claim"],
                    gold_titles=item["supporting_facts_titles"],
                    label=item["label"],
                ).with_inputs("claim")
            )
    random.seed(42)
    random.shuffle(examples)
    return examples[:max_examples]


# ===== Retrieval =====

class BM25Retriever:
    def __init__(self, corpus, k=5):
        self.k = k
        self.corpus = corpus
        self.retriever = bm25s.BM25()
        tokenized = bm25s.tokenize([d["text"] for d in corpus])
        self.retriever.index(tokenized)

    def __call__(self, query):
        tokenized_query = bm25s.tokenize(query)
        results, scores = self.retriever.retrieve(
            tokenized_query, k=self.k
        )
        passages = []
        for idx in results[0]:
            doc = self.corpus[idx]
            passages.append(f"[{doc['title']}] {doc['text']}")
        return "\n\n".join(passages)


# ===== DSPy Program =====

class ResearchHop(dspy.Module):
    def __init__(self):
        self.generate_query = dspy.ChainOfThought(
            "claim, notes -> search_query"
        )
        self.append_notes = dspy.ChainOfThought(
            "claim, notes, passages -> updated_notes: str"
        )

    def forward(self, claim, notes, retriever):
        query_pred = self.generate_query(claim=claim, notes=notes)
        passages = retriever(query_pred.search_query)
        notes_pred = self.append_notes(
            claim=claim, notes=notes, passages=passages
        )
        return notes_pred.updated_notes


class MultiHopResearchAgent(dspy.Module):
    def __init__(self, num_hops=3):
        self.hops = [ResearchHop() for _ in range(num_hops)]
        self.verify = dspy.ChainOfThought(
            "claim, notes -> verdict: bool"
        )

    def forward(self, claim, retriever):
        notes = "No research notes yet."
        for hop in self.hops:
            notes = hop(claim=claim, notes=notes, retriever=retriever)
        result = self.verify(claim=claim, notes=notes)
        return dspy.Prediction(verdict=result.verdict, notes=notes)


# ===== Metrics =====

def extract_titles_from_notes(notes):
    return set(re.findall(r'\[([^\]]+)\]', notes))


def recall_metric(example, prediction, trace=None):
    gold_titles = set(example.gold_titles)
    if not gold_titles:
        return 1.0
    found = extract_titles_from_notes(prediction.notes)
    hits = sum(
        1 for t in gold_titles
        if any(t.lower() in f.lower() for f in found)
    )
    return hits / len(gold_titles)


# ===== Main =====

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["baseline", "mipro", "rl", "compare"],
        default="baseline"
    )
    args = parser.parse_args()

    # Load data
    corpus = load_corpus()
    retriever = BM25Retriever(corpus, k=5)
    all_examples = load_hover_data()
    trainset = all_examples[:400]
    devset = all_examples[400:]

    if args.mode in ("baseline", "compare"):
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm)
        program = MultiHopResearchAgent(num_hops=3)
        # Run baseline evaluation
        evaluator = Evaluate(
            devset=devset, metric=recall_metric,
            num_threads=4, display_progress=True,
        )
        print(f"Baseline: {evaluator(program):.1f}%")

    if args.mode in ("mipro", "compare"):
        from dspy.teleprompt import MIPROv2
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm)
        optimizer = MIPROv2(metric=recall_metric, num_candidates=10)
        optimized = optimizer.compile(
            MultiHopResearchAgent(num_hops=3),
            trainset=trainset,
            num_batches=30,
            max_bootstrapped_demos=3,
            max_labeled_demos=5,
        )
        evaluator = Evaluate(
            devset=devset, metric=recall_metric,
            num_threads=4, display_progress=True,
        )
        print(f"MIPROv2: {evaluator(optimized):.1f}%")

    if args.mode in ("rl", "compare"):
        import arbor
        from arbor import ArborProvider
        from peft import LoraConfig
        from dspy.teleprompt import ArborGRPO

        arbor.init()
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        server_info = arbor.start_server(model_name)

        lm = dspy.LM(
            model=f"openai/arbor:{model_name}",
            provider=ArborProvider(),
            api_base=server_info["base_url"],
        )
        dspy.configure(lm=lm)

        lora_config = LoraConfig(
            r=16, lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
        )

        compiler = ArborGRPO(
            metric=recall_metric,
            num_dspy_examples_per_grpo_step=4,
            num_rollouts_per_grpo_step=4,
            exclude_demos=True,
            num_train_steps=300,
            checkpoint="checkpoints/rl_research_agent",
            train_kwargs={
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "temperature": 1.0,
                "learning_rate": 5e-6,
                "lora_config": lora_config,
                "num_training_gpus": 2,
                "num_inference_gpus": 2,
                "loss_type": "dapo",
                "report_to": "wandb",
            },
        )

        rl_optimized = compiler.compile(
            student=MultiHopResearchAgent(num_hops=3),
            trainset=trainset,
            valset=devset,
        )

        evaluator = Evaluate(
            devset=devset, metric=recall_metric,
            num_threads=4, display_progress=True,
        )
        print(f"ArborGRPO: {evaluator(rl_optimized):.1f}%")
```

---

## Key Takeaways

1. **RL optimization is a last resort**, not a first step. Always try MIPROv2 or SIMBA first.
2. **Multi-hop research** is an ideal testbed for RL because it has many interacting modules.
3. **BM25 retrieval** keeps the focus on the LM: the model must learn to generate good queries for a fixed retriever.
4. **Monitoring with wandb** is essential for long RL training runs. Watch reward curves for convergence.
5. **Cost-benefit analysis** should guide your decision: RL gives ~1 point over MIPROv2 at ~50x the cost in this example.
6. **Checkpointing** is critical. 18-hour training runs must be resumable.
7. **This is all experimental.** Exciting research, but not production-ready.

---

## What's Next

[Phase 10: Multi-Modal DSPy](../../10-multi-modal/10.1-image-audio/blog.md)

---

## Resources

- [DSPy Documentation](https://dspy.ai)
- [Arbor AI GitHub Repository](https://github.com/arbor-ai/arbor)
- [HoVer Dataset](https://hover-nlp.github.io/)
- [BM25S: Fast BM25 in Python](https://github.com/xhluca/bm25s)
- [Weights & Biases](https://wandb.ai)
- [GRPO Paper: DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [Blog 9.1: RL Optimization for DSPy](../9.1-rl-for-dspy/blog.md)
- [Blog 9.2: RL for Complex Multi-Module Programs](../9.2-rl-complex-tasks/blog.md)
- [Blog 4.4: Optimizer Landscape](../../04-optimization/4.4-optimizer-landscape/blog.md)
- [Blog 7.1: BootstrapFinetune](../../07-finetuning/7.1-bootstrap-finetune/blog.md)
