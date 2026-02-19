# 9.2: RL for Complex Multi-Module Programs

## Introduction

In [9.1](../9.1-rl-for-dspy/blog.md), we introduced ArborGRPO and the mechanics of RL optimization in DSPy. We trained a simple single-module program. But the real power of DSPy's RL approach is that it generalizes GRPO to **multi-module LM programs**: pipelines where multiple `Predict`, `ChainOfThought`, or custom modules interact, each contributing to the final output.

This is what makes ArborGRPO fundamentally different from applying standard RL to a single model. When your DSPy program has two or three modules that must coordinate (one crafting a query, another interpreting results, a third synthesizing an answer), the RL reward signal flows through the entire pipeline. All modules improve *together*.

In this blog, we will walk through two concrete case studies where RL optimization was applied to multi-module DSPy programs, examine the results, and discuss when this approach makes practical sense.

> **EXPERIMENTAL WARNING:** Everything in this blog uses experimental DSPy RL features. Results may vary, APIs may change, and training requires significant GPU resources. These case studies are drawn from research examples, not production deployments.

---

## What You'll Learn

- How ArborGRPO handles multi-module DSPy programs
- Case Study 1: PAPILLON, privacy-preserving delegation with dual objectives
- Case Study 2: Multi-hop research, BM25 retrieval with iterative search
- How to define composite metrics for multi-objective RL optimization
- The `multitask=True` parameter for multi-objective training
- Practical results: what improvements to expect and at what cost
- When RL outperforms (or underperforms) prompt optimization

---

## Prerequisites

- Completed [9.1: RL Optimization for DSPy](../9.1-rl-for-dspy/blog.md)
- Familiarity with DSPy's multi-module patterns from [1.4: Custom Modules](../../01-foundations/1.4-custom-modules/blog.md)
- Understanding of RAG concepts from [Phase 5](../../05-retrieval-rag/5.2-building-rag/blog.md)
- Access to 4xH100 GPUs or similar for reproducing experiments

---

## How ArborGRPO Handles Multi-Module Programs

When you pass a multi-module DSPy program to ArborGRPO, Arbor traces every LM call in the forward pass. If your program calls three modules (say, `generate_query`, `retrieve_context`, and `synthesize_answer`), the RL training loop:

1. **Generates rollouts** for the entire pipeline (all modules execute end-to-end)
2. **Scores the final output** using your metric (the reward signal)
3. **Backpropagates the reward** to all LM modules in the program
4. **Updates weights** so that all modules learn to coordinate better

This is crucial: the model does not just learn to answer questions better. It learns to **ask better intermediate questions**, **take better notes**, and **synthesize more effectively**, all in service of the end-to-end objective.

Since all modules share the same local model (e.g., Qwen2.5-1.5B-Instruct), the weight updates affect every module simultaneously. The model becomes better at all roles it plays in the pipeline.

---

## Case Study 1: PAPILLON, Privacy-Preserving Delegation

### The Problem

PAPILLON (from the DSPy examples collection) tackles a fascinating challenge: how can a **small, trusted local model** delegate work to a **powerful but untrusted external model** without leaking private information?

Imagine you have a question that contains sensitive data (medical records, financial details, personal identifiers). You want GPT-4.1-mini's intelligence, but you cannot send the raw question because it would expose private information. The solution: use a local model to **redact** the question before sending it out, then use the external model's response.

### The Architecture

PAPILLON uses two DSPy modules running on different models:

```python
import dspy

class PAPILLON(dspy.Module):
    def __init__(self):
        # Module 1: Runs on local trusted model (Qwen2.5-1.5B)
        # Crafts a redacted version of the question
        self.craft = dspy.ChainOfThought(
            "context, question -> redacted_request"
        )

        # Module 2: Runs on untrusted external model (GPT-4.1-mini)
        # Answers the redacted question
        self.respond = dspy.Predict(
            "redacted_request -> response"
        )

    def forward(self, context, question):
        # Step 1: Local model redacts the question
        redacted = self.craft(context=context, question=question)

        # Step 2: External model answers the redacted question
        result = self.respond(redacted_request=redacted.redacted_request)

        return dspy.Prediction(
            redacted_request=redacted.redacted_request,
            response=result.response,
        )
```

The key insight: only `CraftRedactedRequest` runs on the local model being RL-trained. The external model is frozen; it is just an API call. RL training optimizes the local model to produce better redactions that simultaneously:

1. **Preserve enough information** for the external model to answer correctly (quality)
2. **Remove private information** so the external model does not see sensitive data (privacy)

### The Dataset

PAPILLON uses the **PUPA dataset** (Privacy-Utility Preservation Assessment) from Columbia NLP. Each example contains a context with private information, a question, and annotations for what constitutes a privacy leak.

### The Composite Metric

This is a multi-objective problem. We need to balance quality and privacy:

```python
def papillon_metric(example, prediction, trace=None):
    """
    Composite metric balancing quality and privacy.
    Quality: Does the response correctly answer the question?
    Leakage: Does the redacted request reveal private information?
    """
    # Evaluate answer quality (0.0 to 1.0)
    quality = evaluate_answer_quality(
        prediction.response,
        example.gold_answer
    )

    # Evaluate privacy leakage (0.0 to 1.0, lower is better)
    leakage = evaluate_privacy_leakage(
        prediction.redacted_request,
        example.private_entities
    )

    # Composite score: maximize quality, minimize leakage
    score = (quality + (1.0 - leakage)) / 2.0
    return score
```

The metric returns a score between 0 and 1. A perfect score means the response is correct AND no private information was leaked. The RL training loop optimizes the local model to maximize this composite objective.

### Training Configuration

```python
import arbor
from arbor import ArborProvider
from peft import LoraConfig
from dspy.teleprompt import ArborGRPO

# Initialize Arbor with local model
arbor.init()
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
arbor_server_info = arbor.start_server(model_name)

lm_local = dspy.LM(
    model=f"openai/arbor:{model_name}",
    provider=ArborProvider(),
    api_base=arbor_server_info["base_url"],
)

# External model (not trained, just used for inference)
lm_external = dspy.LM("openai/gpt-4.1-mini")

# LoRA config for efficient training
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

train_kwargs = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "temperature": 1.0,
    "learning_rate": 5e-6,
    "lora_config": lora_config,
    "num_training_gpus": 2,
    "num_inference_gpus": 2,
    "loss_type": "dapo",
}

compiler = ArborGRPO(
    metric=papillon_metric,
    multitask=True,                       # Multi-objective optimization
    num_dspy_examples_per_grpo_step=8,
    num_rollouts_per_grpo_step=4,
    exclude_demos=True,
    num_train_steps=150,
    checkpoint="checkpoints/papillon_rl",
    train_kwargs=train_kwargs,
)

optimized = compiler.compile(
    student=program,
    trainset=trainset,
    valset=devset,
)
```

### Results

On 4xH100 GPUs, training took approximately **3 hours**:

| Metric | Before RL | After RL | Improvement |
|--------|-----------|----------|-------------|
| Composite (quality + privacy) | 54.6% | 60.0% | **+5.4 points** |

A 5.4-point improvement may seem modest, but consider: this is a 1.5B parameter model learning to **simultaneously** preserve privacy and maintain answer quality, two objectives that directly conflict. The fact that RL can improve both in a unified training loop is the fundamental advance.

> **EXPERIMENTAL NOTE:** These results are from research experiments. Your mileage will vary depending on GPU setup, training duration, hyperparameters, and random seeds.

---

## Case Study 2: Multi-Hop Research Agent

### The Problem

Multi-hop fact verification requires gathering evidence from multiple sources. Given a claim like *"The director of Inception also directed a movie about dreams starring Leonardo DiCaprio"*, the system needs to:

1. Search for relevant documents
2. Read and take notes
3. Search again based on what it learned
4. Combine evidence to verify the claim

### The Architecture

```python
import dspy

class ResearchHop(dspy.Module):
    """A single hop: generate a search query and take notes."""

    def __init__(self):
        # Generate a search query based on current notes
        self.generate_query = dspy.ChainOfThought(
            "claim, notes -> search_query"
        )
        # Read retrieved passages and append to notes
        self.append_notes = dspy.ChainOfThought(
            "claim, notes, passages -> updated_notes"
        )

    def forward(self, claim, notes, retriever):
        # Generate a targeted search query
        query_result = self.generate_query(
            claim=claim,
            notes=notes
        )

        # Retrieve passages using BM25
        passages = retriever(query_result.search_query)

        # Update notes with new information
        notes_result = self.append_notes(
            claim=claim,
            notes=notes,
            passages=passages,
        )

        return notes_result.updated_notes


class MultiHopResearcher(dspy.Module):
    """Multi-hop research system that iteratively gathers evidence."""

    def __init__(self, num_hops=3):
        self.hops = [ResearchHop() for _ in range(num_hops)]
        self.verify = dspy.ChainOfThought(
            "claim, notes -> verdict"
        )

    def forward(self, claim, retriever):
        notes = "No notes yet."

        for hop in self.hops:
            notes = hop(claim=claim, notes=notes, retriever=retriever)

        result = self.verify(claim=claim, notes=notes)
        return dspy.Prediction(
            verdict=result.verdict,
            notes=notes,
        )
```

This is a significantly more complex program than PAPILLON. Each `ResearchHop` has two LM modules (`generate_query` and `append_notes`), and there are 3 hops plus a final verification step. That is **7 LM calls per example**, all sharing the same local model. RL training must optimize all 7 calls jointly.

### Retrieval Setup

The retrieval component uses **BM25** over Wikipedia abstracts:

```python
import bm25s

class BM25Retriever:
    def __init__(self, corpus, k=5):
        self.k = k
        self.retriever = bm25s.BM25()
        self.corpus = corpus
        # Tokenize and index
        tokenized = bm25s.tokenize(
            [doc["text"] for doc in corpus]
        )
        self.retriever.index(tokenized)

    def __call__(self, query):
        tokenized_query = bm25s.tokenize(query)
        results, scores = self.retriever.retrieve(
            tokenized_query, k=self.k
        )
        return "\n\n".join(
            self.corpus[idx]["text"]
            for idx in results[0]
        )

# Load corpus (Wikipedia abstracts)
corpus = load_wikipedia_abstracts()  # Your data loading function
retriever = BM25Retriever(corpus, k=5)
```

BM25 is non-differentiable: it is a traditional keyword search algorithm. But that is fine. ArborGRPO does not need differentiable intermediate steps. It only needs the final metric. The RL signal teaches the model to **generate better search queries** that work well with BM25, and to **take better notes** from the retrieved passages.

### Dataset and Metric

The experiment uses the **HoVer dataset** (Hover: A Dataset for Many-Hop Fact Extraction and Claim Verification), which requires multi-hop reasoning over Wikipedia:

```python
def recall_metric(example, prediction, trace=None):
    """
    Recall over gold Wikipedia titles.
    Measures whether the research process found
    the right source documents.
    """
    gold_titles = set(example.gold_titles)

    # Extract titles from the notes
    found_titles = extract_titles_from_notes(prediction.notes)

    if not gold_titles:
        return 1.0

    recall = len(gold_titles & found_titles) / len(gold_titles)
    return recall
```

This metric evaluates whether the multi-hop search process identified the correct source documents. A score of 1.0 means all gold-standard sources were found.

### Training and Results

```python
compiler = ArborGRPO(
    metric=recall_metric,
    num_dspy_examples_per_grpo_step=4,     # Smaller batch (more modules per example)
    num_rollouts_per_grpo_step=4,
    exclude_demos=True,
    num_train_steps=300,                   # More steps for complex program
    checkpoint="checkpoints/multihop_rl",
    train_kwargs=train_kwargs,
)

optimized_researcher = compiler.compile(
    student=MultiHopResearcher(num_hops=3),
    trainset=trainset,
    valset=devset,
)
```

Training on 4xH100 GPUs took approximately **18 hours** (6x longer than PAPILLON), reflecting the much higher complexity (7 LM calls per example vs. 2):

| Metric | Before RL | After RL | Improvement |
|--------|-----------|----------|-------------|
| Recall over gold titles | 61.8% | 66.2% | **+4.4 points** |

Again, a modest but meaningful improvement. The model learned to generate more targeted search queries and extract more relevant information from retrieved passages, all from the reward signal alone.

---

## Multi-Objective Training with `multitask=True`

When your metric involves multiple objectives (like PAPILLON's quality vs. privacy trade-off), you can use the `multitask=True` parameter:

```python
compiler = ArborGRPO(
    metric=composite_metric,
    multitask=True,    # Enable multi-objective optimization
    # ... other params
)
```

With `multitask=True`, ArborGRPO structures the training to balance multiple goals rather than collapsing them all into a single scalar. This is particularly useful when your objectives can conflict (improving one dimension at the expense of another).

> **EXPERIMENTAL NOTE:** The `multitask` parameter behavior is still evolving. In current implementations, it primarily affects how the reward signal is structured during GRPO training. Future versions may offer more fine-grained multi-objective control.

---

## Practical Considerations

### GPU Requirements and Training Time

| Program Complexity | Approximate Training Time (4\u00d7H100) | Estimated Cost |
|-------------------|-------------------------------------|----------------|
| Single module | 1-3 hours | \$15-45 |
| Two modules (PAPILLON) | ~3 hours | ~\$45 |
| 7 modules (Multi-Hop) | ~18 hours | ~\$270 |

Training time scales roughly linearly with the number of LM calls per example. More modules means more rollouts, more gradient computation, and more time.

### Cost vs. Benefit

Let's be honest: is a 4-5 point improvement worth \$45-\$270 in GPU costs and hours of training time?

**Sometimes yes:**
- If this is a model you will deploy locally and run millions of times
- If you are in a research setting exploring RL for LM programs
- If prompt optimization has genuinely plateaued and you need every point

**Usually no:**
- If you can use a larger model with prompt optimization
- If you have not tried MIPROv2 or SIMBA first
- If you need results quickly
- If you do not have multi-GPU access

### RL vs. Prompt Optimization: Head-to-Head

For the multi-hop research task, here is how the approaches compare:

| Approach | Score | Training Time | GPUs Required |
|----------|-------|---------------|---------------|
| Baseline (no optimization) | 61.8% | n/a | 1 GPU |
| MIPROv2 (prompt optimization) | ~65% | 30 min | 1 GPU |
| ArborGRPO (RL) | 66.2% | 18 hours | 4xH100 |

RL ekes out a small additional gain, but at dramatically higher cost. The DSPy team's assessment is candid: RL is "typically worse on a cost/quality basis than MIPROv2 or SIMBA." But the technology is young, and for small local models, it offers a path to improvement that prompt optimization cannot.

---

## Key Takeaways

1. **ArborGRPO trains all modules jointly.** The reward signal flows through the entire multi-module pipeline.
2. **PAPILLON** demonstrates RL for dual-objective optimization (quality + privacy) on a 1.5B model.
3. **Multi-hop research** shows RL can improve complex 7-module programs, learning better search and note-taking strategies.
4. **Results are modest but real:** 4-5 point improvements over baselines.
5. **Cost is high:** multi-GPU setups and hours of training for relatively small gains.
6. **Start with prompt optimization.** MIPROv2 or SIMBA will get you 80% of the way at 1% of the cost.
7. **This is all experimental.** Treat these as research results, not production benchmarks.

---

## What's Next

In [9.P: Project: RL-Optimized Research Agent](../9.P-project-rl-agent/blog.md), you will build a complete RL-optimized research agent from scratch, from data loading to Arbor setup to training to evaluation. This hands-on project will tie together everything from Phase 9 into a working system.

---

## Resources

- [DSPy Documentation: ArborGRPO](https://dspy.ai)
- [Arbor AI GitHub Repository](https://github.com/arbor-ai/arbor)
- [PAPILLON: DSPy Examples](https://github.com/stanfordnlp/dspy/tree/main/examples)
- [HoVer Dataset](https://hover-nlp.github.io/)
- [PUPA Dataset: Columbia NLP](https://github.com/columbia-ai-privacy/pupa)
- [BM25S: Fast BM25 in Python](https://github.com/xhluca/bm25s)
- [Blog 9.1: RL Optimization for DSPy](../9.1-rl-for-dspy/blog.md)
- [Blog 7.1: BootstrapFinetune](../../07-finetuning/7.1-bootstrap-finetune/blog.md)
