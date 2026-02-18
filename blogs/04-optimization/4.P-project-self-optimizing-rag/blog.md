# 4.P: Project: Self-Optimizing RAG

## Introduction

You've learned three optimizers and a decision framework. Now let's put it all together in a real-world scenario: building a **Retrieval-Augmented Generation (RAG)** pipeline and then **optimizing it end-to-end** with MIPROv2.

This project mirrors the official DSPy RAG tutorial pattern. You'll build a baseline RAG system, measure its performance, optimize it, and compare the results, all in about 100 lines of code.

---

## Project Overview

Here's what we'll build:

1. **A RAG module**: retriever + response generator
2. **An evaluation dataset**: question/answer pairs with gold answers
3. **A metric**: SemanticF1 for measuring answer quality
4. **Baseline evaluation**: how the unoptimized program performs
5. **MIPROv2 optimization**: automatic prompt and demo tuning
6. **Before/after comparison**: quantifying the improvement

---

## Step 1: Build the RAG Module

We'll create a simple but complete RAG pipeline. DSPy provides built-in retrieval integration, but for this project we'll use a `dspy.ColBERTv2` retriever pointing to a Wikipedia corpus:

```python

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Configure retriever: ColBERTv2 over Wikipedia abstracts
retriever = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")


class RAG(dspy.Module):
    """A simple Retrieve-then-Read RAG pipeline."""

    def __init__(self, num_passages=3):
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.respond(context=context, question=question)
```

This module:
1. Takes a question
2. Retrieves the top-k most relevant passages
3. Uses `ChainOfThought` to reason over the context and produce a response

---

## Step 2: Prepare Data

We need a training set and a dev set. For this project, we'll create a small dataset of factual questions with known answers:

```python
# Training set: used for optimization
trainset = [
    dspy.Example(question="What is the tallest mountain in the world?", response="Mount Everest").with_inputs("question"),
    dspy.Example(question="Who invented the telephone?", response="Alexander Graham Bell").with_inputs("question"),
    dspy.Example(question="What is the chemical formula for water?", response="H2O").with_inputs("question"),
    dspy.Example(question="Which country has the largest population?", response="China").with_inputs("question"),
    dspy.Example(question="What year did World War II end?", response="1945").with_inputs("question"),
    dspy.Example(question="Who painted the Sistine Chapel ceiling?", response="Michelangelo").with_inputs("question"),
    dspy.Example(question="What is the speed of light in km/s?", response="299,792 km/s").with_inputs("question"),
    dspy.Example(question="What is the largest organ in the human body?", response="The skin").with_inputs("question"),
    dspy.Example(question="Who wrote the theory of relativity?", response="Albert Einstein").with_inputs("question"),
    dspy.Example(question="What is the capital of Australia?", response="Canberra").with_inputs("question"),
    dspy.Example(question="What element has the atomic number 1?", response="Hydrogen").with_inputs("question"),
    dspy.Example(question="Who was the first person to walk on the Moon?", response="Neil Armstrong").with_inputs("question"),
    dspy.Example(question="What is the longest river in the world?", response="The Nile").with_inputs("question"),
    dspy.Example(question="What language has the most native speakers?", response="Mandarin Chinese").with_inputs("question"),
    dspy.Example(question="What is the boiling point of water in Fahrenheit?", response="212°F").with_inputs("question"),
    dspy.Example(question="Who discovered gravity?", response="Isaac Newton").with_inputs("question"),
    dspy.Example(question="What is the largest desert in the world?", response="The Sahara Desert").with_inputs("question"),
    dspy.Example(question="What is the currency of Japan?", response="Japanese yen").with_inputs("question"),
    dspy.Example(question="Who wrote Hamlet?", response="William Shakespeare").with_inputs("question"),
    dspy.Example(question="What is the smallest planet in our solar system?", response="Mercury").with_inputs("question"),
]

# Dev set: used for evaluation (held out from training)
devset = [
    dspy.Example(question="What is the largest planet in our solar system?", response="Jupiter").with_inputs("question"),
    dspy.Example(question="Who developed the polio vaccine?", response="Jonas Salk").with_inputs("question"),
    dspy.Example(question="What is the freezing point of water in Celsius?", response="0°C").with_inputs("question"),
    dspy.Example(question="What is the capital of Brazil?", response="Brasília").with_inputs("question"),
    dspy.Example(question="Who wrote Pride and Prejudice?", response="Jane Austen").with_inputs("question"),
    dspy.Example(question="What is the most abundant gas in Earth's atmosphere?", response="Nitrogen").with_inputs("question"),
    dspy.Example(question="What is the deepest ocean?", response="The Pacific Ocean").with_inputs("question"),
    dspy.Example(question="Who invented the light bulb?", response="Thomas Edison").with_inputs("question"),
    dspy.Example(question="What is the hardest natural substance?", response="Diamond").with_inputs("question"),
    dspy.Example(question="What year did the Berlin Wall fall?", response="1989").with_inputs("question"),
]
```

> **Note:** In a production scenario, you'd want 50+ training examples and 100+ dev examples. This smaller set keeps the project quick and affordable.

---

## Step 3: Define Metric

We'll use `SemanticF1`, which measures the token-level overlap between the predicted response and the gold answer. It's more forgiving than exact match; "Mount Everest" and "Mt. Everest is the tallest" would get a partial score:

```python
def semantic_f1(example, prediction, trace=None):
    """Compute token-level F1 between prediction and gold answer."""
    pred_tokens = set(prediction.response.lower().split())
    gold_tokens = set(example.response.lower().split())

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = pred_tokens & gold_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1
```

> **Tip:** DSPy also provides `dspy.evaluate.SemanticF1` as a built-in metric for convenience. The implementation above keeps things transparent.

---

## Step 4: Baseline Evaluation

Before optimizing, let's measure how the unoptimized RAG performs:

```python
# Create the baseline RAG program
rag = RAG(num_passages=3)

# Run evaluation
evaluate = dspy.Evaluate(
    devset=devset,
    metric=semantic_f1,
    num_threads=4,
    display_progress=True,
    display_table=5,  # Show 5 example results
)

baseline_score = evaluate(rag)
print(f"\nBaseline SemanticF1: {baseline_score:.1f}%")
```

This gives you a concrete number to beat. Typical baseline scores for simple RAG on factual questions range from 30-60% depending on retrieval quality and question complexity.

---

## Step 5: Optimize with MIPROv2

Now let's optimize the RAG pipeline. MIPROv2 will tune both the **instruction** and **demonstrations** for the `respond` predictor:

```python
# Configure the optimizer
tp = dspy.MIPROv2(
    metric=semantic_f1,
    auto="medium",
    num_threads=4,
)

# Compile: this takes 10-20 minutes
optimized_rag = tp.compile(
    rag,
    trainset=trainset,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
)
```

During optimization, you'll see MIPROv2:
1. Generate candidate instructions for the `respond` predictor
2. Bootstrap demonstrations using the training set
3. Run Bayesian trials to find the best instruction + demo combination
4. Return the top-performing configuration

---

## Step 6: Evaluate Optimized Program

Run the same evaluation on the optimized program:

```python
optimized_score = evaluate(optimized_rag)
print(f"\nOptimized SemanticF1: {optimized_score:.1f}%")
print(f"Improvement: {optimized_score - baseline_score:+.1f}%")
```

---

## Step 7: Inspect and Save

Let's see what MIPROv2 discovered:

```python
# Inspect the optimized prompt
result = optimized_rag(question="What is the capital of France?")
dspy.inspect_history(n=1)

# Check what demonstrations were selected
for predictor in optimized_rag.predictors():
    print(f"\nPredictor demos ({len(predictor.demos)}):")
    for i, demo in enumerate(predictor.demos):
        print(f"  Demo {i+1}:")
        print(f"    Q: {demo.get('question', 'N/A')}")
        print(f"    R: {demo.get('response', 'N/A')[:80]}...")

# Save the optimized program for production use
optimized_rag.save("optimized_rag.json")
print("\nOptimized program saved to optimized_rag.json")

# Later, load it back
loaded_rag = RAG(num_passages=3)
loaded_rag.load("optimized_rag.json")
```

---

## Results Summary

Here's what a typical run looks like:

| Stage | SemanticF1 | Cost | Time |
|-------|-----------|------|------|
| Baseline (unoptimized) | ~40-50% | \$0 | Instant |
| MIPROv2 auto="medium" | ~55-70% | ~\$1.50 | ~15-20 min |

The exact numbers depend on the retrieval quality and question difficulty, but a **10-25% improvement** from a single `MIPROv2` run is common.

### What Happened Under the Hood

1. **Instruction optimization**: MIPROv2 likely discovered an instruction like *"Using the provided context passages, give a precise and direct answer to the question. Include specific facts, numbers, or names from the context."*
2. **Demo selection**: The optimizer chose demonstrations that model the ideal output format: concise, factual answers that directly reference the retrieved context.
3. **Combined effect**: Better instructions tell the model *how* to behave; better demos show it *what* good answers look like. Together, they compound for maximum improvement.

---

## What We Learned

Building this project demonstrated several key principles:

1. **RAG modules compose naturally in DSPy**: retriever + responder is just a `Module` with two sub-components.
2. **Metrics drive everything**: `SemanticF1` told the optimizer what "better" means; a different metric would yield different optimizations.
3. **Baseline first, optimize second**: always measure before optimizing so you can quantify the improvement.
4. **MIPROv2 is powerful for multi-component pipelines**: it optimized the response predictor's instruction and demos end-to-end.
5. **Save and version your optimized programs**: the JSON file is your artifact; treat it like a trained model checkpoint.

---

## Next Up

You've mastered DSPy optimization! In Phase 5, we'll dive deep into **retrieval**, building sophisticated RAG systems with multi-hop reasoning, custom retrievers, and agent-based retrieval patterns.

**[Phase 5: Retrieval & RAG Pipelines →](../../05-retrieval-rag/5.1-retrieval-in-dspy/blog.md)**

---

## Resources

- 📖 [DSPy RAG Tutorial](https://dspy.ai/tutorials/rag/)
- 📖 [MIPROv2 Docs](https://dspy.ai/api/optimizers/MIPROv2/)
- 📖 [DSPy Evaluate Docs](https://dspy.ai/api/evaluation/Evaluate/)
- 📖 [DSPy Optimizers Overview](https://dspy.ai/learn/optimization/optimizers/)
- 💻 [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
