# 1.P: Mini-Project - Building a Text Classifier

## Introduction

You've spent four posts learning the foundations: signatures declare *what*, modules define *how*, and custom modules compose them into programs. Now it's time to put it all together.

In this mini-project, you'll build a **text classifier** that categorizes news headlines into topics using a class-based signature with typed output, comparing `Predict` vs `ChainOfThought`, evaluating with a real metric, and saving the final program. No hand-written prompts. No regex parsers. Just clean, composable DSPy.

---

## Project Overview

We'll build a classifier that:

1. Takes a news headline as input
2. Classifies it into one of five categories: **Politics**, **Sports**, **Technology**, **Business**, or **Entertainment**
3. Provides a confidence score
4. Gets evaluated on a small labeled dataset

Along the way, you'll use: class-based signatures with `Literal` types, `dspy.Example` for test data, `dspy.Evaluate` for systematic evaluation, and `save()`/`load()` for persistence.

---

## Step 1: Define the Signature

We want a typed, constrained output, the model should pick from a fixed set of categories, not invent its own. A class-based signature with `Literal` is the right tool:

```python
import dspy
from dotenv import load_dotenv
from typing import Literal

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


class ClassifyHeadline(dspy.Signature):
    """Classify the news headline into exactly one category."""

    headline: str = dspy.InputField(desc="A news headline to classify")
    category: Literal["Politics", "Sports", "Technology", "Business", "Entertainment"] = (
        dspy.OutputField(desc="The topic category of the headline")
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score between 0.0 and 1.0"
    )
```

Three things to notice:

1. **`Literal` constrains the output** to exactly five valid categories. DSPy tells the LM to pick from this list, no free-form text.
2. **`confidence: float`** gives us an automatic numeric output, parsed and validated by DSPy.
3. **The docstring** ("Classify the news headline...") becomes the task instruction in the generated prompt.

---

## Step 2: Build the Module

We'll wrap the signature in a custom module so it's optimizable and serializable:

```python
class HeadlineClassifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict(ClassifyHeadline)

    def forward(self, headline):
        result = self.classify(headline=headline)
        return dspy.Prediction(
            category=result.category,
            confidence=result.confidence,
        )
```

Simple and focused, one predictor, one job. We start with `Predict` and will compare it against `ChainOfThought` later.

Let's test it on a single headline:

```python
classifier = HeadlineClassifier()

result = classifier(headline="SpaceX launches 40 Starlink satellites into orbit")
print(f"Category: {result.category}")
print(f"Confidence: {result.confidence}")
# Category: Technology
# Confidence: 0.95
```

---

## Step 3: Create Test Data

To evaluate systematically, we need labeled examples. DSPy uses `dspy.Example`, a lightweight container for input/output pairs. Each example has an `inputs()` method that marks which fields are inputs (passed to the module) vs. labels (used for evaluation).

```python
test_set = [
    dspy.Example(
        headline="Senate passes bipartisan infrastructure bill",
        category="Politics",
    ).with_inputs("headline"),
    dspy.Example(
        headline="Lakers defeat Celtics in overtime thriller",
        category="Sports",
    ).with_inputs("headline"),
    dspy.Example(
        headline="Apple unveils new M4 chip with neural engine",
        category="Technology",
    ).with_inputs("headline"),
    dspy.Example(
        headline="Fed raises interest rates by 0.25 percentage points",
        category="Business",
    ).with_inputs("headline"),
    dspy.Example(
        headline="Marvel announces three new films for 2027",
        category="Entertainment",
    ).with_inputs("headline"),
    dspy.Example(
        headline="President signs executive order on climate policy",
        category="Politics",
    ).with_inputs("headline"),
    dspy.Example(
        headline="Serena Williams announces return to Grand Slam tennis",
        category="Sports",
    ).with_inputs("headline"),
    dspy.Example(
        headline="Google DeepMind releases open-source AI model",
        category="Technology",
    ).with_inputs("headline"),
    dspy.Example(
        headline="Oil prices surge amid Middle East tensions",
        category="Business",
    ).with_inputs("headline"),
    dspy.Example(
        headline="Taylor Swift breaks streaming record with new album",
        category="Entertainment",
    ).with_inputs("headline"),
]
```

Ten examples, two per category. In a real project you'd have hundreds, but this is enough to demonstrate the evaluation workflow.

> **Tip:** `.with_inputs("headline")` tells DSPy that `headline` is the input field. Everything else (`category`) is a label used only for evaluation, it won't be passed to the module during inference.

---

## Step 4: Define the Metric

A metric function takes an `example` (the ground truth) and a `prediction` (what the module produced), and returns a score. For classification, exact match on the category is the natural choice:

```python
def classification_accuracy(example, prediction, trace=None):
    """Return 1.0 if the predicted category matches the expected category."""
    return prediction.category.strip() == example.category.strip()
```

The `trace` parameter is used by optimizers during compilation, you can safely ignore it for now, but it must be in the signature.

Want a softer metric that also considers confidence? You can weight accuracy by confidence:

```python
def weighted_accuracy(example, prediction, trace=None):
    """Accuracy weighted by confidence. Rewards correct AND confident predictions."""
    is_correct = prediction.category.strip() == example.category.strip()
    if is_correct:
        return prediction.confidence  # Higher confidence = higher score
    return 0.0
```

We'll stick with `classification_accuracy` for our primary evaluation, it's clean and interpretable.

---

## Step 5: Evaluate

DSPy's `Evaluate` class runs your module against a dataset and computes the metric across all examples:

```python
classifier = HeadlineClassifier()

evaluate = dspy.Evaluate(
    devset=test_set,
    metric=classification_accuracy,
    num_threads=4,
    display_progress=True,
    display_table=5,  # Show the first 5 results in a table
)

score = evaluate(classifier)
print(f"\nAccuracy: {score}%")
```

Typical output:

```
Average Metric: 9.00 / 10 (90.0%)

Accuracy: 90.0%
```

`dspy.Evaluate` runs each test example through your module, compares the prediction to the label using your metric, and reports the aggregate score. The `display_table` parameter shows a preview of individual results, great for spotting which examples fail.

---

## Step 6: Compare Strategies

One of DSPy's superpowers is swapping the execution strategy without changing the signature. Let's build a `ChainOfThought` version and compare:

```python
class ReasonedClassifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(ClassifyHeadline)

    def forward(self, headline):
        result = self.classify(headline=headline)
        return dspy.Prediction(
            category=result.category,
            confidence=result.confidence,
        )
```

The only change: `dspy.Predict` → `dspy.ChainOfThought`. The signature, metric, and evaluation code stay identical.

```python
# Evaluate the Predict-based classifier
predict_classifier = HeadlineClassifier()
evaluate_predict = dspy.Evaluate(
    devset=test_set,
    metric=classification_accuracy,
    num_threads=4,
    display_progress=True,
)
predict_score = evaluate_predict(predict_classifier)
print(f"Predict accuracy:        {predict_score}%")

# Evaluate the ChainOfThought-based classifier
cot_classifier = ReasonedClassifier()
evaluate_cot = dspy.Evaluate(
    devset=test_set,
    metric=classification_accuracy,
    num_threads=4,
    display_progress=True,
)
cot_score = evaluate_cot(cot_classifier)
print(f"ChainOfThought accuracy: {cot_score}%")
```

Typical results:

```
Predict accuracy:        90.0%
ChainOfThought accuracy: 100.0%
```

The `ChainOfThought` version often matches or beats `Predict`, the reasoning step helps the model disambiguate tricky headlines like "Oil prices surge amid Middle East tensions" (Business? Politics?). The tradeoff is slightly higher cost and latency per call, since the model generates a `reasoning` field before the classification.

You can also inspect the reasoning for any individual prediction:

```python
result = cot_classifier(headline="Oil prices surge amid Middle East tensions")
print(f"Reasoning: {result.reasoning}")
print(f"Category: {result.category}")
# Reasoning: The headline discusses oil prices, which is a financial/economic topic...
# Category: Business
```

---

## Step 7: Save Your Program

Once you're happy with a classifier, save it so you can reload it later, in a different script, a notebook, or a production service:

```python
# Save the best classifier
cot_classifier.save("headline_classifier.json")
print("Program saved!")
```

To reload:

```python
# In a new script or session
import dspy
from dotenv import load_dotenv

load_dotenv()
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Recreate the same class structure
loaded_classifier = ReasonedClassifier()
loaded_classifier.load("headline_classifier.json")

# Use it immediately
result = loaded_classifier(headline="NASA rover discovers water ice on Mars")
print(f"Category: {result.category}")
# Category: Technology
```

Remember: the class definition (`ReasonedClassifier`, `ClassifyHeadline`) must exist in your code when loading. The save file stores the *learned state* like demos, instructions and not the Python class itself.

---

## What We Learned

This mini-project exercised every core concept from Phase 1:

- **Signatures**: `ClassifyHeadline` declared the contract with a `Literal` type for constrained output and `float` for confidence scoring
- **Modules**: We used `Predict` and `ChainOfThought` interchangeably on the same signature
- **Custom Modules**: `HeadlineClassifier` and `ReasonedClassifier` wrapped predictors in composable, serializable classes
- **Evaluation**: `dspy.Example` held our test data, a metric function defined "correct," and `dspy.Evaluate` ran the benchmark
- **Saving/Loading**: `save()` and `load()` persisted the program state

The key insight: **at no point did we write a prompt.** We declared a signature, chose a module strategy, and let DSPy handle the rest. When we wanted to improve accuracy, we swapped the module and not the prompt.

---

## Next Up

Phase 1 is complete. You can define contracts, choose execution strategies, compose multi-step pipelines, and evaluate them systematically. In Phase 2, we'll take structured outputs to the next level, Pydantic integration, assertions that enforce constraints, and output refinement loops.

**[2.1: Typed Predictors & Pydantic Integration →](../../02-structured-outputs/2.1-typed-predictors/blog.md)**

---

## Resources

- [DSPy Evaluation Documentation](https://dspy.ai/learn/evaluation/overview/)
- [DSPy Modules Documentation](https://dspy.ai/learn/programming/modules/)
- [DSPy Signatures Documentation](https://dspy.ai/learn/programming/signatures/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [Code examples for this post](code/)
