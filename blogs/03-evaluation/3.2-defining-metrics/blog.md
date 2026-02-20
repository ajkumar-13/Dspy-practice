# 3.2: Defining Metrics

## Introduction

Your evaluation set defines *what* to test. Your metric defines *what "good" looks like.* In DSPy, metrics serve double duty: they score your program during evaluation **and** guide the optimizer during automatic prompt tuning. A well-designed metric is the difference between an optimizer that genuinely improves your system and one that chases meaningless numbers.

In this post, we'll cover everything from dead-simple exact match to sophisticated AI-feedback judges, and the critical `trace` parameter that lets your metric behave differently during bootstrapping vs. evaluation.

---

## What You'll Learn

- The anatomy of a DSPy metric function
- Built-in metrics: `answer_exact_match`, `answer_passage_match`, `SemanticF1`, `CompleteAndGrounded`
- Writing custom metrics that check multiple properties
- AI-feedback metrics using DSPy modules as judges
- The `trace` parameter: different behavior for eval vs. bootstrapping
- Advanced: using a full DSPy program as your metric

---

## Prerequisites

- Completed [3.1: Building Evaluation Sets](../3.1-building-eval-sets/blog.md)
- DSPy installed with a configured LM

---

## Anatomy of a DSPy Metric

Every DSPy metric is a Python function with this signature:

```python
def metric(example, pred, trace=None) -> float | int | bool:
    ...
```

Three parameters, always:

- **`example`**: the gold-standard `dspy.Example` with input fields and expected output labels
- **`pred`**: the `dspy.Prediction` your program actually produced
- **`trace`**: controls metric behavior during optimization (more on this below)

The return value can be:
- **`bool`**: pass/fail (useful for bootstrapping: "is this example good enough to use as a demonstration?")
- **`int` or `float`**: a score (useful for evaluation and optimization: "how good is this output?")

Here's the simplest possible metric:

```python
def exact_match(example, pred, trace=None):
    return example.answer.lower().strip() == pred.answer.lower().strip()
```

---

## Built-in Metrics

DSPy ships with several metrics you can use out of the box.

### answer_exact_match

Checks if the predicted answer exactly matches the gold answer (case-insensitive, stripped):

```python
from dspy.evaluate import answer_exact_match, answer_passage_match

# answer_exact_match(example, pred) -> bool
# Compares example.answer vs pred.answer (normalized)
```

### answer_passage_match

Checks if the gold answer appears anywhere in the predicted output, useful for long-form responses where the answer is embedded in a paragraph:

```python
# answer_passage_match(example, pred) -> bool
# Returns True if example.answer is found within pred.answer
```

### SemanticF1

For long-form QA where exact match is too strict, `SemanticF1` computes token-level precision, recall, and F1 between the predicted and gold answers. It captures whether the key ideas are present, even if the wording differs:

```python
from dspy.evaluate import SemanticF1

# Initialize the metric
semantic_f1 = SemanticF1()

# Use it like any metric
score = semantic_f1(example, pred)
print(f"SemanticF1: {score}")
```

`SemanticF1` is an excellent default for open-ended QA tasks. It's more forgiving than exact match but still grounded in the gold answer.

### CompleteAndGrounded

For RAG and retrieval tasks, `CompleteAndGrounded` checks two properties: (1) is the response *complete*, does it cover all the key points in the gold answer? And (2) is it *grounded*, does it stick to information in the retrieved context?

```python
from dspy.evaluate import CompleteAndGrounded

# Great for RAG evaluation
grounded_metric = CompleteAndGrounded()
score = grounded_metric(example, pred)
```

---

## Custom Metrics

Built-in metrics are great starting points, but real projects need custom metrics tailored to their task. The pattern is simple: write a function that checks whatever properties matter to you:

```python
def qa_metric(example, pred, trace=None):
    # Check multiple properties
    answer_correct = example.answer.lower() in pred.answer.lower()
    is_concise = len(pred.answer.split()) < 50
    no_hallucination_markers = "I think" not in pred.answer

    # During evaluation, return a float score
    score = (
        0.6 * answer_correct +
        0.2 * is_concise +
        0.2 * no_hallucination_markers
    )

    # During bootstrapping, be strict: only accept high-quality examples
    if trace is not None:
        return score >= 0.8

    return score
```

Notice the `trace` check. This is the key pattern for metrics that serve both evaluation and optimization. More on this in the dedicated section below.

### Multi-Property Metrics

Often you want to check several independent qualities. Here's a metric for a summarization task:

```python
def summarization_metric(example, pred, trace=None):
    # Check factual overlap
    gold_facts = set(example.answer.lower().split())
    pred_facts = set(pred.summary.lower().split())
    overlap = len(gold_facts & pred_facts) / max(len(gold_facts), 1)

    # Check length constraint
    word_count = len(pred.summary.split())
    length_ok = 20 <= word_count <= 100

    # Check that it doesn't just copy the input
    not_copy = pred.summary.strip() != example.text.strip()

    score = (0.5 * overlap) + (0.25 * length_ok) + (0.25 * not_copy)

    if trace is not None:
        return score >= 0.7

    return score
```

---

## AI-Feedback Metrics

For subjective qualities (fluency, helpfulness, tone, factual accuracy) you can use an LM as a judge. DSPy makes this natural: define a signature, wrap it in a module, and call it inside your metric.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


class AssessQuality(dspy.Signature):
    """Assess the quality of an answer on a scale of 1 to 5."""

    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField(desc="The reference answer")
    predicted_answer: str = dspy.InputField(desc="The answer to assess")
    score: int = dspy.OutputField(desc="Quality score from 1 (terrible) to 5 (perfect)")


assess = dspy.Predict(AssessQuality)


def ai_judge_metric(example, pred, trace=None):
    result = assess(
        question=example.question,
        gold_answer=example.answer,
        predicted_answer=pred.answer,
    )

    # Normalize to 0-1 range
    normalized = (result.score - 1) / 4.0

    if trace is not None:
        return normalized >= 0.75  # Only bootstrap from good examples

    return normalized
```

AI-feedback metrics cost LM calls, so they're slower and more expensive than rule-based metrics. But they can evaluate nuanced properties that no regex can capture.

### Combining Rule-Based and AI Metrics

The best approach is often a hybrid: use fast rule-based checks first, then call the AI judge only when needed:

```python
def hybrid_metric(example, pred, trace=None):
    # Fast check first
    if example.answer.lower().strip() == pred.answer.lower().strip():
        return 1.0  # Perfect match, no need for AI judge

    # Fuzzy check
    if example.answer.lower() in pred.answer.lower():
        return 0.8  # Contains the answer

    # Fall back to AI judge for ambiguous cases
    result = assess(
        question=example.question,
        gold_answer=example.answer,
        predicted_answer=pred.answer,
    )
    return (result.score - 1) / 4.0
```

---

## The trace Parameter

The `trace` parameter is one of DSPy's most important but least understood features. It controls how your metric behaves in different contexts:

- **`trace is None`** → You're in **evaluation or optimization scoring** mode. Return a numeric score (`float` or `int`) for fine-grained ranking.
- **`trace is not None`** → You're in **bootstrapping** mode. The optimizer is deciding whether to keep this example as a demonstration. Return a `bool`: is this example good enough?

```python
def smart_metric(example, pred, trace=None):
    em = example.answer.lower().strip() == pred.answer.lower().strip()
    passage = example.answer.lower() in pred.answer.lower()
    concise = len(pred.answer.split()) < 100

    if trace is not None:
        # Bootstrapping: be strict, require exact match AND conciseness
        return em and concise

    # Evaluation: return a nuanced score
    score = 0.0
    if em:
        score += 0.7
    elif passage:
        score += 0.4
    if concise:
        score += 0.3
    return score
```

Why does this matter? During bootstrapping, the optimizer generates candidate demonstrations by running your program on training examples. The metric acts as a filter: only examples that pass (return `True`) become demonstrations. You want this filter to be **strict** so your demonstrations are high-quality. During evaluation, you want a **nuanced score** so the optimizer can distinguish between "almost right" and "totally wrong."

### Accessing trace for Intermediate Steps

When `trace` is not `None`, it contains the full trace of intermediate module calls. You can use this to validate not just the final output, but the reasoning process:

```python
def trace_aware_metric(example, pred, trace=None):
    final_correct = example.answer.lower() in pred.answer.lower()

    if trace is not None:
        # Check that the chain-of-thought reasoning mentions key concepts
        for step in trace:
            if hasattr(step, 'rationale'):
                mentions_key_concept = "capital" in step.rationale.lower()
                if not mentions_key_concept:
                    return False  # Bad reasoning, don't use as demo
        return final_correct

    return float(final_correct)
```

---

## Advanced: Metrics as DSPy Programs

Here's where it gets powerful: your metric can itself be a full DSPy program, with its own signature, module, and even optimization. This means you can **optimize the judge** alongside the program it evaluates.

```python
class DetailedAssessment(dspy.Signature):
    """Evaluate a QA response on multiple dimensions."""

    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    factual_accuracy: float = dspy.OutputField(desc="0.0-1.0 factual accuracy score")
    completeness: float = dspy.OutputField(desc="0.0-1.0 how complete the answer is")
    clarity: float = dspy.OutputField(desc="0.0-1.0 how clear and well-written")


class QAJudge(dspy.Module):
    def __init__(self):
        self.assess = dspy.ChainOfThought(DetailedAssessment)

    def forward(self, question, gold_answer, predicted_answer):
        return self.assess(
            question=question,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
        )


judge = QAJudge()


def program_metric(example, pred, trace=None):
    result = judge(
        question=example.question,
        gold_answer=example.answer,
        predicted_answer=pred.answer,
    )

    # Weighted combination
    score = (
        0.5 * result.factual_accuracy +
        0.3 * result.completeness +
        0.2 * result.clarity
    )

    if trace is not None:
        return score >= 0.8

    return score
```

Because the judge is a `dspy.Module`, you can save it, load it, and even optimize it with its own evaluation set.

---

## Key Takeaways

- **Every metric is a function**: `def metric(example, pred, trace=None) -> float | bool`. That's the contract.
- **Start with built-ins**: `answer_exact_match` for factoid QA, `SemanticF1` for long-form answers, `CompleteAndGrounded` for RAG.
- **The `trace` parameter is critical**: return `bool` during bootstrapping (strict filter), return `float` during evaluation (nuanced scoring).
- **AI-feedback metrics** bridge the gap for subjective quality, but combine them with fast rule-based checks to control cost.
- **Your metric can be a DSPy program**, fully composable, optimizable, and serializable.

---

## Next Up

You've got data and metrics. Now it's time to put them together and systematically benchmark your DSPy programs. In the next post, we'll cover `dspy.Evaluate`, parallel evaluation, cost tracking, and the iterative development loop.

**[3.3: Running Evaluations →](../3.3-running-evaluations/blog.md)**

---

## Resources

- [DSPy Metrics Documentation](https://dspy.ai/learn/evaluation/metrics/)
- [DSPy Evaluation Overview](https://dspy.ai/learn/evaluation/overview/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [Code examples for this post](code/)
