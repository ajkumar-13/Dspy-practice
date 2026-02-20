# 2.2: Assertions and Constraints


> [!WARNING]
> **Deprecation Notice (DSPy 2.6+):** `dspy.Assert` and `dspy.Suggest` have been replaced by `dspy.Refine` and `dspy.BestOfN`. The concepts in this post remain valuable for understanding constraint-driven design, but for new code, use `dspy.Refine` (iterative improvement with feedback) or `dspy.BestOfN` (parallel candidate selection). See the [migration guide](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/#migration-from-dspysuggest-and-dspyassert) and [Blog 2.3](../2.3-output-refinement/blog.md) for the current API.

## Introduction

In the last post, you learned how typed predictors and Pydantic models give you *structural* guarantees: the LM returns an `int`, a `list[str]`, or a fully validated `BaseModel`. But structure alone isn't enough. What if the LM returns a valid list... but it's empty? What if the sentiment is `"positive"` but the confidence is `0.1`? What if the generated tweet is 400 characters long?

You need **semantic constraints**, rules that go beyond type checking to enforce the *meaning* of your outputs. DSPy provides two powerful primitives for this: `dspy.Assert` and `dspy.Suggest`. Together, they let you program constraints directly into your modules, and DSPy handles the retry logic automatically.

---

## What You'll Learn

- The difference between `dspy.Assert` (hard constraints) and `dspy.Suggest` (soft guidance)
- How to write assertion conditions and feedback messages
- Common assertion patterns: length checks, format validation, content constraints
- How to use assertions inside `forward()` methods of custom modules
- How assertions interact with DSPy's optimization and trace system
- A practical example: generating constrained tweet-length content

---

## Prerequisites

- Completed [2.1: Typed Predictors](../2.1-typed-predictors/blog.md)
- DSPy installed (`uv add dspy`)
- A configured language model (we'll use `openai/gpt-4o-mini`)

---

## Why Assertions Matter

Typed outputs guarantee that a field is an `int` or a `list[str]`. But they can't enforce:

- "The summary must be under 100 words."
- "The entity list must contain at least one PERSON."
- "The generated tweet must be under 280 characters"
- "The confidence score must be above 0.5 when the category is certain."

These are **semantic constraints**: they depend on the *values* of the output, not just their types. Without assertions, you'd write manual retry loops, check conditions yourself, and re-prompt the LM with feedback. DSPy assertions do all of this for you, declaratively.

---

## dspy.Assert: Hard Constraints

`dspy.Assert` enforces a hard constraint. If the condition is `False`, DSPy raises a `dspy.primitives.assertions.DSPyAssertionError`, which triggers automatic backtracking and retry. The LM receives the error message as feedback and tries again.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


class GenerateTweet(dspy.Signature):
    """Generate an engaging tweet about the given topic."""

    topic: str = dspy.InputField(desc="Topic to tweet about")
    tweet: str = dspy.OutputField(desc="An engaging tweet, must be under 280 characters")


class TweetWriter(dspy.Module):
    def __init__(self):
        self.generate = dspy.Predict(GenerateTweet)

    def forward(self, topic):
        result = self.generate(topic=topic)

        # Hard constraint: tweet MUST be under 280 characters
        dspy.Assert(
            len(result.tweet) <= 280,
            f"Tweet is {len(result.tweet)} chars, must be ≤ 280. Shorten it."
        )

        return dspy.Prediction(tweet=result.tweet)


writer = TweetWriter()
result = writer(topic="the future of AI programming with DSPy")
print(f"Tweet ({len(result.tweet)} chars): {result.tweet}")
```

If the LM generates a tweet over 280 characters, DSPy catches the assertion failure, feeds the error message back to the LM, and retries. After a configurable number of retries (default: 5), it raises a hard error.

The two arguments to `dspy.Assert`:
1. **`condition`**: a boolean expression. `True` means the constraint is satisfied.
2. **`message`**: feedback sent to the LM on failure. Make it specific and actionable. "Shorten it" is more useful than "Invalid output."

> **Tip:** Write assertion messages as if you're giving instructions to a colleague. The LM literally receives this text as feedback, so "Tweet is 350 chars, must be ≤ 280. Remove hashtags or shorten sentences." works better than "Too long."

---

## dspy.Suggest: Soft Guidance

`dspy.Suggest` is the gentler sibling of `dspy.Assert`. Instead of raising an error and forcing a retry, it provides **soft guidance**: the constraint is noted, but execution continues even if the condition fails. The suggestion is logged and used during optimization to improve future outputs.

```python
class TweetWriterWithSuggestions(dspy.Module):
    def __init__(self):
        self.generate = dspy.Predict(GenerateTweet)

    def forward(self, topic):
        result = self.generate(topic=topic)

        # Hard constraint: must fit in a tweet
        dspy.Assert(
            len(result.tweet) <= 280,
            f"Tweet is {len(result.tweet)} chars, must be ≤ 280."
        )

        # Soft guidance: should include a hashtag (nice to have, not required)
        dspy.Suggest(
            "#" in result.tweet,
            "Consider including a relevant hashtag for better engagement."
        )

        # Soft guidance: should include an emoji
        dspy.Suggest(
            any(ord(c) > 127 for c in result.tweet),
            "Adding an emoji can increase engagement."
        )

        return dspy.Prediction(tweet=result.tweet)
```

The key difference:
- **`Assert`**: hard stop. The constraint *must* be satisfied, or the LM retries. Use for non-negotiable requirements.
- **`Suggest`**: soft nudge. The constraint *should* be satisfied, but the program continues either way. Use for quality preferences.

| | `dspy.Assert` | `dspy.Suggest` |
|---|---|---|
| **On failure** | Retries with feedback | Continues execution |
| **Use case** | Non-negotiable rules | Quality preferences |
| **Error on repeated failure** | Yes (raises error) | No (logs warning) |
| **During optimization** | Validates traces | Guides optimizer |

---

## Assertion Patterns

Here are practical patterns you'll use repeatedly:

### Length Constraints

```python
# Minimum and maximum length
dspy.Assert(
    50 <= len(result.summary.split()) <= 100,
    f"Summary is {len(result.summary.split())} words. Must be 50-100 words."
)
```

### Format Validation

```python
# Must be valid email format
import re
dspy.Assert(
    re.match(r"^[\w.+-]+@[\w-]+\.[\w.]+$", result.email) is not None,
    "Output must be a valid email address."
)
```

### Content Checks

```python
# Must mention specific keywords
required_keywords = ["AI", "machine learning"]
dspy.Assert(
    any(kw.lower() in result.summary.lower() for kw in required_keywords),
    f"Summary must mention at least one of: {required_keywords}"
)
```

### Non-Empty Collections

```python
# Entity list must not be empty
dspy.Assert(
    len(result.entities) > 0,
    "Must extract at least one entity from the text."
)
```

### Cross-Field Consistency

```python
# Number of scores must match number of keywords
dspy.Assert(
    len(result.scores) == len(result.keywords),
    f"Got {len(result.scores)} scores for {len(result.keywords)} keywords. "
    "Must have exactly one score per keyword."
)
```

---

## Assertions in Custom Modules

Assertions shine brightest inside custom modules, where you can enforce constraints at each step of a multi-step pipeline:

```python
from pydantic import BaseModel, Field
from typing import Literal


class ReviewAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "mixed"] = Field(
        description="Overall sentiment"
    )
    key_points: list[str] = Field(description="Main points from the review")
    rating: float = Field(description="Predicted rating from 1.0 to 5.0")


class AnalyzeReview(dspy.Signature):
    """Analyze a product review and extract structured insights."""

    review: str = dspy.InputField(desc="Product review text")
    analysis: ReviewAnalysis = dspy.OutputField(desc="Structured analysis")


class ReviewAnalyzer(dspy.Module):
    def __init__(self):
        self.analyze = dspy.Predict(AnalyzeReview)

    def forward(self, review):
        result = self.analyze(review=review)
        analysis = result.analysis

        # Hard: rating must be in valid range
        dspy.Assert(
            1.0 <= analysis.rating <= 5.0,
            f"Rating {analysis.rating} out of range. Must be between 1.0 and 5.0."
        )

        # Hard: must extract at least one key point
        dspy.Assert(
            len(analysis.key_points) >= 1,
            "Must extract at least one key point from the review."
        )

        # Soft: key points should be concise
        dspy.Suggest(
            all(len(kp.split()) <= 20 for kp in analysis.key_points),
            "Key points should be concise, under 20 words each."
        )

        # Soft: sentiment should align with rating
        dspy.Suggest(
            (analysis.sentiment == "positive" and analysis.rating >= 3.5) or
            (analysis.sentiment == "negative" and analysis.rating <= 2.5) or
            (analysis.sentiment == "mixed"),
            "Sentiment and rating seem inconsistent. A positive review should have rating ≥ 3.5."
        )

        return dspy.Prediction(analysis=analysis)


analyzer = ReviewAnalyzer()
result = analyzer(
    review="This laptop is incredible! The battery lasts all day, the screen is "
           "gorgeous, and it handles heavy workloads without breaking a sweat. "
           "My only complaint is the webcam quality; it's mediocre at best."
)

print(f"Sentiment: {result.analysis.sentiment}")
print(f"Rating:    {result.analysis.rating}")
print(f"Key points:")
for point in result.analysis.key_points:
    print(f"  - {point}")
```

Notice how assertions create a **validation layer** between the LM output and your application. The typed Pydantic model ensures structural correctness; assertions ensure semantic correctness.

---

## Assertions and Optimization

Assertions don't just work at runtime; they play a crucial role during optimization. When you optimize a DSPy program, the optimizer generates **traces** (records of inputs, intermediate outputs, and final outputs). Assertions validate these traces:

- **`dspy.Assert` failures** discard the trace. The optimizer treats it as a failed example and doesn't learn from it.
- **`dspy.Suggest` failures** penalize the trace. The optimizer still considers it, but with a lower quality score.

This means your assertions automatically guide the optimizer toward producing prompts and demonstrations that satisfy your constraints. You write the constraint once, and it improves both runtime behavior *and* optimization quality.

```python
# During optimization, assertions act as trace validators
optimizer = dspy.BootstrapRS(metric=my_metric, num_threads=4)
optimized = optimizer.compile(
    ReviewAnalyzer(),
    trainset=train_data,
)
# The optimizer will only keep demonstrations where all Assert conditions passed
# Suggest conditions influence the quality score of kept demonstrations
```

> **Important:** Assertions run inside `forward()`, so they fire on every call: during development, evaluation, *and* optimization. Design your assertions to be fast (no expensive API calls) and deterministic.

---

## Key Takeaways

- **`dspy.Assert(condition, message)`** enforces hard constraints with automatic retry. Use it for non-negotiable rules like length limits, format requirements, and valid ranges.
- **`dspy.Suggest(condition, message)`** provides soft guidance without stopping execution. Use it for quality preferences like "include a hashtag" or "keep it concise."
- **Write actionable messages.** The LM receives your message as feedback on retry. Make it specific and instructive.
- **Combine with typed predictors.** Pydantic validates structure; assertions validate semantics. Together, they create bulletproof output pipelines.
- **Assertions guide optimization.** Failed assertions discard bad traces; suggestions penalize low-quality ones. Your constraints improve the optimizer automatically.
- **Keep assertions fast and deterministic.** They run on every call, including during optimization with many examples.

---

## Next Up

Assertions retry when constraints fail, but what if you want to *systematically* generate multiple candidates and pick the best? Or iteratively refine an output until it meets a quality bar? That's where **BestOfN** and **Refine** come in: output refinement strategies that go beyond simple retry.

**[2.3: Output Refinement: BestOfN and Refine →](../2.3-output-refinement/blog.md)**

---

## Resources

- [DSPy Output Refinement (Replaces Assertions)](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/)
- [DSPy Programming Overview](https://dspy.ai/learn/programming/overview/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [Code examples for this post](code/)
