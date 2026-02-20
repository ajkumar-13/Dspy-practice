# 2.3: Output Refinement: BestOfN and Refine

## Introduction

You've learned that typed predictors guarantee structure, and assertions enforce semantic constraints with automatic retry. But sometimes, a single LM call, even with retries, isn't enough to get the quality you need. The model might produce a *valid* output that passes all assertions but still isn't *great*.

This is where **output refinement strategies** come in. DSPy provides two complementary approaches: `dspy.BestOfN` for parallel candidate sampling, and `dspy.Refine` for iterative improvement. Both wrap your existing modules and transparently improve output quality without any changes to your core logic.

---

## What You'll Learn

- How `dspy.BestOfN` generates N candidates and selects the best based on a reward function
- How `dspy.Refine` iteratively improves output based on feedback until a quality threshold is met
- When to use BestOfN vs Refine, and the cost tradeoffs of each
- How to combine refinement strategies with assertions for robust, high-quality outputs
- A practical example: generating high-quality summaries

---

## Prerequisites

- Completed [2.2: Assertions and Constraints](../2.2-assertions/blog.md)
- DSPy installed (`uv add dspy`)
- A configured language model (we'll use `openai/gpt-4o-mini`)

---

## dspy.BestOfN: Parallel Candidate Selection

The idea behind `BestOfN` is simple: instead of taking the first output the LM produces, generate N candidates and pick the best one according to a reward function.

Think of it like asking N different people the same question and choosing the answer you like most. Except all N "people" are the same LM with different random seeds.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


class Summarize(dspy.Signature):
    """Write a concise, informative summary of the given text."""

    text: str = dspy.InputField(desc="Text to summarize")
    summary: str = dspy.OutputField(desc="A concise summary in 2-3 sentences")


class Summarizer(dspy.Module):
    def __init__(self):
        self.summarize = dspy.Predict(Summarize)

    def forward(self, text):
        result = self.summarize(text=text)
        return dspy.Prediction(summary=result.summary)


# Define a reward function
def summary_quality(args, pred):
    """Score summaries: prefer concise, complete summaries."""
    words = pred.summary.split()
    # Penalize very short or very long summaries
    if len(words) < 10:
        return 0.0
    if len(words) > 75:
        return 0.3
    # Use a simple heuristic: reward moderate length
    return min(1.0, len(words) / 40)


# Wrap the module with BestOfN
best_summarizer = dspy.BestOfN(
    module=Summarizer(),
    N=5,
    reward_fn=summary_quality,
    threshold=0.5,
)

article = (
    "Scientists at CERN announced today that they have observed a new particle "
    "in the Large Hadron Collider that doesn't fit neatly into the Standard Model "
    "of particle physics. The particle, tentatively named Xi-prime, was detected "
    "during high-energy proton collisions at unprecedented energy levels. Lead "
    "researcher Dr. Maria Santos called the discovery 'potentially revolutionary' "
    "and noted that it could point toward new physics beyond current theories. "
    "The finding will need to be independently verified by other research groups."
)

result = best_summarizer(text=article)
print(f"Best summary: {result.summary}")
```

### How BestOfN Works

1. Your module is called N times (here, 5 times), each with a different `rollout_id` to ensure diverse outputs
2. Each candidate output is scored using your reward function
3. The first candidate that meets the `threshold` is returned, or the highest-scoring one if none meets it

The reward function signature is `(args: dict, pred: dspy.Prediction) -> float`. The `args` dict contains the keyword arguments passed to the module, and `pred` is the candidate output.

### Using a DSPy Module as Metric

For more sophisticated scoring, you can use another DSPy module as your reward function, essentially having one LM judge another's output:

```python
class JudgeSummary(dspy.Signature):
    """Rate the quality of a summary on a scale of 1-10."""

    original_text: str = dspy.InputField()
    summary: str = dspy.InputField()
    score: float = dspy.OutputField(desc="Quality score from 1.0 to 10.0")
    reasoning: str = dspy.OutputField(desc="Why this score was given")


judge = dspy.ChainOfThought(JudgeSummary)


def llm_judge_reward(args, pred):
    """Use an LM to judge summary quality."""
    result = judge(
        original_text=args["text"],
        summary=pred.summary,
    )
    return result.score / 10.0  # Normalize to 0-1


best_summarizer = dspy.BestOfN(
    module=Summarizer(),
    N=3,
    reward_fn=llm_judge_reward,
    threshold=0.7,
)
```

> **Cost Alert:** BestOfN multiplies your LM costs by N. With an LM judge metric, each candidate requires an *additional* LM call for scoring, so N=5 with a judge metric means 10 LM calls per invocation. Choose N carefully based on your quality/cost tradeoff.

---

## dspy.Refine: Iterative Improvement

While `BestOfN` generates candidates in parallel and picks the best, `dspy.Refine` takes a different approach: it generates one output, checks it against a reward function, and if it's not good enough, **iteratively improves it** by automatically generating feedback and providing it as hints to the module.

Think of it like a writing process: draft, review, revise, review again, until the quality is acceptable or you've hit the maximum number of attempts (N).

```python
class GenerateReport(dspy.Signature):
    """Generate a detailed analytical report on the given topic."""

    topic: str = dspy.InputField(desc="Topic to analyze")
    report: str = dspy.OutputField(
        desc="A detailed report with introduction, analysis, and conclusion"
    )


class ReportWriter(dspy.Module):
    def __init__(self):
        self.write = dspy.Predict(GenerateReport)

    def forward(self, topic):
        result = self.write(topic=topic)
        return dspy.Prediction(report=result.report)


def report_quality(args, pred):
    """Score reports based on structure and completeness."""
    report = pred.report.lower()
    score = 0.0

    # Check for key structural elements
    if "introduction" in report or report.startswith("the") or report.startswith("in"):
        score += 0.25
    if "however" in report or "although" in report or "on the other hand" in report:
        score += 0.25  # Shows balanced analysis
    if "conclusion" in report or "in summary" in report or "overall" in report:
        score += 0.25
    if len(report.split()) >= 100:
        score += 0.25  # Sufficient detail

    return score


# Wrap with Refine
refined_writer = dspy.Refine(
    module=ReportWriter(),
    N=3,               # Maximum 3 refinement attempts
    reward_fn=report_quality,
    threshold=0.75,    # Keep refining until score >= 0.75
)

result = refined_writer(topic="The impact of large language models on software engineering")
print(result.report)
```

### How Refine Works

1. Your module runs and produces an initial output
2. The reward function scores the output
3. If the score is below the `threshold`, DSPy automatically generates detailed feedback about the module's performance using the code in the reward function, and uses this feedback as hints for the next attempt
4. Steps 2-3 repeat until the score meets the threshold or N attempts have been made
5. The best-scoring output across all iterations is returned

The key advantage over `BestOfN`: Refine is *iterative*. Each attempt builds on the feedback from the previous one, so the model learns from its mistakes within the same conversation. `BestOfN` generates independent candidates with different rollout IDs; they don't learn from each other.

---

## Choosing Between BestOfN and Refine

Both strategies improve output quality, but they work differently and suit different scenarios:

| | `dspy.BestOfN` | `dspy.Refine` |
|---|---|---|
| **Strategy** | Parallel sampling (different rollout IDs) | Iterative improvement with auto-feedback |
| **LM calls** | Up to N | Up to N |
| **Diversity** | High (independent samples) | Low (each builds on prior feedback) |
| **Best for** | Creative tasks, diverse outputs | Structured tasks, convergent quality |
| **Cost** | Predictable (up to N calls) | Variable (1 call if first try is good) |
| **Feedback** | None between candidates | Yes, auto-generated from reward_fn code |

### Rules of Thumb

- **Use BestOfN** when you want diversity and your metric clearly distinguishes good from bad. Great for creative writing, alternative phrasings, or when any of several approaches could work.
- **Use Refine** when the output needs to progressively improve toward a specific quality bar. Great for structured reports, formatted outputs, or when the metric provides meaningful directional feedback.
- **Use both together** for maximum quality on critical outputs, but watch the cost.

```python
# Combining: Refine each BestOfN candidate? Or BestOfN with Refine inside?
# The simpler approach: use Refine as the primary strategy, with assertions as guardrails.

class RobustSummarizer(dspy.Module):
    def __init__(self):
        self.summarize = dspy.Predict(Summarize)

    def forward(self, text):
        result = self.summarize(text=text)

        # Assertions catch hard failures
        dspy.Assert(
            len(result.summary.split()) >= 10,
            "Summary too short. Must be at least 10 words."
        )
        dspy.Assert(
            len(result.summary.split()) <= 75,
            "Summary too long. Must be under 75 words."
        )

        return dspy.Prediction(summary=result.summary)


# Then wrap with Refine for quality improvement
refined = dspy.Refine(
    module=RobustSummarizer(),
    N=3,
    reward_fn=summary_quality,
    threshold=0.7,
)
```

---

## Combining with Assertions

Refinement strategies and assertions complement each other perfectly:

- **Assertions** handle *correctness*: hard constraints that must be satisfied (length limits, format requirements, valid ranges)
- **Refinement** handles *quality*: making already-correct outputs better

The execution order matters: assertions fire inside `forward()` on every call, including each BestOfN candidate and each Refine iteration. This means:

1. Each candidate/iteration first passes through your assertion checks
2. If assertions fail, the retry mechanism kicks in before the output even reaches the refinement metric
3. The refinement metric only scores outputs that have already passed all assertions

This creates a powerful quality pipeline:

```
LM Output → Type Validation → Assertions → Refinement Metric → Final Output
```

Each layer catches different problems: types catch structural issues, assertions catch semantic violations, and refinement optimizes for overall quality.

```python
class ConstrainedSummarizer(dspy.Module):
    def __init__(self):
        self.summarize = dspy.Predict(Summarize)

    def forward(self, text):
        result = self.summarize(text=text)

        # Layer 1: Assertions ensure correctness
        dspy.Assert(
            len(result.summary) > 0,
            "Summary cannot be empty."
        )
        dspy.Assert(
            not result.summary.startswith("Sure") and not result.summary.startswith("Here"),
            "Summary should not start with filler phrases like 'Sure' or 'Here is'."
        )

        # Layer 2: Suggestions improve quality
        dspy.Suggest(
            not result.summary.endswith("..."),
            "Summary should be complete, not trailing off with ellipsis."
        )

        return dspy.Prediction(summary=result.summary)


# Layer 3: BestOfN selects the highest quality
best = dspy.BestOfN(
    module=ConstrainedSummarizer(),
    N=3,
    reward_fn=summary_quality,
    threshold=0.5,
)

result = best(text=article)
print(result.summary)
```

---

## Key Takeaways

- **`dspy.BestOfN`** generates N candidates using different rollout IDs and returns the first that meets the threshold, or the highest-scoring one. Best for diverse, creative tasks.
- **`dspy.Refine`** iteratively improves output with auto-generated feedback until a quality threshold is met. Up to N attempts. Best for structured, convergent tasks.
- **Cost vs quality** is the fundamental tradeoff. BestOfN has predictable cost; Refine is cheaper when the first attempt is good.
- **Combine with assertions** for a layered quality pipeline: types → assertions → refinement. Each layer catches different failure modes.
- **Reward function design matters.** Both strategies are only as good as the reward function that guides them. Invest time in writing reward functions that genuinely distinguish quality.

---

## Next Up

You've now mastered the full Phase 2 toolkit: typed predictors for structure, assertions for constraints, and refinement strategies for quality. Time to put it all together in a real project. You'll build a **structured entity extractor** that uses every technique from this phase.

**[2.P: Mini-Project: Entity Extractor →](../2.P-mini-project-entity-extractor/blog.md)**

---

## Resources

- [DSPy Output Refinement Tutorial](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/)
- [DSPy Assertions (Deprecated, see BestOfN/Refine)](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/#migration-from-dspysuggest-and-dspyassert)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [Code examples for this post](code/)
