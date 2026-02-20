"""
11.6: Continuous Evaluation - CI/CD for LLM Applications
=========================================================
Build a pytest-compatible evaluation suite that gates PRs on quality metrics.
"""

import json
import random
from datetime import datetime
from pathlib import Path

import dspy
from dspy.evaluate import Evaluate, SemanticF1

# ---------------------------------------------------------------------------
# Evaluation fixtures (use via conftest.py in practice)
# ---------------------------------------------------------------------------


def load_devset(path="eval/datasets/qa_dev.json"):
    """Load the QA evaluation dataset."""
    with open(path) as f:
        data = json.load(f)
    return [
        dspy.Example(question=ex["question"], answer=ex["answer"]).with_inputs("question")
        for ex in data
    ]


def load_baseline(path="eval/baselines/qa_scores.json"):
    """Load baseline scores for comparison."""
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {"semantic_f1": 0, "empty_rate": 100}


# ---------------------------------------------------------------------------
# Stratified sampling for cost-aware CI
# ---------------------------------------------------------------------------


def stratified_sample(devset, n=50, seed=42):
    """Take a representative sample for CI evaluation."""
    random.seed(seed)
    if len(devset) <= n:
        return devset

    difficulties = {}
    for ex in devset:
        diff = getattr(ex, "difficulty", "unknown")
        difficulties.setdefault(diff, []).append(ex)

    sample = []
    for diff, examples in difficulties.items():
        k = max(1, int(n * len(examples) / len(devset)))
        sample.extend(random.sample(examples, min(k, len(examples))))

    remaining = [ex for ex in devset if ex not in sample]
    if len(sample) < n:
        sample.extend(random.sample(remaining, min(n - len(sample), len(remaining))))

    return sample[:n]


# ---------------------------------------------------------------------------
# Baseline management
# ---------------------------------------------------------------------------

BASELINE_PATH = Path("eval/baselines/qa_scores.json")


def update_baseline(scores: dict):
    """Update the baseline scores after a successful optimization."""
    baseline = {
        "semantic_f1": scores["semantic_f1"],
        "empty_rate": scores.get("empty_rate", 0),
        "updated_at": datetime.now().isoformat(),
        "model": scores.get("model", "unknown"),
        "sample_size": scores.get("sample_size", 0),
    }
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
    print(f"Baseline updated: {baseline}")


# ---------------------------------------------------------------------------
# Evaluation runner (standalone, outside pytest)
# ---------------------------------------------------------------------------


def run_eval(sample_size=50):
    """Run the evaluation suite and report results."""
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    devset = load_devset()
    sample = stratified_sample(devset, n=sample_size)
    baseline = load_baseline()

    qa = dspy.ChainOfThought("question -> answer")

    def metric(ex, pred, trace=None):
        return SemanticF1()(ex, pred, trace)

    evaluator = Evaluate(devset=sample, metric=metric, num_threads=4, display_progress=True)

    result = evaluator(qa)
    score = result.score

    # Check for empty answers
    empty_count = 0
    for ex in sample[:20]:
        result = qa(question=ex.question)
        if not result.answer or len(result.answer.strip()) < 5:
            empty_count += 1
    empty_rate = empty_count / 20 * 100

    # Report
    baseline_score = baseline.get("semantic_f1", 0)
    regression = baseline_score - score

    print(f"\n{'='*50}")
    print("Evaluation Results")
    print(f"{'='*50}")
    print(f"SemanticF1:        {score:.1f}%")
    print(f"Baseline:          {baseline_score:.1f}%")
    print(f"Regression:        {regression:+.1f}%")
    print(f"Empty answer rate: {empty_rate:.0f}%")
    print(f"Sample size:       {len(sample)}")
    print(f"{'='*50}")

    if regression > 5:
        print("FAIL: Quality regression exceeds 5% threshold")
        return False
    if empty_rate > 10:
        print("FAIL: Empty answer rate exceeds 10% threshold")
        return False

    print("PASS: All quality checks passed")
    return True


if __name__ == "__main__":
    success = run_eval(sample_size=50)
    exit(0 if success else 1)
