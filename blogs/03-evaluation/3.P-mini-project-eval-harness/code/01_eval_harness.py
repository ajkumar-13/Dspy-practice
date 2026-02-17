"""
Blog 3.P: Mini-Project: Building an Evaluation Harness
A reusable template for evaluating multiple DSPy programs across multiple metrics.
"""

import dspy
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from datasets import load_dataset
from dspy.evaluate import answer_exact_match, answer_passage_match, SemanticF1

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# ── Step 1: Load the Dataset ─────────────────────────────────────────────────

def load_hotpotqa(num_examples=50):
    """Load a subset of HotPotQA for evaluation."""
    raw = load_dataset("hotpot_qa", "fullwiki", split=f"validation[:{num_examples}]")

    examples = [
        dspy.Example(
            question=row["question"],
            answer=row["answer"],
        ).with_inputs("question")
        for row in raw
    ]

    print(f"Loaded {len(examples)} examples")
    print(f"Sample: {examples[0].question} → {examples[0].answer}")
    return examples


# ── Step 2: Define Multiple Metrics ───────────────────────────────────────────

def metric_exact_match(example, pred, trace=None):
    return answer_exact_match(example, pred)


def metric_passage_match(example, pred, trace=None):
    return answer_passage_match(example, pred)


semantic_f1 = SemanticF1()


def metric_semantic_f1(example, pred, trace=None):
    return semantic_f1(example, pred)


METRICS = {
    "exact_match": metric_exact_match,
    "passage_match": metric_passage_match,
    "semantic_f1": metric_semantic_f1,
}


# ── Step 3: Build Program Variants ───────────────────────────────────────────

class AnswerQuestion(dspy.Signature):
    """Answer the question concisely and accurately."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="A concise, factual answer")


predict_program = dspy.Predict(AnswerQuestion)
cot_program = dspy.ChainOfThought(AnswerQuestion)


class DecomposeAndAnswer(dspy.Module):
    def __init__(self):
        self.decompose = dspy.ChainOfThought(
            "question -> sub_questions: list[str]"
        )
        self.answer = dspy.ChainOfThought(AnswerQuestion)

    def forward(self, question):
        decomposed = self.decompose(question=question)
        sub_qs = ", ".join(decomposed.sub_questions)
        enriched_question = f"{question} (Consider: {sub_qs})"
        result = self.answer(question=enriched_question)
        return dspy.Prediction(answer=result.answer)


decompose_program = DecomposeAndAnswer()

PROGRAMS = {
    "Predict": predict_program,
    "ChainOfThought": cot_program,
    "DecomposeAndAnswer": decompose_program,
}


# ── Step 4: Run Evaluations ──────────────────────────────────────────────────

def run_evaluation_grid(programs, metrics, devset, num_threads=16):
    """Evaluate all programs against all metrics. Returns a results dict."""
    results = {}

    evaluator = dspy.Evaluate(
        devset=devset,
        num_threads=num_threads,
        display_progress=True,
        display_table=0,
    )

    for prog_name, program in programs.items():
        results[prog_name] = {}
        for metric_name, metric_fn in metrics.items():
            print(f"\nEvaluating {prog_name} with {metric_name}...")
            start_time = time.time()

            score = evaluator(program, metric=metric_fn)

            elapsed = time.time() - start_time
            results[prog_name][metric_name] = {
                "score": round(score, 4),
                "time_seconds": round(elapsed, 2),
            }
            print(f"  Score: {score:.2%} ({elapsed:.1f}s)")

    return results


# ── Step 5: Compare and Analyze ──────────────────────────────────────────────

def print_comparison_table(results):
    """Print a formatted comparison table of evaluation results."""
    metric_names = list(next(iter(results.values())).keys())

    header = f"{'Program':<22}"
    for m in metric_names:
        header += f"  {m:>15}"
    header += f"  {'Avg':>8}"

    print("\n" + "=" * len(header))
    print("EVALUATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for prog_name, metrics in results.items():
        row = f"{prog_name:<22}"
        scores = []
        for m in metric_names:
            score = metrics[m]["score"]
            scores.append(score)
            row += f"  {score:>14.2%}"
        avg = sum(scores) / len(scores)
        row += f"  {avg:>7.2%}"
        print(row)

    print("=" * len(header))

    print("\nBest per metric:")
    for m in metric_names:
        best_prog = max(results, key=lambda p: results[p][m]["score"])
        best_score = results[best_prog][m]["score"]
        print(f"  {m}: {best_prog} ({best_score:.2%})")


# ── Step 6: Save Results ─────────────────────────────────────────────────────

def save_results(results, filepath="eval_results.json"):
    """Append evaluation results to a JSON file for historical tracking."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "num_examples": len(devset),
        "model": "openai/gpt-4o-mini",
        "results": results,
    }

    try:
        with open(filepath, "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.append(record)

    with open(filepath, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to {filepath}")
    print(f"Total runs tracked: {len(history)}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    devset = load_hotpotqa(50)
    results = run_evaluation_grid(PROGRAMS, METRICS, devset)
    print_comparison_table(results)
    save_results(results, "eval_results.json")
