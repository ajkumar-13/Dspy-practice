"""
3.4: Synthetic Dataset Bootstrapping
=====================================
Generate, filter, and curate training data for DSPy optimization
using a strong teacher LM.
"""

import json
import random
from collections import Counter

import dspy
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# Schema: what each example looks like
# ---------------------------------------------------------------------------

class QAExample(BaseModel):
    question: str = Field(description="A clear, specific question")
    answer: str = Field(description="A concise, factual answer (1-3 sentences)")
    difficulty: str = Field(description="easy, medium, or hard")
    topic: str = Field(description="The subject area of the question")


# ---------------------------------------------------------------------------
# Seed examples (3-5 manual examples to set the quality bar)
# ---------------------------------------------------------------------------

seed_examples = [
    {
        "question": "What is the difference between a list and a tuple in Python?",
        "answer": "Lists are mutable (can be modified after creation) while tuples are immutable. Lists use square brackets [], tuples use parentheses ().",
        "difficulty": "easy",
        "topic": "python-basics",
    },
    {
        "question": "How does garbage collection work in Python?",
        "answer": "Python uses reference counting as its primary mechanism, supplemented by a cyclic garbage collector that detects and collects reference cycles. Objects are freed when their reference count drops to zero.",
        "difficulty": "hard",
        "topic": "python-internals",
    },
    {
        "question": "What is a decorator in Python?",
        "answer": "A decorator is a function that takes another function as input and extends its behavior without modifying its source code. They use the @decorator_name syntax above a function definition.",
        "difficulty": "medium",
        "topic": "python-functions",
    },
]


# ---------------------------------------------------------------------------
# Generator: use CoT with a strong model
# ---------------------------------------------------------------------------

class GenerateExamples(dspy.Signature):
    """Generate diverse, high-quality question-answer pairs for a Python
    programming quiz. Each example should be factually accurate, specific,
    and cover different subtopics. Vary the difficulty levels. Do NOT repeat
    questions similar to the seed examples."""

    seed_examples: str = dspy.InputField(
        desc="Existing examples to match in style and quality"
    )
    topic_focus: str = dspy.InputField(
        desc="Specific topic area to generate questions about"
    )
    num_examples: int = dspy.InputField(
        desc="Number of examples to generate"
    )
    examples_json: str = dspy.OutputField(
        desc="JSON array of generated examples with question, answer, difficulty, topic fields"
    )


generator = dspy.ChainOfThought(GenerateExamples)


def generate_batch(seeds, topic, batch_size=10):
    """Generate a batch of examples for a specific topic."""
    result = generator(
        seed_examples=json.dumps(seeds, indent=2),
        topic_focus=topic,
        num_examples=batch_size,
    )
    try:
        return json.loads(result.examples_json)
    except json.JSONDecodeError:
        print(f"  Failed to parse JSON for topic: {topic}")
        return []


# ---------------------------------------------------------------------------
# Validation: schema + quality checks
# ---------------------------------------------------------------------------

def validate_example(example):
    """Validate against schema and quality criteria. Returns (is_valid, reason)."""
    try:
        QAExample(**example)
    except (ValidationError, TypeError):
        return False, "schema_error"

    if len(example.get("answer", "")) < 20:
        return False, "answer_too_short"
    if len(example.get("question", "")) < 15:
        return False, "question_too_short"
    if example.get("difficulty") not in ("easy", "medium", "hard"):
        return False, "invalid_difficulty"

    bad_phrases = ["for example", "such as", "etc.", "and so on"]
    answer = example.get("answer", "").lower()
    if any(p in answer for p in bad_phrases):
        return False, "vague_answer"

    return True, "valid"


# ---------------------------------------------------------------------------
# Deduplication: remove near-duplicate questions
# ---------------------------------------------------------------------------

def deduplicate(examples, threshold=0.85):
    """Remove near-duplicate examples based on word-overlap similarity."""
    unique, seen = [], []
    for ex in examples:
        q = ex["question"].lower().strip()
        q_words = set(q.split())
        is_dup = False
        for s in seen:
            s_words = set(s.split())
            if not q_words or not s_words:
                continue
            overlap = len(q_words & s_words) / max(len(q_words), len(s_words))
            if overlap > threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(ex)
            seen.append(q)
    return unique


# ---------------------------------------------------------------------------
# Diversity check
# ---------------------------------------------------------------------------

def check_diversity(examples):
    """Report distribution across difficulty and topics."""
    difficulties = Counter(ex["difficulty"] for ex in examples)
    topics = Counter(ex["topic"] for ex in examples)

    print("Difficulty distribution:")
    for diff, count in sorted(difficulties.items()):
        pct = count / len(examples) * 100
        print(f"  {diff}: {count} ({pct:.0f}%)")

    print(f"\nTopic coverage: {len(topics)} topics")
    for topic, count in sorted(topics.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count}")

    if difficulties.get("hard", 0) < len(examples) * 0.15:
        print("\n  WARNING: Less than 15% hard examples.")
    if len(topics) < 5:
        print("\n  WARNING: Low topic diversity.")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline():
    # Configure teacher LM
    teacher_lm = dspy.LM("openai/gpt-4o", temperature=0.9)
    dspy.configure(lm=teacher_lm)

    topics = [
        "python-basics",
        "data-structures",
        "python-functions",
        "object-oriented-programming",
        "error-handling",
        "file-io",
        "python-internals",
        "standard-library",
        "concurrency",
        "testing",
    ]

    # --- Generate ---
    all_generated = []
    for topic in topics:
        batch = generate_batch(seed_examples, topic, batch_size=20)
        all_generated.extend(batch)
        print(f"Generated {len(batch)} examples for {topic}")
    print(f"\nTotal raw: {len(all_generated)}")

    # --- Validate ---
    validated = []
    rejections = {}
    for ex in all_generated:
        ok, reason = validate_example(ex)
        if ok:
            validated.append(ex)
        else:
            rejections[reason] = rejections.get(reason, 0) + 1
    print(f"Validated: {len(validated)} / {len(all_generated)}")
    print(f"Rejections: {rejections}")

    # --- Deduplicate ---
    deduped = deduplicate(validated)
    print(f"After dedup: {len(deduped)}")

    # --- Diversity ---
    check_diversity(deduped)

    # --- Split ---
    random.seed(42)
    random.shuffle(deduped)
    dspy_examples = [
        dspy.Example(question=ex["question"], answer=ex["answer"]).with_inputs("question")
        for ex in deduped
    ]
    n = len(dspy_examples)
    train_end = int(n * 0.2)
    dev_end = int(n * 0.8)
    trainset = dspy_examples[:train_end]
    devset = dspy_examples[train_end:dev_end]
    testset = dspy_examples[dev_end:]
    print(f"\nTrain: {len(trainset)} | Dev: {len(devset)} | Test: {len(testset)}")

    # --- Optimize ---
    student_lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=student_lm)

    qa = dspy.ChainOfThought("question -> answer")

    def metric(example, prediction, trace=None):
        return dspy.evaluate.SemanticF1()(example, prediction, trace)

    from dspy.evaluate import Evaluate

    evaluator = Evaluate(devset=devset, metric=metric, num_threads=8)
    baseline = evaluator(qa)
    print(f"Baseline: {baseline}")

    optimized = dspy.MIPROv2(metric=metric, auto="medium").compile(
        qa, trainset=trainset
    )
    opt_score = evaluator(optimized)
    print(f"Optimized: {opt_score}")

    test_eval = Evaluate(devset=testset, metric=metric, num_threads=8)
    test_score = test_eval(optimized)
    print(f"Test: {test_score}")


if __name__ == "__main__":
    run_pipeline()
