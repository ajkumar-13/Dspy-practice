"""
Blog 7.P: Project - Model Distillation Pipeline
Run: python 01_distillation.py

End-to-end project demonstrating the complete DSPy distillation workflow:
1. Build a classification system with GPT-4o (teacher)
2. Optimize prompts with MIPROv2
3. Evaluate baseline performance
4. Distill to GPT-4o-mini (student) using BootstrapFinetune
5. Evaluate the distilled model
6. Compare cost, latency, and quality
"""

import time

import dspy
from dotenv import load_dotenv
from dspy.evaluate import Evaluate

load_dotenv()

# -------------------------------------------------------
# Step 1: Build the System
# -------------------------------------------------------

# Configure models
teacher_lm = dspy.LM("openai/gpt-4o")
student_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=teacher_lm)

# Define categories
CATEGORIES = ["politics", "technology", "sports", "health", "finance", "entertainment"]


class NewsClassifier(dspy.Module):
    """Classifies news articles into categories and extracts key entities."""

    def __init__(self):
        self.classify = dspy.ChainOfThought(
            "article -> category: str, entities: list[str], summary: str"
        )

    def forward(self, article):
        result = self.classify(article=article)
        result.category = result.category.strip().lower()
        return result


# Prepare training and test data
all_examples = [
    dspy.Example(
        article="The Senate passed the infrastructure bill with bipartisan support, allocating $1.2 trillion for roads, bridges, and broadband.",
        category="politics",
    ).with_inputs("article"),
    dspy.Example(
        article="Apple unveiled the M3 chip at its latest event, promising 2x faster machine learning performance.",
        category="technology",
    ).with_inputs("article"),
    dspy.Example(
        article="LeBron James scored 40 points to lead the Lakers past the Celtics in overtime.",
        category="sports",
    ).with_inputs("article"),
    dspy.Example(
        article="A new FDA-approved drug shows 60% reduction in migraine frequency in clinical trials.",
        category="health",
    ).with_inputs("article"),
    dspy.Example(
        article="The Federal Reserve held interest rates steady, signaling potential cuts in Q2.",
        category="finance",
    ).with_inputs("article"),
    dspy.Example(
        article="The new Pixar film grossed $200M worldwide in its opening weekend.",
        category="entertainment",
    ).with_inputs("article"),
    dspy.Example(
        article="SpaceX successfully launched its latest Starship prototype from Boca Chica.",
        category="technology",
    ).with_inputs("article"),
    dspy.Example(
        article="The World Health Organization declared the end of the mpox global emergency.",
        category="health",
    ).with_inputs("article"),
    dspy.Example(
        article="Bitcoin surged past $100,000 for the first time amid institutional adoption.",
        category="finance",
    ).with_inputs("article"),
    dspy.Example(
        article="Serena Williams announced her return to competitive tennis at the Australian Open.",
        category="sports",
    ).with_inputs("article"),
    # Add 100-200+ examples for serious distillation
]

trainset = all_examples[:8]
testset = all_examples[8:]


# -------------------------------------------------------
# Step 2: Define Metric and Optimize Prompts
# -------------------------------------------------------

def classification_metric(example, prediction, trace=None):
    """Check category match. Trace mode is stricter for optimization."""
    pred_cat = prediction.category.strip().lower()
    gold_cat = example.category.strip().lower()
    category_correct = pred_cat == gold_cat

    if trace is not None:
        has_entities = hasattr(prediction, "entities") and len(prediction.entities) > 0
        return category_correct and has_entities

    return category_correct


print("=" * 60)
print("Step 2: Optimizing prompts with MIPROv2")
print("=" * 60)

optimizer = dspy.MIPROv2(
    metric=classification_metric,
    auto="light",
    num_threads=4,
)

optimized_teacher = optimizer.compile(
    NewsClassifier(),
    trainset=trainset,
)

optimized_teacher.save("optimized_teacher.json")
print("Optimized teacher saved.")


# -------------------------------------------------------
# Step 3: Baseline Metrics
# -------------------------------------------------------

print("\n" + "=" * 60)
print("Step 3: Evaluating optimized teacher (GPT-4o)")
print("=" * 60)

evaluator = Evaluate(
    devset=testset,
    metric=classification_metric,
    num_threads=4,
    display_progress=True,
)

teacher_score = evaluator(optimized_teacher)
print(f"Teacher accuracy: {teacher_score:.1f}%")

# Measure latency
start = time.time()
for ex in testset:
    optimized_teacher(article=ex.article)
teacher_latency = (time.time() - start) / len(testset)
print(f"Teacher avg latency: {teacher_latency:.2f}s per example")

teacher_cost_info = {
    "model": "gpt-4o",
    "score": teacher_score,
    "latency": teacher_latency,
}


# -------------------------------------------------------
# Step 4: Distill with BootstrapFinetune
# -------------------------------------------------------

print("\n" + "=" * 60)
print("Step 4: Distilling to GPT-4o-mini with BootstrapFinetune")
print("=" * 60)

finetune_optimizer = dspy.BootstrapFinetune(
    metric=classification_metric,
    num_threads=4,
)

# Create student copy with smaller LM
student = NewsClassifier()
student.set_lm(student_lm)

# compile() takes: student, trainset, teacher
dspy.settings.experimental = True
distilled = finetune_optimizer.compile(
    student,
    trainset=trainset,
    teacher=optimized_teacher,
)

distilled.save("distilled_classifier.json")
print("Distilled program saved.")
print("Fine-tuning job submitted. Waiting for completion...")


# -------------------------------------------------------
# Step 5: Evaluate Distilled Model
# -------------------------------------------------------

print("\n" + "=" * 60)
print("Step 5: Evaluating distilled student (fine-tuned GPT-4o-mini)")
print("=" * 60)

student_score = evaluator(distilled)
print(f"Student accuracy: {student_score:.1f}%")

start = time.time()
for ex in testset:
    distilled(article=ex.article)
student_latency = (time.time() - start) / len(testset)
print(f"Student avg latency: {student_latency:.2f}s per example")

student_cost_info = {
    "model": "gpt-4o-mini (fine-tuned)",
    "score": student_score,
    "latency": student_latency,
}


# -------------------------------------------------------
# Step 6: Cost-Quality Analysis
# -------------------------------------------------------

print("\n" + "=" * 60)
print("Step 6: Cost-Quality Analysis")
print("=" * 60)

# Approximate per-token costs (as of early 2025)
COST_PER_1M_INPUT = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "gpt-4o-mini-ft": 0.30,
}
COST_PER_1M_OUTPUT = {
    "gpt-4o": 10.00,
    "gpt-4o-mini": 0.60,
    "gpt-4o-mini-ft": 1.20,
}

# Estimate cost per 1000 classifications (~500 tokens input, ~100 tokens output each)
teacher_cost_1k = (500 * COST_PER_1M_INPUT["gpt-4o"] + 100 * COST_PER_1M_OUTPUT["gpt-4o"]) / 1000
student_cost_1k = (500 * COST_PER_1M_INPUT["gpt-4o-mini-ft"] + 100 * COST_PER_1M_OUTPUT["gpt-4o-mini-ft"]) / 1000

print(f"\n{'Metric':<25} {'Teacher (GPT-4o)':<20} {'Student (4o-mini FT)':<20}")
print("-" * 65)
print(f"{'Accuracy':<25} {teacher_cost_info['score']:<20.1f} {student_cost_info['score']:<20.1f}")
print(f"{'Avg Latency (s)':<25} {teacher_cost_info['latency']:<20.2f} {student_cost_info['latency']:<20.2f}")
print(f"{'Cost per 1K calls':<25} ${teacher_cost_1k:<19.4f} ${student_cost_1k:<19.4f}")
print(f"{'Cost reduction':<25} {'1x (baseline)':<20} {f'{teacher_cost_1k/student_cost_1k:.1f}x cheaper':<20}")

quality_retention = (student_cost_info["score"] / teacher_cost_info["score"]) * 100
print(f"\nQuality retention: {quality_retention:.1f}%")
print(f"Cost reduction: {teacher_cost_1k / student_cost_1k:.1f}x")

if quality_retention >= 95:
    print("\n\u2705 Excellent distillation: student retains >95% of teacher quality!")
elif quality_retention >= 90:
    print("\n\u2705 Good distillation: student retains >90% of teacher quality.")
elif quality_retention >= 80:
    print("\n\u26a0\ufe0f Moderate distillation: consider more training data or a closer model gap.")
else:
    print("\n\u274c Poor distillation: try more training data, a bigger student model, or BetterTogether.")
