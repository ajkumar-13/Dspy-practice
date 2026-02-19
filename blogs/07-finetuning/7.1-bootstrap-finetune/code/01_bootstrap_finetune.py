"""
Blog 7.1: BootstrapFinetune - Prompt to Weights
Run: python 01_bootstrap_finetune.py

Demonstrates teacher-student distillation using dspy.BootstrapFinetune.
The teacher (GPT-4o) generates traces, which are used to fine-tune
a student (GPT-4o-mini).
"""

import dspy
from dspy.evaluate import Evaluate
from dotenv import load_dotenv

load_dotenv()

# Step 1: Configure the teacher (large model)
teacher_lm = dspy.LM("openai/gpt-4o")
student_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=teacher_lm)

# Step 2: Define your program
class Classifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(
            "text -> category: str, confidence: float"
        )

    def forward(self, text):
        return self.classify(text=text)

# Step 3: Prepare training data
trainset = [
    dspy.Example(text="The stock market rallied on strong earnings reports", category="finance").with_inputs("text"),
    dspy.Example(text="New study reveals high-protein diets improve recovery", category="health").with_inputs("text"),
    dspy.Example(text="SpaceX successfully launched 23 Starlink satellites", category="technology").with_inputs("text"),
    dspy.Example(text="Team wins championship after dramatic overtime victory", category="sports").with_inputs("text"),
    # Include 50-200 examples for best results
]

# Step 4: Define a metric
def classification_metric(example, prediction, trace=None):
    return prediction.category.strip().lower() == example.category.strip().lower()

# Step 5: Create teacher and student copies
teacher = Classifier()  # Uses the configured teacher_lm (GPT-4o)

student = Classifier()
student.set_lm(student_lm)  # Student uses GPT-4o-mini

# Step 6: Configure BootstrapFinetune
# Constructor params: metric, multitask, train_kwargs, adapter, exclude_demos, num_threads
bootstrap_finetune = dspy.BootstrapFinetune(
    metric=classification_metric,
    num_threads=4,
)

# Step 7: Compile - generates traces from teacher, fine-tunes student
# compile() params: student, trainset, teacher
dspy.settings.experimental = True  # Fine-tuning is experimental
distilled = bootstrap_finetune.compile(
    student,
    trainset=trainset,
    teacher=teacher,
)

# Step 8: Test the distilled program
result = distilled(text="Researchers discover high-protein diets improve recovery times")
print(f"Category: {result.category}")
print(f"Confidence: {result.confidence}")

# Step 9: Evaluate
devset = [
    dspy.Example(text="NASA announced a new mission to Mars", category="technology").with_inputs("text"),
    dspy.Example(text="Gold prices reached an all-time high amid inflation fears", category="finance").with_inputs("text"),
]

evaluator = Evaluate(devset=devset, metric=classification_metric, num_threads=4, display_progress=True)
score = evaluator(distilled)
print(f"\nDistilled model accuracy: {score:.1f}%")
