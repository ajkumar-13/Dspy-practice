"""
Blog 7.3: Ensemble - Combining Programs
Run: python 01_ensemble.py

Demonstrates combining multiple optimized DSPy program variants
into a single robust ensemble using dspy.Ensemble with majority voting.
"""

import dspy
from dspy.evaluate import Evaluate
from dotenv import load_dotenv

load_dotenv()

# Step 1: Configure the LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Step 2: Define the base program
class QA(dspy.Module):
    def __init__(self):
        self.answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.answer(question=question)

# Step 3: Prepare training data
trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="Who wrote Hamlet?", answer="Shakespeare").with_inputs("question"),
    dspy.Example(question="What is the speed of light in m/s?", answer="299792458").with_inputs("question"),
    dspy.Example(question="What element has atomic number 79?", answer="Gold").with_inputs("question"),
    dspy.Example(question="In what year did World War II end?", answer="1945").with_inputs("question"),
    # Include more examples for best results
]

# Step 4: Define a metric
def exact_match(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()


# -------------------------------------------------------
# Strategy 1: Optimize N variants with different seeds
# -------------------------------------------------------
programs = []
for seed in range(5):
    optimizer = dspy.BootstrapFewShotWithRandomSearch(
        metric=exact_match,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        num_candidate_programs=8,
        seed=seed,
    )
    optimized = optimizer.compile(QA(), trainset=trainset)
    programs.append(optimized)
    print(f"Optimized variant {seed + 1}/5")

# Step 5: Ensemble the variants with majority voting
# Constructor: (reduce_fn, size, deterministic)
ensemble = dspy.Ensemble(reduce_fn=dspy.majority, size=5)
ensembled_program = ensemble.compile(programs)

# Step 6: Test the ensemble
result = ensembled_program(question="What is the capital of Japan?")
print(f"\nEnsemble answer: {result.answer}")


# -------------------------------------------------------
# Strategy 2: Different optimizers for more diversity
# -------------------------------------------------------
opt1 = dspy.BootstrapFewShotWithRandomSearch(metric=exact_match)
prog1 = opt1.compile(QA(), trainset=trainset)

opt2 = dspy.MIPROv2(metric=exact_match, auto="medium")
prog2 = opt2.compile(QA(), trainset=trainset)

diverse_programs = [prog1, prog2]
diverse_ensemble = dspy.Ensemble(reduce_fn=dspy.majority, size=2)
diverse_combined = diverse_ensemble.compile(diverse_programs)


# -------------------------------------------------------
# Strategy 3: Sampled ensemble (cost-effective)
# -------------------------------------------------------
# Run only 3 of the 5 variants per input (cheaper)
sampled_ensemble = dspy.Ensemble(reduce_fn=dspy.majority, size=3)
sampled_combined = sampled_ensemble.compile(programs)


# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------
devset = [
    dspy.Example(question="What is the largest planet in our solar system?", answer="Jupiter").with_inputs("question"),
    dspy.Example(question="Who painted the Mona Lisa?", answer="Leonardo da Vinci").with_inputs("question"),
]

evaluator = Evaluate(devset=devset, metric=exact_match, num_threads=4, display_progress=True)

print("\n--- Full Ensemble (5 of 5) ---")
score_full = evaluator(ensembled_program)
print(f"Full ensemble accuracy: {score_full:.1f}%")

print("\n--- Sampled Ensemble (3 of 5) ---")
score_sampled = evaluator(sampled_combined)
print(f"Sampled ensemble accuracy: {score_sampled:.1f}%")
