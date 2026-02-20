"""
Blog 4.4: Optimizer Landscape - Comparing and Stacking Optimizers
Run: python 01_optimizer_comparison.py
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# -------------------------------------------------------------------------
# Training and evaluation data
# -------------------------------------------------------------------------
trainset = [
    dspy.Example(question="What is the largest planet?", answer="Jupiter").with_inputs("question"),
    dspy.Example(question="Who wrote Romeo and Juliet?", answer="Shakespeare").with_inputs(
        "question"
    ),
    dspy.Example(
        question="What is the boiling point of water in Celsius?", answer="100"
    ).with_inputs("question"),
    dspy.Example(question="What is the chemical symbol for gold?", answer="Au").with_inputs(
        "question"
    ),
    dspy.Example(question="How many continents are there?", answer="7").with_inputs("question"),
    dspy.Example(question="What is the square root of 144?", answer="12").with_inputs("question"),
    dspy.Example(question="Who painted the Mona Lisa?", answer="Leonardo da Vinci").with_inputs(
        "question"
    ),
    dspy.Example(question="What is the smallest prime number?", answer="2").with_inputs("question"),
    dspy.Example(question="What gas do plants absorb?", answer="Carbon dioxide").with_inputs(
        "question"
    ),
    dspy.Example(question="How many legs does a spider have?", answer="8").with_inputs("question"),
]

devset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is the speed of light in km/s?", answer="299792").with_inputs(
        "question"
    ),
    dspy.Example(question="Who discovered gravity?", answer="Isaac Newton").with_inputs("question"),
    dspy.Example(question="What is the longest river?", answer="The Nile").with_inputs("question"),
    dspy.Example(question="What is the currency of Japan?", answer="Yen").with_inputs("question"),
]


def metric(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()


program = dspy.ChainOfThought("question -> answer")

# -------------------------------------------------------------------------
# Demo: Few-Shot Optimizers
# -------------------------------------------------------------------------
print("=" * 60)
print("FEW-SHOT OPTIMIZERS")
print("=" * 60)

# LabeledFewShot: just use your data directly
tp_labeled = dspy.LabeledFewShot(k=4)
optimized_labeled = tp_labeled.compile(program, trainset=trainset)
print("LabeledFewShot compiled.")

# BootstrapFewShotWithRandomSearch
tp_bootstrap = dspy.BootstrapFewShotWithRandomSearch(
    metric=metric,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
    num_candidate_programs=5,
)
optimized_bootstrap = tp_bootstrap.compile(program, trainset=trainset)
print("BootstrapRS compiled.")

# -------------------------------------------------------------------------
# Demo: Optimizer Stacking (BootstrapRS -> MIPROv2)
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("OPTIMIZER STACKING: BootstrapRS -> MIPROv2")
print("=" * 60)

# Step 1: Start with BootstrapRS for a quick baseline
tp1 = dspy.BootstrapFewShotWithRandomSearch(
    metric=metric,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
    num_candidate_programs=5,
)
step1 = tp1.compile(program, trainset=trainset)

# Step 2: Refine with MIPROv2 for better instructions
tp2 = dspy.MIPROv2(metric=metric, auto="medium", num_threads=8)
step2 = tp2.compile(step1, trainset=trainset)

# Step 3: Evaluate the improvement
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=4)
print(f"Baseline:  {evaluate(program)}")
print(f"Step 1:    {evaluate(step1)}")
print(f"Step 2:    {evaluate(step2)}")

# Save the final stacked result
step2.save("optimized_stacked.json")
print("\nStacked optimized program saved to optimized_stacked.json")
