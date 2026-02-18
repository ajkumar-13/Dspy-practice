"""
Blog 4.1: BootstrapRS Optimization
Run: python 01_bootstrap_rs.py
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Build training set
trainset = [
    dspy.Example(question="What is the largest planet?", answer="Jupiter").with_inputs("question"),
    dspy.Example(question="Who wrote Romeo and Juliet?", answer="Shakespeare").with_inputs("question"),
    dspy.Example(question="What is the boiling point of water in Celsius?", answer="100").with_inputs("question"),
    dspy.Example(question="What is the chemical symbol for gold?", answer="Au").with_inputs("question"),
    dspy.Example(question="How many continents are there?", answer="7").with_inputs("question"),
    dspy.Example(question="What is the square root of 144?", answer="12").with_inputs("question"),
    dspy.Example(question="Who painted the Mona Lisa?", answer="Leonardo da Vinci").with_inputs("question"),
    dspy.Example(question="What is the smallest prime number?", answer="2").with_inputs("question"),
    dspy.Example(question="What gas do plants absorb?", answer="Carbon dioxide").with_inputs("question"),
    dspy.Example(question="How many legs does a spider have?", answer="8").with_inputs("question"),
]


# Define metric
def answer_match(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()


# Create program
program = dspy.ChainOfThought("question -> answer")

# Optimize
teleprompter = dspy.BootstrapFewShotWithRandomSearch(
    metric=answer_match,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
    num_candidate_programs=5,
    num_threads=4,
)

optimized = teleprompter.compile(program, trainset=trainset)

# Test the optimized program
result = optimized(question="What is the capital of Japan?")
print(f"Answer: {result.answer}")

# Inspect the optimized prompt
result = optimized(question="What is the speed of light?")
dspy.inspect_history(n=1)

# Access the predictor's demos
for predictor in optimized.predictors():
    print(f"Demos count: {len(predictor.demos)}")
    for demo in predictor.demos:
        print(f"  Q: {demo.get('question', 'N/A')}")
        print(f"  A: {demo.get('answer', 'N/A')}")
        print()

# Save the optimized program
optimized.save("optimized_qa.json")
print("Optimized program saved to optimized_qa.json")

# Later, load it back
loaded = dspy.ChainOfThought("question -> answer")
loaded.load("optimized_qa.json")

# Use it: the demonstrations are restored
result = loaded(question="What is the speed of light?")
print(result.answer)
