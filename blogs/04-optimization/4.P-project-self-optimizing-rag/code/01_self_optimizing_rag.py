"""
Blog 4.P: Self-Optimizing RAG Pipeline
Run: python 01_self_optimizing_rag.py
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Configure retriever: ColBERTv2 over Wikipedia abstracts
retriever = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")


class RAG(dspy.Module):
    """A simple Retrieve-then-Read RAG pipeline."""

    def __init__(self, num_passages=3):
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.respond(context=context, question=question)


# -------------------------------------------------------------------------
# Step 2: Prepare Data
# -------------------------------------------------------------------------

# Training set: used for optimization
trainset = [
    dspy.Example(question="What is the tallest mountain in the world?", response="Mount Everest").with_inputs("question"),
    dspy.Example(question="Who invented the telephone?", response="Alexander Graham Bell").with_inputs("question"),
    dspy.Example(question="What is the chemical formula for water?", response="H2O").with_inputs("question"),
    dspy.Example(question="Which country has the largest population?", response="China").with_inputs("question"),
    dspy.Example(question="What year did World War II end?", response="1945").with_inputs("question"),
    dspy.Example(question="Who painted the Sistine Chapel ceiling?", response="Michelangelo").with_inputs("question"),
    dspy.Example(question="What is the speed of light in km/s?", response="299,792 km/s").with_inputs("question"),
    dspy.Example(question="What is the largest organ in the human body?", response="The skin").with_inputs("question"),
    dspy.Example(question="Who wrote the theory of relativity?", response="Albert Einstein").with_inputs("question"),
    dspy.Example(question="What is the capital of Australia?", response="Canberra").with_inputs("question"),
    dspy.Example(question="What element has the atomic number 1?", response="Hydrogen").with_inputs("question"),
    dspy.Example(question="Who was the first person to walk on the Moon?", response="Neil Armstrong").with_inputs("question"),
    dspy.Example(question="What is the longest river in the world?", response="The Nile").with_inputs("question"),
    dspy.Example(question="What language has the most native speakers?", response="Mandarin Chinese").with_inputs("question"),
    dspy.Example(question="What is the boiling point of water in Fahrenheit?", response="212\u00b0F").with_inputs("question"),
    dspy.Example(question="Who discovered gravity?", response="Isaac Newton").with_inputs("question"),
    dspy.Example(question="What is the largest desert in the world?", response="The Sahara Desert").with_inputs("question"),
    dspy.Example(question="What is the currency of Japan?", response="Japanese yen").with_inputs("question"),
    dspy.Example(question="Who wrote Hamlet?", response="William Shakespeare").with_inputs("question"),
    dspy.Example(question="What is the smallest planet in our solar system?", response="Mercury").with_inputs("question"),
]

# Dev set: used for evaluation (held out from training)
devset = [
    dspy.Example(question="What is the largest planet in our solar system?", response="Jupiter").with_inputs("question"),
    dspy.Example(question="Who developed the polio vaccine?", response="Jonas Salk").with_inputs("question"),
    dspy.Example(question="What is the freezing point of water in Celsius?", response="0\u00b0C").with_inputs("question"),
    dspy.Example(question="What is the capital of Brazil?", response="Bras\u00edlia").with_inputs("question"),
    dspy.Example(question="Who wrote Pride and Prejudice?", response="Jane Austen").with_inputs("question"),
    dspy.Example(question="What is the most abundant gas in Earth's atmosphere?", response="Nitrogen").with_inputs("question"),
    dspy.Example(question="What is the deepest ocean?", response="The Pacific Ocean").with_inputs("question"),
    dspy.Example(question="Who invented the light bulb?", response="Thomas Edison").with_inputs("question"),
    dspy.Example(question="What is the hardest natural substance?", response="Diamond").with_inputs("question"),
    dspy.Example(question="What year did the Berlin Wall fall?", response="1989").with_inputs("question"),
]


# -------------------------------------------------------------------------
# Step 3: Define Metric
# -------------------------------------------------------------------------

def semantic_f1(example, prediction, trace=None):
    """Compute token-level F1 between prediction and gold answer."""
    pred_tokens = set(prediction.response.lower().split())
    gold_tokens = set(example.response.lower().split())

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = pred_tokens & gold_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# -------------------------------------------------------------------------
# Step 4: Baseline Evaluation
# -------------------------------------------------------------------------

# Create the baseline RAG program
rag = RAG(num_passages=3)

# Run evaluation
evaluate = dspy.Evaluate(
    devset=devset,
    metric=semantic_f1,
    num_threads=4,
    display_progress=True,
    display_table=5,
)

baseline_score = evaluate(rag)
print(f"\nBaseline SemanticF1: {baseline_score:.1f}%")


# -------------------------------------------------------------------------
# Step 5: Optimize with MIPROv2
# -------------------------------------------------------------------------

tp = dspy.MIPROv2(
    metric=semantic_f1,
    auto="medium",
    num_threads=4,
)

# Compile: this takes 10-20 minutes
optimized_rag = tp.compile(
    rag,
    trainset=trainset,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
)


# -------------------------------------------------------------------------
# Step 6: Evaluate Optimized Program
# -------------------------------------------------------------------------

optimized_score = evaluate(optimized_rag)
print(f"\nOptimized SemanticF1: {optimized_score:.1f}%")
print(f"Improvement: {optimized_score - baseline_score:+.1f}%")


# -------------------------------------------------------------------------
# Step 7: Inspect and Save
# -------------------------------------------------------------------------

# Inspect the optimized prompt
result = optimized_rag(question="What is the capital of France?")
dspy.inspect_history(n=1)

# Check what demonstrations were selected
for predictor in optimized_rag.predictors():
    print(f"\nPredictor demos ({len(predictor.demos)}):")
    for i, demo in enumerate(predictor.demos):
        print(f"  Demo {i+1}:")
        print(f"    Q: {demo.get('question', 'N/A')}")
        print(f"    R: {demo.get('response', 'N/A')[:80]}...")

# Save the optimized program for production use
optimized_rag.save("optimized_rag.json")
print("\nOptimized program saved to optimized_rag.json")

# Later, load it back
loaded_rag = RAG(num_passages=3)
loaded_rag.load("optimized_rag.json")
