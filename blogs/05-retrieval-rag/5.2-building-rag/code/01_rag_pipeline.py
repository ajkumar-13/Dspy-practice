"""
Blog 5.2: Building RAG Pipelines
Run: python 01_rag_pipeline.py
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Set up retrieval: FAISS-backed local search
embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)
corpus = [
    "The Amazon rainforest produces about 20% of the world's oxygen.",
    "The Sahara Desert is the largest hot desert, spanning 3.6 million square miles.",
    "The Pacific Ocean covers more area than all the Earth's land combined.",
    "Mount Everest grows approximately 4mm taller each year due to tectonic activity.",
    "The Mariana Trench is the deepest point in the ocean at about 36,000 feet.",
    "Antarctica contains about 70% of the world's fresh water in its ice sheet.",
    "The Dead Sea is the lowest point on land at 430 meters below sea level.",
    "Lake Baikal in Russia holds about 20% of the world's unfrozen fresh water.",
    "The Nile River, at 6,650 km, is the longest river in the world.",
    "The Great Barrier Reef is the largest living structure on Earth, visible from space.",
]

search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=3)


class RAG(dspy.Module):
    """A simple Retrieve-then-Read RAG pipeline."""

    def __init__(self):
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        results = search(question)
        # Pass the list of strings directly as context
        return self.respond(context=results, question=question)


# Build evaluation data
trainset = [
    dspy.Example(
        question="What percentage of the world's oxygen does the Amazon produce?",
        response="About 20%",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the largest hot desert in the world?",
        response="The Sahara Desert",
    ).with_inputs("question"),
    dspy.Example(
        question="How deep is the Mariana Trench?",
        response="About 36,000 feet",
    ).with_inputs("question"),
    dspy.Example(
        question="What percentage of the world's fresh water does Antarctica contain?",
        response="About 70%",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the longest river in the world?",
        response="The Nile River at 6,650 km",
    ).with_inputs("question"),
]

devset = [
    dspy.Example(
        question="What ocean covers more area than all land combined?",
        response="The Pacific Ocean",
    ).with_inputs("question"),
    dspy.Example(
        question="How much does Mount Everest grow each year?",
        response="About 4mm per year",
    ).with_inputs("question"),
    dspy.Example(
        question="Where is the lowest point on land?",
        response="The Dead Sea, at 430 meters below sea level",
    ).with_inputs("question"),
]

# Run baseline
rag = RAG()
metric = dspy.SemanticF1()

print("\n=== Evaluating Baseline RAG ===")
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
baseline_score = evaluate(rag)
print(f"Baseline SemanticF1: {baseline_score:.1f}")

# Optimize with MIPROv2
print("\n=== Optimizing with MIPROv2 ===")
optimizer = dspy.MIPROv2(
    metric=metric,
    auto="light",  # Using light for faster demo run
    num_threads=4,
)

optimized_rag = optimizer.compile(rag, trainset=trainset)

print("\n=== Evaluating Optimized RAG ===")
optimized_score = evaluate(optimized_rag)
print(f"Optimized SemanticF1: {optimized_score:.1f}")
print(f"Improvement: {optimized_score - baseline_score:.1f} absolute")

# Inspect results
print("\n=== Inspection ===")
result = optimized_rag(question="What is the deepest point in the ocean?")
print(f"Q: What is the deepest point in the ocean?")
print(f"A: {result.response}")
