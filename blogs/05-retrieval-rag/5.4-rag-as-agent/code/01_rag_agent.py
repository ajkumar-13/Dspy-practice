"""
Blog 5.4: Agentic RAG with ReAct
Run: python 01_rag_agent.py
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Setup a mock corpus and retriever
corpus = [
    "Apollo 11 was the spaceflight that first landed humans on the Moon.",
    "Neil Armstrong and Buzz Aldrin formed the American crew that landed the Apollo Lunar Module Eagle on July 20, 1969.",
    "Michael Collins flew the Command Module Columbia alone in lunar orbit while they were on the Moon's surface.",
    "The Saturn V was the rocket used for the mission.",
    "The mission duration was 8 days, 3 hours, 18 minutes, and 35 seconds.",
]

embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)
retriever = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=3)

# Define tools
def search_wikipedia(query: str) -> str:
    """Searches Wikipedia for the given query and returns snippets."""
    results = retriever(query)
    return "\n\n".join(results)

def calculate_years_ago(year: str) -> str:
    """Calculates how many years ago a given year was (assuming current year is 2025)."""
    try:
        y = int(year)
        diff = 2025 - y
        return f"{diff} years ago"
    except ValueError:
        return "Invalid year format"

# Create the ReAct agent
# dspy.ReAct takes a signature and a list of tool functions
agent = dspy.ReAct("question -> answer", tools=[search_wikipedia, calculate_years_ago])

# Example 1: Simple Retrieval
print("\n=== Example 1: Simple Retrieval ===\n")
question1 = "Who flew the command module during Apollo 11?"
print(f"Question: {question1}")
result1 = agent(question=question1)
print(f"Answer: {result1.answer}")

# Example 2: Multi-step Reasoning (Search + Calc)
print("\n=== Example 2: Multi-step Reasoning ===\n")
question2 = "How many years ago did the Apollo 11 mission happen?"
print(f"Question: {question2}")
result2 = agent(question=question2)
print(f"Answer: {result2.answer}")

# Inspect the trace to see tool calls
print("\n=== Inspection (Last Trace) ===\n")
dspy.inspect_history(n=1)
