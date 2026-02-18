"""
Blog 5.P: Project Research Assistant
Run: python 01_research_assistant.py
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 1. Retrieval Backend (ColBERTv2)
colbert = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")

# 2. Tools
def search_wikipedia(query: str) -> list[str]:
    """Search for relevant Wikipedia passages."""
    results = colbert(query, k=5)
    return [r.long_text for r in results]

def lookup_details(passage_snippet: str, keyword: str) -> str:
    """Find specific sentences containing a keyword within a passage."""
    # Mocking lookup logic for demo simplicity, in reality would re-access full text
    if keyword.lower() in passage_snippet.lower():
        return f"Found details about '{keyword}' in: {passage_snippet[:100]}..."
    return "Keyword not found in snippet."

# 3. Agent
agent = dspy.ReAct("question -> answer", tools=[search_wikipedia, lookup_details])

# 4. Data
# Complex multi-hop questions
dataset = [
    dspy.Example(
        question="Which writer born in Ithaca, New York wrote the novel Infinite Jest?",
        answer="David Foster Wallace",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the capital of the country where the Eiffel Tower is located?",
        answer="Paris",
    ).with_inputs("question"),
    dspy.Example(
        question="Who was the US President when the first iPhone was released?",
        answer="George W. Bush",
    ).with_inputs("question"),
    dspy.Example(
        question="What river flows through the city where the 2024 Olympic Games were held?",
        answer="The Seine",
    ).with_inputs("question"),
]

trainset = dataset[:2]
devset = dataset[2:]

# 5. Metric
metric = dspy.SemanticF1()

# 6. Baseline Evaluation
print("\n=== Baseline Evaluation ===")
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
baseline_score = evaluate(agent)
print(f"Baseline Score: {baseline_score:.1f}")

# 7. Optimization
print("\n=== Optimizing (MIPROv2) ===")
# Using "light" for demo speed, "medium" recommended for real use
optimizer = dspy.MIPROv2(metric=metric, auto="light", num_threads=4)
optimized_agent = optimizer.compile(agent, trainset=trainset)

# 8. Comparison
print("\n=== Optimized Evaluation ===")
optimized_score = evaluate(optimized_agent)
print(f"Optimized Score: {optimized_score:.1f}")
print(f"Improvement: {optimized_score - baseline_score:.1f}")

# Save
optimized_agent.save("research_assistant.json")
