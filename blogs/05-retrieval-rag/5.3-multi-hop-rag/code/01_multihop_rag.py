"""
Blog 5.3: Multi-Hop RAG
Run: python 01_multihop_rag.py
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Setup a mock corpus for multi-hop
corpus = [
    "David Foster Wallace wrote Infinite Jest, which was published in 1996.",
    "Infinite Jest is set in a future North America where the US, Canada, and Mexico have formed O.N.A.N.",
    "The Pale King is an unfinished novel by David Foster Wallace.",
    "David Foster Wallace was born in Ithaca, New York.",
    "The capital of O.N.A.N. in Infinite Jest is Boston.",
]

search = dspy.retrievers.Embeddings(  # type: ignore[attr-defined]
    embedder=dspy.Embedder("openai/text-embedding-3-small", dimensions=512),  # type: ignore[arg-type]
    corpus=corpus,
    k=2,
)


class GenerateSearchQuery(dspy.Signature):
    """Generate a search query to find missing information."""

    context = dspy.InputField(desc="Information we already know")
    question = dspy.InputField(desc="The complex question we need to answer")
    query = dspy.OutputField(desc="A targeted search query for the next step")


class MultiHopRAG(dspy.Module):
    def __init__(self, max_hops=2):
        self.max_hops = max_hops
        self.generate_query = dspy.ChainOfThought(GenerateSearchQuery)
        self.respond = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            # Generate query based on what we know so far
            query_result = self.generate_query(context=context, question=question)
            query = query_result.query
            
            # Search
            passages = search(query)
            
            # Deduplicate and add to context
            new_passages = [p for p in passages if p not in context]
            context.extend(new_passages)
            
            print(f"Hop {hop+1} Query: {query}")
            print(f"Hop {hop+1} Found: {new_passages}")

        # Final answer
        return self.respond(context=context, question=question)


# Run
rag = MultiHopRAG(max_hops=2)
question = "In which city is the capital of the superstate featured in the 1996 novel by David Foster Wallace?"

print(f"\nQuestion: {question}\n")
response = rag(question)
print(f"\nFinal Answer: {response.answer}")
