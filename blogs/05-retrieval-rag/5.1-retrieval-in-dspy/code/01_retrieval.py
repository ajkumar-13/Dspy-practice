"""
Blog 5.1: Retrieval in DSPy
Run: python 01_retrieval.py
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure the LM (for later use in RAG)
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# ==============================================================================
# Example 1: Local Retrieval with Embeddings (FAISS)
# ==============================================================================
print("\n=== Example 1: Local FAISS Retrieval ===\n")

# Step 1: Create an embedder
embedder = dspy.Embedder(
    "openai/text-embedding-3-small",
    dimensions=512,
)

# Step 2: Define a corpus
corpus = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889.",
    "The Great Wall of China stretches over 13,000 miles and was built over many centuries.",
    "The Colosseum in Rome, Italy, is an ancient amphitheater that could hold up to 80,000 spectators.",
    "Machu Picchu is a 15th-century Inca citadel in the Andes Mountains of Peru.",
    "The Taj Mahal is an ivory-white marble mausoleum in Agra, India, built between 1632 and 1653.",
    "The Statue of Liberty was a gift from France to the United States, dedicated in 1886.",
    "Petra is a historical city in southern Jordan, famous for its rock-cut architecture.",
    "Christ the Redeemer is a 30-meter Art Deco statue of Jesus in Rio de Janeiro, Brazil.",
    "The Sydney Opera House opened in 1973 and is one of the most distinctive buildings of the 20th century.",
    "Angkor Wat in Cambodia is the largest religious monument in the world, spanning 400 acres.",
]

# Step 3: Build the retriever
# Note: dspy.retrievers.Embeddings creates a FAISS index automatically
search = dspy.retrievers.Embeddings(
    embedder=embedder,
    corpus=corpus,
    k=3,
)

# Step 4: Use the retriever
query = "Where can I find ancient Roman architecture?"
results = search(query)

print(f"Query: {query}\n")
for i, passage in enumerate(results, 1):
    print(f"  [{i}] {passage}")


# ==============================================================================
# Example 2: Hosted ColBERTv2 Retrieval
# ==============================================================================
print("\n=== Example 2: Hosted ColBERTv2 Retrieval ===\n")

# Connect to a public ColBERTv2 endpoint (Wikipedia abstracts)
colbert = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")

query = "What is the tallest mountain in the world?"
results = colbert(query, k=3)

print(f"Query: {query}\n")
for i, r in enumerate(results, 1):
    # ColBERTv2 returns objects with .long_text, .score, and .pid
    print(f"  [{i}] {r.long_text[:100]}...")


# ==============================================================================
# Example 3: Custom Retriever Function
# ==============================================================================
print("\n=== Example 3: Custom Retriever Function ===\n")


def my_custom_search(query: str, k: int = 5) -> list[str]:
    """
    A custom retriever function.
    DSPy just needs a callable that returns a list of strings.
    """
    custom_corpus = [
        "Python was created by Guido van Rossum and first released in 1991.",
        "JavaScript is the most popular programming language for web development.",
        "Rust emphasizes memory safety without garbage collection.",
        "Go was designed at Google and is known for its simplicity and concurrency support.",
    ]
    # Simple keyword matching (demo only)
    scored = [
        (doc, sum(1 for w in query.lower().split() if w in doc.lower())) for doc in custom_corpus
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored[:k]]


# Use it directly
query = "Who created Python?"
results = my_custom_search(query, k=1)
print(f"Query: {query}")
print(f"Result: {results[0]}")
