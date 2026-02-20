"""
Blog 6.4 - Memory and Conversation History
Run: python 01_memory_agent.py
"""

import json
from pathlib import Path

import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# ============================================================
# Part 1: Basic Conversational QA with dspy.History
# ============================================================


class ConversationalQA(dspy.Signature):
    """Answer the user's question, taking into account the conversation history."""

    history: dspy.History = dspy.InputField(desc="Previous conversation turns")
    question: str = dspy.InputField(desc="The user's current question")
    answer: str = dspy.OutputField(desc="A helpful, context-aware answer")


qa = dspy.ChainOfThought(ConversationalQA)

history = dspy.History(messages=[])

# Turn 1
result1 = qa(history=history, question="What is the capital of France?")
print(f"Turn 1: {result1.answer}")
history = history.append(user="What is the capital of France?", assistant=result1.answer)

# Turn 2 (references prior context)
result2 = qa(history=history, question="What's its population?")
print(f"Turn 2: {result2.answer}")
history = history.append(user="What's its population?", assistant=result2.answer)

# Turn 3 (builds on full conversation)
result3 = qa(history=history, question="How does that compare to London?")
print(f"Turn 3: {result3.answer}")


# ============================================================
# Part 2: Memory-Enabled ReAct Agent
# ============================================================

print("\n" + "=" * 60)
print("Memory-Enabled ReAct Agent (Customer Service)")
print("=" * 60 + "\n")


def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base for information about products and policies."""
    kb = {
        "refund": "Refunds are processed within 5-7 business days. Items must be returned within 30 days.",
        "shipping": "Standard shipping takes 3-5 days. Express shipping is available for $9.99.",
        "warranty": "All products come with a 1-year warranty covering manufacturing defects.",
    }
    query_lower = query.lower()
    for key, value in kb.items():
        if key in query_lower:
            return value
    return "No relevant information found. Please contact support@example.com."


def check_order_status(order_id: str) -> str:
    """Check the status of an order by its order ID."""
    orders = {
        "ORD-001": "Shipped, expected delivery Feb 18, 2026",
        "ORD-002": "Processing, will ship within 24 hours",
        "ORD-003": "Delivered on Feb 12, 2026",
    }
    return orders.get(order_id.upper(), f"Order {order_id} not found.")


agent = dspy.ReAct(
    "history, question -> answer",
    tools=[search_knowledge_base, check_order_status],
    max_iters=5,
)

history = dspy.History(messages=[])

queries = [
    "What is your refund policy?",
    "OK, and what about shipping times?",
    "Can you check order ORD-002 for me?",
    "Based on what you just told me, will it arrive before Friday?",
]

for question in queries:
    result = agent(history=history, question=question)
    print(f"Customer: {question}")
    print(f"Agent:    {result.answer}\n")
    history = history.append(user=question, assistant=result.answer)


# ============================================================
# Part 3: Persistent Memory (File-Based)
# ============================================================

print("=" * 60)
print("Persistent Memory Demo")
print("=" * 60 + "\n")


class PersistentMemory:
    """Simple file-based persistent memory for conversation history."""

    def __init__(self, user_id: str, storage_dir: str = "./memory"):
        self.user_id = user_id
        self.storage_path = Path(storage_dir) / f"{user_id}.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def load_history(self) -> dspy.History:
        """Load conversation history from disk."""
        if self.storage_path.exists():
            data = json.loads(self.storage_path.read_text())
            return dspy.History(messages=data)
        return dspy.History(messages=[])

    def save_history(self, history: dspy.History):
        """Save conversation history to disk."""
        self.storage_path.write_text(json.dumps(history.messages, indent=2))


# Demo persistent memory
memory = PersistentMemory(user_id="demo_user")
history = memory.load_history()

demo_qa = dspy.ChainOfThought(ConversationalQA)
result = demo_qa(history=history, question="Hello! What can you help me with?")
print(f"Agent: {result.answer}")

history = history.append(user="Hello! What can you help me with?", assistant=result.answer)
memory.save_history(history)
print("History saved to ./memory/demo_user.json")
