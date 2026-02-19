# 6.4 Memory and Conversation History


## Introduction

Every agent we've built so far is **stateless**. Each call starts from a blank slate with no memory of prior interactions. That's fine for one-shot questions, but useless for real applications like customer service, tutoring, or multi-turn research. Users expect agents to remember what was said, build on prior context, and maintain a coherent conversation.

DSPy provides the `dspy.History` type for managing conversation history, and patterns for building agents with both **short-term memory** (within a conversation) and **long-term memory** (across sessions). In this post, you'll learn how to build conversational agents that remember.

---

## What You'll Learn

- The `dspy.History` type for managing conversation turns
- Building multi-turn conversational agents
- Accumulating history across turns and passing it as input
- Memory-enabled ReAct agents
- Short-term vs. long-term memory patterns
- Building a customer service agent with conversation context

---

## Prerequisites

- Completed [6.3 MCP Integration](../6.3-mcp-integration/blog.md)
- DSPy installed (`uv add dspy python-dotenv`)
- An OpenAI API key (or any LiteLLM-supported provider)

---

## The `dspy.History` Type

`dspy.History` is DSPy's built-in type for representing conversation history. It holds a sequence of messages (user inputs and assistant responses) that you can pass into any DSPy module as an input field.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


class ConversationalQA(dspy.Signature):
    """Answer the user's question, taking into account the conversation history."""
    history: dspy.History = dspy.InputField(desc="Previous conversation turns")
    question: str = dspy.InputField(desc="The user's current question")
    answer: str = dspy.OutputField(desc="A helpful, context-aware answer")


qa = dspy.ChainOfThought(ConversationalQA)

# Start a conversation
history = dspy.History(messages=[])

# Turn 1
result1 = qa(history=history, question="What is the capital of France?")
print(f"Turn 1: {result1.answer}")

# Update history with the exchange
history = history.append(user="What is the capital of France?", assistant=result1.answer)

# Turn 2 — references prior context
result2 = qa(history=history, question="What's its population?")
print(f"Turn 2: {result2.answer}")

# Update history again
history = history.append(user="What's its population?", assistant=result2.answer)

# Turn 3 — builds on the full conversation
result3 = qa(history=history, question="How does that compare to London?")
print(f"Turn 3: {result3.answer}")
```

The pattern is straightforward:

1. **Initialize** an empty `dspy.History`.
2. **Pass it** alongside the current question to your module.
3. **Append** each user/assistant exchange after every turn.
4. The model sees the full conversation context and can resolve references like "its" or "that."

---

## Building a Conversational Agent Loop

In practice, you'll run this in a loop. Here's a complete interactive agent:

```python

import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


class ChatAgent(dspy.Signature):
    """You are a helpful assistant. Answer questions based on the conversation context."""
    history: dspy.History = dspy.InputField(desc="Conversation history so far")
    question: str = dspy.InputField(desc="The user's current message")
    answer: str = dspy.OutputField(desc="Your response")


agent = dspy.ChainOfThought(ChatAgent)
history = dspy.History(messages=[])

print("Chat with the agent (type 'quit' to exit):\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ("quit", "exit"):
        break

    result = agent(history=history, question=user_input)
    print(f"Agent: {result.answer}\n")

    history = history.append(user=user_input, assistant=result.answer)
```

---

## Memory-Enabled ReAct Agents

You can combine conversation history with tool-using agents. This gives you a ReAct agent that remembers previous interactions while still being able to call tools:

```python

import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


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
        "ORD-001": "Shipped — expected delivery Feb 18, 2026",
        "ORD-002": "Processing — will ship within 24 hours",
        "ORD-003": "Delivered on Feb 12, 2026",
    }
    return orders.get(order_id.upper(), f"Order {order_id} not found.")


# Build a memory-enabled ReAct agent
agent = dspy.ReAct(
    "history, question -> answer",
    tools=[search_knowledge_base, check_order_status],
    max_iters=5,
)

# Simulate a multi-turn customer service interaction
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
```

Notice the signature is `"history, question -> answer"`. The `history` field is an explicit input, so the ReAct agent sees the full conversation context while deciding which tools to call.

---

## Short-Term vs. Long-Term Memory

The patterns above implement **short-term memory**. History persists within a conversation session but is lost when the program ends. For production agents, you often need **long-term memory** that persists across sessions.

### Short-Term Memory (In-Session)

- Managed by `dspy.History`. Lives in memory.
- Automatically passed to the module at each turn.
- Lost when the process ends.
- Best for: single conversations, chat sessions, task-oriented dialogs.

### Long-Term Memory (Cross-Session)

For persistent memory, you need an external store. A simple pattern:

```python
import json
from pathlib import Path


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


# Usage
memory = PersistentMemory(user_id="customer_123")
history = memory.load_history()

# ... run conversation turns, updating history ...

memory.save_history(history)
```

For production deployments, replace the file-based store with Redis, a database, or a dedicated memory service. The DSPy pattern stays the same; you just change where `dspy.History` is loaded from and saved to.

---

## Key Takeaways

- **`dspy.History`** is DSPy's built-in type for managing conversation turns. Pass it as an input field to any module.
- **Accumulate history** after each turn with `history.append(user=..., assistant=...)`. The model sees the full conversation context.
- **Memory-enabled ReAct agents** use the signature `"history, question -> answer"` to combine tool use with conversation awareness.
- **Short-term memory** lives in `dspy.History` during a session; **long-term memory** requires external persistence (files, databases, memory services).
- **The pattern is always the same:** load history, pass to module, get response, append to history, save.

---

## Next Up

Some tasks require routing between different models, sending sensitive data to a local model while delegating non-sensitive work to the cloud. In the next post, we'll explore **Papillon**, a privacy-conscious delegation pattern for multi-model agent architectures.

**[6.5: Papillon: Privacy-Conscious Delegation →](../6.5-papillon/blog.md)**

---

## Resources

- 📖 [DSPy Conversation History Tutorial](https://dspy.ai/tutorials/conversation_history/)
- 📖 [Mem0 ReAct Agent Tutorial](https://dspy.ai/tutorials/mem0_react_agent/)
- 📖 [DSPy History API Reference](https://dspy.ai/api/types/History/)
- 📖 [Building Customer Service Agents](https://dspy.ai/tutorials/customer_service_agent/)
- 💻 [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
