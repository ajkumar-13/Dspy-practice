# 5.3: Multi-Hop RAG

## Introduction

Some questions can't be answered with a single retrieval step. "Who was the president when the tallest building in the world was completed?" requires first finding out *which* building, *when* it was completed, and *then* looking up who was president at that time. This is **multi-hop reasoning**, and it is where simple RAG pipelines fall apart.

DSPy makes multi-hop RAG surprisingly clean. Instead of orchestrating a brittle chain of API calls and prompt templates, you define a module that iteratively retrieves and reasons, and then let the optimizer tune the entire pipeline end-to-end.

---

## What You'll Learn

- What multi-hop reasoning is and when you need it
- Building an iterative retrieve-then-reason loop in a `dspy.Module`
- Query decomposition: breaking complex questions into sub-questions
- Hop validation: ensuring each retrieval step adds new information
- Using assertions to prevent redundant hops
- Evaluating multi-hop pipelines on complex questions

---

## Prerequisites

- Completed [5.2: Building RAG Pipelines](../5.2-building-rag/blog.md)
- Familiarity with `dspy.ChainOfThought` and `dspy.Suggest`/`dspy.Assert`

---

## When Single-Hop Isn't Enough

Consider these questions:

| Question | Hops Required |
|----------|--------------|
| "What is the capital of France?" | 1 (single fact lookup) |
| "What river flows through the capital of France?" | 2 (find capital, then find river) |
| "Who designed the most famous bridge over the river that flows through the French capital?" | 3 (capital, river, bridge, designer) |

Single-hop RAG retrieves passages once and generates an answer. For multi-fact questions, it often hallucinates or gives incomplete answers because the first retrieval doesn't contain all the necessary information.

Multi-hop RAG solves this with an **iterative loop**: retrieve, reason, generate a follow-up query, retrieve again, and then answer.

---

## Building a Multi-Hop Module

Here's a complete multi-hop RAG module that performs up to `max_hops` rounds of retrieval:

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Set up retrieval: using ColBERTv2 for Wikipedia
search = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")


class GenerateSearchQuery(dspy.Signature):
    context: list[str] = dspy.InputField(desc="previously retrieved passages")
    question: str = dspy.InputField(desc="the original question to answer")
    search_query: str = dspy.OutputField(desc="a focused search query for the next hop")


class MultiHopRAG(dspy.Module):
    """Multi-hop RAG that iteratively retrieves and reasons."""

    def __init__(self, max_hops: int = 3, passages_per_hop: int = 3):
        self.max_hops = max_hops
        self.generate_query = [
            dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)
        ]
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        context = []

        for hop in range(self.max_hops):
            # Generate a search query based on what we know so far
            query = self.generate_query[hop](
                context=context,
                question=question,
            ).search_query

            # Retrieve new passages
            results = search(query, k=3)
            passages = [r.long_text for r in results]
            context = deduplicate(context + passages)

        # Final response using all accumulated context
        return self.respond(context=context, question=question)


def deduplicate(passages: list[str]) -> list[str]:
    """Remove duplicate passages while preserving order."""
    seen = set()
    unique = []
    for p in passages:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


# Test it
multihop = MultiHopRAG(max_hops=2)
result = multihop(question="What river flows through the city where the Eiffel Tower is located?")
print(f"Response: {result.response}")
```

### How It Works

1.  **Hop 1:** The model sees an empty context and the original question. It generates a search query (e.g., "Where is the Eiffel Tower located?") and retrieves passages about Paris.
2.  **Hop 2:** The model now has context about Paris. It generates a follow-up query (e.g., "What river flows through Paris?") and retrieves passages about the Seine.
3.  **Final response:** With context from both hops, the model answers: "The Seine river flows through Paris, where the Eiffel Tower is located."

Each hop has its **own predictor** (`self.generate_query[hop]`), which means the optimizer can tune each hop independently with different instructions and demonstrations for the first query versus subsequent queries.

---

## Query Decomposition

An alternative to iterative hopping is to decompose the question upfront into sub-questions, then answer each one:

```python
class DecomposeAndAnswer(dspy.Module):
    """Decompose a complex question into sub-questions, answer each, then synthesize."""

    def __init__(self):
        self.decompose = dspy.ChainOfThought(
            "question -> sub_questions: list[str]"
        )
        self.answer_sub = dspy.ChainOfThought(
            "context, question -> answer"
        )
        self.synthesize = dspy.ChainOfThought(
            "question, sub_answers: list[str] -> response"
        )

    def forward(self, question):
        # Step 1: Break into sub-questions
        sub_questions = self.decompose(question=question).sub_questions

        # Step 2: Answer each sub-question with retrieval
        sub_answers = []
        for sq in sub_questions:
            results = search(sq, k=3)
            context = [r.long_text for r in results]
            answer = self.answer_sub(context=context, question=sq).answer
            sub_answers.append(f"Q: {sq} → A: {answer}")

        # Step 3: Synthesize the final response
        return self.synthesize(question=question, sub_answers=sub_answers)


decompose = DecomposeAndAnswer()
result = decompose(
    question="Which country is larger: the country where the Taj Mahal is located, or the country where Machu Picchu is located?"
)
print(f"Response: {result.response}")
```

The decomposition approach works well when the sub-questions are **independent** (you can even parallelize the retrieval). The iterative approach works better when each hop **depends** on the previous one.

---

## Hop Validation with Assertions

A common problem with multi-hop RAG is that the model generates redundant queries that retrieve the same passages. You can use `dspy.Suggest` to nudge the model toward novel queries:

```python
class ValidatedMultiHopRAG(dspy.Module):
    """Multi-hop RAG with assertions to prevent redundant hops."""

    def __init__(self, max_hops: int = 3):
        self.max_hops = max_hops
        self.generate_query = [
            dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)
        ]
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        context = []
        previous_queries = []

        for hop in range(self.max_hops):
            query = self.generate_query[hop](
                context=context,
                question=question,
            ).search_query

            # Validate: query should be different from previous queries
            dspy.Suggest(
                query not in previous_queries,
                f"Search query '{query}' was already used. Generate a different query "
                f"that seeks NEW information not already in the context.",
            )
            previous_queries.append(query)

            # Retrieve
            results = search(query, k=3)
            passages = [r.long_text for r in results]

            # Validate: new passages should add information
            new_passages = [p for p in passages if p not in context]
            dspy.Suggest(
                len(new_passages) > 0,
                "The retrieval returned no new information. Try a more specific "
                "or different search query.",
            )

            context = deduplicate(context + passages)

        return self.respond(context=context, question=question)
```

`dspy.Suggest` is a soft constraint. If the suggestion fails, DSPy retries the predictor with the feedback message. This self-correcting loop prevents the model from wasting hops on duplicate retrievals.

---

## Evaluating Multi-Hop RAG

Multi-hop questions need datasets that actually require multiple reasoning steps. HotPotQA is the classic benchmark:

```python
# Load HotPotQA examples
trainset = [
    dspy.Example(
        question="What is the nationality of the architect who designed the building where the UN General Assembly meets?",
        response="Brazilian",
    ).with_inputs("question"),
    dspy.Example(
        question="In what year was the university founded where the inventor of the World Wide Web studied?",
        response="1209",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the population of the city where the headquarters of the company that makes the iPhone is located?",
        response="About 66,000 (Cupertino)",
    ).with_inputs("question"),
    # ... more examples
]

# Evaluate
metric = dspy.SemanticF1()
evaluate = dspy.Evaluate(devset=trainset, metric=metric, num_threads=4, display_progress=True)

multihop = MultiHopRAG(max_hops=2)
score = evaluate(multihop)
print(f"Multi-Hop RAG SemanticF1: {score:.1f}%")
```

You can then optimize with MIPROv2 just like a single-hop pipeline. The optimizer will tune the query generation and response predictors across all hops.

---

## Key Takeaways

- **Multi-hop RAG extends standard RAG** by adding a loop: retrieve, read, generate the next query, and retrieve again.
- **Query generation is the key optimization target**: `MIPROv2` learns *how* to search for missing information.
- **Deduplication matters**: filtering out repeated passages prevents the context window from filling up with redundancy.
- **Context handling**: concatenate all passages from all hops into a single context list for the final answer generation.

---

## Next Up

Multi-hop RAG uses a fixed number of hops. But what if the model could *decide* when to retrieve, what to search for, and when it has enough context? That is **agentic RAG**: using `dspy.ReAct` to turn retrieval into a tool the model wields autonomously.

**[5.4: RAG as Agent →](../5.4-rag-as-agent/blog.md)**

---

## Resources

- [DSPy Multi-Hop Search Tutorial](https://dspy.ai/tutorials/multihop_search/)
- [HotPotQA Dataset](https://hotpotqa.github.io/)
- [DSPy Assertions Guide](https://dspy.ai/learn/programming/assertions/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
