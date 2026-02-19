# 8.1: Understanding RLM (Recursive Language Models)

## Introduction

A new challenge has emerged in LLM applications: **context management**. As documents, logs, and datasets grow larger, simply stuffing everything into a prompt degrades performance, a phenomenon known as **context rot**. The model's attention becomes diluted across irrelevant information, and quality suffers.

DSPy addresses this with **`dspy.RLM`**, a Recursive Language Model module that takes a fundamentally different approach. Instead of forcing all context into a single prompt, RLM lets the LLM **write Python code** to programmatically explore and process large inputs in a sandboxed REPL. The LLM decides what to look at, how to filter it, and when it has enough information to answer. In this blog, you'll learn what RLM is, how it works, and when to reach for it.

---

## What You'll Learn

- What Recursive Language Models are and how they differ from standard prompting
- How `dspy.RLM` uses a sandboxed Python REPL to explore large contexts
- The key differences between `dspy.RLM`, `dspy.ChainOfThought`, and `dspy.Predict`
- How to configure RLM with `max_iterations`, `sub_lm`, and custom tools
- When to use RLM vs. other DSPy modules

---

## Prerequisites

- Completed Phases 1-7 of the Learn DSPy series
- Familiarity with `dspy.ChainOfThought` (covered in [1.3: First Modules](../../01-foundations/1.3-first-modules/blog.md))
- Deno installed for the sandboxed Python interpreter ([installation guide](https://docs.deno.com/runtime/getting_started/installation/))

---

## What Are Recursive Language Models?

Standard DSPy modules like `dspy.Predict` and `dspy.ChainOfThought` pass the full input context into the LM prompt. For small inputs, this works fine. But for large contexts (long documents, extensive logs, large datasets) performance degrades because the model's attention is spread across too much irrelevant information.

**Recursive Language Models** fix this by introducing an **iterative exploration loop**. Instead of processing everything at once, the LLM:

1. Receives **metadata** about the context (type, length, preview) but not the full content
2. Writes **Python code** to explore specific parts of the data
3. Sees the **code output** and decides what to explore next
4. Repeats until it has enough information, then calls `SUBMIT()` with the final answer

Here's the mental model:

```
Predict:           Full Context → Answer
ChainOfThought:    Full Context → Reasoning → Answer
RLM:               Context Metadata → [Code → Output → Code → Output → ...] → SUBMIT(Answer)
```

The critical difference: RLM **separates the data space from the token space**. The LLM never processes the full context directly. Instead, it dynamically loads only the parts it needs through code execution.

### Why This Matters

When you pass a 100-page document into a standard prompt:

- The model's attention is diluted across irrelevant sections
- Important details get lost in the middle (the "lost in the middle" problem)
- Token costs scale linearly with context size

With RLM, the model programmatically searches for relevant information, reads only what it needs, and processes chunks independently. This produces more reliable answers at lower effective cost for large contexts.

---

## The `dspy.RLM` Module

`dspy.RLM` wraps a DSPy signature, just like `dspy.Predict` or `dspy.ChainOfThought`. The difference is in how it processes the signature: through an iterative REPL loop.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure a strong model (any capable LM works)
lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

# Use RLM for large context processing
rlm = dspy.RLM("context, question -> answer")

# The LLM will write code to explore the context
result = rlm(
    context="...very long document or dataset...",
    question="What is the total revenue mentioned?",
)
print(result.answer)
```

Under the hood, `dspy.RLM` launches a sandboxed Python REPL (using Deno and Pyodide) where the LLM can write and execute Python code. The LLM iteratively explores the input data, uses `llm_query()` for semantic analysis of specific snippets, and calls `SUBMIT()` when it has the final answer.

### How RLM Handles the REPL Loop

When you call an `RLM` module:

1. **DSPy formats the signature** and provides the LLM with metadata about each input field (type, length, preview)
2. **The LLM writes Python code** to explore the data (e.g., `print(context[:2000])`)
3. **The code executes in a sandbox**, and the LLM sees the output
4. **The LLM can call `llm_query(prompt)`** for semantic sub-LLM calls on extracted snippets
5. **Steps 2-4 repeat** until the LLM calls `SUBMIT(output)` with the final answer

```python
# The REPL loop is automatic. You just call it like any module.
rlm = dspy.RLM("document, question -> answer", max_iterations=10)

result = rlm(
    document="...500K+ characters of text...",
    question="What were the key findings from Q3?",
)
print(result.answer)  # The LLM explored the document programmatically

# You can inspect what code the LLM executed
for step in result.trajectory:
    print(f"Code:\n{step['code']}")
    print(f"Output:\n{step['output']}\n")
```

---

## RLM vs. ChainOfThought vs. Predict

This is the most important distinction in Phase 8. These modules represent fundamentally different approaches to processing information.

| Aspect | `dspy.Predict` | `dspy.ChainOfThought` | `dspy.RLM` |
|--------|---------------|----------------------|-------------|
| **Mechanism** | Direct input-to-output | Prompted step-by-step reasoning | Iterative REPL code execution |
| **Context handling** | Full context in prompt | Full context in prompt | Metadata only; explores via code |
| **Best for** | Simple, small inputs | Moderate reasoning tasks | Large contexts, data exploration |
| **Model requirement** | Any LM | Any LM | Any strong LM + Deno installed |
| **Output** | Prediction fields only | Prediction + `rationale` | Prediction + `trajectory` + `final_reasoning` |
| **Token efficiency** | Scales with context | Scales with context | Only processes relevant parts |
| **Latency** | Fast | Moderate | Slower (multiple iterations) |

### When They Produce Different Results

For small inputs, all three produce similar answers. The gap widens with large, complex contexts:

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

large_document = open("annual_report.txt").read()  # 200K+ characters

# Predict: stuffs everything into the prompt (may lose details)
predict = dspy.Predict("document, question -> answer")
predict_result = predict(document=large_document, question="What was Q3 revenue?")

# ChainOfThought: reasons over full context (still may miss details)
cot = dspy.ChainOfThought("document, question -> answer")
cot_result = cot(document=large_document, question="What was Q3 revenue?")

# RLM: programmatically searches for the answer (more reliable)
rlm = dspy.RLM("document, question -> answer", max_iterations=15)
rlm_result = rlm(document=large_document, question="What was Q3 revenue?")
```

On large documents, RLM consistently outperforms because it can search, filter, and focus on the relevant sections rather than relying on the model's attention to find a needle in a haystack.

---

## Configuration and Usage

### Constructor Parameters

```python
dspy.RLM(
    signature,              # Signature string or class
    max_iterations=20,      # Max REPL iterations before stopping
    max_llm_calls=50,       # Max LLM calls (main + sub-LLM)
    max_output_chars=10000, # Max chars in final output
    verbose=False,          # Print REPL steps in real-time
    tools=None,             # Custom tools available in the REPL
    sub_lm=None,            # Cheaper LM for llm_query() calls
    interpreter=None,       # Custom code interpreter
)
```

### Basic Usage Pattern

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

# Simple usage
qa = dspy.RLM("context, question -> answer")
result = qa(
    context="...large text...",
    question="Summarize the key findings",
)
print(result.answer)
```

### Using a Cheaper Sub-LM

The `sub_lm` parameter lets the LLM delegate semantic analysis to a cheaper model, reducing cost while the main LM handles strategy:

```python
main_lm = dspy.LM("openai/gpt-4o")
cheap_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=main_lm)

# Main LM decides strategy; sub-LM handles extraction via llm_query()
rlm = dspy.RLM("data, query -> summary", sub_lm=cheap_lm)
```

### In a Custom Module

`dspy.RLM` works like other DSPy modules inside custom programs:

```python
class DocumentQA(dspy.Module):
    def __init__(self):
        self.analyze = dspy.RLM(
            "document, question -> answer",
            max_iterations=15,
        )

    def forward(self, document: str, question: str):
        return self.analyze(document=document, question=question)

qa = DocumentQA()
result = qa(document="...long report...", question="What are the risks?")
print(result.answer)
```

### Built-in Tools in the REPL

Inside the REPL, the LLM has access to:

- `llm_query(prompt)`: Make a sub-LLM call for semantic analysis
- `llm_query_batched(prompts)`: Batch sub-LLM calls
- `print()`: View intermediate results
- `SUBMIT(...)`: Return the final answer
- Standard libraries: `re`, `json`, `collections`, `math`

### Custom Tools

You can provide additional tools that the LLM can call inside the REPL:

```python
def fetch_metadata(doc_id: str) -> str:
    """Fetch metadata for a document ID."""
    return database.get_metadata(doc_id)

rlm = dspy.RLM(
    "documents, query -> answer",
    tools=[fetch_metadata],
)
```

---

## When to Use RLM

**Use `dspy.RLM` when:**

- Your context is too large to fit effectively in a single prompt
- The task benefits from programmatic exploration (searching, filtering, aggregating)
- You need the LLM to decide how to decompose the problem, not you
- You're working with structured or semi-structured data (logs, tables, reports)
- Standard prompting produces unreliable results due to context rot

**Stick with `dspy.ChainOfThought` when:**

- The context fits comfortably in the model's window
- The task requires step-by-step reasoning over the full input
- You want visible rationale for each prediction
- Latency is critical (RLM is slower due to multiple iterations)

**Stick with `dspy.Predict` when:**

- The task is simple and direct
- No reasoning or exploration is needed
- You want minimal latency and token usage

---

## Key Takeaways

1. **Recursive Language Models** explore large contexts through an iterative REPL loop, where the LLM writes Python code to search, filter, and analyze data programmatically.

2. **`dspy.RLM`** is DSPy's module for handling contexts too large for standard prompting. It's not a drop-in replacement for `Predict` or `ChainOfThought`; it's a different paradigm for a different problem.

3. **RLM separates data from tokens**: the LLM sees metadata about the context and dynamically loads only what it needs, avoiding context rot.

4. **Use `sub_lm` for cost control**: let the main LM decide strategy while a cheaper sub-LM handles extraction via `llm_query()`.

5. **Requires Deno**: RLM uses a sandboxed Python interpreter (Pyodide WASM via Deno) for secure code execution.

---

## Next Up

Now that you understand what RLM is and when to use it, it's time to build with it. In the next blog, we'll tackle practical patterns: processing large documents, combining RLM with retrieval, and measuring the cost-quality tradeoffs.

[8.2: Building with RLM →](../8.2-building-with-rlm/blog.md)

---

## Resources

- 📖 [DSPy RLM API Docs](https://dspy.ai/api/modules/RLM/)
- 📖 [DSPy Module Documentation](https://dspy.ai/learn/programming/modules/)
- 📖 [Deno Installation Guide](https://docs.deno.com/runtime/getting_started/installation/)
- 📂 [Blog code samples](code/)
