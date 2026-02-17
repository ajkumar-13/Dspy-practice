# 1.3: First Modules - Predict, CoT, PoT, and ReAct

## Introduction

In the previous post, you learned that **signatures** declare *what* an LM should do, the inputs, outputs, and types. But a signature alone doesn't decide *how* the LM approaches the task. Should it answer directly? Think step by step? Write code? Use external tools?

That's the job of **modules**.

A DSPy module wraps a signature with a **strategy** for executing it. The same signature, `"question -> answer"`, produces fundamentally different behavior depending on which module you use. `Predict` fires a single LM call. `ChainOfThought` forces step-by-step reasoning before answering. `ProgramOfThought` generates and runs Python code. `ReAct` orchestrates an iterative loop of reasoning and tool use.

This separation of *what* from *how* is DSPy's core design insight. It means you can swap strategies without rewriting your logic, optimize each strategy independently, and compose them into sophisticated pipelines.

Let's meet the modules.

---

## What You'll Learn

- **`dspy.Predict`**: direct LM call, the foundation of everything
- **`dspy.ChainOfThought`**: adds a `reasoning` field for step-by-step thinking
- **`dspy.ProgramOfThought`**: generates and executes Python code to compute answers
- **`dspy.ReAct`**: iterative reasoning with tool calls
- **Other built-in modules**: `MultiChainComparison`, `Parallel`, `BestOfN`, `Refine`, `RLM`, `CodeAct`
- How to **compare module outputs** side by side on the same task
- How to **compose modules** inside a `dspy.Module` subclass

---

## Prerequisites

- Completed [1.2: Signatures - The Contract System](../1.2-signatures/blog.md)
- DSPy installed (`uv add dspy`)
- A configured language model (we'll use `openai/gpt-4o-mini` in examples)

---

## The Module Hierarchy

Every built-in module follows the same pattern: it takes a signature (inline string or class) and returns a callable that produces a `dspy.Prediction`. The hierarchy looks like this:

```
dspy.Module (base class)
├── dspy.Predict                 - direct call
├── dspy.ChainOfThought          - adds reasoning before output
├── dspy.ProgramOfThought        - generates code to compute output
├── dspy.ReAct                   - iterative reasoning + tool use
├── dspy.MultiChainComparison    - multiple CoT paths, then compare
├── dspy.Parallel                - runs multiple instances concurrently
├── dspy.BestOfN                 - generates N candidates, picks the best
├── dspy.Refine                  - iteratively refines output via feedback
├── dspy.RLM                     - recursive reasoning via sandboxed Python REPL
└── dspy.CodeAct                 - code-based action agent
```

The key insight: **all of these are interchangeable**. Any signature that works with `Predict` works with `ChainOfThought`, `ProgramOfThought`, or any other module. You choose the module based on how much reasoning power you need and not based on the task structure.

Let's set up our environment before diving in:

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)
```

---

## dspy.Predict: The Foundation

`Predict` is the simplest module. It takes your signature, formats it into a prompt via the adapter, sends it to the LM, and parses the response. No tricks, no extra fields, just a clean, direct call.

```python
predict = dspy.Predict("question -> answer")

result = predict(question="What is the speed of light in km/s?")
print(result.answer)
# Approximately 299,792 km/s
```

`Predict` works with class-based signatures too:

```python
class FactCheck(dspy.Signature):
    """Determine if the claim is true or false based on the context."""

    context: str = dspy.InputField(desc="Background information")
    claim: str = dspy.InputField(desc="The claim to verify")
    verdict: bool = dspy.OutputField(desc="True if the claim is supported")

checker = dspy.Predict(FactCheck)
result = checker(
    context="The Eiffel Tower is 330 meters tall and located in Paris, France.",
    claim="The Eiffel Tower is over 300 meters tall."
)
print(result.verdict)  # True
```

**When to use Predict:** Simple, factual tasks where the LM can answer directly like classification, extraction, translation, straightforward Q&A. It's the fastest and cheapest option since it makes exactly one LM call with no extra output fields.

---

## dspy.ChainOfThought: Think Before Answering

`ChainOfThought` (CoT) automatically injects a `reasoning` output field *before* your declared outputs. This forces the LM to articulate its thought process before producing the final answer, a technique that dramatically improves accuracy on tasks requiring logic, math, or multi-step inference.

```python
cot = dspy.ChainOfThought("question -> answer")

result = cot(question="A store sells apples for $2 each. If I buy 3 apples "
                       "and pay with a $20 bill, how much change do I get?")
print(result.reasoning)
# Each apple costs $2, and I'm buying 3 apples. Total cost = 3 × $2 = $6.
# I pay with a $20 bill. Change = $20 - $6 = $14.
print(result.answer)
# $14
```

Notice: you didn't add `reasoning` to your signature, `ChainOfThought` adds it automatically. The `reasoning` field appears on the prediction object alongside your declared outputs.

CoT shines on tasks that benefit from deliberation:

```python
class DiagnosticQA(dspy.Signature):
    """Answer the medical knowledge question with careful reasoning."""

    symptoms: str = dspy.InputField(desc="Patient symptoms described")
    question: str = dspy.InputField(desc="Diagnostic question")
    answer: str = dspy.OutputField(desc="Evidence-based answer")

cot_qa = dspy.ChainOfThought(DiagnosticQA)
result = cot_qa(
    symptoms="Persistent cough, fever for 5 days, shortness of breath",
    question="What conditions should be considered?"
)
print(result.reasoning)  # Step-by-step clinical reasoning
print(result.answer)     # List of differential diagnoses
```

**When to use ChainOfThought:** Reasoning-heavy tasks like math problems, logic puzzles, complex Q&A, multi-step analysis. The extra tokens spent on reasoning usually pay for themselves in answer quality. It costs slightly more than `Predict` (one LM call, but longer output), and the tradeoff is almost always worth it.

---

## dspy.ProgramOfThought: Code-First Reasoning

`ProgramOfThought` (PoT) takes a different approach entirely. Instead of reasoning in natural language, it asks the LM to **generate Python code** that computes the answer. DSPy then executes that code in a sandboxed environment and returns the result.

This is devastating on quantitative tasks where natural language reasoning introduces arithmetic errors:

```python
pot = dspy.ProgramOfThought("question -> answer")

result = pot(question="What is the sum of the first 50 prime numbers?")
print(result.answer)
# 5117
```

Behind the scenes, the LM generated something like:

```python
def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

primes = [n for n in range(2, 300) if is_prime(n)][:50]
answer = sum(primes)
```

DSPy executed this code and extracted the result. No rounding errors, no "approximately", just the correct answer computed deterministically.

```python
# PoT excels on data manipulation tasks
pot = dspy.ProgramOfThought("data_description, question -> answer")

result = pot(
    data_description="A list of temperatures in Celsius: [22, 25, 19, 30, 28, 21, 24]",
    question="What is the standard deviation of these temperatures, rounded to 2 decimal places?"
)
print(result.answer)
# 3.65
```

**When to use ProgramOfThought:** Math, statistics, data processing, algorithmic tasks, anything where code computes a more reliable answer than natural language reasoning. Be aware that it makes multiple LM calls (code generation + potential retries) and requires a Python execution environment.

---

## dspy.ReAct: Reasoning with Tools

`ReAct` implements the **Reasoning + Acting** pattern. It gives the LM access to tools (Python functions) and lets it iteratively reason about what to do, call a tool, observe the result, and decide what to do next. This is DSPy's built-in agent loop.

First, define tools as regular Python functions:

```python
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a topic."""
    # Simplified for demonstration; in practice, call the Wikipedia API
    results = {
        "python programming": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "eiffel tower": "The Eiffel Tower is a wrought-iron lattice tower in Paris, 330 meters tall, built in 1889.",
    }
    return results.get(query.lower(), f"No results found for '{query}'.")

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"
```

Now use them with `ReAct`:

```python
react = dspy.ReAct(
    "question -> answer",
    tools=[search_wikipedia, calculate]
)

result = react(question="How tall is the Eiffel Tower in feet?")
print(result.answer)
# Approximately 1,083 feet
```

Behind the scenes, `ReAct` runs an iterative loop:

1. **Think:** "I need to find the height of the Eiffel Tower."
2. **Act:** Call `search_wikipedia("eiffel tower")` → "...330 meters tall..."
3. **Think:** "Now I need to convert 330 meters to feet."
4. **Act:** Call `calculate("330 * 3.28084")` → "1082.68"
5. **Think:** "I have the answer."
6. **Finish:** Return "Approximately 1,083 feet."

Each tool must be a callable with a **docstring** (DSPy uses it to tell the LM what the tool does) and **type annotations** on its parameters. The `ReAct` module handles the orchestration, parsing tool calls, executing them, and feeding results back.

**When to use ReAct:** Tasks requiring external information or multi-step actions like database lookups, API calls, web search, calculations within a broader reasoning context. It's the most expensive module (multiple LM calls per invocation) but the most capable.

---

## Other Built-in Modules

DSPy ships several more modules beyond the big four. Here's a quick overview:

| Module | What It Does | Use Case |
|--------|-------------|----------|
| `dspy.MultiChainComparison` | Generates multiple chains of thought, then compares them to pick the best answer | Tasks where reasoning can go astray; cross-checking reduces errors |
| `dspy.Parallel` | Runs multiple module instances concurrently | Batch processing, independent sub-tasks |
| `dspy.BestOfN` | Generates N outputs and selects the best one using a metric function | When quality matters more than cost: sample and filter |
| `dspy.Refine` | Iteratively refines output based on feedback from a metric | Outputs that need polish, such as summaries, code, and structured data |
| `dspy.RLM` | Extended reasoning via thinking tokens (reasoning language models) | Deep reasoning with models that support extended thinking |
| `dspy.CodeAct` | Code-based action agent that plans and acts via code generation | Complex tool-use scenarios where code is the action language |

These are more specialized. You'll reach for them when the core four aren't enough, and we'll explore several in depth in later posts.

---

## Comparing Module Outputs

The power of interchangeable modules becomes clear when you run the same signature through each one. Let's compare:

```python
import dspy
from dotenv import load_dotenv

load_dotenv()
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

QUESTION = "If a rectangle has a perimeter of 30 cm and one side is 8 cm, what is its area?"

# --- Predict: Direct answer ---
predict = dspy.Predict("question -> answer")
r1 = predict(question=QUESTION)
print(f"Predict:          {r1.answer}")

# --- ChainOfThought: Reason first ---
cot = dspy.ChainOfThought("question -> answer")
r2 = cot(question=QUESTION)
print(f"ChainOfThought:   {r2.answer}")
print(f"  Reasoning:      {r2.reasoning[:100]}...")

# --- ProgramOfThought: Compute via code ---
pot = dspy.ProgramOfThought("question -> answer")
r3 = pot(question=QUESTION)
print(f"ProgramOfThought: {r3.answer}")
```

Typical output:

```
Predict:          56 cm²
ChainOfThought:   56 cm²
  Reasoning:      Perimeter = 2(l + w) = 30, so l + w = 15. One side is 8 cm, so the other is 7 cm...
ProgramOfThought: 56
```

All three arrive at the correct answer (perimeter 30 → half perimeter 15 → sides 8 and 7 → area 56), but through different strategies. On this simple problem, `Predict` suffices. On harder problems (ambiguous wording, multi-step logic, large numbers), `ChainOfThought` and `ProgramOfThought` pull ahead.

---

## Module Composition

Individual modules are building blocks. The real power comes when you compose them inside a **custom `dspy.Module` subclass**. The pattern is straightforward:

- **`__init__`** declares sub-modules
- **`forward`** defines the control flow

```python
class FactCheckedQA(dspy.Module):
    def __init__(self):
        self.generate_answer = dspy.ChainOfThought("question -> answer")
        self.verify = dspy.ChainOfThought(
            "question, proposed_answer -> is_correct: bool, corrected_answer"
        )

    def forward(self, question):
        # Step 1: Generate an initial answer with reasoning
        initial = self.generate_answer(question=question)

        # Step 2: Verify and potentially correct the answer
        verification = self.verify(
            question=question,
            proposed_answer=initial.answer
        )

        # Return the corrected answer if the original was wrong
        if verification.is_correct:
            return dspy.Prediction(answer=initial.answer)
        else:
            return dspy.Prediction(answer=verification.corrected_answer)

# Use it like any other module
qa = FactCheckedQA()
result = qa(question="What year did the Berlin Wall fall?")
print(result.answer)
# 1989
```

This two-step pipeline first generate, then verify, it's a common pattern. The custom module orchestrates two `ChainOfThought` calls with regular Python logic in between. DSPy can optimize both sub-modules independently, and you can swap `ChainOfThought` for `Predict` or `ProgramOfThought` without changing the pipeline structure.

We'll dive much deeper into custom module architecture in the next post.

---

## Key Takeaways

- **Modules define HOW a signature is executed.** Same signature, different module = different strategy.
- **`Predict`** is the simplest, needs one direct LM call. Use it for straightforward tasks.
- **`ChainOfThought`** adds a `reasoning` field, dramatically improving accuracy on complex tasks.
- **`ProgramOfThought`** generates Python code, giving you exact computation instead of natural language estimation.
- **`ReAct`** orchestrates iterative reasoning with tool calls, it's DSPy's built-in agent pattern.
- **All modules are interchangeable.** You can swap them freely to find the best strategy for your task.
- **Modules compose.** Combine them inside `dspy.Module` subclasses with `__init__` + `forward` to build multi-step pipelines.

---

## Next Up

You've seen how built-in modules work individually. Next, we'll learn how to build **custom modules** i.e. multi-step pipelines that chain LM calls, conditional logic, and data transformations into powerful programs.

**[1.4: Building Custom Modules →](../1.4-custom-modules/blog.md)**

---

## Resources

- [DSPy Modules Documentation](https://dspy.ai/learn/programming/modules/)
- [DSPy ChainOfThought Guide](https://dspy.ai/api/modules/ChainOfThought/)
- [DSPy ReAct Guide](https://dspy.ai/api/modules/ReAct/)
- [ReAct Paper, Yao et al., 2023](https://arxiv.org/abs/2210.03629)
- [Program of Thoughts Paper, Chen et al., 2023](https://arxiv.org/abs/2211.12588)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [Code examples for this post](code/)
