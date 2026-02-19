# 6.1 ReAct Agents in DSPy


## Introduction

Agents are LM programs that don't just generate text. They **take actions**. They read, search, calculate, call APIs, and reason about what to do next. The dominant pattern for building agents is **ReAct** (Reasoning + Acting), which interleaves chain-of-thought reasoning with tool calls in an iterative loop until the model decides it has enough information to answer.

DSPy makes ReAct agents **programmable, composable, and optimizable**. You don't write agent prompts. Instead, you define tools as Python functions, declare a signature, and let `dspy.ReAct` handle the reasoning loop. And because it's a DSPy module, you can optimize it with MIPROv2 just like any other module. The DSPy docs report jumping from **8% to 42% accuracy** on complex agent benchmarks purely through optimization.

---

## What You'll Learn

- How the ReAct pattern works: Thought, Tool Name, Tool Args, Observation
- Building tools as Python functions with docstrings and type hints
- Creating agents with `dspy.ReAct`
- Inspecting agent trajectories for debugging
- Error recovery and iteration control
- Optimizing ReAct agents with MIPROv2 for dramatic accuracy gains

---

## Prerequisites

- Completed [Phase 5: Retrieval & RAG](../../05-retrieval-rag/5.1-retrieval-in-dspy/blog.md)
- DSPy installed (`uv add dspy python-dotenv`)
- An OpenAI API key (or any LiteLLM-supported provider)

---

## The ReAct Pattern

ReAct is a loop. At each iteration, the model produces three things:

1. **Thought**: reasoning about what it knows and what it still needs
2. **Tool Name**: which tool to call (or `finish` to stop)
3. **Tool Args**: the arguments to pass to the chosen tool

The framework executes the tool and feeds the result back as an **Observation**. The model then reasons again with the new information. This loop continues until the model calls `finish` or hits the maximum number of iterations.

```
Thought: I need to find the current weather in Paris.
Tool: get_weather
Args: {"city": "Paris"}
Observation: {"temperature": 18, "condition": "partly cloudy"}

Thought: I now have the weather data. I can answer the question.
Tool: finish
Args: {"answer": "It's 18°C and partly cloudy in Paris."}
```

In DSPy, you don't implement this loop yourself. `dspy.ReAct` handles the entire cycle: prompting, tool dispatch, observation feeding, and termination.

---

## Building Your First Agent

Let's build a ReAct agent with two simple tools: one for weather lookups and one for basic calculations.

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# --- Define tools as plain Python functions ---

def get_weather(city: str) -> str:
    """Get the current weather for a city. Returns temperature and conditions."""
    # In production, this would call a real weather API
    weather_data = {
        "paris": "18°C, partly cloudy",
        "london": "14°C, rainy",
        "tokyo": "25°C, sunny",
        "new york": "22°C, clear skies",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Supports basic arithmetic (+, -, *, /, **)."""
    try:
        # Only allow safe mathematical operations
        allowed_chars = set("0123456789.+-*/() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic arithmetic operations are supported."
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


# --- Build the ReAct agent ---
agent = dspy.ReAct(
    "question -> answer",
    tools=[get_weather, calculate],
    max_iters=5,
)

# --- Use the agent ---
result = agent(question="What's the weather in Paris and Tokyo? What's the temperature difference?")
print(f"Answer: {result.answer}")
```

A few things to note:

- **Tools are plain Python functions.** DSPy uses the function name, docstring, and type hints to tell the LM what's available. No special decorators or registration required.
- **`max_iters=5`** caps the reasoning loop at 5 iterations to prevent runaway agents.
- **The signature `"question -> answer"`** is the same one you'd use with `ChainOfThought`. ReAct simply adds tool-calling capabilities on top.

---

## Tool Design Best Practices

The quality of your tools directly affects agent performance. Follow these guidelines:

```python
# GOOD: Clear name, detailed docstring, typed parameters
def search_database(query: str, max_results: int = 5) -> list[str]:
    """Search the product database for items matching the query.
    Returns a list of product descriptions sorted by relevance."""
    ...

# BAD: Vague name, no docstring, no types
def search(q, n=5):
    ...
```

**Tool design checklist:**

1. **Descriptive function names**: the LM reads the function name to decide which tool to use.
2. **Detailed docstrings**: explain what the tool does, what it returns, and any constraints.
3. **Type hints on all parameters**: DSPy uses them to generate the tool schema.
4. **Simple parameter types**: stick to `str`, `int`, `float`, `bool`. Avoid complex objects.
5. **Return strings or simple types**: the result becomes the observation text.
6. **Handle errors gracefully**: return error messages instead of raising exceptions.

---

## Inspecting Agent Trajectories

Debugging agents requires visibility into every step of the reasoning loop. DSPy provides this through the `trajectory` attribute on the result:

```python
result = agent(question="What's 15% of the temperature in London?")

# Inspect the full trajectory
for step_name, step_value in result.trajectory.items():
    print(f"{step_name}: {step_value}")
```

The trajectory is a dictionary containing each step's thought, tool call, and observation. Typical keys look like:

```
thought_0: I need to find the temperature in London first.
tool_name_0: get_weather
tool_args_0: {"city": "London"}
observation_0: 14°C, rainy
thought_1: The temperature is 14°C. Now I need to calculate 15% of 14.
tool_name_1: calculate
tool_args_1: {"expression": "14 * 0.15"}
observation_1: 2.1
thought_2: 15% of 14°C is 2.1°C. I have my answer.
tool_name_2: finish
tool_args_2: {"answer": "15% of the temperature in London (14°C) is 2.1°C."}
```

This is invaluable for debugging: you can see exactly where the agent went wrong, whether it chose the right tool, and whether the tool returned useful results.

You can also use `dspy.inspect_history(n=1)` to see the raw prompt that was sent to the LM at any step.

---

## Error Recovery

ReAct agents in DSPy have built-in error recovery. If a tool call fails (bad arguments, an exception, or a timeout), the error message is fed back as the observation, and the model gets a chance to retry or try a different approach:

```python
def unreliable_api(query: str) -> str:
    """Call an external API that might fail."""
    import random
    if random.random() < 0.5:
        raise ConnectionError("API temporarily unavailable")
    return f"Results for: {query}"

# The agent handles errors gracefully — the error becomes an observation
# and the model can retry or adjust its approach
agent = dspy.ReAct(
    "question -> answer",
    tools=[unreliable_api],
    max_iters=8,
)
```

When a tool raises an exception, DSPy catches it and returns the error as the observation. The model sees something like `"Error: API temporarily unavailable"` and can reason about whether to retry, use a different tool, or answer with what it has.

---

## Optimizing Agents

This is where DSPy truly shines. Because `dspy.ReAct` is a module, you can optimize it with any DSPy optimizer. The DSPy documentation shows that **MIPROv2 boosted a ReAct agent from 8% to 42% accuracy** on the HotPotQA benchmark, a 5x improvement with zero code changes.

```python
import dspy
from dspy.evaluate import SemanticF1

# Define your evaluation metric
metric = SemanticF1()

# Create a training set (list of dspy.Example objects)
trainset = [
    dspy.Example(
        question="What is the capital of the country where the Eiffel Tower is located?",
        answer="Paris"
    ).with_inputs("question"),
    dspy.Example(
        question="What is 20% of the temperature in Tokyo if it's 25°C?",
        answer="5°C"
    ).with_inputs("question"),
    # ... more examples
]

# Optimize with MIPROv2
optimizer = dspy.MIPROv2(
    metric=metric,
    auto="medium",
    num_threads=4,
)

optimized_agent = optimizer.compile(
    agent,
    trainset=trainset,
    requires_permission_to_run=False,
)

# Save the optimized agent
optimized_agent.save("optimized_agent.json")
```

For complex agent tasks, using a **teacher LM** during optimization can dramatically improve results. The teacher generates high-quality reasoning traces that the student model learns from:

```python
# Use GPT-4o as teacher while keeping gpt-4o-mini as the runtime model
gpt4o = dspy.LM("openai/gpt-4o")

optimized_agent = optimizer.compile(
    agent,
    trainset=trainset,
    requires_permission_to_run=False,
    teacher_settings=dict(lm=gpt4o),
)
```

This pattern is powerful: you pay for GPT-4o during optimization (a one-time cost), but deploy with the cheaper GPT-4o-mini, getting closer to GPT-4o quality at a fraction of the runtime cost.

---

## Key Takeaways

- **`dspy.ReAct` implements the Reasoning + Acting loop**. It iterates through Thought, Tool, and Observation cycles until the agent has enough information.
- **Tools are plain Python functions**. DSPy extracts the name, docstring, and type hints to build tool schemas automatically.
- **`max_iters` controls the loop**. Set it high enough for complex tasks, low enough to prevent runaway costs.
- **`result.trajectory` exposes the full reasoning trace**, essential for debugging agent behavior.
- **Error recovery is built-in**. Exceptions become observations, giving the model a chance to adapt.
- **Optimization works on agents too**. MIPROv2 can dramatically improve agent accuracy (8% to 42% in DSPy's docs).
- **Teacher LMs during optimization**. Use a powerful model to generate traces, then deploy with a cheaper model.

---

## Next Up

You've built your first agent with `dspy.ReAct`. But ReAct isn't the only way to use tools in DSPy. In the next post, we'll explore the `dspy.Tool` class, manual tool handling with `ToolCalls`, native function calling, and async tools, giving you fine-grained control over how your agents interact with the world.

**[6.2: Advanced Tool Use →](../6.2-advanced-tool-use/blog.md)**

---

## Resources

- 📖 [DSPy Agents Tutorial](https://dspy.ai/tutorials/agents/)
- 📖 [DSPy ReAct API Reference](https://dspy.ai/api/modules/ReAct/)
- 📖 [Building Customer Service Agents](https://dspy.ai/tutorials/customer_service_agent/)
- 📖 [MIPROv2 Optimizer](https://dspy.ai/api/optimizers/MIPROv2/)
- 📖 [ReAct Paper (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- 💻 [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
