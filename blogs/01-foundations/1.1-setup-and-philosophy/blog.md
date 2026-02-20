# 1.1: Setup & Philosophy - Why DSPy Changes Everything

## Introduction

If you've spent any time building applications with large language models, you've probably experienced the **prompt engineering death spiral**: you tweak a prompt to fix one edge case, and three others break. You switch from GPT-4 to Claude, and half your prompts need rewriting. You try to add structured output, and suddenly you're wrestling with regex parsers and retry loops. It's brittle, model-specific, and maddeningly hard to iterate on.

**DSPy changes everything.**

Born out of Stanford NLP research and now boasting over 32,000 GitHub stars, DSPy introduces a radical idea: **stop writing prompts. Start writing programs.** Instead of treating language models as black boxes that you coax with carefully worded instructions, DSPy treats LM calls as **programmable, composable, optimizable modules**, just like functions in regular software engineering.

The result? Your LM applications become modular, testable, portable across models, and most importantly, automatically improvable. In this first blog, we'll set up our development environment and deeply understand the philosophy that makes DSPy a paradigm shift.

---

## What You'll Learn

- Why prompt engineering doesn't scale, and what DSPy does instead
- How to install DSPy and set up a clean development environment with `uv`
- The core philosophy: **Signatures**, **Modules**, and **Optimizers**
- How to configure language models: OpenAI, Anthropic, local models via Ollama
- How DSPy's adapter system formats prompts behind the scenes
- How to inspect and debug LM calls
- How to write your very first DSPy program

---

## Prerequisites

- **Python 3.12+** installed on your system
- **`uv`** package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- An API key for at least one LLM provider (OpenAI, Anthropic, etc.)
- Basic Python knowledge (functions, classes, f-strings)

---

## Setting Up Your Environment

We'll use `uv` for fast, reproducible dependency management. If you haven't installed it yet:

```bash
# Install uv (Windows PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or on macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Now let's create our project:

```bash
# Create a new project directory
mkdir learn-dspy && cd learn-dspy

# Initialize a Python project with uv
uv init
uv venv --python 3.12

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install DSPy
uv add dspy python-dotenv
```

Next, create a `.env` file in your project root to store API keys securely:

```bash
# .env
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```


Verify installation:

```python
import dspy
print(dspy.__version__)  # Should print 3.1.3 or later
```

---

## The Core Philosophy: Programming over Prompting

DSPy's tagline is simple: **"Programming > Prompting."** But what does that actually mean?

Think about it this way. If you're building a neural network in PyTorch, you don't hand-write the gradient computations. You define layers, connect them, specify a loss function, and let the framework handle the optimization. **DSPy does the same thing for language model programs.**

Here's the analogy in full:

| PyTorch | DSPy |
|---------|------|
| Layers (nn.Linear, nn.Conv2d) | Modules (Predict, ChainOfThought) |
| Forward pass | Module composition (pipelines) |
| Loss function | Metric function |
| Optimizer (Adam, SGD) | Optimizer (BootstrapRS, MIPROv2) |
| Backpropagation | Prompt optimization / fine-tuning |

DSPy rests on **three pillars**:

### Pillar 1: Signatures Define WHAT

A **Signature** declares the input/output contract of an LM call: *what* you want the model to do, without specifying *how* to prompt it.

```python
# A signature is just: "input_fields -> output_fields"
"question -> answer"
"document -> summary"
"context, question -> reasoning, answer"
```

That's it. No system prompts, no few-shot examples pasted inline, no "you are a helpful assistant." Just a clean declaration of intent.

### Pillar 2: Modules Define HOW

A **Module** wraps a signature with a strategy for executing it. DSPy ships with several built-in modules:

- `dspy.Predict`: a simple LM call
- `dspy.ChainOfThought`: adds step-by-step reasoning before the answer
- `dspy.ReAct`: an agent loop with tool use
- `dspy.ProgramOfThought`: generates and executes code to find answers

```python
# Same signature, different strategies
simple = dspy.Predict("question -> answer")
reasoned = dspy.ChainOfThought("question -> answer")
```

### Pillar 3: Optimizers Improve Quality Automatically

Here's where DSPy truly shines. Instead of manually tweaking prompts, you define a **metric** (how to measure quality) and let an **optimizer** automatically find better prompts, few-shot examples, or even fine-tuning data.

```python
# Define what "good" means
def accuracy(example, prediction, trace=None):
    return prediction.answer.lower() == example.answer.lower()

# Let DSPy figure out the best way to prompt the model
optimizer = dspy.BootstrapRS(metric=accuracy, num_threads=4)
optimized_program = optimizer.compile(my_program, trainset=train_data)
```

We'll go deep on optimizers in Phase 4. For now, just appreciate the elegance: **you write the program once, and the framework improves it for you.**

---

## Configuring Language Models

DSPy uses a unified `dspy.LM` interface powered by LiteLLM under the hood, giving you access to 100+ providers with a single, consistent API.

### Basic Configuration

```python
import dspy
from dotenv import load_dotenv

load_dotenv()  # Loads OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.

# Configure an OpenAI model
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)
```

The `dspy.configure(lm=lm)` call sets the **default language model globally**. Every DSPy module will use this LM unless you override it.

### Multiple Providers

```python
# OpenAI
lm_openai = dspy.LM("openai/gpt-4o-mini")

# Anthropic
lm_claude = dspy.LM("anthropic/claude-sonnet-4-5-20250929")

# Ollama (local)
lm_local = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434")

# Databricks
lm_db = dspy.LM("databricks/databricks-meta-llama-3-1-70b-instruct")
```

### Thread-Safe LM Switching with `dspy.context`

Need to use a different model for just one part of your pipeline? Don't reconfigure globally; use `dspy.context`:

```python
dspy.configure(lm=lm_openai)  # Default: GPT-4o-mini

# Temporarily switch to Claude for this block only
with dspy.context(lm=lm_claude):
    result = dspy.Predict("question -> answer")(
        question="Explain quantum entanglement."
    )
    print(result.answer)

# Back to GPT-4o-mini outside the context block
```

This is **thread-safe**, so you can use different models in different threads without conflicts. This is critical for production deployments.

### Caching

DSPy caches LM responses **by default** to save money and speed up development. This is fantastic during iteration but can bite you when you want fresh responses:

```python
# Disable cache for a specific call
result = predict(question="What's the weather?", cache=False)

# Use rollout_id for diverse BUT cached outputs
# Each unique rollout_id generates a new cached response
result1 = predict(question="Tell me a joke.", rollout_id=1)
result2 = predict(question="Tell me a joke.", rollout_id=2)
# result1 and result2 will be different jokes, both cached
```

> **Tip:** During development, caching saves you real money. A single `dspy.Evaluate` run might make hundreds of LM calls, caching means you only pay once per unique input.

### Token Usage Tracking

Keep an eye on costs by enabling usage tracking:

```python
lm = dspy.LM("openai/gpt-4o-mini", track_usage=True)
dspy.configure(lm=lm)

# After making some calls...
predict = dspy.Predict("question -> answer")
predict(question="What is DSPy?")

# Check token usage
print(lm.usage)
# {'prompt_tokens': 42, 'completion_tokens': 128, 'total_tokens': 170}
```

### OpenAI Responses API

For applications leveraging OpenAI's newer Responses API:

```python
lm = dspy.LM("openai/gpt-4o-mini", model_type="responses")
dspy.configure(lm=lm)
```

---

## Understanding Adapters

You might be wondering: if we're not writing prompts, who is? The answer is **Adapters**.

An adapter sits between your DSPy modules and the raw LM API. It takes your signature's field names, any instructions, and the input values, and formats them into an actual prompt the model can understand. When the model responds, the adapter parses the output back into structured fields.

### ChatAdapter (Default)

The `ChatAdapter` is DSPy's default. It uses a distinctive marker format to delimit fields:

```
[[ ## question ## ]]
What is the capital of France?

[[ ## answer ## ]]
Paris
```

These `[[ ## field_name ## ]]` markers make it easy for DSPy to reliably parse the model's response back into named fields. You'll see these markers if you inspect the raw prompts DSPy sends.

### JSONAdapter

For models with strong native JSON output support (like GPT-4o with JSON mode), `JSONAdapter` formats the prompt to request JSON responses:

```python
dspy.configure(
    lm=lm,
    adapter=dspy.JSONAdapter()
)
```

This is especially useful when working with typed predictors (covered in Blog 2.1) where you need structured data types like lists, integers, or Pydantic models.

### TwoStepAdapter

The `TwoStepAdapter` first asks the LM to generate a free-form response, then asks it to extract structured fields from that response. It's useful when models struggle with the `ChatAdapter` format:

```python
dspy.configure(
    lm=lm,
    adapter=dspy.TwoStepAdapter()
)
```

> **Common Pitfall:** If you're getting parsing errors with a model (especially smaller or local models), try switching to `JSONAdapter` or `TwoStepAdapter` before assuming your signature is wrong.

---

## Inspecting LM Calls

When things don't work as expected, you need to see what DSPy is actually sending to and receiving from the LM. DSPy gives you two tools for this:

### `dspy.inspect_history`

This prints the last `n` LM interactions in a human-readable format:

```python
predict = dspy.Predict("question -> answer")
predict(question="Why is the sky blue?")

# See the last interaction
dspy.inspect_history(n=1)
```

This will show you the exact system message, user message, and assistant response, including the `[[ ## field_name ## ]]` markers that the adapter generated.

### `lm.history`

For programmatic access to call metadata:

```python
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

predict = dspy.Predict("question -> answer")
predict(question="What is 2 + 2?")

# Access the raw history
last_call = lm.history[-1]
print(last_call["messages"])   # The messages sent
print(last_call["response"])   # The raw response object
```

> **Tip:** Always inspect history when debugging. The most common issues (wrong field names, unexpected formatting, missing context) will become obvious when you see the raw prompt.

---

## Your First DSPy Program

Let's put it all together. Here's a complete, runnable DSPy program:

```python

import dspy
from dotenv import load_dotenv

# 1. Load API keys from .env
load_dotenv()

# 2. Configure the language model
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 3. Create a module with a signature
qa = dspy.Predict("question -> answer")

# 4. Call it like a function
response = qa(question="What is DSPy and why should I use it?")
print(f"Answer: {response.answer}")

# 5. Try ChainOfThought for better reasoning
cot = dspy.ChainOfThought("question -> answer")
response = cot(question="If a train leaves Chicago at 9 AM traveling at 60 mph, "
                        "and another leaves New York at 10 AM traveling at 80 mph, "
                        "when do they meet?")
print(f"\nReasoning: {response.reasoning}")
print(f"Answer: {response.answer}")

# 6. Inspect what happened under the hood
dspy.inspect_history(n=1)
```

Notice a few things:

1. **No prompt was written.** DSPy generated it from the signature `"question -> answer"`.
2. **The output is structured.** You access `response.answer` as an attribute, not by parsing raw text.
3. **`ChainOfThought` added reasoning automatically.** Same signature, richer behavior, just by swapping the module.

This is the power of DSPy: **you declare what you want, compose how it happens, and let the framework handle the rest.**

---

## Code Examples

The full working code for this blog is in [code/01_basic_setup.py](code/01_basic_setup.py). Clone the repo, add your API keys to `.env`, and run it to see DSPy in action.

---

## Key Takeaways

- **DSPy replaces prompt engineering with programming.** You write signatures and modules, not prompts.
- **Signatures are contracts**: they declare inputs and outputs without dictating how the model should respond.
- **Modules are strategies**: `Predict`, `ChainOfThought`, and `ReAct` each bring different prompting strategies to the same signature.
- **Optimizers close the loop**: they automatically find better prompts, demonstrations, and fine-tuning data for your program.
- **`dspy.LM` is provider-agnostic**: switch between OpenAI, Anthropic, and local models with one line of code.
- **Caching is on by default**: great for development, but remember `cache=False` and `rollout_id` when you need fresh or diverse responses.
- **Always inspect history when debugging**: `dspy.inspect_history(n=1)` is your best friend.

---

## Next Up

Now that your environment is set up and you understand the philosophy, it's time to go deep on the most fundamental concept in DSPy: **Signatures**, the contract system that replaces prompts.

**[1.2: Signatures - The Contract System →](../1.2-signatures/blog.md)**

---

## Resources

- [DSPy Official Docs: Getting Started](https://dspy.ai/)
- [DSPy Programming Overview](https://dspy.ai/learn/programming/overview/)
- [Language Model Configuration](https://dspy.ai/learn/programming/language_models/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [LiteLLM Supported Providers](https://docs.litellm.ai/docs/providers)
- [uv Package Manager](https://docs.astral.sh/uv/)