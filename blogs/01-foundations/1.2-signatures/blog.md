# 1.2: Signatures - The Contract System

## Introduction

In traditional LLM development, you spend your time crafting prompts, carefully worded paragraphs of instructions, edge-case handling, and formatting rules. Change the model, and you rewrite the prompt. Change the task slightly, and you patch it with more instructions. It's fragile, ad hoc, and deeply un-engineering.

DSPy's answer is the **Signature**, a clean, declarative contract that says *what* the language model should do without specifying *how* to prompt it. Think of a signature as a **function type hint for AI**: it declares the inputs going in and the outputs you expect back, and DSPy handles everything else (formatting, parsing, type coercion, and even optimization).

If you've used typed function signatures in Python, TypeScript, or Rust, you already have the right mental model. A DSPy Signature is the same idea, applied to language model calls.

In this post, we'll explore every flavor of signature, from one-line inline strings to richly annotated class-based contracts with Pydantic types and multi-modal inputs. By the end, you'll be able to define precise, portable, optimizable contracts for any LM task.

---

## What You'll Learn

- **Inline signatures**: the fastest way to define a contract
- **Class-based signatures**: full control with `InputField`, `OutputField`, and docstring instructions
- **Type hints**: `str`, `int`, `float`, `bool`, `list[str]`, and custom Pydantic models
- **Field descriptions**: how `desc=` subtly guides LM behavior
- **Multi-modal signatures**: using `dspy.Image` for vision tasks
- **Custom Pydantic types**: structured extraction with rich output schemas
- **Signature design patterns**: practical tips from real-world usage

---

## Prerequisites

- Completed [1.1: Setup & Philosophy](../1.1-setup-and-philosophy/blog.md)
- DSPy installed (`uv add dspy`)
- A configured language model (we'll use `openai/gpt-4o-mini` in examples)

---

## Understanding Signatures

Every LM call in DSPy flows through a signature. A signature answers three questions:

1. **What goes in?** The input fields (question, context, document, etc.)
2. **What comes out?** The output fields (answer, summary, score, etc.)
3. **What's the task?** An optional instruction string or docstring

That's the entire contract. DSPy's **adapter system** takes this contract and dynamically constructs the actual prompt sent to the model, complete with formatting, type instructions, and (after optimization) few-shot examples. You never see or manage that prompt directly.

Here's why this matters: because the prompt is generated from a structured contract, DSPy can **optimize** it. It can add examples, rewrite instructions, or adjust formatting, all without you changing a single line of code.

There are two ways to define signatures: **inline** (a string) and **class-based** (a Python class). Let's start simple.

---

## Inline Signatures

Inline signatures are strings with a simple syntax: `"input_field1, input_field2 -> output_field1, output_field2"`. They're perfect for prototyping, one-off calls, and straightforward tasks.

### The Simplest Signature

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# One input, one output, both default to str
predict = dspy.Predict("question -> answer")

result = predict(question="What is the capital of France?")
print(result.answer)
# Paris
```

The field names matter. DSPy uses them as semantic cues when constructing the prompt. A field called `question` signals Q&A behavior; a field called `document` signals summarization or extraction. **Choose descriptive names.**

### Multiple Inputs and Outputs

```python
# Two inputs, two outputs
qa = dspy.Predict("context, question -> reasoning, answer")

result = qa(
    context="The Eiffel Tower was completed in 1889 for the World's Fair.",
    question="When was the Eiffel Tower built?"
)
print(result.reasoning)
# The context states the Eiffel Tower was completed in 1889...
print(result.answer)
# 1889
```

### Typed Inline Signatures

You can append type annotations directly in the string using Python-style hints:

```python
# Typed output: DSPy will parse the response as a float
scorer = dspy.Predict("review -> sentiment_score: float")

result = scorer(review="This product is absolutely wonderful! Best purchase ever.")
print(result.sentiment_score)
# 0.95
print(type(result.sentiment_score))
# <class 'float'>
```

```python
# Boolean output: DSPy resolves this to Yes/No internally
classifier = dspy.Predict("email -> is_spam: bool")

result = classifier(email="Congratulations! You've won a free iPhone. Click here now!")
print(result.is_spam)
# True
```

```python
# List output
extractor = dspy.Predict("sentence -> keywords: list[str]")

result = extractor(sentence="Machine learning and deep learning are subfields of AI.")
print(result.keywords)
# ['machine learning', 'deep learning', 'AI']
```

> **Tip:** Inline signatures default all fields to `str` unless you add a type annotation. For anything beyond simple text in, text out, prefer explicit types.

---

## Class-Based Signatures

When you need more control (adding descriptions, instructions, complex types), graduate to class-based signatures. These subclass `dspy.Signature` and use `dspy.InputField()` and `dspy.OutputField()` to declare fields.

### Basic Class Signature

```python
class QA(dspy.Signature):
    """Answer the question based on the given context."""

    context: str = dspy.InputField(desc="Relevant factual information")
    question: str = dspy.InputField(desc="The user's question")
    answer: str = dspy.OutputField(desc="A concise, factual answer")

qa = dspy.Predict(QA)
result = qa(
    context="Python was created by Guido van Rossum and released in 1991.",
    question="Who created Python?"
)
print(result.answer)
# Guido van Rossum
```

Three things to notice:

1. **The docstring becomes the task instruction.** DSPy injects it into the prompt as the system-level directive. Write it clearly, it's the single most important lever for guiding behavior.
2. **`desc=` provides field-level guidance.** These descriptions tell the LM *what kind of content* each field should contain. They're not prompts, they're metadata that DSPy weaves into the generated prompt.
3. **Type hints are enforced.** DSPy will attempt to parse and validate the LM's output against your declared types.

### Adding Instructions via Docstring

The docstring is your main tool for shaping behavior in class-based signatures:

```python
class Summarizer(dspy.Signature):
    """Summarize the document in exactly 2-3 sentences.
    Focus on the key findings and conclusions.
    Use simple, accessible language."""

    document: str = dspy.InputField(desc="The full text to summarize")
    summary: str = dspy.OutputField(desc="A 2-3 sentence summary")

summarize = dspy.Predict(Summarizer)
result = summarize(document="...")
```

### Inline Signatures with Instructions

You can also attach instructions to inline signatures using the `dspy.Signature()` constructor:

```python
sig = dspy.Signature(
    "document -> summary",
    instructions="Summarize in exactly 2-3 sentences. Use simple language."
)

summarize = dspy.Predict(sig)
result = summarize(document="A long article about climate change...")
print(result.summary)
```

This gives you the brevity of inline syntax with the expressiveness of instructions, a nice middle ground.

---

## Type Resolution

DSPy doesn't just accept type hints, it **resolves** them into prompt formatting and output parsing logic. Here's how each type is handled internally:

| Type | Prompt Behavior | Output Parsing |
|------|----------------|----------------|
| `str` | Open text field | Raw string (default) |
| `int` | Expects integer | Parsed to `int` |
| `float` | Expects decimal number | Parsed to `float` |
| `bool` | Presented as Yes/No | Converted to `True`/`False` |
| `list[str]` | Expects a JSON list | Parsed to Python list |
| `list[int]` | Expects a list of integers | Parsed to list of ints |
| `dict` | Expects a JSON object | Parsed to Python dict |
| Pydantic model | Expects structured JSON matching the schema | Validated against the model |

### Type Resolution in Practice

```python
class MovieAnalysis(dspy.Signature):
    """Analyze a movie and provide structured information."""

    movie_title: str = dspy.InputField(desc="Title of the movie")
    genre: str = dspy.OutputField(desc="Primary genre")
    year: int = dspy.OutputField(desc="Release year")
    rating: float = dspy.OutputField(desc="Rating out of 10")
    is_sequel: bool = dspy.OutputField(desc="Whether it's a sequel")
    themes: list[str] = dspy.OutputField(desc="Key themes explored")

analyze = dspy.Predict(MovieAnalysis)
result = analyze(movie_title="The Dark Knight")

print(result.genre)      # "Action" (str)
print(result.year)       # 2008 (int)
print(result.rating)     # 9.0 (float)
print(result.is_sequel)  # True (bool)
print(result.themes)     # ['justice', 'chaos', 'morality'] (list[str])
```

Every field comes back as the correct Python type. No manual parsing, no `json.loads()`, no regex. DSPy handles it.

---

## Multi-Modal Signatures

DSPy supports vision models through the `dspy.Image` type. You can pass images as inputs alongside text, enabling multi-modal pipelines.

```python
class ImageDescription(dspy.Signature):
    """Describe the contents of the image in detail."""

    image: dspy.Image = dspy.InputField(desc="The image to describe")
    description: str = dspy.OutputField(desc="A detailed description of the image")

# Load an image from a URL
img = dspy.Image(url="https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg")

# Use a vision-capable model
vision_lm = dspy.LM("openai/gpt-4o")
with dspy.context(lm=vision_lm):
    describe = dspy.Predict(ImageDescription)
    result = describe(image=img)
    print(result.description)
```

### Combining Image and Text Inputs

```python
class ImageQA(dspy.Signature):
    """Answer the question about the given image."""

    image: dspy.Image = dspy.InputField(desc="The image to analyze")
    question: str = dspy.InputField(desc="A question about the image")
    answer: str = dspy.OutputField(desc="The answer based on the image")

qa = dspy.Predict(ImageQA)
result = qa(
    image=dspy.Image(url="https://example.com/chart.png"),
    question="What is the trend shown in this chart?"
)
```

> **Tip:** Multi-modal signatures require a vision-capable model (e.g., `gpt-4o`, `claude-sonnet-4-5`). Using them with a text-only model will raise an error.

---

## Custom Types with Pydantic

This is where signatures get truly powerful. You can use **Pydantic models** as output field types, enabling the LM to produce richly structured data that's validated at extraction time.

### Structured Extraction

```python
from pydantic import BaseModel, Field

class ContactInfo(BaseModel):
    name: str = Field(description="Full name of the person")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")
    company: str = Field(description="Company or organization")

class ExtractContact(dspy.Signature):
    """Extract contact information from the text."""

    text: str = dspy.InputField(desc="Text containing contact information")
    contact: ContactInfo = dspy.OutputField(desc="Extracted contact details")

extractor = dspy.Predict(ExtractContact)
result = extractor(
    text="Please reach out to Jane Smith at jane.smith@acme.com or call 555-0123. She's the CTO at Acme Corp."
)

print(result.contact.name)     # Jane Smith
print(result.contact.email)    # jane.smith@acme.com
print(result.contact.phone)    # 555-0123
print(result.contact.company)  # Acme Corp
print(type(result.contact))    # <class 'ContactInfo'>
```

The output is a fully validated Pydantic object. You get autocomplete, type safety, and serialization for free.

### Nested Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import Optional

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Person(BaseModel):
    name: str
    age: int
    address: Optional[Address] = None
    hobbies: list[str] = Field(default_factory=list)

class ExtractPerson(dspy.Signature):
    """Extract structured person information from the biography text."""

    bio: str = dspy.InputField(desc="A short biography")
    person: Person = dspy.OutputField(desc="Structured person data")

extractor = dspy.Predict(ExtractPerson)
result = extractor(
    bio="John Doe, 34, lives at 123 Oak St, Springfield, IL 62704. He enjoys hiking and photography."
)

print(result.person.name)              # John Doe
print(result.person.age)               # 34
print(result.person.address.city)      # Springfield
print(result.person.hobbies)           # ['hiking', 'photography']
```

### Lists of Pydantic Models

```python
class Entity(BaseModel):
    name: str = Field(description="The entity name")
    entity_type: str = Field(description="Type: PERSON, ORG, LOCATION, DATE")
    context: str = Field(description="Brief context about the entity")

class ExtractEntities(dspy.Signature):
    """Extract all named entities from the text."""

    text: str = dspy.InputField()
    entities: list[Entity] = dspy.OutputField(desc="All named entities found")

extractor = dspy.Predict(ExtractEntities)
result = extractor(
    text="Apple CEO Tim Cook announced new products at the September event in Cupertino."
)

for entity in result.entities:
    print(f"{entity.name} ({entity.entity_type}): {entity.context}")
# Apple (ORG): Technology company
# Tim Cook (PERSON): CEO of Apple
# September (DATE): Time of the event
# Cupertino (LOCATION): Location of the event
```

> **Tip:** Pydantic's `Field(description=...)` works alongside DSPy's `desc=`. Use Pydantic descriptions for individual model attributes and DSPy's `desc=` for the output field as a whole.

---

## Signature Design Patterns

After working with many DSPy signatures, some clear patterns emerge. Here are the most useful ones.

### Pattern 1: Separate Reasoning from Answer

```python
class ReasonedAnswer(dspy.Signature):
    """Answer the question step by step."""

    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning process")
    answer: str = dspy.OutputField(desc="Final concise answer")
```

By giving the model a `reasoning` field that comes *before* the `answer` field, you're effectively building Chain-of-Thought into your signature. The model writes out its reasoning first, then produces a more accurate final answer. (DSPy's `ChainOfThought` module does this automatically, but it's useful to understand the mechanism.)

### Pattern 2: Confidence Scoring

```python
class ConfidentClassifier(dspy.Signature):
    """Classify the text and rate your confidence."""

    text: str = dspy.InputField()
    category: str = dspy.OutputField(desc="One of: positive, negative, neutral")
    confidence: float = dspy.OutputField(desc="Confidence score from 0.0 to 1.0")
```

### Pattern 3: Multi-Aspect Analysis

```python
class ReviewAnalysis(dspy.Signature):
    """Analyze a product review across multiple dimensions."""

    review: str = dspy.InputField(desc="The product review text")
    sentiment: str = dspy.OutputField(desc="Overall sentiment: positive, negative, mixed")
    pros: list[str] = dspy.OutputField(desc="List of positive aspects mentioned")
    cons: list[str] = dspy.OutputField(desc="List of negative aspects mentioned")
    purchase_intent: bool = dspy.OutputField(desc="Whether the reviewer would buy again")
    summary: str = dspy.OutputField(desc="One-sentence summary of the review")
```

### Pattern 4: Compositional Signatures

Signatures are first-class objects. You can store them, pass them around, and compose them dynamically:

```python
# Build signatures dynamically based on configuration
def make_classifier(categories: list[str]):
    category_str = ", ".join(categories)
    
    class DynamicClassifier(dspy.Signature):
        f"""Classify the text into one of these categories: {category_str}"""
        text: str = dspy.InputField()
        category: str = dspy.OutputField(desc=f"One of: {category_str}")
        reasoning: str = dspy.OutputField(desc="Why this category was chosen")
    
    return DynamicClassifier

# Create specialized classifiers
topic_sig = make_classifier(["tech", "sports", "politics", "entertainment"])
mood_sig = make_classifier(["happy", "sad", "angry", "neutral"])

topic_classifier = dspy.Predict(topic_sig)
mood_classifier = dspy.Predict(mood_sig)
```

---

## Best Practices

1. **Name fields semantically.** `answer` is better than `output`. `patient_summary` is better than `text`. The field name IS part of the prompt.

2. **Keep docstrings focused.** Write clear, concise instructions. The docstring should describe the task in one to three sentences and not a wall of rules.

3. **Use `desc=` liberally.** Field descriptions are your best tool for guiding output format and content without over-specifying the prompt.

4. **Start inline, graduate to class-based.** Begin with `"question -> answer"` for prototyping. Move to a class when you need descriptions, types, or multiple fields.

5. **Order output fields intentionally.** DSPy generates outputs in the declared order. Put reasoning before answers, and simple fields before complex ones.

6. **Prefer specific types over str.** Use `int`, `float`, `bool`, `list[str]`, or Pydantic models whenever possible. They eliminate parsing bugs and make your code more robust.

7. **Don't over-engineer.** A signature with 10 output fields is probably trying to do too much. Split it into multiple focused signatures composed in a module pipeline.

---

## Key Takeaways

- **Signatures are contracts, not prompts.** They declare what the LM should do (inputs, outputs, types) and DSPy generates the actual prompt.
- **Inline signatures** (`"question -> answer"`) are great for quick prototyping and simple tasks.
- **Class-based signatures** give you docstring instructions, field descriptions via `desc=`, and full type-hint support.
- **Type hints are resolved automatically:** `str` → open text, `bool` → Yes/No, `int`/`float` → numeric parsing, `list` → JSON list, Pydantic → structured JSON.
- **`dspy.Image`** enables multi-modal contracts with vision models.
- **Pydantic models as output types** unlock structured extraction with validation. This is one of DSPy's most powerful features.
- **Signatures are first-class objects**: you can compose, modify, and dynamically generate them.

---

## Next Up

In **[1.3: First Modules - Predict, ChainOfThought & Beyond](../1.3-first-modules/blog.md)**, we'll take these signatures and plug them into DSPy's built-in modules. You'll see how the same signature behaves differently depending on whether you use `Predict`, `ChainOfThought`, or `ProgramOfThought`, and why that separation of *what* from *how* is the core insight of DSPy.


---

## Resources

- [DSPy Signatures Documentation](https://dspy.ai/learn/programming/signatures/)
- [DSPy Typed Predictors Guide](https://dspy.ai/learn/programming/typed_predictors/)
- [Pydantic Model Documentation](https://docs.pydantic.dev/latest/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [Code examples for this post](code/)
