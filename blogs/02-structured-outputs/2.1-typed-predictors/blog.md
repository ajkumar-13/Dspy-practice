# 2.1: Typed Predictors and Structured Output


## Introduction

In Phase 1, you learned the fundamentals: signatures declare inputs and outputs, modules define execution strategies, and custom modules compose them into pipelines. You even used `Literal` types and `float` outputs in the text classifier mini-project. But that was just scratching the surface.

Real applications demand **structured, validated output**, not just strings. You need lists of entities, nested JSON objects, boolean flags, numeric scores, and complex data models. You need to know that your LM returns a *list of exactly three keywords*, not a comma-separated paragraph. You need a *Pydantic model with validated fields*, not a blob of text that might or might not parse.

DSPy integrates deeply with Python's type system and Pydantic to make this effortless. In this post, you'll learn how to use typed output fields, Pydantic models, and the right adapter to get reliable structured output from any language model.

---

## What You'll Learn

- How to use typed output fields: `list[str]`, `list[int]`, `bool`, `float`, `Literal`
- How to use Pydantic `BaseModel` as an output field type for structured extraction
- How to build nested Pydantic models for complex data
- The difference between `ChatAdapter` and `JSONAdapter`, and when to use each
- How `dspy.configure(adapter=dspy.JSONAdapter())` changes prompt formatting
- How type validation and error handling work under the hood
- A practical entity extraction example with fully structured output

---

## Prerequisites

- Completed [Phase 1: Foundations](../../01-foundations/1.1-setup-and-philosophy/blog.md)
- DSPy installed (`uv add dspy pydantic`)
- A configured language model (we'll use `openai/gpt-4o-mini`)

---

## Basic Typed Outputs

You've already seen `Literal` and `float` types in the text classifier project. Let's formalize what DSPy supports out of the box. In a class-based signature, you can annotate any output field with a Python type, and DSPy will instruct the LM to produce that type and validate the result.

```python
import dspy
from dotenv import load_dotenv
from typing import Literal

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


class AnalyzeText(dspy.Signature):
    """Analyze the given text and extract metadata."""

    text: str = dspy.InputField(desc="Text to analyze")
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField(
        desc="Overall sentiment"
    )
    word_count: int = dspy.OutputField(desc="Approximate word count")
    is_question: bool = dspy.OutputField(desc="Whether the text is a question")
    confidence: float = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")


analyze = dspy.Predict(AnalyzeText)
result = analyze(text="I absolutely love this new framework! It makes everything so easy.")

print(f"Sentiment:  {result.sentiment}")    # positive
print(f"Word count: {result.word_count}")   # 11
print(f"Question:   {result.is_question}")  # False
print(f"Confidence: {result.confidence}")   # 0.95
print(f"Types:      {type(result.word_count).__name__}, {type(result.is_question).__name__}")
# Types: int, bool
```

DSPy handles the parsing and type coercion automatically. An `int` field returns an actual Python `int`, not a string. A `bool` field returns `True` or `False`, not `"yes"` or `"no"`. And `Literal` constrains the LM to exactly the options you specify.

### Working with Lists

Lists are one of the most useful typed outputs. Need a list of keywords? A list of extracted names? Just annotate:

```python
class ExtractKeywords(dspy.Signature):
    """Extract the most important keywords from the text."""

    text: str = dspy.InputField()
    keywords: list[str] = dspy.OutputField(desc="5 to 10 important keywords")
    relevance_scores: list[float] = dspy.OutputField(
        desc="Relevance score (0-1) for each keyword, in the same order"
    )


extract = dspy.Predict(ExtractKeywords)
result = extract(text="Machine learning models are transforming healthcare by enabling "
                      "early disease detection through medical imaging analysis.")

print(f"Keywords: {result.keywords}")
# ['machine learning', 'healthcare', 'disease detection', 'medical imaging', ...]
print(f"Scores:   {result.relevance_scores}")
# [0.95, 0.9, 0.88, 0.85, ...]
print(f"Type:     {type(result.keywords)}")
# <class 'list'>
```

DSPy supports `list[str]`, `list[int]`, `list[float]`, and `list[bool]` natively. For more complex list items, you'll want Pydantic models, which we'll cover next.

---

## Pydantic Models as Types

Basic types cover simple cases, but real-world extraction demands structured objects. Imagine extracting entities from a news article: each entity has a name, a type, and a description. You could use three parallel lists, but that's fragile and hard to work with. Instead, define a Pydantic model:

```python
from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str = Field(description="The entity name as it appears in the text")
    entity_type: str = Field(description="Type: PERSON, ORG, LOCATION, DATE, or EVENT")
    description: str = Field(description="Brief one-sentence description of this entity")


class ExtractEntities(dspy.Signature):
    """Extract all named entities from the given text."""

    text: str = dspy.InputField(desc="Text to extract entities from")
    entities: list[Entity] = dspy.OutputField(desc="List of extracted entities")


extract = dspy.Predict(ExtractEntities)
result = extract(
    text="On January 15, 2025, Sundar Pichai announced that Google would "
         "open a new AI research lab in Tokyo, Japan."
)

for entity in result.entities:
    print(f"  {entity.name} ({entity.entity_type}): {entity.description}")
# Sundar Pichai (PERSON): CEO of Google who made the announcement
# Google (ORG): Technology company opening the new lab
# Tokyo (LOCATION): City in Japan where the new lab will be located
# Japan (LOCATION): Country where the new research lab will be opened
# January 15, 2025 (DATE): Date of the announcement
```

Each entity is a proper Pydantic `BaseModel` instance with validated fields. You get autocomplete in your IDE, type checking, and the full power of Pydantic validation.

> **Tip:** Use `Field(description=...)` on your Pydantic model fields. DSPy passes these descriptions to the LM, guiding it toward the right format. Think of descriptions as soft constraints on the model's output.

---

## Nested Structures

Pydantic models can nest arbitrarily. For example, an analysis result might contain a list of entities, each with a list of mentions:

```python
class Mention(BaseModel):
    text: str = Field(description="The exact text span of the mention")
    start_index: int = Field(description="Character offset where the mention starts")


class DetailedEntity(BaseModel):
    name: str = Field(description="Canonical name of the entity")
    entity_type: Literal["PERSON", "ORG", "LOCATION", "DATE"] = Field(
        description="Entity type"
    )
    mentions: list[Mention] = Field(
        description="All mentions of this entity in the text"
    )


class DetailedExtraction(dspy.Signature):
    """Extract entities with all their mentions from the text."""

    text: str = dspy.InputField()
    entities: list[DetailedEntity] = dspy.OutputField(
        desc="Entities with their mentions"
    )


extract = dspy.Predict(DetailedExtraction)
result = extract(
    text="Google announced a partnership with Google DeepMind. "
         "The Google team in London will lead the effort."
)

for entity in result.entities:
    print(f"{entity.name} ({entity.entity_type}):")
    for mention in entity.mentions:
        print(f"  - '{mention.text}' at position {mention.start_index}")
```

Nested structures work naturally. DSPy serializes the Pydantic schema for the LM, and the response is validated against the full nested model. If validation fails, DSPy retries automatically.

---

## Choosing Your Adapter: ChatAdapter vs JSONAdapter

When you use typed outputs and Pydantic models, the **adapter** controls how DSPy communicates the expected output format to the LM. This choice significantly impacts reliability.

### ChatAdapter (Default)

The `ChatAdapter` is DSPy's default. It uses text markers to delimit output fields:

```
[[ ## text ## ]]
On January 15, 2025, Sundar Pichai announced...

[[ ## entities ## ]]
[{"name": "Sundar Pichai", "entity_type": "PERSON", ...}, ...]
```

The LM generates field values between `[[ ## field_name ## ]]` markers, and DSPy parses them. For Pydantic models, the field value is expected as JSON between the markers.

**Strengths:** Works with most models, including smaller and local models. The marker format is intuitive and models rarely get confused by it.

**Weaknesses:** For deeply nested or complex Pydantic models, parsing can be less reliable. The model might produce JSON that isn't perfectly formatted between the markers.

### JSONAdapter

The `JSONAdapter` asks the LM to return its entire response as a single JSON object:

```python
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
```

With `JSONAdapter`, the prompt explicitly instructs the LM to return JSON, and if the model supports native JSON mode (like GPT-4o), DSPy enables it automatically.

```python
# Configure with JSONAdapter
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    adapter=dspy.JSONAdapter(),
)

extract = dspy.Predict(ExtractEntities)
result = extract(
    text="Elon Musk founded SpaceX in Hawthorne, California in 2002."
)

for entity in result.entities:
    print(f"  {entity.name} ({entity.entity_type})")
# Works identically, but the underlying prompt format is JSON
```

**Strengths:** More reliable for complex Pydantic models, nested structures, and lists. Takes advantage of native JSON mode when available.

**Weaknesses:** Some smaller or local models struggle with JSON-only responses. The format gives the model less "thinking room" compared to ChatAdapter's free-form markers.

### When to Use Which?

| Scenario | Recommended Adapter |
|----------|-------------------|
| Simple typed outputs (`str`, `int`, `bool`) | ChatAdapter (default) |
| Complex Pydantic models | JSONAdapter |
| Deeply nested structures | JSONAdapter |
| Smaller/local models (Ollama) | ChatAdapter |
| Models with native JSON mode (GPT-4o, Claude) | JSONAdapter |
| Mixed simple + complex outputs | ChatAdapter (safer default) |

> **Tip:** Start with the default `ChatAdapter`. If you see parsing errors or malformed outputs, switch to `JSONAdapter`. You can also switch adapters per-block using `dspy.context`:
>
> ```python
> with dspy.context(adapter=dspy.JSONAdapter()):
>     result = extract(text="...")
> ```

---

## Error Handling

What happens when the LM returns output that doesn't match your types? DSPy handles this automatically through its retry mechanism:

1. DSPy sends the prompt to the LM
2. The LM response is parsed and validated against your type annotations
3. If validation fails (e.g., the LM returned `"yes"` for a `bool` field), DSPy retries with feedback
4. After multiple failures, DSPy raises an error

You can control retry behavior and catch errors gracefully:

```python
class StrictOutput(dspy.Signature):
    """Extract a precise numeric answer."""

    question: str = dspy.InputField()
    answer: int = dspy.OutputField(desc="The exact integer answer")
    unit: Literal["meters", "kilometers", "miles"] = dspy.OutputField()


predict = dspy.Predict(StrictOutput)

try:
    result = predict(question="How tall is Mount Everest in meters?")
    print(f"{result.answer} {result.unit}")  # 8849 meters
except Exception as e:
    print(f"Failed to get structured output: {e}")
```

In practice, with GPT-4o-mini and similar models, type validation rarely fails for simple types. The retry mechanism handles most transient issues automatically. For maximum reliability with complex models, combine typed predictors with `JSONAdapter` and assertions (covered in the next post).

---

## Practical Example: Entity Extraction Pipeline

Let's put everything together in a complete, practical pipeline that extracts structured entities from a news article:

```python
import dspy
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())


class Entity(BaseModel):
    name: str = Field(description="Entity name")
    category: Literal["PERSON", "ORG", "LOCATION", "DATE", "MONEY"] = Field(
        description="Entity category"
    )
    context: str = Field(description="How this entity is relevant in the text")


class ArticleAnalysis(BaseModel):
    summary: str = Field(description="One-sentence summary of the article")
    entities: list[Entity] = Field(description="All named entities found")
    topic: Literal["politics", "business", "technology", "science", "sports"] = Field(
        description="Primary topic"
    )


class AnalyzeArticle(dspy.Signature):
    """Analyze a news article and extract structured information."""

    article: str = dspy.InputField(desc="The news article text")
    analysis: ArticleAnalysis = dspy.OutputField(desc="Structured analysis result")


class ArticleAnalyzer(dspy.Module):
    def __init__(self):
        self.analyze = dspy.Predict(AnalyzeArticle)

    def forward(self, article):
        result = self.analyze(article=article)
        return dspy.Prediction(analysis=result.analysis)


# Use it
analyzer = ArticleAnalyzer()
result = analyzer(
    article="Amazon announced today that it will invest $4 billion in Anthropic, "
            "the AI safety startup founded by former OpenAI researchers Dario and "
            "Daniela Amodei. The deal, signed in San Francisco, makes Amazon the "
            "largest investor in the Claude chatbot maker."
)

analysis = result.analysis
print(f"Summary: {analysis.summary}")
print(f"Topic:   {analysis.topic}")
print(f"\nEntities:")
for entity in analysis.entities:
    print(f"  {entity.name} ({entity.category}): {entity.context}")
```

This pipeline returns a fully validated `ArticleAnalysis` object with typed fields, constrained enums, and nested entity lists, all from a single LM call. No regex parsing, no JSON wrangling, no retry loops.

---

## Key Takeaways

- **Typed output fields** (`int`, `bool`, `float`, `list[str]`, `Literal`) give you automatic parsing and validation. No manual type coercion needed.
- **Pydantic `BaseModel`** as an output type enables arbitrarily complex structured extraction. Use `Field(description=...)` to guide the LM.
- **Nested Pydantic models** work naturally. DSPy serializes the full schema and validates the response against it.
- **`ChatAdapter`** (default) uses `[[ ## field ## ]]` text markers. **`JSONAdapter`** requests JSON output and enables native JSON mode. Start with ChatAdapter; switch to JSONAdapter for complex structures.
- **Error handling** is automatic. DSPy retries with feedback when type validation fails.
- **The key insight:** structured output in DSPy is a declaration, not an implementation. You declare the shape, DSPy handles the rest.

---

## Next Up

Typed outputs guarantee structure, but what about *semantic* constraints? What if you need the summary to be under 100 words, the entity list to be non-empty, or the confidence score to be above 0.5? That's where **DSPy Assertions** come in: programmatic constraints with automatic retry.

**[2.2: Assertions and Constraints →](../2.2-assertions/blog.md)**

---

## Resources

- [DSPy Typed Predictors Documentation](https://dspy.ai/learn/programming/signatures/)
- [DSPy Adapters Guide](https://dspy.ai/learn/programming/language_models/#setting-up-the-lm-and-adapter)
- [Pydantic Documentation](https://docs.pydantic.dev/latest/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [Code examples for this post](code/)
