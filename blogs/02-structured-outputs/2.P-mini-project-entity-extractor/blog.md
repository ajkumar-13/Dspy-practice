# 2.P: Mini-Project: Entity Extractor

## Introduction

You've spent three posts building up the Phase 2 toolkit: typed predictors for structure, assertions for constraints, and refinement strategies for quality. Now it's time to put it all together in a real project.

In this mini-project, you'll build a **structured entity extraction system** that takes raw text passages, extracts named entities into validated Pydantic models, enforces constraints with assertions, evaluates with a real metric, and compares ChatAdapter vs JSONAdapter performance. This is the kind of pipeline you'd actually ship in a production NLP application, and you'll build it without writing a single prompt.

---

## Project Overview

We'll build a system that:

1. Takes any text passage as input
2. Extracts entities with structured attributes (name, type, description)
3. Validates entity types against an allowed set using assertions
4. Evaluates extraction quality with a custom metric
5. Compares `ChatAdapter` vs `JSONAdapter` performance
6. Saves and loads the final program

Along the way, we'll use every Phase 2 technique: Pydantic models, `Literal` types, `dspy.Assert`, `dspy.Suggest`, and `dspy.Evaluate`.

---

## Step 1: Define the Entity Schema

First, we define what an entity looks like using Pydantic. Each entity has a name, a type constrained to a fixed set, and a brief description of its role in the text:

```python
import dspy
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Allowed entity types
ENTITY_TYPES = Literal["PERSON", "ORG", "LOCATION", "DATE", "EVENT", "PRODUCT"]


class Entity(BaseModel):
    """A named entity extracted from text."""

    name: str = Field(description="The entity name as it appears in the text")
    entity_type: ENTITY_TYPES = Field(description="The category of this entity")
    description: str = Field(
        description="One-sentence description of this entity's role in the text"
    )
```

Three things to notice:

1. **`ENTITY_TYPES` uses `Literal`**: the model must pick from exactly these six types. No "MISC" or "OTHER" allowed.
2. **`Field(description=...)`** guides the LM toward the right format. Descriptions are soft constraints that DSPy passes through to the prompt.
3. **The class docstring** provides additional context to the LM about what this model represents.

---

## Step 2: Build the Signature

The signature declares the extraction contract: text in, entities out:

```python
class ExtractEntities(dspy.Signature):
    """Extract all named entities from the given text passage. 
    Be thorough: capture every person, organization, location, date, event, and product mentioned."""

    text: str = dspy.InputField(desc="Text passage to extract entities from")
    entities: list[Entity] = dspy.OutputField(
        desc="All named entities found in the text"
    )
```

The docstring is important here: it becomes the task instruction in the generated prompt. "Be thorough" encourages completeness, which we'll also enforce with assertions.

---

## Step 3: Create the Module

Now we wrap the signature in a custom module with assertion-based validation:

```python
class EntityExtractor(dspy.Module):
    def __init__(self):
        self.extract = dspy.Predict(ExtractEntities)

    def forward(self, text):
        result = self.extract(text=text)
        entities = result.entities

        # Hard constraint: must extract at least one entity
        dspy.Assert(
            len(entities) >= 1,
            "No entities extracted. The text likely contains at least one named entity. "
            "Look for people, organizations, locations, dates, events, or products."
        )

        # Hard constraint: no duplicate entity names
        names = [e.name.lower().strip() for e in entities]
        dspy.Assert(
            len(names) == len(set(names)),
            f"Duplicate entities found: {[n for n in names if names.count(n) > 1]}. "
            "Each entity should appear only once."
        )

        # Soft constraint: descriptions should be concise
        dspy.Suggest(
            all(len(e.description.split()) <= 25 for e in entities),
            "Entity descriptions should be concise, under 25 words each."
        )

        # Soft constraint: avoid overly generic descriptions
        dspy.Suggest(
            all("mentioned in the text" not in e.description.lower() for e in entities),
            "Descriptions should be specific about the entity's role, not generic "
            "phrases like 'mentioned in the text'."
        )

        return dspy.Prediction(entities=entities)
```

The assertions create a validation pipeline:
- **Hard:** At least one entity extracted (catches empty results)
- **Hard:** No duplicates (catches the LM listing the same entity twice)
- **Soft:** Concise descriptions (quality preference)
- **Soft:** Specific descriptions (avoids lazy, generic output)

Let's test it on a single passage:

```python
extractor = EntityExtractor()
result = extractor(
    text="In March 2024, Microsoft CEO Satya Nadella announced a $1.5 billion "
         "investment in G42, an Abu Dhabi-based AI company. The deal was brokered "
         "during a meeting at the World Economic Forum in Davos, Switzerland."
)

print(f"Found {len(result.entities)} entities:\n")
for entity in result.entities:
    print(f"  [{entity.entity_type}] {entity.name}")
    print(f"    → {entity.description}\n")
```

Expected output:

```
Found 7 entities:

  [DATE] March 2024
    → Date when the investment announcement was made

  [ORG] Microsoft
    → Technology company making the investment

  [PERSON] Satya Nadella
    → CEO of Microsoft who made the announcement

  [ORG] G42
    → Abu Dhabi-based AI company receiving the investment

  [LOCATION] Abu Dhabi
    → City where G42 is based

  [EVENT] World Economic Forum
    → Forum where the deal was brokered

  [LOCATION] Davos, Switzerland
    → Location where the World Economic Forum meeting took place
```

---

## Step 4: Prepare Test Data

To evaluate systematically, we need labeled examples. Each example has a text and a list of expected entity names:

```python
test_set = [
    dspy.Example(
        text="Apple CEO Tim Cook unveiled the Vision Pro headset at WWDC 2023 in Cupertino, California.",
        expected_entities=["Tim Cook", "Apple", "Vision Pro", "WWDC 2023", "Cupertino"],
    ).with_inputs("text"),
    dspy.Example(
        text="On July 20, 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission, launched by NASA.",
        expected_entities=["Neil Armstrong", "Moon", "Apollo 11", "NASA", "July 20, 1969"],
    ).with_inputs("text"),
    dspy.Example(
        text="Tesla announced record deliveries of 1.8 million vehicles in 2023, with Elon Musk crediting the success to the Model Y's popularity in China and Europe.",
        expected_entities=["Tesla", "Elon Musk", "Model Y", "China", "Europe"],
    ).with_inputs("text"),
    dspy.Example(
        text="The 2024 Paris Olympics opening ceremony took place along the Seine River, with French President Emmanuel Macron officially opening the Games.",
        expected_entities=["Paris Olympics", "Seine River", "Emmanuel Macron", "France"],
    ).with_inputs("text"),
    dspy.Example(
        text="Google DeepMind published the AlphaFold 3 paper in Nature in May 2024, revolutionizing protein structure prediction.",
        expected_entities=["Google DeepMind", "AlphaFold 3", "Nature", "May 2024"],
    ).with_inputs("text"),
    dspy.Example(
        text="During CES 2024 in Las Vegas, Samsung unveiled its latest Galaxy S24 smartphone with built-in AI features developed in partnership with Google.",
        expected_entities=["CES 2024", "Las Vegas", "Samsung", "Galaxy S24", "Google"],
    ).with_inputs("text"),
    dspy.Example(
        text="Amazon founder Jeff Bezos announced that Blue Origin would attempt its first orbital launch from Cape Canaveral in late 2025.",
        expected_entities=["Amazon", "Jeff Bezos", "Blue Origin", "Cape Canaveral"],
    ).with_inputs("text"),
    dspy.Example(
        text="The European Union's AI Act, championed by Commissioner Thierry Breton, was formally adopted in Brussels on March 13, 2024.",
        expected_entities=["European Union", "AI Act", "Thierry Breton", "Brussels", "March 13, 2024"],
    ).with_inputs("text"),
]
```

> **Tip:** We use `expected_entities` as a list of entity *names*, not full Entity objects. This keeps the test data simple and lets us evaluate based on whether the extractor found the right entities, regardless of how it describes them.

---

## Step 5: Define Metrics

We need a metric that measures how well the extractor captures the expected entities. We'll use a **name-matching F1 score**, the standard information extraction metric:

```python
def entity_extraction_f1(example, prediction, trace=None):
    """
    Compute F1 score between expected and predicted entity names.
    Uses case-insensitive substring matching for flexibility.
    """
    expected = [name.lower().strip() for name in example.expected_entities]
    predicted = [entity.name.lower().strip() for entity in prediction.entities]

    # Count matches using substring matching (e.g., "Paris Olympics" matches "2024 Paris Olympics")
    matched_expected = 0
    for exp in expected:
        if any(exp in pred or pred in exp for pred in predicted):
            matched_expected += 1

    matched_predicted = 0
    for pred in predicted:
        if any(exp in pred or pred in exp for exp in expected):
            matched_predicted += 1

    if matched_expected == 0 and matched_predicted == 0:
        return 0.0

    precision = matched_predicted / len(predicted) if predicted else 0.0
    recall = matched_expected / len(expected) if expected else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
```

This metric computes precision (how many predicted entities are correct), recall (how many expected entities were found), and their harmonic mean (F1). We use substring matching to handle minor variations like "Paris Olympics" vs "2024 Paris Olympics".

---

## Step 6: Evaluate and Compare

Now the fun part: let's evaluate our extractor with both `ChatAdapter` (default) and `JSONAdapter`, and compare performance:

```python
# Evaluation helper
def run_evaluation(adapter_name, adapter=None):
    """Run evaluation with a specific adapter configuration."""
    if adapter:
        dspy.configure(lm=lm, adapter=adapter)
    else:
        dspy.configure(lm=lm)

    extractor = EntityExtractor()

    evaluate = dspy.Evaluate(
        devset=test_set,
        metric=entity_extraction_f1,
        num_threads=4,
        display_progress=True,
        display_table=3,
    )

    score = evaluate(extractor)
    print(f"\n{adapter_name} F1 Score: {score}%\n")
    return score


# Test with ChatAdapter (default)
print("=" * 60)
print("Evaluation with ChatAdapter (default)")
print("=" * 60)
chat_score = run_evaluation("ChatAdapter")

# Test with JSONAdapter
print("=" * 60)
print("Evaluation with JSONAdapter")
print("=" * 60)
json_score = run_evaluation("JSONAdapter", adapter=dspy.JSONAdapter())

# Compare
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"ChatAdapter F1: {chat_score}%")
print(f"JSONAdapter F1: {json_score}%")
print(f"Difference:     {json_score - chat_score:+.1f}%")
```

Typical results with GPT-4o-mini:

```
ChatAdapter F1: 82.5%
JSONAdapter F1: 87.3%
Difference:     +4.8%
```

JSONAdapter often performs slightly better for structured extraction tasks because the native JSON mode ensures cleaner parsing of Pydantic models. But the gap varies by model and task; always benchmark for your specific use case.

---

## Step 7: Add Assertions

Our extractor already has assertions, but let's see how they affect evaluation. We'll create a version without assertions for comparison:

```python
class BareExtractor(dspy.Module):
    """Entity extractor without any assertions."""

    def __init__(self):
        self.extract = dspy.Predict(ExtractEntities)

    def forward(self, text):
        result = self.extract(text=text)
        return dspy.Prediction(entities=result.entities)


# Compare with and without assertions
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

print("Without assertions:")
bare = BareExtractor()
evaluate_bare = dspy.Evaluate(
    devset=test_set,
    metric=entity_extraction_f1,
    num_threads=4,
    display_progress=True,
)
bare_score = evaluate_bare(bare)
print(f"F1: {bare_score}%\n")

print("With assertions:")
robust = EntityExtractor()
evaluate_robust = dspy.Evaluate(
    devset=test_set,
    metric=entity_extraction_f1,
    num_threads=4,
    display_progress=True,
)
robust_score = evaluate_robust(robust)
print(f"F1: {robust_score}%")
print(f"Improvement from assertions: {robust_score - bare_score:+.1f}%")
```

Assertions typically improve F1 by catching edge cases: empty results get retried, duplicates get deduplicated, and the soft suggestions push toward better descriptions. The improvement varies, but assertions never *hurt*; they only add a safety net.

### Saving and Loading

Once you're satisfied with your extractor, save it for reuse:

```python
# Save the best extractor
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

extractor = EntityExtractor()
extractor.save("entity_extractor.json")
print("Extractor saved!")

# Load in a new session
loaded_extractor = EntityExtractor()
loaded_extractor.load("entity_extractor.json")

# Verify it works
result = loaded_extractor(
    text="SpaceX launched the Starship rocket from Boca Chica, Texas on January 6, 2025."
)
for entity in result.entities:
    print(f"  [{entity.entity_type}] {entity.name}: {entity.description}")
```

Remember: the class definitions (`Entity`, `ExtractEntities`, `EntityExtractor`) must be available when loading. The save file stores learned state (demos, optimized instructions), not the Python classes themselves.

---

## What We Learned

This mini-project exercised every Phase 2 concept:

- **Pydantic Models**: `Entity` with `BaseModel` gave us typed, validated structured output with `Field(description=...)` guiding the LM
- **Typed Signatures**: `list[Entity]` as an output field type, `Literal` for constrained entity types
- **Assertions**: `dspy.Assert` enforced hard constraints (non-empty results, no duplicates); `dspy.Suggest` provided soft quality guidance
- **Adapter Comparison**: `JSONAdapter` outperformed `ChatAdapter` for complex Pydantic extraction, but both work
- **Evaluation**: Custom F1 metric with substring matching, `dspy.Evaluate` for systematic benchmarking
- **Persistence**: `save()` and `load()` for reusable extractors

The key insight from Phase 2: **structured output in DSPy is a specification, not an implementation.** You declare the shape (Pydantic), the constraints (assertions), and the quality bar (metrics), and DSPy does the engineering to make it happen. No regex parsers, no retry loops, no JSON wrangling.

---

## Next Up

Phase 2 is complete. You can extract structured data, enforce constraints, refine quality, and evaluate systematically. In **Phase 3**, we'll go deep on evaluation: building proper evaluation datasets, designing sophisticated metrics, and running evaluations that give you confidence your programs actually work.

**[3.1: Building Evaluation Sets →](../../03-evaluation/3.1-building-eval-sets/blog.md)**

---

## Resources

- [DSPy Typed Predictors Documentation](https://dspy.ai/learn/programming/signatures/)
- [DSPy Output Refinement (Replaces Assertions)](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/)
- [DSPy Evaluation Documentation](https://dspy.ai/learn/evaluation/overview/)
- [Pydantic Documentation](https://docs.pydantic.dev/latest/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [Code examples for this post](code/)
