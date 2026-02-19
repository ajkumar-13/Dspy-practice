# 10.1: Working with Images and Audio in DSPy

## Introduction

Language models are not just about text anymore. The latest generation of models (GPT-4o, Gemini, and others) can see images, hear audio, and generate speech. But if you have tried building multi-modal applications, you know the pain: every provider has its own API for uploading images, encoding audio, and handling binary data. You end up writing more glue code than logic.

DSPy brings the same clean abstraction to multi-modal inputs that it brought to text. With `dspy.Image` and `dspy.Audio`, you can pass images and audio clips through signatures and modules exactly like you pass strings. The best part? The same optimizers that improve your text pipelines (`MIPROv2`, `BootstrapFewShotWithRandomSearch`) work on multi-modal programs too. No special handling required.

In this blog, you will learn how to use DSPy's multi-modal primitives, configure the right models for each modality, and optimize multi-modal pipelines with the tools you already know.

---

## What You'll Learn

- How to use `dspy.Image` to pass images into DSPy programs
- How to use `dspy.Audio` to pass audio clips into DSPy programs
- Configuring vision-capable and audio-capable language models
- Building image analysis and spoken QA pipelines
- Iterative image prompt refinement with feedback loops
- Loading and preprocessing multi-modal datasets
- Optimizing multi-modal programs with DSPy optimizers
- Practical tips for token costs and caching with multi-modal inputs

---

## Prerequisites

- Completed Phases 1 to 4 (Foundations, Structured Outputs, Evaluation, Optimization)
- An OpenAI API key with access to `gpt-4o-mini` (vision) and `gpt-4o-mini-audio-preview` (audio)
- `uv add dspy python-dotenv numpy` installed
- Basic familiarity with DSPy signatures and modules

---

## The dspy.Image Primitive

DSPy represents images with the `dspy.Image` class. You can create an image from a URL, a local file path, or a base64-encoded string. Once created, you use it in signatures just like any other field type.

### Creating Images

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure a vision-capable LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Create an image from a URL
image = dspy.Image.from_url(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
)

# Alternative constructor (equivalent)
image_alt = dspy.Image(url="https://example.com/photo.jpg")
```

Both approaches produce the same result: a `dspy.Image` object that DSPy can serialize and pass to vision-capable models. Under the hood, DSPy handles the encoding and formatting required by each provider. You never touch base64 strings or multipart uploads.

### Using Images in Signatures

To accept an image as input, declare a field with the `dspy.Image` type in your signature:

```python
class DescribeImage(dspy.Signature):
    """Describe the contents of an image in detail."""

    current_image: dspy.Image = dspy.InputField()
    description: str = dspy.OutputField(desc="a detailed description of the image")


# Use it with Predict
describer = dspy.Predict(DescribeImage)
result = describer(current_image=image)
print(result.description)
```

That is all you need. No special wrappers, no provider-specific code. The signature tells DSPy the shape of the input, the module handles the reasoning strategy, and the adapter formats everything correctly for whichever LM you are using.

### Image Analysis with ChainOfThought

For richer analysis, use `ChainOfThought`. The model will reason step-by-step before producing its answer:

```python
class ImageQA(dspy.Signature):
    """Answer questions about an image."""

    image: dspy.Image = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="concise answer to the question")


qa = dspy.ChainOfThought(ImageQA)
result = qa(
    image=dspy.Image.from_url(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    ),
    question="What animal is in this image, and what is it doing?",
)
print(f"Answer: {result.answer}")
```

### Iterative Image Prompt Refinement

One powerful pattern from the DSPy docs is **iterative prompt refinement for image generation**. The idea: you generate an image from a prompt, then use a vision model to compare the result against your desired prompt and refine it. This creates a feedback loop where the model critiques and improves its own prompts.

```python
class ImagePromptRefiner(dspy.Signature):
    """Compare a generated image against the desired prompt, provide feedback, and
    produce a revised prompt that better captures the desired result."""

    desired_prompt: str = dspy.InputField(desc="what the image should depict")
    current_image: dspy.Image = dspy.InputField(desc="the currently generated image")
    current_prompt: str = dspy.InputField(desc="the prompt used to generate current_image")
    feedback: str = dspy.OutputField(desc="what is wrong or missing in the current image")
    image_strictly_matches_desired_prompt: bool = dspy.OutputField()
    revised_prompt: str = dspy.OutputField(desc="improved prompt for image generation")


refiner = dspy.Predict(ImagePromptRefiner)

# Iterative refinement loop
desired = "A photorealistic red fox sitting in a snowy forest at sunset"
current_prompt = desired
max_iterations = 5

for i in range(max_iterations):
    # In practice, you would call an image generation API here (e.g., FAL/Flux Pro)
    # generated_image = generate_image(current_prompt)
    generated_image = dspy.Image.from_url("https://example.com/generated.jpg")  # placeholder

    result = refiner(
        desired_prompt=desired,
        current_image=generated_image,
        current_prompt=current_prompt,
    )

    print(f"Iteration {i + 1}: {result.feedback}")

    if result.image_strictly_matches_desired_prompt:
        print("Image matches desired prompt!")
        break

    current_prompt = result.revised_prompt
```

This pattern is incredibly useful for automated creative workflows. The vision model acts as a quality gate, and the loop converges on prompts that produce the desired image.

---

## The dspy.Audio Primitive

DSPy handles audio with the `dspy.Audio` class. You can create audio objects from numpy arrays (e.g., loaded from a file or dataset), and pass them through signatures just like text or images.

### Creating Audio Objects

```python
import numpy as np
import dspy
from dotenv import load_dotenv

load_dotenv()

# Create audio from a numpy array
# Typical audio: 16kHz sampling rate, float32 values between -1 and 1
sample_rate = 16000
duration = 2.0  # seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
audio_array = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440Hz sine wave

audio = dspy.Audio.from_array(audio_array, sampling_rate=sample_rate)
```

### Configuring an Audio-Capable LM

Audio input requires a model that supports audio modality. OpenAI provides dedicated audio preview models:

```python
# For audio input tasks (speech-to-text, spoken QA)
audio_lm = dspy.LM("gpt-4o-mini-audio-preview-2024-12-17")
dspy.configure(lm=audio_lm)
```

> **Important:** Standard models like `gpt-4o-mini` do **not** accept audio input. You must use an audio-preview model for audio-based signatures. For audio generation (text-to-speech), use `gpt-4o-mini-tts` via the OpenAI client directly.

### Spoken QA: Answering Questions from Audio

The spoken QA pattern is one of the most practical multi-modal applications. Given an audio clip (a podcast segment, a lecture recording, a voice note), the model answers questions about its content:

```python
class SpokenQASignature(dspy.Signature):
    """Answer the question based on the audio clip."""

    passage_audio: dspy.Audio = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="factoid answer between 1 and 5 words")


spoken_qa = dspy.ChainOfThought(SpokenQASignature)

# In practice, you would load real audio:
# audio = dspy.Audio.from_array(my_audio_array, sampling_rate=16000)
# result = spoken_qa(passage_audio=audio, question="What year was the treaty signed?")
# print(result.answer)
```

This is remarkably clean: the same signature-based approach you use for text QA, but with audio input. DSPy handles all the encoding and API formatting behind the scenes.

### Loading Audio Datasets

For training and evaluation, you will often load audio from HuggingFace datasets. DSPy's `DataLoader` can handle this with preprocessing:

```python
from dspy.utils import DataLoader

dl = DataLoader()

# Load an audio QA dataset from HuggingFace
dataset = dl.from_huggingface(
    "dataset_name",
    split="train",
    input_keys=("audio", "question"),
)

# Preprocess: convert audio arrays to dspy.Audio objects
def preprocess_audio_example(example):
    audio_data = example["audio"]
    example["passage_audio"] = dspy.Audio.from_array(
        np.array(audio_data["array"], dtype=np.float32),
        sampling_rate=audio_data["sampling_rate"],
    )
    return example.with_inputs("passage_audio", "question")


trainset = [preprocess_audio_example(ex) for ex in dataset]
```

---

## Configuring LMs for Multi-Modal Tasks

Different modalities need different models. Here is a quick reference:

| Task | Model | DSPy Configuration |
|------|-------|-------------------|
| Vision (image input) | `gpt-4o-mini` | `dspy.LM("openai/gpt-4o-mini")` |
| Audio input (spoken QA) | `gpt-4o-mini-audio-preview-2024-12-17` | `dspy.LM("gpt-4o-mini-audio-preview-2024-12-17")` |
| Audio generation (TTS) | `gpt-4o-mini-tts` | Via OpenAI client directly |
| Multi-modal (text + image) | `gpt-4o`, `gpt-4o-mini` | `dspy.LM("openai/gpt-4o-mini")` |

You can use different models for different modules in the same pipeline using `dspy.context`:

```python
vision_lm = dspy.LM("openai/gpt-4o-mini")
audio_lm = dspy.LM("gpt-4o-mini-audio-preview-2024-12-17")

# Use vision LM for image tasks
dspy.configure(lm=vision_lm)
image_result = describer(current_image=image)

# Switch to audio LM for audio tasks
with dspy.context(lm=audio_lm):
    audio_result = spoken_qa(passage_audio=audio, question="What was discussed?")
```

---

## Multi-Modal Optimization

Here is the single most important takeaway: **DSPy's optimizers work on multi-modal programs with no changes.** The same `BootstrapFewShotWithRandomSearch` and `MIPROv2` that optimize your text pipelines will optimize your image and audio pipelines.

### Optimizing an Image QA Pipeline

```python
# Evaluate baseline
metric = dspy.evaluate.answer_exact_match
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)

qa_module = dspy.ChainOfThought(ImageQA)
baseline_score = evaluate(qa_module)
print(f"Baseline: {baseline_score:.1f}%")

# Optimize with MIPROv2
optimizer = dspy.MIPROv2(
    metric=metric,
    auto="light",
    num_threads=4,
)
optimized_qa = optimizer.compile(qa_module, trainset=trainset, max_bootstrapped_demos=3)

optimized_score = evaluate(optimized_qa)
print(f"Optimized: {optimized_score:.1f}%")
```

### Optimizing Audio Pipelines: Important Caveats

Audio tokens are significantly more expensive than text tokens, and audio data is large. The DSPy docs recommend a **conservative approach** when optimizing audio pipelines:

```python
# For audio pipelines, use conservative settings
optimizer = dspy.MIPROv2(
    metric=metric,
    auto="light",
    num_threads=2,  # fewer threads to manage costs
)

optimized_audio_qa = optimizer.compile(
    spoken_qa,
    trainset=trainset,
    max_bootstrapped_demos=2,  # 0 to 2 few-shot examples (audio demos are large)
)
```

Key guidelines for audio optimization:

1. **Configure optimizers conservatively.** Use 0 to 2 few-shot examples and fewer candidates/trials. Audio tokens are expensive, and each few-shot demo includes the full audio encoding.
2. **Use `data_aware_proposer=False`** when your dataset contains audio. The proposer may struggle to reason about raw audio data in its context.
3. **Keep datasets small.** A trainset of 20 to 50 examples is plenty for audio tasks.

```python
# MIPROv2 with data_aware_proposer=False for audio datasets
optimizer = dspy.MIPROv2(
    metric=metric,
    auto="light",
    num_threads=2,
    data_aware_proposer=False,  # disable when dataset contains audio
)
```

---

## Practical Considerations

### Token Costs

Multi-modal inputs consume significantly more tokens than text:

- **Images:** A typical image costs 1,000 to 5,000 tokens depending on resolution and detail level.
- **Audio:** Audio tokens are among the most expensive. A 30-second clip can cost as much as thousands of text tokens.

Monitor your costs carefully during development. Use DSPy's built-in caching to avoid re-processing the same inputs.

### Caching

DSPy caches multi-modal LM calls by default, just like text calls. This means:

- The first call with a given image/audio processes and sends the data.
- Subsequent calls with the same input hit the cache instantly.
- Use `cache=False` in your LM call if you need fresh responses.

```python
# Cache is on by default, great for development
result1 = describer(current_image=image)  # calls the API
result2 = describer(current_image=image)  # cache hit, instant
```

### Model Selection

Not all models support all modalities. Before building a pipeline:

1. Check that your model supports the input modality (vision, audio).
2. Check the model's context window. Images and audio consume significant space.
3. Consider cost trade-offs: `gpt-4o-mini` is much cheaper than `gpt-4o` for vision tasks, with strong performance.

---

## Putting It All Together

Here is a complete example combining image analysis with structured output:

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


class ImageContentAnalysis(dspy.Signature):
    """Analyze an image and produce structured content analysis."""

    image: dspy.Image = dspy.InputField()
    main_subject: str = dspy.OutputField(desc="primary subject of the image")
    setting: str = dspy.OutputField(desc="where the image takes place")
    mood: str = dspy.OutputField(desc="overall mood or tone of the image")
    colors: list[str] = dspy.OutputField(desc="dominant colors in the image")
    description: str = dspy.OutputField(desc="detailed 2-3 sentence description")


analyzer = dspy.ChainOfThought(ImageContentAnalysis)

image = dspy.Image.from_url(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
)

result = analyzer(image=image)
print(f"Subject: {result.main_subject}")
print(f"Setting: {result.setting}")
print(f"Mood: {result.mood}")
print(f"Colors: {result.colors}")
print(f"Description: {result.description}")
```

The key insight: multi-modal DSPy is just DSPy. You use the same signatures, modules, and optimizers you already know. The only difference is the field types and the model you configure.

---

## Key Takeaways

- **`dspy.Image`** wraps images from URLs, files, or base64. Use it as a field type in signatures.
- **`dspy.Audio`** wraps audio from numpy arrays. Requires audio-capable models.
- **Vision models** (`gpt-4o-mini`) work with standard `dspy.LM()` configuration.
- **Audio models** require specific preview models like `gpt-4o-mini-audio-preview-2024-12-17`.
- **Optimization works the same**, but use conservative settings for audio (0 to 2 demos, `data_aware_proposer=False`).
- **Caching is your friend.** Multi-modal tokens are expensive, and DSPy caches by default.
- **Iterative refinement** with vision models creates powerful feedback loops for image generation.

---

## What's Next

Now that you understand DSPy's multi-modal primitives, it is time to combine them into full pipelines. In the next blog, you will build multi-modal workflows that chain vision, audio, and text processing into cohesive systems.

[10.2: Building Multi-Modal Pipelines](../10.2-multi-modal-pipelines/blog.md)

---

## Resources

- [DSPy Multi-Modality Guide](https://dspy.ai/learn/programming/multi_modality/)
- [DSPy Image API Reference](https://dspy.ai/api/primitives/Image/)
- [DSPy Audio API Reference](https://dspy.ai/api/primitives/Audio/)
- [Image Generation Prompting Tutorial](https://dspy.ai/tutorials/image_generation_prompting/)
- [Audio Tutorial](https://dspy.ai/tutorials/audio/)
- [OpenAI Vision Guide](https://platform.openai.com/docs/guides/vision)
- [OpenAI Audio Guide](https://platform.openai.com/docs/guides/audio)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [Blog 1.2: Signatures](../../01-foundations/1.2-signatures/blog.md)
- [Blog 4.2: MIPROv2](../../04-optimization/4.2-miprov2/blog.md)
