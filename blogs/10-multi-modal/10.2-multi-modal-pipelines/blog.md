# 10.2: Building Multi-Modal Pipelines

## Introduction

In [10.1](../10.1-image-audio/blog.md), you learned to work with individual modalities, passing images and audio through DSPy signatures. But the real power of multi-modal AI comes from **combining modalities in a single pipeline**: image to text analysis, text to speech generation, audio to text to structured output, or even full cross-modal workflows where images generate text feedback that generates better images.

In this blog, you will build complete multi-modal pipelines that chain vision, audio, and text processing together. You will learn to build a TTS (text-to-speech) pipeline with emotion-aware prompt optimization, an iterative image generation system, and custom multi-modal modules that combine multiple analysis stages. You will also tackle the practical challenges: evaluating multi-modal outputs, managing token costs, and selecting the right models.

---

## What You'll Learn

- Building end-to-end multi-modal pipelines with DSPy modules
- TTS pipeline with emotion-style prompting and audio similarity metrics
- Iterative image generation with vision-based feedback loops
- Custom multi-modal modules combining vision analysis with text generation
- Evaluating multi-modal outputs (audio cosine similarity, LLM-as-judge for images)
- Cross-modal workflows: text to image to text feedback cycles
- Practical considerations: token costs, caching, and model selection

---

## Prerequisites

- Completed [10.1: Working with Images and Audio](../10.1-image-audio/blog.md)
- Familiarity with DSPy optimizers ([Blog 4.2: MIPROv2](../../04-optimization/4.2-miprov2/blog.md))
- `uv add dspy python-dotenv numpy openai` installed
- OpenAI API key with access to `gpt-4o-mini`, `gpt-4o-mini-audio-preview-2024-12-17`, and `gpt-4o-mini-tts`

---

## TTS Pipeline: Emotion-Aware Speech Synthesis

One of the most compelling multi-modal patterns is **optimizing text-to-speech with DSPy**. Instead of hand-tuning voice instructions for a TTS model, you define a signature that generates emotion-aware prompts, then let DSPy optimize those prompts using audio similarity metrics.

### The EmotionStylePrompt Signature

The first stage takes a raw text line and a target style (e.g., "excited", "melancholic") and generates an instruction for the TTS model:

```python
import dspy
from dotenv import load_dotenv

load_dotenv()


class EmotionStylePromptSignature(dspy.Signature):
    """Generate a TTS instruction that will make the speech model
    deliver the given line in the specified emotional style."""

    raw_line: str = dspy.InputField(desc="the text line to be spoken")
    target_style: str = dspy.InputField(desc="desired emotional style, e.g., 'excited', 'sad'")
    openai_instruction: str = dspy.OutputField(
        desc="detailed instruction for the TTS model describing voice qualities, "
        "pacing, tone, and emotional delivery"
    )
```

### The EmotionStylePrompter Module

The module wraps `ChainOfThought` for instruction generation and then calls the TTS model directly via the OpenAI client:

```python
import numpy as np
from openai import OpenAI


class EmotionStylePrompter(dspy.Module):
    """Generate emotion-styled speech from text using optimizable prompts."""

    def __init__(self):
        self.generate_instruction = dspy.ChainOfThought(EmotionStylePromptSignature)
        self.openai_client = OpenAI()

    def forward(self, raw_line: str, target_style: str):
        # Step 1: Generate the TTS instruction (optimizable by DSPy)
        result = self.generate_instruction(
            raw_line=raw_line,
            target_style=target_style,
        )

        # Step 2: Call TTS model with the generated instruction
        tts_response = self.openai_client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=raw_line,
            instructions=result.openai_instruction,
            response_format="pcm",
        )

        # Convert raw PCM bytes to numpy array
        audio_array = np.frombuffer(tts_response.content, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0  # normalize to [-1, 1]

        return dspy.Prediction(
            openai_instruction=result.openai_instruction,
            audio=dspy.Audio.from_array(audio_array, sampling_rate=24000),
        )
```

Notice the architecture: the **instruction generation** step is a standard DSPy module that can be optimized. The TTS call itself is a deterministic API call. This means `MIPROv2` can discover better instructions that produce more emotionally appropriate speech, without you hand-tuning a single prompt.

### Audio Similarity Metrics with Wav2Vec 2.0

How do you evaluate whether two audio clips sound similar? One effective approach uses **Wav2Vec 2.0 embeddings** and cosine similarity:

```python
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class AudioSimilarityMetric:
    """Compute cosine similarity between audio embeddings using Wav2Vec 2.0."""

    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.model.eval()

    def get_embedding(self, audio_array: np.ndarray, sampling_rate: int = 16000):
        inputs = self.processor(
            audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling over time dimension
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def __call__(self, example, prediction, trace=None):
        # Get embeddings for reference and predicted audio
        ref_embedding = self.get_embedding(example.reference_audio_array)
        pred_embedding = self.get_embedding(prediction.audio_array)

        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            ref_embedding.unsqueeze(0),
            pred_embedding.unsqueeze(0),
        ).item()

        # Return similarity score (0 to 1)
        return similarity
```

This metric lets DSPy optimizers compare different instruction strategies and select the one that produces speech most similar to a reference recording.

### Optimizing the TTS Pipeline

With the metric defined, optimization uses `MIPROv2` as usual, and the results can be dramatic. The optimizer discovers instruction patterns that produce more emotionally compelling speech:

```python
# Configure for TTS optimization
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Build trainset of (raw_line, target_style, reference_audio) examples
trainset = [...]  # your training examples

# Optimize
audio_metric = AudioSimilarityMetric()
optimizer = dspy.MIPROv2(
    metric=audio_metric,
    auto="light",
    num_threads=2,
    data_aware_proposer=False,  # audio data in dataset
)

tts_module = EmotionStylePrompter()
optimized_tts = optimizer.compile(tts_module, trainset=trainset, max_bootstrapped_demos=1)
```

After optimization, the model might discover instructions like *"Speak with a trembling voice that builds to a crescendo, pausing before key emotional words"* instead of the generic instructions it started with.

---

## Iterative Image Generation Pipeline

In [10.1](../10.1-image-audio/blog.md), you saw the basic image prompt refinement pattern. Let's build a complete pipeline that integrates with a real image generation API (using FAL/Flux Pro as an example):

```python
import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


class ImagePromptRefiner(dspy.Signature):
    """Analyze a generated image, compare it with the desired prompt,
    and produce a revised prompt for better results."""

    desired_prompt: str = dspy.InputField(desc="what the image should depict")
    current_image: dspy.Image = dspy.InputField(desc="the currently generated image")
    current_prompt: str = dspy.InputField(desc="the prompt used to generate current_image")
    feedback: str = dspy.OutputField(desc="detailed analysis of what is wrong or missing")
    image_strictly_matches_desired_prompt: bool = dspy.OutputField(
        desc="True only if the image perfectly matches the desired prompt"
    )
    revised_prompt: str = dspy.OutputField(
        desc="improved prompt that addresses the feedback"
    )


class IterativeImageGenerator(dspy.Module):
    """Generate images iteratively, refining the prompt based on visual feedback."""

    def __init__(self, max_iterations: int = 5):
        self.refiner = dspy.Predict(ImagePromptRefiner)
        self.max_iterations = max_iterations

    def generate_image(self, prompt: str) -> dspy.Image:
        """Call your image generation API here.
        Example with FAL/Flux Pro:
            import fal_client
            result = fal_client.subscribe("fal-ai/flux-pro/v1.1", arguments={"prompt": prompt})
            return dspy.Image.from_url(result["images"][0]["url"])
        """
        # Placeholder: replace with your image gen API
        return dspy.Image.from_url("https://example.com/generated.jpg")

    def forward(self, desired_prompt: str):
        current_prompt = desired_prompt
        all_feedback = []

        for i in range(self.max_iterations):
            # Generate image from current prompt
            current_image = self.generate_image(current_prompt)

            # Get feedback from vision model
            result = self.refiner(
                desired_prompt=desired_prompt,
                current_image=current_image,
                current_prompt=current_prompt,
            )

            all_feedback.append(result.feedback)

            if result.image_strictly_matches_desired_prompt:
                return dspy.Prediction(
                    final_image=current_image,
                    final_prompt=current_prompt,
                    iterations=i + 1,
                    feedback_history=all_feedback,
                )

            current_prompt = result.revised_prompt

        # Return best effort after max iterations
        return dspy.Prediction(
            final_image=current_image,
            final_prompt=current_prompt,
            iterations=self.max_iterations,
            feedback_history=all_feedback,
        )


# Usage
generator = IterativeImageGenerator(max_iterations=5)
result = generator(desired_prompt="A watercolor painting of a lighthouse during a storm")
print(f"Converged in {result.iterations} iterations")
print(f"Final prompt: {result.final_prompt}")
```

The inner `refiner` module is optimizable. `MIPROv2` can discover better instructions for how the vision model should critique images and suggest prompt revisions.

---

## Building Custom Multi-Modal Modules

Real-world multi-modal applications often chain multiple analysis stages. Here is a pattern for combining vision analysis with text generation in a structured pipeline:

```python
class VisualAnalysis(dspy.Signature):
    """Analyze the visual content of an image."""

    image: dspy.Image = dspy.InputField()
    objects: list[str] = dspy.OutputField(desc="objects detected in the image")
    scene_description: str = dspy.OutputField(desc="description of the scene")
    visual_style: str = dspy.OutputField(desc="artistic or photographic style")


class ContentSynthesis(dspy.Signature):
    """Synthesize visual analysis with accompanying text into a cohesive summary."""

    scene_description: str = dspy.InputField()
    objects: list[str] = dspy.InputField()
    visual_style: str = dspy.InputField()
    accompanying_text: str = dspy.InputField()
    summary: str = dspy.OutputField(desc="cohesive summary combining visual and textual content")
    key_themes: list[str] = dspy.OutputField(desc="main themes across both modalities")


class MultiModalAnalyzer(dspy.Module):
    """Chain: image -> visual analysis -> synthesis with text -> summary."""

    def __init__(self):
        self.analyze_image = dspy.ChainOfThought(VisualAnalysis)
        self.synthesize = dspy.ChainOfThought(ContentSynthesis)

    def forward(self, image: dspy.Image, accompanying_text: str):
        # Step 1: Analyze the image
        visual = self.analyze_image(image=image)

        # Step 2: Synthesize with text
        synthesis = self.synthesize(
            scene_description=visual.scene_description,
            objects=visual.objects,
            visual_style=visual.visual_style,
            accompanying_text=accompanying_text,
        )

        return dspy.Prediction(
            objects=visual.objects,
            scene_description=visual.scene_description,
            visual_style=visual.visual_style,
            summary=synthesis.summary,
            key_themes=synthesis.key_themes,
        )


# Usage
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

analyzer = MultiModalAnalyzer()
result = analyzer(
    image=dspy.Image.from_url("https://example.com/product-photo.jpg"),
    accompanying_text="Our new sustainable water bottle features recycled ocean plastics.",
)
print(f"Themes: {result.key_themes}")
print(f"Summary: {result.summary}")
```

Each sub-module is independently optimizable. When you run `MIPROv2`, it discovers better instructions for both the visual analysis stage and the synthesis stage.

---

## Multi-Modal Evaluation

### Exact Match for QA Tasks

For tasks that produce text answers (e.g., spoken QA, image QA), use the same evaluation tools as text pipelines:

```python
metric = dspy.evaluate.answer_exact_match
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
score = evaluate(spoken_qa)
```

### Custom Metrics for Audio

For audio quality evaluation, cosine similarity on learned embeddings (as shown in the TTS section above) works well. Key options include:

- **Wav2Vec 2.0 embeddings.** Captures speech characteristics, prosody, and emotion.
- **Mel-spectrogram similarity.** Compares audio at the frequency level.
- **Transcription accuracy.** Run STT on output and compare to expected text.

### Custom Metrics for Images: LLM-as-Judge

For image quality, use the vision model itself as a judge:

```python
class ImageQualityJudge(dspy.Signature):
    """Judge whether the generated image matches the desired description."""

    image: dspy.Image = dspy.InputField()
    desired_description: str = dspy.InputField()
    quality_score: float = dspy.OutputField(desc="score from 0.0 to 1.0")
    reasoning: str = dspy.OutputField(desc="explanation of the score")


judge = dspy.Predict(ImageQualityJudge)


def image_quality_metric(example, prediction, trace=None):
    result = judge(
        image=prediction.final_image,
        desired_description=example.desired_prompt,
    )
    return result.quality_score
```

---

## Cross-Modal Workflows

Some of the most interesting multi-modal applications involve **cross-modal loops** where outputs from one modality feed into another:

```python
class TextToImageFeedback(dspy.Module):
    """Text to Image to Text feedback cycle."""

    def __init__(self):
        self.enhance_prompt = dspy.ChainOfThought(
            "concept -> detailed_image_prompt"
        )
        self.critique_image = dspy.ChainOfThought(
            "image: dspy.Image, original_concept: str -> critique, score: float"
        )

    def forward(self, concept: str):
        # Text to enhanced prompt
        enhanced = self.enhance_prompt(concept=concept)

        # Enhanced prompt to image (via external API)
        image = generate_image(enhanced.detailed_image_prompt)  # your image gen function

        # Image to text critique
        critique = self.critique_image(
            image=image,
            original_concept=concept,
        )

        return dspy.Prediction(
            image=image,
            prompt_used=enhanced.detailed_image_prompt,
            critique=critique.critique,
            score=critique.score,
        )
```

---

## Practical Considerations

### Token Costs for Multi-Modal

Multi-modal tokens are significantly more expensive than text:

| Modality | Approximate Cost (per input) | Notes |
|----------|------------------------------|-------|
| Text (1K tokens) | ~\$0.00015 (gpt-4o-mini) | Baseline |
| Image (standard) | ~\$0.001 to \$0.005 | Depends on resolution |
| Audio (30 sec) | ~\$0.005 to \$0.02 | Audio-preview pricing |

**Tip:** During development, minimize few-shot demos that include images or audio. Each demo multiplies the media cost by the number of examples.

### Caching Multi-Modal Results

DSPy caches all LM calls by default, including multi-modal ones. For iterative development, this is essential:

```python
# First run processes the image (costs tokens)
result = analyzer(image=my_image, accompanying_text="test text")

# Second run with same inputs (cache hit, free)
result2 = analyzer(image=my_image, accompanying_text="test text")
```

For production with variable inputs, consider caching at the module level:

```python
# Disable cache for production evaluation
lm = dspy.LM("openai/gpt-4o-mini", cache=False)
```

### Model Selection Based on Modality

Choose models based on what your pipeline needs:

- **Image understanding only:** `gpt-4o-mini` (cost-effective) or `gpt-4o` (highest quality)
- **Audio input:** `gpt-4o-mini-audio-preview-2024-12-17` (currently the only DSPy-compatible option)
- **Audio generation:** `gpt-4o-mini-tts` (via OpenAI client, not through DSPy directly)
- **Multi-step pipelines:** Use `dspy.context(lm=...)` to assign different models to different stages

---

## Key Takeaways

- **TTS pipelines** combine DSPy instruction optimization with direct API calls for speech synthesis.
- **Audio similarity metrics** (Wav2Vec 2.0 cosine similarity) let optimizers evaluate audio quality.
- **Iterative image generation** uses vision models as quality gates in feedback loops.
- **Custom multi-modal modules** chain vision, text, and synthesis stages, each independently optimizable.
- **Cross-modal workflows** pass data between modalities for richer analysis.
- **LLM-as-judge** works for image quality evaluation. The vision model scores its own outputs.
- **Token costs matter.** Multi-modal inputs are expensive; cache aggressively and limit few-shot demos.
- **`MIPROv2` with `data_aware_proposer=False`** is essential when datasets contain audio data.

---

## What's Next

You have built multi-modal pipelines from individual components. Now it is time to put everything together in a real-world project: a multi-modal content analyzer that accepts images and text, produces structured analysis, and is optimized end-to-end.

[10.P: Project: Multi-Modal Content Analyzer](../10.P-project-content-analyzer/blog.md)

---

## Resources

- [DSPy Multi-Modality Guide](https://dspy.ai/learn/programming/multi_modality/)
- [Image Generation Prompting Tutorial](https://dspy.ai/tutorials/image_generation_prompting/)
- [Audio Tutorial](https://dspy.ai/tutorials/audio/)
- [DSPy Evaluation Guide](https://dspy.ai/learn/evaluation/overview/)
- [Wav2Vec 2.0 on HuggingFace](https://huggingface.co/facebook/wav2vec2-base)
- [OpenAI TTS Guide](https://platform.openai.com/docs/guides/text-to-speech)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [Blog 10.1: Working with Images and Audio](../10.1-image-audio/blog.md)
- [Blog 4.2: MIPROv2](../../04-optimization/4.2-miprov2/blog.md)
