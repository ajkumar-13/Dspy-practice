"""
Blog 10.1: Working with Images and Audio in DSPy
Requires: pip install -U dspy python-dotenv numpy
Requires: OpenAI API key with gpt-4o-mini and gpt-4o-mini-audio-preview access
"""

import dspy
import numpy as np
from dotenv import load_dotenv

load_dotenv()


# =====================================================
# Image Basics
# =====================================================

def image_basics():
    """Demonstrate dspy.Image creation and usage."""
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    # Create an image from a URL
    image = dspy.Image.from_url(
        "https://upload.wikimedia.org/wikipedia/commons/"
        "thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    )

    # Alternative constructor
    image_alt = dspy.Image(url="https://example.com/photo.jpg")

    # Describe the image
    class DescribeImage(dspy.Signature):
        """Describe the contents of an image in detail."""
        current_image: dspy.Image = dspy.InputField()
        description: str = dspy.OutputField(
            desc="a detailed description of the image"
        )

    describer = dspy.Predict(DescribeImage)
    result = describer(current_image=image)
    print(f"Description: {result.description}")
    return result


# =====================================================
# Image QA with ChainOfThought
# =====================================================

def image_qa():
    """Image question answering with step-by-step reasoning."""
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    class ImageQA(dspy.Signature):
        """Answer questions about an image."""
        image: dspy.Image = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(
            desc="concise answer to the question"
        )

    qa = dspy.ChainOfThought(ImageQA)
    result = qa(
        image=dspy.Image.from_url(
            "https://upload.wikimedia.org/wikipedia/commons/"
            "thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
        ),
        question="What animal is in this image, and what is it doing?",
    )
    print(f"Answer: {result.answer}")
    return result


# =====================================================
# Iterative Image Prompt Refinement
# =====================================================

def iterative_refinement():
    """Refine image generation prompts using vision feedback."""
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    class ImagePromptRefiner(dspy.Signature):
        """Compare a generated image against the desired prompt,
        provide feedback, and produce a revised prompt."""
        desired_prompt: str = dspy.InputField(
            desc="what the image should depict"
        )
        current_image: dspy.Image = dspy.InputField(
            desc="the currently generated image"
        )
        current_prompt: str = dspy.InputField(
            desc="the prompt used to generate current_image"
        )
        feedback: str = dspy.OutputField(
            desc="what is wrong or missing in the current image"
        )
        image_strictly_matches_desired_prompt: bool = dspy.OutputField()
        revised_prompt: str = dspy.OutputField(
            desc="improved prompt for image generation"
        )

    refiner = dspy.Predict(ImagePromptRefiner)

    desired = "A photorealistic red fox sitting in a snowy forest at sunset"
    current_prompt = desired
    max_iterations = 5

    for i in range(max_iterations):
        # In practice, call an image generation API (e.g., FAL/Flux Pro)
        generated_image = dspy.Image.from_url(
            "https://example.com/generated.jpg"
        )

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

    return current_prompt


# =====================================================
# Audio Basics
# =====================================================

def audio_basics():
    """Demonstrate dspy.Audio creation and spoken QA."""
    # Create audio from a numpy array (440Hz sine wave)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_array = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    audio = dspy.Audio.from_array(audio_array, sampling_rate=sample_rate)
    print(f"Created audio: {sample_rate}Hz, {duration}s")

    # Spoken QA signature
    class SpokenQASignature(dspy.Signature):
        """Answer the question based on the audio clip."""
        passage_audio: dspy.Audio = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(
            desc="factoid answer between 1 and 5 words"
        )

    spoken_qa = dspy.ChainOfThought(SpokenQASignature)

    # Note: requires audio-preview model
    # audio_lm = dspy.LM("gpt-4o-mini-audio-preview-2024-12-17")
    # dspy.configure(lm=audio_lm)
    # result = spoken_qa(passage_audio=audio, question="What is the frequency?")
    print("SpokenQA module created (requires audio-preview model to run)")
    return spoken_qa


# =====================================================
# Structured Image Analysis
# =====================================================

def structured_image_analysis():
    """Complete multi-modal example with structured output."""
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    class ImageContentAnalysis(dspy.Signature):
        """Analyze an image and produce structured content analysis."""
        image: dspy.Image = dspy.InputField()
        main_subject: str = dspy.OutputField(
            desc="primary subject of the image"
        )
        setting: str = dspy.OutputField(
            desc="where the image takes place"
        )
        mood: str = dspy.OutputField(
            desc="overall mood or tone of the image"
        )
        colors: list[str] = dspy.OutputField(
            desc="dominant colors in the image"
        )
        description: str = dspy.OutputField(
            desc="detailed 2-3 sentence description"
        )

    analyzer = dspy.ChainOfThought(ImageContentAnalysis)

    image = dspy.Image.from_url(
        "https://upload.wikimedia.org/wikipedia/commons/"
        "thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    )

    result = analyzer(image=image)
    print(f"Subject: {result.main_subject}")
    print(f"Setting: {result.setting}")
    print(f"Mood: {result.mood}")
    print(f"Colors: {result.colors}")
    print(f"Description: {result.description}")
    return result


# =====================================================
# Multi-Modal Optimization
# =====================================================

def optimize_image_qa(trainset, devset):
    """Optimize an image QA pipeline with MIPROv2."""
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    class ImageQA(dspy.Signature):
        """Answer questions about an image."""
        image: dspy.Image = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(
            desc="concise answer to the question"
        )

    metric = dspy.evaluate.answer_exact_match
    evaluate = dspy.Evaluate(
        devset=devset, metric=metric,
        num_threads=4, display_progress=True,
    )

    qa_module = dspy.ChainOfThought(ImageQA)
    baseline_score = evaluate(qa_module)
    print(f"Baseline: {baseline_score:.1f}%")

    optimizer = dspy.MIPROv2(
        metric=metric, auto="light", num_threads=4,
    )
    optimized_qa = optimizer.compile(
        qa_module, trainset=trainset, max_bootstrapped_demos=3,
    )

    optimized_score = evaluate(optimized_qa)
    print(f"Optimized: {optimized_score:.1f}%")
    return optimized_qa


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  Blog 10.1: Images and Audio in DSPy")
    print("=" * 50)

    print("\n--- Image Basics ---")
    image_basics()

    print("\n--- Image QA ---")
    image_qa()

    print("\n--- Audio Basics ---")
    audio_basics()

    print("\n--- Structured Image Analysis ---")
    structured_image_analysis()

    print("\nDone! See blog for optimization and refinement examples.")
