"""
Blog 10.2: Building Multi-Modal Pipelines
Requires: pip install -U dspy python-dotenv numpy openai torch transformers
Requires: OpenAI API key with gpt-4o-mini, gpt-4o-mini-audio-preview, gpt-4o-mini-tts
"""

import numpy as np
import dspy
from dotenv import load_dotenv

load_dotenv()


# =====================================================
# TTS Pipeline: Emotion-Aware Speech Synthesis
# =====================================================

class EmotionStylePromptSignature(dspy.Signature):
    """Generate a TTS instruction that will make the speech model
    deliver the given line in the specified emotional style."""

    raw_line: str = dspy.InputField(desc="the text line to be spoken")
    target_style: str = dspy.InputField(
        desc="desired emotional style, e.g., 'excited', 'sad'"
    )
    openai_instruction: str = dspy.OutputField(
        desc="detailed instruction for the TTS model describing "
        "voice qualities, pacing, tone, and emotional delivery"
    )


class EmotionStylePrompter(dspy.Module):
    """Generate emotion-styled speech from text using optimizable prompts."""

    def __init__(self):
        self.generate_instruction = dspy.ChainOfThought(
            EmotionStylePromptSignature
        )
        # OpenAI client for direct TTS API calls
        from openai import OpenAI
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
        audio_array = audio_array.astype(np.float32) / 32768.0

        return dspy.Prediction(
            openai_instruction=result.openai_instruction,
            audio=dspy.Audio.from_array(audio_array, sampling_rate=24000),
        )


# =====================================================
# Audio Similarity Metric (Wav2Vec 2.0)
# =====================================================

class AudioSimilarityMetric:
    """Compute cosine similarity between audio embeddings."""

    def __init__(self):
        import torch
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )
        self.model.eval()
        self.torch = torch

    def get_embedding(self, audio_array, sampling_rate=16000):
        inputs = self.processor(
            audio_array, sampling_rate=sampling_rate,
            return_tensors="pt", padding=True,
        )
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def __call__(self, example, prediction, trace=None):
        ref_emb = self.get_embedding(example.reference_audio_array)
        pred_emb = self.get_embedding(prediction.audio_array)
        similarity = self.torch.nn.functional.cosine_similarity(
            ref_emb.unsqueeze(0), pred_emb.unsqueeze(0),
        ).item()
        return similarity


# =====================================================
# Iterative Image Generation Pipeline
# =====================================================

class ImagePromptRefiner(dspy.Signature):
    """Analyze a generated image, compare it with the desired prompt,
    and produce a revised prompt for better results."""

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
        desc="detailed analysis of what is wrong or missing"
    )
    image_strictly_matches_desired_prompt: bool = dspy.OutputField(
        desc="True only if the image perfectly matches"
    )
    revised_prompt: str = dspy.OutputField(
        desc="improved prompt that addresses the feedback"
    )


class IterativeImageGenerator(dspy.Module):
    """Generate images iteratively with vision-based feedback."""

    def __init__(self, max_iterations=5):
        self.refiner = dspy.Predict(ImagePromptRefiner)
        self.max_iterations = max_iterations

    def generate_image(self, prompt):
        """Replace with your image generation API."""
        return dspy.Image.from_url("https://example.com/generated.jpg")

    def forward(self, desired_prompt):
        current_prompt = desired_prompt
        all_feedback = []

        for i in range(self.max_iterations):
            current_image = self.generate_image(current_prompt)
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

        return dspy.Prediction(
            final_image=current_image,
            final_prompt=current_prompt,
            iterations=self.max_iterations,
            feedback_history=all_feedback,
        )


# =====================================================
# Custom Multi-Modal Analyzer
# =====================================================

class VisualAnalysis(dspy.Signature):
    """Analyze the visual content of an image."""
    image: dspy.Image = dspy.InputField()
    objects: list[str] = dspy.OutputField(desc="objects detected")
    scene_description: str = dspy.OutputField(desc="scene description")
    visual_style: str = dspy.OutputField(desc="artistic or photographic style")


class ContentSynthesis(dspy.Signature):
    """Synthesize visual analysis with text into a cohesive summary."""
    scene_description: str = dspy.InputField()
    objects: list[str] = dspy.InputField()
    visual_style: str = dspy.InputField()
    accompanying_text: str = dspy.InputField()
    summary: str = dspy.OutputField(
        desc="cohesive summary combining visual and textual content"
    )
    key_themes: list[str] = dspy.OutputField(
        desc="main themes across both modalities"
    )


class MultiModalAnalyzer(dspy.Module):
    """Chain: image -> visual analysis -> synthesis with text."""

    def __init__(self):
        self.analyze_image = dspy.ChainOfThought(VisualAnalysis)
        self.synthesize = dspy.ChainOfThought(ContentSynthesis)

    def forward(self, image, accompanying_text):
        visual = self.analyze_image(image=image)
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


# =====================================================
# Image Quality Judge (LLM-as-Judge)
# =====================================================

class ImageQualityJudge(dspy.Signature):
    """Judge whether the generated image matches the description."""
    image: dspy.Image = dspy.InputField()
    desired_description: str = dspy.InputField()
    quality_score: float = dspy.OutputField(desc="score from 0.0 to 1.0")
    reasoning: str = dspy.OutputField(desc="explanation of the score")


judge = dspy.Predict(ImageQualityJudge)


def image_quality_metric(example, prediction, trace=None):
    """LLM-as-judge metric for image quality evaluation."""
    result = judge(
        image=prediction.final_image,
        desired_description=example.desired_prompt,
    )
    return result.quality_score


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    print("=" * 50)
    print("  Blog 10.2: Multi-Modal Pipelines")
    print("=" * 50)

    # Iterative image generation
    print("\n--- Iterative Image Generation ---")
    generator = IterativeImageGenerator(max_iterations=3)
    result = generator(
        desired_prompt="A watercolor painting of a lighthouse during a storm"
    )
    print(f"Converged in {result.iterations} iterations")
    print(f"Final prompt: {result.final_prompt}")

    # Multi-modal analyzer
    print("\n--- Multi-Modal Analyzer ---")
    analyzer = MultiModalAnalyzer()
    result = analyzer(
        image=dspy.Image.from_url(
            "https://upload.wikimedia.org/wikipedia/commons/"
            "thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
        ),
        accompanying_text="Meet our newest rescue cat! Adoption is the best.",
    )
    print(f"Themes: {result.key_themes}")
    print(f"Summary: {result.summary}")

    print("\nDone! See blog for TTS and optimization examples.")
