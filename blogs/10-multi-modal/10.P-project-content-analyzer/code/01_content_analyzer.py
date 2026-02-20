"""
Blog 10.P: Multi-Modal Content Analyzer
Requires: pip install -U dspy python-dotenv
Requires: OpenAI API key with gpt-4o-mini access
"""

import time

import dspy
from dotenv import load_dotenv

load_dotenv()


# =====================================================
# Signatures
# =====================================================


class ImageAnalysisSignature(dspy.Signature):
    """Analyze the visual content of an image, identifying objects, scene,
    artistic style, and dominant colors."""

    image: dspy.Image = dspy.InputField()
    objects: list[str] = dspy.OutputField(desc="objects and subjects detected in the image")
    scene_description: str = dspy.OutputField(desc="description of the scene or setting")
    visual_style: str = dspy.OutputField(desc="photographic or artistic style")
    dominant_colors: list[str] = dspy.OutputField(desc="3-5 dominant colors in the image")


class TextAnalysisSignature(dspy.Signature):
    """Analyze accompanying text to extract themes, entities, and intent."""

    text: str = dspy.InputField(desc="the accompanying text (caption, description, etc.)")
    key_themes: list[str] = dspy.OutputField(desc="main themes or topics in the text")
    entities: list[str] = dspy.OutputField(desc="named entities (people, brands, places)")
    intent: str = dspy.OutputField(
        desc="communicative intent (e.g., 'promotional', 'informational')"
    )
    tone: str = dspy.OutputField(desc="writing tone (e.g., 'casual', 'professional', 'humorous')")


class ContentSynthesisSignature(dspy.Signature):
    """Combine visual and text analyses into a unified understanding."""

    scene_description: str = dspy.InputField()
    objects: list[str] = dspy.InputField()
    visual_style: str = dspy.InputField()
    dominant_colors: list[str] = dspy.InputField()
    key_themes: list[str] = dspy.InputField()
    entities: list[str] = dspy.InputField()
    intent: str = dspy.InputField()
    text_tone: str = dspy.InputField()
    coherence_assessment: str = dspy.OutputField(
        desc="how well the image and text complement each other"
    )
    unified_summary: str = dspy.OutputField(
        desc="2-3 sentence summary integrating visual and textual content"
    )
    content_category: str = dspy.OutputField(
        desc="category (e.g., 'product marketing', 'lifestyle', 'news')"
    )
    target_audience: str = dspy.OutputField(desc="likely target audience for this content")


class SentimentSignature(dspy.Signature):
    """Determine overall sentiment and emotional tone."""

    unified_summary: str = dspy.InputField()
    coherence_assessment: str = dspy.InputField()
    content_category: str = dspy.InputField()
    sentiment: str = dspy.OutputField(
        desc="overall sentiment: 'positive', 'negative', 'neutral', or 'mixed'"
    )
    emotional_tone: str = dspy.OutputField(
        desc="specific emotional tone (e.g., 'aspirational', 'urgent', 'nostalgic')"
    )
    confidence: float = dspy.OutputField(desc="confidence score from 0.0 to 1.0")


# =====================================================
# Component Modules
# =====================================================


class ImageAnalyzer(dspy.Module):
    """Analyze visual content from an image."""

    def __init__(self):
        self.analyze = dspy.ChainOfThought(ImageAnalysisSignature)

    def forward(self, image):
        return self.analyze(image=image)


class TextAnalyzer(dspy.Module):
    """Extract themes, entities, and intent from text."""

    def __init__(self):
        self.analyze = dspy.ChainOfThought(TextAnalysisSignature)

    def forward(self, text):
        return self.analyze(text=text)


class ContentSynthesizer(dspy.Module):
    """Combine visual and text analysis into unified understanding."""

    def __init__(self):
        self.synthesize = dspy.ChainOfThought(ContentSynthesisSignature)

    def forward(self, visual_result, text_result):
        return self.synthesize(
            scene_description=visual_result.scene_description,
            objects=visual_result.objects,
            visual_style=visual_result.visual_style,
            dominant_colors=visual_result.dominant_colors,
            key_themes=text_result.key_themes,
            entities=text_result.entities,
            intent=text_result.intent,
            text_tone=text_result.tone,
        )


class SentimentAnalyzer(dspy.Module):
    """Determine overall sentiment and emotional tone."""

    def __init__(self):
        self.analyze = dspy.ChainOfThought(SentimentSignature)

    def forward(self, synthesis_result):
        return self.analyze(
            unified_summary=synthesis_result.unified_summary,
            coherence_assessment=synthesis_result.coherence_assessment,
            content_category=synthesis_result.content_category,
        )


# =====================================================
# Pipeline
# =====================================================


class ContentAnalysisPipeline(dspy.Module):
    """Full multi-modal content analysis: image + text -> structured report."""

    def __init__(self):
        self.image_analyzer = ImageAnalyzer()
        self.text_analyzer = TextAnalyzer()
        self.synthesizer = ContentSynthesizer()
        self.sentiment_analyzer = SentimentAnalyzer()

    def forward(self, image, text):
        visual_result = self.image_analyzer(image=image)
        text_result = self.text_analyzer(text=text)
        synthesis = self.synthesizer(visual_result, text_result)
        sentiment = self.sentiment_analyzer(synthesis)

        return dspy.Prediction(
            objects=visual_result.objects,
            scene_description=visual_result.scene_description,
            visual_style=visual_result.visual_style,
            dominant_colors=visual_result.dominant_colors,
            key_themes=text_result.key_themes,
            entities=text_result.entities,
            intent=text_result.intent,
            text_tone=text_result.tone,
            coherence_assessment=synthesis.coherence_assessment,
            unified_summary=synthesis.unified_summary,
            content_category=synthesis.content_category,
            target_audience=synthesis.target_audience,
            sentiment=sentiment.sentiment,
            emotional_tone=sentiment.emotional_tone,
            confidence=sentiment.confidence,
        )


# =====================================================
# Evaluation
# =====================================================


def make_example(image_url, text, content_category, sentiment, intent):
    """Helper to create evaluation examples."""
    return dspy.Example(
        image=dspy.Image.from_url(image_url),
        text=text,
        content_category=content_category,
        sentiment=sentiment,
        intent=intent,
    ).with_inputs("image", "text")


def content_analysis_metric(example, prediction, trace=None):
    """Multi-dimensional metric for content analysis quality."""
    scores = []

    if hasattr(prediction, "content_category") and hasattr(example, "content_category"):
        pred_cat = prediction.content_category.lower().strip()
        gold_cat = example.content_category.lower().strip()
        scores.append(1.0 if pred_cat == gold_cat else 0.0)

    if hasattr(prediction, "sentiment") and hasattr(example, "sentiment"):
        pred_sent = prediction.sentiment.lower().strip()
        gold_sent = example.sentiment.lower().strip()
        scores.append(1.0 if pred_sent == gold_sent else 0.0)

    if hasattr(prediction, "intent") and hasattr(example, "intent"):
        pred_intent = prediction.intent.lower().strip()
        gold_intent = example.intent.lower().strip()
        scores.append(1.0 if pred_intent == gold_intent else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


# LLM-as-Judge for summary quality
class SummaryQualityJudge(dspy.Signature):
    """Judge the quality of a multi-modal content summary."""

    unified_summary: str = dspy.InputField()
    content_category: str = dspy.InputField()
    expected_category: str = dspy.InputField()
    quality_score: float = dspy.OutputField(desc="quality from 0.0 to 1.0")


summary_judge = dspy.Predict(SummaryQualityJudge)


def enhanced_metric(example, prediction, trace=None):
    """Combines exact-match fields with LLM-judged summary quality."""
    base_score = content_analysis_metric(example, prediction, trace)

    if trace is None and hasattr(prediction, "unified_summary"):
        judge_result = summary_judge(
            unified_summary=prediction.unified_summary,
            content_category=prediction.content_category,
            expected_category=example.content_category,
        )
        summary_score = float(judge_result.quality_score)
        return 0.7 * base_score + 0.3 * summary_score

    return base_score


# =====================================================
# Production: Robust Pipeline with Fallback
# =====================================================


class RobustContentAnalysisPipeline(ContentAnalysisPipeline):
    """Production version with fallback handling."""

    def forward(self, image, text):
        try:
            return super().forward(image=image, text=text)
        except Exception:
            text_result = self.text_analyzer(text=text)
            return dspy.Prediction(
                objects=[],
                scene_description="Image analysis unavailable",
                visual_style="unknown",
                dominant_colors=[],
                key_themes=text_result.key_themes,
                entities=text_result.entities,
                intent=text_result.intent,
                text_tone=text_result.tone,
                coherence_assessment="Image unavailable, text-only analysis",
                unified_summary=f"Text: {', '.join(text_result.key_themes)}",
                content_category="unknown",
                target_audience="general",
                sentiment="neutral",
                emotional_tone="neutral",
                confidence=0.5,
            )


def analyze_batch(pipeline, items, delay=0.5):
    """Process content items in batch with rate-limiting."""
    results = []
    for i, item in enumerate(items):
        try:
            result = pipeline(image=item["image"], text=item["text"])
            results.append({"status": "success", "result": result, "index": i})
        except Exception as e:
            results.append({"status": "error", "error": str(e), "index": i})
        time.sleep(delay)
    return results


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    print("=" * 60)
    print("  Blog 10.P: Multi-Modal Content Analyzer")
    print("=" * 60)

    # Build datasets
    trainset = [
        make_example(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/"
            "Good_Food_Display_-_NCI_Visuals_Online.jpg/800px-Good_Food_Display_-_NCI_Visuals_Online.jpg",
            "Fresh organic produce delivered to your door every week! "
            "Use code FRESH20 for 20% off your first order.",
            "product marketing",
            "positive",
            "promotional",
        ),
        make_example(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
            "After months of searching, we finally found our perfect companion.",
            "lifestyle",
            "positive",
            "personal sharing",
        ),
        make_example(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/"
            "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/"
            "1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
            "Van Gogh's Starry Night (1889) remains one of the most "
            "recognized paintings in Western art.",
            "entertainment",
            "neutral",
            "informational",
        ),
    ]

    devset = [
        make_example(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
            "PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
            "Transparency in design is about building trust.",
            "informational",
            "neutral",
            "informational",
        ),
    ]

    # Run baseline
    print("\n--- Baseline Evaluation ---")
    pipeline = ContentAnalysisPipeline()
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=content_analysis_metric,
        num_threads=4,
        display_progress=True,
    )
    baseline_score = evaluate(pipeline)
    print(f"Baseline score: {baseline_score:.1f}%")

    # Optimize with MIPROv2
    print("\n--- MIPROv2 Optimization ---")
    optimizer = dspy.MIPROv2(
        metric=content_analysis_metric,
        auto="light",
        num_threads=4,
    )
    optimized_pipeline = optimizer.compile(
        ContentAnalysisPipeline(),
        trainset=trainset,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
    )
    optimized_score = evaluate(optimized_pipeline)
    print(f"Optimized score: {optimized_score:.1f}%")
    print(f"Improvement: {optimized_score - baseline_score:+.1f}%")

    # Save
    optimized_pipeline.save("optimized_content_analyzer")
    print("\nSaved optimized pipeline to 'optimized_content_analyzer'")
