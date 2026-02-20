"""
Blog 11.2: Async Programming and Streaming in DSPy
Requires: pip install -U dspy python-dotenv
Requires: OpenAI API key
"""

import asyncio

import dspy
from dotenv import load_dotenv
from dspy.streaming import StatusMessageProvider, StreamListener

load_dotenv()


# =====================================================
# Section 1: Basic Async Usage
# =====================================================


async def demo_basic_async():
    """Basic async call with acall()."""
    predict = dspy.Predict("question -> answer")
    output = await predict.acall(question="What is the capital of France?")
    print(f"Async answer: {output.answer}")


async def demo_concurrent_async():
    """Fire multiple LM calls concurrently."""
    predict = dspy.Predict("question -> answer")

    questions = [
        "What is the speed of light?",
        "Who wrote Hamlet?",
        "What is the largest ocean?",
    ]

    tasks = [predict.acall(question=q) for q in questions]
    results = await asyncio.gather(*tasks)

    for q, r in zip(questions, results):
        print(f"Q: {q}\nA: {r.answer}\n")


# =====================================================
# Section 2: Custom Async Modules
# =====================================================


class AsyncPipeline(dspy.Module):
    """Two-step async pipeline: analyze then summarize."""

    def __init__(self):
        self.analyze = dspy.Predict("text -> sentiment, topics")
        self.summarize = dspy.Predict("text, sentiment, topics -> summary")

    async def aforward(self, text, **kwargs):
        analysis = await self.analyze.acall(text=text)
        result = await self.summarize.acall(
            text=text,
            sentiment=analysis.sentiment,
            topics=analysis.topics,
        )
        return result


async def demo_async_pipeline():
    """Use a custom async module."""
    pipeline = AsyncPipeline()
    result = await pipeline.acall(text="DSPy enables programming LMs declaratively.")
    print(f"Summary: {result.summary}")


# =====================================================
# Section 3: Output Token Streaming
# =====================================================


async def demo_basic_streaming():
    """Stream tokens from a ChainOfThought program."""
    program = dspy.ChainOfThought("question -> answer")
    listener = StreamListener(signature_field_name="answer")
    streaming_program = dspy.streamify(program, stream_listeners=[listener])

    output = streaming_program(question="Explain black holes in simple terms")

    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk.chunk, end="", flush=True)
        elif isinstance(chunk, dspy.Prediction):
            print("\n\n--- Final answer ---")
            print(chunk.answer[:100] + "...")


# =====================================================
# Section 4: Status Message Streaming
# =====================================================


class MyStatusProvider(StatusMessageProvider):
    """Custom status messages for pipeline progress."""

    def lm_start_status_message(self, instance, inputs):
        return "Thinking..."

    def lm_end_status_message(self, instance, outputs):
        return "Got response"

    def module_start_status_message(self, instance, inputs):
        name = instance.__class__.__name__
        return f"Starting {name}..."

    def module_end_status_message(self, instance, outputs):
        name = instance.__class__.__name__
        return f"Finished {name}"


async def demo_status_streaming():
    """Stream status messages alongside token output."""
    program = dspy.ChainOfThought("question -> answer")
    listener = StreamListener(signature_field_name="answer")

    streaming_program = dspy.streamify(
        program,
        stream_listeners=[listener],
        status_message_provider=MyStatusProvider(),
    )

    output = streaming_program(question="What causes tides?")

    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StatusMessage):
            print(f"\n[STATUS] {chunk.message}")
        elif isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk.chunk, end="", flush=True)
        elif isinstance(chunk, dspy.Prediction):
            print(f"\n\nFinal: {chunk.answer[:100]}...")


# =====================================================
# Section 5: Complete Async Streaming Pipeline
# =====================================================


class ResearchPipeline(dspy.Module):
    """Multi-step research pipeline with async support."""

    def __init__(self):
        self.analyze = dspy.ChainOfThought("question -> key_points")
        self.synthesize = dspy.ChainOfThought("question, key_points -> answer")

    async def aforward(self, question, **kwargs):
        analysis = await self.analyze.acall(question=question)
        return await self.synthesize.acall(
            question=question,
            key_points=analysis.key_points,
        )


class PipelineStatus(StatusMessageProvider):
    def module_start_status_message(self, instance, inputs):
        return f"Starting {instance.__class__.__name__}..."

    def lm_start_status_message(self, instance, inputs):
        return "Generating response..."


async def demo_complete_pipeline():
    """Full async streaming pipeline with status messages."""
    pipeline = ResearchPipeline()
    listener = StreamListener(
        signature_field_name="answer",
        predict_name="synthesize",
    )
    streaming_pipeline = dspy.streamify(
        pipeline,
        stream_listeners=[listener],
        status_message_provider=PipelineStatus(),
    )

    output = streaming_pipeline(
        question="What are the key challenges in quantum computing?",
    )

    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StatusMessage):
            print(f"\n[{chunk.message}]")
        elif isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk.chunk, end="", flush=True)
        elif isinstance(chunk, dspy.Prediction):
            print("\n\nDone!")


# =====================================================
# Main
# =====================================================


async def main():
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    print("=" * 60)
    print("  Blog 11.2: Async Programming and Streaming")
    print("=" * 60)

    print("\n--- 1. Basic Async ---")
    await demo_basic_async()

    print("\n--- 2. Concurrent Async ---")
    await demo_concurrent_async()

    print("\n--- 3. Custom Async Pipeline ---")
    await demo_async_pipeline()

    print("\n--- 4. Token Streaming ---")
    await demo_basic_streaming()

    print("\n--- 5. Status Message Streaming ---")
    await demo_status_streaming()

    print("\n--- 6. Complete Pipeline ---")
    await demo_complete_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
