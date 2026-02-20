"""
Blog 11.3: Deploying DSPy Applications
Requires: pip install -U dspy python-dotenv fastapi uvicorn httpx
Requires: OpenAI API key

This file contains the complete deployment code split into logical modules.
In production, you would split these into separate files:
    - program.py (DSPy program definition)
    - config.py (configuration management)
    - optimize.py (optimization script)
    - app.py (FastAPI application)
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import dspy
from dotenv import load_dotenv
from dspy.streaming import StreamListener
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dspy_api")


# =====================================================
# Section 1: Program Definition
# =====================================================

class QASignature(dspy.Signature):
    """Answer a question with detailed reasoning."""

    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="A clear, comprehensive answer")
    key_concepts: str = dspy.OutputField(desc="Key concepts used in the answer")
    question_type: str = dspy.OutputField(
        desc="Type of question: factual, opinion, analytical"
    )


class QAProgram(dspy.Module):
    def __init__(self):
        self.predict = dspy.ChainOfThought(QASignature)

    def forward(self, question, **kwargs):
        return self.predict(question=question)

    async def aforward(self, question, **kwargs):
        return await self.predict.acall(question=question)


# =====================================================
# Section 2: Configuration
# =====================================================

class Settings:
    MODEL: str = os.getenv("DSPY_MODEL", "openai/gpt-4o-mini")
    TRACK_USAGE: bool = os.getenv("TRACK_USAGE", "true").lower() == "true"
    CACHE_DISK_ENABLED: bool = os.getenv("CACHE_DISK_ENABLED", "true").lower() == "true"
    CACHE_MEMORY_ENABLED: bool = os.getenv("CACHE_MEMORY_ENABLED", "true").lower() == "true"
    CACHE_DISK_SIZE: int = int(os.getenv("CACHE_DISK_SIZE", str(5_000_000_000)))
    CACHE_MEMORY_ENTRIES: int = int(os.getenv("CACHE_MEMORY_ENTRIES", str(2_000_000)))
    PROGRAM_STATE_PATH: str | None = os.getenv("PROGRAM_STATE_PATH")
    ENABLE_STREAMING: bool = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "1"))


settings = Settings()


# =====================================================
# Section 3: Metrics Tracker
# =====================================================

class MetricsTracker:
    def __init__(self):
        self.total_requests = 0
        self.total_errors = 0
        self.total_tokens = 0
        self.latencies = []
        self.start_time = time.time()

    def record_request(self, latency: float, usage: dict):
        self.total_requests += 1
        self.latencies.append(latency)
        for model_usage in usage.values():
            self.total_tokens += model_usage.get("prompt_tokens", 0)
            self.total_tokens += model_usage.get("completion_tokens", 0)

    def record_error(self):
        self.total_errors += 1

    def get_summary(self):
        uptime = time.time() - self.start_time
        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": round(
                self.total_errors / max(self.total_requests, 1), 4
            ),
            "avg_latency": round(
                sum(self.latencies) / max(len(self.latencies), 1), 3
            ),
            "total_tokens": self.total_tokens,
        }


metrics = MetricsTracker()


# =====================================================
# Section 4: FastAPI Application
# =====================================================

program: QAProgram = None
streaming_program = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global program, streaming_program

    dspy.configure(
        lm=dspy.LM(settings.MODEL),
        track_usage=settings.TRACK_USAGE,
    )
    dspy.configure_cache(
        enable_disk_cache=settings.CACHE_DISK_ENABLED,
        enable_memory_cache=settings.CACHE_MEMORY_ENABLED,
        disk_size_limit_bytes=settings.CACHE_DISK_SIZE,
        memory_max_entries=settings.CACHE_MEMORY_ENTRIES,
    )

    program = QAProgram()
    if settings.PROGRAM_STATE_PATH:
        program.load(settings.PROGRAM_STATE_PATH)
        logger.info(f"Loaded optimized state: {settings.PROGRAM_STATE_PATH}")

    if settings.ENABLE_STREAMING:
        listener = StreamListener(signature_field_name="answer")
        streaming_program = dspy.streamify(program, stream_listeners=[listener])
        logger.info("Streaming enabled")

    logger.info(f"API ready | Model: {settings.MODEL}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="DSPy Production API",
    description="A production-ready QA API powered by DSPy",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        metrics.record_error()
        logger.error(f"Unhandled error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )


# =====================================================
# Section 5: Request/Response Models
# =====================================================

class PredictRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=10000,
        description="The question to answer",
    )


class PredictResponse(BaseModel):
    answer: str
    key_concepts: str | None = None
    question_type: str | None = None
    usage: dict | None = None
    latency_seconds: float | None = None


# =====================================================
# Section 6: Endpoints
# =====================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": settings.MODEL,
        "optimized": settings.PROGRAM_STATE_PATH is not None,
    }


@app.get("/metrics")
async def get_metrics():
    if not settings.ENABLE_METRICS:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    return metrics.get_summary()


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start = time.time()
    try:
        result = await program.acall(question=request.question)
        latency = time.time() - start
        usage = result.get_lm_usage() if settings.TRACK_USAGE else {}
        metrics.record_request(latency, usage)
        logger.info(
            f"Predicted in {latency:.3f}s | "
            f"Q: {request.question[:50]}... | Tokens: {usage}"
        )
        return PredictResponse(
            answer=result.answer,
            key_concepts=getattr(result, "key_concepts", None),
            question_type=getattr(result, "question_type", None),
            usage=usage,
            latency_seconds=round(latency, 3),
        )
    except dspy.DSPyAssertionError as e:
        metrics.record_error()
        logger.warning(f"Assertion failed: {e}")
        raise HTTPException(
            status_code=422, detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        metrics.record_error()
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict/stream")
async def predict_stream(request: PredictRequest):
    if not settings.ENABLE_STREAMING or streaming_program is None:
        raise HTTPException(status_code=404, detail="Streaming not enabled")

    async def event_generator():
        start = time.time()
        try:
            output = streaming_program(question=request.question)
            async for chunk in output:
                if isinstance(chunk, dspy.streaming.StreamResponse):
                    data = json.dumps({
                        "type": "token",
                        "field": chunk.signature_field_name,
                        "content": chunk.chunk,
                    })
                    yield f"data: {data}\n\n"
                elif isinstance(chunk, dspy.Prediction):
                    latency = time.time() - start
                    usage = (
                        chunk.get_lm_usage() if settings.TRACK_USAGE else {}
                    )
                    metrics.record_request(latency, usage)
                    data = json.dumps({
                        "type": "complete",
                        "answer": chunk.answer,
                        "usage": usage,
                        "latency_seconds": round(latency, 3),
                    })
                    yield f"data: {data}\n\n"
        except Exception as e:
            metrics.record_error()
            logger.error(f"Streaming failed: {e}", exc_info=True)
            data = json.dumps({"type": "error", "detail": str(e)})
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/batch")
async def batch_predict(questions: list[PredictRequest]):
    async def process_one(q: PredictRequest):
        try:
            result = await program.acall(question=q.question)
            return {
                "question": q.question,
                "answer": result.answer,
                "error": None,
            }
        except Exception as e:
            return {
                "question": q.question,
                "answer": None,
                "error": str(e),
            }

    results = await asyncio.gather(*[process_one(q) for q in questions])
    return {"results": results, "count": len(results)}


# =====================================================
# Section 7: Optimization Script
# =====================================================

def run_optimization():
    """Run MIPROv2 optimization and save state."""
    dspy.configure(
        lm=dspy.LM(settings.MODEL), track_usage=True,
    )

    program = QAProgram()

    trainset = [
        dspy.Example(
            question="What is photosynthesis?",
            answer=(
                "Photosynthesis is the process by which plants convert "
                "sunlight, water, and CO2 into glucose and oxygen."
            ),
        ).with_inputs("question"),
        dspy.Example(
            question="What causes earthquakes?",
            answer=(
                "Earthquakes are caused by tectonic plate movements "
                "releasing stored energy."
            ),
        ).with_inputs("question"),
    ]

    def metric(example, prediction, trace=None):
        return len(prediction.answer) > 20

    optimizer = dspy.MIPROv2(metric=metric, auto="light")
    optimized = optimizer.compile(program, trainset=trainset)
    optimized.save("optimized_state.json", save_program=False)
    print("Saved optimized program state")


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "optimize":
        run_optimization()
    else:
        import uvicorn
        uvicorn.run(
            "01_deployment:app",
            host=settings.HOST,
            port=settings.PORT,
            workers=settings.WORKERS,
            reload=True,
        )
