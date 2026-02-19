# 11.3: Deploying DSPy Applications

## Introduction

You've built a DSPy program, optimized it, and it runs perfectly in your notebook. Now you need to deploy it as a real service that handles actual user traffic. This means packaging your program into an API server, containerizing it for consistent deployment, managing configuration, and setting up proper health checks and monitoring.

This post walks through the complete deployment journey: from a standalone DSPy program to a containerized API service.

---

## What You'll Learn

- Wrapping DSPy programs in a FastAPI application
- Managing startup and shutdown with `lifespan` events
- Streaming endpoints with Server-Sent Events (SSE)
- Configuration management with environment variables
- Docker containerization for consistent deployment
- Health checks and graceful shutdown
- Multi-worker deployment with Uvicorn

---

## Prerequisites

- Completed [11.2: Async Programming and Streaming in DSPy](../11.2-async-streaming/blog.md)
- DSPy installed (`uv add dspy python-dotenv fastapi uvicorn httpx`)
- Familiarity with FastAPI basics
- An OpenAI API key (or any LiteLLM-supported provider)

---

## Step 1: Define Your DSPy Program

Start with a clean program definition. This module performs question answering with context analysis, a pattern that benefits significantly from optimization:

```python
# program.py
import dspy


class QASignature(dspy.Signature):
    """Answer a question with detailed reasoning."""
    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="A clear, comprehensive answer")
    key_concepts: str = dspy.OutputField(desc="Key concepts used in the answer")
    question_type: str = dspy.OutputField(desc="Type of question: factual, opinion, analytical")


class QAProgram(dspy.Module):
    def __init__(self):
        self.predict = dspy.ChainOfThought(QASignature)

    def forward(self, question, **kwargs):
        return self.predict(question=question)

    async def aforward(self, question, **kwargs):
        return await self.predict.acall(question=question)
```

### Optimize and Save

Run optimization once, then deploy the saved state:

```python
# optimize.py
import dspy
from program import QAProgram
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)

program = QAProgram()

# Create training data
trainset = [
    dspy.Example(
        question="What is photosynthesis?",
        answer="Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
    ).with_inputs("question"),
    dspy.Example(
        question="What causes earthquakes?",
        answer="Earthquakes are caused by the sudden release of energy from tectonic plate movements.",
    ).with_inputs("question"),
]

# Optimize with MIPROv2
optimizer = dspy.MIPROv2(metric=lambda example, prediction, trace=None: len(prediction.answer) > 20, auto="light")
optimized = optimizer.compile(program, trainset=trainset)

# Save the optimized state
optimized.save("optimized_state.json", save_program=False)
print("Saved optimized program state")
```

---

## Step 2: Configuration Management

Centralize all configuration in a dedicated module using environment variables with sensible defaults:

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LM
    MODEL: str = os.getenv("DSPY_MODEL", "openai/gpt-4o-mini")
    TRACK_USAGE: bool = os.getenv("TRACK_USAGE", "true").lower() == "true"

    # Cache
    CACHE_DISK_ENABLED: bool = os.getenv("CACHE_DISK_ENABLED", "true").lower() == "true"
    CACHE_MEMORY_ENABLED: bool = os.getenv("CACHE_MEMORY_ENABLED", "true").lower() == "true"
    CACHE_DISK_SIZE: int = int(os.getenv("CACHE_DISK_SIZE", str(5_000_000_000)))
    CACHE_MEMORY_ENTRIES: int = int(os.getenv("CACHE_MEMORY_ENTRIES", str(2_000_000)))

    # Program
    PROGRAM_STATE_PATH: str | None = os.getenv("PROGRAM_STATE_PATH")

    # Features
    ENABLE_STREAMING: bool = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "1"))


settings = Settings()
```

Example `.env` file:

```env
OPENAI_API_KEY=sk-...
DSPY_MODEL=openai/gpt-4o-mini
PROGRAM_STATE_PATH=optimized_state.json
TRACK_USAGE=true
ENABLE_STREAMING=true
ENABLE_METRICS=true
```

---

## Step 3: Build the FastAPI Application

The main application ties everything together: program loading, endpoints, streaming, and monitoring.

### Application Lifecycle

Use FastAPI's `lifespan` context manager to initialize the DSPy program once at startup:

```python
# app.py
import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import dspy
from dspy.streaming import StreamListener
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from program import QAProgram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dspy_api")

# Globals
program: QAProgram = None
streaming_program = None
```

### Metrics Tracker

A simple in-memory metrics tracker for latency, throughput, and token usage:

```python
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
            "error_rate": round(self.total_errors / max(self.total_requests, 1), 4),
            "avg_latency": round(sum(self.latencies) / max(len(self.latencies), 1), 3),
            "total_tokens": self.total_tokens,
        }


metrics = MetricsTracker()
```

### Lifespan Handler

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global program, streaming_program

    # Configure DSPy
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

    # Initialize program
    program = QAProgram()
    if settings.PROGRAM_STATE_PATH:
        program.load(settings.PROGRAM_STATE_PATH)
        logger.info(f"Loaded optimized state: {settings.PROGRAM_STATE_PATH}")

    # Set up streaming
    if settings.ENABLE_STREAMING:
        listener = StreamListener(signature_field_name="answer")
        streaming_program = dspy.streamify(program, stream_listeners=[listener])
        logger.info("Streaming enabled")

    logger.info(f"API ready | Model: {settings.MODEL}")
    yield
    logger.info("Shutting down")
```

### FastAPI App with Middleware

```python
app = FastAPI(
    title="DSPy Production API",
    description="A production-ready question-answering API powered by DSPy",
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
```

### Request/Response Models

```python
class PredictRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000, description="The question to answer")


class PredictResponse(BaseModel):
    answer: str
    key_concepts: str | None = None
    question_type: str | None = None
    usage: dict | None = None
    latency_seconds: float | None = None
```

### Endpoints

```python
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
            f"Predicted in {latency:.3f}s | Q: {request.question[:50]}... | Tokens: {usage}"
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
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
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
                    usage = chunk.get_lm_usage() if settings.TRACK_USAGE else {}
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
            return {"question": q.question, "answer": result.answer, "error": None}
        except Exception as e:
            return {"question": q.question, "answer": None, "error": str(e)}

    results = await asyncio.gather(*[process_one(q) for q in questions])
    return {"results": results, "count": len(results)}
```

---

## Step 4: Docker Containerization

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml ./

RUN uv pip install --system dspy python-dotenv fastapi uvicorn httpx

COPY app.py program.py config.py ./
COPY optimized_state.json* ./

RUN mkdir -p /app/.cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### docker-compose.yml

```yaml
version: "3.8"

services:
  dspy-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DSPY_MODEL=openai/gpt-4o-mini
      - PROGRAM_STATE_PATH=optimized_state.json
      - TRACK_USAGE=true
      - ENABLE_STREAMING=true
      - ENABLE_METRICS=true
      - CACHE_DISK_ENABLED=true
      - CACHE_MEMORY_ENABLED=true
    volumes:
      - cache-data:/app/.cache
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G

volumes:
  cache-data:
```

### Build and Run

```bash
# Build the image
docker build -t dspy-api .

# Run with environment variables
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e PROGRAM_STATE_PATH=optimized_state.json \
  dspy-api

# Or use docker-compose
docker-compose up --build
```

---

## Step 5: Running Locally (Development)

For local development without Docker:

```bash
# Run with uvicorn (single worker for dev)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Run with multiple workers for production
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Key Takeaways

- **Separate optimization from serving**: run `optimize.py` once, deploy `app.py` repeatedly.
- **Use `lifespan` for initialization**: load the program once at startup, not per-request.
- **`acall()` for all endpoints**: non-blocking async ensures high throughput.
- **SSE for streaming**: `dspy.streamify()` + `StreamingResponse` delivers real-time output.
- **Centralize configuration**: environment variables with sensible defaults in `config.py`.
- **Docker for consistency**: containerize for reproducible deployments.
- **Health checks**: `/health` endpoint for container orchestration and monitoring.

---

**Next up:** [11.4: Debugging and Observability](../11.4-debugging-observability/blog.md) covers inspecting LM history, logging, usage tracking, and building custom observability tools.

---

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [DSPy Async Documentation](https://dspy.ai/learn/programming/async/)
