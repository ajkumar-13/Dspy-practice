# 11.P: Project: Production-Ready DSPy API

## Introduction

You've learned caching, async programming, streaming, deployment, and debugging across this entire phase. Now it's time to build a **complete, production-ready API** that ties everything together, from an optimized DSPy program to a deployed, monitored FastAPI service with streaming, health checks, error handling, and Docker containerization.

This is the capstone project for Phase 11. By the end, you'll have a deployable service that you can use as a template for any DSPy-powered API.

---

## What You'll Build

1. **A DSPy program**: a question-answering pipeline with chain-of-thought reasoning
2. **An optimization script**: optimize and save the program state
3. **A FastAPI application**: async endpoints, streaming via SSE, health checks, metrics
4. **Error handling middleware**: graceful failure with proper HTTP status codes
5. **Caching configuration**: production-tuned cache layers
6. **Docker deployment**: Dockerfile and docker-compose for containerized deployment
7. **Monitoring and logging**: structured logging, usage tracking, performance metrics

---

## Prerequisites

- Completed [11.4: Debugging and Observability](../11.4-debugging-observability/blog.md)
- DSPy installed (`uv add dspy python-dotenv fastapi uvicorn httpx`)

---

## Project Structure

```
production-api/
    program.py          # DSPy program definition
    optimize.py         # Optimization script
    config.py           # Configuration management
    app.py              # FastAPI application
    .env                # Environment variables
    Dockerfile          # Container definition
    docker-compose.yml  # Multi-container setup
    test_api.py         # API test suite
```

---

## File 1: `program.py`

Start with a clean program definition. This module performs question answering with context analysis, a pattern that benefits significantly from optimization.

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

Key design decisions:
- **Chain-of-thought** for better reasoning quality
- **Multiple output fields** to get structured metadata alongside the answer
- **Both `forward` and `aforward`** so the program works in sync and async contexts

---

## File 2: `optimize.py`

Run this once to optimize the program and save its state. You deploy the saved state, not the optimizer:

```python
# optimize.py
import dspy
from program import QAProgram
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)

program = QAProgram()

# Training examples
trainset = [
    dspy.Example(
        question="What is photosynthesis?",
        answer="Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen using chlorophyll.",
    ).with_inputs("question"),
    dspy.Example(
        question="What causes earthquakes?",
        answer="Earthquakes occur when tectonic plates along fault lines suddenly release accumulated stress, sending seismic waves through the Earth.",
    ).with_inputs("question"),
    dspy.Example(
        question="How does a neural network learn?",
        answer="Neural networks learn through backpropagation, adjusting connection weights to minimize the difference between predicted and actual outputs across training examples.",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the greenhouse effect?",
        answer="The greenhouse effect is when gases like CO2 and methane trap heat in Earth's atmosphere, warming the surface beyond what direct sunlight alone would achieve.",
    ).with_inputs("question"),
]


def metric(example, prediction, trace=None):
    """Score based on answer quality."""
    answer = prediction.answer.lower()
    # Reward longer, more detailed answers
    length_ok = len(prediction.answer) > 50
    # Reward mentioning key concepts
    has_concepts = hasattr(prediction, "key_concepts") and len(prediction.key_concepts) > 10
    return length_ok and has_concepts


# Optimize
optimizer = dspy.MIPROv2(metric=metric, auto="light")
optimized = optimizer.compile(program, trainset=trainset)

# Save state only (recommended for deployment)
optimized.save("optimized_state.json", save_program=False)

# Verify
usage = optimized(question="What is gravity?").get_lm_usage()
print(f"Optimization complete. Test usage: {usage}")
print("Saved to optimized_state.json")
```

---

## File 3: `config.py`

Centralize all configuration with environment variable overrides:

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

Example `.env`:

```env
OPENAI_API_KEY=sk-...
DSPY_MODEL=openai/gpt-4o-mini
PROGRAM_STATE_PATH=optimized_state.json
TRACK_USAGE=true
ENABLE_STREAMING=true
ENABLE_METRICS=true
CACHE_DISK_ENABLED=true
CACHE_MEMORY_ENABLED=true
CACHE_DISK_SIZE=5000000000
CACHE_MEMORY_ENTRIES=2000000
```

---

## File 4: `app.py`

This is the core of the project, a complete production API:

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dspy_api")


# --- Metrics Tracker ---

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
        avg_lat = sum(self.latencies) / max(len(self.latencies), 1)
        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": round(self.total_errors / max(self.total_requests, 1), 4),
            "avg_latency_seconds": round(avg_lat, 3),
            "p95_latency_seconds": round(
                sorted(self.latencies)[int(len(self.latencies) * 0.95)]
                if self.latencies else 0, 3
            ),
            "total_tokens": self.total_tokens,
            "requests_per_second": round(self.total_requests / max(uptime, 1), 2),
        }


metrics = MetricsTracker()


# --- Lifespan ---

program: QAProgram = None
streaming_program = None


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
        logger.info(f"Loaded optimized state from {settings.PROGRAM_STATE_PATH}")

    # Set up streaming
    if settings.ENABLE_STREAMING:
        listener = StreamListener(signature_field_name="answer")
        streaming_program = dspy.streamify(program, stream_listeners=[listener])
        logger.info("Streaming endpoint enabled")

    logger.info(f"API ready | model={settings.MODEL} | streaming={settings.ENABLE_STREAMING}")
    yield
    logger.info("API shutting down")


# --- FastAPI App ---

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


# --- Models ---

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


# --- Endpoints ---

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": settings.MODEL,
        "optimized": settings.PROGRAM_STATE_PATH is not None,
        "streaming": settings.ENABLE_STREAMING,
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
            "X-Accel-Buffering": "no",  # Disable nginx buffering
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

## File 5: `Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency specification
COPY pyproject.toml ./

# Install dependencies
RUN uv pip install --system dspy python-dotenv fastapi uvicorn httpx

# Copy application code
COPY app.py program.py config.py ./

# Copy optimized program state (if exists)
COPY optimized_state.json* ./

# Create cache directory
RUN mkdir -p /app/.cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn — 4 workers for production
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

## File 6: `docker-compose.yml`

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

---

## Running the Project

### Local Development

```bash
# 1. Optimize the program (run once)
python optimize.py

# 2. Start the API
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t dspy-api .
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY dspy-api
```

### Production (Multi-Worker)

```bash
# Run with 4 workers for production
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Testing the API

### Quick Tests with curl

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'

# Stream
curl -N -X POST http://localhost:8000/predict/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain quantum computing"}'

# Batch
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '[{"question": "What is AI?"}, {"question": "What is ML?"}]'

# Metrics
curl http://localhost:8000/metrics
```

### Python Test Script

```python
# test_api.py
import asyncio
import time
import httpx


async def test_api():
    base = "http://localhost:8000"

    async with httpx.AsyncClient(timeout=30) as client:
        # Health
        r = await client.get(f"{base}/health")
        print(f"Health: {r.json()}")

        # Predict
        r = await client.post(
            f"{base}/predict",
            json={"question": "What is photosynthesis?"},
        )
        data = r.json()
        print(f"Predict: {data['answer'][:80]}...")
        print(f"Latency: {data['latency_seconds']}s")

        # Batch
        r = await client.post(
            f"{base}/batch",
            json=[
                {"question": "What is AI?"},
                {"question": "What is ML?"},
                {"question": "What is DL?"},
            ],
        )
        batch = r.json()
        print(f"Batch: {batch['count']} results")
        for item in batch["results"]:
            status = "OK" if item["answer"] else f"ERROR: {item['error']}"
            print(f"  - {item['question']}: {status}")

        # Stream
        print("\nStreaming:")
        async with client.stream(
            "POST",
            f"{base}/predict/stream",
            json={"question": "Explain gravity briefly"},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    content = line[6:]
                    if content == "[DONE]":
                        print("\n[DONE]")
                        break
                    import json
                    chunk = json.loads(content)
                    if chunk["type"] == "token":
                        print(chunk["content"], end="", flush=True)
                    elif chunk["type"] == "complete":
                        print(f"\nComplete: latency={chunk['latency_seconds']}s")

        # Metrics
        r = await client.get(f"{base}/metrics")
        print(f"\nMetrics: {r.json()}")


asyncio.run(test_api())
```

---

## Load Testing

For production readiness, run basic load tests:

```python
# load_test.py
import asyncio
import time
import httpx


async def run_load_test(concurrent_requests=5, total_requests=20):
    base = "http://localhost:8000"
    questions = [
        "What is machine learning?",
        "How does DNA replication work?",
        "What causes inflation?",
        "Explain the water cycle",
        "What is dark energy?",
    ]

    results = []
    start = time.time()

    async with httpx.AsyncClient(timeout=60) as client:
        sem = asyncio.Semaphore(concurrent_requests)

        async def make_request(i):
            async with sem:
                q = questions[i % len(questions)]
                req_start = time.time()
                r = await client.post(
                    f"{base}/predict",
                    json={"question": q},
                )
                return {
                    "status": r.status_code,
                    "latency": time.time() - req_start,
                }

        tasks = [make_request(i) for i in range(total_requests)]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start

    # Report
    latencies = [r["latency"] for r in results]
    errors = sum(1 for r in results if r["status"] != 200)

    print(f"\n--- Results ---")
    print(f"Total time: {total_time:.1f}s")
    print(f"Requests: {total_requests} ({errors} errors)")
    print(f"Throughput: {total_requests / total_time:.1f} req/s")
    print(f"Avg latency: {sum(latencies) / len(latencies):.2f}s")
    print(f"P95 latency: {sorted(latencies)[int(len(latencies) * 0.95)]:.2f}s")
    print(f"Max latency: {max(latencies):.2f}s")


asyncio.run(run_load_test(concurrent_requests=5, total_requests=20))
```

Key things to monitor during load testing:
- **Throughput**: requests per second under load
- **P95 latency**: 95th percentile response time
- **Error rate**: how many requests fail under pressure
- **Memory usage**: does the cache grow unbounded?
- **Token consumption**: are cache hits working at scale?

---

## Production Monitoring Checklist

| Component | What to Monitor | How |
|---|---|---|
| **API availability** | Uptime, health check | `/health` endpoint + alerting |
| **Latency** | Avg, P95, P99 response times | `/metrics` endpoint |
| **Errors** | Error rate, error types | Structured logging |
| **Token usage** | Prompt + completion tokens | `get_lm_usage()` in metrics |
| **Cache performance** | Hit rate, cache size | Cache statistics |
| **Costs** | Dollar spend per day | Token usage x model pricing |
| **Memory** | RAM usage over time | Container metrics |

---

## Key Takeaways

1. **Separate optimization from serving**: run `optimize.py` once, deploy `app.py` repeatedly
2. **Use `lifespan` for init**: load the program once at startup, not per-request
3. **`acall()` everywhere**: non-blocking async for all endpoints
4. **SSE for streaming**: `dspy.streamify()` + `StreamingResponse` for real-time output
5. **Batch endpoints**: `asyncio.gather()` for concurrent multi-question handling
6. **Centralize configuration**: environment variables with sensible defaults
7. **Monitor everything**: latency, tokens, errors, cache hits

---

**Next up:** [Phase 12: Advanced Architectures and Research](../../12-advanced/12.1-real-world-architectures/blog.md) explores real-world DSPy architectures, research papers, and contributing to the framework.

---

## Resources

- [DSPy Documentation](https://dspy.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [DSPy Saving/Loading](https://dspy.ai/learn/saving_loading/)
- [DSPy Streaming](https://dspy.ai/learn/programming/streaming/)
- [DSPy Async](https://dspy.ai/learn/programming/async/)
