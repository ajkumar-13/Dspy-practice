# DSPy Mastery: Zero to Pro

**A comprehensive blog series covering DSPy from your first `dspy.Predict` to production-grade, self-optimizing AI systems.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![DSPy](https://img.shields.io/badge/DSPy-2.6+-green.svg)](https://dspy.ai)
[![CI](https://github.com/ajkumar-13/Dspy-practice/actions/workflows/ci.yml/badge.svg)](https://github.com/ajkumar-13/Dspy-practice/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://docs.astral.sh/uv/)

---

## 📖 What Is This?

A **42-part blog series** that teaches you DSPy, Stanford's framework for programming (not prompting) language models. Each blog is a self-contained lesson with explanations, runnable code examples, and hands-on projects.

**By the end, you'll be able to:**
- Build LLM applications without writing a single prompt
- Create self-optimizing AI pipelines that improve automatically
- Deploy production-ready systems with caching, streaming, and observability
- Work with cutting-edge techniques: GEPA, RLM, RL optimization, multi-modal

## 🏗️ Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/learn-dspy.git
cd learn-dspy

# Install uv (if not already installed)
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (OpenAI, Anthropic, etc.)

# Run your first example
uv run blogs/01-foundations/1.1-setup-and-philosophy/code/01_basic_setup.py
```

## 🗺️ Series Overview

### Phase 1 — Foundations: Think in Signatures, Not Prompts
> *Learn the building blocks that make DSPy different from every other LLM framework.*

| # | Blog | Description |
|---|------|-------------|
| 1.1 | [Setup & Philosophy](blogs/01-foundations/1.1-setup-and-philosophy/blog.md) | Install DSPy, configure LMs, understand Programming > Prompting |
| 1.2 | [Signatures](blogs/01-foundations/1.2-signatures/blog.md) | Inline & class-based signatures — the contract system |
| 1.3 | [First Modules](blogs/01-foundations/1.3-first-modules/blog.md) | Predict, ChainOfThought, ProgramOfThought |
| 1.4 | [Custom Modules](blogs/01-foundations/1.4-custom-modules/blog.md) | Build multi-step programs by composing modules |
| 1.P | [Project: Text Classifier](blogs/01-foundations/1.P-mini-project-text-classifier/blog.md) | No-prompt support ticket classifier |

### Phase 2 — Structured Outputs & Guardrails
> *Get validated, structured outputs with programmatic constraints.*

| # | Blog | Description |
|---|------|-------------|
| 2.1 | [Typed Predictors](blogs/02-structured-outputs/2.1-typed-predictors/blog.md) | Pydantic integration for structured LM outputs |
| 2.2 | [Assertions](blogs/02-structured-outputs/2.2-assertions/blog.md) | Assert & Suggest, programmatic constraints with retry |
| 2.3 | [Output Refinement](blogs/02-structured-outputs/2.3-output-refinement/blog.md) | Best-of-N sampling and iterative refinement |
| 2.P | [Project: Entity Extractor](blogs/02-structured-outputs/2.P-mini-project-entity-extractor/blog.md) | Extract structured data from legal documents |

### Phase 3 — Evaluation & Metrics
> *You can't optimize what you can't measure.*

| # | Blog | Description |
|---|------|-------------|
| 3.1 | [Building Eval Sets](blogs/03-evaluation/3.1-building-eval-sets/blog.md) | Create development sets with `dspy.Example` |
| 3.2 | [Defining Metrics](blogs/03-evaluation/3.2-defining-metrics/blog.md) | Built-in, custom, and LLM-as-Judge metrics |
| 3.3 | [Running Evaluations](blogs/03-evaluation/3.3-running-evaluations/blog.md) | Systematic benchmarking with `dspy.Evaluate` |
| 3.P | [Project: Eval Harness](blogs/03-evaluation/3.P-mini-project-eval-harness/blog.md) | Reusable evaluation framework with model comparison |

### Phase 4 — Optimization: Self-Improving Pipelines
> *DSPy's superpower, your programs optimize themselves.*

| # | Blog | Description |
|---|------|-------------|
| 4.1 | [BootstrapRS](blogs/04-optimization/4.1-bootstrap-rs/blog.md) | Few-shot example optimization |
| 4.2 | [MIPROv2](blogs/04-optimization/4.2-miprov2/blog.md) | Bayesian instruction + demo optimization |
| 4.3 | [GEPA](blogs/04-optimization/4.3-gepa/blog.md) | Reflective prompt evolution (advanced) |
| 4.4 | [Optimizer Landscape](blogs/04-optimization/4.4-optimizer-landscape/blog.md) | When to use which optimizer — decision framework |
| 4.P | [Project: Self-Optimizing RAG](blogs/04-optimization/4.P-project-self-optimizing-rag/blog.md) | Progressive optimization: BootstrapRS → MIPROv2 → GEPA |

### Phase 5 — Retrieval & RAG Pipelines
> *Build retrieval-augmented generation systems that actually work.*

| # | Blog | Description |
|---|------|-------------|
| 5.1 | [Retrieval in DSPy](blogs/05-retrieval-rag/5.1-retrieval-in-dspy/blog.md) | ColBERTv2, vector DBs, custom retrieval |
| 5.2 | [Building RAG](blogs/05-retrieval-rag/5.2-building-rag/blog.md) | Simple RAG, CoT RAG, optimized RAG |
| 5.3 | [Multi-Hop RAG](blogs/05-retrieval-rag/5.3-multi-hop-rag/blog.md) | Iterative query refinement across hops |
| 5.4 | [RAG as Agent](blogs/05-retrieval-rag/5.4-rag-as-agent/blog.md) | Dynamic, agent-driven retrieval decisions |
| 5.P | [Project: Research Assistant](blogs/05-retrieval-rag/5.P-project-research-assistant/blog.md) | Multi-hop research agent with 3+ retrieval hops |

### Phase 6 — Agents & Tool Use
> *Build autonomous agents that reason, act, and use tools.*

| # | Blog | Description |
|---|------|-------------|
| 6.1 | [ReAct Agents](blogs/06-agents/6.1-react-agents/blog.md) | Reason + Act loop with tool integration |
| 6.2 | [Advanced Tool Use](blogs/06-agents/6.2-advanced-tool-use/blog.md) | Multi-tool agents, agent optimization |
| 6.3 | [MCP Integration](blogs/06-agents/6.3-mcp-integration/blog.md) | Model Context Protocol for external tools |
| 6.4 | [Memory Agents](blogs/06-agents/6.4-memory-agents/blog.md) | Conversation history, Mem0, stateful agents |
| 6.5 | [PAPILLON](blogs/06-agents/6.5-papillon/blog.md) | Privacy-conscious task delegation |
| 6.P | [Project: Financial Analyst](blogs/06-agents/6.P-project-financial-analyst/blog.md) | Autonomous stock analysis agent |

### Phase 7 — Finetuning & Weight Optimization
> *Distill knowledge from large models into smaller, cheaper ones.*

| # | Blog | Description |
|---|------|-------------|
| 7.1 | [BootstrapFinetune](blogs/07-finetuning/7.1-bootstrap-finetune/blog.md) | Auto-generated training data from traces |
| 7.2 | [BetterTogether](blogs/07-finetuning/7.2-better-together/blog.md) | Joint prompt + weight optimization |
| 7.3 | [Ensemble Methods](blogs/07-finetuning/7.3-ensemble/blog.md) | Combine multiple programs for robustness |
| 7.P | [Project: Distillation](blogs/07-finetuning/7.P-project-distillation/blog.md) | GPT-4o → Llama 3 knowledge distillation |

### Phase 8 — RLM: Recursive Language Models 🔬
> *Process contexts 100x larger than your model's window.*

| # | Blog | Description |
|---|------|-------------|
| 8.1 | [Understanding RLM](blogs/08-rlm/8.1-understanding-rlm/blog.md) | Recursive processing of massive contexts |
| 8.2 | [Building with RLM](blogs/08-rlm/8.2-building-with-rlm/blog.md) | Practical RLM programs + retrieval combo |
| 8.P | [Project: Document Analyzer](blogs/08-rlm/8.P-project-document-analyzer/blog.md) | Analyze 500-page manuals with RLM |

### Phase 9 — RL Optimization
> *Reinforcement learning for DSPy programs.*

| # | Blog | Description |
|---|------|-------------|
| 9.1 | [RL for DSPy](blogs/09-rl-optimization/9.1-rl-for-dspy/blog.md) | RL optimization fundamentals |
| 9.2 | [RL Complex Tasks](blogs/09-rl-optimization/9.2-rl-complex-tasks/blog.md) | RL for PAPILLON, multi-hop, and more |
| 9.P | [Project: RL Agent](blogs/09-rl-optimization/9.P-project-rl-agent/blog.md) | RL-optimized research agent vs MIPROv2 vs GEPA |

### Phase 10 — Multi-Modal DSPy
> *Image, audio, and text in a single optimizable pipeline.*

| # | Blog | Description |
|---|------|-------------|
| 10.1 | [Image & Audio](blogs/10-multi-modal/10.1-image-audio/blog.md) | Vision LMs, image gen, audio processing |
| 10.2 | [Multi-Modal Pipelines](blogs/10-multi-modal/10.2-multi-modal-pipelines/blog.md) | End-to-end multi-modal optimization |
| 10.P | [Project: Content Analyzer](blogs/10-multi-modal/10.P-project-content-analyzer/blog.md) | Image + text → structured data + captions |

### Phase 11 — Production Engineering
> *Ship it. For real.*

| # | Blog | Description |
|---|------|-------------|
| 11.1 | [Caching & Performance](blogs/11-production/11.1-caching-performance/blog.md) | Built-in caching, dev vs prod strategies |
| 11.2 | [Async & Streaming](blogs/11-production/11.2-async-streaming/blog.md) | High-throughput async, real-time streaming |
| 11.3 | [Deployment](blogs/11-production/11.3-deployment/blog.md) | Save, load, and deploy as APIs |
| 11.4 | [Debugging & Observability](blogs/11-production/11.4-debugging-observability/blog.md) | Prompt inspection, tracing, MLflow |
| 11.P | [Project: Production API](blogs/11-production/11.P-project-production-api/blog.md) | FastAPI + caching + streaming + observability |

### Phase 12 — Advanced Architectures & Research
> *Push the boundaries. Contribute back.*

| # | Blog | Description |
|---|------|-------------|
| 12.1 | [Real-World Architectures](blogs/12-advanced/12.1-real-world-architectures/blog.md) | Pipelines, agent orchestration, optimization cascades, EDD, PAPILLON |
| 12.2 | [Real-World Applications](blogs/12-advanced/12.2-real-world-applications/blog.md) | Customer service, content gen, data extraction, research, code gen, education |
| 12.3 | [Research Papers](blogs/12-advanced/12.3-research-papers/blog.md) | Core DSPy paper, MIPROv2, ColBERTv2, SIMBA, GEPA, PAPILLON, Arbor |
| 12.4 | [Contributing](blogs/12-advanced/12.4-contributing/blog.md) | Contributing to the DSPy project |

## 📁 Project Structure

```
learn-dspy/
├── .github/workflows/ci.yml           # CI: lint + test on every push
├── README.md                          # This file
├── plan.md                            # Full learning plan
├── pyproject.toml                     # UV project config + dependencies
├── .python-version                    # Python 3.12
├── .env.example                       # API key template
├── .gitignore
│
├── tests/                             # Pytest test suite
│   ├── conftest.py                    # Shared fixtures
│   ├── test_signatures.py             # Signature field validation
│   ├── test_typed_predictors.py       # Pydantic model constraints
│   ├── test_examples.py              # dspy.Example & Prediction
│   ├── test_metrics.py               # Custom metric logic
│   ├── test_modules.py               # Module instantiation
│   └── test_tools.py                 # Agent tool functions
│
└── blogs/
    ├── 01-foundations/
    │   ├── 1.1-setup-and-philosophy/
    │   │   ├── blog.md                # Blog post
    │   │   └── code/                  # Runnable examples
    │   │       └── 01_basic_setup.py
    │   ├── 1.2-signatures/
    │   ├── 1.3-first-modules/
    │   ├── 1.4-custom-modules/
    │   └── 1.P-mini-project-text-classifier/
    │
    ├── 02-structured-outputs/
    │   ├── 2.1-typed-predictors/
    │   ├── 2.2-assertions/
    │   ├── 2.3-output-refinement/
    │   └── 2.P-mini-project-entity-extractor/
    │
    ├── 03-evaluation/
    │   ├── 3.1-building-eval-sets/
    │   ├── 3.2-defining-metrics/
    │   ├── 3.3-running-evaluations/
    │   └── 3.P-mini-project-eval-harness/
    │
    ├── 04-optimization/
    │   ├── 4.1-bootstrap-rs/
    │   ├── 4.2-miprov2/
    │   ├── 4.3-gepa/
    │   ├── 4.4-optimizer-landscape/
    │   └── 4.P-project-self-optimizing-rag/
    │
    ├── 05-retrieval-rag/
    │   ├── 5.1-retrieval-in-dspy/
    │   ├── 5.2-building-rag/
    │   ├── 5.3-multi-hop-rag/
    │   ├── 5.4-rag-as-agent/
    │   └── 5.P-project-research-assistant/
    │
    ├── 06-agents/
    │   ├── 6.1-react-agents/
    │   ├── 6.2-advanced-tool-use/
    │   ├── 6.3-mcp-integration/
    │   ├── 6.4-memory-agents/
    │   ├── 6.5-papillon/
    │   └── 6.P-project-financial-analyst/
    │
    ├── 07-finetuning/
    │   ├── 7.1-bootstrap-finetune/
    │   ├── 7.2-better-together/
    │   ├── 7.3-ensemble/
    │   └── 7.P-project-distillation/
    │
    ├── 08-rlm/
    │   ├── 8.1-understanding-rlm/
    │   ├── 8.2-building-with-rlm/
    │   └── 8.P-project-document-analyzer/
    │
    ├── 09-rl-optimization/
    │   ├── 9.1-rl-for-dspy/
    │   ├── 9.2-rl-complex-tasks/
    │   └── 9.P-project-rl-agent/
    │
    ├── 10-multi-modal/
    │   ├── 10.1-image-audio/
    │   ├── 10.2-multi-modal-pipelines/
    │   └── 10.P-project-content-analyzer/
    │
    ├── 11-production/
    │   ├── 11.1-caching-performance/
    │   ├── 11.2-async-streaming/
    │   ├── 11.3-deployment/
    │   ├── 11.4-debugging-observability/
    │   └── 11.P-project-production-api/
    │
    └── 12-advanced/
        ├── 12.1-real-world-architectures/
        ├── 12.2-real-world-applications/
        ├── 12.3-research-papers/
        └── 12.4-contributing/
```

## 🛠️ Development

### Prerequisites

- **Python 3.12+**
- **uv** — [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
- **API keys** for OpenAI, Anthropic, or a local model via Ollama

### Running Examples

```bash
# Run any code example
uv run blogs/01-foundations/1.1-setup-and-philosophy/code/01_basic_setup.py

# Install optional dependencies for specific phases
uv sync --extra rag          # Phase 5: ChromaDB, FAISS
uv sync --extra finetuning   # Phase 7: Transformers, Datasets
uv sync --extra production   # Phase 11: FastAPI, MLflow
uv sync --extra dev          # Development tools: Jupyter, Ruff, Pytest
```

### Running Tests

```bash
# Run all tests (no LLM API key required)
uv run python -m pytest tests/ -v
```

Tests validate signatures, Pydantic models, metrics, tool functions, and module structure without calling any LLM.

### Linting

```bash
# Check code style
uv run ruff check .
uv run ruff format --check .
```

### Using a Local Model (Free)

You can follow along without paid API keys using [Ollama](https://ollama.com/):

```bash
# Install Ollama and pull a model
ollama pull llama3.2

# Use it in any example by changing the LM config:
# dspy.configure(lm=dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434"))
```

## 💡 Key Insight

> **Don't skip Phases 3-4 (Evaluation & Optimization).** Most people jump straight to building agents. But DSPy's superpower is *systematic optimization* - without a good metric and eval set, you're just writing regular code with extra steps.

## 📚 Resources

| Resource | Link |
|----------|------|
| DSPy Docs | [dspy.ai](https://dspy.ai) |
| DSPy GitHub | [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) |
| DSPy Discord | [Community](https://discord.gg/XCGy2WDCQB) |
| DSPy Paper | [arxiv.org/abs/2310.03714](https://arxiv.org/abs/2310.03714) |

## 📄 License

This project is licensed under the MIT License, see the [LICENSE](LICENSE) file for details.

---

**Star this repo** ⭐ if you find it useful, it helps others discover the series!
