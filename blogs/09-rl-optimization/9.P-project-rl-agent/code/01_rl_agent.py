"""
Blog 9.P: RL-Optimized Research Agent
Requires: pip install -U dspy arbor-ai bm25s peft python-dotenv wandb
Requires: Multiple GPUs (4xH100 recommended)
EXPERIMENTAL: proof of concept, not production-ready

Usage:
    python 01_rl_agent.py --mode baseline
    python 01_rl_agent.py --mode mipro
    python 01_rl_agent.py --mode rl
    python 01_rl_agent.py --mode compare
"""

import argparse
import json
import random
import re

import bm25s
import dspy
from dotenv import load_dotenv
from dspy.evaluate import Evaluate

load_dotenv()


# =====================================================
# Data Loading
# =====================================================


def load_corpus(path="data/wiki_abstracts.jsonl"):
    """Load Wikipedia abstracts corpus (JSONL format)."""
    corpus = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    print(f"Loaded {len(corpus):,} documents")
    return corpus


def load_hover_data(path="data/hover_train.jsonl", max_examples=500):
    """Load HoVer dataset for multi-hop fact verification."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            examples.append(
                dspy.Example(
                    claim=item["claim"],
                    gold_titles=item["supporting_facts_titles"],
                    label=item["label"],
                ).with_inputs("claim")
            )
    random.seed(42)
    random.shuffle(examples)
    return examples[:max_examples]


# =====================================================
# BM25 Retriever
# =====================================================


class BM25Retriever:
    """BM25S-based retriever for Wikipedia abstracts."""

    def __init__(self, corpus, k=5):
        self.k = k
        self.corpus = corpus
        self.retriever = bm25s.BM25()
        tokenized = bm25s.tokenize([d["text"] for d in corpus])
        self.retriever.index(tokenized)
        print(f"BM25 index built over {len(corpus):,} documents")

    def __call__(self, query):
        tokenized_query = bm25s.tokenize(query)
        results, scores = self.retriever.retrieve(tokenized_query, k=self.k)
        passages = []
        for idx in results[0]:
            doc = self.corpus[idx]
            passages.append(f"[{doc['title']}] {doc['text']}")
        return "\n\n".join(passages)


# =====================================================
# DSPy Multi-Hop Research Agent
# =====================================================


class ResearchHop(dspy.Module):
    """A single research hop: search + update notes."""

    def __init__(self):
        self.generate_query = dspy.ChainOfThought("claim, notes -> search_query")
        self.append_notes = dspy.ChainOfThought("claim, notes, passages -> updated_notes: str")

    def forward(self, claim, notes, retriever):
        query_pred = self.generate_query(claim=claim, notes=notes)
        passages = retriever(query_pred.search_query)
        notes_pred = self.append_notes(claim=claim, notes=notes, passages=passages)
        return notes_pred.updated_notes


class MultiHopResearchAgent(dspy.Module):
    """Complete multi-hop research agent with N hops."""

    def __init__(self, num_hops=3):
        self.hops = [ResearchHop() for _ in range(num_hops)]
        self.verify = dspy.ChainOfThought("claim, notes -> verdict: bool")

    def forward(self, claim, retriever):
        notes = "No research notes yet."
        for hop in self.hops:
            notes = hop(claim=claim, notes=notes, retriever=retriever)
        result = self.verify(claim=claim, notes=notes)
        return dspy.Prediction(verdict=result.verdict, notes=notes)


# =====================================================
# Evaluation Metrics
# =====================================================


def extract_titles_from_notes(notes):
    """Extract Wikipedia titles from [brackets] in notes."""
    return set(re.findall(r"\[([^\]]+)\]", notes))


def recall_metric(example, prediction, trace=None):
    """Recall over gold Wikipedia titles."""
    gold_titles = set(example.gold_titles)
    if not gold_titles:
        return 1.0
    found = extract_titles_from_notes(prediction.notes)
    hits = sum(1 for t in gold_titles if any(t.lower() in f.lower() for f in found))
    return hits / len(gold_titles)


# =====================================================
# Main Entry Point
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-Optimized Research Agent")
    parser.add_argument(
        "--mode",
        choices=["baseline", "mipro", "rl", "compare"],
        default="baseline",
        help="Which optimization mode to run",
    )
    args = parser.parse_args()

    # Load data
    corpus = load_corpus()
    retriever = BM25Retriever(corpus, k=5)
    all_examples = load_hover_data()
    trainset = all_examples[:400]
    devset = all_examples[400:]
    print(f"Train: {len(trainset)}, Dev: {len(devset)}")

    # Baseline evaluation
    if args.mode in ("baseline", "compare"):
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm)
        program = MultiHopResearchAgent(num_hops=3)

        class EvalWrapper(dspy.Module):
            def __init__(self, inner, ret):
                super().__init__()
                self.inner = inner
                self.ret = ret

            def forward(self, claim):
                return self.inner(claim=claim, retriever=self.ret)

        evaluator = Evaluate(
            devset=devset,
            metric=recall_metric,
            num_threads=4,
            display_progress=True,
        )
        baseline_score = evaluator(EvalWrapper(program, retriever))
        print(f"Baseline: {baseline_score:.1f}%")

    # MIPROv2 optimization
    if args.mode in ("mipro", "compare"):
        from dspy.teleprompt import MIPROv2

        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm)
        optimizer = MIPROv2(metric=recall_metric, num_candidates=10)
        optimized = optimizer.compile(
            MultiHopResearchAgent(num_hops=3),
            trainset=trainset,
            num_batches=30,
            max_bootstrapped_demos=3,
            max_labeled_demos=5,
        )
        evaluator = Evaluate(
            devset=devset,
            metric=recall_metric,
            num_threads=4,
            display_progress=True,
        )
        mipro_score = evaluator(EvalWrapper(optimized, retriever))
        print(f"MIPROv2: {mipro_score:.1f}%")

    # ArborGRPO RL training
    if args.mode in ("rl", "compare"):
        import arbor
        from arbor import ArborProvider
        from dspy.teleprompt import ArborGRPO
        from peft import LoraConfig

        arbor.init()
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        server_info = arbor.start_server(model_name)

        lm = dspy.LM(
            model=f"openai/arbor:{model_name}",
            provider=ArborProvider(),
            api_base=server_info["base_url"],
        )
        dspy.configure(lm=lm)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        compiler = ArborGRPO(
            metric=recall_metric,
            num_dspy_examples_per_grpo_step=4,
            num_rollouts_per_grpo_step=4,
            exclude_demos=True,
            num_train_steps=300,
            checkpoint="checkpoints/rl_research_agent",
            train_kwargs={
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "temperature": 1.0,
                "learning_rate": 5e-6,
                "lora_config": lora_config,
                "num_training_gpus": 2,
                "num_inference_gpus": 2,
                "loss_type": "dapo",
                "report_to": "wandb",
            },
        )

        print("Starting RL training (this will take ~18 hours)...")
        rl_optimized = compiler.compile(
            student=MultiHopResearchAgent(num_hops=3),
            trainset=trainset,
            valset=devset,
        )

        evaluator = Evaluate(
            devset=devset,
            metric=recall_metric,
            num_threads=4,
            display_progress=True,
        )
        rl_score = evaluator(EvalWrapper(rl_optimized, retriever))
        print(f"ArborGRPO: {rl_score:.1f}%")
