"""
7.4: Local Distillation - API to Ollama
========================================
Complete workflow for distilling an optimized API-based DSPy program
to a local model running via Ollama.
"""

import time

import dspy
from dspy.evaluate import Evaluate, SemanticF1

# ---------------------------------------------------------------------------
# Step 1: Define the task and training data
# ---------------------------------------------------------------------------

class TechnicalQA(dspy.Signature):
    """Answer technical programming questions with clear, accurate explanations."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Clear, concise technical explanation")


trainset = [
    dspy.Example(
        question="What is a race condition?",
        answer="A race condition occurs when two or more threads access shared data concurrently, and the outcome depends on the order of execution.",
    ).with_inputs("question"),
    dspy.Example(
        question="Explain the difference between processes and threads.",
        answer="Processes are independent execution units with their own memory space. Threads are lighter-weight units within a process that share the same memory.",
    ).with_inputs("question"),
    dspy.Example(
        question="What is a mutex?",
        answer="A mutex (mutual exclusion) is a synchronization primitive that prevents multiple threads from accessing a shared resource simultaneously.",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the GIL in Python?",
        answer="The Global Interpreter Lock is a mutex in CPython that allows only one thread to execute Python bytecode at a time, limiting true parallelism for CPU-bound tasks.",
    ).with_inputs("question"),
]

devset = [
    dspy.Example(
        question="What is deadlock?",
        answer="Deadlock occurs when two or more threads are each waiting for the other to release a resource, creating a circular dependency where none can proceed.",
    ).with_inputs("question"),
    dspy.Example(
        question="What is a semaphore?",
        answer="A semaphore is a synchronization primitive that controls access to a shared resource through a counter, allowing a fixed number of threads to access it simultaneously.",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the actor model?",
        answer="The actor model is a concurrency paradigm where independent actors communicate through asynchronous message passing, avoiding shared state entirely.",
    ).with_inputs("question"),
]


# ---------------------------------------------------------------------------
# Step 2: Optimize on the teacher model
# ---------------------------------------------------------------------------

def optimize_teacher():
    """Optimize the program using the API model."""
    teacher_lm = dspy.LM("openai/gpt-4o")
    dspy.configure(lm=teacher_lm)

    qa_program = dspy.ChainOfThought(TechnicalQA)

    def metric(ex, pred, trace=None):
        return SemanticF1()(ex, pred, trace)

    optimizer = dspy.MIPROv2(metric=metric, auto="light")
    optimized = optimizer.compile(qa_program, trainset=trainset)

    # Save the optimized teacher
    optimized.save("optimized_teacher", save_program=True)
    print("Teacher optimized and saved.")
    return optimized


# ---------------------------------------------------------------------------
# Step 3: Distill to local model
# ---------------------------------------------------------------------------

def distill_to_local(optimized_teacher):
    """Distill the optimized teacher to a local Ollama model."""
    student_lm = dspy.LM(
        "ollama_chat/llama3.2",
        api_base="http://localhost:11434",
    )
    dspy.configure(lm=student_lm)

    def metric(ex, pred, trace=None):
        return SemanticF1()(ex, pred, trace)

    distiller = dspy.BootstrapFinetune(metric=metric)
    distilled = distiller.compile(optimized_teacher, trainset=trainset)

    print("Distillation complete!")
    return distilled


# ---------------------------------------------------------------------------
# Step 4: Evaluate and compare
# ---------------------------------------------------------------------------

def compare(optimized_teacher, distilled):
    """Compare teacher vs student on the dev set."""
    teacher_lm = dspy.LM("openai/gpt-4o")
    student_lm = dspy.LM(
        "ollama_chat/llama3.2", api_base="http://localhost:11434"
    )

    def metric(ex, pred, trace=None):
        return SemanticF1()(ex, pred, trace)

    evaluator = Evaluate(devset=devset, metric=metric, num_threads=4)

    # Teacher score
    dspy.configure(lm=teacher_lm)
    teacher_score = evaluator(optimized_teacher)

    # Student score
    dspy.configure(lm=student_lm)
    student_score = evaluator(distilled)

    print(f"\nTeacher (GPT-4o):      {teacher_score:.1f}%")
    print(f"Student (Llama local): {student_score:.1f}%")
    print(f"Quality retention:     {student_score / teacher_score * 100:.0f}%")


# ---------------------------------------------------------------------------
# Step 5: Benchmark latency
# ---------------------------------------------------------------------------

def benchmark(program, questions, lm):
    """Benchmark a program on a list of questions."""
    dspy.configure(lm=lm)
    latencies = []
    for q in questions:
        start = time.perf_counter()
        program(question=q)
        latencies.append(time.perf_counter() - start)
    return {
        "avg_ms": sum(latencies) / len(latencies) * 1000,
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Step 1: Optimize teacher ===")
    teacher = optimize_teacher()

    print("\n=== Step 2: Distill to local ===")
    student = distill_to_local(teacher)

    print("\n=== Step 3: Compare ===")
    compare(teacher, student)
