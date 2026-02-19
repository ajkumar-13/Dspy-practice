"""
Blog 9.1: RL Optimization with ArborGRPO
Requires: pip install -U dspy arbor-ai peft
Requires: Multiple GPUs (4xH100 recommended)
EXPERIMENTAL: proof of concept, not production-ready
"""

import dspy
import arbor
from arbor import ArborProvider
from peft import LoraConfig
from dspy.teleprompt import ArborGRPO
from dspy.evaluate import Evaluate
from dotenv import load_dotenv

load_dotenv()


# =====================================================
# Step 1: Initialize Arbor and Model
# =====================================================

arbor.init()

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
arbor_server_info = arbor.start_server(model_name)

lm = dspy.LM(
    model=f"openai/arbor:{model_name}",
    provider=ArborProvider(),
    api_base=arbor_server_info["base_url"],
)

dspy.configure(lm=lm)
print(f"Arbor server running at: {arbor_server_info['base_url']}")


# =====================================================
# Step 2: Define DSPy Program
# =====================================================

class SimpleQA(dspy.Module):
    def __init__(self):
        self.answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.answer(question=question)


program = SimpleQA()


# =====================================================
# Step 3: Define Metric and Data
# =====================================================

def answer_metric(example, prediction, trace=None):
    """Simple exact-match metric."""
    gold = example.answer.lower().strip()
    pred = prediction.answer.lower().strip()
    return gold in pred or pred in gold


trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What year did WWII end?", answer="1945").with_inputs("question"),
    dspy.Example(question="Who wrote Romeo and Juliet?", answer="Shakespeare").with_inputs("question"),
    dspy.Example(question="What is the largest planet?", answer="Jupiter").with_inputs("question"),
    dspy.Example(question="What is the speed of light in m/s?", answer="299792458").with_inputs("question"),
    # ... add hundreds more examples for meaningful RL training
]

devset = trainset[:3]


# =====================================================
# Step 4: Configure RL Training
# =====================================================

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

train_kwargs = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "temperature": 1.0,
    "learning_rate": 5e-6,
    "lora_config": lora_config,
    "num_training_gpus": 2,
    "num_inference_gpus": 2,
    "loss_type": "dapo",
    "report_to": "wandb",
}


# =====================================================
# Step 5: Compile with ArborGRPO
# =====================================================

compiler = ArborGRPO(
    metric=answer_metric,
    num_dspy_examples_per_grpo_step=8,
    num_rollouts_per_grpo_step=4,
    exclude_demos=True,
    num_train_steps=100,
    checkpoint="checkpoints/rl_qa",
    train_kwargs=train_kwargs,
)

print("Starting RL training...")
print("This will take several hours on 4xH100 GPUs.")

optimized_program = compiler.compile(
    student=program,
    trainset=trainset,
    valset=devset,
)


# =====================================================
# Step 6: Evaluate
# =====================================================

evaluator = Evaluate(
    devset=devset,
    metric=answer_metric,
    num_threads=4,
    display_progress=True,
)

score = evaluator(optimized_program)
print(f"RL-optimized score: {score}%")
