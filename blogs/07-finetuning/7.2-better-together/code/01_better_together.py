"""
Blog 7.2: BetterTogether - Joint Optimization
Run: python 01_better_together.py

Demonstrates joint prompt+weight optimization using dspy.BetterTogether.
Combines MIPROv2 (prompt optimization) and BootstrapFinetune (weight optimization)
to co-adapt modules running on different sized models.
"""

import dspy
from dspy.evaluate import Evaluate
from dotenv import load_dotenv

load_dotenv()

# Step 1: Configure models
large_lm = dspy.LM("openai/gpt-4o")
small_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=large_lm)

# Step 2: Define a multi-step pipeline with mixed model sizes
class AnalysisPipeline(dspy.Module):
    def __init__(self):
        # Complex reasoning stays on the large model
        self.analyze = dspy.ChainOfThought(
            "document -> analysis: str, key_findings: list[str]"
        )
        # Classification moves to the small model
        self.classify = dspy.Predict(
            "analysis, key_findings -> category: str, risk_level: str"
        )

    def forward(self, document):
        result = self.analyze(document=document)
        return self.classify(
            analysis=result.analysis,
            key_findings=result.key_findings,
        )

# Step 3: Prepare training data
trainset = [
    dspy.Example(
        document="Q3 revenue increased 15% YoY driven by cloud services growth",
        category="growth",
        risk_level="low",
    ).with_inputs("document"),
    dspy.Example(
        document="Company faces regulatory scrutiny over data privacy practices",
        category="risk",
        risk_level="high",
    ).with_inputs("document"),
    dspy.Example(
        document="Annual report shows stable earnings with modest 2% increase",
        category="stable",
        risk_level="low",
    ).with_inputs("document"),
    dspy.Example(
        document="Supply chain disruptions caused a 10% decline in production",
        category="decline",
        risk_level="medium",
    ).with_inputs("document"),
    # Include 100+ examples for best results
]

# Step 4: Define a metric
def pipeline_metric(example, prediction, trace=None):
    category_match = prediction.category.strip().lower() == example.category.strip().lower()
    risk_match = prediction.risk_level.strip().lower() == example.risk_level.strip().lower()
    return category_match and risk_match

# Step 5: Create the pipeline and assign the small LM to the classify module
pipeline = AnalysisPipeline()
pipeline.classify.set_lm(small_lm)  # Classification runs on the smaller model

# Step 6: Configure BetterTogether
# Constructor: (metric, prompt_optimizer, weight_optimizer, seed)
better_together = dspy.BetterTogether(
    metric=pipeline_metric,
    prompt_optimizer=dspy.MIPROv2,
    weight_optimizer=dspy.BootstrapFinetune,
)

# Step 7: Compile with alternating strategy
# compile(): (student, trainset, strategy, valset_ratio)
# Strategy "p -> w -> p" means: prompt optimize, then weight optimize, then prompt optimize again
dspy.settings.experimental = True
optimized = better_together.compile(
    pipeline,
    trainset=trainset,
    strategy="p -> w -> p",
)

# Step 8: Test the jointly-optimized pipeline
result = optimized(document="Tech startup secures $50M Series B funding round")
print(f"Category: {result.category}")
print(f"Risk Level: {result.risk_level}")

# Step 9: Evaluate the optimized pipeline
devset = [
    dspy.Example(
        document="Market volatility increased amid geopolitical tensions",
        category="risk",
        risk_level="high",
    ).with_inputs("document"),
    dspy.Example(
        document="New product launch exceeded sales expectations by 30%",
        category="growth",
        risk_level="low",
    ).with_inputs("document"),
]

evaluator = Evaluate(devset=devset, metric=pipeline_metric, num_threads=4, display_progress=True)
score = evaluator(optimized)
print(f"\nJointly-optimized pipeline accuracy: {score:.1f}%")
