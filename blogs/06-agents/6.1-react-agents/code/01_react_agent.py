"""
Blog 6.1 - ReAct Agent with Weather and Calculator Tools
Run: python 01_react_agent.py
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# --- Define tools as plain Python functions ---


def get_weather(city: str) -> str:
    """Get the current weather for a city. Returns temperature and conditions."""
    weather_data = {
        "paris": "18°C, partly cloudy",
        "london": "14°C, rainy",
        "tokyo": "25°C, sunny",
        "new york": "22°C, clear skies",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Supports basic arithmetic (+, -, *, /, **)."""
    try:
        allowed_chars = set("0123456789.+-*/() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic arithmetic operations are supported."
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


# --- Build the ReAct agent ---
agent = dspy.ReAct(
    "question -> answer",
    tools=[get_weather, calculate],
    max_iters=5,
)

# --- Use the agent ---
result = agent(question="What's the weather in Paris and Tokyo? What's the temperature difference?")
print(f"Answer: {result.answer}")

# --- Inspect the agent trajectory ---
print("\n--- Trajectory ---")
for step_name, step_value in result.trajectory.items():
    print(f"{step_name}: {step_value}")

# --- Optimize with MIPROv2 ---
from dspy.evaluate import SemanticF1

metric = SemanticF1()

trainset = [
    dspy.Example(
        question="What is the capital of the country where the Eiffel Tower is located?",
        answer="Paris",
    ).with_inputs("question"),
    dspy.Example(
        question="What is 20% of the temperature in Tokyo if it's 25°C?",
        answer="5°C",
    ).with_inputs("question"),
]

optimizer = dspy.MIPROv2(
    metric=metric,
    auto="medium",
    num_threads=4,
)

optimized_agent = optimizer.compile(
    agent,
    trainset=trainset,
    requires_permission_to_run=False,
)

# Save the optimized agent
optimized_agent.save("optimized_agent.json")
print("\nOptimized agent saved to optimized_agent.json")
