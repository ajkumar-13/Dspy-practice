"""
Blog 1.1: Setup & Philosophy
Basic DSPy setup and LM configuration examples.
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# --- Configure an OpenAI model ---
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# --- Your first DSPy call ---
predict = dspy.Predict("question -> answer")
result = predict(question="What is DSPy?")
print(result.answer)

# --- Configure Anthropic ---
# lm_anthropic = dspy.LM("anthropic/claude-sonnet-4-5-20250929")
# dspy.configure(lm=lm_anthropic)

# --- Configure a local model via Ollama ---
# lm_local = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434")
# dspy.configure(lm=lm_local)

# --- Inspect what DSPy sent to the LM ---
dspy.inspect_history(n=1)
