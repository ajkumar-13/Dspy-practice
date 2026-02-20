"""
Phase 1 Mini-Project: No-Prompt Text Classifier
Support ticket classifier using only DSPy signatures and modules.
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


class TicketClassification(dspy.Signature):
    """Classify a customer support ticket by priority and extract key information."""

    ticket: str = dspy.InputField(desc="The customer support ticket text")
    priority: str = dspy.OutputField(desc="One of: Urgent, Standard, Low")
    category: str = dspy.OutputField(desc="Category like: Billing, Technical, Account, Shipping")
    sentiment: str = dspy.OutputField(
        desc="Customer sentiment: Angry, Frustrated, Neutral, Satisfied"
    )
    summary: str = dspy.OutputField(desc="One-sentence summary of the issue")


class TicketClassifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(TicketClassification)

    def forward(self, ticket: str):
        return self.classify(ticket=ticket)


# --- Test with sample tickets ---
if __name__ == "__main__":
    classifier = TicketClassifier()

    tickets = [
        "My account has been hacked! Someone changed my password and is making purchases. I need this resolved IMMEDIATELY.",
        "Hey, just wondering when my order #12345 will arrive. No rush, just curious.",
        "Your billing system charged me twice this month. I want a refund and an explanation. This is the third time this has happened.",
    ]

    for ticket in tickets:
        print("=" * 60)
        result = classifier(ticket=ticket)
        print(f"Ticket: {ticket[:80]}...")
        print(f"Priority: {result.priority}")
        print(f"Category: {result.category}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Summary: {result.summary}")
        print()
