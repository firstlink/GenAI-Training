"""
Lab 1 - Exercise 1: First API Call (Claude)
Solution for making your first Anthropic Claude API call

Learning Objectives:
- Initialize Anthropic client
- Make a basic message request
- Extract response and token usage
- Understand Claude's message structure
"""

from anthropic import Anthropic
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

print("=" * 70)
print("EXERCISE 1: FIRST API CALL - CLAUDE")
print("=" * 70)

# Make API call
print("\n📤 Sending request to Claude...")
message = client.messages.create(
    model="claude-3-haiku-20240307",  # Fast and economical Claude model
    max_tokens=100,  # Limit response length
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

# Extract and display answer
answer = message.content[0].text
print(f"\n💬 Answer: {answer}")

# Show token usage
print(f"\n📊 Token Usage:")
print(f"  Input: {message.usage.input_tokens} tokens")
print(f"  Output: {message.usage.output_tokens} tokens")
print(f"  Total: {message.usage.input_tokens + message.usage.output_tokens} tokens")

# Calculate approximate cost (Claude Haiku pricing)
cost_per_1m_input = 0.25  # $0.25 per 1M input tokens
cost_per_1m_output = 1.25  # $1.25 per 1M output tokens

input_cost = (message.usage.input_tokens / 1_000_000) * cost_per_1m_input
output_cost = (message.usage.output_tokens / 1_000_000) * cost_per_1m_output
total_cost = input_cost + output_cost

print(f"\n💰 Approximate Cost:")
print(f"  Input: ${input_cost:.6f}")
print(f"  Output: ${output_cost:.6f}")
print(f"  Total: ${total_cost:.6f}")

print("\n✅ Exercise 1 Complete!")
print("\n💡 Key Takeaways:")
print("  - Claude doesn't use system messages (uses user/assistant)")
print("  - Claude Haiku is extremely fast and cost-effective")
print("  - Response is in message.content[0].text")
print("  - Claude is great for analysis and reasoning tasks")
