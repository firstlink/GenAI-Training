"""
Lab 1 - Exercise 1: First API Call (OpenAI)
Solution for making your first OpenAI API call

Learning Objectives:
- Initialize OpenAI client
- Make a basic chat completion request
- Extract response and token usage
- Understand message structure (system + user roles)
"""

from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

print("=" * 70)
print("EXERCISE 1: FIRST API CALL - OPENAI")
print("=" * 70)

# Make API call
print("\n📤 Sending request to OpenAI...")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # Fast and cost-effective model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=50,  # Limit response length
    temperature=0.7  # Balanced creativity
)

# Extract and display answer
answer = response.choices[0].message.content
print(f"\n💬 Answer: {answer}")

# Show token usage (important for cost tracking)
print(f"\n📊 Token Usage:")
print(f"  Prompt: {response.usage.prompt_tokens} tokens")
print(f"  Completion: {response.usage.completion_tokens} tokens")
print(f"  Total: {response.usage.total_tokens} tokens")

# Calculate approximate cost (GPT-3.5-turbo pricing)
cost_per_1k_input = 0.0015  # $0.0015 per 1K input tokens
cost_per_1k_output = 0.002  # $0.002 per 1K output tokens

input_cost = (response.usage.prompt_tokens / 1000) * cost_per_1k_input
output_cost = (response.usage.completion_tokens / 1000) * cost_per_1k_output
total_cost = input_cost + output_cost

print(f"\n💰 Approximate Cost:")
print(f"  Input: ${input_cost:.6f}")
print(f"  Output: ${output_cost:.6f}")
print(f"  Total: ${total_cost:.6f}")

print("\n✅ Exercise 1 Complete!")
print("\n💡 Key Takeaways:")
print("  - System message sets the AI's behavior")
print("  - User message contains your question/request")
print("  - Always track token usage for cost management")
print("  - GPT-3.5-turbo is fast and economical")
