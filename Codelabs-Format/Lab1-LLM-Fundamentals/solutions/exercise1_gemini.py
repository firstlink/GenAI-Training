"""
Lab 1 - Exercise 1: First API Call (Gemini)
Solution for making your first Google Gemini API call

Learning Objectives:
- Configure Google Generative AI
- Make a basic content generation request
- Extract response text
- Understand Gemini's simpler API structure
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

print("=" * 70)
print("EXERCISE 1: FIRST API CALL - GEMINI")
print("=" * 70)

# Initialize model
model = genai.GenerativeModel('gemini-pro')

# Make API call
print("\n📤 Sending request to Gemini...")
response = model.generate_content('What is the capital of France?')

# Extract and display answer
print(f"\n💬 Answer: {response.text}")

# Note: Gemini's free tier doesn't provide detailed token usage
# But we can estimate based on response
print(f"\n📊 Response Info:")
print(f"  Model: gemini-pro")
print(f"  Characters: {len(response.text)}")
print(f"  Estimated tokens: ~{len(response.text) // 4}")

print("\n✅ Exercise 1 Complete!")
print("\n💡 Key Takeaways:")
print("  - Gemini has a simpler API structure")
print("  - Gemini Pro is free with rate limits")
print("  - Great for getting started without costs")
print("  - Response access is straightforward: response.text")
print("\n⚠️ Note: Gemini's free tier has rate limits")
print("  - Use for learning and light testing")
print("  - For production, consider OpenAI or Claude")
