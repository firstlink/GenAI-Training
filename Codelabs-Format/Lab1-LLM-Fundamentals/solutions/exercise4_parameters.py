"""
Lab 1 - Exercise 4: Parameter Comparison
Solution for comparing different LLM parameters

Learning Objectives:
- Understand max_tokens effect
- Compare top_p vs temperature
- Test presence_penalty and frequency_penalty
- Learn optimal parameter combinations
"""

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def compare_max_tokens(prompt, token_limits):
    """Compare same prompt with different max_tokens"""
    print(f"\n📝 Prompt: '{prompt}'")
    print("=" * 70)

    for max_tok in token_limits:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tok,
            temperature=0.7
        )

        print(f"\n🔢 max_tokens={max_tok}:")
        print(f"   Response: {response.choices[0].message.content}")
        print(f"   Actual tokens used: {response.usage.completion_tokens}")


def compare_top_p_vs_temperature(prompt):
    """Compare top_p and temperature effects"""
    print(f"\n📝 Prompt: '{prompt}'")
    print("=" * 70)

    configs = [
        {"temp": 0.0, "top_p": 1.0, "desc": "Deterministic"},
        {"temp": 1.0, "top_p": 1.0, "desc": "Creative (high temp)"},
        {"temp": 0.7, "top_p": 0.5, "desc": "Focused (low top_p)"},
        {"temp": 0.7, "top_p": 0.9, "desc": "Balanced"},
    ]

    for config in configs:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=config["temp"],
            top_p=config["top_p"]
        )

        print(f"\n⚙️ {config['desc']}:")
        print(f"   (temperature={config['temp']}, top_p={config['top_p']})")
        print(f"   Response: {response.choices[0].message.content[:150]}...")


def test_penalties(prompt):
    """Test presence and frequency penalties"""
    print(f"\n📝 Prompt: '{prompt}'")
    print("=" * 70)

    configs = [
        {"presence": 0.0, "frequency": 0.0, "desc": "No penalties"},
        {"presence": 1.0, "frequency": 0.0, "desc": "High presence penalty"},
        {"presence": 0.0, "frequency": 1.0, "desc": "High frequency penalty"},
        {"presence": 0.5, "frequency": 0.5, "desc": "Balanced penalties"},
    ]

    for config in configs:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.8,
            presence_penalty=config["presence"],
            frequency_penalty=config["frequency"]
        )

        print(f"\n⚙️ {config['desc']}:")
        print(f"   (presence={config['presence']}, frequency={config['frequency']})")
        print(f"   Response: {response.choices[0].message.content}")


# ============================================================================
# MAIN DEMO
# ============================================================================

print("=" * 70)
print("EXERCISE 4: PARAMETER COMPARISON")
print("=" * 70)

# Task 4.1: max_tokens Comparison
print("\n\n🔢 TASK 4.1: MAX_TOKENS EFFECT")
print("=" * 70)

prompt1 = "Explain what Python is in simple terms"
compare_max_tokens(prompt1, token_limits=[10, 50, 100, 200])

# Task 4.2: top_p vs temperature
print("\n\n🎛️ TASK 4.2: TOP_P VS TEMPERATURE")
print("=" * 70)

prompt2 = "Write a creative tagline for an AI company"
compare_top_p_vs_temperature(prompt2)

# Task 4.3: Penalties
print("\n\n⚖️ TASK 4.3: PRESENCE & FREQUENCY PENALTIES")
print("=" * 70)

prompt3 = "Write a short paragraph about AI using the word 'technology' multiple times"
test_penalties(prompt3)

# Parameter Reference Guide
print("\n\n📖 PARAMETER REFERENCE GUIDE")
print("=" * 70)

guide = {
    "temperature": {
        "range": "0.0 - 2.0",
        "default": "0.7",
        "purpose": "Controls randomness",
        "tip": "Lower = focused, Higher = creative"
    },
    "top_p": {
        "range": "0.0 - 1.0",
        "default": "1.0",
        "purpose": "Nucleus sampling (alternative to temperature)",
        "tip": "Use top_p OR temperature, not both"
    },
    "max_tokens": {
        "range": "1 - 4096 (model dependent)",
        "default": "Varies",
        "purpose": "Limits response length",
        "tip": "Set based on expected response length"
    },
    "presence_penalty": {
        "range": "-2.0 - 2.0",
        "default": "0.0",
        "purpose": "Penalizes topics already mentioned",
        "tip": "Use for diverse topic coverage"
    },
    "frequency_penalty": {
        "range": "-2.0 - 2.0",
        "default": "0.0",
        "purpose": "Penalizes repeated tokens",
        "tip": "Use to reduce repetition"
    }
}

for param, info in guide.items():
    print(f"\n📌 {param.upper()}:")
    for key, value in info.items():
        print(f"   {key.capitalize():12} {value}")

print("\n\n✅ Exercise 4 Complete!")
print("\n💡 Key Takeaways:")
print("  - max_tokens: Always set to avoid runaway costs")
print("  - temperature: Most important for controlling output style")
print("  - top_p: Alternative to temperature (don't use both)")
print("  - presence_penalty: Encourages new topics")
print("  - frequency_penalty: Reduces word repetition")

print("\n\n🎯 RECOMMENDED SETTINGS:")
print("=" * 70)

recommendations = {
    "Factual Q&A": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 200,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    },
    "Creative Writing": {
        "temperature": 0.9,
        "top_p": 1.0,
        "max_tokens": 500,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.5
    },
    "Code Generation": {
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 1000,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.3
    },
    "Chatbot": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 300,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.2
    }
}

for use_case, params in recommendations.items():
    print(f"\n📋 {use_case}:")
    for param, value in params.items():
        print(f"   {param:20} {value}")
