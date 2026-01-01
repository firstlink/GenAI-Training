"""
Lab 1 - Exercise 3: Temperature Experiments
Solution for understanding temperature parameter effects

Learning Objectives:
- Understand how temperature affects randomness
- Compare outputs at different temperature values
- Learn when to use high vs low temperature
- See deterministic vs creative responses
"""

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def test_temperature(prompt, temperature, num_runs=3):
    """
    Test a prompt at specific temperature multiple times

    Args:
        prompt: The question/prompt to test
        temperature: Temperature value (0.0 to 2.0)
        num_runs: Number of times to run (to see variation)

    Returns:
        List of responses
    """
    responses = []

    for i in range(num_runs):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=temperature
        )
        responses.append(response.choices[0].message.content)

    return responses


# ============================================================================
# MAIN DEMO
# ============================================================================

print("=" * 70)
print("EXERCISE 3: TEMPERATURE EXPERIMENTS")
print("=" * 70)

# Task 3.1: Temperature Comparison
print("\n🌡️ TASK 3.1: TEMPERATURE COMPARISON")
print("=" * 70)

prompt = "Complete this sentence: The best thing about AI is"

temperatures = [0.0, 0.5, 1.0, 1.5]

print(f"\n📝 Prompt: '{prompt}'")
print("\nRunning same prompt at different temperatures...")

for temp in temperatures:
    print(f"\n🌡️ Temperature: {temp}")
    print("-" * 70)

    responses = test_temperature(prompt, temp, num_runs=3)

    for i, response in enumerate(responses, 1):
        print(f"  Run {i}: {response}")

    # Check if responses are identical
    if len(set(responses)) == 1:
        print("  ⚠️ All responses IDENTICAL (deterministic)")
    else:
        print(f"  ✨ {len(set(responses))} different responses (creative)")

# Task 3.2: Use Case Scenarios
print("\n\n📋 TASK 3.2: TEMPERATURE USE CASES")
print("=" * 70)

use_cases = [
    {
        "name": "Factual Question",
        "prompt": "What is 25 * 4?",
        "recommended_temp": 0.0,
        "reason": "We want consistent, accurate answer"
    },
    {
        "name": "Creative Writing",
        "prompt": "Write a creative opening line for a sci-fi story",
        "recommended_temp": 0.9,
        "reason": "We want diverse, creative outputs"
    },
    {
        "name": "Code Generation",
        "prompt": "Write a Python function to reverse a string",
        "recommended_temp": 0.3,
        "reason": "Focused but allowing some variation in style"
    },
    {
        "name": "Brainstorming",
        "prompt": "Give me 3 business ideas for AI startups",
        "recommended_temp": 0.8,
        "reason": "Creative but not too random"
    }
]

for use_case in use_cases:
    print(f"\n📌 {use_case['name']}")
    print(f"   Recommended Temperature: {use_case['recommended_temp']}")
    print(f"   Reason: {use_case['reason']}")
    print(f"   Prompt: '{use_case['prompt']}'")

    # Generate response at recommended temperature
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": use_case['prompt']}],
        max_tokens=100,
        temperature=use_case['recommended_temp']
    )

    print(f"   Response: {response.choices[0].message.content}")
    print()

# Task 3.3: Temperature Guide
print("\n\n📖 TASK 3.3: TEMPERATURE SELECTION GUIDE")
print("=" * 70)

temperature_guide = {
    0.0: {
        "description": "Deterministic (same output every time)",
        "use_for": ["Math", "Factual Q&A", "Classification", "Translation"]
    },
    0.3: {
        "description": "Focused (mostly consistent, slight variation)",
        "use_for": ["Code generation", "Summarization", "Analysis"]
    },
    0.7: {
        "description": "Balanced (default - good mix)",
        "use_for": ["General chat", "Customer support", "Q&A"]
    },
    1.0: {
        "description": "Creative (diverse outputs)",
        "use_for": ["Creative writing", "Brainstorming", "Story generation"]
    },
    1.5: {
        "description": "Very creative (high variation)",
        "use_for": ["Experimental", "Art", "Poetry"]
    },
    2.0: {
        "description": "Maximum randomness (unpredictable)",
        "use_for": ["Rarely used - too random for most tasks"]
    }
}

print("\nTemperature Selection Guide:")
for temp, info in temperature_guide.items():
    print(f"\n🌡️ {temp}:")
    print(f"   {info['description']}")
    print(f"   Best for: {', '.join(info['use_for'])}")

print("\n\n✅ Exercise 3 Complete!")
print("\n💡 Key Takeaways:")
print("  - Temperature 0.0 = Deterministic (same every time)")
print("  - Temperature 1.0+ = Creative (different every time)")
print("  - Use LOW temp (0.0-0.3) for factual/technical tasks")
print("  - Use HIGH temp (0.7-1.0) for creative tasks")
print("  - Default 0.7 works well for most general use")
print("  - Temperature > 1.5 is rarely useful")

# Practical Exercise
print("\n\n🎯 PRACTICAL CHALLENGE:")
print("=" * 70)
print("Try these prompts yourself with different temperatures:")
print("  1. 'What is the capital of Japan?' (try 0.0 vs 1.0)")
print("  2. 'Write a haiku about coding' (try 0.3 vs 1.2)")
print("  3. 'Explain recursion in Python' (try 0.0 vs 0.7)")
print("\nObserve how temperature affects:")
print("  - Consistency of factual answers")
print("  - Creativity in writing")
print("  - Variation in technical explanations")
