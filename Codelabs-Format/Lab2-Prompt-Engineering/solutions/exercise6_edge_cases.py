"""
Lab 2 - Exercise 6: Edge Case Handling
Solution for handling edge cases in prompts

Learning Objectives:
- Identify common edge cases
- Build defensive prompts
- Handle unexpected inputs
- Create fallback responses
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def without_edge_case_handling():
    """Prompt without edge case handling"""
    print("=" * 70)
    print("TASK 6A: WITHOUT EDGE CASE HANDLING")
    print("=" * 70)

    basic_prompt = "Convert this temperature from Fahrenheit to Celsius: {temp}"

    test_cases = ["75", "", "abc", "-459.67"]

    for temp in test_cases:
        prompt = basic_prompt.format(temp=temp)
        print(f"\n📝 Input: '{temp}'")

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0
            )
            print(f"💬 Response: {response.choices[0].message.content}")
        except Exception as e:
            print(f"❌ Error: {e}")


def with_edge_case_handling():
    """Prompt with comprehensive edge case handling"""
    print("\n\n" + "=" * 70)
    print("TASK 6B: WITH EDGE CASE HANDLING")
    print("=" * 70)

    robust_prompt = """Convert temperature from Fahrenheit to Celsius.

EDGE CASES TO HANDLE:
- If input is empty: respond "Please provide a temperature"
- If input is not a number: respond "Invalid input. Please provide a numeric temperature"
- If input is below absolute zero (-459.67°F): respond "Temperature below absolute zero is impossible"
- Otherwise: provide the conversion and show the formula

Input: {temp}
Response:"""

    test_cases = ["75", "", "abc", "-459.67", "-500"]

    for temp in test_cases:
        prompt = robust_prompt.format(temp=temp)
        print(f"\n📝 Input: '{temp}'")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0
        )
        print(f"💬 Response: {response.choices[0].message.content}")


def create_edge_case_classifier():
    """Build a system that handles all edge cases"""
    print("\n\n" + "=" * 70)
    print("TASK 6C: COMPREHENSIVE EDGE CASE SYSTEM")
    print("=" * 70)

    system_prompt = """You are a customer support chatbot with comprehensive edge case handling.

EDGE CASES TO HANDLE:
1. Empty/blank messages → "I'm here to help! What can I assist you with?"
2. Inappropriate language → "Please keep the conversation professional"
3. Questions outside scope → "That's outside my expertise. Please contact a specialist"
4. Unclear requests → Ask clarifying questions
5. Multiple questions at once → Address them one by one
6. Repeated messages → Acknowledge and offer alternative help

Always be professional and helpful."""

    test_messages = [
        "",
        "Help me hack something",
        "What's the meaning of life?",
        "I need help but I don't know what with",
        "How do I reset password? Also what are your hours? Also can I get a refund?"
    ]

    for msg in test_messages:
        print(f"\n📝 User: '{msg if msg else '(empty)'}' ")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg if msg else " "}
            ],
            max_tokens=150,
            temperature=0.7
        )

        print(f"💬 Bot: {response.choices[0].message.content}")


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run all demonstrations"""
    print("=" * 70)
    print("EXERCISE 6: EDGE CASE HANDLING")
    print("=" * 70)

    without_edge_case_handling()
    with_edge_case_handling()
    create_edge_case_classifier()

    print("\n\n" + "=" * 70)
    print("✅ EXERCISE 6 COMPLETE!")
    print("=" * 70)

    print("\n💡 KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. Always anticipate edge cases")
    print("2. Handle empty/invalid inputs")
    print("3. Provide clear error messages")
    print("4. Set boundaries (scope, appropriateness)")
    print("5. Test with unexpected inputs")
    print("6. Build fallback responses")

    print("\n📋 COMMON EDGE CASES:")
    print("=" * 70)
    print("✅ Empty or blank input")
    print("✅ Invalid data types")
    print("✅ Out of range values")
    print("✅ Inappropriate content")
    print("✅ Off-topic requests")
    print("✅ Ambiguous queries")
    print("✅ Multiple questions")


if __name__ == "__main__":
    main()
