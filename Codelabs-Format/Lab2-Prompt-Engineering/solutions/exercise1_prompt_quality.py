"""
Lab 2 - Exercise 1: Understanding Prompt Quality
Solution for comparing vague vs specific prompts

Learning Objectives:
- Experience the impact of prompt specificity
- Learn the 7-part prompt structure
- Understand context and constraints
- See how details improve output quality
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def test_vague_prompt():
    """
    Test how LLMs respond to vague prompts

    Problem: Vague prompts lead to unfocused, generic responses
    """
    print("=" * 70)
    print("TASK 1A: VAGUE PROMPT")
    print("=" * 70)

    vague_prompt = "Tell me about returns"

    print(f"\n📝 Prompt: '{vague_prompt}'")
    print("\n⚠️ Problem: This prompt is ambiguous!")
    print("   Could mean:")
    print("   - Product returns (e-commerce)")
    print("   - Financial returns (investments)")
    print("   - Return statements (programming)")
    print("   - Tax returns")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": vague_prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )

    answer = response.choices[0].message.content
    print(f"\n💬 Response:\n{answer}")
    print(f"\n📊 Tokens used: {response.usage.total_tokens}")
    print("\n❌ Result: Response is unfocused and may not be what we wanted!")


def test_specific_prompt():
    """
    Test how LLMs respond to specific, well-structured prompts

    Uses 7-part structure:
    1. Role/Task
    2. Context
    3. Instructions
    4. Examples
    5. Constraints
    6. Output Format
    7. Tone
    """
    print("\n\n" + "=" * 70)
    print("TASK 1B: SPECIFIC PROMPT")
    print("=" * 70)

    specific_prompt = """
As a customer service agent for TechStore (an electronics retailer),
explain our 30-day product return policy.

Include:
- Eligibility requirements
- Return process steps
- Timeframe for refunds

Keep it under 100 words and use a professional, helpful tone.
"""

    print("\n📝 Prompt Structure Analysis:")
    print("   ✅ Role: Customer service agent for TechStore")
    print("   ✅ Context: Electronics retailer, 30-day policy")
    print("   ✅ Instructions: Explain policy")
    print("   ✅ Requirements: Eligibility, process, timeframe")
    print("   ✅ Constraints: Under 100 words")
    print("   ✅ Tone: Professional, helpful")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": specific_prompt}
        ],
        max_tokens=200,
        temperature=0.7
    )

    answer = response.choices[0].message.content
    print(f"\n💬 Response:\n{answer}")
    print(f"\n📊 Tokens used: {response.usage.total_tokens}")
    print(f"📏 Word count: ~{len(answer.split())} words")
    print("\n✅ Result: Focused, relevant, and actionable response!")


def test_7_part_structure():
    """
    Demonstrate the complete 7-part prompt structure

    7 Parts:
    1. Task/Role
    2. Context
    3. Instructions
    4. Examples
    5. Constraints
    6. Output Format
    7. Tone/Style
    """
    print("\n\n" + "=" * 70)
    print("TASK 1C: COMPLETE 7-PART PROMPT STRUCTURE")
    print("=" * 70)

    seven_part_prompt = """
1. TASK/ROLE:
You are a technical documentation writer for a software company.

2. CONTEXT:
We're documenting a new API endpoint that allows users to upload images.
Our audience is developers with basic REST API knowledge.

3. INSTRUCTIONS:
Write documentation for the POST /api/v1/images endpoint.
Explain what it does, required parameters, and response format.

4. EXAMPLES:
Include a curl example showing how to upload an image.

5. CONSTRAINTS:
- Keep it under 150 words
- Use code blocks for examples
- Don't assume advanced knowledge

6. OUTPUT FORMAT:
Use markdown format with:
- Header for endpoint name
- Parameters section
- Example section
- Response section

7. TONE/STYLE:
Clear, concise, and beginner-friendly.
"""

    print("\n📝 7-Part Prompt:")
    print(seven_part_prompt)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": seven_part_prompt}
        ],
        max_tokens=400,
        temperature=0.7
    )

    answer = response.choices[0].message.content
    print("\n💬 Response:")
    print(answer)
    print("\n✅ Result: Well-structured, complete documentation!")


def compare_prompts_side_by_side():
    """Compare multiple prompts with different levels of specificity"""
    print("\n\n" + "=" * 70)
    print("BONUS: SIDE-BY-SIDE COMPARISON")
    print("=" * 70)

    prompts = {
        "❌ Terrible": "write code",

        "⚠️ Poor": "Write Python code for a function",

        "🤔 Okay": "Write a Python function that calculates the factorial of a number",

        "✅ Good": """Write a Python function called 'factorial' that:
- Takes an integer n as input
- Returns the factorial of n
- Handles edge cases (n=0, negative numbers)
- Includes a docstring""",

        "⭐ Excellent": """Write a production-ready Python function with these specifications:

Function: factorial(n: int) -> int
Purpose: Calculate factorial of a number
Requirements:
- Input validation (reject negative numbers)
- Handle edge case: 0! = 1
- Raise ValueError for invalid input
- Include comprehensive docstring
- Add type hints
- Keep it under 15 lines

Example usage:
>>> factorial(5)
120
>>> factorial(0)
1

Style: Clean, pythonic code with clear error messages.
"""
    }

    print("\nPrompt Quality Spectrum:")
    print("=" * 70)

    for quality, prompt in prompts.items():
        print(f"\n{quality} PROMPT:")
        print(f"'{prompt[:60]}...'")

    print("\n\n💡 Key Differences:")
    print("=" * 70)
    print("❌ Terrible: No context, no constraints")
    print("⚠️ Poor: Vague requirements, missing details")
    print("🤔 Okay: Basic task definition, lacks specifics")
    print("✅ Good: Clear requirements, some structure")
    print("⭐ Excellent: Complete specification, examples, constraints")


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run all demonstrations"""
    print("=" * 70)
    print("EXERCISE 1: UNDERSTANDING PROMPT QUALITY")
    print("=" * 70)

    # Task 1A: Vague prompt
    test_vague_prompt()

    # Task 1B: Specific prompt
    test_specific_prompt()

    # Task 1C: 7-part structure
    test_7_part_structure()

    # Bonus: Comparison
    compare_prompts_side_by_side()

    # Key takeaways
    print("\n\n" + "=" * 70)
    print("✅ EXERCISE 1 COMPLETE!")
    print("=" * 70)

    print("\n💡 KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. Specific prompts → Better outputs")
    print("2. Include context and constraints")
    print("3. Define desired output format")
    print("4. Specify tone and style")
    print("5. Use the 7-part structure for complex tasks")
    print("6. Examples improve quality dramatically")
    print("7. More details = More control")

    print("\n📖 7-PART PROMPT STRUCTURE:")
    print("=" * 70)
    print("1. Task/Role     - What should the AI be/do?")
    print("2. Context       - What's the background?")
    print("3. Instructions  - What are the specific steps?")
    print("4. Examples      - Show what you want")
    print("5. Constraints   - What are the limits?")
    print("6. Output Format - How should it look?")
    print("7. Tone/Style    - What's the voice?")

    print("\n🎯 PRACTICAL TIPS:")
    print("=" * 70)
    print("✅ Always provide context")
    print("✅ Be specific about what you want")
    print("✅ Set clear constraints (length, format, etc.)")
    print("✅ Specify the tone/style")
    print("✅ Include examples when possible")
    print("✅ Define the output format")
    print("✅ Test and refine your prompts")


if __name__ == "__main__":
    main()
