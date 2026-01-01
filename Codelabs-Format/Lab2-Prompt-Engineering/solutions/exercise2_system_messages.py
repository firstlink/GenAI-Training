"""
Lab 2 - Exercise 2: System Messages
Solution for using system messages to control AI behavior

Learning Objectives:
- Understand the power of system messages
- Set AI personality and behavior
- Create specialized assistants
- Control output format and style
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def test_without_system_message():
    """Test response without a system message"""
    print("=" * 70)
    print("TASK 2A: WITHOUT SYSTEM MESSAGE")
    print("=" * 70)

    user_question = "How do I handle a customer complaint?"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": user_question}
        ],
        max_tokens=200
    )

    print(f"\n📝 Question: {user_question}")
    print(f"\n💬 Response (no system message):")
    print(response.choices[0].message.content)
    print("\n⚠️ Result: Generic, not role-specific")


def test_with_system_message():
    """Test response with a clear system message"""
    print("\n\n" + "=" * 70)
    print("TASK 2B: WITH SYSTEM MESSAGE")
    print("=" * 70)

    system_message = """You are a customer service training instructor with 10 years of experience.
You teach empathetic, professional customer service techniques.
Always provide actionable steps and real examples."""

    user_question = "How do I handle a customer complaint?"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_question}
        ],
        max_tokens=250
    )

    print(f"\n🎭 System Message:")
    print(system_message)
    print(f"\n📝 Question: {user_question}")
    print(f"\n💬 Response (with system message):")
    print(response.choices[0].message.content)
    print("\n✅ Result: Role-specific, actionable, professional!")


def create_specialized_assistants():
    """Create different AI personalities with system messages"""
    print("\n\n" + "=" * 70)
    print("TASK 2C: SPECIALIZED ASSISTANTS")
    print("=" * 70)

    question = "Explain what a REST API is"

    assistants = {
        "👨‍🏫 Teacher": """You are a patient teacher explaining concepts to beginners.
Use simple language, analogies, and real-world examples.
Break complex ideas into digestible chunks.""",

        "🤓 Expert": """You are a technical expert speaking to other professionals.
Use precise terminology, assume technical knowledge, and be concise.
Focus on architecture and best practices.""",

        "🎨 Storyteller": """You are a creative storyteller who explains technical concepts through narratives.
Use metaphors, characters, and engaging stories.
Make learning fun and memorable.""",

        "📝 Documentation Writer": """You are a technical documentation writer.
Provide clear, structured information with bullet points.
Include code examples and use cases."""
    }

    print(f"\n📝 Question for all assistants: '{question}'\n")

    for name, system_msg in assistants.items():
        print("=" * 70)
        print(f"{name}")
        print("=" * 70)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": question}
            ],
            max_tokens=200,
            temperature=0.7
        )

        print(f"\n💬 Response:")
        print(response.choices[0].message.content)
        print()


def test_output_format_control():
    """Use system messages to control output format"""
    print("\n\n" + "=" * 70)
    print("TASK 2D: OUTPUT FORMAT CONTROL")
    print("=" * 70)

    user_request = "Give me 3 tips for writing clean code"

    formats = {
        "JSON": """You are a helpful assistant that ALWAYS responds in valid JSON format.
Your response must be a JSON object with appropriate keys.""",

        "Markdown": """You are a helpful assistant that ALWAYS responds in Markdown format.
Use proper headers, bullet points, and formatting.""",

        "Numbered List": """You are a helpful assistant that ALWAYS responds as a numbered list.
Each item should be clear and actionable.""",

        "Table": """You are a helpful assistant that ALWAYS responds as a markdown table.
Use columns to organize information clearly."""
    }

    print(f"\n📝 Request: '{user_request}'\n")

    for format_name, system_msg in formats.items():
        print("=" * 70)
        print(f"📊 Format: {format_name}")
        print("=" * 70)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_request}
            ],
            max_tokens=250,
            temperature=0.7
        )

        print(f"\n💬 Response:")
        print(response.choices[0].message.content)
        print()


def test_behavior_constraints():
    """Use system messages to set behavior constraints"""
    print("\n\n" + "=" * 70)
    print("TASK 2E: BEHAVIOR CONSTRAINTS")
    print("=" * 70)

    system_message = """You are a helpful assistant with these constraints:

ALWAYS:
- Be concise (max 50 words per response)
- Provide actionable advice
- End with a follow-up question

NEVER:
- Use jargon without explaining it
- Give vague or generic advice
- Provide unsolicited information"""

    questions = [
        "How do I improve my Python skills?",
        "What's the best way to learn machine learning?"
    ]

    print(f"\n🎭 System Constraints:")
    print(system_message)

    for question in questions:
        print("\n" + "=" * 70)
        print(f"📝 Question: {question}")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ],
            max_tokens=150,
            temperature=0.7
        )

        answer = response.choices[0].message.content
        print(f"\n💬 Response:")
        print(answer)
        print(f"\n📊 Word count: {len(answer.split())} words")


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run all demonstrations"""
    print("=" * 70)
    print("EXERCISE 2: SYSTEM MESSAGES")
    print("=" * 70)

    # Task 2A: Without system message
    test_without_system_message()

    # Task 2B: With system message
    test_with_system_message()

    # Task 2C: Specialized assistants
    create_specialized_assistants()

    # Task 2D: Output format control
    test_output_format_control()

    # Task 2E: Behavior constraints
    test_behavior_constraints()

    # Key takeaways
    print("\n\n" + "=" * 70)
    print("✅ EXERCISE 2 COMPLETE!")
    print("=" * 70)

    print("\n💡 KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. System messages set the AI's role and behavior")
    print("2. They persist across the entire conversation")
    print("3. Use them to create specialized assistants")
    print("4. Control output format and style")
    print("5. Set constraints and guidelines")
    print("6. More specific system messages = more control")
    print("7. System messages are invisible to users")

    print("\n📖 SYSTEM MESSAGE BEST PRACTICES:")
    print("=" * 70)
    print("✅ Define a clear role/persona")
    print("✅ Specify expertise level")
    print("✅ Set tone and style")
    print("✅ Define output format")
    print("✅ Set behavioral constraints")
    print("✅ Include do's and don'ts")
    print("✅ Be specific about what to avoid")

    print("\n🎯 SYSTEM MESSAGE TEMPLATE:")
    print("=" * 70)
    print("""
You are a [ROLE] with [EXPERTISE].

Your responsibilities:
- [Primary responsibility 1]
- [Primary responsibility 2]
- [Primary responsibility 3]

Guidelines:
- [Guideline 1]
- [Guideline 2]
- [Guideline 3]

ALWAYS:
- [Required behavior 1]
- [Required behavior 2]

NEVER:
- [Prohibited behavior 1]
- [Prohibited behavior 2]

Output format: [Specify format]
Tone: [Specify tone]
""")

    print("\n🔥 POWER TIPS:")
    print("=" * 70)
    print("💡 Test different system messages to find what works best")
    print("💡 Combine role + expertise + constraints for maximum control")
    print("💡 Use examples in system messages for complex formats")
    print("💡 Keep system messages focused - don't overload")
    print("💡 Update system message when switching conversation topics")


if __name__ == "__main__":
    main()
