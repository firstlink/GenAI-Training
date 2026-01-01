"""
Lab 2 - Exercise 7: Tone and Style Control
Solution for controlling AI tone and writing style

Learning Objectives:
- Control conversational tone
- Adjust formality levels
- Create consistent brand voice
- Switch between writing styles
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def demonstrate_tone_variations():
    """Show same content in different tones"""
    print("=" * 70)
    print("TASK 7A: TONE VARIATIONS")
    print("=" * 70)

    message = "Your payment is overdue. Please pay immediately."

    tones = {
        "Professional": "Rewrite this in a professional, neutral tone",
        "Friendly": "Rewrite this in a warm, friendly tone",
        "Empathetic": "Rewrite this in an empathetic, understanding tone",
        "Urgent": "Rewrite this in an urgent but polite tone",
        "Casual": "Rewrite this in a casual, conversational tone"
    }

    print(f"\n📝 Original message: '{message}'\n")

    for tone_name, instruction in tones.items():
        prompt = f"{instruction}: '{message}'"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )

        print(f"🎭 {tone_name}:")
        print(f"   {response.choices[0].message.content}\n")


def demonstrate_formality_levels():
    """Control formality from casual to formal"""
    print("\n" + "=" * 70)
    print("TASK 7B: FORMALITY LEVELS")
    print("=" * 70)

    topic = "Explain what cloud computing is"

    formality_levels = [
        ("Very Casual", "Explain like you're talking to a friend at a coffee shop"),
        ("Casual", "Explain in a relaxed, conversational way"),
        ("Neutral", "Explain clearly and professionally"),
        ("Formal", "Explain in a formal, business tone"),
        ("Very Formal", "Explain in academic/technical language")
    ]

    print(f"\n📝 Topic: {topic}\n")

    for level, instruction in formality_levels:
        prompt = f"{instruction}: {topic}"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )

        print(f"📊 {level}:")
        print(f"   {response.choices[0].message.content}\n")


def create_brand_voice():
    """Create consistent brand voice"""
    print("\n" + "=" * 70)
    print("TASK 7C: BRAND VOICE CONSISTENCY")
    print("=" * 70)

    brand_voice = """You are writing for TechFlow, a developer tools company.

Brand voice guidelines:
- Tone: Friendly but professional
- Style: Clear, concise, jargon-free
- Personality: Helpful and enthusiastic about technology
- Language: Active voice, second person ("you")
- Avoid: Corporate speak, buzzwords, excessive formality

Key phrases we use:
- "Let's make it simple"
- "Here's how"
- "You've got this"

Write in this voice."""

    requests = [
        "Announce a new feature",
        "Respond to a customer complaint",
        "Explain a technical concept"
    ]

    print(f"\n🎨 Brand Voice: TechFlow\n")

    for request in requests:
        print(f"📝 Task: {request}")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": brand_voice},
                {"role": "user", "content": request}
            ],
            max_tokens=150,
            temperature=0.7
        )

        print(f"💬 Response:\n{response.choices[0].message.content}\n")
        print("-" * 70 + "\n")


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run all demonstrations"""
    print("=" * 70)
    print("EXERCISE 7: TONE AND STYLE CONTROL")
    print("=" * 70)

    demonstrate_tone_variations()
    demonstrate_formality_levels()
    create_brand_voice()

    print("\n" + "=" * 70)
    print("✅ EXERCISE 7 COMPLETE!")
    print("=" * 70)

    print("\n💡 KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. Tone dramatically affects message perception")
    print("2. Specify desired tone in prompts")
    print("3. Create brand voice guidelines")
    print("4. Use system messages for consistency")
    print("5. Test tone with diverse scenarios")
    print("6. Formality should match audience and context")


if __name__ == "__main__":
    main()
