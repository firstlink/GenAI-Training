"""
Lab 2 - Exercise 4: Chain-of-Thought Prompting
Solution for teaching AI to show reasoning steps

Learning Objectives:
- Understand chain-of-thought (CoT) prompting
- Improve reasoning accuracy with step-by-step thinking
- Apply CoT to math, logic, and analysis tasks
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def without_cot():
    """Solve problems without chain-of-thought"""
    print("=" * 70)
    print("TASK 4A: WITHOUT CHAIN-OF-THOUGHT")
    print("=" * 70)

    problem = "If a store has 15 apples and sells 60% of them, then buys 8 more, how many apples does it have?"

    prompt = f"Solve this problem: {problem}"

    print(f"\n📝 Prompt: {prompt}")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0
    )

    print(f"\n💬 Response: {response.choices[0].message.content}")
    print("\n⚠️ May jump to answer without showing work")


def with_cot():
    """Solve problems with chain-of-thought"""
    print("\n\n" + "=" * 70)
    print("TASK 4B: WITH CHAIN-OF-THOUGHT")
    print("=" * 70)

    problem = "If a store has 15 apples and sells 60% of them, then buys 8 more, how many apples does it have?"

    prompt = f"""Solve this problem step by step:

{problem}

Let's think through this:
1. First, identify what we know
2. Calculate intermediate steps
3. Arrive at the final answer

Show your work for each step."""

    print(f"\n📝 Prompt with CoT instructions:")
    print(prompt)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0
    )

    print(f"\n💬 Response:")
    print(response.choices[0].message.content)
    print("\n✅ Shows reasoning steps, easier to verify!")


def cot_few_shot():
    """Combine CoT with few-shot learning"""
    print("\n\n" + "=" * 70)
    print("TASK 4C: CHAIN-OF-THOUGHT FEW-SHOT")
    print("=" * 70)

    prompt = """Solve these word problems step-by-step.

Example:
Problem: John has 5 apples. He buys 3 more, then gives 2 away. How many does he have?

Solution:
Step 1: John starts with 5 apples
Step 2: He buys 3 more: 5 + 3 = 8 apples
Step 3: He gives 2 away: 8 - 2 = 6 apples
Answer: 6 apples

Problem: A train travels 60 miles in 1 hour. How far does it travel in 2.5 hours at the same speed?

Solution:
Step 1: The train's speed is 60 miles per hour
Step 2: Distance = Speed × Time
Step 3: Distance = 60 × 2.5 = 150 miles
Answer: 150 miles

Now solve:
Problem: Sarah has $50. She spends $12 on lunch and $8 on coffee. Later, she finds $5. How much does she have now?

Solution:"""

    print(f"\n📝 Few-shot CoT prompt (showing examples):")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0
    )

    print(f"\n💬 Response:")
    print(response.choices[0].message.content)


def cot_for_logic():
    """Use CoT for logical reasoning"""
    print("\n\n" + "=" * 70)
    print("TASK 4D: CHAIN-OF-THOUGHT FOR LOGIC")
    print("=" * 70)

    prompt = """Analyze this scenario logically, step by step:

Scenario: All project managers at TechCorp must attend the weekly meeting. Sarah attended the weekly meeting. Is Sarah a project manager?

Let's reason through this:
1. Identify the given information
2. Identify what we're trying to determine
3. Apply logical reasoning
4. State the conclusion with justification"""

    print(f"\n📝 Logic problem with CoT:")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0
    )

    print(f"\n💬 Response:")
    print(response.choices[0].message.content)


def cot_for_analysis():
    """Use CoT for text analysis"""
    print("\n\n" + "=" * 70)
    print("TASK 4E: CHAIN-OF-THOUGHT FOR ANALYSIS")
    print("=" * 70)

    prompt = """Analyze the sentiment of this customer review step-by-step:

Review: "The product arrived quickly, which was great. However, the quality is not what I expected based on the price. It works, but I'm disappointed."

Analysis steps:
1. Identify positive elements
2. Identify negative elements
3. Weigh the overall sentiment
4. Provide final classification with reasoning"""

    print(f"\n📝 Sentiment analysis with CoT:")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0
    )

    print(f"\n💬 Response:")
    print(response.choices[0].message.content)


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run all demonstrations"""
    print("=" * 70)
    print("EXERCISE 4: CHAIN-OF-THOUGHT PROMPTING")
    print("=" * 70)

    without_cot()
    with_cot()
    cot_few_shot()
    cot_for_logic()
    cot_for_analysis()

    print("\n\n" + "=" * 70)
    print("✅ EXERCISE 4 COMPLETE!")
    print("=" * 70)

    print("\n💡 KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. CoT = Ask AI to 'think step by step'")
    print("2. Dramatically improves accuracy on complex tasks")
    print("3. Makes reasoning transparent and verifiable")
    print("4. Works especially well for math and logic")
    print("5. Combine with few-shot for best results")
    print("6. Use phrases like 'Let's think through this'")
    print("7. Temperature=0 for consistent reasoning")

    print("\n📖 COT PROMPTING PATTERNS:")
    print("=" * 70)
    print('✅ "Let\'s solve this step by step"')
    print('✅ "Think through this carefully"')
    print('✅ "Break this down into steps"')
    print('✅ "Show your reasoning"')
    print('✅ "Let\'s analyze this systematically"')

    print("\n🎯 WHEN TO USE COT:")
    print("=" * 70)
    print("✅ Math word problems")
    print("✅ Logical reasoning")
    print("✅ Multi-step calculations")
    print("✅ Complex analysis tasks")
    print("✅ Debugging and troubleshooting")
    print("✅ Decision-making scenarios")


if __name__ == "__main__":
    main()
