"""
Lab 2 - Exercise 3: Few-Shot Learning
Solution for teaching AI through examples

Learning Objectives:
- Understand few-shot learning
- Create effective examples
- Build zero-shot vs few-shot comparisons
- Implement sentiment analysis and intent classification
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def zero_shot_classification():
    """Attempt classification without examples (zero-shot)"""
    print("=" * 70)
    print("TASK 3A: ZERO-SHOT CLASSIFICATION")
    print("=" * 70)

    prompt = "Classify the sentiment of this text as positive, negative, or neutral: 'This product is okay, nothing special.'"

    print(f"\n📝 Prompt: {prompt}")
    print("\n⚠️ Note: No examples provided")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0
    )

    print(f"\n💬 Response: {response.choices[0].message.content}")
    print("\n🤔 Result: Works, but may be inconsistent or verbose")


def few_shot_classification():
    """Use examples to guide classification (few-shot)"""
    print("\n\n" + "=" * 70)
    print("TASK 3B: FEW-SHOT CLASSIFICATION")
    print("=" * 70)

    few_shot_prompt = """Classify the sentiment as positive, negative, or neutral.

Examples:
Text: "I love this product! It's amazing!"
Sentiment: positive

Text: "This is the worst purchase I've ever made."
Sentiment: negative

Text: "The product arrived on time."
Sentiment: neutral

Text: "It's okay, does the job."
Sentiment: neutral

Now classify this:
Text: "This product is okay, nothing special."
Sentiment:"""

    print(f"\n📝 Prompt with examples:")
    print(few_shot_prompt)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": few_shot_prompt}
        ],
        max_tokens=10,
        temperature=0
    )

    print(f"\n💬 Response: {response.choices[0].message.content}")
    print("\n✅ Result: Consistent, concise, follows the pattern!")


def build_sentiment_analyzer():
    """Build a production-ready sentiment analyzer with few-shot"""
    print("\n\n" + "=" * 70)
    print("TASK 3C: PRODUCTION SENTIMENT ANALYZER")
    print("=" * 70)

    def analyze_sentiment(text):
        """Analyze sentiment using few-shot learning"""

        prompt = f"""You are a sentiment analysis system. Classify each text as exactly one word: positive, negative, or neutral.

Examples:
Text: "Absolutely love it! Best purchase ever!"
Sentiment: positive

Text: "Complete waste of money. Very disappointed."
Sentiment: negative

Text: "Product works as described."
Sentiment: neutral

Text: "Not bad, could be better."
Sentiment: neutral

Text: "Terrible quality, broke after one day!"
Sentiment: negative

Text: "Exceeded my expectations! Highly recommend."
Sentiment: positive

Now classify:
Text: "{text}"
Sentiment:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0
        )

        return response.choices[0].message.content.strip().lower()

    # Test cases
    test_texts = [
        "This is fantastic! I'm so happy with it.",
        "Worst experience ever. Never buying again.",
        "It arrived yesterday.",
        "Meh, it's alright I guess.",
        "Amazing product! Worth every penny!",
        "Broke after 2 days. Total garbage."
    ]

    print("\n🧪 Testing sentiment analyzer:\n")
    for text in test_texts:
        sentiment = analyze_sentiment(text)
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(sentiment, "❓")
        print(f"{emoji} '{text[:50]}...' → {sentiment}")


def intent_classification():
    """Use few-shot for intent classification"""
    print("\n\n" + "=" * 70)
    print("TASK 3D: INTENT CLASSIFICATION")
    print("=" * 70)

    def classify_intent(user_message):
        """Classify user intent using few-shot"""

        prompt = f"""Classify the user's intent as one of: question, complaint, purchase, cancel, other

Examples:
User: "How do I reset my password?"
Intent: question

User: "I want to buy the premium plan"
Intent: purchase

User: "This service is terrible! It never works!"
Intent: complaint

User: "Please cancel my subscription"
Intent: cancel

User: "What are your hours?"
Intent: question

User: "I'd like to upgrade my account"
Intent: purchase

User: "The app keeps crashing, this is unacceptable"
Intent: complaint

Now classify:
User: "{user_message}"
Intent:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0
        )

        return response.choices[0].message.content.strip().lower()

    # Test cases
    test_messages = [
        "How much does the pro plan cost?",
        "I want to purchase 5 licenses",
        "Your product is broken! Fix it!",
        "I need to cancel my order",
        "When does support close?",
        "This is ridiculous, nothing works"
    ]

    print("\n🧪 Testing intent classifier:\n")

    intent_emoji = {
        "question": "❓",
        "complaint": "😠",
        "purchase": "💰",
        "cancel": "❌",
        "other": "🤷"
    }

    for msg in test_messages:
        intent = classify_intent(msg)
        emoji = intent_emoji.get(intent, "❓")
        print(f"{emoji} '{msg}' → {intent}")


def category_classification():
    """Multi-class classification with few-shot"""
    print("\n\n" + "=" * 70)
    print("TASK 3E: PRODUCT CATEGORY CLASSIFICATION")
    print("=" * 70)

    def classify_product(description):
        """Classify product into category"""

        prompt = f"""Classify the product into one category: electronics, clothing, food, books, toys, other

Examples:
Product: "iPhone 15 Pro Max smartphone"
Category: electronics

Product: "Men's cotton t-shirt, blue"
Category: clothing

Product: "Organic honey, 16oz jar"
Category: food

Product: "Python Programming: Complete Guide"
Category: books

Product: "LEGO Star Wars set"
Category: toys

Product: "Wireless Bluetooth headphones"
Category: electronics

Product: "Women's running shoes"
Category: clothing

Now classify:
Product: "{description}"
Category:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0
        )

        return response.choices[0].message.content.strip().lower()

    # Test products
    test_products = [
        "MacBook Air laptop",
        "Winter jacket for kids",
        "Dark chocolate bar",
        "Mystery novel by Agatha Christie",
        "Action figure superhero",
        "USB-C charging cable"
    ]

    print("\n🧪 Testing product classifier:\n")

    for product in test_products:
        category = classify_product(product)
        print(f"📦 '{product}' → {category}")


def structured_output_extraction():
    """Use few-shot to extract structured data"""
    print("\n\n" + "=" * 70)
    print("TASK 3F: STRUCTURED DATA EXTRACTION")
    print("=" * 70)

    def extract_contact_info(text):
        """Extract structured contact info from text"""

        prompt = f"""Extract contact information in JSON format.

Examples:
Text: "Hi, I'm John Smith. Email me at john@example.com or call 555-1234"
Output: {{"name": "John Smith", "email": "john@example.com", "phone": "555-1234"}}

Text: "Contact: Sarah Johnson, sarah.j@company.org"
Output: {{"name": "Sarah Johnson", "email": "sarah.j@company.org", "phone": null}}

Text: "Mike Williams can be reached at 555-9876"
Output: {{"name": "Mike Williams", "email": null, "phone": "555-9876"}}

Now extract from:
Text: "{text}"
Output:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0
        )

        return response.choices[0].message.content.strip()

    # Test texts
    test_contacts = [
        "I'm Emma Davis. Reach me at emma.d@email.com or 555-4567",
        "For support, contact Alex Brown at alex@support.com",
        "Call Robert Miller at 555-7890"
    ]

    print("\n🧪 Testing contact extractor:\n")

    for text in test_contacts:
        result = extract_contact_info(text)
        print(f"📧 Input: '{text}'")
        print(f"   Output: {result}\n")


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run all demonstrations"""
    print("=" * 70)
    print("EXERCISE 3: FEW-SHOT LEARNING")
    print("=" * 70)

    # Task 3A: Zero-shot
    zero_shot_classification()

    # Task 3B: Few-shot
    few_shot_classification()

    # Task 3C: Sentiment analyzer
    build_sentiment_analyzer()

    # Task 3D: Intent classification
    intent_classification()

    # Task 3E: Category classification
    category_classification()

    # Task 3F: Structured extraction
    structured_output_extraction()

    # Key takeaways
    print("\n\n" + "=" * 70)
    print("✅ EXERCISE 3 COMPLETE!")
    print("=" * 70)

    print("\n💡 KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. Few-shot learning = Teaching by example")
    print("2. Examples dramatically improve consistency")
    print("3. 3-5 examples usually sufficient")
    print("4. Cover edge cases in examples")
    print("5. Examples should be diverse")
    print("6. Temperature=0 for consistent classification")
    print("7. Few-shot works for many tasks")

    print("\n📖 FEW-SHOT BEST PRACTICES:")
    print("=" * 70)
    print("✅ Provide 3-5 diverse examples")
    print("✅ Cover common cases AND edge cases")
    print("✅ Use consistent format across examples")
    print("✅ Include examples of ALL classes")
    print("✅ Put examples before the actual task")
    print("✅ Use temperature=0 for consistency")
    print("✅ Keep examples clear and unambiguous")

    print("\n🎯 FEW-SHOT TEMPLATE:")
    print("=" * 70)
    print("""
Task description: [What to do]

Examples:
Input: [Example 1 input]
Output: [Example 1 output]

Input: [Example 2 input]
Output: [Example 2 output]

Input: [Example 3 input]
Output: [Example 3 output]

Now complete:
Input: [Actual input]
Output:
""")

    print("\n🔥 WHEN TO USE FEW-SHOT:")
    print("=" * 70)
    print("✅ Classification tasks (sentiment, intent, category)")
    print("✅ Structured output extraction")
    print("✅ Format conversion (text → JSON, etc.)")
    print("✅ Custom task definitions")
    print("✅ When you need consistent outputs")
    print("✅ Domain-specific tasks")


if __name__ == "__main__":
    main()
