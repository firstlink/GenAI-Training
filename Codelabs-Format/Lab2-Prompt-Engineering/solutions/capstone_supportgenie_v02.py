"""
Lab 2 - Capstone: SupportGenie v0.2
Enhanced with Advanced Prompt Engineering

NEW in v0.2:
- 7-part prompt structure
- Few-shot learning for intent classification
- Chain-of-thought reasoning
- Edge case handling
- Tone control by context
- Reusable prompt templates
- Response quality improvements

Upgrades from v0.1:
- Better intent understanding
- Context-aware responses
- Improved error handling
- Professional prompt engineering
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class PromptTemplate:
    """Reusable prompt template"""
    def __init__(self, template):
        self.template = template

    def format(self, **kwargs):
        return self.template.format(**kwargs)


class SupportGenieV02:
    """
    Enhanced AI Customer Support with Advanced Prompt Engineering

    v0.2 Features:
    - Advanced intent classification (few-shot)
    - Context-aware system messages
    - Edge case handling
    - Chain-of-thought for complex issues
    - Professional tone control
    """

    # Intent classification with few-shot learning
    INTENT_CLASSIFIER = PromptTemplate("""Classify user intent as ONE of: question, complaint, purchase, cancel, praise, other

Examples:
"How do I reset my password?" → question
"This is terrible! Nothing works!" → complaint
"I want to buy the premium plan" → purchase
"Please cancel my subscription" → cancel
"Love your product! Amazing!" → praise

User message: "{message}"
Intent:""")

    # Enhanced system messages by mode
    SYSTEM_MESSAGES = {
        "support": """You are SupportGenie v0.2, an advanced customer support AI.

ROLE: Professional customer support specialist
EXPERTISE: Product knowledge, troubleshooting, empathetic communication

RESPONSE STRUCTURE (use for complex issues):
1. Acknowledge the customer's concern empathetically
2. Analyze the situation (chain-of-thought if needed)
3. Provide clear, actionable solution
4. Verify understanding
5. Offer additional help

TONE: Professional yet warm, patient, solution-oriented

EDGE CASES:
- Unclear request → Ask clarifying questions
- Outside scope → Direct to appropriate resource
- Angry customer → Extra empathy, immediate action
- Simple question → Concise, direct answer

CONSTRAINTS:
- Keep responses under 150 words unless troubleshooting
- Use bullet points for multi-step instructions
- Always end with "Is there anything else I can help with?"

NEVER: Make promises you can't keep, be defensive, use jargon without explanation""",

        "sales": """You are SupportGenie v0.2 Sales Assistant.

ROLE: Consultative sales advisor
APPROACH: Need-based selling, not pushy

RESPONSE PATTERN:
1. Understand customer needs (ask questions)
2. Recommend appropriate solution
3. Explain benefits (not just features)
4. Address potential concerns
5. Clear next steps

TONE: Helpful and enthusiastic, not aggressive

EDGE CASES:
- Price concerns → Explain value, offer options
- Comparison questions → Honest, feature-focused
- Not ready to buy → Provide information, stay helpful

ALWAYS: Listen first, recommend second""",

        "technical": """You are SupportGenie v0.2 Technical Support.

ROLE: Technical troubleshooting specialist
EXPERTISE: System diagnostics, step-by-step problem solving

TROUBLESHOOTING APPROACH (chain-of-thought):
1. Gather information about the issue
2. Identify likely causes
3. Provide diagnostic steps
4. Offer solution based on findings
5. Verify resolution

COMMUNICATION:
- Use technical terms when appropriate
- Explain complex concepts clearly
- Provide code/examples when helpful

TONE: Professional, precise, patient

EDGE CASES:
- Vague description → Ask specific diagnostic questions
- Complex issue → Break into smaller steps
- Quick fix available → Provide it immediately"""
    }

    def __init__(self, mode="support", company_name="TechCorp"):
        self.mode = mode
        self.company_name = company_name
        self.conversation_history = []
        self.session_start = datetime.now()
        self.message_count = 0

        # Initialize with enhanced system message
        system_msg = self.SYSTEM_MESSAGES[mode].replace("TechCorp", company_name)
        self.conversation_history.append({"role": "system", "content": system_msg})

    def classify_intent(self, message):
        """Classify user intent using few-shot learning"""
        prompt = self.INTENT_CLASSIFIER.format(message=message)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )

        return response.choices[0].message.content.strip().lower()

    def handle_edge_cases(self, message):
        """Check for edge cases before processing"""
        if not message or message.strip() == "":
            return "I'm here to help! What can I assist you with today?"

        if len(message) < 3:
            return "Could you provide a bit more detail so I can help you better?"

        # Check for inappropriate content (simplified)
        inappropriate_keywords = ["hack", "exploit", "illegal"]
        if any(word in message.lower() for word in inappropriate_keywords):
            return "I'm here to provide legitimate support. How can I help you with our products or services?"

        return None  # No edge case detected

    def get_response(self, user_message):
        """Get enhanced response with advanced prompting"""

        # Handle edge cases
        edge_case_response = self.handle_edge_cases(user_message)
        if edge_case_response:
            return edge_case_response

        # Classify intent
        intent = self.classify_intent(user_message)

        # Add intent context to conversation
        self.conversation_history.append({
            "role": "user",
            "content": f"[Intent: {intent}] {user_message}"
        })

        # Get response
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.conversation_history,
                max_tokens=500,
                temperature=0.7
            )

            assistant_message = response.choices[0].message.content

            # Add to history (without intent label)
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            self.message_count += 1

            return {
                "response": assistant_message,
                "intent": intent,
                "tokens": response.usage.total_tokens
            }

        except Exception as e:
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                "intent": "error",
                "tokens": 0
            }

    def run_demo(self):
        """Run demonstration of v0.2 capabilities"""
        print("=" * 70)
        print(f"SUPPORTGENIE v0.2 - {self.mode.upper()} MODE")
        print("=" * 70)
        print(f"Company: {self.company_name}")
        print("Enhanced with: Advanced Prompt Engineering")
        print("=" * 70)

        demo_scenarios = {
            "support": [
                "I can't log into my account",
                "This is frustrating! It's been 3 days!",
                "help",  # Edge case: too short
                ""  # Edge case: empty
            ],
            "sales": [
                "What plans do you offer?",
                "How does your pricing compare to competitors?",
                "I'm not sure if I need this yet"
            ],
            "technical": [
                "The app crashes when I click export",
                "What error message do you see?",
                "Let me check that"
            ]
        }

        scenarios = demo_scenarios.get(self.mode, demo_scenarios["support"])

        for i, msg in enumerate(scenarios, 1):
            print(f"\n{'='*70}")
            print(f"Scenario {i}")
            print(f"{'='*70}")
            print(f"\n👤 Customer: '{msg if msg else '(empty)'}' ")

            result = self.get_response(msg)

            if isinstance(result, dict):
                print(f"\n🏷️  Intent Detected: {result['intent']}")
                print(f"\n🤖 {self.company_name} Agent:")
                print(f"{result['response']}")
                print(f"\n📊 Tokens: {result['tokens']}")
            else:
                print(f"\n🤖 {self.company_name} Agent:")
                print(f"{result}")


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Demonstrate SupportGenie v0.2"""
    print("\n" * 2)
    print("=" * 70)
    print("SUPPORTGENIE v0.2 - ADVANCED PROMPT ENGINEERING")
    print("=" * 70)

    print("\n🆕 NEW FEATURES IN v0.2:")
    print("  ✅ Few-shot intent classification")
    print("  ✅ 7-part structured prompts")
    print("  ✅ Chain-of-thought reasoning")
    print("  ✅ Comprehensive edge case handling")
    print("  ✅ Context-aware tone control")
    print("  ✅ Enhanced system messages")

    print("\n📋 Select demo mode:")
    print("  1. Customer Support")
    print("  2. Sales Assistant")
    print("  3. Technical Support")
    print("  4. Run all demos")

    choice = input("\nChoice (1-4, or Enter for all): ").strip()

    if choice == "1":
        genie = SupportGenieV02(mode="support", company_name="TechCorp")
        genie.run_demo()
    elif choice == "2":
        genie = SupportGenieV02(mode="sales", company_name="TechCorp")
        genie.run_demo()
    elif choice == "3":
        genie = SupportGenieV02(mode="technical", company_name="TechCorp")
        genie.run_demo()
    else:
        # Run all demos
        for mode in ["support", "sales", "technical"]:
            print("\n" * 2)
            genie = SupportGenieV02(mode=mode, company_name="TechCorp")
            genie.run_demo()

    print("\n\n" + "=" * 70)
    print("✅ SUPPORTGENIE v0.2 DEMONSTRATION COMPLETE")
    print("=" * 70)

    print("\n🎯 IMPROVEMENTS FROM v0.1:")
    print("  • Intent classification for better routing")
    print("  • Structured response patterns")
    print("  • Edge case handling (empty, short, inappropriate)")
    print("  • Chain-of-thought for complex issues")
    print("  • Professional prompt templates")
    print("  • Consistent brand voice")

    print("\n📈 PROMPT ENGINEERING TECHNIQUES USED:")
    print("  • 7-part prompt structure in system messages")
    print("  • Few-shot learning for classification")
    print("  • Clear role and expertise definition")
    print("  • Explicit constraints and guidelines")
    print("  • Edge case instructions")
    print("  • Tone and style specifications")

    print("\n🔮 COMING IN v3.0 (Lab 5):")
    print("  • RAG with knowledge base")
    print("  • Document retrieval")
    print("  • Fact-based responses")
    print("  • Source citations")


if __name__ == "__main__":
    main()
