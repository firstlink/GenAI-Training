"""
Lab 1 - Exercise 7: Build SimpleChatbot
Solution for building a basic conversational chatbot

Learning Objectives:
- Implement conversation history tracking
- Manage message context
- Build a simple chat loop
- Handle user input and responses
"""

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class SimpleChatbot:
    """
    A basic chatbot with conversation history

    Features:
    - Maintains conversation context
    - Configurable system message
    - Token and cost tracking
    - Simple chat loop
    """

    def __init__(self, system_message="You are a helpful assistant.", model="gpt-3.5-turbo"):
        """
        Initialize the chatbot

        Args:
            system_message: Sets the chatbot's behavior
            model: OpenAI model to use
        """
        self.model = model
        self.conversation_history = [
            {"role": "system", "content": system_message}
        ]
        self.total_tokens = 0
        self.total_cost = 0.0

    def add_message(self, role, content):
        """Add a message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })

    def get_response(self, user_message):
        """
        Get chatbot response to user message

        Args:
            user_message: The user's input

        Returns:
            The chatbot's response
        """
        # Add user message to history
        self.add_message("user", user_message)

        # Get response from API
        response = client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            max_tokens=500,
            temperature=0.7
        )

        # Extract assistant's reply
        assistant_message = response.choices[0].message.content

        # Add assistant's reply to history
        self.add_message("assistant", assistant_message)

        # Track tokens and cost
        self.total_tokens += response.usage.total_tokens
        cost = (response.usage.prompt_tokens / 1000 * 0.0015) + \
               (response.usage.completion_tokens / 1000 * 0.002)
        self.total_cost += cost

        return assistant_message

    def get_history_length(self):
        """Get number of messages in history"""
        # Subtract 1 for system message
        return len(self.conversation_history) - 1

    def clear_history(self):
        """Clear conversation history (keep system message)"""
        system_msg = self.conversation_history[0]
        self.conversation_history = [system_msg]
        self.total_tokens = 0
        self.total_cost = 0.0

    def get_stats(self):
        """Get conversation statistics"""
        return {
            "messages": self.get_history_length(),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost
        }

    def chat_loop(self):
        """
        Run an interactive chat loop

        Commands:
            'quit' or 'exit' - End the conversation
            'clear' - Clear conversation history
            'stats' - Show usage statistics
        """
        print("=" * 70)
        print("SIMPLE CHATBOT")
        print("=" * 70)
        print("\nCommands:")
        print("  'quit' or 'exit' - End conversation")
        print("  'clear' - Clear history")
        print("  'stats' - Show statistics")
        print("\nStart chatting!\n")

        while True:
            # Get user input
            user_input = input("You: ").strip()

            # Check for commands
            if user_input.lower() in ['quit', 'exit']:
                print("\n👋 Goodbye! Chat ended.")
                break

            elif user_input.lower() == 'clear':
                self.clear_history()
                print("✅ Conversation history cleared.\n")
                continue

            elif user_input.lower() == 'stats':
                stats = self.get_stats()
                print(f"\n📊 Statistics:")
                print(f"   Messages: {stats['messages']}")
                print(f"   Total tokens: {stats['total_tokens']}")
                print(f"   Total cost: ${stats['total_cost']:.6f}\n")
                continue

            elif not user_input:
                continue

            # Get and display response
            try:
                response = self.get_response(user_input)
                print(f"\nBot: {response}\n")

            except Exception as e:
                print(f"\n❌ Error: {e}\n")


# ============================================================================
# MAIN DEMO
# ============================================================================

print("=" * 70)
print("EXERCISE 7: BUILD SIMPLECHATBOT")
print("=" * 70)

# Task 7.1: Basic Chatbot
print("\n\n🤖 TASK 7.1: BASIC CHATBOT DEMO")
print("=" * 70)

bot = SimpleChatbot(
    system_message="You are a friendly and helpful assistant.",
    model="gpt-3.5-turbo"
)

print("\nHaving a short conversation with the bot...\n")

conversation = [
    "Hello! What's your name?",
    "What can you help me with?",
    "Tell me a joke about programming"
]

for user_msg in conversation:
    print(f"You: {user_msg}")
    response = bot.get_response(user_msg)
    print(f"Bot: {response}\n")

print(f"📊 Conversation stats: {bot.get_stats()}\n")

# Task 7.2: Different Personalities
print("\n\n🎭 TASK 7.2: CHATBOT PERSONALITIES")
print("=" * 70)

personalities = [
    {
        "name": "Pirate",
        "system": "You are a friendly pirate. Respond like a pirate, using pirate language.",
        "question": "What's the weather like?"
    },
    {
        "name": "Shakespeare",
        "system": "You are Shakespeare. Respond in Shakespearean English.",
        "question": "How are you today?"
    },
    {
        "name": "Tech Expert",
        "system": "You are a technical expert. Give detailed, technical responses.",
        "question": "What is an API?"
    }
]

for personality in personalities:
    print(f"\n🎭 {personality['name']} Chatbot:")
    print("-" * 70)

    bot = SimpleChatbot(system_message=personality['system'])
    print(f"You: {personality['question']}")
    response = bot.get_response(personality['question'])
    print(f"Bot: {response}\n")

# Task 7.3: Specialized Chatbot
print("\n\n💼 TASK 7.3: SPECIALIZED CHATBOT (Customer Support)")
print("=" * 70)

support_bot = SimpleChatbot(
    system_message="""You are a customer support agent for TechCorp.
You are helpful, professional, and empathetic.
Always be polite and try to solve the customer's problem.
If you can't help, suggest they contact a human agent."""
)

support_scenarios = [
    "I can't log into my account",
    "How do I reset my password?",
    "Thank you for your help!"
]

print("\n📞 Customer Support Conversation:\n")

for msg in support_scenarios:
    print(f"Customer: {msg}")
    response = support_bot.get_response(msg)
    print(f"Support: {response}\n")

print(f"📊 Support session stats: {support_bot.get_stats()}")

# Task 7.4: Context Awareness
print("\n\n🧠 TASK 7.4: CONTEXT AWARENESS DEMO")
print("=" * 70)

context_bot = SimpleChatbot(
    system_message="You are a helpful assistant with good memory."
)

print("\nDemonstrating context retention...\n")

context_conversation = [
    ("My name is Alice", "Initial introduction"),
    ("What programming languages do you recommend?", "Question"),
    ("What was my name again?", "Memory test"),
    ("Which of those languages is best for web development?", "Context reference")
]

for user_msg, note in context_conversation:
    print(f"You: {user_msg}")
    print(f"     ({note})")
    response = context_bot.get_response(user_msg)
    print(f"Bot: {response}\n")

print("\n✅ Exercise 7 Complete!")
print("\n💡 Key Takeaways:")
print("  - Conversation history enables context")
print("  - System message sets personality/behavior")
print("  - Each message adds to token count")
print("  - Clear history when context gets too long")
print("  - Track costs for production use")

print("\n\n🎯 INTERACTIVE MODE:")
print("=" * 70)
print("\nReady to chat? Uncomment the line below to start:\n")
print("# bot = SimpleChatbot()")
print("# bot.chat_loop()")
print("\nOr run this script and use the interactive chatbot!")

# Uncomment to enable interactive mode
if __name__ == "__main__":
    response = input("\nWould you like to start an interactive chat? (y/n): ")
    if response.lower() == 'y':
        interactive_bot = SimpleChatbot(
            system_message="You are a helpful, friendly assistant."
        )
        interactive_bot.chat_loop()
