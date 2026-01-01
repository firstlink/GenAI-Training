"""
Lab 1 - Capstone Project: SupportGenie v0.1
Professional AI-Powered Customer Support Chatbot

Features:
- Intelligent customer support conversations
- Multiple conversation modes (support, sales, technical)
- Conversation history with context
- Token and cost tracking
- Session management
- Export conversation logs
- Professional error handling

This is v0.1 - Foundation version with core chatbot capabilities.
Future versions will add:
- v0.2: Advanced prompt engineering (Lab 2)
- v3.0: RAG with knowledge base (Lab 5)
"""

from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
import json

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


class SupportGenie:
    """
    Professional AI Customer Support Chatbot

    A production-ready chatbot with:
    - Configurable support modes
    - Conversation tracking
    - Cost monitoring
    - Session management
    """

    # Support mode configurations
    MODES = {
        "support": {
            "name": "Customer Support",
            "system_message": """You are SupportGenie, a professional customer support agent.

Your role:
- Help customers with their questions and issues
- Be empathetic, patient, and professional
- Provide clear, step-by-step solutions
- If you can't help, suggest contacting a human agent

Guidelines:
- Always be polite and friendly
- Use the customer's name if they provide it
- Acknowledge their frustration if they're upset
- Provide specific, actionable solutions
- End on a positive note"""
        },
        "sales": {
            "name": "Sales Assistant",
            "system_message": """You are SupportGenie, a helpful sales assistant.

Your role:
- Answer product questions
- Help customers find the right solution
- Provide pricing information
- Guide customers through purchase decisions

Guidelines:
- Be consultative, not pushy
- Ask questions to understand needs
- Recommend products that fit customer requirements
- Be honest about limitations"""
        },
        "technical": {
            "name": "Technical Support",
            "system_message": """You are SupportGenie, a technical support specialist.

Your role:
- Diagnose technical issues
- Provide detailed troubleshooting steps
- Explain technical concepts clearly
- Escalate complex issues when needed

Guidelines:
- Ask diagnostic questions
- Provide step-by-step instructions
- Use technical terms appropriately
- Verify the solution worked"""
        }
    }

    def __init__(self, mode="support", model="gpt-3.5-turbo", company_name="TechCorp"):
        """
        Initialize SupportGenie

        Args:
            mode: Support mode (support, sales, technical)
            model: OpenAI model to use
            company_name: Company name to personalize responses
        """
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode. Choose from: {list(self.MODES.keys())}")

        self.mode = mode
        self.model = model
        self.company_name = company_name

        # Customize system message with company name
        system_message = self.MODES[mode]["system_message"]
        system_message = system_message.replace("TechCorp", company_name)

        # Initialize conversation
        self.conversation_history = [
            {"role": "system", "content": system_message}
        ]

        # Tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()
        self.total_tokens = 0
        self.total_cost = 0.0
        self.message_count = 0

    def get_response(self, user_message, stream=False):
        """
        Get chatbot response

        Args:
            user_message: User's input
            stream: Enable streaming response

        Returns:
            Assistant's response (string or stream)
        """
        # Add user message
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        try:
            if stream:
                # Streaming response
                response_stream = client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    max_tokens=500,
                    temperature=0.7,
                    stream=True
                )

                # Collect full response
                full_response = ""
                for chunk in response_stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content

                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })

                self.message_count += 1

            else:
                # Non-streaming response
                response = client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    max_tokens=500,
                    temperature=0.7
                )

                # Extract response
                assistant_message = response.choices[0].message.content

                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })

                # Update tracking
                self.total_tokens += response.usage.total_tokens
                cost = self._calculate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
                self.total_cost += cost
                self.message_count += 1

                return assistant_message

        except Exception as e:
            error_msg = f"I apologize, but I'm experiencing technical difficulties. Error: {str(e)}"
            self.conversation_history.append({
                "role": "assistant",
                "content": error_msg
            })
            return error_msg

    def _calculate_cost(self, prompt_tokens, completion_tokens):
        """Calculate cost for API call"""
        # GPT-3.5-turbo pricing
        input_cost = (prompt_tokens / 1000) * 0.0015
        output_cost = (completion_tokens / 1000) * 0.002
        return input_cost + output_cost

    def get_stats(self):
        """Get session statistics"""
        duration = datetime.now() - self.session_start

        return {
            "session_id": self.session_id,
            "mode": self.MODES[self.mode]["name"],
            "duration": str(duration).split('.')[0],  # Remove microseconds
            "messages": self.message_count,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "avg_cost_per_message": self.total_cost / self.message_count if self.message_count > 0 else 0
        }

    def export_conversation(self, filename=None):
        """
        Export conversation to JSON file

        Args:
            filename: Output filename (optional)

        Returns:
            Filename where conversation was saved
        """
        if filename is None:
            filename = f"conversation_{self.session_id}.json"

        export_data = {
            "session_id": self.session_id,
            "company": self.company_name,
            "mode": self.mode,
            "model": self.model,
            "start_time": self.session_start.isoformat(),
            "conversation": self.conversation_history[1:],  # Exclude system message
            "stats": self.get_stats()
        }

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        return filename

    def clear_history(self):
        """Clear conversation (keep system message)"""
        system_msg = self.conversation_history[0]
        self.conversation_history = [system_msg]
        self.total_tokens = 0
        self.total_cost = 0.0
        self.message_count = 0

    def run(self):
        """Run interactive chat interface"""
        print("=" * 70)
        print(f"SUPPORTGENIE v0.1 - {self.MODES[self.mode]['name']}")
        print("=" * 70)
        print(f"Company: {self.company_name}")
        print(f"Session ID: {self.session_id}")
        print("\nCommands:")
        print("  'quit' - End conversation and show stats")
        print("  'clear' - Clear conversation history")
        print("  'stats' - Show session statistics")
        print("  'export' - Export conversation to JSON")
        print("  'mode <name>' - Switch mode (support/sales/technical)")
        print("\nStart chatting!\n")

        while True:
            user_input = input("You: ").strip()

            # Handle commands
            if not user_input:
                continue

            elif user_input.lower() == 'quit':
                print("\n" + "=" * 70)
                print("CONVERSATION ENDED")
                print("=" * 70)
                stats = self.get_stats()
                for key, value in stats.items():
                    print(f"{key.replace('_', ' ').title():25} {value}")
                print("=" * 70)
                break

            elif user_input.lower() == 'clear':
                self.clear_history()
                print("✅ Conversation cleared.\n")
                continue

            elif user_input.lower() == 'stats':
                stats = self.get_stats()
                print("\n📊 Session Statistics:")
                for key, value in stats.items():
                    print(f"   {key.replace('_', ' ').title():25} {value}")
                print()
                continue

            elif user_input.lower() == 'export':
                filename = self.export_conversation()
                print(f"✅ Conversation exported to: {filename}\n")
                continue

            elif user_input.lower().startswith('mode '):
                new_mode = user_input.split()[1]
                if new_mode in self.MODES:
                    self.__init__(mode=new_mode, company_name=self.company_name)
                    print(f"✅ Switched to {self.MODES[new_mode]['name']} mode\n")
                else:
                    print(f"❌ Invalid mode. Choose from: {list(self.MODES.keys())}\n")
                continue

            # Get response
            try:
                response = self.get_response(user_input)
                print(f"\n{self.company_name} Agent: {response}\n")

            except Exception as e:
                print(f"\n❌ Error: {e}\n")


# ============================================================================
# DEMO & TESTING
# ============================================================================

def demo_supportgenie():
    """Demonstrate SupportGenie capabilities"""
    print("=" * 70)
    print("SUPPORTGENIE v0.1 - DEMONSTRATION")
    print("=" * 70)

    # Test each mode
    modes = ["support", "sales", "technical"]

    for mode in modes:
        print(f"\n\n{'=' * 70}")
        print(f"MODE: {SupportGenie.MODES[mode]['name'].upper()}")
        print("=" * 70)

        genie = SupportGenie(mode=mode, company_name="TechCorp")

        # Sample conversations for each mode
        if mode == "support":
            conversations = [
                "I can't log into my account",
                "It says my password is incorrect"
            ]
        elif mode == "sales":
            conversations = [
                "What products do you offer?",
                "I need something for small business"
            ]
        else:  # technical
            conversations = [
                "My app keeps crashing",
                "It happens when I click the export button"
            ]

        for msg in conversations:
            print(f"\nCustomer: {msg}")
            response = genie.get_response(msg)
            print(f"Agent: {response}")

        # Show stats
        stats = genie.get_stats()
        print(f"\n📊 {mode.title()} session stats:")
        print(f"   Messages: {stats['messages']}")
        print(f"   Cost: ${stats['total_cost']:.6f}")


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("SUPPORTGENIE v0.1 - AI CUSTOMER SUPPORT")
    print("=" * 70)

    # Run demo or interactive mode
    choice = input("\nChoose mode:\n  1. Demo (automated)\n  2. Interactive chat\n  3. Exit\n\nChoice: ")

    if choice == "1":
        demo_supportgenie()
    elif choice == "2":
        print("\nSelect support mode:")
        print("  1. Customer Support")
        print("  2. Sales Assistant")
        print("  3. Technical Support")

        mode_choice = input("\nChoice (1-3): ")
        mode_map = {"1": "support", "2": "sales", "3": "technical"}

        if mode_choice in mode_map:
            company = input("Enter company name (default: TechCorp): ").strip() or "TechCorp"
            genie = SupportGenie(mode=mode_map[mode_choice], company_name=company)
            genie.run()
        else:
            print("Invalid choice")
    else:
        print("Goodbye!")


if __name__ == "__main__":
    main()
