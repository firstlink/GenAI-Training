"""
Lab 6 - Exercise 1: Basic Calculator Agents
Solution for building first agents with tool calling for OpenAI and Claude

Learning Objectives:
- Implement tool/function calling with OpenAI
- Implement tool use with Claude (Anthropic)
- Understand agent execution loop
- Compare API differences between providers
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

# Load environment variables
load_dotenv()


# ============================================================================
# SHARED CALCULATOR FUNCTION
# ============================================================================

def calculator(expression: str) -> float:
    """
    Evaluate a mathematical expression

    Args:
        expression: Mathematical expression as string (e.g., "25 * 4 + 10")

    Returns:
        Result of calculation

    Raises:
        Exception: If expression is invalid or dangerous
    """
    try:
        # Security: Limit to basic math operations
        allowed_chars = set('0123456789+-*/.()')
        if not all(c in allowed_chars or c.isspace() for c in expression):
            return f"Error: Expression contains invalid characters"

        result = eval(expression)
        return result
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# OPENAI CALCULATOR AGENT
# ============================================================================

class OpenAICalculatorAgent:
    """
    Calculator agent using OpenAI's function calling

    Features:
    - Tool definition with OpenAI schema
    - Automatic tool selection
    - Tool execution and result integration
    """

    def __init__(self, api_key: str = None):
        """
        Initialize OpenAI calculator agent

        Args:
            api_key: OpenAI API key (or from environment)
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))

        # Define tool schema for OpenAI
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Performs mathematical calculations. Use for any arithmetic operations, percentages, or math problems.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate, e.g., '25 * 4 + 10' or '340 * 0.15'"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]

    def run(self, user_message: str, verbose: bool = True) -> str:
        """
        Run calculator agent with user message

        Args:
            user_message: User's question or request
            verbose: Print detailed execution steps

        Returns:
            Final agent response
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"OpenAI Calculator Agent")
            print('='*60)
            print(f"USER: {user_message}")
            print('='*60)

        # Initial call to GPT
        messages = [{"role": "user", "content": user_message}]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"  # Let model decide if tool is needed
        )

        response_message = response.choices[0].message

        # Check if model wants to use a tool
        if response_message.tool_calls:
            if verbose:
                print("\n🔧 Agent is using a tool!")

            # Get tool call details
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"   Tool: {function_name}")
                print(f"   Arguments: {function_args}")

            # Execute the function
            if function_name == "calculator":
                result = calculator(function_args["expression"])
                if verbose:
                    print(f"   Result: {result}")

            # Send result back to GPT for final response
            messages.append(response_message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": str(result)
            })

            # Get final response
            final_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

            final_answer = final_response.choices[0].message.content

            if verbose:
                print(f"\n💬 Agent: {final_answer}")
                print('='*60)

            return final_answer

        else:
            # No tool needed - direct response
            if verbose:
                print(f"\n💬 Agent (no tool): {response_message.content}")
                print('='*60)

            return response_message.content


# ============================================================================
# CLAUDE CALCULATOR AGENT
# ============================================================================

class ClaudeCalculatorAgent:
    """
    Calculator agent using Claude's tool use

    Features:
    - Tool definition with Claude schema
    - Tool use detection
    - Result integration with conversation
    """

    def __init__(self, api_key: str = None):
        """
        Initialize Claude calculator agent

        Args:
            api_key: Anthropic API key (or from environment)
        """
        self.client = Anthropic(api_key=api_key or os.getenv('ANTHROPIC_API_KEY'))

        # Define tool schema for Claude
        self.tools = [
            {
                "name": "calculator",
                "description": "Performs mathematical calculations. Use for any arithmetic operations, percentages, or math problems.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate, e.g., '25 * 4 + 10' or '340 * 0.15'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        ]

    def run(self, user_message: str, verbose: bool = True) -> str:
        """
        Run calculator agent with user message

        Args:
            user_message: User's question or request
            verbose: Print detailed execution steps

        Returns:
            Final agent response
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Claude Calculator Agent")
            print('='*60)
            print(f"USER: {user_message}")
            print('='*60)

        # Initial call to Claude
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=self.tools,
            messages=[{"role": "user", "content": user_message}]
        )

        # Check if Claude wants to use a tool
        if response.stop_reason == "tool_use":
            if verbose:
                print("\n🔧 Agent is using a tool!")

            # Extract tool use block
            tool_use = next(block for block in response.content if block.type == "tool_use")

            tool_name = tool_use.name
            tool_input = tool_use.input

            if verbose:
                print(f"   Tool: {tool_name}")
                print(f"   Arguments: {tool_input}")

            # Execute the function
            if tool_name == "calculator":
                result = calculator(tool_input["expression"])
                if verbose:
                    print(f"   Result: {result}")

            # Send result back to Claude
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                tools=self.tools,
                messages=[
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": response.content},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": str(result)
                            }
                        ]
                    }
                ]
            )

            final_answer = response.content[0].text

            if verbose:
                print(f"\n💬 Agent: {final_answer}")
                print('='*60)

            return final_answer

        else:
            # No tool needed
            final_answer = response.content[0].text

            if verbose:
                print(f"\n💬 Agent (no tool): {final_answer}")
                print('='*60)

            return final_answer


# ============================================================================
# COMPARISON UTILITY
# ============================================================================

def compare_agents(query: str, openai_agent: OpenAICalculatorAgent, claude_agent: ClaudeCalculatorAgent):
    """
    Compare OpenAI and Claude agents side-by-side

    Args:
        query: Test query
        openai_agent: OpenAI calculator agent
        claude_agent: Claude calculator agent
    """
    print(f"\n{'#'*70}")
    print(f"COMPARISON: {query}")
    print('#'*70)

    # Test OpenAI
    openai_result = openai_agent.run(query, verbose=True)

    # Test Claude
    try:
        claude_result = claude_agent.run(query, verbose=True)
    except Exception as e:
        print(f"\n⚠️  Claude agent error: {e}")
        claude_result = None

    # Summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print('='*70)
    print(f"OpenAI result: {openai_result}")
    if claude_result:
        print(f"Claude result: {claude_result}")
    print('='*70)


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Demonstrate calculator agents with OpenAI and Claude"""
    print("="*70)
    print("EXERCISE 1: BASIC CALCULATOR AGENTS")
    print("="*70)

    # Check API keys
    has_openai = bool(os.getenv('OPENAI_API_KEY'))
    has_claude = bool(os.getenv('ANTHROPIC_API_KEY'))

    print("\n🔑 API Key Status:")
    print(f"   OpenAI: {'✅' if has_openai else '❌'}")
    print(f"   Claude: {'✅' if has_claude else '❌'}")

    if not has_openai:
        print("\n❌ Error: OPENAI_API_KEY required for this exercise")
        return

    # Initialize agents
    print("\n" + "="*70)
    print("INITIALIZING AGENTS")
    print("="*70)

    openai_agent = OpenAICalculatorAgent()
    print("✅ OpenAI Calculator Agent initialized")

    claude_agent = None
    if has_claude:
        claude_agent = ClaudeCalculatorAgent()
        print("✅ Claude Calculator Agent initialized")
    else:
        print("⚠️  Claude agent skipped (no API key)")

    # Test cases
    test_cases = [
        "What is 15% of 340?",
        "Calculate 25 * 17 + 100",
        "What is 1000 divided by 8?",
        "Hello, how are you?"  # Should NOT use calculator
    ]

    # Test OpenAI Agent
    print("\n" + "="*70)
    print("TEST: OPENAI CALCULATOR AGENT")
    print("="*70)

    for i, test_query in enumerate(test_cases, 1):
        print(f"\n--- Test {i}/{len(test_cases)} ---")
        openai_agent.run(test_query, verbose=True)

    # Test Claude Agent (if available)
    if claude_agent:
        print("\n" + "="*70)
        print("TEST: CLAUDE CALCULATOR AGENT")
        print("="*70)

        for i, test_query in enumerate(test_cases, 1):
            print(f"\n--- Test {i}/{len(test_cases)} ---")
            try:
                claude_agent.run(test_query, verbose=True)
            except Exception as e:
                print(f"Error: {e}")

    # Comparison (if both available)
    if claude_agent:
        print("\n" + "="*70)
        print("SIDE-BY-SIDE COMPARISON")
        print("="*70)

        compare_agents("What is 20% of 500?", openai_agent, claude_agent)

    # Summary
    print("\n" + "="*70)
    print("✅ EXERCISE 1 COMPLETE!")
    print("="*70)

    print("\n💡 KEY LEARNINGS:")
    print("   1. Tool calling enables agents to use external functions")
    print("   2. OpenAI uses 'function calling' with tool_calls")
    print("   3. Claude uses 'tool use' with tool_use blocks")
    print("   4. Both require: tool definition → execution → result integration")
    print("   5. Agents intelligently decide when to use tools")

    print("\n🔍 API DIFFERENCES:")
    print("   OpenAI:")
    print("     • Schema: 'type': 'function' wrapper")
    print("     • Parameters: 'parameters' key")
    print("     • Response: tool_calls array")
    print("     • Result: role='tool' message")
    print("\n   Claude:")
    print("     • Schema: Direct tool definition")
    print("     • Parameters: 'input_schema' key")
    print("     • Response: stop_reason='tool_use'")
    print("     • Result: tool_result in conversation")

    print("\n🎯 AGENT EXECUTION LOOP:")
    print("   1. User query → LLM")
    print("   2. LLM decides to use tool")
    print("   3. Execute tool function")
    print("   4. Send result back to LLM")
    print("   5. LLM generates final response")


if __name__ == "__main__":
    main()
