"""
Lab 6 - All Exercises: AI Agents & Tool Calling
Consolidated solution covering all Lab 6 concepts

Includes:
- Exercise 1: Basic Calculator Agent (OpenAI & Claude)
- Exercise 2: Multi-Tool Assistant
- Exercise 3: Conditional Workflow Agent
- Exercise 4: Resilient Agent with Error Handling
- Capstone: AgentHub v1.0 Multi-Agent Platform

Learning Objectives:
- Implement tool/function calling with OpenAI and Claude
- Build multi-tool agents with intelligent tool selection
- Create conditional workflows and branching logic
- Handle errors gracefully in production
- Design multi-agent systems with routing
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)


# ============================================================================
# EXERCISE 1: BASIC CALCULATOR AGENT
# ============================================================================

def calculator(expression: str) -> float:
    """Basic calculator function"""
    try:
        allowed_chars = set('0123456789+-*/.()')
        if not all(c in allowed_chars or c.isspace() for c in expression):
            return f"Error: Invalid characters"
        return eval(expression)
    except Exception as e:
        return f"Error: {str(e)}"


class BasicCalculatorAgent:
    """Simple calculator agent with OpenAI"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.tools = [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Performs mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                }
            }
        }]

    def run(self, query: str) -> str:
        """Execute calculator agent"""
        messages = [{"role": "user", "content": query}]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        if response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            result = calculator(function_args["expression"])

            messages.append(response_message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "calculator",
                "content": str(result)
            })

            final_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return final_response.choices[0].message.content

        return response_message.content


# ============================================================================
# EXERCISE 2: MULTI-TOOL ASSISTANT
# ============================================================================

def get_current_datetime(format: str = "full") -> str:
    """Get current date/time"""
    now = datetime.now()
    if format == "date":
        return now.strftime("%Y-%m-%d")
    elif format == "time":
        return now.strftime("%H:%M:%S")
    return now.strftime("%Y-%m-%d %H:%M:%S")


def search_knowledge_base(query: str) -> str:
    """Search simulated knowledge base"""
    knowledge = {
        "python": "Python is a high-level programming language known for simplicity.",
        "machine learning": "ML is a subset of AI that enables systems to learn from data.",
        "ai agents": "AI agents are autonomous systems that use tools and reasoning.",
        "databases": "Databases store organized data. SQL uses tables, NoSQL uses flexible schemas."
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower or query_lower in key:
            return f"**{key.title()}**: {value}"

    return "No information found in knowledge base."


def send_notification(message: str, priority: str = "normal") -> str:
    """Send notification (simulated)"""
    return f"✓ Notification sent (Priority: {priority}): {message}"


class MultiToolAssistant:
    """Agent with multiple tools"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Performs mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Math expression"}
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_datetime",
                    "description": "Gets current date and time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "format": {
                                "type": "string",
                                "enum": ["full", "date", "time"],
                                "description": "Output format"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "Search knowledge base for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "send_notification",
                    "description": "Send a notification or reminder",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Notification message"},
                            "priority": {
                                "type": "string",
                                "enum": ["low", "normal", "high"],
                                "description": "Priority level"
                            }
                        },
                        "required": ["message"]
                    }
                }
            }
        ]

        self.functions = {
            "calculator": calculator,
            "get_current_datetime": get_current_datetime,
            "search_knowledge_base": search_knowledge_base,
            "send_notification": send_notification
        }

    def run(self, query: str, verbose: bool = False) -> str:
        """Execute multi-tool assistant"""
        messages = [{"role": "user", "content": query}]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        if response_message.tool_calls:
            if verbose:
                print(f"🔧 Using {len(response_message.tool_calls)} tool(s)")

            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if verbose:
                    print(f"   • {function_name}({json.dumps(function_args)})")

                function_to_call = self.functions[function_name]
                result = function_to_call(**function_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": str(result)
                })

            final_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return final_response.choices[0].message.content

        return response_message.content


# ============================================================================
# EXERCISE 3: CONDITIONAL WORKFLOW AGENT
# ============================================================================

def check_business_hours() -> dict:
    """Check if within business hours (9 AM - 5 PM)"""
    now = datetime.now()
    hour = now.hour
    is_open = 9 <= hour < 17

    return {
        "is_open": is_open,
        "current_hour": hour,
        "message": f"Business is {'open' if is_open else 'closed'} (Hour: {hour})"
    }


def send_email(recipient: str, subject: str, body: str) -> str:
    """Send email (simulated)"""
    return f"✓ Email sent to {recipient}\n   Subject: {subject}"


def create_ticket(title: str, description: str, priority: str = "normal") -> str:
    """Create support ticket (simulated)"""
    import random
    ticket_id = random.randint(1000, 9999)
    return f"✓ Ticket #{ticket_id} created\n   Title: {title}\n   Priority: {priority}"


class ConditionalWorkflowAgent:
    """Agent with conditional logic based on business hours"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "check_business_hours",
                    "description": "Check if business is currently open (9 AM - 5 PM)",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Send email during business hours for urgent matters",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recipient": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"}
                        },
                        "required": ["recipient", "subject", "body"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_ticket",
                    "description": "Create ticket for non-urgent matters or outside business hours",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "priority": {"type": "string", "enum": ["low", "normal", "high"]}
                        },
                        "required": ["title", "description"]
                    }
                }
            }
        ]

        self.functions = {
            "check_business_hours": check_business_hours,
            "send_email": send_email,
            "create_ticket": create_ticket
        }

    def run(self, query: str, max_iterations: int = 5) -> str:
        """Execute conditional workflow"""
        system_prompt = """You are a support assistant. Follow these rules:
1. First check business hours
2. If OPEN and urgent: use send_email
3. If CLOSED or non-urgent: use create_ticket
4. Explain your decision"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        for iteration in range(max_iterations):
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            if not response_message.tool_calls:
                return response_message.content

            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                function_to_call = self.functions[function_name]
                result = function_to_call(**function_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": str(result) if not isinstance(result, dict) else json.dumps(result)
                })

        return "Max iterations reached"


# ============================================================================
# EXERCISE 4: RESILIENT AGENT WITH ERROR HANDLING
# ============================================================================

def safe_calculator(expression: str) -> Dict[str, Any]:
    """Calculator with comprehensive error handling"""
    try:
        if not expression or not isinstance(expression, str):
            return {
                "success": False,
                "error": "Invalid expression: must be non-empty string",
                "error_type": "ValidationError"
            }

        dangerous_keywords = ['import', 'exec', 'eval', '__', 'open']
        if any(kw in expression.lower() for kw in dangerous_keywords):
            return {
                "success": False,
                "error": "Expression contains forbidden operations",
                "error_type": "SecurityError"
            }

        result = eval(expression)
        return {"success": True, "result": result, "expression": expression}

    except ZeroDivisionError:
        return {
            "success": False,
            "error": "Cannot divide by zero",
            "error_type": "ZeroDivisionError"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Calculation failed: {str(e)}",
            "error_type": type(e).__name__
        }


def primary_search(query: str) -> Dict[str, Any]:
    """Primary search (may fail 40% of time)"""
    import random
    if random.random() < 0.4:
        return {
            "success": False,
            "error": "Primary search service unavailable",
            "service": "primary"
        }
    return {
        "success": True,
        "results": [f"Primary result for: {query}"],
        "service": "primary"
    }


def backup_search(query: str) -> Dict[str, Any]:
    """Backup search (more reliable, 10% failure)"""
    import random
    if random.random() < 0.1:
        return {
            "success": False,
            "error": "Backup search unavailable",
            "service": "backup"
        }
    return {
        "success": True,
        "results": [f"Backup result for: {query}"],
        "service": "backup"
    }


class ResilientAgent:
    """Agent with comprehensive error handling"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.logger = logging.getLogger("ResilientAgent")

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "safe_calculator",
                    "description": "Performs calculations with error handling",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Math expression"}
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "primary_search",
                    "description": "Primary search (fast but may be unavailable)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "backup_search",
                    "description": "Backup search (use if primary fails)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        self.functions = {
            "safe_calculator": safe_calculator,
            "primary_search": primary_search,
            "backup_search": backup_search
        }

    def run(self, query: str, max_iterations: int = 5) -> str:
        """Execute resilient agent"""
        system_prompt = """You are a resilient assistant with error handling.
Rules:
1. If a tool fails, check the error in the result
2. For search: if primary_search fails, try backup_search
3. Inform user about errors in a friendly way
4. Continue trying alternatives when possible"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        for iteration in range(max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    timeout=30
                )

                response_message = response.choices[0].message

                if not response_message.tool_calls:
                    return response_message.content

                messages.append(response_message)

                for tool_call in response_message.tool_calls:
                    try:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        if function_name in self.functions:
                            result = self.functions[function_name](**function_args)
                        else:
                            result = {"success": False, "error": f"Unknown tool: {function_name}"}

                        if isinstance(result, dict) and not result.get("success", True):
                            self.logger.warning(f"Tool {function_name} failed: {result.get('error')}")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps(result) if isinstance(result, dict) else str(result)
                        })

                    except Exception as e:
                        self.logger.error(f"Tool execution error: {str(e)}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps({"success": False, "error": str(e)})
                        })

            except Exception as e:
                self.logger.error(f"LLM API error: {str(e)}")
                return f"I encountered an error: {str(e)}"

        return "I couldn't complete the task within the iteration limit."


# ============================================================================
# CAPSTONE: AGENTHUB v1.0 - MULTI-AGENT PLATFORM
# ============================================================================

class BaseAgent:
    """Base agent class with common functionality"""

    def __init__(self, name: str, tools: List[Dict], functions: Dict):
        self.name = name
        self.tools = tools
        self.functions = functions
        self.logger = logging.getLogger(name)
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def run(self, user_message: str, system_prompt: str = None) -> str:
        """Run the agent"""
        messages = [{"role": "user", "content": user_message}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        max_iterations = 5
        for iteration in range(max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto"
                )

                response_message = response.choices[0].message

                if not response_message.tool_calls:
                    return response_message.content

                messages.append(response_message)

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    if function_name in self.functions:
                        result = self.functions[function_name](**function_args)
                    else:
                        result = {"success": False, "error": "Unknown tool"}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(result) if isinstance(result, dict) else str(result)
                    })

            except Exception as e:
                self.logger.error(f"Error: {str(e)}")
                return f"Error: {str(e)}"

        return "Task incomplete - max iterations reached"


class MathAgent(BaseAgent):
    """Agent specialized in mathematics"""

    def __init__(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Performs mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                }
            }
        }]
        super().__init__("MathAgent", tools, {"calculator": calculator})


class InfoAgent(BaseAgent):
    """Agent specialized in information retrieval"""

    def __init__(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "Search knowledge base",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_datetime",
                    "description": "Get current date/time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "format": {
                                "type": "string",
                                "enum": ["full", "date", "time"]
                            }
                        },
                        "required": []
                    }
                }
            }
        ]

        functions = {
            "search_knowledge_base": search_knowledge_base,
            "get_current_datetime": get_current_datetime
        }
        super().__init__("InfoAgent", tools, functions)


class TaskAgent(BaseAgent):
    """Agent specialized in task management"""

    def __init__(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "send_notification",
                    "description": "Send a notification",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                            "priority": {"type": "string", "enum": ["low", "normal", "high"]}
                        },
                        "required": ["message"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_ticket",
                    "description": "Create a support ticket",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "priority": {"type": "string", "enum": ["low", "normal", "high"]}
                        },
                        "required": ["title", "description"]
                    }
                }
            }
        ]

        functions = {
            "send_notification": send_notification,
            "create_ticket": create_ticket
        }
        super().__init__("TaskAgent", tools, functions)


class AgentHub:
    """
    Central hub that routes queries to specialized agents

    Features:
    - Intelligent query routing
    - Multiple specialized agents
    - Error handling
    - Logging and monitoring
    """

    def __init__(self):
        self.logger = logging.getLogger("AgentHub")
        self.agents = {
            "math": MathAgent(),
            "info": InfoAgent(),
            "task": TaskAgent()
        }
        self.logger.info("AgentHub initialized with 3 specialized agents")

    def route_query(self, user_message: str) -> str:
        """Determine which agent should handle the query"""
        message_lower = user_message.lower()

        # Math keywords
        if any(word in message_lower for word in ["calculate", "compute", "math", "%", "+", "-", "*", "/"]):
            return "math"

        # Task keywords
        elif any(word in message_lower for word in ["remind", "notification", "ticket", "alert"]):
            return "task"

        # Info keywords
        elif any(word in message_lower for word in ["search", "find", "what is", "tell me", "date", "time"]):
            return "info"

        # Default to info
        return "info"

    def process(self, user_message: str, verbose: bool = True) -> str:
        """Process a user message by routing to appropriate agent"""
        if verbose:
            print(f"\n{'='*70}")
            print(f"🏢 AGENTHUB v1.0")
            print('='*70)
            print(f"USER: {user_message}\n")

        # Route to appropriate agent
        agent_type = self.route_query(user_message)
        agent = self.agents[agent_type]

        if verbose:
            print(f"🔀 Routing to: {agent.name}")
            print(f"{'='*70}\n")

        self.logger.info(f"Routing to {agent.name}")

        # Execute agent
        try:
            result = agent.run(user_message)

            if verbose:
                print(f"\n💬 {agent.name} Response:")
                print(f"   {result}")
                print('='*70)

            return result

        except Exception as e:
            error_msg = f"Error in {agent.name}: {str(e)}"
            self.logger.error(error_msg)

            if verbose:
                print(f"\n❌ Error: {error_msg}")
                print('='*70)

            return f"I encountered an error: {str(e)}"


# ============================================================================
# DEMONSTRATION & TESTING
# ============================================================================

def demo_exercise1():
    """Demo Exercise 1: Basic Calculator Agent"""
    print("\n" + "="*70)
    print("EXERCISE 1: BASIC CALCULATOR AGENT")
    print("="*70)

    agent = BasicCalculatorAgent()

    tests = [
        "What is 15% of 340?",
        "Calculate 25 * 17",
        "Hello!"  # Should not use calculator
    ]

    for query in tests:
        print(f"\nQuery: {query}")
        result = agent.run(query)
        print(f"Response: {result}")


def demo_exercise2():
    """Demo Exercise 2: Multi-Tool Assistant"""
    print("\n" + "="*70)
    print("EXERCISE 2: MULTI-TOOL ASSISTANT")
    print("="*70)

    agent = MultiToolAssistant()

    tests = [
        "What is 20% of 500?",
        "What's today's date?",
        "Tell me about Python",
        "Remind me to review agents"
    ]

    for query in tests:
        print(f"\nQuery: {query}")
        result = agent.run(query, verbose=True)
        print(f"Response: {result}")


def demo_exercise3():
    """Demo Exercise 3: Conditional Workflow Agent"""
    print("\n" + "="*70)
    print("EXERCISE 3: CONDITIONAL WORKFLOW AGENT")
    print("="*70)

    agent = ConditionalWorkflowAgent()

    tests = [
        "I need urgent help with a system outage",
        "I have a question about my account"
    ]

    for query in tests:
        print(f"\nQuery: {query}")
        result = agent.run(query)
        print(f"Response: {result}")


def demo_exercise4():
    """Demo Exercise 4: Resilient Agent"""
    print("\n" + "="*70)
    print("EXERCISE 4: RESILIENT AGENT WITH ERROR HANDLING")
    print("="*70)

    agent = ResilientAgent()

    tests = [
        "Calculate 50 * 20",
        "Calculate 10 / 0",  # Error case
        "Search for Python information"  # May trigger fallback
    ]

    for query in tests:
        print(f"\nQuery: {query}")
        result = agent.run(query)
        print(f"Response: {result}")


def demo_capstone():
    """Demo Capstone: AgentHub v1.0"""
    print("\n" + "="*70)
    print("CAPSTONE: AGENTHUB v1.0 - MULTI-AGENT PLATFORM")
    print("="*70)

    hub = AgentHub()

    tests = [
        "What is 25% of 340?",
        "Tell me about Python",
        "What's today's date?",
        "Remind me to review AI agents",
        "Calculate 127 * 43",
        "Create a ticket for the login bug"
    ]

    for query in tests:
        hub.process(query)


def main():
    """Run all demonstrations"""
    print("="*70)
    print("LAB 6: AI AGENTS & TOOL CALLING - ALL EXERCISES")
    print("="*70)

    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n❌ Error: OPENAI_API_KEY not found")
        print("   Please set your API key in .env file")
        return

    print("\n✅ OpenAI API key loaded")

    # Run all demos
    try:
        demo_exercise1()
        demo_exercise2()
        demo_exercise3()
        demo_exercise4()
        demo_capstone()

        # Summary
        print("\n" + "="*70)
        print("✅ ALL EXERCISES COMPLETED!")
        print("="*70)

        print("\n💡 KEY CONCEPTS COVERED:")
        print("   ✓ Tool/function calling with LLMs")
        print("   ✓ Multi-tool agents with intelligent selection")
        print("   ✓ Conditional workflows and branching logic")
        print("   ✓ Error handling and resilience")
        print("   ✓ Multi-agent systems with routing")
        print("   ✓ Production-ready agent design")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")


if __name__ == "__main__":
    main()
