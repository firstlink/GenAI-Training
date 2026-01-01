# 🛠️ Lab 6: AI Agents & Tool Calling - Hands-On Lab

> **Duration:** 90-120 minutes
> **Difficulty:** Intermediate-Advanced
> **Prerequisites:** Labs 1-5 completed, Python environment set up

---

## 📋 Lab Overview

In this hands-on lab, you'll build progressively more sophisticated AI agents, culminating in a **production-ready intelligent agent system**. You'll implement tool calling with both OpenAI and Claude, create multi-tool agents, and build sophisticated workflows.

### What You'll Build

By the end of this lab, you'll have created:

1. ✅ **Basic Calculator Agent** - Single tool calling with both OpenAI and Claude
2. ✅ **Multi-Tool Assistant** - Agent with 5+ different tools
3. ✅ **Conditional Workflow Agent** - Smart decision-making based on conditions
4. ✅ **Resilient Agent System** - Production error handling and fallbacks
5. ✅ **AgentHub v1.0** - Complete multi-agent platform (Capstone)

---

## 🎯 Learning Objectives

By completing this lab, you will:
- ✓ Implement tool/function calling with OpenAI and Claude APIs
- ✓ Build agents that can use multiple tools intelligently
- ✓ Create conditional workflows and branching logic
- ✓ Handle errors gracefully in production agents
- ✓ Design sophisticated multi-step agent behaviors
- ✓ Build reusable agent frameworks

---

## 📂 Setup

### Step 1: Create Lab Directory

```bash
mkdir lab6_ai_agents
cd lab6_ai_agents
```

### Step 2: Install Dependencies

```bash
pip install openai anthropic python-dotenv
```

### Step 3: Create Environment File

Create `.env` in your lab directory:

```bash
# .env
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

### Step 4: Test Setup

Create `test_setup.py`:

```python
import os
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

load_dotenv()

print("✓ OpenAI key loaded" if os.getenv('OPENAI_API_KEY') else "✗ OpenAI key missing")
print("✓ Anthropic key loaded" if os.getenv('ANTHROPIC_API_KEY') else "✗ Anthropic key missing")

# Test API connections
try:
    openai_client = OpenAI()
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=10
    )
    print("✓ OpenAI API working")
except Exception as e:
    print(f"✗ OpenAI API error: {e}")

try:
    anthropic_client = Anthropic()
    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        messages=[{"role": "user", "content": "Hi"}]
    )
    print("✓ Anthropic API working")
except Exception as e:
    print(f"✗ Anthropic API error: {e}")
```

Run: `python test_setup.py`

Expected output:
```
✓ OpenAI key loaded
✓ Anthropic key loaded
✓ OpenAI API working
✓ Anthropic API working
```

---

## 🔧 Exercise 1: Basic Calculator Agent (20-25 minutes)

**Goal:** Build your first agent with single tool calling for both OpenAI and Claude.

### Part A: OpenAI Calculator Agent

Create `exercise1_openai_calculator.py`:

```python
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# STEP 1: Define the calculator function
def calculator(expression: str) -> float:
    """
    Evaluate a mathematical expression

    Args:
        expression: Mathematical expression as string

    Returns:
        Result of calculation
    """
    try:
        result = eval(expression)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# STEP 2: Define the tool schema for OpenAI
tools = [
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
                        "description": "Mathematical expression to evaluate, e.g., '25 * 4 + 10'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# STEP 3: Build the agent
def run_calculator_agent(user_message):
    """
    Run a simple calculator agent with OpenAI
    """
    print(f"\n{'='*60}")
    print(f"USER: {user_message}")
    print('='*60)

    # Initial call to GPT
    messages = [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    # Check if model wants to use a tool
    if response_message.tool_calls:
        print("\n🔧 Agent is using a tool!")

        # Get tool call details
        tool_call = response_message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        print(f"   Tool: {function_name}")
        print(f"   Arguments: {function_args}")

        # Execute the function
        if function_name == "calculator":
            result = calculator(function_args["expression"])
            print(f"   Result: {result}")

        # Send result back to GPT
        messages.append(response_message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": str(result)
        })

        # Get final response
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        final_answer = final_response.choices[0].message.content
        print(f"\n💬 Agent: {final_answer}")
    else:
        # No tool needed
        print(f"\n💬 Agent: {response_message.content}")

    print('='*60)

# STEP 4: Test the agent
if __name__ == "__main__":
    # Test cases
    run_calculator_agent("What is 15% of 340?")
    run_calculator_agent("Calculate 25 * 17 + 100")
    run_calculator_agent("Hello, how are you?")  # Should NOT use calculator
```

**Run it:** `python exercise1_openai_calculator.py`

**Expected Output:**
```
============================================================
USER: What is 15% of 340?
============================================================

🔧 Agent is using a tool!
   Tool: calculator
   Arguments: {'expression': '340 * 0.15'}
   Result: 51.0

💬 Agent: 15% of 340 is 51.
============================================================
```

### Part B: Claude Calculator Agent

Create `exercise1_claude_calculator.py`:

```python
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# STEP 1: Define the calculator function (same as before)
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression"""
    try:
        result = eval(expression)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# STEP 2: Define the tool schema for Claude
tools = [
    {
        "name": "calculator",
        "description": "Performs mathematical calculations. Use for any arithmetic operations, percentages, or math problems.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g., '25 * 4 + 10'"
                }
            },
            "required": ["expression"]
        }
    }
]

# STEP 3: Build the agent
def run_claude_calculator_agent(user_message):
    """
    Run a simple calculator agent with Claude
    """
    print(f"\n{'='*60}")
    print(f"USER: {user_message}")
    print('='*60)

    # Initial call to Claude
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": user_message}]
    )

    # Check if Claude wants to use a tool
    if response.stop_reason == "tool_use":
        print("\n🔧 Agent is using a tool!")

        # Extract tool use
        tool_use = next(block for block in response.content if block.type == "tool_use")

        tool_name = tool_use.name
        tool_input = tool_use.input

        print(f"   Tool: {tool_name}")
        print(f"   Arguments: {tool_input}")

        # Execute the function
        if tool_name == "calculator":
            result = calculator(tool_input["expression"])
            print(f"   Result: {result}")

        # Send result back to Claude
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=tools,
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
        print(f"\n💬 Agent: {final_answer}")
    else:
        # No tool needed
        final_answer = response.content[0].text
        print(f"\n💬 Agent: {final_answer}")

    print('='*60)

# STEP 4: Test the agent
if __name__ == "__main__":
    run_claude_calculator_agent("What is 15% of 340?")
    run_claude_calculator_agent("Calculate 25 * 17 + 100")
    run_claude_calculator_agent("Hello, how are you?")
```

**Run it:** `python exercise1_claude_calculator.py`

### ✅ Checkpoint 1

**What you learned:**
- ✓ How to define tools for OpenAI (function calling)
- ✓ How to define tools for Claude (tool use)
- ✓ The agent execution loop: Call LLM → Execute Tool → Send Result → Get Final Response
- ✓ Key differences between OpenAI and Claude APIs

**Self-Check Questions:**
1. What are the three parts of a tool definition?
2. How does OpenAI's tool schema differ from Claude's?
3. When should an agent use a tool vs. responding directly?

<details>
<summary>📝 Click to see answers</summary>

1. **Three parts:** Definition (schema), Implementation (function), Registration (passing to API)
2. **Differences:** OpenAI uses `"type": "function"` wrapper and `"parameters"`, Claude uses direct schema with `"input_schema"`
3. **When to use tool:** For tasks requiring external data, calculations, or actions beyond the LLM's knowledge

</details>

---

## 🔧 Exercise 2: Multi-Tool Assistant (25-30 minutes)

**Goal:** Build an agent with multiple tools that can intelligently select which one to use.

Create `exercise2_multi_tool_assistant.py`:

```python
import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ==================== STEP 1: IMPLEMENT TOOLS ====================

def calculator(expression: str) -> float:
    """Perform mathematical calculations"""
    try:
        result = eval(expression)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def get_current_datetime(format: str = "full") -> str:
    """
    Get current date and time

    Args:
        format: "full", "date", or "time"

    Returns:
        Formatted current datetime
    """
    now = datetime.now()

    if format == "date":
        return now.strftime("%Y-%m-%d")
    elif format == "time":
        return now.strftime("%H:%M:%S")
    else:  # full
        return now.strftime("%Y-%m-%d %H:%M:%S")

def search_knowledge_base(query: str) -> str:
    """
    Search a simulated knowledge base

    Args:
        query: Search query

    Returns:
        Search results
    """
    knowledge = {
        "python": "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms.",
        "machine learning": "Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.",
        "ai agents": "AI agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve goals using tools and reasoning.",
        "web development": "Web development involves creating websites and web applications using HTML, CSS, JavaScript, and backend technologies.",
        "databases": "Databases are organized collections of data. SQL databases use structured tables while NoSQL databases offer more flexible schemas."
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower or query_lower in key:
            return f"**{key.title()}**: {value}"

    return "No information found in knowledge base."

def send_notification(message: str, priority: str = "normal") -> str:
    """
    Send a notification (simulated)

    Args:
        message: Notification message
        priority: "low", "normal", or "high"

    Returns:
        Confirmation message
    """
    return f"✓ Notification sent (Priority: {priority}): {message}"

# ==================== STEP 2: DEFINE TOOL SCHEMAS ====================

TOOLS = [
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
                        "description": "Mathematical expression to evaluate, e.g., '25 * 4 + 10'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Gets the current date and time. Use when user asks about 'today', 'now', current time, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["full", "date", "time"],
                        "description": "Format of output: 'full' (date and time), 'date' (only date), 'time' (only time)"
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
            "description": "Search through the knowledge base for information on programming, technology, and computing topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query or topic to look up"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "Send a notification or reminder. Use when user asks to remind them or send a notification.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Notification message content"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "description": "Priority level of the notification"
                    }
                },
                "required": ["message"]
            }
        }
    }
]

# ==================== STEP 3: TOOL EXECUTION MAPPING ====================

AVAILABLE_FUNCTIONS = {
    "calculator": calculator,
    "get_current_datetime": get_current_datetime,
    "search_knowledge_base": search_knowledge_base,
    "send_notification": send_notification
}

# ==================== STEP 4: BUILD MULTI-TOOL AGENT ====================

def run_multi_tool_agent(user_message, verbose=True):
    """
    Run an agent with multiple tools

    Args:
        user_message: User's query
        verbose: Print detailed execution steps

    Returns:
        Final agent response
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"USER: {user_message}")
        print('='*70)

    messages = [{"role": "user", "content": user_message}]

    # Initial LLM call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    # Check if tools were called
    if response_message.tool_calls:
        if verbose:
            print(f"\n🔧 TOOL CALLS: {len(response_message.tool_calls)} tool(s) requested")

        # Add assistant's response to conversation
        messages.append(response_message)

        # Execute each tool call
        for i, tool_call in enumerate(response_message.tool_calls):
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"\n  [{i+1}] Tool: {function_name}")
                print(f"      Args: {function_args}")

            # Execute the function
            function_to_call = AVAILABLE_FUNCTIONS[function_name]
            function_result = function_to_call(**function_args)

            if verbose:
                print(f"      Result: {function_result}")

            # Add tool result to conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": str(function_result)
            })

        # Get final response with tool results
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        final_answer = final_response.choices[0].message.content

        if verbose:
            print(f"\n💬 AGENT RESPONSE:")
            print(f"   {final_answer}")
            print('='*70)

        return final_answer

    else:
        # No tools needed
        if verbose:
            print(f"\n💬 AGENT RESPONSE (No tools needed):")
            print(f"   {response_message.content}")
            print('='*70)

        return response_message.content

# ==================== STEP 5: TEST THE AGENT ====================

if __name__ == "__main__":
    # Test 1: Math calculation
    print("\n" + "="*70)
    print("TEST 1: Calculator")
    print("="*70)
    run_multi_tool_agent("What is 25 * 17?")

    # Test 2: Date/time
    print("\n" + "="*70)
    print("TEST 2: DateTime")
    print("="*70)
    run_multi_tool_agent("What's today's date?")

    # Test 3: Knowledge search
    print("\n" + "="*70)
    print("TEST 3: Knowledge Search")
    print("="*70)
    run_multi_tool_agent("Tell me about machine learning")

    # Test 4: Notification
    print("\n" + "="*70)
    print("TEST 4: Notification")
    print("="*70)
    run_multi_tool_agent("Remind me to review AI agents")

    # Test 5: Multiple tools in sequence
    print("\n" + "="*70)
    print("TEST 5: Sequential Tools")
    print("="*70)
    run_multi_tool_agent("What's today's date and what is 15% of 200?")

    # Test 6: No tool needed
    print("\n" + "="*70)
    print("TEST 6: Conversational (No Tools)")
    print("="*70)
    run_multi_tool_agent("Hello, how are you?")
```

**Run it:** `python exercise2_multi_tool_assistant.py`

### Your Task: Add Two New Tools

Add these two additional tools to your agent:

1. **Weather Tool** (simulated):
```python
def get_weather(location: str) -> str:
    """Get weather for a location (simulated)"""
    import random
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]
    temp = random.randint(50, 85)
    condition = random.choice(conditions)
    return f"Weather in {location}: {temp}°F, {condition}"
```

2. **Translation Tool** (simulated):
```python
def translate_text(text: str, target_language: str) -> str:
    """Translate text (simulated)"""
    # Simple simulation
    translations = {
        "spanish": {"hello": "hola", "goodbye": "adiós"},
        "french": {"hello": "bonjour", "goodbye": "au revoir"}
    }
    # In reality, call a translation API
    return f"[Simulated translation to {target_language}]: {text}"
```

**Test your additions with:**
- "What's the weather in Paris?"
- "Translate 'hello' to Spanish"

### ✅ Checkpoint 2

**What you learned:**
- ✓ How to register multiple tools with an agent
- ✓ Tool selection is automatic based on descriptions
- ✓ Agents can use multiple tools sequentially
- ✓ Tool execution mapping with dictionaries

**Challenge Questions:**
1. How does the LLM decide which tool to use?
2. What happens if two tools could solve the same query?
3. How would you improve tool descriptions for better selection?

---

## 🔧 Exercise 3: Conditional Workflow Agent (25-30 minutes)

**Goal:** Build an agent that makes decisions based on tool results and executes conditional workflows.

Create `exercise3_conditional_agent.py`:

```python
import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ==================== STEP 1: CONDITIONAL TOOLS ====================

def check_business_hours() -> dict:
    """
    Check if current time is within business hours (9 AM - 5 PM)

    Returns:
        Dictionary with is_open status and current hour
    """
    now = datetime.now()
    hour = now.hour

    # Business hours: 9 AM to 5 PM
    is_open = 9 <= hour < 17

    return {
        "is_open": is_open,
        "current_hour": hour,
        "message": f"Business is {'open' if is_open else 'closed'} (Current hour: {hour})"
    }

def send_email(recipient: str, subject: str, body: str) -> str:
    """
    Send an email (simulated)

    Args:
        recipient: Email recipient
        subject: Email subject
        body: Email body

    Returns:
        Confirmation message
    """
    return f"✓ Email sent to {recipient}\n   Subject: {subject}\n   Body: {body[:50]}..."

def create_ticket(title: str, description: str, priority: str = "normal") -> str:
    """
    Create a support ticket (simulated)

    Args:
        title: Ticket title
        description: Ticket description
        priority: Priority level (low, normal, high)

    Returns:
        Ticket confirmation
    """
    import random
    ticket_id = random.randint(1000, 9999)
    return f"✓ Ticket created: #{ticket_id}\n   Title: {title}\n   Priority: {priority}"

# ==================== STEP 2: DEFINE CONDITIONAL TOOLS ====================

CONDITIONAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_business_hours",
            "description": "Check if the business is currently open based on business hours (9 AM - 5 PM)",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient. Only use during business hours for urgent matters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"}
                },
                "required": ["recipient", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a support ticket. Use this for non-urgent matters or outside business hours.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Ticket title"},
                    "description": {"type": "string", "description": "Detailed description"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "description": "Priority level"
                    }
                },
                "required": ["title", "description"]
            }
        }
    }
]

CONDITIONAL_FUNCTIONS = {
    "check_business_hours": check_business_hours,
    "send_email": send_email,
    "create_ticket": create_ticket
}

# ==================== STEP 3: BUILD CONDITIONAL AGENT ====================

def conditional_agent(user_message):
    """
    Agent that uses conditional logic based on business hours
    """
    print(f"\n{'='*70}")
    print(f"USER: {user_message}")
    print('='*70)

    # System prompt with conditional logic instructions
    system_prompt = """You are a helpful assistant that handles customer requests.

IMPORTANT RULES:
1. First, check if it's business hours using check_business_hours
2. If business is OPEN and the matter is urgent: use send_email
3. If business is CLOSED or matter is non-urgent: use create_ticket
4. Always explain your decision to the user

Follow this decision tree strictly."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    iteration = 0
    max_iterations = 5

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Step {iteration} ---")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=CONDITIONAL_TOOLS,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        # Check if done
        if not response_message.tool_calls:
            print(f"\n💬 AGENT: {response_message.content}")
            print('='*70)
            return response_message.content

        # Process tool calls
        print(f"🔧 Tool calls: {len(response_message.tool_calls)}")
        messages.append(response_message)

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"   • {function_name}({json.dumps(function_args)})")

            # Execute
            function_to_call = CONDITIONAL_FUNCTIONS[function_name]
            result = function_to_call(**function_args)

            print(f"     Result: {result}")

            # Add to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": str(result) if not isinstance(result, dict) else json.dumps(result)
            })

    return "Max iterations reached"

# ==================== STEP 4: TEST CONDITIONAL LOGIC ====================

if __name__ == "__main__":
    # Test 1: Urgent request (will check hours first)
    print("\n" + "="*70)
    print("TEST 1: Urgent Issue")
    print("="*70)
    conditional_agent("I need urgent help with a system outage")

    # Test 2: Non-urgent request
    print("\n" + "="*70)
    print("TEST 2: Non-Urgent Issue")
    print("="*70)
    conditional_agent("I have a question about my account settings")

    # Test 3: Feature request
    print("\n" + "="*70)
    print("TEST 3: Feature Request")
    print("="*70)
    conditional_agent("Can you add dark mode to the application?")
```

**Run it:** `python exercise3_conditional_agent.py`

### Your Task: Add Inventory Check Workflow

Add a branching workflow for inventory management:

```python
def check_inventory(product_id: str) -> dict:
    """Check product inventory"""
    inventory = {
        "PROD001": {"name": "Laptop", "in_stock": True, "quantity": 15},
        "PROD002": {"name": "Mouse", "in_stock": True, "quantity": 50},
        "PROD003": {"name": "Monitor", "in_stock": False, "quantity": 0}
    }
    return inventory.get(product_id, {"error": "Product not found"})

def create_order(product_id: str, quantity: int) -> dict:
    """Create order for in-stock items"""
    import random
    order_id = f"ORD{random.randint(10000, 99999)}"
    return {"order_id": order_id, "product_id": product_id, "status": "confirmed"}

def create_backorder(product_id: str, quantity: int) -> dict:
    """Create backorder for out-of-stock items"""
    import random
    backorder_id = f"BACK{random.randint(10000, 99999)}"
    return {"backorder_id": backorder_id, "product_id": product_id, "status": "pending", "eta": "14 days"}
```

**Workflow:**
1. Check inventory
2. IF in stock → create_order
3. IF out of stock → create_backorder

**Test with:** "I want to order 2 units of PROD001" and "I want to order 1 unit of PROD003"

### ✅ Checkpoint 3

**What you learned:**
- ✓ Conditional workflows with if/then logic
- ✓ Sequential tool calling where outputs inform next steps
- ✓ System prompts guide agent decision-making
- ✓ Branching paths based on tool results

---

## 🔧 Exercise 4: Resilient Agent with Error Handling (20-25 minutes)

**Goal:** Build a production-ready agent with comprehensive error handling and fallback mechanisms.

Create `exercise4_resilient_agent.py`:

```python
import os
import json
import logging
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== STEP 1: SAFE TOOL IMPLEMENTATIONS ====================

def safe_calculator(expression: str) -> Dict[str, Any]:
    """
    Calculator with comprehensive error handling

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Dictionary with result or error information
    """
    try:
        # Input validation
        if not expression or not isinstance(expression, str):
            return {
                "success": False,
                "error": "Invalid expression: must be a non-empty string",
                "error_type": "ValidationError"
            }

        # Security check
        dangerous_keywords = ['import', 'exec', 'eval', '__', 'open', 'file']
        if any(keyword in expression.lower() for keyword in dangerous_keywords):
            return {
                "success": False,
                "error": "Expression contains forbidden operations",
                "error_type": "SecurityError"
            }

        # Attempt calculation
        result = eval(expression)

        logger.info(f"Calculation successful: {expression} = {result}")
        return {
            "success": True,
            "result": result,
            "expression": expression
        }

    except ZeroDivisionError:
        logger.error(f"Division by zero: {expression}")
        return {
            "success": False,
            "error": "Cannot divide by zero",
            "error_type": "ZeroDivisionError",
            "expression": expression
        }

    except SyntaxError as e:
        logger.error(f"Syntax error: {expression}")
        return {
            "success": False,
            "error": f"Invalid mathematical syntax: {str(e)}",
            "error_type": "SyntaxError",
            "expression": expression
        }

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            "success": False,
            "error": f"Calculation failed: {str(e)}",
            "error_type": type(e).__name__,
            "expression": expression
        }

def risky_api_call(endpoint: str) -> Dict[str, Any]:
    """
    Simulated API call that might fail (30% failure rate)

    Args:
        endpoint: API endpoint to call

    Returns:
        Dictionary with result or error
    """
    import random

    if random.random() < 0.3:  # 30% failure rate
        return {
            "success": False,
            "error": "API connection timeout",
            "error_type": "TimeoutError"
        }

    return {
        "success": True,
        "data": f"Data from {endpoint}",
        "endpoint": endpoint
    }

def primary_search(query: str) -> Dict[str, Any]:
    """Primary search (may fail)"""
    import random

    if random.random() < 0.4:  # 40% failure rate
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
    """Backup search (more reliable)"""
    import random

    if random.random() < 0.1:  # 10% failure rate
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

# ==================== STEP 2: DEFINE RESILIENT TOOLS ====================

RESILIENT_TOOLS = [
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
            "name": "risky_api_call",
            "description": "Makes an API call that might fail occasionally",
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "API endpoint"}
                },
                "required": ["endpoint"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "primary_search",
            "description": "Primary search service (fast but may be unavailable)",
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
            "description": "Backup search service (use if primary fails)",
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

RESILIENT_FUNCTIONS = {
    "safe_calculator": safe_calculator,
    "risky_api_call": risky_api_call,
    "primary_search": primary_search,
    "backup_search": backup_search
}

# ==================== STEP 3: BUILD RESILIENT AGENT ====================

def resilient_agent(user_message, max_iterations=5):
    """
    Agent with comprehensive error handling
    """
    print(f"\n{'='*70}")
    print(f"USER: {user_message}")
    print('='*70)

    system_prompt = """You are a resilient assistant with error handling capabilities.

IMPORTANT RULES:
1. If a tool fails, check the error in the result
2. For search: if primary_search fails, try backup_search
3. Always inform the user about errors in a friendly way
4. Continue trying alternatives when possible

Handle errors gracefully and inform users of any issues."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        logger.info(f"Iteration {iteration}/{max_iterations}")

        try:
            # LLM API call
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=RESILIENT_TOOLS,
                tool_choice="auto",
                timeout=30
            )

            response_message = response.choices[0].message

            # Check if done
            if not response_message.tool_calls:
                final_answer = response_message.content
                print(f"\n💬 AGENT: {final_answer}")
                print('='*70)
                return final_answer

            # Process tool calls
            print(f"\n🔧 Tool calls: {len(response_message.tool_calls)}")
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                try:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"   → {function_name}({json.dumps(function_args)})")

                    # Execute tool
                    if function_name not in RESILIENT_FUNCTIONS:
                        result = {"success": False, "error": f"Unknown tool: {function_name}"}
                    else:
                        function_to_call = RESILIENT_FUNCTIONS[function_name]
                        result = function_to_call(**function_args)

                    # Log result
                    if isinstance(result, dict) and not result.get("success", True):
                        logger.warning(f"Tool {function_name} failed: {result.get('error')}")
                        print(f"     ⚠️  Error: {result.get('error')}")
                    else:
                        print(f"     ✓ Success: {result}")

                    # Add result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(result) if isinstance(result, dict) else str(result)
                    })

                except Exception as e:
                    error_msg = f"Tool execution error: {str(e)}"
                    logger.error(error_msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps({"success": False, "error": error_msg})
                    })

        except Exception as e:
            error_msg = f"LLM API error: {str(e)}"
            logger.error(error_msg)
            print(f"\n❌ ERROR: {error_msg}")
            print('='*70)
            return f"I encountered an error: {error_msg}"

    logger.warning("Max iterations reached")
    return "I couldn't complete the task within the iteration limit."

# ==================== STEP 4: TEST RESILIENT AGENT ====================

if __name__ == "__main__":
    # Test 1: Valid calculation
    print("\n" + "="*70)
    print("TEST 1: Valid Calculation")
    print("="*70)
    resilient_agent("Calculate 50 * 20")

    # Test 2: Error handling (division by zero)
    print("\n" + "="*70)
    print("TEST 2: Error Handling")
    print("="*70)
    resilient_agent("Calculate 10 / 0")

    # Test 3: Fallback search (run multiple times to see primary fail → backup)
    print("\n" + "="*70)
    print("TEST 3: Fallback Search (run 3 times)")
    print("="*70)
    for i in range(3):
        print(f"\n--- Run {i+1} ---")
        resilient_agent("Search for information about Python")

    # Test 4: Risky API (may fail randomly)
    print("\n" + "="*70)
    print("TEST 4: Risky API Call")
    print("="*70)
    resilient_agent("Fetch data from /api/users")
```

**Run it:** `python exercise4_resilient_agent.py`

### ✅ Checkpoint 4

**What you learned:**
- ✓ Safe tool implementations with try-except blocks
- ✓ Structured error responses with success flags
- ✓ Fallback mechanisms (primary → backup)
- ✓ User-friendly error messages
- ✓ Logging for debugging

---

## 🎯 Capstone Project: AgentHub v1.0 (30-40 minutes)

**Goal:** Build a complete multi-agent platform that combines everything you've learned.

### Project Requirements

Create `capstone_agenthub.py` with:

1. **Multiple specialized agents:**
   - MathAgent (calculator)
   - InfoAgent (search + datetime)
   - TaskAgent (notifications + tickets)

2. **Router agent** that directs queries to the right specialist

3. **Error handling** throughout

4. **Logging and monitoring**

Create `capstone_agenthub.py`:

```python
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)

# ==================== TOOLS LIBRARY ====================

def calculator(expression: str) -> Dict[str, Any]:
    """Safe calculator with error handling"""
    try:
        result = eval(expression)
        return {"success": True, "result": result}
    except ZeroDivisionError:
        return {"success": False, "error": "Cannot divide by zero"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_current_datetime(format: str = "full") -> str:
    """Get current date/time"""
    now = datetime.now()
    if format == "date":
        return now.strftime("%Y-%m-%d")
    elif format == "time":
        return now.strftime("%H:%M:%S")
    return now.strftime("%Y-%m-%d %H:%M:%S")

def search_knowledge_base(query: str) -> str:
    """Search knowledge base"""
    knowledge = {
        "python": "Python is a high-level programming language.",
        "ai agents": "AI agents are autonomous systems that use tools.",
        "machine learning": "ML is a subset of AI focused on learning from data."
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return "No information found."

def send_notification(message: str, priority: str = "normal") -> str:
    """Send notification"""
    return f"✓ Notification ({priority}): {message}"

def create_ticket(title: str, description: str) -> str:
    """Create support ticket"""
    import random
    ticket_id = random.randint(1000, 9999)
    return f"✓ Ticket #{ticket_id} created: {title}"

# ==================== BASE AGENT CLASS ====================

class BaseAgent:
    """Base agent class with common functionality"""

    def __init__(self, name: str, tools: List[Dict], functions: Dict):
        self.name = name
        self.tools = tools
        self.functions = functions
        self.logger = logging.getLogger(name)
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def run(self, user_message: str, system_prompt: str = None) -> str:
        """Run the agent with a user message"""
        self.logger.info(f"Processing: {user_message}")

        messages = [{"role": "user", "content": user_message}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        iteration = 0
        max_iterations = 5

        while iteration < max_iterations:
            iteration += 1

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto"
                )

                response_message = response.choices[0].message

                # Check if done
                if not response_message.tool_calls:
                    return response_message.content

                # Process tool calls
                messages.append(response_message)

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    self.logger.debug(f"Calling {function_name}({function_args})")

                    # Execute tool
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

# ==================== SPECIALIZED AGENTS ====================

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

        functions = {"calculator": calculator}

        super().__init__("MathAgent", tools, functions)

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
                                "enum": ["full", "date", "time"],
                                "description": "Output format"
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
            },
            {
                "type": "function",
                "function": {
                    "name": "create_ticket",
                    "description": "Create a support ticket",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Ticket title"},
                            "description": {"type": "string", "description": "Ticket description"}
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

# ==================== AGENT HUB (ROUTER) ====================

class AgentHub:
    """
    Central hub that routes queries to specialized agents
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
        """
        Determine which agent should handle the query
        """
        message_lower = user_message.lower()

        # Math keywords
        if any(word in message_lower for word in ["calculate", "compute", "math", "%", "+", "-", "*", "/"]):
            return "math"

        # Task keywords
        elif any(word in message_lower for word in ["remind", "notification", "ticket", "alert", "create ticket"]):
            return "task"

        # Info keywords (date, search, etc.)
        elif any(word in message_lower for word in ["search", "find", "what is", "tell me about", "date", "time", "today"]):
            return "info"

        # Default to info agent
        else:
            return "info"

    def process(self, user_message: str, verbose: bool = True) -> str:
        """
        Process a user message by routing to appropriate agent

        Args:
            user_message: User's query
            verbose: Print detailed output

        Returns:
            Agent's response
        """
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

# ==================== MAIN ====================

def main():
    """Run AgentHub demo"""
    hub = AgentHub()

    # Test queries
    test_queries = [
        "What is 25% of 340?",
        "Tell me about Python",
        "What's today's date?",
        "Remind me to review AI agents tomorrow",
        "Calculate 127 * 43",
        "Create a ticket for the login bug",
        "Search for information about machine learning"
    ]

    print("\n" + "="*70)
    print("🚀 AGENTHUB v1.0 - MULTI-AGENT PLATFORM")
    print("="*70)
    print(f"\nRunning {len(test_queries)} test queries...\n")

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'#'*70}")
        print(f"TEST {i}/{len(test_queries)}")
        print('#'*70)
        hub.process(query)

        # Small delay for readability
        import time
        time.sleep(0.5)

    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED")
    print("="*70)
    print("\nAgentHub v1.0 Statistics:")
    print(f"  • Agents: {len(hub.agents)}")
    print(f"  • Queries processed: {len(test_queries)}")
    print(f"  • Agents available: {', '.join(hub.agents.keys())}")
    print("="*70)

if __name__ == "__main__":
    main()
```

**Run it:** `python capstone_agenthub.py`

### Your Enhancements

Add these features to make AgentHub even better:

1. **Add a WeatherAgent** with simulated weather tool
2. **Implement query history** - track all queries and responses
3. **Add confidence scores** to routing decisions
4. **Create an agent performance dashboard** showing success rates
5. **Add user preferences** - remember user's preferred notification priority

### ✅ Capstone Complete!

**What you built:**
- ✓ Multi-agent system with specialized agents
- ✓ Intelligent routing based on query analysis
- ✓ Comprehensive error handling
- ✓ Logging and monitoring
- ✓ Modular, extensible architecture

---

## 🎓 Lab Summary

### What You Accomplished

Congratulations! You've built:

1. ✅ **Single-tool agents** with both OpenAI and Claude
2. ✅ **Multi-tool agents** with intelligent tool selection
3. ✅ **Conditional workflows** with branching logic
4. ✅ **Resilient agents** with error handling and fallbacks
5. ✅ **AgentHub v1.0** - Production multi-agent platform

### Key Concepts Mastered

- ✓ Tool/function calling with LLMs
- ✓ Tool schema definition and registration
- ✓ Agent execution loops
- ✓ Conditional workflows and branching
- ✓ Error handling and resilience
- ✓ Multi-agent coordination
- ✓ Production-ready agent design

### Skills You Can Now Apply

1. **Build autonomous agents** for real-world applications
2. **Design tool libraries** for specific domains
3. **Implement error handling** in production systems
4. **Create multi-agent architectures** for complex workflows
5. **Debug and monitor** agent systems

---

## 📚 Additional Challenges

### Challenge 1: Research Agent
Build an agent that:
1. Searches multiple sources (3+ tools)
2. Compares results
3. Synthesizes findings
4. Provides sourced answers

### Challenge 2: Customer Support Agent
Create a support agent that:
1. Classifies issues (technical, billing, general)
2. Checks business hours
3. Routes appropriately
4. Escalates when needed

### Challenge 3: Data Pipeline Agent
Build an agent that:
1. Validates input data
2. Transforms data (multiple steps)
3. Handles errors gracefully
4. Reports success/failure

---

## 🔧 Troubleshooting

### Common Issues

**Issue:** Tool not being called
```
Solution: Improve tool description
- Be specific about when to use the tool
- Include keywords from common queries
- Add examples in the description
```

**Issue:** Wrong tool selected
```
Solution: Refine tool descriptions
- Make descriptions more distinct
- Add usage guidelines
- Use system prompts to guide selection
```

**Issue:** Agent gets stuck in loops
```
Solution: Add max iterations
- Set reasonable iteration limits (5-10)
- Log iterations for debugging
- Add loop detection logic
```

---

## 🎯 Next Steps

Ready to continue your agent journey?

1. **Lab 7: Agent Memory & Planning**
   - ReAct pattern (Reasoning + Acting)
   - Agent memory systems
   - Multi-step planning
   - Task decomposition

2. **Lab 8: Advanced Multi-Agent Systems**
   - Agent coordination
   - Parallel agent execution
   - Agent frameworks (LangChain, CrewAI)
   - Production deployment

3. **Real-World Projects**
   - Build a research agent
   - Create an agentic RAG system
   - Deploy agents to production

---

## 📖 Additional Resources

### Documentation
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Claude Tool Use](https://docs.anthropic.com/claude/docs/tool-use)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)

### Further Reading
- ReAct: Reasoning and Acting in LLMs
- Agent design patterns and best practices
- Production agent deployment strategies

---

## ✅ Lab 6 Complete!

**Congratulations!** 🎉

You've successfully completed Lab 6: AI Agents & Tool Calling. You now have the skills to build sophisticated, production-ready AI agents.

**Time to celebrate your achievement!** 🚀

---

**Ready for Lab 7?** → Continue to [Lab 7: Agent Memory & Planning](../Lab7-Agent-Memory/codelab.md)
