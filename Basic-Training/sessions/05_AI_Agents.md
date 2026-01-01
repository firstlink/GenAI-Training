# Session 5: AI Agents with Memory

**Duration**: 90 minutes  
**Difficulty**: Intermediate to Advanced  
**Colab Notebook**: [05_AI_Agents.ipynb](../notebooks/05_AI_Agents.ipynb)

---

## Learning Objectives

- 🎯 Understand AI agent architectures  
- 🎯 Implement the ReAct (Reasoning + Acting) pattern  
- 🎯 Build agent execution loops  
- 🎯 Create memory systems (short-term, long-term)  
- 🎯 Handle autonomous multi-step tasks  
- 🎯 Make SupportGenie truly autonomous

---

## Capstone: SupportGenie v0.5 - Autonomous Agent

**Upgrades**:
- Autonomous task planning
- Conversation memory  
- Customer profile memory
- Multi-step problem solving
- Learns from past interactions

---

## Part 1: What is an AI Agent?

### Definition

An **AI Agent** is a system that:
1. **Observes** the environment/user input
2. **Thinks** about what to do (reasoning)
3. **Acts** using tools/functions
4. **Remembers** context and history
5. **Iterates** until task complete

### Agent vs Chatbot

**Chatbot** (What we built so far):
- User asks → LLM responds
- Single turn interaction
- No persistent memory beyond conversation

**Agent**:
- User gives task → Agent plans → Uses tools → Iterates → Completes task
- Multi-turn, autonomous
- Remembers past interactions

---

## Part 2: The ReAct Pattern

### ReAct = Reasoning + Acting

The agent alternates between thinking and acting:

```
Thought: What do I need to do?
Action: [Use a tool]
Observation: [Tool result]
Thought: Based on the observation, what next?
Action: [Use another tool]
Observation: [Result]
Thought: Task complete!
Final Answer: [Response to user]
```

### Example: Booking a Restaurant

```
User: "Book a restaurant for 4 people tonight in Seattle"

Thought: I need today's date first
Action: get_current_date()
Observation: 2024-12-29

Thought: Now search for restaurants
Action: search_restaurants(location="Seattle", party_size=4, date="2024-12-29")
Observation: [List of 5 restaurants]

Thought: Check availability at top choice
Action: check_availability(restaurant="The Pink Door", date="2024-12-29", time="19:00", party_size=4)
Observation: {available: true, times: ["18:30", "19:00", "20:00"]}

Thought: Make the reservation
Action: make_reservation(restaurant="The Pink Door", date="2024-12-29", time="19:00", party_size=4)
Observation: {confirmation: "RES-12345", status: "confirmed"}

Thought: Task complete!
Final Answer: "I've booked The Pink Door for 4 people tonight at 7 PM. Confirmation: RES-12345"
```

---

## Part 3: Agent Execution Loop

```python
class SimpleAgent:
    def __init__(self, tools, max_iterations=10):
        self.client = OpenAI()
        self.tools = tools
        self.max_iterations = max_iterations
        self.conversation_history = []

    def run(self, task):
        """Execute agent loop"""
        
        self.conversation_history = [
            {"role": "user", "content": task}
        ]

        for iteration in range(self.max_iterations):
            # 1. THINK: Get agent's next action
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=self.conversation_history,
                tools=self.tools
            )

            response_message = response.choices[0].message

            # 2. CHECK: Is task complete?
            if not response_message.tool_calls:
                return response_message.content  # Done!

            # 3. ACT: Execute tool calls
            self.conversation_history.append(response_message)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                # Execute function
                result = self.execute_tool(function_name, function_args)

                # 4. OBSERVE: Add result to history
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

        return "Max iterations reached. Task incomplete."

    def execute_tool(self, name, args):
        # Execute the actual tool
        # Implementation depends on your tools
        pass
```

---

## Part 4: Memory Systems

### Types of Memory

1. **Short-term**: Current conversation
2. **Working Memory**: Current task context
3. **Long-term**: Past interactions, customer profiles

### Implementing Conversation Memory

```python
class ConversationMemory:
    def __init__(self, max_messages=20):
        self.messages = []
        self.max_messages = max_messages

    def add_message(self, role, content):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })

        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            # Keep system message + recent messages
            system_msgs = [m for m in self.messages if m['role'] == 'system']
            recent = self.messages[-(self.max_messages-len(system_msgs)):]
            self.messages = system_msgs + recent

    def get_messages(self):
        return [{"role": m['role'], "content": m['content']} 
                for m in self.messages]

    def summarize_old_context(self, llm_client):
        """Summarize old messages to save tokens"""
        if len(self.messages) > 15:
            old_messages = self.messages[:10]
            
            # Ask LLM to summarize
            summary_prompt = f"Summarize this conversation:\n{old_messages}"
            summary = llm_client.chat.completions.create(...)
            
            # Replace old messages with summary
            self.messages = [
                {"role": "system", "content": f"Previous conversation summary: {summary}"}
            ] + self.messages[10:]
```

### Customer Profile Memory

```python
class CustomerMemory:
    def __init__(self):
        self.profiles = {}  # In production: use database

    def get_profile(self, customer_id):
        """Get customer profile"""
        if customer_id not in self.profiles:
            # Fetch from database
            self.profiles[customer_id] = self.load_from_db(customer_id)
        
        return self.profiles[customer_id]

    def update_profile(self, customer_id, updates):
        """Update customer preferences"""
        profile = self.get_profile(customer_id)
        profile.update(updates)
        self.save_to_db(customer_id, profile)

    def load_from_db(self, customer_id):
        # Mock implementation
        return {
            "customer_id": customer_id,
            "name": "John Smith",
            "email": "john@email.com",
            "past_orders": ["ORD-123", "ORD-456"],
            "preferences": {
                "communication": "email",
                "language": "en"
            },
            "past_issues": [
                {"date": "2024-11-15", "issue": "Late delivery", "resolved": True}
            ]
        }
```

---

## Part 5: SupportGenie v0.5 - Autonomous Agent

```python
class SupportGenieV5:
    """
    Autonomous agent with memory and multi-step reasoning
    """

    SYSTEM_MESSAGE = """You are SupportGenie, an autonomous customer support agent.

    You can:
    - Look up order status
    - Create support tickets
    - Check customer accounts
    - Update customer preferences

    When given a task:
    1. Break it down into steps
    2. Use tools as needed
    3. Remember context from conversation
    4. Adapt based on results
    5. Confirm completion with user

    Be proactive and helpful. If you need more information, ask clearly."""

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.conversation_memory = ConversationMemory()
        self.customer_memory = CustomerMemory()
        self.setup_tools()

    def setup_tools(self):
        self.tools = [
            # get_order_status tool
            # create_support_ticket tool
            # check_account_info tool
            # etc.
        ]

    def chat(self, user_message, customer_id=None):
        """Agent chat with memory"""

        # Load customer context if available
        context = ""
        if customer_id:
            profile = self.customer_memory.get_profile(customer_id)
            context = f"\n\nCustomer Profile:\n{json.dumps(profile, indent=2)}"

        # Add to conversation memory
        full_message = user_message + context
        self.conversation_memory.add_message("user", full_message)

        # Run agent loop
        for iteration in range(10):  # Max 10 iterations
            messages = [
                {"role": "system", "content": self.SYSTEM_MESSAGE}
            ] + self.conversation_memory.get_messages()

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=self.tools
            )

            response_message = response.choices[0].message

            # No more actions needed
            if not response_message.tool_calls:
                self.conversation_memory.add_message(
                    "assistant", 
                    response_message.content
                )
                return response_message.content

            # Execute tool calls
            for tool_call in response_message.tool_calls:
                result = self.execute_tool(tool_call)
                # Add to memory...

        return "Task taking longer than expected. Would you like me to create a ticket for human review?"
```

---

## Part 6: Advanced Agent Patterns

### Pattern 1: Plan-and-Execute

```python
# First, create a plan
plan = agent.create_plan(task)
# Then execute each step
for step in plan:
    result = agent.execute_step(step)
```

### Pattern 2: Self-Critique

```python
# Agent reviews its own work
response = agent.generate_response(query)
critique = agent.critique(response)
if critique.needs_improvement:
    response = agent.improve(response, critique)
```

### Pattern 3: Human-in-the-Loop

```python
# Agent asks for human help when uncertain
if agent.confidence < 0.7:
    human_input = agent.ask_human(question)
    agent.continue_with_input(human_input)
```

---

## Exercises

1. Build an agent that can book appointments
2. Create a research agent that searches and summarizes
3. Implement agent memory with SQLite database
4. Add self-critique to improve responses

---

## Common Mistakes

❌ Infinite loops - Always set max_iterations  
❌ No memory management - Conversation grows too large  
❌ Ignoring errors - Tools can fail  
❌ Over-automation - Some tasks need human approval

---

## Key Takeaways

✅ Agents autonomously complete multi-step tasks  
✅ ReAct pattern: Reason → Act → Observe → Repeat  
✅ Memory is critical for context  
✅ Always limit iterations  
✅ Handle errors gracefully

---

**Session 5 Complete!** 🎉  
**Next**: [Session 6: Multi-Agent Systems →](06_Multi_Agent.md)
