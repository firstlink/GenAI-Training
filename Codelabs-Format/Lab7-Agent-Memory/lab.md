# 🛠️ Lab 7: Agent Memory & Planning - Hands-On Lab

> **Duration:** 100-130 minutes
> **Difficulty:** Advanced
> **Prerequisites:** Lab 6 completed, Python environment set up

---

## 📋 Lab Overview

In this hands-on lab, you'll build increasingly sophisticated agents with memory systems and planning capabilities, culminating in a **production-ready intelligent agent** with full memory and ReAct planning.

### What You'll Build

By the end of this lab, you'll have created:

1. ✅ **Memory Agent** - Agent with short-term, working, and long-term memory
2. ✅ **ReAct Agent** - Agent using Thought → Action → Observation loop
3. ✅ **Planning Agent** - Agent that plans before executing
4. ✅ **Reflective Agent** - Agent with self-reflection and error correction
5. ✅ **IntelliAgent v1.0** - Complete agent with memory + planning (Capstone)

---

## 🎯 Learning Objectives

By completing this lab, you will:
- ✓ Implement short-term memory (conversation history)
- ✓ Build working memory for task tracking
- ✓ Create long-term memory with vector databases
- ✓ Implement the ReAct framework
- ✓ Build planning agents that create strategies
- ✓ Add self-reflection for error correction
- ✓ Combine memory and planning in production agents

---

## 📂 Setup

### Step 1: Create Lab Directory

```bash
mkdir lab7_agent_memory
cd lab7_agent_memory
```

### Step 2: Install Dependencies

```bash
pip install openai anthropic python-dotenv chromadb sentence-transformers
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
import chromadb

load_dotenv()

# Test API keys
print("✓ OpenAI key loaded" if os.getenv('OPENAI_API_KEY') else "✗ OpenAI key missing")

# Test OpenAI connection
try:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=10
    )
    print("✓ OpenAI API working")
except Exception as e:
    print(f"✗ OpenAI API error: {e}")

# Test ChromaDB
try:
    chroma_client = chromadb.Client()
    print("✓ ChromaDB working")
except Exception as e:
    print(f"✗ ChromaDB error: {e}")

print("\n✅ Setup complete! Ready for exercises.")
```

Run: `python test_setup.py`

---

## 🔧 Exercise 1: Agent with Memory Systems (25-30 minutes)

**Goal:** Build an agent with short-term, working, and long-term memory.

### Part A: Short-Term Memory

Create `exercise1a_short_term_memory.py`:

```python
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ==================== SHORT-TERM MEMORY ====================

class ShortTermMemory:
    """Manages conversation history"""

    def __init__(self, max_messages: int = 20):
        self.messages: List[Dict] = []
        self.max_messages = max_messages

    def add_system_message(self, content: str):
        """Add system message at the beginning"""
        self.messages.insert(0, {"role": "system", "content": content})

    def add_user_message(self, content: str):
        """Add user message"""
        self.messages.append({"role": "user", "content": content})
        self._trim_if_needed()

    def add_assistant_message(self, content: str):
        """Add assistant message"""
        self.messages.append({"role": "assistant", "content": content})
        self._trim_if_needed()

    def _trim_if_needed(self):
        """Keep only recent messages + system message"""
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        other_msgs = [m for m in self.messages if m["role"] != "system"]

        if len(other_msgs) > self.max_messages:
            other_msgs = other_msgs[-self.max_messages:]

        self.messages = system_msgs + other_msgs

    def get_messages(self) -> List[Dict]:
        """Get all messages"""
        return self.messages

    def clear(self):
        """Clear all except system messages"""
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        self.messages = system_msgs

# ==================== CONVERSATIONAL AGENT ====================

class ConversationalAgent:
    """Agent with conversation memory"""

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.memory = ShortTermMemory(max_messages=20)
        self.memory.add_system_message(system_prompt)

    def chat(self, user_message: str) -> str:
        """Chat with memory"""
        print(f"\n{'='*60}")
        print(f"User: {user_message}")

        # Add to memory
        self.memory.add_user_message(user_message)

        # Get response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.memory.get_messages()
        )

        assistant_message = response.choices[0].message.content
        self.memory.add_assistant_message(assistant_message)

        print(f"Agent: {assistant_message}")
        print('='*60)

        return assistant_message

    def reset(self):
        """Reset conversation"""
        self.memory.clear()
        print("🔄 Conversation reset")

# ==================== TEST ====================

if __name__ == "__main__":
    agent = ConversationalAgent()

    print("\n" + "="*60)
    print("CONVERSATIONAL AGENT WITH SHORT-TERM MEMORY")
    print("="*60)

    # Conversation demonstrating memory
    agent.chat("Hi, my name is Alice")
    agent.chat("What's my name?")  # Should remember: Alice
    agent.chat("I love Python programming")
    agent.chat("What do I love?")  # Should remember: Python

    print("\n" + "="*60)
    print("✅ Exercise 1A Complete!")
    print("="*60)
```

**Run it:** `python exercise1a_short_term_memory.py`

### Part B: Working Memory

Create `exercise1b_working_memory.py`:

```python
from typing import Any, Dict, Optional
from datetime import datetime

# ==================== WORKING MEMORY ====================

class WorkingMemory:
    """Manages task context and progress"""

    def __init__(self):
        self.task_name: Optional[str] = None
        self.task_status: str = "idle"
        self.variables: Dict[str, Any] = {}
        self.steps_completed: list = []
        self.current_step: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    def start_task(self, task_name: str):
        """Start a new task"""
        self.task_name = task_name
        self.task_status = "in_progress"
        self.started_at = datetime.now()
        self.variables = {}
        self.steps_completed = []
        print(f"\n📋 Started task: {task_name}")

    def set_variable(self, key: str, value: Any):
        """Store a variable"""
        self.variables[key] = value
        print(f"   💾 Stored: {key} = {value}")

    def get_variable(self, key: str) -> Any:
        """Retrieve a variable"""
        return self.variables.get(key)

    def complete_step(self, step_name: str):
        """Mark a step as completed"""
        self.steps_completed.append({
            "step": step_name,
            "completed_at": datetime.now()
        })
        self.current_step = None
        print(f"   ✅ Completed step: {step_name}")

    def start_step(self, step_name: str):
        """Start a new step"""
        self.current_step = step_name
        print(f"   🔄 Starting step: {step_name}")

    def complete_task(self, success: bool = True):
        """Complete the task"""
        self.task_status = "completed" if success else "failed"
        self.completed_at = datetime.now()

        duration = (self.completed_at - self.started_at).total_seconds()
        print(f"\n{'✅' if success else '❌'} Task {self.task_status}: {self.task_name}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Steps completed: {len(self.steps_completed)}")

    def get_summary(self) -> Dict[str, Any]:
        """Get task summary"""
        return {
            "task_name": self.task_name,
            "status": self.task_status,
            "steps_completed": len(self.steps_completed),
            "variables": self.variables
        }

# ==================== TEST ====================

if __name__ == "__main__":
    import json

    memory = WorkingMemory()

    print("\n" + "="*60)
    print("WORKING MEMORY DEMONSTRATION")
    print("="*60)

    # Simulate a multi-step task
    memory.start_task("Calculate compound interest")

    memory.start_step("Get input values")
    memory.set_variable("principal", 1000)
    memory.set_variable("rate", 0.05)
    memory.set_variable("time", 3)
    memory.complete_step("Get input values")

    memory.start_step("Calculate final amount")
    principal = memory.get_variable("principal")
    rate = memory.get_variable("rate")
    time = memory.get_variable("time")
    amount = principal * ((1 + rate) ** time)
    memory.set_variable("final_amount", amount)
    memory.complete_step("Calculate final amount")

    memory.start_step("Calculate interest gained")
    interest = amount - principal
    memory.set_variable("interest", interest)
    memory.complete_step("Calculate interest gained")

    memory.complete_task(success=True)

    print("\n" + "="*60)
    print("Task Summary:")
    print(json.dumps(memory.get_summary(), indent=2))
    print("="*60)

    print("\n✅ Exercise 1B Complete!")
```

**Run it:** `python exercise1b_working_memory.py`

### Part C: Long-Term Memory with ChromaDB

Create `exercise1c_long_term_memory.py`:

```python
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from datetime import datetime

# ==================== LONG-TERM MEMORY ====================

class LongTermMemory:
    """Manages persistent memory with vector database"""

    def __init__(self, collection_name: str = "agent_memory"):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="./agent_memory_db")

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Agent long-term memory"}
        )

        print(f"✓ Long-term memory initialized")
        print(f"  Collection: {collection_name}")
        print(f"  Memories stored: {self.collection.count()}")

    def store_memory(
        self,
        content: str,
        memory_type: str = "general",
        metadata: Dict = None
    ) -> str:
        """Store a memory"""
        # Generate embedding
        embedding = self.embedding_model.encode(content)

        # Create unique ID
        memory_id = f"mem_{datetime.now().timestamp()}"

        # Prepare metadata
        mem_metadata = {
            "type": memory_type,
            "created_at": datetime.now().isoformat(),
            **(metadata or {})
        }

        # Store in ChromaDB
        self.collection.add(
            documents=[content],
            embeddings=[embedding.tolist()],
            ids=[memory_id],
            metadatas=[mem_metadata]
        )

        print(f"💾 Stored: {content[:50]}{'...' if len(content) > 50 else ''}")
        return memory_id

    def retrieve_memories(
        self,
        query: str,
        n_results: int = 5,
        memory_type: str = None
    ) -> List[Dict]:
        """Retrieve relevant memories"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)

        # Build filter
        where_filter = {"type": memory_type} if memory_type else None

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        memories = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                memories.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "relevance": 1 / (1 + results['distances'][0][i])
                })

        return memories

    def clear_memories(self):
        """Clear all memories"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"description": "Agent long-term memory"}
        )
        print("🗑️  Cleared all memories")

# ==================== TEST ====================

if __name__ == "__main__":
    ltm = LongTermMemory()

    print("\n" + "="*60)
    print("LONG-TERM MEMORY DEMONSTRATION")
    print("="*60)

    # Store some memories
    print("\n📝 Storing memories...")
    ltm.store_memory(
        "User's name is Alice",
        memory_type="user_fact"
    )

    ltm.store_memory(
        "User likes Python programming",
        memory_type="user_preference"
    )

    ltm.store_memory(
        "User works as a data scientist",
        memory_type="user_fact"
    )

    ltm.store_memory(
        "User prefers detailed explanations",
        memory_type="user_preference"
    )

    # Retrieve relevant memories
    print("\n🔍 Querying memories...")
    print("\nQuery: 'What does the user do?'")
    memories = ltm.retrieve_memories("What does the user do?", n_results=3)

    for i, mem in enumerate(memories):
        print(f"\n[{i+1}] Relevance: {mem['relevance']:.4f}")
        print(f"    Content: {mem['content']}")
        print(f"    Type: {mem['metadata']['type']}")

    print("\n" + "="*60)
    print("✅ Exercise 1C Complete!")
    print("="*60)
```

**Run it:** `python exercise1c_long_term_memory.py`

### ✅ Checkpoint 1

**What you learned:**
- ✓ How to implement short-term memory for conversation history
- ✓ How to use working memory for task tracking
- ✓ How to build long-term memory with vector databases
- ✓ Memory trimming and management strategies

**Self-Check Questions:**
1. What's the difference between short-term and long-term memory?
2. When should you use working memory?
3. How does semantic search work in long-term memory?

---

## 🔧 Exercise 2: ReAct Agent (25-30 minutes)

**Goal:** Build an agent using the Thought → Action → Observation loop.

Create `exercise2_react_agent.py`:

```python
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ==================== TOOLS ====================

def calculator(expression: str) -> Dict[str, Any]:
    """Calculator tool"""
    try:
        result = eval(expression)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

def search_info(query: str) -> Dict[str, Any]:
    """Simulated search tool"""
    knowledge = {
        "paris": "Paris is the capital of France with population of ~2.2 million (city proper). Founded in 3rd century BC.",
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991. Known for readability.",
        "ai": "AI (Artificial Intelligence) is the simulation of human intelligence by machines and computer systems.",
        "machine learning": "Machine learning is a subset of AI that enables systems to learn from data without explicit programming."
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return {"success": True, "result": value}

    return {"success": False, "result": "No information found"}

# ==================== TOOL DEFINITIONS ====================

REACT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_info",
            "description": "Search for information on a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

REACT_FUNCTIONS = {
    "calculator": calculator,
    "search_info": search_info
}

# ==================== REACT AGENT ====================

def react_agent(user_query: str, verbose: bool = True):
    """
    ReAct agent showing explicit reasoning
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"USER QUERY: {user_query}")
        print('='*70)

    # System prompt for ReAct
    system_prompt = """You are a helpful assistant using the ReAct (Reasoning + Acting) framework.

For each step:
1. THINK about what to do next
2. Use a tool if needed (ACTION)
3. OBSERVE the result
4. DECIDE if you have enough information

Show your reasoning clearly by thinking step-by-step before each action."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    iteration = 0
    max_iterations = 5

    while iteration < max_iterations:
        iteration += 1

        if verbose:
            print(f"\n{'─'*70}")
            print(f"ITERATION {iteration}")
            print('─'*70)

        # Get response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=REACT_TOOLS,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        # Check if done
        if not response_message.tool_calls:
            final_answer = response_message.content

            if verbose:
                print(f"\n💭 FINAL THOUGHT:")
                print(final_answer)
                print('='*70)

            return final_answer

        # Show thinking
        if verbose and response_message.content:
            print(f"\n💭 THOUGHT:")
            print(response_message.content)

        # Process tool calls (ACTIONS)
        messages.append(response_message)

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"\n🔧 ACTION: {function_name}")
                print(f"   Args: {json.dumps(arguments)}")

            # Execute tool
            function = REACT_FUNCTIONS[function_name]
            result = function(**arguments)

            if verbose:
                print(f"\n👁️  OBSERVATION:")
                print(f"   {json.dumps(result, indent=3)}")

            # Add result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(result)
            })

    return "Max iterations reached"

# ==================== TEST ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("REACT AGENT DEMONSTRATION")
    print("="*70)

    # Test 1: Simple calculation
    print("\n" + "#"*70)
    print("TEST 1: Simple Calculation")
    print("#"*70)
    react_agent("What is 25% of 840?")

    # Test 2: Information retrieval
    print("\n" + "#"*70)
    print("TEST 2: Information Retrieval")
    print("#"*70)
    react_agent("Tell me about Python programming language")

    # Test 3: Multi-step reasoning
    print("\n" + "#"*70)
    print("TEST 3: Multi-Step Reasoning")
    print("#"*70)
    react_agent("Search for Paris population, then calculate how many people per 100,000")

    print("\n" + "="*70)
    print("✅ Exercise 2 Complete!")
    print("="*70)
```

**Run it:** `python exercise2_react_agent.py`

### ✅ Checkpoint 2

**What you learned:**
- ✓ The ReAct (Reasoning + Acting) framework
- ✓ Thought → Action → Observation loop
- ✓ Explicit reasoning for transparency
- ✓ Multi-step problem solving

---

## 🔧 Exercise 3: Planning Agent (25-30 minutes)

**Goal:** Build an agent that creates a plan before executing.

Create `exercise3_planning_agent.py`:

```python
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ==================== PLANNING AGENT ====================

class PlanningAgent:
    """Agent that plans before executing"""

    def __init__(self, tools, functions):
        self.tools = tools
        self.functions = functions
        self.plan = []
        self.execution_log = []

    def create_plan(self, user_query: str) -> List[str]:
        """Create a step-by-step plan"""
        print(f"\n{'='*70}")
        print("PLANNING PHASE")
        print('='*70)

        planning_prompt = f"""Given this task, create a detailed step-by-step plan.

Task: {user_query}

Available tools:
- calculator: for mathematical calculations
- search_info: for finding information

Create a numbered plan with specific steps. Each step should be clear and actionable."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a planning assistant. Create clear, actionable plans."},
                {"role": "user", "content": planning_prompt}
            ]
        )

        plan_text = response.choices[0].message.content
        print(f"\n📋 PLAN:")
        print(plan_text)

        # Parse plan into steps
        lines = plan_text.split('\n')
        steps = [line.strip() for line in lines if line.strip() and any(c.isdigit() for c in line[:3])]

        self.plan = steps
        return steps

    def execute_plan(self) -> Dict:
        """Execute the created plan"""
        print(f"\n{'='*70}")
        print("EXECUTION PHASE")
        print('='*70)

        messages = [
            {"role": "system", "content": "Execute the plan step by step. Follow each step carefully."},
            {"role": "user", "content": f"Execute this plan:\n" + "\n".join(self.plan)}
        ]

        step_num = 0
        max_iterations = len(self.plan) + 5

        while step_num < max_iterations:
            step_num += 1
            print(f"\n{'─'*70}")
            print(f"EXECUTION STEP {step_num}")
            print('─'*70)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            if response_message.content:
                print(f"💭 {response_message.content[:150]}...")

            # Check if done
            if not response_message.tool_calls:
                final_answer = response_message.content
                print(f"\n✅ Execution complete!")
                print(f"\nFinal Answer:\n{final_answer}")
                return {
                    "success": True,
                    "answer": final_answer,
                    "plan": self.plan,
                    "execution_log": self.execution_log
                }

            # Execute tools
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                print(f"\n🔧 Tool: {function_name}")
                print(f"   Args: {json.dumps(arguments)}")

                result = self.functions[function_name](**arguments)

                print(f"   Result: {json.dumps(result)}")

                self.execution_log.append({
                    "step": step_num,
                    "tool": function_name,
                    "arguments": arguments,
                    "result": result
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(result)
                })

        return {
            "success": False,
            "error": "Max iterations reached"
        }

    def plan_and_execute(self, user_query: str) -> Dict:
        """Complete plan-then-execute workflow"""
        # Phase 1: Planning
        self.create_plan(user_query)

        # Phase 2: Execution
        result = self.execute_plan()

        return result

# ==================== TEST ====================

if __name__ == "__main__":
    from exercise2_react_agent import REACT_TOOLS, REACT_FUNCTIONS

    print("\n" + "="*70)
    print("PLANNING AGENT DEMONSTRATION")
    print("="*70)

    agent = PlanningAgent(REACT_TOOLS, REACT_FUNCTIONS)

    result = agent.plan_and_execute(
        "Find information about machine learning, then calculate how many years it's been since AI was coined in 1956 (current year 2024)"
    )

    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"Success: {result['success']}")
    print(f"Steps in plan: {len(result.get('plan', []))}")
    print(f"Tools executed: {len(result.get('execution_log', []))}")

    print("\n✅ Exercise 3 Complete!")
```

**Run it:** `python exercise3_planning_agent.py`

### ✅ Checkpoint 3

**What you learned:**
- ✓ Plan-then-execute pattern
- ✓ Creating structured plans from goals
- ✓ Executing plans systematically
- ✓ Tracking execution progress

---

## 🔧 Exercise 4: Self-Reflective Agent (20-25 minutes)

**Goal:** Build an agent that reflects on its progress and adjusts course.

Create `exercise4_reflective_agent.py`:

```python
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ==================== REFLECTIVE AGENT ====================

class ReflectiveAgent:
    """Agent with self-reflection capabilities"""

    def __init__(self, tools, functions):
        self.tools = tools
        self.functions = functions
        self.reflection_history = []
        self.action_history = []

    def reflect(self, action_history: List[Dict]) -> str:
        """Reflect on actions taken"""
        if not action_history:
            return ""

        history_text = "Actions taken so far:\n"
        for i, action in enumerate(action_history):
            history_text += f"{i+1}. {action['tool']}({json.dumps(action['args'])}) → {action['result']}\n"

        reflection_prompt = f"""{history_text}

Reflect on these actions:
1. Are we making progress toward the goal?
2. Did any action fail or produce unexpected results?
3. Should we change our approach?
4. What should we do next?

Provide a brief reflection."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a reflective assistant that evaluates progress."},
                {"role": "user", "content": reflection_prompt}
            ]
        )

        reflection = response.choices[0].message.content
        return reflection

    def run(self, user_query: str, reflection_interval: int = 3):
        """Run agent with periodic reflection"""
        print(f"\n{'='*70}")
        print(f"TASK: {user_query}")
        print('='*70)

        system_prompt = """You are a thoughtful ReAct agent that learns from mistakes.

Use the ReAct framework:
- THOUGHT: Reason about what to do
- ACTION: Use a tool
- OBSERVATION: Analyze the result
- REFLECTION: Evaluate if approach is working

If something fails, try a different approach."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        iteration = 0
        max_iterations = 10

        while iteration < max_iterations:
            iteration += 1
            print(f"\n{'─'*70}")
            print(f"ITERATION {iteration}")
            print('─'*70)

            # Periodic reflection
            if len(self.action_history) > 0 and len(self.action_history) % reflection_interval == 0:
                reflection = self.reflect(self.action_history)
                print(f"\n🤔 REFLECTION:")
                print(reflection)

                self.reflection_history.append({
                    "iteration": iteration,
                    "reflection": reflection
                })

                # Add reflection to context
                messages.append({
                    "role": "user",
                    "content": f"Reflection on progress: {reflection}"
                })

            # Get next action
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            if response_message.content:
                print(f"\n💭 {response_message.content}")

            # Check if done
            if not response_message.tool_calls:
                print(f"\n✅ COMPLETE")
                return {
                    "answer": response_message.content,
                    "action_history": self.action_history,
                    "reflections": self.reflection_history,
                    "iterations": iteration
                }

            # Execute actions
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                print(f"\n🔧 ACTION: {function_name}({json.dumps(arguments)})")

                # Execute
                try:
                    result = self.functions[function_name](**arguments)
                    success = result.get("success", True) if isinstance(result, dict) else True
                except Exception as e:
                    result = {"success": False, "error": str(e)}
                    success = False

                print(f"👁️  OBSERVATION: {json.dumps(result)}")

                if not success:
                    print("⚠️  Action failed!")

                # Record action
                self.action_history.append({
                    "tool": function_name,
                    "args": arguments,
                    "result": result,
                    "success": success
                })

                # Add to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(result)
                })

        return {
            "answer": "Max iterations reached",
            "action_history": self.action_history,
            "reflections": self.reflection_history
        }

# ==================== TEST ====================

if __name__ == "__main__":
    from exercise2_react_agent import REACT_TOOLS, REACT_FUNCTIONS

    print("\n" + "="*70)
    print("REFLECTIVE AGENT DEMONSTRATION")
    print("="*70)

    agent = ReflectiveAgent(REACT_TOOLS, REACT_FUNCTIONS)

    result = agent.run(
        "Calculate 20% of 500, search for info about AI, then calculate 10% of the first result",
        reflection_interval=2
    )

    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"Iterations: {result['iterations']}")
    print(f"Actions taken: {len(result['action_history'])}")
    print(f"Reflections made: {len(result['reflections'])}")

    print("\n✅ Exercise 4 Complete!")
```

**Run it:** `python exercise4_reflective_agent.py`

### ✅ Checkpoint 4

**What you learned:**
- ✓ Self-reflection for progress evaluation
- ✓ Error detection and course correction
- ✓ Adaptive agent behavior
- ✓ Learning from mistakes

---

## 🎯 Capstone Project: IntelliAgent v1.0 (30-40 minutes)

**Goal:** Build a complete intelligent agent combining memory systems and planning.

### Project Requirements

Create `capstone_intelliagent.py` with:

1. **Full memory system** (short-term + working + long-term)
2. **ReAct framework** with explicit reasoning
3. **Planning capabilities** for complex tasks
4. **Self-reflection** and error correction
5. **Production-ready** error handling

Create `capstone_intelliagent.py`:

```python
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

# Import memory classes from previous exercises
import sys
sys.path.append('.')
from exercise1a_short_term_memory import ShortTermMemory
from exercise1b_working_memory import WorkingMemory
from exercise1c_long_term_memory import LongTermMemory

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== TOOLS ====================

def calculator(expression: str) -> Dict[str, Any]:
    """Safe calculator"""
    try:
        result = eval(expression)
        return {"success": True, "result": result}
    except ZeroDivisionError:
        return {"success": False, "error": "Division by zero"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def search_info(query: str) -> Dict[str, Any]:
    """Knowledge base search"""
    knowledge = {
        "paris": "Paris is the capital of France, population ~2.2 million",
        "python": "Python created by Guido van Rossum in 1991",
        "ai": "AI is the simulation of human intelligence by machines",
        "machine learning": "ML is a subset of AI focused on learning from data"
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return {"success": True, "result": value}

    return {"success": False, "result": "No information found"}

# ==================== INTELLIAGENT ====================

class IntelliAgent:
    """
    Production-ready intelligent agent with:
    - Full memory system (short-term, working, long-term)
    - ReAct framework
    - Planning capabilities
    - Self-reflection
    """

    def __init__(self, name: str = "IntelliAgent"):
        self.name = name

        # Memory systems
        self.short_term = ShortTermMemory(max_messages=20)
        self.working = WorkingMemory()
        self.long_term = LongTermMemory(collection_name=f"{name.lower()}_memory")

        # Tools
        self.tools = self._setup_tools()
        self.functions = {
            "calculator": calculator,
            "search_info": search_info
        }

        # State
        self.action_history = []
        self.reflection_history = []

        # System prompt
        system_prompt = f"""You are {name}, an intelligent assistant with memory and planning capabilities.

You use the ReAct framework:
1. THOUGHT: Reason about what to do next
2. ACTION: Use tools when needed
3. OBSERVATION: Analyze results
4. REFLECTION: Periodically evaluate progress

You have access to:
- Short-term memory (conversation history)
- Working memory (task variables)
- Long-term memory (persistent facts)

Plan complex tasks before executing. Reflect on your progress. Learn from mistakes."""

        self.short_term.add_system_message(system_prompt)

        logger.info(f"{name} initialized with full capabilities")

    def _setup_tools(self):
        """Setup tool definitions"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform mathematical calculations",
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
                    "name": "search_info",
                    "description": "Search for information",
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

    def remember(self, fact: str, memory_type: str = "user_fact"):
        """Store in long-term memory"""
        self.long_term.store_memory(fact, memory_type=memory_type)
        logger.info(f"Stored memory: {fact}")

    def reflect(self) -> str:
        """Reflect on recent actions"""
        if not self.action_history:
            return ""

        recent_actions = self.action_history[-5:]  # Last 5 actions
        history_text = "Recent actions:\n"
        for i, action in enumerate(recent_actions):
            history_text += f"{i+1}. {action['tool']} → {action.get('success', 'unknown')}\n"

        reflection_prompt = f"""{history_text}

Quick reflection: Are we making progress? Any issues?"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": reflection_prompt}]
        )

        return response.choices[0].message.content

    def execute_task(self, user_message: str, use_long_term: bool = True, max_iterations: int = 10):
        """
        Execute a task with full capabilities
        """
        print(f"\n{'='*70}")
        print(f"{self.name}: EXECUTING TASK")
        print('='*70)
        print(f"Task: {user_message}\n")

        # Start task in working memory
        self.working.start_task(user_message)

        # Retrieve relevant long-term memories
        if use_long_term and self.long_term.collection.count() > 0:
            memories = self.long_term.retrieve_memories(user_message, n_results=3)
            if memories and memories[0]['relevance'] > 0.5:
                print("🧠 Retrieved memories:")
                memory_context = "Relevant memories:\n"
                for mem in memories[:3]:
                    if mem['relevance'] > 0.5:
                        print(f"   • {mem['content']}")
                        memory_context += f"- {mem['content']}\n"

                # Add to short-term memory
                self.short_term.add_user_message(f"{memory_context}\nTask: {user_message}")
            else:
                self.short_term.add_user_message(user_message)
        else:
            self.short_term.add_user_message(user_message)

        # Main execution loop
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            print(f"\n{'─'*60}")
            print(f"Iteration {iteration}")
            print('─'*60)

            # Reflection every 3 iterations
            if iteration > 1 and iteration % 3 == 0:
                reflection = self.reflect()
                print(f"\n🤔 Reflection: {reflection[:100]}...")
                self.reflection_history.append(reflection)

            # Get response
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=self.short_term.get_messages(),
                    tools=self.tools,
                    tool_choice="auto"
                )

                response_message = response.choices[0].message

                # Check if done
                if not response_message.tool_calls:
                    final_answer = response_message.content
                    self.short_term.add_assistant_message(final_answer)
                    self.working.complete_task(success=True)

                    print(f"\n✅ COMPLETE")
                    print(f"\n{self.name}: {final_answer}")
                    print('='*70)

                    return {
                        "success": True,
                        "answer": final_answer,
                        "iterations": iteration,
                        "actions": len(self.action_history),
                        "reflections": len(self.reflection_history)
                    }

                # Show reasoning
                if response_message.content:
                    print(f"💭 {response_message.content[:100]}...")

                # Execute tools
                self.short_term.messages.append(response_message)

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    print(f"\n🔧 {function_name}({json.dumps(arguments)})")

                    # Execute
                    result = self.functions[function_name](**arguments)
                    success = result.get("success", True) if isinstance(result, dict) else True

                    print(f"   Result: {json.dumps(result)}")

                    # Record action
                    self.action_history.append({
                        "iteration": iteration,
                        "tool": function_name,
                        "args": arguments,
                        "result": result,
                        "success": success
                    })

                    # Store in working memory if result has data
                    if success and isinstance(result, dict) and "result" in result:
                        self.working.set_variable(f"{function_name}_result", result["result"])

                    # Add to conversation
                    self.short_term.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(result)
                    })

            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {str(e)}")
                self.working.complete_task(success=False)
                return {
                    "success": False,
                    "error": str(e),
                    "iterations": iteration
                }

        # Max iterations reached
        self.working.complete_task(success=False)
        return {
            "success": False,
            "error": "Max iterations reached",
            "iterations": iteration
        }

# ==================== MAIN ====================

def main():
    """Demonstrate IntelliAgent capabilities"""

    print("\n" + "="*70)
    print("🧠 INTELLIAGENT v1.0 - INTELLIGENT AGENT SYSTEM")
    print("="*70)

    agent = IntelliAgent(name="IntelliAgent")

    # Store some long-term memories
    print("\n📝 Storing long-term memories...")
    agent.remember("User's name is Bob", memory_type="user_fact")
    agent.remember("User loves data science", memory_type="user_preference")

    # Test tasks
    test_tasks = [
        "Calculate 15% of 600",
        "What is Python and when was it created?",
        "Calculate 20% of 500, then search for info about machine learning",
    ]

    for i, task in enumerate(test_tasks, 1):
        print(f"\n{'#'*70}")
        print(f"TASK {i}/{len(test_tasks)}")
        print('#'*70)

        result = agent.execute_task(task)

        print(f"\n📊 Task Summary:")
        print(f"   Success: {result['success']}")
        print(f"   Iterations: {result.get('iterations', 0)}")
        print(f"   Actions: {result.get('actions', 0)}")
        print(f"   Reflections: {result.get('reflections', 0)}")

    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*70)
    print(f"\nIntelliAgent v1.0 Statistics:")
    print(f"  • Total actions: {len(agent.action_history)}")
    print(f"  • Reflections: {len(agent.reflection_history)}")
    print(f"  • Long-term memories: {agent.long_term.collection.count()}")
    print("="*70)

if __name__ == "__main__":
    main()
```

**Run it:** `python capstone_intelliagent.py`

### Your Enhancements

Add these features to make IntelliAgent even more powerful:

1. **Dynamic replanning** - Replan when actions fail
2. **Memory consolidation** - Merge similar memories
3. **Conversation summarization** - Summarize and store old conversations
4. **Learning from mistakes** - Store failed approaches to avoid repeating
5. **Multi-step planning** - Create hierarchical plans for complex tasks

### ✅ Capstone Complete!

**What you built:**
- ✓ Full memory system (short, working, long-term)
- ✓ ReAct framework with explicit reasoning
- ✓ Self-reflection and adaptation
- ✓ Production error handling
- ✓ Comprehensive logging

---

## 🎓 Lab Summary

### What You Accomplished

Congratulations! You've built:

1. ✅ **Memory systems** - Short-term, working, and long-term memory
2. ✅ **ReAct agents** - Transparent reasoning with Thought → Action → Observation
3. ✅ **Planning agents** - Create strategies before executing
4. ✅ **Reflective agents** - Self-evaluate and adjust course
5. ✅ **IntelliAgent v1.0** - Complete intelligent agent system

### Key Concepts Mastered

**Memory:**
- ✓ Short-term memory for conversation history
- ✓ Working memory for task state
- ✓ Long-term memory with vector databases
- ✓ Memory retrieval with semantic search

**Planning & Reasoning:**
- ✓ ReAct framework (Reasoning + Acting)
- ✓ Plan-then-execute pattern
- ✓ Self-reflection and error correction
- ✓ Dynamic replanning

**Production Skills:**
- ✓ Error handling in agents
- ✓ Logging and monitoring
- ✓ Modular agent architecture
- ✓ Memory management strategies

---

## 📚 Additional Challenges

### Challenge 1: Hierarchical Planning
Build an agent that:
1. Breaks complex tasks into subtasks
2. Creates plans for each subtask
3. Executes in dependency order
4. Tracks progress hierarchically

### Challenge 2: Memory Decay
Implement memory decay where:
1. Old memories become less relevant over time
2. Relevance score decreases with age
3. Very old memories are archived or deleted
4. Important memories are preserved

### Challenge 3: Multi-Agent Collaboration
Create multiple specialized agents that:
1. Each have their own memory and expertise
2. Communicate and share information
3. Delegate tasks to appropriate agent
4. Combine results from multiple agents

---

## 🔧 Troubleshooting

### Common Issues

**Issue:** Long-term memory not persisting
```
Solution: Check ChromaDB path
- Ensure './agent_memory_db' directory is created
- Check file permissions
- Verify ChromaDB is properly installed
```

**Issue:** Agent gets stuck in loops
```
Solution: Improve reflection
- Lower reflection_interval to detect loops faster
- Add loop detection logic
- Set reasonable max_iterations
```

**Issue:** Memory retrieval returns irrelevant results
```
Solution: Adjust relevance threshold
- Increase threshold (e.g., 0.7 instead of 0.5)
- Improve memory descriptions
- Use metadata filtering
```

---

## 🎯 Next Steps

Ready for more advanced topics?

1. **Lab 8: Advanced Multi-Agent Systems**
   - Research agent architecture
   - Agentic RAG systems
   - Agent frameworks (LangChain, CrewAI)
   - Production deployment

2. **Real-World Projects**
   - Build a research agent with memory
   - Create a personal assistant
   - Deploy agents to production
   - Integrate with real APIs

---

## 📖 Additional Resources

### Documentation
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Memory](https://python.langchain.com/docs/modules/memory/)

### Further Reading
- ReAct paper: "Synergizing Reasoning and Acting in Language Models"
- Agent architecture patterns
- Memory systems in AI

---

## ✅ Lab 7 Complete!

**Congratulations!** 🎉

You've successfully completed Lab 7: Agent Memory & Planning. You now have the skills to build sophisticated agents with memory, planning, and self-reflection capabilities.

**Time to celebrate your achievement!** 🚀

---

**Ready for Lab 8?** → Continue to [Lab 8: Advanced Multi-Agent Systems](../Lab8-Advanced-Agents/codelab.md)
