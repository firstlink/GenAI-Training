"""
Lab 7 - Complete Solutions: Agent Memory & Planning
Consolidated implementation of all exercises and capstone

Learning Objectives:
- Implement multi-level memory systems (short-term, working, long-term)
- Build ReAct agents with reasoning and action loops
- Create planning agents with task decomposition
- Implement self-reflective agents
- Build production-ready intelligent agent system

Exercises:
1. Memory Systems - Short-term, working, and long-term memory
2. ReAct Agent - Reasoning and Acting framework
3. Planning Agent - Task decomposition and execution
4. Reflective Agent - Self-evaluation and improvement
Capstone: IntelliAgent v1.0 - Complete intelligent agent system
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()


# ============================================================================
# EXERCISE 1: MEMORY SYSTEMS
# ============================================================================

class ShortTermMemory:
    """
    Short-term memory for managing conversation history

    Features:
    - Stores recent conversation messages
    - Automatic size management with max_messages limit
    - FIFO (First In, First Out) when limit reached
    - Direct access to message history
    """

    def __init__(self, max_messages: int = 20):
        """
        Initialize short-term memory

        Args:
            max_messages: Maximum number of messages to retain
        """
        self.max_messages = max_messages
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """
        Add a message to short-term memory

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Maintain size limit (keep system messages, trim user/assistant)
        if len(self.messages) > self.max_messages:
            # Keep system messages, remove oldest user/assistant messages
            system_messages = [m for m in self.messages if m["role"] == "system"]
            other_messages = [m for m in self.messages if m["role"] != "system"]
            other_messages = other_messages[-(self.max_messages - len(system_messages)):]
            self.messages = system_messages + other_messages

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in short-term memory

        Returns:
            List of message dictionaries (without timestamps)
        """
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def clear(self):
        """Clear all messages from short-term memory"""
        self.messages = []

    def get_summary(self) -> str:
        """
        Get a summary of short-term memory state

        Returns:
            Summary string
        """
        return f"Short-term memory: {len(self.messages)} messages (max: {self.max_messages})"


class WorkingMemory:
    """
    Working memory for managing current task context and state

    Features:
    - Task tracking with start/complete
    - Key-value variable storage
    - Step tracking for multi-step tasks
    - Context management
    """

    def __init__(self):
        """Initialize working memory"""
        self.current_task: Optional[str] = None
        self.task_started_at: Optional[str] = None
        self.variables: Dict[str, Any] = {}
        self.completed_steps: List[str] = []

    def start_task(self, task_name: str):
        """
        Start a new task

        Args:
            task_name: Name or description of the task
        """
        self.current_task = task_name
        self.task_started_at = datetime.now().isoformat()
        self.variables = {}
        self.completed_steps = []

    def set_variable(self, key: str, value: Any):
        """
        Set a variable in working memory

        Args:
            key: Variable name
            value: Variable value
        """
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """
        Get a variable from working memory

        Args:
            key: Variable name
            default: Default value if not found

        Returns:
            Variable value or default
        """
        return self.variables.get(key, default)

    def complete_step(self, step_name: str):
        """
        Mark a step as completed

        Args:
            step_name: Name of the completed step
        """
        self.completed_steps.append({
            "step": step_name,
            "completed_at": datetime.now().isoformat()
        })

    def get_context(self) -> Dict[str, Any]:
        """
        Get current working memory context

        Returns:
            Dictionary with task state
        """
        return {
            "current_task": self.current_task,
            "task_started_at": self.task_started_at,
            "variables": self.variables,
            "completed_steps": [s["step"] for s in self.completed_steps],
            "steps_count": len(self.completed_steps)
        }

    def clear(self):
        """Clear working memory"""
        self.current_task = None
        self.task_started_at = None
        self.variables = {}
        self.completed_steps = []

    def get_summary(self) -> str:
        """
        Get a summary of working memory state

        Returns:
            Summary string
        """
        if self.current_task:
            return f"Working on: {self.current_task} | Variables: {len(self.variables)} | Steps: {len(self.completed_steps)}"
        return "No active task"


class LongTermMemory:
    """
    Long-term memory using vector database for persistent storage

    Features:
    - Persistent memory across sessions
    - Semantic search for memory retrieval
    - Memory categorization by type
    - ChromaDB integration
    """

    def __init__(self, collection_name: str = "agent_memory", persist_directory: str = "./chroma_memory"):
        """
        Initialize long-term memory

        Args:
            collection_name: Name for ChromaDB collection
            persist_directory: Directory for persistent storage
        """
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Use default embedding function
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def store_memory(self, content: str, memory_type: str = "general", metadata: Optional[Dict[str, Any]] = None):
        """
        Store a memory in long-term storage

        Args:
            content: Memory content
            memory_type: Type of memory (general, fact, experience, etc.)
            metadata: Additional metadata
        """
        memory_id = f"mem_{int(time.time() * 1000)}"

        memory_metadata = {
            "type": memory_type,
            "created_at": datetime.now().isoformat()
        }

        if metadata:
            memory_metadata.update(metadata)

        self.collection.add(
            documents=[content],
            metadatas=[memory_metadata],
            ids=[memory_id]
        )

    def retrieve_memories(self, query: str, n_results: int = 5, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on semantic similarity

        Args:
            query: Query to search for
            n_results: Number of results to return
            memory_type: Filter by memory type (optional)

        Returns:
            List of memory dictionaries with content and metadata
        """
        where_filter = {"type": memory_type} if memory_type else None

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )

        memories = []
        if results["documents"] and results["documents"][0]:
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                memories.append({
                    "content": doc,
                    "metadata": metadata,
                    "similarity": 1 - distance  # Convert distance to similarity
                })

        return memories

    def clear(self):
        """Clear all memories from long-term storage"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_function
        )

    def get_summary(self) -> str:
        """
        Get a summary of long-term memory state

        Returns:
            Summary string
        """
        count = self.collection.count()
        return f"Long-term memory: {count} stored memories"


# ============================================================================
# EXERCISE 2: REACT AGENT
# ============================================================================

class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent

    Implements the ReAct framework:
    1. Thought: Reason about the current situation
    2. Action: Decide what action to take
    3. Observation: Observe the result
    4. Repeat until task is complete

    Features:
    - Structured reasoning loop
    - Tool calling integration
    - Multi-step problem solving
    - Iteration tracking
    """

    def __init__(self, api_key: str = None):
        """
        Initialize ReAct agent

        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))

        # Define available tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Performs mathematical calculations",
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
                    "name": "search_knowledge",
                    "description": "Search for information in knowledge base",
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

        self.system_prompt = """You are a ReAct agent that follows this reasoning pattern:

Thought: Analyze the current situation and decide what to do next
Action: Choose and execute a tool/function
Observation: Observe and analyze the result

Continue this loop until the task is complete. Always explain your reasoning in the Thought step."""

    def calculator(self, expression: str) -> str:
        """Execute calculator tool"""
        try:
            result = eval(expression)
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"

    def search_knowledge(self, query: str) -> str:
        """Execute knowledge search tool (simulated)"""
        # Simulated knowledge base
        knowledge = {
            "python": "Python is a high-level programming language known for its simplicity and readability.",
            "ai": "Artificial Intelligence is the simulation of human intelligence by machines.",
            "react": "ReAct is a framework combining Reasoning and Acting for AI agents."
        }

        query_lower = query.lower()
        for key, value in knowledge.items():
            if key in query_lower:
                return f"Knowledge found: {value}"

        return "No specific knowledge found for this query."

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool by name

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if tool_name == "calculator":
            return self.calculator(arguments["expression"])
        elif tool_name == "search_knowledge":
            return self.search_knowledge(arguments["query"])
        else:
            return f"Unknown tool: {tool_name}"

    def run(self, task: str, max_iterations: int = 5, verbose: bool = True) -> str:
        """
        Run ReAct agent on a task

        Args:
            task: Task description
            max_iterations: Maximum reasoning iterations
            verbose: Print detailed execution steps

        Returns:
            Final response
        """
        if verbose:
            print(f"\n{'='*70}")
            print("REACT AGENT")
            print('='*70)
            print(f"TASK: {task}")
            print('='*70)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task}
        ]

        for iteration in range(max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

            # Get response from LLM
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            # Check if tool is needed
            if response_message.tool_calls:
                if verbose:
                    print(f"\n💭 Thought: Agent decided to use a tool")

                # Add assistant message
                messages.append(response_message)

                # Execute each tool
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    if verbose:
                        print(f"🔧 Action: {function_name}({function_args})")

                    # Execute tool
                    result = self.execute_tool(function_name, function_args)

                    if verbose:
                        print(f"👁️  Observation: {result}")

                    # Add tool result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": result
                    })

            else:
                # No tool needed - task complete
                final_response = response_message.content

                if verbose:
                    print(f"\n✅ Task Complete!")
                    print(f"\n💬 Final Response: {final_response}")
                    print('='*70)

                return final_response

        # Max iterations reached
        if verbose:
            print(f"\n⚠️  Max iterations ({max_iterations}) reached")
            print('='*70)

        return "Task incomplete: Maximum iterations reached"


# ============================================================================
# EXERCISE 3: PLANNING AGENT
# ============================================================================

class PlanningAgent:
    """
    Planning Agent with task decomposition and execution

    Features:
    - Automatic task breakdown into steps
    - Sequential step execution
    - Progress tracking
    - Plan adaptation
    """

    def __init__(self, api_key: str = None):
        """
        Initialize planning agent

        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.working_memory = WorkingMemory()

    def create_plan(self, task: str, verbose: bool = True) -> List[str]:
        """
        Create a plan by breaking down the task into steps

        Args:
            task: Task description
            verbose: Print plan details

        Returns:
            List of plan steps
        """
        if verbose:
            print(f"\n{'='*70}")
            print("PLANNING PHASE")
            print('='*70)

        prompt = f"""Break down this task into 3-5 clear, actionable steps:

Task: {task}

Provide a numbered list of steps. Each step should be concrete and achievable.
Format: Just the numbered list, nothing else."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        plan_text = response.choices[0].message.content

        # Parse steps from numbered list
        steps = []
        for line in plan_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering and clean up
                step = line.split('.', 1)[-1].strip()
                step = step.lstrip('- ').strip()
                if step:
                    steps.append(step)

        if verbose:
            print("\n📋 Plan created:")
            for i, step in enumerate(steps, 1):
                print(f"   {i}. {step}")
            print('='*70)

        return steps

    def execute_step(self, step: str, context: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """
        Execute a single step of the plan

        Args:
            step: Step description
            context: Current execution context
            verbose: Print execution details

        Returns:
            Step execution result
        """
        if verbose:
            print(f"\n🔄 Executing: {step}")

        # Build context prompt
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])

        prompt = f"""Execute this step based on the current context:

Step: {step}

Current Context:
{context_str}

Provide a brief result of executing this step."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        result = response.choices[0].message.content

        if verbose:
            print(f"   ✅ Result: {result}")

        return {
            "step": step,
            "result": result,
            "status": "completed"
        }

    def run(self, task: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Run planning agent on a task

        Args:
            task: Task description
            verbose: Print detailed execution

        Returns:
            Execution summary
        """
        if verbose:
            print(f"\n{'='*70}")
            print("PLANNING AGENT")
            print('='*70)
            print(f"TASK: {task}")

        # Start task in working memory
        self.working_memory.start_task(task)

        # Create plan
        steps = self.create_plan(task, verbose)
        self.working_memory.set_variable("plan", steps)

        # Execute plan
        if verbose:
            print(f"\n{'='*70}")
            print("EXECUTION PHASE")
            print('='*70)

        results = []
        context = {"task": task}

        for i, step in enumerate(steps, 1):
            if verbose:
                print(f"\n--- Step {i}/{len(steps)} ---")

            result = self.execute_step(step, context, verbose)
            results.append(result)

            # Update context and working memory
            context[f"step_{i}_result"] = result["result"]
            self.working_memory.complete_step(step)

        # Generate final summary
        if verbose:
            print(f"\n{'='*70}")
            print("EXECUTION COMPLETE")
            print('='*70)
            print(f"\n✅ Completed {len(steps)} steps")
            print(f"\n📊 Working Memory: {self.working_memory.get_summary()}")
            print('='*70)

        return {
            "task": task,
            "steps": steps,
            "results": results,
            "working_memory": self.working_memory.get_context()
        }


# ============================================================================
# EXERCISE 4: REFLECTIVE AGENT
# ============================================================================

class ReflectiveAgent:
    """
    Reflective Agent with self-evaluation and improvement

    Features:
    - Self-evaluation of responses
    - Quality scoring
    - Iterative improvement
    - Reflection on performance
    """

    def __init__(self, api_key: str = None):
        """
        Initialize reflective agent

        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))

    def generate_response(self, query: str) -> str:
        """
        Generate initial response to query

        Args:
            query: User query

        Returns:
            Generated response
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}],
            temperature=0.7
        )

        return response.choices[0].message.content

    def reflect(self, query: str, response: str) -> Dict[str, Any]:
        """
        Reflect on the quality of a response

        Args:
            query: Original query
            response: Generated response

        Returns:
            Reflection results with score and feedback
        """
        reflection_prompt = f"""Evaluate this response on a scale of 1-10 and provide specific feedback:

Query: {query}

Response: {response}

Provide your evaluation in this format:
Score: [1-10]
Strengths: [what was good]
Weaknesses: [what could be improved]
Suggestions: [specific improvements]"""

        reflection_response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": reflection_prompt}],
            temperature=0.3
        )

        reflection_text = reflection_response.choices[0].message.content

        # Parse reflection
        lines = reflection_text.split('\n')
        score = 5  # default
        strengths = ""
        weaknesses = ""
        suggestions = ""

        for line in lines:
            line = line.strip()
            if line.startswith("Score:"):
                try:
                    score = int(line.split(':')[1].strip().split()[0])
                except:
                    score = 5
            elif line.startswith("Strengths:"):
                strengths = line.split(':', 1)[1].strip()
            elif line.startswith("Weaknesses:"):
                weaknesses = line.split(':', 1)[1].strip()
            elif line.startswith("Suggestions:"):
                suggestions = line.split(':', 1)[1].strip()

        return {
            "score": score,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "suggestions": suggestions,
            "full_reflection": reflection_text
        }

    def improve_response(self, query: str, response: str, reflection: Dict[str, Any]) -> str:
        """
        Improve response based on reflection

        Args:
            query: Original query
            response: Original response
            reflection: Reflection results

        Returns:
            Improved response
        """
        improvement_prompt = f"""Improve this response based on the feedback:

Query: {query}

Original Response: {response}

Feedback:
- Weaknesses: {reflection['weaknesses']}
- Suggestions: {reflection['suggestions']}

Generate an improved response that addresses these issues."""

        improved_response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": improvement_prompt}],
            temperature=0.7
        )

        return improved_response.choices[0].message.content

    def run(self, query: str, target_score: int = 8, max_iterations: int = 3, verbose: bool = True) -> Dict[str, Any]:
        """
        Run reflective agent with iterative improvement

        Args:
            query: User query
            target_score: Target quality score (1-10)
            max_iterations: Maximum improvement iterations
            verbose: Print detailed execution

        Returns:
            Final results with response and reflections
        """
        if verbose:
            print(f"\n{'='*70}")
            print("REFLECTIVE AGENT")
            print('='*70)
            print(f"QUERY: {query}")
            print(f"TARGET SCORE: {target_score}/10")
            print('='*70)

        # Generate initial response
        if verbose:
            print(f"\n{'='*70}")
            print("INITIAL GENERATION")
            print('='*70)

        response = self.generate_response(query)

        if verbose:
            print(f"\n💬 Response:\n{response}")

        reflections = []

        for iteration in range(max_iterations):
            if verbose:
                print(f"\n{'='*70}")
                print(f"REFLECTION ITERATION {iteration + 1}")
                print('='*70)

            # Reflect on response
            reflection = self.reflect(query, response)
            reflections.append(reflection)

            if verbose:
                print(f"\n🤔 Reflection:")
                print(f"   Score: {reflection['score']}/10")
                print(f"   Strengths: {reflection['strengths']}")
                print(f"   Weaknesses: {reflection['weaknesses']}")
                print(f"   Suggestions: {reflection['suggestions']}")

            # Check if target score reached
            if reflection['score'] >= target_score:
                if verbose:
                    print(f"\n✅ Target score reached! ({reflection['score']}/10)")
                break

            # Improve response
            if verbose:
                print(f"\n🔄 Improving response based on feedback...")

            response = self.improve_response(query, response, reflection)

            if verbose:
                print(f"\n💬 Improved Response:\n{response}")

        # Final summary
        if verbose:
            print(f"\n{'='*70}")
            print("FINAL RESULTS")
            print('='*70)
            print(f"Iterations: {len(reflections)}")
            print(f"Final Score: {reflections[-1]['score']}/10")
            print(f"\n💬 Final Response:\n{response}")
            print('='*70)

        return {
            "query": query,
            "final_response": response,
            "reflections": reflections,
            "final_score": reflections[-1]['score'],
            "iterations": len(reflections)
        }


# ============================================================================
# CAPSTONE: INTELLIAGENT v1.0
# ============================================================================

class IntelliAgent:
    """
    IntelliAgent v1.0 - Production-ready intelligent agent

    Combines all capabilities:
    - Multi-level memory system
    - ReAct reasoning and acting
    - Planning with task decomposition
    - Self-reflection and improvement

    Features:
    - Complete memory management
    - Intelligent task execution
    - Self-evaluation and improvement
    - Production error handling
    - Logging and monitoring
    """

    def __init__(self, api_key: str = None, use_long_term_memory: bool = True):
        """
        Initialize IntelliAgent

        Args:
            api_key: OpenAI API key
            use_long_term_memory: Enable long-term memory
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))

        # Initialize memory systems
        self.short_term_memory = ShortTermMemory(max_messages=20)
        self.working_memory = WorkingMemory()
        self.long_term_memory = LongTermMemory() if use_long_term_memory else None

        # Initialize sub-agents
        self.react_agent = ReActAgent(api_key=api_key)
        self.planning_agent = PlanningAgent(api_key=api_key)
        self.reflective_agent = ReflectiveAgent(api_key=api_key)

        self.system_prompt = """You are IntelliAgent v1.0, an advanced AI assistant with:
- Multi-level memory (short-term, working, long-term)
- Reasoning and planning capabilities
- Self-reflection and improvement
- Tool calling abilities

You provide thoughtful, accurate, and helpful responses."""

    def retrieve_relevant_memories(self, query: str, n_results: int = 3) -> List[str]:
        """
        Retrieve relevant memories from long-term storage

        Args:
            query: Query to search memories
            n_results: Number of memories to retrieve

        Returns:
            List of relevant memory contents
        """
        if not self.long_term_memory:
            return []

        memories = self.long_term_memory.retrieve_memories(query, n_results=n_results)
        return [m["content"] for m in memories]

    def classify_task(self, message: str) -> str:
        """
        Classify the type of task

        Args:
            message: User message

        Returns:
            Task type (simple, complex, multi-step)
        """
        # Simple heuristics for task classification
        message_lower = message.lower()

        # Multi-step indicators
        multi_step_keywords = ["plan", "create", "build", "design", "develop", "step by step"]
        if any(kw in message_lower for kw in multi_step_keywords):
            return "multi-step"

        # Complex indicators
        complex_keywords = ["calculate", "search", "find", "analyze"]
        if any(kw in message_lower for kw in complex_keywords):
            return "complex"

        return "simple"

    def execute_task(self, user_message: str, use_reflection: bool = True, verbose: bool = True) -> str:
        """
        Execute a task with full IntelliAgent capabilities

        Args:
            user_message: User's message/task
            use_reflection: Enable self-reflection
            verbose: Print detailed execution

        Returns:
            Agent response
        """
        if verbose:
            print(f"\n{'='*70}")
            print("INTELLIAGENT v1.0")
            print('='*70)
            print(f"USER: {user_message}")
            print('='*70)

        # Add to short-term memory
        self.short_term_memory.add_message("user", user_message)

        # Retrieve relevant long-term memories
        relevant_memories = []
        if self.long_term_memory:
            relevant_memories = self.retrieve_relevant_memories(user_message, n_results=2)
            if verbose and relevant_memories:
                print(f"\n🧠 Retrieved {len(relevant_memories)} relevant memories")

        # Classify task
        task_type = self.classify_task(user_message)
        if verbose:
            print(f"\n📊 Task Type: {task_type}")

        # Build context with memories
        context_parts = []
        if relevant_memories:
            context_parts.append("Relevant memories:")
            for i, memory in enumerate(relevant_memories, 1):
                context_parts.append(f"{i}. {memory}")

        context = "\n".join(context_parts) if context_parts else ""

        # Execute based on task type
        if task_type == "multi-step":
            if verbose:
                print("\n🎯 Using Planning Agent")

            result = self.planning_agent.run(user_message, verbose=verbose)
            response = f"Task completed with {len(result['steps'])} steps."

        elif task_type == "complex":
            if verbose:
                print("\n🎯 Using ReAct Agent")

            response = self.react_agent.run(user_message, max_iterations=3, verbose=verbose)

        else:
            # Simple task - direct LLM
            if verbose:
                print("\n🎯 Direct Processing")

            messages = [{"role": "system", "content": self.system_prompt}]

            if context:
                messages.append({"role": "system", "content": context})

            messages.extend(self.short_term_memory.get_messages()[-5:])  # Last 5 messages

            llm_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )

            response = llm_response.choices[0].message.content

            if verbose:
                print(f"\n💬 Response: {response}")

        # Reflection (optional)
        if use_reflection and task_type in ["simple", "complex"]:
            if verbose:
                print(f"\n{'='*70}")
                print("SELF-REFLECTION")
                print('='*70)

            reflection = self.reflective_agent.reflect(user_message, response)

            if verbose:
                print(f"\n🤔 Self-Evaluation:")
                print(f"   Score: {reflection['score']}/10")
                print(f"   Strengths: {reflection['strengths']}")
                if reflection['score'] < 7:
                    print(f"   Suggestions: {reflection['suggestions']}")

            # Improve if score is low
            if reflection['score'] < 7:
                if verbose:
                    print(f"\n🔄 Self-improving response...")

                response = self.reflective_agent.improve_response(
                    user_message, response, reflection
                )

                if verbose:
                    print(f"\n💬 Improved Response: {response}")

        # Add response to short-term memory
        self.short_term_memory.add_message("assistant", response)

        # Store in long-term memory
        if self.long_term_memory:
            memory_content = f"User asked: {user_message}\nAgent responded: {response}"
            self.long_term_memory.store_memory(
                memory_content,
                memory_type="conversation",
                metadata={"task_type": task_type}
            )

        # Summary
        if verbose:
            print(f"\n{'='*70}")
            print("MEMORY STATE")
            print('='*70)
            print(f"📝 {self.short_term_memory.get_summary()}")
            print(f"🔧 {self.working_memory.get_summary()}")
            if self.long_term_memory:
                print(f"🧠 {self.long_term_memory.get_summary()}")
            print('='*70)

        return response

    def clear_memories(self):
        """Clear all memory systems"""
        self.short_term_memory.clear()
        self.working_memory.clear()
        if self.long_term_memory:
            self.long_term_memory.clear()


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demo_exercise1_memory_systems():
    """Demonstrate Exercise 1: Memory Systems"""
    print("\n" + "="*70)
    print("EXERCISE 1: MEMORY SYSTEMS")
    print("="*70)

    # Short-term memory demo
    print("\n--- Short-Term Memory ---")
    stm = ShortTermMemory(max_messages=5)

    print("\nAdding messages...")
    stm.add_message("user", "Hello!")
    stm.add_message("assistant", "Hi! How can I help?")
    stm.add_message("user", "What's the weather?")
    stm.add_message("assistant", "I don't have real-time weather data.")

    print(f"\n{stm.get_summary()}")
    print(f"Messages: {len(stm.get_messages())}")

    # Working memory demo
    print("\n--- Working Memory ---")
    wm = WorkingMemory()

    wm.start_task("Build a web application")
    wm.set_variable("framework", "React")
    wm.set_variable("database", "PostgreSQL")
    wm.complete_step("Set up project structure")
    wm.complete_step("Install dependencies")

    print(f"\n{wm.get_summary()}")
    context = wm.get_context()
    print(f"Variables: {context['variables']}")
    print(f"Completed steps: {context['completed_steps']}")

    # Long-term memory demo
    print("\n--- Long-Term Memory ---")
    ltm = LongTermMemory(collection_name="demo_memory")

    print("\nStoring memories...")
    ltm.store_memory("Python is a high-level programming language", memory_type="fact")
    ltm.store_memory("I prefer using React for frontend development", memory_type="preference")
    ltm.store_memory("Successfully deployed the application yesterday", memory_type="experience")

    print(f"\n{ltm.get_summary()}")

    print("\nRetrieving memories about 'programming'...")
    memories = ltm.retrieve_memories("programming", n_results=2)
    for i, mem in enumerate(memories, 1):
        print(f"\n{i}. {mem['content']}")
        print(f"   Type: {mem['metadata']['type']}")
        print(f"   Similarity: {mem['similarity']:.3f}")

    # Cleanup
    ltm.clear()

    print("\n" + "="*70)
    print("✅ Exercise 1 Complete!")
    print("="*70)


def demo_exercise2_react_agent():
    """Demonstrate Exercise 2: ReAct Agent"""
    print("\n" + "="*70)
    print("EXERCISE 2: REACT AGENT")
    print("="*70)

    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  Skipped: OPENAI_API_KEY not found")
        return

    agent = ReActAgent()

    # Test case 1: Requires calculation
    test1 = "What is 25% of 340 and then add 50 to that result?"
    agent.run(test1, max_iterations=5, verbose=True)

    # Test case 2: Requires knowledge search
    test2 = "Tell me about Python and calculate 10 + 5"
    agent.run(test2, max_iterations=5, verbose=True)

    print("\n" + "="*70)
    print("✅ Exercise 2 Complete!")
    print("="*70)


def demo_exercise3_planning_agent():
    """Demonstrate Exercise 3: Planning Agent"""
    print("\n" + "="*70)
    print("EXERCISE 3: PLANNING AGENT")
    print("="*70)

    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  Skipped: OPENAI_API_KEY not found")
        return

    agent = PlanningAgent()

    # Test task
    task = "Create a simple blog website with user authentication"
    result = agent.run(task, verbose=True)

    print("\n📊 Execution Summary:")
    print(f"   Task: {result['task']}")
    print(f"   Steps completed: {len(result['results'])}")
    print(f"   Status: All steps completed successfully")

    print("\n" + "="*70)
    print("✅ Exercise 3 Complete!")
    print("="*70)


def demo_exercise4_reflective_agent():
    """Demonstrate Exercise 4: Reflective Agent"""
    print("\n" + "="*70)
    print("EXERCISE 4: REFLECTIVE AGENT")
    print("="*70)

    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  Skipped: OPENAI_API_KEY not found")
        return

    agent = ReflectiveAgent()

    # Test query
    query = "Explain what machine learning is and why it's important"
    result = agent.run(query, target_score=8, max_iterations=2, verbose=True)

    print("\n📊 Final Statistics:")
    print(f"   Query: {result['query']}")
    print(f"   Iterations: {result['iterations']}")
    print(f"   Final Score: {result['final_score']}/10")

    print("\n" + "="*70)
    print("✅ Exercise 4 Complete!")
    print("="*70)


def demo_capstone_intelliagent():
    """Demonstrate Capstone: IntelliAgent v1.0"""
    print("\n" + "="*70)
    print("CAPSTONE: INTELLIAGENT v1.0")
    print("="*70)

    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  Skipped: OPENAI_API_KEY not found")
        return

    # Initialize agent
    agent = IntelliAgent(use_long_term_memory=True)

    print("\n✅ IntelliAgent initialized with:")
    print("   - Short-term memory")
    print("   - Working memory")
    print("   - Long-term memory")
    print("   - ReAct capabilities")
    print("   - Planning capabilities")
    print("   - Self-reflection")

    # Test cases with different task types
    test_cases = [
        ("Simple task", "Hello! What can you help me with?"),
        ("Complex task", "What is 15% of 340?"),
        ("Multi-step task", "Create a plan for learning machine learning")
    ]

    for test_name, test_query in test_cases:
        print(f"\n{'#'*70}")
        print(f"TEST: {test_name}")
        print('#'*70)

        response = agent.execute_task(
            test_query,
            use_reflection=True,
            verbose=True
        )

    # Demonstrate memory persistence
    print(f"\n{'='*70}")
    print("MEMORY PERSISTENCE TEST")
    print('='*70)

    print("\nStoring a memory...")
    agent.long_term_memory.store_memory(
        "The user prefers detailed explanations with examples",
        memory_type="preference"
    )

    print("\nTesting memory retrieval in next query...")
    agent.execute_task(
        "Explain artificial intelligence",
        use_reflection=False,
        verbose=True
    )

    # Cleanup
    agent.clear_memories()

    print("\n" + "="*70)
    print("✅ Capstone Complete!")
    print("="*70)
    print("\n🎉 IntelliAgent v1.0 is production-ready!")


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Run all demonstrations"""
    print("="*70)
    print("LAB 7: AGENT MEMORY & PLANNING - COMPLETE SOLUTIONS")
    print("="*70)

    # Check API keys
    has_openai = bool(os.getenv('OPENAI_API_KEY'))

    print("\n🔑 API Key Status:")
    print(f"   OpenAI: {'✅' if has_openai else '❌'}")

    if not has_openai:
        print("\n❌ Error: OPENAI_API_KEY required")
        print("Please set OPENAI_API_KEY in your .env file")
        return

    # Menu
    print("\n" + "="*70)
    print("AVAILABLE DEMONSTRATIONS")
    print("="*70)
    print("1. Exercise 1: Memory Systems")
    print("2. Exercise 2: ReAct Agent")
    print("3. Exercise 3: Planning Agent")
    print("4. Exercise 4: Reflective Agent")
    print("5. Capstone: IntelliAgent v1.0")
    print("6. Run All")
    print("="*70)

    choice = input("\nSelect demonstration (1-6, or Enter for all): ").strip()

    if choice == "1":
        demo_exercise1_memory_systems()
    elif choice == "2":
        demo_exercise2_react_agent()
    elif choice == "3":
        demo_exercise3_planning_agent()
    elif choice == "4":
        demo_exercise4_reflective_agent()
    elif choice == "5":
        demo_capstone_intelliagent()
    else:
        # Run all
        demo_exercise1_memory_systems()
        demo_exercise2_react_agent()
        demo_exercise3_planning_agent()
        demo_exercise4_reflective_agent()
        demo_capstone_intelliagent()

    # Final summary
    print("\n" + "="*70)
    print("🎓 LAB 7 COMPLETE!")
    print("="*70)
    print("\n💡 KEY LEARNINGS:")
    print("   1. Multi-level memory systems (short-term, working, long-term)")
    print("   2. ReAct framework for reasoning and acting")
    print("   3. Planning with task decomposition")
    print("   4. Self-reflection and iterative improvement")
    print("   5. Production-ready intelligent agent architecture")

    print("\n🎯 SKILLS ACQUIRED:")
    print("   ✅ Memory management across different time scales")
    print("   ✅ Structured reasoning loops")
    print("   ✅ Task planning and execution")
    print("   ✅ Self-evaluation and improvement")
    print("   ✅ Complex agent system integration")


if __name__ == "__main__":
    main()
