# Lab 7 Solutions: Agent Memory & Planning

## 📚 Overview

Complete solutions for Lab 7 covering agent memory systems, ReAct framework, planning agents, self-reflection, and production-ready intelligent agent architecture. Build sophisticated agents with multi-level memory and advanced reasoning capabilities.

---

## 📁 Files Included

### Solutions
- **`all_exercises.py`** - Consolidated solution with all concepts (1100+ lines)

### Configuration
- **`.env.example`** - Environment configuration template
- **`README.md`** - This comprehensive guide

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install required libraries
pip install openai chromadb python-dotenv
```

### Environment Setup

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Add your API keys to `.env`:
```bash
# Required
OPENAI_API_KEY=sk-your-openai-key-here
```

### Run Solutions

```bash
# Run all exercises (interactive menu)
python all_exercises.py

# Run specific exercise
# The script will prompt you to select which demonstration to run
```

---

## 📖 What's Covered

### Exercise 1: Memory Systems

**Concepts:**
- Multi-level memory architecture
- Short-term memory (conversation history)
- Working memory (task context)
- Long-term memory (persistent storage)
- Vector database integration

**Implementation:**

```python
from all_exercises import ShortTermMemory, WorkingMemory, LongTermMemory

# Short-term memory - conversation history
stm = ShortTermMemory(max_messages=20)
stm.add_message("user", "Hello!")
stm.add_message("assistant", "Hi! How can I help?")

# Working memory - task context
wm = WorkingMemory()
wm.start_task("Build a web app")
wm.set_variable("framework", "React")
wm.complete_step("Set up project")

# Long-term memory - persistent storage
ltm = LongTermMemory()
ltm.store_memory("Python is my favorite language", memory_type="preference")
memories = ltm.retrieve_memories("programming languages", n_results=3)
```

**Memory Types:**

| Memory Type | Purpose | Storage | Lifespan |
|------------|---------|---------|----------|
| **Short-term** | Conversation history | In-memory list | Current session |
| **Working** | Task context & variables | In-memory dict | Current task |
| **Long-term** | Knowledge & experiences | ChromaDB | Persistent |

**Features:**
- Automatic size management for short-term memory
- Task tracking and progress in working memory
- Semantic search for long-term memory
- Memory categorization by type

---

### Exercise 2: ReAct Agent

**Concepts:**
- ReAct framework (Reasoning + Acting)
- Thought → Action → Observation loop
- Tool calling integration
- Multi-step reasoning

**ReAct Loop:**
```
1. THOUGHT: "I need to calculate 25% of 340"
2. ACTION: Use calculator tool with "340 * 0.25"
3. OBSERVATION: "Calculation result: 85"
4. THOUGHT: "The answer is 85"
5. RESPONSE: "25% of 340 is 85"
```

**Implementation:**
```python
from all_exercises import ReActAgent

agent = ReActAgent()

# Agent automatically follows Thought → Action → Observation loop
result = agent.run(
    "What is 25% of 340 and then add 50 to that?",
    max_iterations=5,
    verbose=True
)
```

**Available Tools:**
- `calculator` - Mathematical operations
- `search_knowledge` - Knowledge base search

**How It Works:**
1. Agent receives task
2. **Thought**: Reasons about what to do
3. **Action**: Calls appropriate tool
4. **Observation**: Analyzes tool result
5. Repeat until task complete
6. Generate final response

---

### Exercise 3: Planning Agent

**Concepts:**
- Task decomposition
- Plan-then-execute pattern
- Sequential step execution
- Progress tracking with working memory

**Planning Flow:**
```
User Task
    ↓
Create Plan (break into steps)
    ↓
┌─────────────────┐
│ Execute Step 1  │
│ Execute Step 2  │
│ Execute Step 3  │
│     ...         │
└─────────────────┘
    ↓
Return Results
```

**Implementation:**
```python
from all_exercises import PlanningAgent

agent = PlanningAgent()

# Agent automatically creates plan and executes steps
result = agent.run(
    "Create a simple blog website with user authentication",
    verbose=True
)

print(f"Completed {len(result['steps'])} steps")
print(f"Results: {result['results']}")
```

**Example Plan:**
```
Task: "Create a blog website"

Generated Plan:
1. Set up project structure and dependencies
2. Design database schema for users and posts
3. Implement user authentication system
4. Create blog post CRUD operations
5. Build frontend interface
```

**Features:**
- Automatic task breakdown into 3-5 steps
- Sequential execution with context passing
- Progress tracking in working memory
- Step result aggregation

---

### Exercise 4: Reflective Agent

**Concepts:**
- Self-evaluation of responses
- Quality scoring (1-10 scale)
- Iterative improvement
- Reflection-based learning

**Reflection Process:**
```
Generate Response
    ↓
Reflect on Quality
    ↓
Score < Target? ──No──→ Done
    │
   Yes
    ↓
Improve Response
    ↓
Reflect Again
    ↓
Score ≥ Target? ─Yes──→ Done
```

**Implementation:**
```python
from all_exercises import ReflectiveAgent

agent = ReflectiveAgent()

# Agent generates, reflects, and improves until target score reached
result = agent.run(
    "Explain what machine learning is",
    target_score=8,      # Target quality: 8/10
    max_iterations=3,
    verbose=True
)

print(f"Final score: {result['final_score']}/10")
print(f"Iterations: {result['iterations']}")
```

**Reflection Components:**
- **Score**: 1-10 quality rating
- **Strengths**: What was good
- **Weaknesses**: What could be improved
- **Suggestions**: Specific improvements

**Example Reflection:**
```
Original Response: "Machine learning is when computers learn."

Reflection:
  Score: 4/10
  Strengths: Simple and concise
  Weaknesses: Too brief, lacks detail and examples
  Suggestions: Add explanation of how it works, provide examples

Improved Response: "Machine learning is a branch of AI where
computers learn from data without being explicitly programmed.
For example, email spam filters learn to identify spam by
analyzing thousands of emails..."

New Score: 8/10 ✅
```

---

### Capstone: IntelliAgent v1.0

**Architecture:**
```
         IntelliAgent v1.0
               |
    ┌──────────┼──────────┐
    ↓          ↓          ↓
 Memory    ReAct      Planning
 Systems   Agent       Agent
    ↓          ↓          ↓
Short-term  Tools    Task Decomp
Working     Loop     Execution
Long-term   Reason   Progress
    ↓          ↓          ↓
         Reflective Agent
         Self-Evaluation
```

**Integrated Features:**

**1. Multi-Level Memory:**
- Short-term: Last 20 messages
- Working: Current task variables
- Long-term: Persistent knowledge

**2. Intelligent Task Classification:**
```python
def classify_task(message):
    if "plan" or "create" in message:
        return "multi-step"  # Use planning agent
    elif "calculate" or "search" in message:
        return "complex"     # Use ReAct agent
    else:
        return "simple"      # Direct processing
```

**3. Automatic Routing:**
- **Simple tasks** → Direct LLM with memory context
- **Complex tasks** → ReAct agent with tools
- **Multi-step tasks** → Planning agent with decomposition

**4. Self-Reflection:**
- Evaluates all responses
- Improves low-quality responses (<7/10)
- Learns from improvements

**Usage:**
```python
from all_exercises import IntelliAgent

# Initialize with all capabilities
agent = IntelliAgent(use_long_term_memory=True)

# Simple task
agent.execute_task(
    "Hello! What can you help me with?",
    use_reflection=True,
    verbose=True
)

# Complex task (uses ReAct)
agent.execute_task(
    "What is 15% of 340 and tell me about Python?",
    use_reflection=True,
    verbose=True
)

# Multi-step task (uses Planning)
agent.execute_task(
    "Create a plan for learning machine learning",
    use_reflection=True,
    verbose=True
)
```

**Features:**
- ✅ Complete memory management
- ✅ Intelligent task routing
- ✅ ReAct reasoning loops
- ✅ Planning and decomposition
- ✅ Self-evaluation and improvement
- ✅ Long-term memory persistence
- ✅ Production error handling
- ✅ Comprehensive logging

---

## 💡 Key Concepts

### Memory Architecture

**Why Multi-Level Memory?**
Different types of information require different storage:

1. **Short-term Memory**: Recent conversation context
   - Fast access
   - Limited capacity
   - Temporary storage

2. **Working Memory**: Current task state
   - Active variables
   - Progress tracking
   - Task-specific context

3. **Long-term Memory**: Persistent knowledge
   - Semantic search
   - Unlimited storage
   - Cross-session persistence

### ReAct Framework

**Traditional vs ReAct:**

**Traditional Agent:**
```
User query → LLM → Response
```

**ReAct Agent:**
```
User query
    ↓
THOUGHT: "I need to calculate this"
    ↓
ACTION: calculator(340 * 0.25)
    ↓
OBSERVATION: "Result is 85"
    ↓
THOUGHT: "Now I have the answer"
    ↓
Response
```

**Benefits:**
- Transparent reasoning
- Tool integration
- Multi-step problem solving
- Explainable decisions

### Planning Pattern

**Plan-then-Execute vs Direct Execution:**

**Direct Execution:**
- Start working immediately
- May miss steps
- Hard to track progress

**Plan-then-Execute:**
- Create complete plan first
- Execute systematically
- Track progress clearly
- Easier debugging

### Self-Reflection

**Why Self-Reflection?**
- Quality assurance
- Automatic improvement
- Learning from mistakes
- Consistent high-quality output

**Reflection Loop:**
```python
while score < target_score and iterations < max_iterations:
    # Generate response
    response = generate(query)

    # Reflect
    reflection = reflect(query, response)

    # Improve if needed
    if reflection.score < target_score:
        response = improve(response, reflection)
```

---

## 🎯 Best Practices

### Memory Management

✅ **Set Appropriate Limits:**
```python
# Too large - wastes tokens
stm = ShortTermMemory(max_messages=100)  # ❌

# Good balance
stm = ShortTermMemory(max_messages=20)   # ✅
```

✅ **Clear Memory When Appropriate:**
```python
# After completing a task
agent.working_memory.clear()

# When switching contexts
agent.short_term_memory.clear()
```

✅ **Categorize Long-term Memories:**
```python
ltm.store_memory(content, memory_type="fact")
ltm.store_memory(content, memory_type="preference")
ltm.store_memory(content, memory_type="experience")
```

### ReAct Design

✅ **Set Iteration Limits:**
```python
# Prevent infinite loops
agent.run(task, max_iterations=5)  # ✅
```

✅ **Provide Clear Tool Descriptions:**
```python
{
    "name": "calculator",
    "description": "Performs mathematical calculations. Use for arithmetic operations, percentages, or math problems."
}
```

✅ **Handle Tool Errors:**
```python
def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
```

### Planning Best Practices

✅ **Appropriate Task Breakdown:**
```python
# Too granular
steps = ["Open IDE", "Type code", "Save file", ...]  # ❌

# Good granularity
steps = ["Set up project", "Implement features", "Test"]  # ✅
```

✅ **Pass Context Between Steps:**
```python
context = {"task": task}
for step in steps:
    result = execute_step(step, context)
    context[f"step_result"] = result
```

### Reflection Guidelines

✅ **Set Realistic Target Scores:**
```python
# Too high - may never reach
target_score = 10  # ❌

# Realistic
target_score = 8   # ✅
```

✅ **Limit Iterations:**
```python
# Prevent excessive improvement attempts
max_iterations = 3  # ✅
```

---

## 🔧 Troubleshooting

### Issue: Memory Growing Too Large

**Problem:** Short-term memory consuming too many tokens.

**Solutions:**
1. Reduce `max_messages`:
   ```python
   stm = ShortTermMemory(max_messages=10)
   ```
2. Summarize old conversations before adding to memory
3. Clear memory between major topic changes

### Issue: ReAct Agent Not Using Tools

**Problem:** Agent responds directly instead of using tools.

**Solutions:**
1. Improve tool descriptions with usage examples
2. Add explicit instruction in system prompt:
   ```python
   "You have access to tools. Always consider using them when appropriate."
   ```
3. Use stronger model (gpt-4o instead of gpt-4o-mini)

### Issue: Planning Creates Too Many Steps

**Problem:** Plan has 10+ steps, too granular.

**Solutions:**
1. Specify step count in planning prompt:
   ```python
   "Break down this task into 3-5 clear, actionable steps"
   ```
2. Post-process to combine related steps
3. Use higher-level task descriptions

### Issue: Reflection Always Scores High

**Problem:** Agent always gives itself 9-10, no improvement happens.

**Solutions:**
1. Use more critical reflection prompt:
   ```python
   "Be critical and identify specific areas for improvement"
   ```
2. Use separate model instance for reflection
3. Add specific evaluation criteria

### Issue: Long-term Memory Retrieval Irrelevant

**Problem:** Retrieved memories not relevant to query.

**Solutions:**
1. Improve memory content quality:
   ```python
   # Bad
   ltm.store_memory("ok")

   # Good
   ltm.store_memory("User prefers detailed technical explanations with code examples")
   ```
2. Increase n_results to get more options
3. Add metadata filtering:
   ```python
   memories = ltm.retrieve_memories(query, memory_type="preference")
   ```

---

## 📊 Performance Tips

### Memory Optimization

**Token Usage:**
```python
# Expensive - includes all 100 messages
messages = stm.get_messages()

# Optimized - only recent context
messages = stm.get_messages()[-10:]  # Last 10 messages
```

**Long-term Memory:**
```python
# Limit retrieval to most relevant
memories = ltm.retrieve_memories(query, n_results=3)  # Not 20
```

### ReAct Optimization

**Faster Tools:**
```python
# Slow - external API call
def search_web(query):
    return requests.get(api_url).json()

# Fast - local lookup
def search_cache(query):
    return cache.get(query, default_response)
```

**Reduce Iterations:**
```python
# Development
agent.run(task, max_iterations=10)

# Production
agent.run(task, max_iterations=3)
```

### Planning Optimization

**Parallel Step Execution:**
```python
# If steps are independent, execute in parallel
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    results = executor.map(execute_step, independent_steps)
```

### Overall Performance

**Model Selection:**
- Simple tasks: `gpt-4o-mini` (fast, cheap)
- Complex reasoning: `gpt-4o` (better quality)
- Reflection: `gpt-4o-mini` (sufficient for evaluation)

**Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def generate_plan(task: str):
    # Expensive operation cached
    pass
```

---

## 🧪 Testing

### Validate Syntax

```bash
cd solutions/
python3 -m py_compile all_exercises.py
```

### Run Demonstrations

```bash
# Run all demonstrations
python all_exercises.py

# Select specific demo from menu:
# 1. Memory Systems
# 2. ReAct Agent
# 3. Planning Agent
# 4. Reflective Agent
# 5. IntelliAgent v1.0
# 6. Run All
```

### Manual Testing Checklist

- [ ] Short-term memory maintains size limit
- [ ] Working memory tracks task state
- [ ] Long-term memory persists across runs
- [ ] ReAct agent uses tools appropriately
- [ ] Planning agent breaks down tasks
- [ ] Reflective agent improves low scores
- [ ] IntelliAgent routes tasks correctly
- [ ] Memory retrieval returns relevant results

---

## 📚 Architecture Patterns

### Memory Pattern: Three-Tier Architecture

```python
class IntelligentAgent:
    def __init__(self):
        # Tier 1: Fast, temporary
        self.short_term = ShortTermMemory()

        # Tier 2: Task-specific
        self.working = WorkingMemory()

        # Tier 3: Persistent, searchable
        self.long_term = LongTermMemory()

    def process(self, message):
        # Retrieve from long-term
        context = self.long_term.retrieve(message)

        # Add to short-term
        self.short_term.add(message)

        # Track in working
        self.working.set_variable("current_query", message)

        # Process with full context
        response = self.generate(message, context)

        # Store in long-term
        self.long_term.store(f"Q: {message} A: {response}")
```

### ReAct Pattern: Observation Loop

```python
def react_loop(task, max_iterations=5):
    messages = [{"role": "user", "content": task}]

    for i in range(max_iterations):
        # Thought phase
        response = llm.generate(messages)

        # Action phase
        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = execute_tool(tool_call)

                # Observation phase
                messages.append({
                    "role": "tool",
                    "content": result
                })
        else:
            # No more actions needed
            return response.content
```

### Planning Pattern: Divide and Conquer

```python
def plan_and_execute(task):
    # Divide: Break into steps
    steps = create_plan(task)

    # Conquer: Execute each step
    results = []
    context = {"task": task}

    for step in steps:
        result = execute_step(step, context)
        results.append(result)
        context[f"step_{len(results)}"] = result

    return aggregate_results(results)
```

---

## 🎓 What You've Learned

✅ Multi-level memory architecture (short-term, working, long-term)
✅ ReAct framework for reasoning and acting
✅ Plan-then-execute pattern for complex tasks
✅ Self-reflection and iterative improvement
✅ Intelligent task routing and classification
✅ Production-ready agent design
✅ Memory persistence with vector databases
✅ Tool integration with reasoning loops
✅ Quality assurance through reflection
✅ Complex agent system architecture

---

## 🌟 Advanced Topics

### Memory Consolidation

Transfer important short-term memories to long-term:

```python
def consolidate_memories(short_term, long_term):
    """Move important conversations to long-term storage"""
    messages = short_term.get_messages()

    # Identify important exchanges
    important = [
        m for m in messages
        if is_important(m)  # Custom importance logic
    ]

    # Store in long-term
    for msg in important:
        long_term.store_memory(
            msg["content"],
            memory_type="important_conversation"
        )
```

### Hierarchical Planning

Plans that create sub-plans:

```python
def hierarchical_plan(task, max_depth=2):
    """Create nested plans for complex tasks"""
    steps = create_plan(task)

    for step in steps:
        if is_complex(step) and max_depth > 0:
            step.sub_plan = hierarchical_plan(
                step,
                max_depth - 1
            )

    return steps
```

### Multi-Agent Reflection

Agents reviewing each other's work:

```python
class PeerReviewSystem:
    def __init__(self):
        self.agent_a = ReflectiveAgent()
        self.agent_b = ReflectiveAgent()

    def peer_review(self, task):
        # Agent A generates
        response_a = self.agent_a.generate(task)

        # Agent B reviews
        review = self.agent_b.reflect(task, response_a)

        # Agent A improves
        improved = self.agent_a.improve(response_a, review)

        return improved
```

### Episodic Memory

Remembering specific experiences:

```python
class EpisodicMemory(LongTermMemory):
    def store_episode(self, event: Dict[str, Any]):
        """Store a complete episode with context"""
        episode = {
            "timestamp": datetime.now().isoformat(),
            "event": event["description"],
            "context": event["context"],
            "outcome": event["outcome"],
            "emotions": event.get("emotions", [])
        }

        self.store_memory(
            json.dumps(episode),
            memory_type="episode",
            metadata={"timestamp": episode["timestamp"]}
        )
```

---

## 📖 Additional Resources

### Documentation
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)

### Further Reading
- Memory systems in cognitive science
- Agent architecture patterns
- Multi-agent coordination
- Self-improving AI systems

### Related Papers
- ReAct: Synergizing Reasoning and Acting in Language Models
- Chain-of-Thought Prompting
- Self-Refine: Iterative Refinement with Self-Feedback

---

## 📊 Comparison: Agent Capabilities

| Capability | Basic Agent | ReAct Agent | Planning Agent | IntelliAgent |
|-----------|-------------|-------------|----------------|--------------|
| Tool Use | ❌ | ✅ | ❌ | ✅ |
| Reasoning Loop | ❌ | ✅ | ❌ | ✅ |
| Task Decomposition | ❌ | ❌ | ✅ | ✅ |
| Memory (Short) | ❌ | ❌ | ❌ | ✅ |
| Memory (Long) | ❌ | ❌ | ❌ | ✅ |
| Self-Reflection | ❌ | ❌ | ❌ | ✅ |
| Task Routing | ❌ | ❌ | ❌ | ✅ |

---

**Production Ready! 🚀**

*You now have the skills to build sophisticated AI agents with advanced memory, reasoning, and planning capabilities for real-world applications.*
