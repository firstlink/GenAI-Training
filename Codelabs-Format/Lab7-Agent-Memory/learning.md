# 📚 Lab 7: Agent Memory & Planning - Learning Material

> **Duration:** 40 minutes
> **Level:** Advanced
> **Prerequisites:** Lab 6 (AI Agents & Tool Calling)

---

## 🎯 Learning Objectives

By the end of this module, you will understand:
- ✓ The three types of agent memory (short-term, working, long-term)
- ✓ How agents maintain conversation context
- ✓ Working memory for task tracking
- ✓ Long-term memory with vector databases
- ✓ The ReAct (Reasoning + Acting) framework
- ✓ Thought → Action → Observation loops
- ✓ Task planning and decomposition
- ✓ Self-reflection and error correction
- ✓ When and why agents should plan

---

## 📖 Table of Contents

1. [Understanding Agent Memory](#1-understanding-agent-memory)
2. [Short-Term Memory](#2-short-term-memory-conversation-history)
3. [Working Memory](#3-working-memory-task-context)
4. [Long-Term Memory](#4-long-term-memory-persistent-storage)
5. [The ReAct Framework](#5-the-react-framework)
6. [Agent Planning Strategies](#6-agent-planning-strategies)
7. [Self-Reflection & Error Correction](#7-self-reflection--error-correction)
8. [When to Use Memory & Planning](#8-when-to-use-memory--planning)

---

## 1. Understanding Agent Memory

### What is Agent Memory?

Agent memory allows AI systems to:
- **Remember** past interactions and context
- **Learn** from previous conversations
- **Maintain continuity** across sessions
- **Personalize** responses based on history
- **Improve** decision-making with experience

### The Three Types of Memory

```
┌────────────────────────────────────────────────────────────┐
│                   AGENT MEMORY SYSTEM                       │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. SHORT-TERM MEMORY                               │   │
│  │  (Conversation History)                             │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  • Current conversation messages                    │   │
│  │  • Recent tool calls and results                    │   │
│  │  • Immediate context                                │   │
│  │  • Duration: Current session                        │   │
│  │  • Storage: Message array in memory                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  2. WORKING MEMORY                                  │   │
│  │  (Task Context)                                     │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  • Variables and state for current task             │   │
│  │  • Intermediate calculation results                 │   │
│  │  • Task progress tracking                           │   │
│  │  • Duration: Until task complete                    │   │
│  │  • Storage: In-memory dictionaries/objects          │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  3. LONG-TERM MEMORY                                │   │
│  │  (Persistent Storage)                               │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  • User preferences and facts                       │   │
│  │  • Past conversations (summarized)                  │   │
│  │  • Learned knowledge and patterns                   │   │
│  │  • Duration: Indefinite (persists across sessions)  │   │
│  │  • Storage: Vector database (ChromaDB, Pinecone)    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### Human Memory Analogy

Think of it like human memory:

| Memory Type | Human Equivalent | Agent Example |
|------------|------------------|---------------|
| **Short-term** | What you just heard in conversation | Last 10-20 messages in chat |
| **Working** | Notes you're taking during a task | Variables like `total_price = 150` |
| **Long-term** | Facts you remember about a person | "User prefers Python over Java" |

---

## 2. Short-Term Memory (Conversation History)

### What is Short-Term Memory?

Short-term memory is the **conversation history** - the messages exchanged between user and agent in the current session.

### How It Works

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hi, I'm Alice"},
    {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
    {"role": "user", "content": "What's my name?"},
    {"role": "assistant", "content": "Your name is Alice!"}
]
```

The agent can answer "What's my name?" because it **remembers** the conversation history.

### Memory Management

**Problem:** Conversations can get very long
**Solution:** Trim old messages, keeping only recent context

```python
# Keep only last 20 messages (plus system message)
max_messages = 20
if len(messages) > max_messages:
    system_msgs = [m for m in messages if m["role"] == "system"]
    recent_msgs = [m for m in messages if m["role"] != "system"][-max_messages:]
    messages = system_msgs + recent_msgs
```

### Why Trim?

1. **Token limits:** APIs have maximum context length (e.g., 128K tokens)
2. **Cost:** Fewer tokens = lower API costs
3. **Focus:** Too much context can confuse the model
4. **Performance:** Smaller context = faster responses

### Example: Agent Without Memory vs With Memory

**Without Memory:**
```
User: Hi, I'm Bob
Agent: Hello! How can I help you?

User: What's my name?
Agent: I don't know your name. You haven't told me.
```

**With Memory:**
```
User: Hi, I'm Bob
Agent: Hello Bob! How can I help you?

User: What's my name?
Agent: Your name is Bob!
```

---

## 3. Working Memory (Task Context)

### What is Working Memory?

Working memory stores **task-specific information** while the agent works on a problem. It's like a notepad for the current task.

### Use Cases

1. **Multi-step calculations**
   - Store intermediate results
   - Track progress through steps

2. **Data collection tasks**
   - Accumulate information from multiple sources
   - Build up a complete answer

3. **Stateful workflows**
   - Track which steps are complete
   - Know what to do next

### Working Memory Structure

```python
working_memory = {
    "task_name": "Calculate compound interest",
    "status": "in_progress",  # idle, in_progress, completed, failed
    "variables": {
        "principal": 1000,
        "rate": 0.05,
        "time": 3,
        "amount": 1157.63  # Calculated
    },
    "steps_completed": [
        "Get input values",
        "Calculate final amount"
    ],
    "current_step": "Calculate interest gained",
    "intermediate_results": [
        {"step": 1, "result": 1157.63, "description": "Final amount"}
    ]
}
```

### Example: Multi-Step Math Problem

**Task:** "Calculate 20% of 500, store it as 'tax', then add it to 500"

**Working Memory Progression:**

```
Step 1:
working_memory.variables = {}

Step 2: Calculate 20% of 500
working_memory.variables = {"tax": 100}

Step 3: Add to 500
working_memory.variables = {
    "tax": 100,
    "total": 600
}

Task complete!
```

### Benefits of Working Memory

✓ **Track progress** through complex tasks
✓ **Store intermediate results** for later use
✓ **Resume tasks** if interrupted
✓ **Debug issues** by inspecting state
✓ **Avoid redundant calculations** by caching results

---

## 4. Long-Term Memory (Persistent Storage)

### What is Long-Term Memory?

Long-term memory **persists across sessions** using a database. It stores facts, preferences, and knowledge indefinitely.

### Storage: Vector Databases

Long-term memory uses **vector databases** (like ChromaDB) to store information as:
- **Document:** The actual text/fact
- **Embedding:** 384-dimensional vector representing semantic meaning
- **Metadata:** Additional info (type, timestamp, category)

### How Vector Memory Works

```
1. STORE a fact:
   "User prefers Python over JavaScript"
   ↓
   Convert to embedding (vector)
   ↓
   Store in ChromaDB

2. RETRIEVE relevant facts:
   Query: "What languages does user like?"
   ↓
   Convert query to embedding
   ↓
   Find similar embeddings (semantic search)
   ↓
   Return: "User prefers Python over JavaScript"
```

### Types of Long-Term Memories

```python
memory_types = {
    "user_fact": "User's name is Alice",
    "user_preference": "User likes detailed explanations",
    "user_skill": "User is proficient in Python",
    "conversation_summary": "Discussed machine learning basics on 2024-01-15",
    "learned_knowledge": "User works as a data scientist at TechCorp"
}
```

### Retrieval Process

```
┌──────────────────────────────────────────────────────────┐
│  LONG-TERM MEMORY RETRIEVAL                              │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  User Query: "What do I do for work?"                     │
│       ↓                                                   │
│  Convert to embedding [0.23, -0.41, 0.56, ...]           │
│       ↓                                                   │
│  Search vector database for similar embeddings           │
│       ↓                                                   │
│  ┌─────────────────────────────────────────────────┐     │
│  │  Top 3 Matches (by similarity):                 │     │
│  │  1. "User works as data scientist" (0.92)       │     │
│  │  2. "User is skilled in Python" (0.76)          │     │
│  │  3. "User joined TechCorp in 2023" (0.71)       │     │
│  └─────────────────────────────────────────────────┘     │
│       ↓                                                   │
│  Filter by relevance threshold (> 0.70)                  │
│       ↓                                                   │
│  Return matched memories to agent                        │
│       ↓                                                   │
│  Agent uses memories to generate answer:                 │
│  "You work as a data scientist at TechCorp!"            │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Memory Lifecycle

```
┌────────────┐
│ STORE      │  User: "I prefer dark mode"
│            │  → Store: "User prefers dark mode"
└─────┬──────┘
      │
      ↓
┌────────────┐
│ RETRIEVE   │  User: "What do I prefer?"
│            │  → Retrieve: "User prefers dark mode"
└─────┬──────┘
      │
      ↓
┌────────────┐
│ UPDATE     │  User: "Actually, I like light mode now"
│            │  → Update memory or store new version
└─────┬──────┘
      │
      ↓
┌────────────┐
│ FORGET     │  Old memories decay or are deleted
│            │  (optional - can implement memory decay)
└────────────┘
```

### Example: Agent with Full Memory System

```
Session 1:
User: "Hi, I'm Alice and I love Python"
Agent: *Stores in long-term memory*

[Agent shuts down, restart later]

Session 2 (days later):
User: "What do you know about me?"
Agent: *Retrieves from long-term memory*
      "Your name is Alice and you love Python programming!"
```

---

## 5. The ReAct Framework

### What is ReAct?

**ReAct** = **Rea**soning + **Act**ing

A framework where agents **show their thinking** before taking actions, making their decision-making transparent and debuggable.

### Traditional Agent vs ReAct Agent

```
┌─────────────────────────────────────────────────────────┐
│  TRADITIONAL AGENT (Black Box)                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  User: "What is 15% of 340?"                            │
│       ↓                                                  │
│  [Tool: calculator("340 * 0.15")]  ← Hidden reasoning   │
│       ↓                                                  │
│  Agent: "15% of 340 is 51"                              │
│                                                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  REACT AGENT (Transparent)                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  User: "What is 15% of 340?"                            │
│       ↓                                                  │
│  💭 THOUGHT:                                            │
│  "I need to calculate 15% of 340. This requires         │
│   multiplication: 340 × 0.15"                           │
│       ↓                                                  │
│  🔧 ACTION: calculator("340 * 0.15")                    │
│       ↓                                                  │
│  👁️ OBSERVATION: Result = 51.0                         │
│       ↓                                                  │
│  💭 THOUGHT:                                            │
│  "The calculation is complete. I have the answer."      │
│       ↓                                                  │
│  Agent: "15% of 340 is 51"                              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### The ReAct Loop

```
┌───────────────────────────────────────────────────────┐
│                    REACT CYCLE                         │
│                                                        │
│      ┌─────────────────────┐                         │
│      │   1. THOUGHT         │                         │
│      │   (Reasoning)        │                         │
│      │                      │                         │
│      │  "What should I do?" │                         │
│      │  "What info do I     │                         │
│      │   need?"             │                         │
│      └──────────┬───────────┘                         │
│                 │                                      │
│                 ↓                                      │
│      ┌─────────────────────┐                         │
│      │   2. ACTION          │                         │
│      │   (Tool Use)         │                         │
│      │                      │                         │
│      │  Call specific tool  │                         │
│      │  with parameters     │                         │
│      └──────────┬───────────┘                         │
│                 │                                      │
│                 ↓                                      │
│      ┌─────────────────────┐                         │
│      │   3. OBSERVATION     │                         │
│      │   (Result)           │                         │
│      │                      │                         │
│      │  "What did I get?"   │                         │
│      │  "Was it successful?"│                         │
│      └──────────┬───────────┘                         │
│                 │                                      │
│                 ↓                                      │
│      ┌─────────────────────┐                         │
│      │   4. DECISION        │                         │
│      │                      │                         │
│      │  Done? → Answer      │                         │
│      │  Not done? → THOUGHT │─────┐                  │
│      └─────────────────────┘      │                  │
│                                    │                  │
│                    ┌───────────────┘                  │
│                    │                                  │
│                    └─→ Loop back to THOUGHT           │
│                                                        │
└───────────────────────────────────────────────────────┘
```

### ReAct Example: Research Task

**Task:** "Find information about Python and calculate how old it is"

```
ITERATION 1:
💭 THOUGHT: "I need to find when Python was created first"
🔧 ACTION: search_info("Python programming language")
👁️ OBSERVATION: "Python created by Guido van Rossum in 1991"

ITERATION 2:
💭 THOUGHT: "Now I know it was created in 1991. Current year is 2024.
              I need to calculate: 2024 - 1991"
🔧 ACTION: calculator("2024 - 1991")
👁️ OBSERVATION: Result = 33

ITERATION 3:
💭 THOUGHT: "I have all the information needed. Python is 33 years old."
FINAL ANSWER: "Python was created in 1991, making it 33 years old."
```

### Benefits of ReAct

✓ **Transparency:** See exactly why agent made each decision
✓ **Debuggability:** Identify where reasoning went wrong
✓ **Trustworthiness:** Users can verify the logic
✓ **Error correction:** Agent can catch its own mistakes
✓ **Explainability:** Understand the full reasoning chain

---

## 6. Agent Planning Strategies

### What is Planning?

Planning is when an agent **creates a strategy** before executing, rather than deciding step-by-step.

### Plan-Then-Execute Pattern

```
┌────────────────────────────────────────────────────────┐
│  REACTIVE (No Planning)                                │
├────────────────────────────────────────────────────────┤
│  User: "Research Paris and calculate travel budget"    │
│     ↓                                                   │
│  Act → Observe → Act → Observe → Act → Observe         │
│  (Figures out what to do as it goes)                   │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  PLAN-THEN-EXECUTE                                     │
├────────────────────────────────────────────────────────┤
│  User: "Research Paris and calculate travel budget"    │
│     ↓                                                   │
│  ┌──────────────────────────────────────────────┐      │
│  │ PLANNING PHASE:                              │      │
│  │ 1. Search for Paris info                     │      │
│  │ 2. Search for flight prices                  │      │
│  │ 3. Search for hotel prices                   │      │
│  │ 4. Calculate total: flights + hotels + food  │      │
│  └──────────────────────────────────────────────┘      │
│     ↓                                                   │
│  ┌──────────────────────────────────────────────┐      │
│  │ EXECUTION PHASE:                             │      │
│  │ Execute steps 1-4 in order                   │      │
│  └──────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────┘
```

### Types of Planning

#### 1. Linear Planning
Sequential steps, one after another:
```
1. Do A
2. Do B
3. Do C
4. Done
```

#### 2. Hierarchical Planning
Break complex tasks into subtasks:
```
Main Task: Plan vacation
├── Subtask 1: Research destination
│   ├── Find weather info
│   ├── Find attractions
│   └── Read reviews
├── Subtask 2: Book travel
│   ├── Book flights
│   └── Book hotel
└── Subtask 3: Create itinerary
    ├── Day 1 plan
    ├── Day 2 plan
    └── Day 3 plan
```

#### 3. Conditional Planning
Plans with if/then branches:
```
1. Check budget
2. If budget > $2000:
     - Book premium flight
   Else:
     - Book economy flight
3. Search hotels in budget range
4. Book hotel
```

### When to Use Planning

| Use Planning When... | Skip Planning When... |
|---------------------|----------------------|
| ✓ Task is complex with many steps | ✗ Task is simple (1-2 steps) |
| ✓ Steps have dependencies | ✗ Steps are independent |
| ✓ Need to optimize order | ✗ Order doesn't matter |
| ✓ Resources are constrained | ✗ No constraints |
| ✓ Explaining approach is valuable | ✗ Speed is critical |

### Example: Travel Planning Agent

```
PLANNING PHASE:
┌─────────────────────────────────────────────────┐
│ Goal: Plan a 3-day trip to Paris                │
│                                                  │
│ Plan:                                            │
│ 1. Search Paris weather for travel dates        │
│ 2. Find top 5 attractions                       │
│ 3. Search round-trip flights                    │
│ 4. Search hotels near attractions               │
│ 5. Calculate total budget                       │
│ 6. Create day-by-day itinerary                  │
└─────────────────────────────────────────────────┘

EXECUTION PHASE:
Step 1: ✅ Weather found: 18°C, partly cloudy
Step 2: ✅ Attractions: Eiffel Tower, Louvre, Notre Dame...
Step 3: ✅ Flight: $650 round-trip
Step 4: ✅ Hotel: $120/night × 3 = $360
Step 5: ✅ Total budget: $1,210 (flights + hotel + food estimate)
Step 6: ✅ Itinerary created

RESULT: Complete 3-day Paris trip plan with budget
```

---

## 7. Self-Reflection & Error Correction

### What is Self-Reflection?

Self-reflection is when an agent **evaluates its own progress** and adjusts course if needed.

### Reflection Points

```
┌──────────────────────────────────────────────────────┐
│  AGENT WORKFLOW WITH REFLECTION                      │
├──────────────────────────────────────────────────────┤
│                                                       │
│  Action 1: Search for information                    │
│  Action 2: Calculate result                          │
│  Action 3: Search more details                       │
│      ↓                                                │
│  🤔 REFLECTION POINT                                 │
│  "Am I making progress toward the goal?"             │
│  "Did any actions fail?"                             │
│  "Should I change my approach?"                      │
│      ↓                                                │
│  Decision: Adjust approach / Continue / Give up      │
│      ↓                                                │
│  Action 4: Based on reflection...                    │
│                                                       │
└──────────────────────────────────────────────────────┘
```

### Reflection Questions

An agent should ask itself:

1. **Progress Check:**
   - "Am I closer to the goal than before?"
   - "Is this approach working?"

2. **Error Detection:**
   - "Did any tools return errors?"
   - "Are results what I expected?"

3. **Resource Check:**
   - "How many attempts have I made?"
   - "Am I using tools efficiently?"

4. **Strategy Evaluation:**
   - "Is there a better way to solve this?"
   - "Should I try a different tool?"

### Example: Agent with Self-Reflection

```
Task: "Find the population of Paris and calculate persons per square km"

ATTEMPT 1:
Action: search("Paris")
Result: "Paris is the capital of France..."
🤔 REFLECTION: "I got general info, but no population number.
                I need to search more specifically."

ATTEMPT 2:
Action: search("Paris population 2024")
Result: "Paris population: approximately 2.2 million"
🤔 REFLECTION: "Good! I have population. Now I need area."

ATTEMPT 3:
Action: search("Paris area square kilometers")
Result: "Paris area: 105.4 km²"
🤔 REFLECTION: "Perfect! I have both numbers. Now calculate."

ATTEMPT 4:
Action: calculator("2200000 / 105.4")
Result: 20,872
🤔 REFLECTION: "Calculation complete. I have the answer!"

Final: "Paris has about 20,872 persons per square km"
```

### Dynamic Replanning

When reflection reveals a problem, the agent can **replan**:

```
ORIGINAL PLAN:
1. Search for flight prices
2. Search for hotel prices
3. Calculate total budget

EXECUTION:
Step 1: ✅ Flight prices found
Step 2: ❌ Hotel API returned error
🤔 REFLECTION: "Hotel search failed. I need a new approach."

NEW PLAN (Replanning):
1. ✅ (Already done) Flight prices
2. Try alternative hotel search method
3. If that fails, use estimated hotel costs
4. Calculate budget with available data

EXECUTION CONTINUES:
Step 2b: ✅ Alternative search succeeded
Step 3: ✅ Budget calculated
```

### Benefits of Self-Reflection

✓ **Adaptive:** Changes approach when stuck
✓ **Robust:** Recovers from errors automatically
✓ **Efficient:** Avoids repeating failed approaches
✓ **Transparent:** Shows reasoning for changes
✓ **Intelligent:** Learns what works and what doesn't

---

## 8. When to Use Memory & Planning

### Memory Usage Decision Tree

```
                 ┌──────────────────┐
                 │ Does agent need  │
                 │ to remember info?│
                 └────────┬─────────┘
                          │
            ┌─────────────┴─────────────┐
            │                            │
         ┌──▼──┐                     ┌──▼──┐
         │ YES │                     │ NO  │
         └──┬──┘                     └──┬──┘
            │                            │
    ┌───────┴────────┐                  │
    │ For how long?  │            Use stateless
    └───────┬────────┘             agent
            │
    ┌───────┴─────────────────┐
    │                          │
┌───▼────┐              ┌─────▼─────┐
│Current │              │ Across    │
│session │              │ sessions  │
│ only   │              │           │
└───┬────┘              └─────┬─────┘
    │                         │
┌───▼────────┐          ┌─────▼─────────┐
│Short-term  │          │Long-term      │
│memory      │          │memory         │
│(messages)  │          │(vector DB)    │
└────────────┘          └───────────────┘
```

### Planning Usage Decision Tree

```
                 ┌──────────────────┐
                 │ Is task complex? │
                 │ (>3 steps)       │
                 └────────┬─────────┘
                          │
            ┌─────────────┴─────────────┐
            │                            │
         ┌──▼──┐                     ┌──▼──┐
         │ YES │                     │ NO  │
         └──┬──┘                     └──┬──┘
            │                            │
    ┌───────┴────────┐             Use reactive
    │ Dependencies?  │              (no planning)
    └───────┬────────┘
            │
    ┌───────┴─────────────────┐
    │                          │
┌───▼────┐              ┌─────▼────────┐
│Steps   │              │Steps are     │
│depend  │              │independent   │
│on each │              │              │
│other   │              │              │
└───┬────┘              └─────┬────────┘
    │                         │
┌───▼──────────┐        ┌─────▼─────────┐
│Use           │        │Parallel or    │
│sequential    │        │simple         │
│planning      │        │planning       │
└──────────────┘        └───────────────┘
```

### Best Practices

#### Memory Best Practices

✓ **Trim aggressively:** Keep only essential context
✓ **Categorize:** Tag memories by type (fact, preference, skill)
✓ **Update regularly:** Replace outdated information
✓ **Filter by relevance:** Don't retrieve irrelevant memories
✓ **Summarize periodically:** Condense long conversations

#### Planning Best Practices

✓ **Plan at the right level:** Not too detailed, not too vague
✓ **Allow flexibility:** Plans can change during execution
✓ **Set checkpoints:** Validate progress at key points
✓ **Time-box planning:** Don't spend too long planning
✓ **Learn from failures:** Improve planning from mistakes

### Real-World Applications

| Application | Memory Needed | Planning Needed |
|------------|---------------|-----------------|
| **Chat** bot | Short-term (conversation) | No planning |
| **Research Agent** | Long-term (sources) + Working (findings) | Yes - search strategy |
| **Customer Support** | Long-term (user history) + Short-term | Conditional planning |
| **Personal Assistant** | All three types | Yes - task scheduling |
| **Code Generator** | Working (current code structure) | Yes - implementation plan |
| **Data Analyzer** | Working (analysis results) | Yes - analysis steps |

---

## 🎓 Summary

### Key Concepts Recap

**Memory Systems:**
- ✓ **Short-term:** Conversation history (messages)
- ✓ **Working:** Task variables and state
- ✓ **Long-term:** Persistent facts (vector DB)

**ReAct Framework:**
- ✓ **Thought:** Reasoning about what to do
- ✓ **Action:** Using tools
- ✓ **Observation:** Analyzing results
- ✓ **Loop:** Repeat until task complete

**Planning:**
- ✓ **Plan-then-execute:** Create strategy first
- ✓ **Dynamic replanning:** Adapt when things fail
- ✓ **Self-reflection:** Evaluate progress periodically

### Architecture Diagram

```
┌────────────────────────────────────────────────────────┐
│              COMPLETE AGENT ARCHITECTURE                │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌──────────────┐               │
│  │  Short-term  │◄────►│   Working    │               │
│  │   Memory     │      │   Memory     │               │
│  │ (messages)   │      │ (task state) │               │
│  └──────┬───────┘      └──────┬───────┘               │
│         │                      │                       │
│         └──────────┬───────────┘                       │
│                    │                                   │
│              ┌─────▼─────┐                            │
│              │           │                            │
│              │  REACT    │                            │
│              │  AGENT    │                            │
│              │           │                            │
│              │  • Think  │                            │
│              │  • Act    │                            │
│              │  • Observe│                            │
│              │  • Reflect│                            │
│              │           │                            │
│              └─────┬─────┘                            │
│                    │                                   │
│         ┌──────────┴───────────┐                      │
│         │                      │                      │
│    ┌────▼─────┐         ┌─────▼──────┐               │
│    │ Long-term│         │  Planning  │               │
│    │  Memory  │         │   System   │               │
│    │(Vector DB)│         │            │               │
│    └──────────┘         └────────────┘               │
│                                                         │
└────────────────────────────────────────────────────────┘
```

---

## 📝 Knowledge Check

Test your understanding with these questions:

### Question 1: Memory Types
Which type of memory would you use to store "User prefers dark mode" across sessions?

<details>
<summary>Click to see answer</summary>

**Answer:** Long-term memory (vector database)

**Explanation:** This is a persistent preference that should be remembered even after the session ends, making it perfect for long-term memory storage.

</details>

### Question 2: ReAct Benefits
What is the main advantage of the ReAct framework over traditional agents?

<details>
<summary>Click to see answer</summary>

**Answer:** Transparency and debuggability

**Explanation:** ReAct makes the agent's reasoning visible through explicit "Thought" steps, allowing developers to see exactly why the agent made each decision. This makes it much easier to debug and understand agent behavior.

</details>

### Question 3: When to Plan
When should an agent create a plan before executing?

<details>
<summary>Click to see answer</summary>

**Answer:** When the task is complex (>3 steps) and steps have dependencies

**Explanation:** Planning is most valuable for complex tasks where the order of operations matters and where understanding the full strategy upfront leads to better outcomes.

</details>

### Question 4: Working Memory
What should be stored in working memory during a multi-step calculation?

<details>
<summary>Click to see answer</summary>

**Answer:** Intermediate results and variables for the current task

**Explanation:** Working memory is perfect for storing temporary values like "tax = 100" or "subtotal = 500" that are only needed for the duration of the current task.

</details>

### Question 5: Self-Reflection
How does self-reflection improve agent performance?

<details>
<summary>Click to see answer</summary>

**Answer:** It allows the agent to detect errors, evaluate progress, and adjust its approach when needed

**Explanation:** Self-reflection acts like a checkpoint system where the agent periodically asks "Is this working?" and can change strategy if the current approach isn't making progress.

</details>

---

## 🚀 Ready for Hands-On Practice!

You now understand:
- ✅ The three types of agent memory
- ✅ How memory systems work together
- ✅ The ReAct framework for transparent reasoning
- ✅ Planning strategies and when to use them
- ✅ Self-reflection for error correction

**Next Step:** Move to the hands-on lab to build these systems yourself!

[→ Continue to Hands-On Lab](lab.md)

---

**Learning Module Complete!** 🎉
Time to put theory into practice with real code examples.
