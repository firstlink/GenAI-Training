# Lab 6: AI Agents & Tool Calling

## 📚 Learning Material

**Duration:** 40 minutes
**Difficulty:** Intermediate to Advanced
**Prerequisites:** Labs 1-5 completed

---

## 🎯 Learning Objectives

By the end of this learning module, you will understand:
- ✅ What AI agents are and how they differ from chatbots
- ✅ The agent execution loop (Observe → Think → Act → Repeat)
- ✅ Tool/function calling concepts and implementation
- ✅ Agent decision-making and planning
- ✅ When to use agents vs. simple LLM calls
- ✅ Common agent architectures and patterns
- ✅ Error handling and safety in agent systems

---

## 📖 Table of Contents

1. [Introduction: Chatbot vs. Agent](#1-introduction-chatbot-vs-agent)
2. [The Agent Loop](#2-the-agent-loop)
3. [Tool Calling Fundamentals](#3-tool-calling-fundamentals)
4. [Agent Architecture](#4-agent-architecture)
5. [Planning and Reasoning](#5-planning-and-reasoning)
6. [Error Handling and Safety](#6-error-handling-and-safety)
7. [When to Use Agents](#7-when-to-use-agents)
8. [Review & Key Takeaways](#8-review--key-takeaways)

---

## 1. Introduction: Chatbot vs. Agent

### What is an AI Agent?

An **AI Agent** is an AI system that can:
1. **Reason** about what actions to take
2. **Plan** sequences of steps
3. **Use tools** to interact with external systems
4. **Adapt** based on results
5. **Iterate** until goals are achieved

### The Key Difference

```
┌────────────────────────────────────────────────────────┐
│  TRADITIONAL CHATBOT (Simple LLM)                      │
├────────────────────────────────────────────────────────┤
│                                                        │
│  User: "What's the weather in Paris?"                 │
│                  ↓                                     │
│              [  LLM  ]                                 │
│                  ↓                                     │
│  Response: "I don't have access to current weather    │
│            data. Please check weather.com"            │
│                                                        │
│  Limitations:                                          │
│  ❌ No access to real-time data                       │
│  ❌ Can't take actions                                │
│  ❌ Just generates text                               │
│                                                        │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  AI AGENT (LLM + Tools)                                │
├────────────────────────────────────────────────────────┤
│                                                        │
│  User: "What's the weather in Paris?"                 │
│                  ↓                                     │
│  Agent: [Thinks] "I need current weather data"        │
│                  ↓                                     │
│  Agent: [Acts] Calls weather_api(city="Paris")        │
│                  ↓                                     │
│  API Result: {temp: 18°C, conditions: "Cloudy"}       │
│                  ↓                                     │
│  Agent: [Responds] "The current weather in Paris is   │
│         18°C with cloudy skies."                       │
│                                                        │
│  Capabilities:                                         │
│  ✅ Access to real-time data via APIs                 │
│  ✅ Can take actions (search, calculate, etc.)        │
│  ✅ Provides accurate, current information            │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Real-World Example

**Question:** "What's 15% off of $245.99, and can I afford it if I have $200?"

**Chatbot (No Tools):**
```
"Let me calculate that for you. 15% of $245.99 is approximately
$36.90, so the discounted price would be around $209.09.
With $200, you would be about $9.09 short."
```
❌ Risk of calculation errors
❌ No verification

**Agent (With Calculator Tool):**
```
[Thinks] "I need to calculate 15% discount and compare to budget"
[Action 1] calculate(245.99 * 0.15) → Result: 36.8985
[Action 2] calculate(245.99 - 36.8985) → Result: 209.09
[Action 3] calculate(209.09 - 200) → Result: 9.09
[Response] "With 15% off, the price would be $209.09. You have
$200, so you'd be $9.09 short of being able to afford it."
```
✅ Accurate calculations
✅ Verified results
✅ Step-by-step reasoning

**⏱️ Duration so far:** 5 minutes

---

## 2. The Agent Loop

### The Core Execution Pattern

Every agent follows this pattern:

```
┌─────────────────────────────────────────────────────────┐
│  THE AGENT LOOP                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. OBSERVE                                             │
│     ┌─────────────────────────────────────┐            │
│     │ • Read user input                    │            │
│     │ • Review conversation context        │            │
│     │ • Check available tools              │            │
│     │ • Examine current state              │            │
│     └─────────────────────────────────────┘            │
│                       ↓                                 │
│  2. THINK (Reasoning)                                   │
│     ┌─────────────────────────────────────┐            │
│     │ • What is being asked?               │            │
│     │ • What information do I need?        │            │
│     │ • Which tool should I use?           │            │
│     │ • What are the parameters?           │            │
│     │ • Do I have enough info to answer?   │            │
│     └─────────────────────────────────────┘            │
│                       ↓                                 │
│  3. DECIDE                                              │
│     ┌─────────────────────────────────────┐            │
│     │ Choice A: Use a tool                 │            │
│     │ Choice B: Respond to user            │            │
│     │ Choice C: Ask for clarification      │            │
│     └─────────────────────────────────────┘            │
│                       ↓                                 │
│  4. ACT                                                 │
│     ┌─────────────────────────────────────┐            │
│     │ If Tool: Execute function            │            │
│     │ If Response: Generate answer         │            │
│     │ If Clarify: Ask question             │            │
│     └─────────────────────────────────────┘            │
│                       ↓                                 │
│  5. EVALUATE                                            │
│     ┌─────────────────────────────────────┐            │
│     │ • Did the action succeed?            │            │
│     │ • Do I have enough information now?  │            │
│     │ • Is the task complete?              │            │
│     └─────────────────────────────────────┘            │
│                       ↓                                 │
│              ┌────────────────┐                        │
│              │  Task Complete?│                        │
│              └───┬────────┬───┘                        │
│                 NO       YES                            │
│                  │         │                            │
│                  └─────────┘                            │
│                 (Loop)  (Done)                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Detailed Example: Weather Query

Let's trace through "What should I wear in Paris tomorrow?"

**Iteration 1:**
```
OBSERVE: User asks about clothing for Paris tomorrow
THINK: I need:
  - Current date
  - Tomorrow's weather in Paris
  - Temperature range
DECIDE: Use get_weather tool
ACT: get_weather(city="Paris", date="tomorrow")
RESULT: {
  date: "2024-12-01",
  temp_high: 15°C,
  temp_low: 8°C,
  conditions: "Light rain",
  precipitation: 60%
}
EVALUATE: Have weather info, can now answer
```

**Iteration 2:**
```
THINK: Based on weather data:
  - Cool temperatures (8-15°C)
  - Rain expected
  - Need layers + rain protection
DECIDE: Have enough info, respond to user
ACT: Generate response
RESPONSE: "Tomorrow in Paris will be cool (8-15°C) with light rain.
I'd recommend:
  - Layered clothing (jacket or sweater)
  - Waterproof jacket or umbrella
  - Comfortable walking shoes"
EVALUATE: Task complete ✓
```

### Key Characteristics

**1. Iterative**
- Agents can make multiple tool calls
- Each step builds on previous results

**2. Adaptive**
- If a tool fails, try alternative approach
- Adjust plan based on results

**3. Goal-Oriented**
- Continues until task is complete
- Knows when to stop

**⏱️ Duration so far:** 12 minutes

---

## 3. Tool Calling Fundamentals

### What is Tool Calling?

**Tool Calling** = Allowing the LLM to invoke functions/APIs to accomplish tasks.

### Anatomy of a Tool

Every tool has three parts:

```
┌────────────────────────────────────────────────────────┐
│  1. DEFINITION (Schema)                                │
│     What the tool is and how to use it                 │
├────────────────────────────────────────────────────────┤
│                                                        │
│  {                                                     │
│    "name": "get_weather",                             │
│    "description": "Get current weather for a city",   │
│    "parameters": {                                     │
│      "type": "object",                                │
│      "properties": {                                   │
│        "city": {                                       │
│          "type": "string",                            │
│          "description": "The city name"               │
│        },                                             │
│        "units": {                                      │
│          "type": "string",                            │
│          "enum": ["celsius", "fahrenheit"],           │
│          "description": "Temperature units"           │
│        }                                              │
│      },                                               │
│      "required": ["city"]                             │
│    }                                                  │
│  }                                                    │
│                                                        │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  2. IMPLEMENTATION (Function)                          │
│     The actual Python code that executes               │
├────────────────────────────────────────────────────────┤
│                                                        │
│  def get_weather(city: str, units: str = "celsius"):  │
│      """Get current weather for a city"""             │
│                                                        │
│      # Call weather API                               │
│      response = weather_api.get(city)                 │
│                                                        │
│      # Convert units if needed                        │
│      temp = convert_temperature(response.temp, units) │
│                                                        │
│      return {                                         │
│          "temperature": temp,                         │
│          "conditions": response.conditions,           │
│          "humidity": response.humidity                │
│      }                                                │
│                                                        │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  3. REGISTRATION                                       │
│     Telling the LLM the tool exists                    │
├────────────────────────────────────────────────────────┤
│                                                        │
│  tools = [                                            │
│      {                                                │
│          "type": "function",                          │
│          "function": {                                │
│              "name": "get_weather",                   │
│              "description": "...",                    │
│              "parameters": {...}                      │
│          }                                            │
│      }                                                │
│  ]                                                    │
│                                                        │
│  # Pass to LLM                                        │
│  openai.chat.completions.create(                      │
│      model="gpt-4",                                   │
│      messages=messages,                               │
│      tools=tools  # ← LLM now knows about this tool   │
│  )                                                    │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### How Tool Calling Works

**Step-by-Step Flow:**

```
┌─────────────────────────────────────────────────────────┐
│  USER → AGENT WORKFLOW                                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. User: "What's 25 × 37?"                             │
│                  ↓                                      │
│  2. You send to LLM with available tools                │
│     messages = [{"role": "user", "content": "25×37"}]   │
│     tools = [calculator_tool_definition]                │
│                  ↓                                      │
│  3. LLM decides to use tool (returns tool call)         │
│     {                                                   │
│       "role": "assistant",                             │
│       "tool_calls": [{                                  │
│         "function": {                                   │
│           "name": "calculate",                         │
│           "arguments": '{"expression": "25 * 37"}'     │
│         }                                              │
│       }]                                               │
│     }                                                  │
│                  ↓                                      │
│  4. You execute the function                            │
│     result = calculate("25 * 37")  # Returns: 925       │
│                  ↓                                      │
│  5. You send result back to LLM                         │
│     messages.append({                                   │
│       "role": "tool",                                  │
│       "content": "925"                                 │
│     })                                                 │
│                  ↓                                      │
│  6. LLM generates final response                        │
│     "25 multiplied by 37 equals 925."                  │
│                  ↓                                      │
│  7. You return to user                                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Tool Definition Best Practices

**1. Clear, Descriptive Names**
```python
✅ Good: get_current_weather, search_database, send_email
❌ Bad: tool1, get_stuff, do_thing
```

**2. Detailed Descriptions**
```python
✅ Good:
"Get the current weather conditions for a specific city. Returns
temperature, conditions (sunny/cloudy/rainy), humidity, and wind speed."

❌ Bad:
"Gets weather"
```

**3. Well-Defined Parameters**
```python
✅ Good:
{
  "city": {
    "type": "string",
    "description": "City name (e.g., 'Paris', 'New York')"
  },
  "units": {
    "type": "string",
    "enum": ["celsius", "fahrenheit"],
    "description": "Temperature unit preference"
  }
}

❌ Bad:
{
  "location": {"type": "string"},
  "format": {"type": "string"}
}
```

**⏱️ Duration so far:** 22 minutes

---

## 4. Agent Architecture

### Common Agent Patterns

#### Pattern 1: ReAct (Reason + Act)

**Most common pattern for agents.**

```
┌─────────────────────────────────────────────────────────┐
│  ReAct PATTERN                                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  User Question                                          │
│         ↓                                               │
│  ┌─────────────────────────────────────┐               │
│  │ THOUGHT: What do I need to do?      │               │
│  │ "I need to find current price of    │               │
│  │  Bitcoin and compare to yesterday"  │               │
│  └─────────────────────────────────────┘               │
│         ↓                                               │
│  ┌─────────────────────────────────────┐               │
│  │ ACTION: Use tool                    │               │
│  │ get_crypto_price(coin="BTC",        │               │
│  │                  date="today")      │               │
│  └─────────────────────────────────────┘               │
│         ↓                                               │
│  ┌─────────────────────────────────────┐               │
│  │ OBSERVATION: Result from tool       │               │
│  │ {"price": 42000, "change": "+3.5%"} │               │
│  └─────────────────────────────────────┘               │
│         ↓                                               │
│  ┌─────────────────────────────────────┐               │
│  │ THOUGHT: Is this enough?            │               │
│  │ "Yes, I have the info needed"       │               │
│  └─────────────────────────────────────┘               │
│         ↓                                               │
│  ┌─────────────────────────────────────┐               │
│  │ ANSWER: Final response              │               │
│  │ "Bitcoin is currently at $42,000,   │               │
│  │  up 3.5% from yesterday."           │               │
│  └─────────────────────────────────────┘               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Key Feature:** Explicit reasoning steps visible in the process.

#### Pattern 2: Tool-Calling Loop

**OpenAI/Anthropic native pattern.**

```
Start → LLM Call → Tool Call? → Yes → Execute → Add to History → Loop Back
                        ↓ No
                    Final Answer
```

**Advantages:**
- Clean, simple implementation
- Native to modern LLMs
- Automatic tool selection

#### Pattern 3: Plan-and-Execute

**For complex, multi-step tasks.**

```
1. PLAN Phase:
   User: "Plan a trip to Paris"
   Agent: Creates plan:
     - Search flights
     - Find hotels
     - Research attractions
     - Estimate budget

2. EXECUTE Phase:
   For each step in plan:
     - Execute with tools
     - Gather results
     - Move to next step

3. SYNTHESIZE Phase:
   Combine all results into coherent response
```

### Agent Components in Detail

```
┌─────────────────────────────────────────────────────────┐
│  COMPLETE AGENT SYSTEM                                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────┐           │
│  │  LLM (Reasoning Engine)                 │           │
│  │  - GPT-4, Claude, etc.                  │           │
│  │  - Makes decisions                      │           │
│  │  - Generates responses                  │           │
│  └─────────────────────────────────────────┘           │
│                      ↕                                  │
│  ┌─────────────────────────────────────────┐           │
│  │  Agent Controller                       │           │
│  │  - Manages execution loop               │           │
│  │  - Routes tool calls                    │           │
│  │  - Handles errors                       │           │
│  └─────────────────────────────────────────┘           │
│                      ↕                                  │
│  ┌─────────────────────────────────────────┐           │
│  │  Tool Registry                          │           │
│  │  - Stores available tools               │           │
│  │  - Validates tool calls                 │           │
│  │  - Executes functions                   │           │
│  └─────────────────────────────────────────┘           │
│                      ↕                                  │
│  ┌─────────────────────────────────────────┐           │
│  │  Memory System                          │           │
│  │  - Conversation history                 │           │
│  │  - Tool call history                    │           │
│  │  - Context management                   │           │
│  └─────────────────────────────────────────┘           │
│                      ↕                                  │
│  ┌─────────────────────────────────────────┐           │
│  │  Safety Layer                           │           │
│  │  - Input validation                     │           │
│  │  - Rate limiting                        │           │
│  │  - Error handling                       │           │
│  └─────────────────────────────────────────┘           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**⏱️ Duration so far:** 30 minutes

---

## 5. Planning and Reasoning

### How Agents Plan

Agents use different planning strategies:

**1. Single-Step (Reactive)**
```
Question → Tool Call → Answer
Simple, fast, works for straightforward tasks
```

**2. Multi-Step (Sequential)**
```
Question → Plan steps → Execute step 1 → Execute step 2 → ... → Answer
Good for tasks requiring multiple pieces of information
```

**3. Iterative Refinement**
```
Question → Initial attempt → Evaluate → Refine → Try again → Answer
Useful when first attempt might fail or be incomplete
```

### Example: Complex Planning

**Task:** "Find the best restaurant near me that's open now and has vegan options"

**Agent's Plan:**
```
Step 1: Get user's location
  Tool: get_user_location()

Step 2: Get current time
  Tool: get_current_time()

Step 3: Search restaurants
  Tool: search_restaurants(
    location=user_location,
    dietary="vegan",
    open_now=true
  )

Step 4: Rank by rating
  Tool: sort_by_rating(restaurants)

Step 5: Return top result
  Generate response with recommendation
```

### Decision Trees in Agents

Agents make decisions at each step:

```
                    Start
                      │
           ┌──────────┴──────────┐
           │                     │
    Simple Question?      Complex Question?
           │                     │
           ↓                     ↓
    Use single tool     Create multi-step plan
           │                     │
           ↓                     ↓
    Execute & respond    Execute sequence
                               │
                               ↓
                        Aggregate results
                               │
                               ↓
                        Generate response
```

**⏱️ Duration so far:** 35 minutes

---

## 6. Error Handling and Safety

### Common Agent Errors

**1. Tool Execution Failures**
```python
User: "What's the weather in XYZ123?" (Invalid city)
Tool: get_weather("XYZ123") → Error: City not found

Agent should:
✅ Catch error gracefully
✅ Ask user for clarification
✅ Suggest alternatives
❌ Don't crash or hallucinate
```

**2. Infinite Loops**
```python
# Bad: Agent keeps calling same tool repeatedly
Agent: Call tool → Result insufficient → Call tool again → ...

Prevention:
✅ Max iteration limit (e.g., 10 steps)
✅ Track tool call history
✅ Detect repeated patterns
```

**3. Hallucinated Tool Calls**
```python
# Agent tries to call a tool that doesn't exist
Agent: "I'll use the make_coffee() tool..."

Prevention:
✅ Strict tool validation
✅ Clear tool descriptions
✅ Reject invalid tool calls
```

### Safety Considerations

```
┌────────────────────────────────────────────────────────┐
│  AGENT SAFETY CHECKLIST                                │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ✓ Input Validation                                   │
│    - Validate all user inputs                         │
│    - Sanitize data before tool calls                  │
│                                                        │
│  ✓ Tool Restrictions                                  │
│    - Whitelist allowed tools                          │
│    - No dangerous operations (delete, format, etc.)   │
│    - Require confirmations for critical actions       │
│                                                        │
│  ✓ Rate Limiting                                      │
│    - Limit tool calls per minute                      │
│    - Prevent abuse/spam                               │
│                                                        │
│  ✓ Error Handling                                     │
│    - Try-catch around all tool calls                  │
│    - Graceful degradation                             │
│    - Informative error messages                       │
│                                                        │
│  ✓ Monitoring                                         │
│    - Log all tool calls                               │
│    - Track failures                                   │
│    - Alert on anomalies                               │
│                                                        │
│  ✓ Cost Controls                                      │
│    - Token usage limits                               │
│    - Tool call cost tracking                          │
│    - Budget alerts                                    │
│                                                        │
└────────────────────────────────────────────────────────┘
```

**⏱️ Duration so far:** 38 minutes

---

## 7. When to Use Agents

### Decision Framework

```
┌──────────────────────────────┬────────────┬──────────────┐
│  Scenario                    │  Use Agent?│  Why         │
├──────────────────────────────┼────────────┼──────────────┤
│ Need current information     │  ✓ YES     │  Tools can   │
│ (weather, stock prices)      │            │  fetch data  │
├──────────────────────────────┼────────────┼──────────────┤
│ Mathematical calculations    │  ✓ YES     │  Calculator  │
│                              │            │  tool needed │
├──────────────────────────────┼────────────┼──────────────┤
│ Multi-step tasks             │  ✓ YES     │  Agent loop  │
│ (research, planning)         │            │  handles it  │
├──────────────────────────────┼────────────┼──────────────┤
│ Creative writing             │  ✗ NO      │  No tools    │
│ (stories, poems)             │            │  needed      │
├──────────────────────────────┼────────────┼──────────────┤
│ Simple Q&A from docs         │  ✗ NO      │  RAG is      │
│ (already in context)         │            │  sufficient  │
├──────────────────────────────┼────────────┼──────────────┤
│ Need to take actions         │  ✓ YES     │  Tools for   │
│ (send email, create ticket)  │            │  actions     │
└──────────────────────────────┴────────────┴──────────────┘
```

### Agent vs. RAG vs. Simple LLM

```
Simple LLM:
├─ Best for: General knowledge questions
├─ Example: "Explain photosynthesis"
└─ Why: No external data needed

RAG System:
├─ Best for: Questions about your documents
├─ Example: "What's in our Q3 report?"
└─ Why: Needs to search your knowledge base

Agent System:
├─ Best for: Tasks requiring actions or current data
├─ Example: "Email the team about tomorrow's weather"
└─ Why: Needs both tools (email, weather) and reasoning
```

### Cost Considerations

**Agents are more expensive:**
```
Simple LLM Call:
- 1 API call
- ~500 tokens
- Cost: $0.01

Agent with Tools:
- 3-5 API calls (reasoning + tool loops)
- ~2000 tokens
- Cost: $0.04

When it's worth it: When accuracy and capability matter more than cost
```

**⏱️ Duration so far:** 40 minutes

---

## 8. Review & Key Takeaways

### 🎯 What You Learned

✅ **Agent Definition** - AI systems that reason, plan, use tools, and iterate
✅ **Agent Loop** - Observe → Think → Decide → Act → Evaluate → Repeat
✅ **Tool Calling** - How LLMs invoke functions to accomplish tasks
✅ **Agent Architecture** - ReAct, Tool-Calling Loop, Plan-and-Execute patterns
✅ **Planning** - Single-step, multi-step, iterative refinement strategies
✅ **Safety** - Error handling, validation, rate limiting, monitoring
✅ **When to Use** - Agents vs. RAG vs. simple LLM decision framework

### 💡 Key Concepts

**1. Agents ≠ Chatbots**
```
Chatbot: Generates text
Agent: Reasons + Takes actions + Uses tools
```

**2. The Loop is Everything**
```
Agents iterate until task is complete
Each iteration builds on previous results
```

**3. Tools Extend Capabilities**
```
LLM alone: Limited to training data
LLM + Tools: Can access any information/service
```

**4. Planning is Critical**
```
Simple task: Single tool call
Complex task: Multi-step plan → Execute → Synthesize
```

**5. Safety First**
```
Always validate inputs
Limit iterations
Handle errors gracefully
Monitor and log
```

### 🧠 Knowledge Check

<details>
<summary><strong>Question 1:</strong> What's the main difference between a chatbot and an agent?</summary>

**Answer:**
A chatbot generates text based on its training data. An agent can:
- Reason about what actions to take
- Use tools to access external information
- Take actions (call APIs, perform calculations, etc.)
- Iterate until a task is complete
</details>

<details>
<summary><strong>Question 2:</strong> What are the 5 steps in the agent loop?</summary>

**Answer:**
1. **Observe** - Read input and context
2. **Think** - Reason about what to do
3. **Decide** - Choose tool or response
4. **Act** - Execute the decision
5. **Evaluate** - Check if task is complete, loop if not
</details>

<details>
<summary><strong>Question 3:</strong> What are the three parts of a tool definition?</summary>

**Answer:**
1. **Definition (Schema)** - Name, description, parameters
2. **Implementation** - The actual Python function
3. **Registration** - Telling the LLM the tool exists
</details>

<details>
<summary><strong>Question 4:</strong> When should you use an agent vs. simple RAG?</summary>

**Answer:**
Use **Agent** when:
- Need current/real-time information
- Need to perform calculations
- Task requires multiple steps
- Need to take actions (send email, create records)

Use **RAG** when:
- Answering questions from your documents
- Information is already in your knowledge base
- Don't need external tools
</details>

<details>
<summary><strong>Question 5:</strong> What are three important safety considerations for agents?</summary>

**Answer:**
1. **Input Validation** - Sanitize and validate all inputs
2. **Iteration Limits** - Prevent infinite loops (max 10 steps)
3. **Error Handling** - Try-catch around tool calls, graceful failures
</details>

### 🚀 Ready for Hands-On Practice?

You now understand:
- ✅ The agent loop and execution pattern
- ✅ Tool calling fundamentals
- ✅ Agent architecture patterns
- ✅ Planning and reasoning strategies
- ✅ Safety and error handling

**Next step**: [Hands-On Lab →](lab.md)

In the lab, you'll:
1. Build your first tool-calling agent
2. Create and register custom tools
3. Implement the agent execution loop
4. Handle multi-step reasoning
5. Add error handling and safety
6. Build a production agent system

---

### 📚 Additional Resources

**Want to dive deeper?**
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use Documentation](https://docs.anthropic.com/claude/docs/tool-use)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [ReAct Paper (Original Research)](https://arxiv.org/abs/2210.03629)

---

**Learning Material Complete!** ✅
[← Back to README](../README.md) | [Start Hands-On Lab →](lab.md)
