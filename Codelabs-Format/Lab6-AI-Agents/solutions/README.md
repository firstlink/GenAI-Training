# Lab 6 Solutions: AI Agents & Tool Calling

## 📚 Overview

Complete solutions for Lab 6 covering AI agents with tool calling, multi-tool assistants, conditional workflows, error handling, and multi-agent systems. Build production-ready agents that can use external tools and make intelligent decisions.

---

## 📁 Files Included

### Exercise Solutions
- **`exercise1_calculator_agents.py`** - Basic calculator agents (OpenAI & Claude)
- **`all_exercises.py`** - Consolidated solution with all concepts

### Configuration
- **`.env.example`** - Environment configuration template
- **`README.md`** - This comprehensive guide

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install required libraries
pip install openai anthropic python-dotenv
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

# Optional (for Claude comparison)
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Run Solutions

```bash
# Run individual exercise
python exercise1_calculator_agents.py

# Run all exercises (recommended)
python all_exercises.py
```

---

## 📖 What's Covered

### Exercise 1: Basic Calculator Agent

**Concepts:**
- Tool/function calling with OpenAI
- Tool use with Claude (Anthropic)
- Agent execution loop
- API differences between providers

**Key Implementation:**
```python
from all_exercises import BasicCalculatorAgent

agent = BasicCalculatorAgent()
result = agent.run("What is 15% of 340?")
# Agent uses calculator tool automatically
```

**Agent Execution Flow:**
1. User query → LLM
2. LLM decides to use tool
3. Execute tool function
4. Send result back to LLM
5. LLM generates final response

**OpenAI vs Claude:**
- **OpenAI**: Uses `tool_calls` array with `type: "function"`
- **Claude**: Uses `tool_use` blocks with `stop_reason: "tool_use"`

---

### Exercise 2: Multi-Tool Assistant

**Concepts:**
- Registering multiple tools
- Intelligent tool selection
- Sequential tool usage
- Tool execution mapping

**Available Tools:**
- `calculator` - Mathematical operations
- `get_current_datetime` - Current date/time
- `search_knowledge_base` - Information lookup
- `send_notification` - Send notifications

**Usage:**
```python
from all_exercises import MultiToolAssistant

agent = MultiToolAssistant()

# Agent automatically selects appropriate tool
agent.run("What is 20% of 500?")  # Uses calculator
agent.run("What's today's date?")  # Uses datetime
agent.run("Tell me about Python")  # Uses knowledge base
agent.run("Remind me to review agents")  # Uses notifications
```

**How Tool Selection Works:**
- LLM reads tool descriptions
- Matches query intent to tool purpose
- Automatically selects best tool(s)
- Can use multiple tools in sequence

---

### Exercise 3: Conditional Workflow Agent

**Concepts:**
- Conditional logic based on tool results
- Branching workflows
- Sequential tool calling
- Business rules implementation

**Workflow Example:**
```
User Request
    ↓
Check Business Hours
    ↓
  ┌─────┴─────┐
  ↓           ↓
OPEN        CLOSED
  ↓           ↓
Send Email  Create Ticket
```

**Implementation:**
```python
from all_exercises import ConditionalWorkflowAgent

agent = ConditionalWorkflowAgent()

# During business hours (9 AM - 5 PM)
agent.run("I need urgent help with a system outage")
# → Checks hours → Open → Sends email

# Outside business hours
agent.run("I have a question about my account")
# → Checks hours → Closed → Creates ticket
```

**Key Features:**
- System prompts guide decision-making
- Tools inform next actions
- Agents explain their decisions
- Fallback paths for different conditions

---

### Exercise 4: Resilient Agent with Error Handling

**Concepts:**
- Safe tool implementations
- Structured error responses
- Fallback mechanisms
- Production error handling
- Logging and monitoring

**Error Handling Patterns:**

**1. Input Validation:**
```python
def safe_calculator(expression: str):
    if not expression:
        return {"success": False, "error": "Invalid input"}
```

**2. Security Checks:**
```python
dangerous_keywords = ['import', 'exec', 'open']
if any(kw in expression for kw in dangerous_keywords):
    return {"success": False, "error": "Forbidden operation"}
```

**3. Graceful Degradation:**
```python
# Try primary service
result = primary_search(query)
if not result["success"]:
    # Fall back to backup
    result = backup_search(query)
```

**4. User-Friendly Errors:**
```python
# Instead of: "ZeroDivisionError: division by zero"
# Return: "Cannot divide by zero. Please try a different calculation."
```

**Usage:**
```python
from all_exercises import ResilientAgent

agent = ResilientAgent()

# Valid calculation
agent.run("Calculate 50 * 20")
# → Success: 1000

# Error handling
agent.run("Calculate 10 / 0")
# → Error caught: "Cannot divide by zero"

# Automatic fallback
agent.run("Search for Python information")
# → Tries primary → If fails, uses backup
```

---

### Capstone: AgentHub v1.0 - Multi-Agent Platform

**Architecture:**
```
        AgentHub (Router)
              |
    ┌─────────┼─────────┐
    ↓         ↓         ↓
MathAgent  InfoAgent  TaskAgent
    |         |         |
Calculator  Search   Notifications
          DateTime    Tickets
```

**Specialized Agents:**

**1. MathAgent:**
- Handles mathematical calculations
- Percentages, arithmetic, formulas

**2. InfoAgent:**
- Knowledge base search
- Current date/time queries
- Information retrieval

**3. TaskAgent:**
- Send notifications
- Create support tickets
- Task management

**Routing Logic:**
```python
class AgentHub:
    def route_query(self, query):
        if "calculate" in query or "%" in query:
            return "math"
        elif "search" in query or "what is" in query:
            return "info"
        elif "remind" in query or "ticket" in query:
            return "task"
        return "info"  # Default
```

**Usage:**
```python
from all_exercises import AgentHub

hub = AgentHub()

# Automatically routes to appropriate agent
hub.process("What is 25% of 340?")
# → Routes to MathAgent

hub.process("Tell me about Python")
# → Routes to InfoAgent

hub.process("Remind me to review agents")
# → Routes to TaskAgent
```

**Features:**
- ✅ Intelligent query routing
- ✅ Multiple specialized agents
- ✅ Error handling throughout
- ✅ Logging and monitoring
- ✅ Modular, extensible design

---

## 💡 Key Concepts

### Tool/Function Calling

**What is it?**
Tool calling allows LLMs to use external functions and APIs, extending their capabilities beyond text generation.

**When to use tools:**
- External data access (APIs, databases)
- Calculations and computations
- Actions and side effects
- Real-time information
- Domain-specific operations

**Tool Definition Components:**
1. **Name**: Identifier for the function
2. **Description**: When and how to use it
3. **Parameters**: Input schema (JSON Schema)
4. **Implementation**: The actual function code

### Agent Execution Loop

**Standard Pattern:**
```python
while not done:
    # 1. Send message to LLM
    response = llm.create(messages, tools)

    # 2. Check if tools needed
    if response.tool_calls:
        # 3. Execute each tool
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call)

        # 4. Add results to conversation
        messages.append(tool_results)
    else:
        # 5. Done - return final answer
        return response.content
```

### Error Handling Strategies

**1. Validation at Entry:**
```python
if not valid_input(args):
    return error_response("Invalid input")
```

**2. Try-Except Blocks:**
```python
try:
    result = risky_operation()
except SpecificError as e:
    return friendly_error(e)
```

**3. Structured Responses:**
```python
{
    "success": True/False,
    "result": "..." if success,
    "error": "..." if failure,
    "error_type": "..." for debugging
}
```

**4. Fallback Mechanisms:**
```python
result = primary_method()
if not result.success:
    result = backup_method()
```

### Multi-Agent Design Patterns

**1. Router Pattern:**
- Central router agent
- Specialized sub-agents
- Query classification
- Route to appropriate specialist

**2. Pipeline Pattern:**
- Sequential agent chain
- Each agent processes and passes forward
- Validation at each step
- Final agent produces output

**3. Coordinator Pattern:**
- Orchestrator agent
- Parallel sub-agents
- Result aggregation
- Conflict resolution

---

## 🎯 Best Practices

### Tool Design

✅ **Write Clear Descriptions:**
```python
# Good
"description": "Performs mathematical calculations. Use for arithmetic operations, percentages, or math problems."

# Bad
"description": "Does math"
```

✅ **Include Keywords:**
```python
# Include terms users might use
"description": "... Use when user asks about 'calculate', 'compute', 'math', percentages, or numerical operations."
```

✅ **Specify Parameters Clearly:**
```python
"expression": {
    "type": "string",
    "description": "Mathematical expression to evaluate, e.g., '25 * 4 + 10'"
}
```

✅ **Provide Examples:**
Include usage examples in descriptions to guide the LLM.

### Error Handling

✅ **Validate Early:**
Check inputs before expensive operations.

✅ **Return Structured Errors:**
Use consistent error format across all tools.

✅ **Log for Debugging:**
```python
logger.error(f"Tool {name} failed: {error}")
```

✅ **User-Friendly Messages:**
Transform technical errors into helpful messages.

### Agent Architecture

✅ **Single Responsibility:**
Each agent/tool should do one thing well.

✅ **Composable Tools:**
Design tools that can work together.

✅ **Iteration Limits:**
Always set max iterations to prevent infinite loops.

✅ **Timeout Handling:**
Set timeouts for LLM and tool calls.

### Security

✅ **Input Sanitization:**
```python
# Don't allow dangerous operations
forbidden = ['import', 'exec', 'eval', '__']
if any(f in input for f in forbidden):
    return error("Forbidden operation")
```

✅ **Least Privilege:**
Tools should have minimum necessary permissions.

✅ **API Key Protection:**
Never log or expose API keys.

✅ **Rate Limiting:**
Implement rate limits for tool usage.

---

## 🔧 Troubleshooting

### Issue: Tool Not Being Called

**Problem:** LLM responds directly instead of using tool.

**Solutions:**
1. Improve tool description with keywords
2. Add examples to description
3. Use system prompt to encourage tool use:
   ```python
   "You have access to tools. Use them when appropriate."
   ```

### Issue: Wrong Tool Selected

**Problem:** LLM chooses incorrect tool.

**Solutions:**
1. Make tool descriptions more distinct
2. Add usage guidelines to descriptions
3. Include negative examples: "Do NOT use for..."
4. Refine routing logic in multi-agent systems

### Issue: Agent Loops Indefinitely

**Problem:** Agent keeps making tool calls without completing.

**Solutions:**
1. Set `max_iterations` limit (5-10 typical)
2. Add loop detection logic
3. Check tool results for success/failure
4. Ensure tools return clear completion signals

### Issue: Errors Not Handled

**Problem:** Uncaught exceptions crash agent.

**Solutions:**
1. Wrap tool execution in try-except
2. Return structured error responses
3. Add timeout to prevent hanging
4. Log errors for debugging:
   ```python
   logging.error(f"Tool execution failed: {e}")
   ```

### Issue: Poor Performance

**Problem:** Slow agent responses.

**Solutions:**
1. Use faster models (gpt-4o-mini vs gpt-4)
2. Reduce max_tokens
3. Cache repeated tool results
4. Parallelize independent tool calls
5. Set appropriate timeouts

---

## 📊 Performance Tips

### Speed Optimization

**Use Faster Models:**
- Development: `gpt-4o-mini` (fast, cheap)
- Production complex: `gpt-4o` or `claude-sonnet`

**Parallel Tool Execution:**
```python
# If tools are independent, execute in parallel
with ThreadPoolExecutor() as executor:
    results = executor.map(execute_tool, tool_calls)
```

**Caching:**
```python
@lru_cache(maxsize=100)
def cached_tool(query):
    return expensive_operation(query)
```

### Cost Optimization

**Token Management:**
- Keep tool descriptions concise
- Limit conversation history length
- Use streaming for long responses

**Model Selection:**
- Simple tasks: gpt-4o-mini ($0.15/1M input tokens)
- Complex reasoning: gpt-4o ($2.50/1M input tokens)

**Batching:**
Process multiple queries in batch when possible.

### Quality Optimization

**Better Prompts:**
- Clear system instructions
- Explicit tool usage guidelines
- Example conversations

**Tool Design:**
- Granular tools (specific purposes)
- Consistent naming conventions
- Rich metadata in responses

**Testing:**
- Unit test each tool
- Integration test agent workflows
- Monitor success rates

---

## 🧪 Testing

### Validate Syntax

```bash
cd solutions/
python3 -m py_compile exercise1_calculator_agents.py
python3 -m py_compile all_exercises.py
```

### Run Tests

```bash
# Test individual exercise
python exercise1_calculator_agents.py

# Test all exercises
python all_exercises.py
```

### Manual Testing Checklist

- [ ] Calculator works for basic math
- [ ] Calculator handles division by zero
- [ ] Multi-tool agent selects correct tools
- [ ] Conditional agent checks business hours
- [ ] Resilient agent handles errors gracefully
- [ ] AgentHub routes to correct agents
- [ ] All agents provide clear responses

---

## 📚 Next Steps

After completing Lab 6:
1. **Lab 7:** Agent Memory & Planning (ReAct pattern, task decomposition)
2. **Lab 8:** Advanced Multi-Agent Systems (coordination, frameworks)
3. Build custom agents for your domain
4. Deploy agents to production

---

## 🎓 What You've Learned

✅ Tool/function calling with OpenAI and Claude
✅ Multi-tool agents with intelligent selection
✅ Conditional workflows and branching logic
✅ Production error handling and resilience
✅ Multi-agent systems with routing
✅ Agent design patterns and best practices
✅ Security and validation in agent systems
✅ Performance optimization strategies

---

## 🌟 Advanced Topics

### Tool Chaining

Execute tools in sequence where output of one feeds into another:

```python
# Get data → Process → Store → Notify
result1 = fetch_data()
result2 = process(result1)
result3 = store(result2)
notify("Complete")
```

### Dynamic Tool Loading

Load tools at runtime based on context:

```python
available_tools = load_tools_for_user(user_id)
agent = Agent(tools=available_tools)
```

### Tool Versioning

Handle multiple versions of tools:

```python
tools = {
    "calculator_v1": old_calculator,
    "calculator_v2": new_calculator
}
```

### Observability

Track agent behavior:

```python
def log_tool_use(tool_name, args, result):
    logger.info({
        "tool": tool_name,
        "args": args,
        "success": result.get("success"),
        "latency": elapsed_time
    })
```

---

## 📖 Additional Resources

### Documentation
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Claude Tool Use Documentation](https://docs.anthropic.com/claude/docs/tool-use)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)

### Further Reading
- ReAct: Reasoning and Acting in LLMs
- Agent design patterns
- Production deployment best practices
- Multi-agent coordination strategies

---

**Ready for Production! 🚀**

*You now have the skills to build sophisticated AI agents with tool calling for real-world applications.*
