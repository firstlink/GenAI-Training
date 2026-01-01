# Lab 8 Solutions: Advanced Multi-Agent Systems

## 📚 Overview

Complete solutions for Lab 8 covering advanced multi-agent systems, research agents, agentic RAG, LangChain framework integration, and production-ready research platforms. This capstone lab combines all concepts from the Advanced GenAI Training course.

---

## 📁 Files Included

### Solutions
- **`all_exercises.py`** - Consolidated solution with all concepts (1200+ lines)

### Configuration
- **`.env.example`** - Environment configuration template
- **`README.md`** - This comprehensive guide

---

## 🚀 Quick Start

### Prerequisites

```bash
# Core dependencies
pip install openai chromadb sentence-transformers python-dotenv

# Optional: LangChain (for Exercise 3)
pip install langchain langchain-openai
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

# Optional
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Run Solutions

```bash
# Run all exercises (interactive menu)
python all_exercises.py

# The script will prompt you to select which demonstration to run
```

---

## 📖 What's Covered

### Exercise 1: Research Agent

**Concepts:**
- Autonomous information gathering
- Multi-step research workflows
- Result synthesis and reporting
- Configurable research depth

**Implementation:**

```python
from all_exercises import ResearchAgent

agent = ResearchAgent()

# Agent autonomously searches and synthesizes information
result = agent.research(
    "machine learning applications",
    depth="moderate",  # brief, moderate, or comprehensive
    verbose=True
)

print(f"Report: {result['report']}")
print(f"Searches performed: {result['searches_performed']}")
```

**Research Flow:**
```
User Query
    ↓
Iteration 1: Search for general information
    ↓
Iteration 2: Search for specific details
    ↓
Iteration 3: Gather diverse perspectives
    ↓
Iteration N: Synthesize findings
    ↓
Generate Report
```

**Features:**
- **Web Search Tool**: Simulated web search (easily replaceable with real APIs)
- **Depth Control**: Brief (3 iterations), Moderate (5), Comprehensive (8)
- **Search History**: Tracks all searches performed
- **Automatic Synthesis**: LLM synthesizes findings into coherent report

**Example Output:**
```
RESEARCH AGENT: machine learning applications
==================================================

--- Research Iteration 1 ---
🔧 web_search({"query": "machine learning applications"})
   Found 3 results

--- Research Iteration 2 ---
🔧 web_search({"query": "modern ML use cases 2024"})
   Found 2 results

✅ RESEARCH COMPLETE
==================================================

Machine learning has numerous applications across industries:

1. Healthcare: Diagnostic systems and drug discovery
2. Autonomous Vehicles: Self-driving technology
3. Recommendation Systems: Personalized content
4. Natural Language Processing: Translation and chatbots

[Full synthesized report...]
```

---

### Exercise 2: Agentic RAG System

**Concepts:**
- Agent-controlled retrieval decisions
- Dynamic vs static RAG
- Multi-hop reasoning
- Retrieval tracking

**Traditional RAG vs Agentic RAG:**

| Feature | Traditional RAG | Agentic RAG |
|---------|----------------|-------------|
| **Retrieval** | Always retrieves | Decides when to retrieve |
| **Query Understanding** | Fixed query | Can reformulate queries |
| **Multi-hop** | Single retrieval | Multiple retrievals possible |
| **Efficiency** | May retrieve unnecessarily | Only retrieves when needed |

**Implementation:**

```python
from all_exercises import KnowledgeBase, AgenticRAG

# Setup knowledge base
kb = KnowledgeBase()

# Create agentic RAG system
rag = AgenticRAG(kb)

# Test 1: General knowledge (NO retrieval)
result = rag.answer_question("What is 25 * 4?")
# Agent recognizes it knows the answer, doesn't retrieve
# Retrievals: 0

# Test 2: KB-specific (DOES retrieve)
result = rag.answer_question(
    "According to the knowledge base, what is supervised learning?"
)
# Agent recognizes need for specific KB info, retrieves
# Retrievals: 1+

# Test 3: Multi-hop (MULTIPLE retrievals)
result = rag.answer_question(
    "Compare machine learning and deep learning based on the KB"
)
# Agent retrieves info about both topics
# Retrievals: 2+
```

**Agentic Decision Flow:**
```
User Question
    ↓
Agent Analyzes Question
    ↓
  ┌─────┴─────┐
  ↓           ↓
General     Specific
Knowledge   KB Info
  ↓           ↓
Answer      Retrieve
Directly    Documents
            ↓
            Answer with
            Context
```

**Features:**
- **Smart Retrieval**: Only retrieves when necessary
- **Multi-Hop**: Can retrieve multiple times for complex questions
- **Semantic Search**: Vector similarity for relevant documents
- **Retrieval Logging**: Tracks what was retrieved and why

**Example with Retrieval:**
```
AGENTIC RAG
==================================================
Question: According to the knowledge base, what is supervised learning?

--- Iteration 1 ---
💭 Agent: I need to retrieve information from the knowledge base

📚 Retrieving: 'supervised learning'
   [1] supervised_learning (relevance: 0.892)
   [2] machine_learning (relevance: 0.734)

--- Iteration 2 ---

✅ ANSWER:
==================================================
According to the knowledge base, supervised learning trains
models on labeled data, learning to map inputs to outputs.
Common applications include classification and regression tasks.

Retrievals: 1
```

---

### Exercise 3: LangChain Agent

**Concepts:**
- Framework-based agent development
- Tool decorators
- AgentExecutor pattern
- Rapid prototyping

**Why LangChain?**
- **Pre-built Patterns**: Common agent architectures ready to use
- **Tool Integration**: Easy tool creation with decorators
- **Error Handling**: Built-in parsing and error management
- **Ecosystem**: Large collection of integrations

**Key Components:**

```python
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI

# 1. Define tools with decorator
@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    return str(eval(expression))

@tool
def text_analyzer(text: str) -> str:
    """Analyze text statistics."""
    words = len(text.split())
    return f"Words: {words}"

# 2. Create LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# 3. Create agent
tools = [calculator, text_analyzer]
agent = create_tool_calling_agent(llm, tools, prompt)

# 4. Create executor
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Use agent
result = executor.invoke({"input": "What is 15% of 340?"})
```

**Benefits:**
- ✅ Rapid development with pre-built patterns
- ✅ Automatic tool schema generation
- ✅ Built-in error handling
- ✅ Verbose logging for debugging
- ✅ Extensive ecosystem of tools

**Note:** Exercise 3 in the consolidated solution provides conceptual overview. For full LangChain implementation, see lab materials.

---

### Exercise 4: Multi-Agent System

**Concepts:**
- Specialized agent roles
- Coordinator pattern
- Agent-to-agent collaboration
- Result synthesis

**Architecture:**

```
    CoordinatorAgent
           |
    ┌──────┼──────┐
    ↓      ↓      ↓
Researcher Analyst Writer
    |      |      |
  Search  Analyze Create
  Info   Patterns Content
```

**Agent Specializations:**

| Agent | Role | Temperature | Use Cases |
|-------|------|-------------|-----------|
| **Researcher** | Gather information | 0.0 | Finding facts, sources |
| **Analyst** | Identify patterns | 0.0 | Data analysis, insights |
| **Writer** | Create content | 0.7 | Articles, summaries |
| **Coordinator** | Delegate tasks | 0.0 | Task routing, synthesis |

**Implementation:**

```python
from all_exercises import MultiAgentSystem

system = MultiAgentSystem()

# System automatically delegates to appropriate agents
result = system.execute_task(
    "Research multi-agent AI systems, analyze benefits, and write a summary",
    verbose=True
)
```

**Execution Flow:**

```
Task: "Research X, analyze Y, write Z"
    ↓
Coordinator analyzes task
    ↓
Delegates to: researcher, analyst, writer
    ↓
┌────────────┬────────────┬────────────┐
│ Researcher │  Analyst   │   Writer   │
│  executes  │  executes  │  executes  │
└────────────┴────────────┴────────────┘
    ↓           ↓           ↓
  Result 1   Result 2   Result 3
    └───────────┴───────────┘
              ↓
    Coordinator synthesizes
              ↓
        Final Result
```

**Coordinator Logic:**

```python
class CoordinatorAgent:
    def delegate(self, task: str) -> Dict[str, str]:
        # 1. Analyze task requirements
        # 2. Determine which agents needed
        # 3. Execute with each agent
        # 4. Collect results

        if "research" in task:
            use researcher
        if "analyze" in task:
            use analyst
        if "write" in task:
            use writer
```

**Features:**
- **Automatic Delegation**: Coordinator selects appropriate agents
- **Parallel Execution**: Agents can work independently
- **Result Synthesis**: Coordinator combines outputs coherently
- **Specialization**: Each agent optimized for specific tasks

---

### Capstone: ResearchHub v1.0

**Complete Production Research Platform**

Combines ALL concepts:
- ✅ Research agents
- ✅ Agentic RAG
- ✅ Multi-agent collaboration
- ✅ Vector database
- ✅ Advanced workflows

**Architecture:**

```
      ResearchHub v1.0
            |
    ┌───────┼───────┐
    ↓       ↓       ↓
  Tools   Memory  Agents
    |       |       |
Web Search  KB    Research
Retrieval  Vector  Analysis
Analysis   DB     Synthesis
```

**Features:**

**1. Multiple Tool Integration:**
```python
hub = ResearchHub()

# Available tools:
# - web_search: Search for information
# - retrieve_documents: Query knowledge base
# - analyze_findings: Analyze research results
```

**2. Configurable Research Modes:**
```python
# Quick research (3 iterations)
hub.conduct_research("AI agents", mode="quick")

# Standard research (5 iterations)
hub.conduct_research("Agentic RAG", mode="standard")

# Comprehensive research (8 iterations)
hub.conduct_research("Multi-agent systems", mode="comprehensive")
```

**3. Research History Tracking:**
```python
# All research is tracked
stats = hub.get_stats()
# {
#   "total_research_conducted": 5,
#   "kb_documents": 7,
#   "tools_available": 3
# }
```

**4. Production Features:**
- ✅ Logging with Python logging module
- ✅ Error handling throughout
- ✅ Persistent storage with ChromaDB
- ✅ Structured result format
- ✅ Timestamped research history
- ✅ Configurable iteration limits

**Usage Example:**

```python
from all_exercises import ResearchHub

# Initialize platform
hub = ResearchHub()

# Conduct research
result = hub.conduct_research(
    "Multi-agent AI systems",
    mode="standard",
    verbose=True
)

# Access results
print(result['report'])
print(f"Iterations: {result['iterations']}")
print(f"Timestamp: {result['timestamp']}")

# Get platform stats
stats = hub.get_stats()
print(f"Total research: {stats['total_research_conducted']}")
```

**Research Workflow:**

```
Topic Input
    ↓
Initialize Research Context
    ↓
┌─────────────────────┐
│  Iteration Loop     │
│  ┌──────────────┐   │
│  │ Web Search   │   │
│  │ KB Retrieval │   │
│  │ Analysis     │   │
│  └──────────────┘   │
└─────────────────────┘
    ↓
Synthesize Report
    ↓
Store in History
    ↓
Return Results
```

---

## 💡 Key Concepts

### Research Agent Design

**Core Components:**
1. **Tool Integration**: Web search, databases, APIs
2. **Iteration Control**: Configurable depth
3. **Result Synthesis**: Coherent report generation
4. **History Tracking**: Search audit trail

**Best Practices:**
```python
# ✅ Good: Configurable depth
agent.research(topic, depth="moderate")

# ✅ Good: Track searches
self.search_history.append(result)

# ✅ Good: Limit iterations
max_iterations = {"brief": 3, "moderate": 5}[depth]
```

### Agentic RAG Patterns

**Decision Points:**

```python
def should_retrieve(question: str) -> bool:
    """Agent decides if retrieval is needed"""

    # General knowledge → No retrieval
    if is_general_knowledge(question):
        return False

    # KB-specific → Retrieve
    if mentions_knowledge_base(question):
        return True

    # Complex → Multiple retrievals
    if is_multi_hop_question(question):
        return True
```

**Multi-Hop Retrieval:**
```
Question: "Compare A and B"
    ↓
Retrieve info about A
    ↓
Retrieve info about B
    ↓
Analyze both
    ↓
Generate comparison
```

### Multi-Agent Coordination

**Coordinator Pattern:**

```python
class Coordinator:
    def delegate(self, task):
        # 1. Parse task requirements
        required_agents = self.analyze_task(task)

        # 2. Execute with agents
        results = {}
        for agent_name in required_agents:
            results[agent_name] = self.agents[agent_name].execute(task)

        # 3. Synthesize
        return self.synthesize(results)
```

**Agent Specialization Benefits:**
- **Expertise**: Each agent optimized for specific domain
- **Modularity**: Easy to add/remove agents
- **Scalability**: Can run agents in parallel
- **Quality**: Specialized models/temperatures per role

---

## 🎯 Best Practices

### Research Agent

✅ **Set Appropriate Iteration Limits:**
```python
# Development: More iterations for thoroughness
agent.research(topic, depth="comprehensive")  # 8 iterations

# Production: Fewer iterations for speed
agent.research(topic, depth="brief")  # 3 iterations
```

✅ **Track and Log Searches:**
```python
class ResearchTools:
    def __init__(self):
        self.search_history = []  # Audit trail

    def web_search(self, query):
        result = perform_search(query)
        self.search_history.append(result)  # Track
        return result
```

✅ **Use Real Search APIs in Production:**
```python
# Development: Simulated search
def web_search(query):
    return simulated_results

# Production: Real API
def web_search(query):
    return tavily_api.search(query)
    # or serpapi.search(query)
    # or brave_search.query(query)
```

### Agentic RAG

✅ **Clear Retrieval Instructions:**
```python
system_prompt = """
CRITICAL: Only retrieve when you need specific information
from the knowledge base. If you already know the answer,
respond directly without retrieval.
"""
```

✅ **Track Retrieval Decisions:**
```python
self.retrieval_log = [
    {"query": "supervised learning", "reason": "KB-specific"},
    {"query": "deep learning", "reason": "multi-hop"}
]
```

✅ **Optimize Retrieval Count:**
```python
# Too many documents → noise
retrieve_documents(query, n_results=10)  # ❌

# Optimal → relevant results
retrieve_documents(query, n_results=2-3)  # ✅
```

### Multi-Agent Systems

✅ **Clear Agent Roles:**
```python
class ResearcherAgent:
    role = "gathering information and finding sources"

class AnalystAgent:
    role = "analyzing data and identifying patterns"
```

✅ **Appropriate Temperatures:**
```python
# Factual tasks: Low temperature
ResearcherAgent(temperature=0.0)  # ✅

# Creative tasks: Higher temperature
WriterAgent(temperature=0.7)  # ✅
```

✅ **Result Synthesis:**
```python
def synthesize(results: Dict[str, str]) -> str:
    """Combine agent outputs coherently"""
    # Don't just concatenate
    # Use LLM to synthesize meaningfully
    prompt = f"Synthesize these results: {results}"
    return llm.generate(prompt)
```

---

## 🔧 Troubleshooting

### Issue: Research Agent Not Searching

**Problem:** Agent returns answer without using search tool.

**Solutions:**
1. Improve system prompt:
   ```python
   "You MUST use web_search to find information. Do not rely on general knowledge."
   ```
2. Use `tool_choice="required"` to force tool use
3. Check tool description clarity

### Issue: Agentic RAG Always Retrieves

**Problem:** Agent retrieves even for simple questions.

**Solutions:**
1. Strengthen non-retrieval instruction:
   ```python
   "ONLY retrieve when you need specific KB information you don't know"
   ```
2. Add examples of when NOT to retrieve
3. Check if retrieval threshold is too low

### Issue: Multi-Agent System Uses Wrong Agent

**Problem:** Coordinator delegates to inappropriate agent.

**Solutions:**
1. Improve agent role descriptions:
   ```python
   "researcher: ONLY for finding new information, NOT for analysis"
   ```
2. Add negative examples: "Do NOT use researcher for writing"
3. Make delegation prompt more explicit

### Issue: ChromaDB Persistence Errors

**Problem:** Database files corrupted or locked.

**Solutions:**
1. Delete database directory:
   ```bash
   rm -rf ./agentic_rag_db ./researchhub_db
   ```
2. Restart Python process
3. Check file permissions

### Issue: Too Many API Calls

**Problem:** Exceeding rate limits or budget.

**Solutions:**
1. Reduce iteration limits:
   ```python
   max_iterations = 3  # Instead of 8
   ```
2. Add caching:
   ```python
   @lru_cache(maxsize=100)
   def web_search(query):
       ...
   ```
3. Use cheaper model:
   ```python
   model="gpt-4o-mini"  # Instead of gpt-4o
   ```

---

## 📊 Performance Tips

### Speed Optimization

**Reduce Iterations:**
```python
# Development
mode = "comprehensive"  # 8 iterations

# Production
mode = "quick"  # 3 iterations
```

**Parallel Agent Execution:**
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    results = {
        name: executor.submit(agent.execute, task)
        for name, agent in agents.items()
    }
```

**Cache Results:**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def retrieve_documents(query: str, n_results: int = 2):
    # Expensive operation cached
    pass
```

### Cost Optimization

**Use Appropriate Models:**
```python
# Simple tasks
model = "gpt-4o-mini"  # $0.15/1M tokens

# Complex reasoning
model = "gpt-4o"  # $2.50/1M tokens
```

**Limit Context Size:**
```python
# Expensive: Include all history
messages = all_messages

# Optimized: Recent messages only
messages = recent_messages[-10:]
```

**Batch Operations:**
```python
# Process multiple queries in one session
for query in queries:
    results.append(process(query))
```

### Quality Optimization

**Better Prompts:**
```python
# Vague
"Research this topic"

# Specific
"Research this topic. Find 3-5 key facts, identify trends, cite sources."
```

**Specialized Agents:**
```python
# Generic agent for everything
GenericAgent()

# Specialized agents for specific tasks
ResearcherAgent(), AnalystAgent(), WriterAgent()
```

**Validation:**
```python
def validate_research(result):
    """Ensure quality of research output"""
    if len(result['report']) < 100:
        return False  # Too brief
    if result['searches_performed'] == 0:
        return False  # No research done
    return True
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
# 1. Research Agent
# 2. Agentic RAG System
# 3. LangChain Agent (Info)
# 4. Multi-Agent System
# 5. ResearchHub v1.0
# 6. Run All
```

### Manual Testing Checklist

- [ ] Research agent performs multiple searches
- [ ] Agentic RAG skips retrieval for general knowledge
- [ ] Agentic RAG retrieves for KB-specific questions
- [ ] Multi-agent coordinator delegates correctly
- [ ] ResearchHub completes full research cycle
- [ ] Knowledge base persists across runs
- [ ] All agents respect iteration limits
- [ ] Error handling works for API failures

---

## 📚 Architecture Patterns

### Pattern 1: Research Agent Loop

```python
def research_loop(topic, max_iterations):
    """Iterative research pattern"""
    messages = [initial_prompt(topic)]

    for i in range(max_iterations):
        response = llm.generate(messages)

        if no_more_searches_needed(response):
            return synthesize_report(response)

        # Execute searches
        for search in response.tool_calls:
            result = execute_search(search)
            messages.append(result)

    return final_report(messages)
```

### Pattern 2: Agentic Retrieval

```python
def agentic_retrieval(question):
    """Agent decides when to retrieve"""
    messages = [
        system_prompt_with_retrieval_rules,
        user_question
    ]

    while not done:
        response = llm.generate(messages, tools=[retrieve])

        if response.uses_retrieval_tool:
            # Agent decided to retrieve
            docs = retrieve(response.query)
            messages.append(docs)
        else:
            # Agent has answer
            return response.content
```

### Pattern 3: Multi-Agent Coordination

```python
def coordinate_agents(task, agents):
    """Coordinator pattern"""
    # 1. Analyze task
    required_agents = determine_agents_needed(task)

    # 2. Execute with each agent
    results = {}
    for agent_name in required_agents:
        results[agent_name] = agents[agent_name].execute(task)

    # 3. Synthesize
    return synthesize_results(results)
```

---

## 🎓 What You've Learned

✅ Research agent architectures and implementation
✅ Agentic RAG vs traditional RAG patterns
✅ LangChain framework for rapid development
✅ Multi-agent coordination and collaboration
✅ Tool integration and function calling
✅ Production deployment strategies
✅ Vector database integration
✅ Agent specialization patterns
✅ Result synthesis techniques
✅ Complete research platform architecture

---

## 🌟 Advanced Topics

### Real-Time Collaboration

Multiple agents working simultaneously:

```python
class RealtimeMultiAgent:
    def collaborate(self, task):
        # All agents work in parallel
        with ThreadPoolExecutor() as executor:
            futures = {
                name: executor.submit(agent.execute, task)
                for name, agent in self.agents.items()
            }

            results = {
                name: future.result()
                for name, future in futures.items()
            }

        return self.synthesize(results)
```

### Hierarchical Agent Systems

Agents managing other agents:

```python
class HierarchicalSystem:
    def __init__(self):
        # Top level coordinator
        self.main_coordinator = MainCoordinator()

        # Team coordinators
        self.research_team = ResearchTeam()
        self.analysis_team = AnalysisTeam()

        # Specialist agents under each team
        self.research_team.add_agents([...])
        self.analysis_team.add_agents([...])
```

### Self-Improving Agents

Agents that learn from feedback:

```python
class SelfImprovingAgent:
    def execute_with_learning(self, task):
        # Execute task
        result = self.execute(task)

        # Self-evaluate
        score = self.evaluate(result)

        # If poor quality, retry with improvements
        if score < threshold:
            feedback = self.generate_feedback(result)
            result = self.retry_with_feedback(task, feedback)

        # Store successful patterns
        self.memory.store_successful_pattern(task, result)

        return result
```

### Agent Communication Protocol

Structured messaging between agents:

```python
class AgentMessage:
    def __init__(self, sender, recipient, message_type, content):
        self.sender = sender
        self.recipient = recipient
        self.type = message_type  # request, response, notification
        self.content = content
        self.timestamp = datetime.now()

class AgentCommunicator:
    def send_message(self, message: AgentMessage):
        """Route message to recipient agent"""
        recipient = self.agents[message.recipient]
        response = recipient.handle_message(message)
        return response
```

---

## 📖 Additional Resources

### Documentation
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [LangChain Documentation](https://python.langchain.com/docs)
- [ChromaDB Documentation](https://docs.trychroma.com)

### Further Reading
- Multi-agent systems and coordination
- Agentic RAG architectures
- Production AI deployment
- Agent frameworks comparison

### Research Papers
- "ReAct: Synergizing Reasoning and Acting in Language Models"
- "Toolformer: Language Models Can Teach Themselves to Use Tools"
- "Communicative Agents for Software Development"

---

## 📊 Comparison: System Capabilities

| Capability | Research Agent | Agentic RAG | Multi-Agent | ResearchHub |
|-----------|----------------|-------------|-------------|-------------|
| Web Search | ✅ | ❌ | ❌ | ✅ |
| KB Retrieval | ❌ | ✅ | ❌ | ✅ |
| Smart Retrieval | ❌ | ✅ | ❌ | ✅ |
| Multi-Agent | ❌ | ❌ | ✅ | ✅ |
| Specialization | ❌ | ❌ | ✅ | ✅ |
| Research Depth | ✅ | ❌ | ❌ | ✅ |
| History Tracking | ✅ | ✅ | ❌ | ✅ |
| Production Ready | ⚠️ | ⚠️ | ⚠️ | ✅ |

---

**Production Ready! 🚀**

*Congratulations! You've completed all 8 labs and mastered advanced multi-agent AI systems. You're now ready to build production AI applications!*
