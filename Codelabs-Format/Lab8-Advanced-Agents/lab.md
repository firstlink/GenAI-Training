# 🛠️ Lab 8: Advanced Multi-Agent Systems - Hands-On Lab

> **Duration:** 120-150 minutes
> **Difficulty:** Advanced
> **Prerequisites:** Labs 6-7 completed, Python environment set up

---

## 📋 Lab Overview

In this capstone hands-on lab, you'll build increasingly sophisticated multi-agent systems, culminating in a **production-ready intelligent research platform** that combines research agents, agentic RAG, frameworks, and multi-agent collaboration.

### What You'll Build

By the end of this lab, you'll have created:

1. ✅ **Research Agent** - Autonomous information gathering system
2. ✅ **Agentic RAG System** - Smart retrieval with decision-making
3. ✅ **LangChain Agent** - Framework-powered agent
4. ✅ **Multi-Agent System** - Collaborative agent team
5. ✅ **ResearchHub v1.0** - Production multi-agent research platform (Capstone)

---

## 🎯 Learning Objectives

By completing this lab, you will:
- ✓ Build research agents that gather and synthesize information
- ✓ Implement agentic RAG with dynamic retrieval
- ✓ Use LangChain framework for rapid agent development
- ✓ Create multi-agent systems with specialization
- ✓ Implement agent-to-agent communication
- ✓ Combine all concepts into production systems

---

## 📂 Setup

### Step 1: Create Lab Directory

```bash
mkdir lab8_advanced_agents
cd lab8_advanced_agents
```

### Step 2: Install Dependencies

```bash
pip install openai anthropic python-dotenv
pip install chromadb sentence-transformers
pip install langchain langchain-openai langgraph
pip install requests beautifulsoup4
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
print("✓ Anthropic key loaded" if os.getenv('ANTHROPIC_API_KEY') else "✗ Anthropic key missing")

# Test OpenAI
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

# Test LangChain
try:
    from langchain_openai import ChatOpenAI
    print("✓ LangChain installed")
except Exception as e:
    print(f"✗ LangChain error: {e}")

print("\n✅ Setup complete! Ready for exercises.")
```

Run: `python test_setup.py`

---

## 🔧 Exercise 1: Research Agent (25-30 minutes)

**Goal:** Build an autonomous research agent that searches, retrieves, and synthesizes information.

Create `exercise1_research_agent.py`:

```python
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from datetime import datetime

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ==================== RESEARCH TOOLS ====================

class ResearchTools:
    """Tools for research agent"""

    def __init__(self):
        self.search_history = []

    def web_search(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        """
        Simulated web search (in production, use real search API)
        """
        print(f"🔍 Searching: '{query}'")

        # Simulated knowledge base
        knowledge = {
            "machine learning": [
                {
                    "title": "Machine Learning Fundamentals",
                    "snippet": "ML enables systems to learn from data. Key types: supervised, unsupervised, and reinforcement learning.",
                    "source": "ML Guide"
                },
                {
                    "title": "ML Applications 2024",
                    "snippet": "Modern ML powers healthcare diagnostics, autonomous vehicles, and recommendation systems.",
                    "source": "Tech Review"
                }
            ],
            "climate change": [
                {
                    "title": "Climate Science Consensus",
                    "snippet": "97% of scientists agree climate change is human-caused, primarily through CO2 emissions.",
                    "source": "IPCC"
                },
                {
                    "title": "Climate Impact on Ecosystems",
                    "snippet": "Rising temperatures affect biodiversity, oceans, and weather patterns globally.",
                    "source": "Nature"
                }
            ],
            "renewable energy": [
                {
                    "title": "Solar Power Advances",
                    "snippet": "Solar efficiency reached 25% with perovskite materials, reducing costs significantly.",
                    "source": "Energy Institute"
                },
                {
                    "title": "Wind Energy Growth",
                    "snippet": "Wind capacity grew 15% globally in 2023, with offshore wind leading expansion.",
                    "source": "Renewable News"
                }
            ]
        }

        # Find matching results
        query_lower = query.lower()
        results = []

        for key, articles in knowledge.items():
            if key in query_lower or any(word in query_lower for word in key.split()):
                results.extend(articles[:num_results])

        if not results:
            results = [{
                "title": f"General info about {query}",
                "snippet": "Information available from various sources.",
                "source": "Web"
            }]

        result_data = {
            "query": query,
            "num_results": len(results[:num_results]),
            "results": results[:num_results],
            "timestamp": datetime.now().isoformat()
        }

        self.search_history.append(result_data)
        return result_data

# ==================== RESEARCH AGENT ====================

class ResearchAgent:
    """Autonomous research agent"""

    def __init__(self):
        self.tools = ResearchTools()
        self.findings = []

        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information. Returns relevant articles.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results (default: 3)",
                                "default": 3
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        self.functions = {
            "web_search": self.tools.web_search
        }

    def research(self, topic: str, depth: str = "moderate") -> Dict[str, Any]:
        """
        Conduct research on a topic

        Args:
            topic: Research topic
            depth: "brief", "moderate", or "comprehensive"

        Returns:
            Research report
        """
        print(f"\n{'='*70}")
        print(f"RESEARCH AGENT: {topic}")
        print('='*70)

        # Determine iterations based on depth
        max_iterations = {"brief": 3, "moderate": 5, "comprehensive": 8}[depth]

        system_prompt = f"""You are an expert research agent. Your task is to research this topic: {topic}

Guidelines:
1. Search for information using web_search tool
2. Gather diverse perspectives
3. Identify key facts and insights
4. Track sources
5. Synthesize findings into a coherent report

For {depth} research, make {max_iterations} iterations maximum."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research this topic: {topic}"}
        ]

        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Research Iteration {iteration} ---")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tool_definitions,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            # Check if done
            if not response_message.tool_calls:
                final_report = response_message.content

                print(f"\n{'='*70}")
                print("✅ RESEARCH COMPLETE")
                print('='*70)
                print(final_report)

                return {
                    "topic": topic,
                    "report": final_report,
                    "searches_performed": len(self.tools.search_history),
                    "iterations": iteration
                }

            # Execute tool calls
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                print(f"\n🔧 {function_name}({json.dumps(arguments)})")

                # Execute
                result = self.functions[function_name](**arguments)

                print(f"   Found {result['num_results']} results")

                # Add to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(result)
                })

        return {
            "topic": topic,
            "report": "Research incomplete - max iterations reached",
            "searches_performed": len(self.tools.search_history)
        }

# ==================== TEST ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RESEARCH AGENT DEMONSTRATION")
    print("="*70)

    agent = ResearchAgent()

    # Test 1: Brief research
    print("\n" + "#"*70)
    print("TEST 1: Brief Research")
    print("#"*70)
    result = agent.research("machine learning applications", depth="brief")
    print(f"\nSearches performed: {result['searches_performed']}")

    # Test 2: Moderate research
    print("\n" + "#"*70)
    print("TEST 2: Moderate Research")
    print("#"*70)
    agent2 = ResearchAgent()
    result = agent2.research("renewable energy trends", depth="moderate")
    print(f"\nSearches performed: {result['searches_performed']}")

    print("\n✅ Exercise 1 Complete!")
```

**Run it:** `python exercise1_research_agent.py`

### ✅ Checkpoint 1

**What you learned:**
- ✓ Research agent architecture
- ✓ Multi-step information gathering
- ✓ Result synthesis and reporting
- ✓ Dynamic search strategies

---

## 🔧 Exercise 2: Agentic RAG System (30-35 minutes)

**Goal:** Build a RAG system where the agent decides when and what to retrieve.

Create `exercise2_agentic_rag.py`:

```python
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ==================== KNOWLEDGE BASE SETUP ====================

class KnowledgeBase:
    """Vector database for Agentic RAG"""

    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./agentic_rag_db")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.collection = self.chroma_client.get_or_create_collection(
            name="agentic_rag_docs"
        )

        # Populate if empty
        if self.collection.count() == 0:
            self._populate()

    def _populate(self):
        """Populate with sample documents"""
        documents = [
            {
                "text": "Machine learning is a subset of AI that enables systems to learn from data without explicit programming. Key types include supervised, unsupervised, and reinforcement learning.",
                "metadata": {"topic": "machine_learning", "category": "basics"}
            },
            {
                "text": "Deep learning uses neural networks with multiple layers to automatically learn hierarchical representations. It excels at tasks like image recognition and natural language processing.",
                "metadata": {"topic": "deep_learning", "category": "advanced"}
            },
            {
                "text": "Supervised learning trains models on labeled data, learning to map inputs to outputs. Common applications include classification and regression tasks.",
                "metadata": {"topic": "supervised_learning", "category": "techniques"}
            },
            {
                "text": "Unsupervised learning discovers patterns in unlabeled data through techniques like clustering and dimensionality reduction.",
                "metadata": {"topic": "unsupervised_learning", "category": "techniques"}
            },
            {
                "text": "Natural language processing enables computers to understand and generate human language. Applications include translation, sentiment analysis, and chatbots.",
                "metadata": {"topic": "nlp", "category": "applications"}
            }
        ]

        for i, doc in enumerate(documents):
            embedding = self.embedding_model.encode(doc["text"])
            self.collection.add(
                documents=[doc["text"]],
                embeddings=[embedding.tolist()],
                ids=[f"doc_{i}"],
                metadatas=[doc["metadata"]]
            )

        print(f"✓ Knowledge base populated with {len(documents)} documents")

# ==================== AGENTIC RAG ====================

class AgenticRAG:
    """RAG system where agent decides when to retrieve"""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.retrieval_log = []

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_documents",
                    "description": "Retrieve relevant documents from the knowledge base. Use ONLY when you need specific information you don't know.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for documents"
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Number of documents to retrieve (default: 2)",
                                "default": 2
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        self.functions = {
            "retrieve_documents": self.retrieve_documents
        }

    def retrieve_documents(self, query: str, n_results: int = 2) -> Dict:
        """Retrieve documents from knowledge base"""
        print(f"\n📚 Retrieving: '{query}'")

        query_embedding = self.embedding_model.encode(query)

        results = self.kb.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        documents = []
        for i in range(len(results['documents'][0])):
            doc = {
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "relevance": 1 / (1 + results['distances'][0][i])
            }
            documents.append(doc)
            print(f"   [{i+1}] {doc['metadata'].get('topic')} (relevance: {doc['relevance']:.3f})")

        self.retrieval_log.append({
            "query": query,
            "num_docs": len(documents)
        })

        return {
            "success": True,
            "query": query,
            "documents": documents,
            "count": len(documents)
        }

    def answer_question(self, question: str) -> Dict:
        """Answer question using agentic RAG"""
        print(f"\n{'='*70}")
        print(f"AGENTIC RAG")
        print('='*70)
        print(f"Question: {question}\n")

        system_prompt = """You are an intelligent assistant with access to a knowledge base about machine learning and AI.

CRITICAL INSTRUCTIONS:
- If you already know the answer from general knowledge, answer directly WITHOUT retrieving
- ONLY use retrieve_documents when you need specific information from the knowledge base
- You can retrieve multiple times if needed
- Think step-by-step about whether retrieval is necessary

Examples:
- "What is 2+2?" → NO retrieval needed, answer directly
- "What is machine learning?" → Could answer from general knowledge OR retrieve for specifics
- "According to the knowledge base, what is deep learning?" → MUST retrieve"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        iteration = 0
        max_iterations = 5

        while iteration < max_iterations:
            iteration += 1
            print(f"--- Iteration {iteration} ---")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            # Check if done
            if not response_message.tool_calls:
                answer = response_message.content

                print(f"\n{'='*70}")
                print("✅ ANSWER:")
                print('='*70)
                print(answer)

                return {
                    "question": question,
                    "answer": answer,
                    "retrievals": len(self.retrieval_log),
                    "retrieval_log": self.retrieval_log
                }

            # Show agent reasoning
            if response_message.content:
                print(f"\n💭 Agent: {response_message.content}")

            messages.append(response_message)

            # Execute retrievals
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                result = self.functions[function_name](**arguments)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(result)
                })

        return {"question": question, "answer": "Max iterations reached"}

# ==================== TEST ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("AGENTIC RAG DEMONSTRATION")
    print("="*70)

    kb = KnowledgeBase()
    agentic_rag = AgenticRAG(kb)

    # Test 1: General knowledge (should NOT retrieve)
    print("\n" + "#"*70)
    print("TEST 1: General Knowledge (No Retrieval Expected)")
    print("#"*70)
    result = agentic_rag.answer_question("What is 25 * 4?")
    print(f"\nRetrievals: {result['retrievals']} (expected: 0)")

    # Test 2: KB-specific question (SHOULD retrieve)
    print("\n" + "#"*70)
    print("TEST 2: Knowledge Base Query (Retrieval Expected)")
    print("#"*70)
    agentic_rag.retrieval_log = []  # Reset log
    result = agentic_rag.answer_question("According to the knowledge base, what is supervised learning?")
    print(f"\nRetrievals: {result['retrievals']} (expected: 1+)")

    # Test 3: Multi-hop question (MULTIPLE retrievals)
    print("\n" + "#"*70)
    print("TEST 3: Multi-Hop Query (Multiple Retrievals Expected)")
    print("#"*70)
    agentic_rag.retrieval_log = []
    result = agentic_rag.answer_question("Compare machine learning and deep learning based on the knowledge base")
    print(f"\nRetrievals: {result['retrievals']} (expected: 2+)")

    print("\n✅ Exercise 2 Complete!")
```

**Run it:** `python exercise2_agentic_rag.py`

### ✅ Checkpoint 2

**What you learned:**
- ✓ Agentic RAG decision-making
- ✓ Dynamic retrieval strategies
- ✓ Multi-hop reasoning
- ✓ Retrieval vs general knowledge

---

## 🔧 Exercise 3: LangChain Framework Agent (20-25 minutes)

**Goal:** Build agents using the LangChain framework.

Create `exercise3_langchain_agent.py`:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ==================== DEFINE TOOLS ====================

@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations. Input should be a math expression like '25 * 4'."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def text_analyzer(text: str) -> str:
    """Analyze text and return statistics: word count, character count, sentence count."""
    words = len(text.split())
    chars = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')

    return f"""Text Analysis:
• Words: {words}
• Characters: {chars}
• Sentences: {sentences}
• Avg word length: {chars/words:.1f} chars"""

@tool
def knowledge_search(query: str) -> str:
    """Search knowledge base for information on AI and ML topics."""
    knowledge = {
        "langchain": "LangChain is a framework for developing LLM-powered applications.",
        "agents": "Agents are autonomous systems that use LLMs to reason and act.",
        "rag": "RAG combines LLMs with external knowledge retrieval.",
        "tools": "Tools extend agent capabilities with functions and APIs."
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return f"**{key.title()}**: {value}"

    return "No information found."

# ==================== CREATE LANGCHAIN AGENT ====================

def create_langchain_agent():
    """Create a LangChain agent with tools"""
    print("\n" + "="*70)
    print("CREATING LANGCHAIN AGENT")
    print("="*70)

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Define tools
    tools = [calculator, text_analyzer, knowledge_search]

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to tools. Use them when appropriate."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create executor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True
    )

    print("✓ Agent created with tools: calculator, text_analyzer, knowledge_search")

    return executor

# ==================== TEST ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LANGCHAIN AGENT DEMONSTRATION")
    print("="*70)

    agent = create_langchain_agent()

    # Test 1: Calculator
    print("\n" + "#"*70)
    print("TEST 1: Calculator")
    print("#"*70)
    result = agent.invoke({"input": "What is 15% of 340?"})
    print(f"\n✅ Answer: {result['output']}")

    # Test 2: Text Analysis
    print("\n" + "#"*70)
    print("TEST 2: Text Analysis")
    print("#"*70)
    result = agent.invoke({
        "input": "Analyze this text: 'LangChain makes building AI agents easy and powerful. It provides tools for rapid development.'"
    })
    print(f"\n✅ Answer: {result['output']}")

    # Test 3: Knowledge Search
    print("\n" + "#"*70)
    print("TEST 3: Knowledge Search")
    print("#"*70)
    result = agent.invoke({"input": "What is RAG?"})
    print(f"\n✅ Answer: {result['output']}")

    # Test 4: Multiple Tools
    print("\n" + "#"*70)
    print("TEST 4: Multiple Tools")
    print("#"*70)
    result = agent.invoke({
        "input": "Calculate 100 * 5, then analyze the word 'LangChain', then tell me about agents"
    })
    print(f"\n✅ Answer: {result['output']}")

    print("\n✅ Exercise 3 Complete!")
```

**Run it:** `python exercise3_langchain_agent.py`

### ✅ Checkpoint 3

**What you learned:**
- ✓ LangChain framework basics
- ✓ Tool decorator (@tool)
- ✓ AgentExecutor usage
- ✓ Rapid agent development

---

## 🔧 Exercise 4: Multi-Agent System (25-30 minutes)

**Goal:** Build a collaborative multi-agent system with specialized agents.

Create `exercise4_multi_agent.py`:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, List

load_dotenv()

# ==================== BASE AGENT ====================

class BaseAgent:
    """Base class for all agents"""

    def __init__(self, name: str, role: str, temperature: float = 0):
        self.name = name
        self.role = role
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    def execute(self, task: str) -> str:
        """Execute a task"""
        raise NotImplementedError

# ==================== SPECIALIZED AGENTS ====================

class ResearcherAgent(BaseAgent):
    """Agent specialized in research"""

    def __init__(self):
        super().__init__(
            name="Researcher",
            role="conducting research and gathering information",
            temperature=0
        )

    def execute(self, task: str) -> str:
        """Research a topic"""
        print(f"\n🔬 {self.name} researching...")

        prompt = f"""You are a research specialist. Research this topic and provide 3-4 key findings: {task}

Focus on facts, insights, and important information."""

        response = self.llm.invoke([
            SystemMessage(content=f"You are an expert at {self.role}."),
            HumanMessage(content=prompt)
        ])

        return response.content

class AnalystAgent(BaseAgent):
    """Agent specialized in analysis"""

    def __init__(self):
        super().__init__(
            name="Analyst",
            role="analyzing information and identifying patterns",
            temperature=0
        )

    def execute(self, task: str) -> str:
        """Analyze information"""
        print(f"\n📊 {self.name} analyzing...")

        prompt = f"""You are an analysis specialist. Analyze this information and provide insights: {task}

Identify patterns, trends, and important takeaways."""

        response = self.llm.invoke([
            SystemMessage(content=f"You are an expert at {self.role}."),
            HumanMessage(content=prompt)
        ])

        return response.content

class WriterAgent(BaseAgent):
    """Agent specialized in writing"""

    def __init__(self):
        super().__init__(
            name="Writer",
            role="creating clear, engaging content",
            temperature=0.7
        )

    def execute(self, task: str) -> str:
        """Write content"""
        print(f"\n✍️  {self.name} writing...")

        prompt = f"""You are a professional writer. Create content based on this: {task}

Write clearly and engagingly in 2-3 paragraphs."""

        response = self.llm.invoke([
            SystemMessage(content=f"You are an expert at {self.role}."),
            HumanMessage(content=prompt)
        ])

        return response.content

# ==================== COORDINATOR AGENT ====================

class CoordinatorAgent(BaseAgent):
    """Coordinates multiple specialist agents"""

    def __init__(self, agents: Dict[str, BaseAgent]):
        super().__init__(
            name="Coordinator",
            role="coordinating team of specialist agents",
            temperature=0
        )
        self.agents = agents

    def delegate(self, task: str) -> Dict[str, str]:
        """Determine which agents to use and delegate"""
        print(f"\n👔 {self.name} analyzing task...")

        available_agents = ", ".join(self.agents.keys())

        prompt = f"""Task: {task}

Available specialist agents: {available_agents}

Which agent(s) should handle this task? Consider:
- researcher: for finding information
- analyst: for analyzing data/information
- writer: for creating written content

Respond with ONLY the agent name(s), comma-separated."""

        response = self.llm.invoke([
            SystemMessage(content="You are a task coordinator."),
            HumanMessage(content=prompt)
        ])

        # Parse assigned agents
        assigned = [a.strip().lower() for a in response.content.split(',')]
        assigned = [a for a in assigned if a in self.agents]

        print(f"   Assigned to: {', '.join(assigned)}")

        # Execute with assigned agents
        results = {}
        for agent_name in assigned:
            agent = self.agents[agent_name]
            result = agent.execute(task)
            results[agent_name] = result

        return results

    def synthesize(self, task: str, results: Dict[str, str]) -> str:
        """Synthesize results from multiple agents"""
        print(f"\n👔 {self.name} synthesizing results...")

        results_text = "\n\n".join([
            f"**{name.title()}**: {result}"
            for name, result in results.items()
        ])

        prompt = f"""Original task: {task}

Results from specialist agents:
{results_text}

Synthesize these results into a cohesive final answer."""

        response = self.llm.invoke([
            SystemMessage(content="You are an expert at synthesis."),
            HumanMessage(content=prompt)
        ])

        return response.content

# ==================== MULTI-AGENT SYSTEM ====================

class MultiAgentSystem:
    """Complete multi-agent system"""

    def __init__(self):
        # Create specialist agents
        self.agents = {
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent()
        }

        # Create coordinator
        self.coordinator = CoordinatorAgent(self.agents)

    def execute_task(self, task: str) -> str:
        """Execute task using multi-agent system"""
        print(f"\n{'='*70}")
        print(f"MULTI-AGENT SYSTEM")
        print('='*70)
        print(f"Task: {task}")
        print('='*70)

        # Delegate to agents
        results = self.coordinator.delegate(task)

        # Synthesize
        final_result = self.coordinator.synthesize(task, results)

        print(f"\n{'='*70}")
        print("✅ FINAL RESULT:")
        print('='*70)
        print(final_result)

        return final_result

# ==================== TEST ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-AGENT SYSTEM DEMONSTRATION")
    print("="*70)

    system = MultiAgentSystem()

    # Test 1: Research + Analysis + Writing
    print("\n" + "#"*70)
    print("TEST 1: Complete Workflow (Research → Analyze → Write)")
    print("#"*70)
    system.execute_task(
        "Research the benefits of multi-agent AI systems, analyze the key advantages, "
        "and write a brief summary"
    )

    # Test 2: Research + Writing
    print("\n" + "#"*70)
    print("TEST 2: Research + Writing")
    print("#"*70)
    system.execute_task("Research LangChain framework and write an introduction")

    print("\n✅ Exercise 4 Complete!")
```

**Run it:** `python exercise4_multi_agent.py`

### ✅ Checkpoint 4

**What you learned:**
- ✓ Multi-agent architectures
- ✓ Specialized agent roles
- ✓ Coordinator pattern
- ✓ Agent-to-agent collaboration

---

## 🎯 Capstone Project: ResearchHub v1.0 (40-50 minutes)

**Goal:** Build a production-ready research platform combining all concepts.

Create `capstone_researchhub.py`:

```python
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== RESEARCHHUB v1.0 ====================

class ResearchHub:
    """
    Production multi-agent research platform combining:
    - Research agents
    - Agentic RAG
    - Multi-agent collaboration
    - Advanced workflows
    """

    def __init__(self):
        self.name = "ResearchHub v1.0"

        # Initialize knowledge base
        self.kb = self._setup_knowledge_base()

        # Initialize tools
        self.tools = self._setup_tools()
        self.functions = self._setup_functions()

        # Research history
        self.research_history = []

        logger.info(f"{self.name} initialized")

    def _setup_knowledge_base(self):
        """Setup vector database"""
        chroma_client = chromadb.PersistentClient(path="./researchhub_db")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        collection = chroma_client.get_or_create_collection(name="research_docs")

        # Populate if empty
        if collection.count() == 0:
            docs = [
                "AI agents are autonomous systems that can reason, plan, and take actions using tools.",
                "Multi-agent systems involve multiple specialized agents collaborating on complex tasks.",
                "LangChain is a framework for building LLM-powered applications with agents and tools.",
                "Agentic RAG systems use agents to decide when and what to retrieve from knowledge bases.",
                "Research agents can autonomously gather, analyze, and synthesize information from multiple sources."
            ]

            for i, doc in enumerate(docs):
                embedding = embedding_model.encode(doc)
                collection.add(
                    documents=[doc],
                    embeddings=[embedding.tolist()],
                    ids=[f"doc_{i}"]
                )

        logger.info(f"Knowledge base ready with {collection.count()} documents")

        return {"collection": collection, "model": embedding_model}

    def _setup_tools(self):
        """Setup tool definitions"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information on a topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "num_results": {"type": "integer", "default": 3}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve_documents",
                    "description": "Retrieve documents from knowledge base",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "n_results": {"type": "integer", "default": 2}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_findings",
                    "description": "Analyze research findings to identify key insights",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "findings": {"type": "string", "description": "Research findings to analyze"}
                        },
                        "required": ["findings"]
                    }
                }
            }
        ]

    def _setup_functions(self):
        """Setup function implementations"""
        return {
            "web_search": self.web_search,
            "retrieve_documents": self.retrieve_documents,
            "analyze_findings": self.analyze_findings
        }

    def web_search(self, query: str, num_results: int = 3) -> Dict:
        """Simulated web search"""
        logger.info(f"Web search: {query}")

        knowledge = {
            "agents": ["AI agents use LLMs for reasoning", "Agents can use tools autonomously"],
            "langchain": ["LangChain simplifies agent development", "LangChain provides pre-built patterns"],
            "rag": ["RAG retrieves relevant context", "Agentic RAG adds decision-making"]
        }

        results = []
        for key, snippets in knowledge.items():
            if key in query.lower():
                results.extend([{"snippet": s, "source": "Web"} for s in snippets])

        return {"success": True, "query": query, "results": results[:num_results]}

    def retrieve_documents(self, query: str, n_results: int = 2) -> Dict:
        """Retrieve from knowledge base"""
        logger.info(f"KB retrieval: {query}")

        query_embedding = self.kb["model"].encode(query)

        results = self.kb["collection"].query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents"]
        )

        docs = [{"content": doc} for doc in results['documents'][0]]

        return {"success": True, "query": query, "documents": docs}

    def analyze_findings(self, findings: str) -> Dict:
        """Analyze research findings"""
        logger.info("Analyzing findings...")

        # Simple analysis (in production, use LLM)
        insights = [
            "Key patterns identified",
            "Important themes extracted",
            "Conclusions drawn"
        ]

        return {"success": True, "insights": insights}

    def conduct_research(self, topic: str, mode: str = "comprehensive") -> Dict:
        """
        Conduct comprehensive research

        Args:
            topic: Research topic
            mode: "quick", "standard", or "comprehensive"

        Returns:
            Research report
        """
        print(f"\n{'='*70}")
        print(f"🏢 {self.name} - RESEARCH MODE: {mode.upper()}")
        print('='*70)
        print(f"Topic: {topic}\n")

        max_iterations = {"quick": 3, "standard": 5, "comprehensive": 8}[mode]

        system_prompt = f"""You are ResearchHub, an advanced research platform.

Your capabilities:
- web_search: Search for information
- retrieve_documents: Query knowledge base
- analyze_findings: Analyze research results

Conduct {mode} research on: {topic}

Process:
1. Search and retrieve information
2. Analyze findings
3. Synthesize comprehensive report"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research: {topic}"}
        ]

        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            print(f"--- Iteration {iteration} ---")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            if not response_message.tool_calls:
                report = response_message.content

                print(f"\n{'='*70}")
                print("✅ RESEARCH COMPLETE")
                print('='*70)
                print(report)

                result = {
                    "topic": topic,
                    "mode": mode,
                    "report": report,
                    "iterations": iteration,
                    "timestamp": datetime.now().isoformat()
                }

                self.research_history.append(result)

                return result

            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                print(f"\n🔧 {function_name}({list(arguments.keys())})")

                result = self.functions[function_name](**arguments)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(result)
                })

        return {"error": "Max iterations reached"}

    def get_stats(self) -> Dict:
        """Get platform statistics"""
        return {
            "total_research_conducted": len(self.research_history),
            "kb_documents": self.kb["collection"].count(),
            "tools_available": len(self.tools)
        }

# ==================== MAIN ====================

def main():
    """Demonstrate ResearchHub"""
    print("\n" + "="*70)
    print("🏢 RESEARCHHUB v1.0 - PRODUCTION RESEARCH PLATFORM")
    print("="*70)

    hub = ResearchHub()

    # Research tasks
    tasks = [
        ("Multi-agent AI systems", "quick"),
        ("Agentic RAG vs traditional RAG", "standard")
    ]

    for i, (topic, mode) in enumerate(tasks, 1):
        print(f"\n{'#'*70}")
        print(f"RESEARCH TASK {i}/{len(tasks)}")
        print('#'*70)

        hub.conduct_research(topic, mode=mode)

    # Show stats
    print(f"\n{'='*70}")
    print("📊 PLATFORM STATISTICS")
    print('='*70)
    stats = hub.get_stats()
    for key, value in stats.items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    print('='*70)

if __name__ == "__main__":
    main()
```

**Run it:** `python capstone_researchhub.py`

### ✅ Capstone Complete!

**What you built:**
- ✓ Complete research platform
- ✓ Multi-tool integration
- ✓ Agentic RAG capabilities
- ✓ Research history tracking
- ✓ Production architecture

---

## 🎓 Lab Summary

### What You Accomplished

Congratulations! You've built:

1. ✅ **Research Agent** - Autonomous information gathering
2. ✅ **Agentic RAG** - Smart retrieval with decision-making
3. ✅ **LangChain Agent** - Framework-powered development
4. ✅ **Multi-Agent System** - Collaborative specialists
5. ✅ **ResearchHub v1.0** - Production research platform

### Skills Mastered

- ✓ Research agent architectures
- ✓ Agentic RAG vs traditional RAG
- ✓ LangChain framework usage
- ✓ Multi-agent coordination
- ✓ Production deployment patterns

---

## 📚 Additional Challenges

### Challenge 1: Add More Agents
Extend the multi-agent system with:
- CriticAgent (reviews and critiques)
- EditorAgent (edits and improves content)
- FactCheckerAgent (validates information)

### Challenge 2: Advanced RAG
Enhance agentic RAG with:
- Query reformulation
- Source ranking
- Contradiction detection
- Citation tracking

### Challenge 3: LangGraph Workflow
Build a LangGraph workflow with:
- Conditional routing
- Parallel processing
- State management
- Error recovery

---

## 🔧 Troubleshooting

**Issue:** Framework not installed
```bash
Solution: pip install langchain langchain-openai
```

**Issue:** ChromaDB errors
```bash
Solution: Delete ./agentic_rag_db and ./researchhub_db
```

---

## 🎯 Next Steps

**You've completed the Advanced GenAI Training!** 🎉

**What's Next:**
1. Build real-world projects
2. Deploy to production
3. Contribute to open source
4. Continue learning advanced patterns

---

## ✅ Lab 8 Complete!

**Congratulations!** 🚀

You've mastered advanced multi-agent systems and completed all 8 labs of the Advanced GenAI Training course!

---

**Ready to build the future with AI agents!**
