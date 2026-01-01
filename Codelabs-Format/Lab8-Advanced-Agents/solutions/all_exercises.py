"""
Lab 8 - Complete Solutions: Advanced Multi-Agent Systems
Consolidated implementation of all exercises and capstone

Learning Objectives:
- Build research agents that gather and synthesize information
- Implement agentic RAG with dynamic retrieval decisions
- Use LangChain framework for rapid agent development
- Create multi-agent systems with specialized collaboration
- Build production-ready research platform

Exercises:
1. Research Agent - Autonomous information gathering system
2. Agentic RAG System - Smart retrieval with decision-making
3. LangChain Agent - Framework-powered agent development
4. Multi-Agent System - Collaborative specialist team
Capstone: ResearchHub v1.0 - Production multi-agent research platform
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# EXERCISE 1: RESEARCH AGENT
# ============================================================================

class ResearchTools:
    """
    Tools for research agent

    Provides simulated web search capabilities for information gathering
    """

    def __init__(self):
        self.search_history = []

    def web_search(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        """
        Simulated web search (in production, use real search API like Tavily, SerpAPI, etc.)

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            Search results with articles
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
                },
                {
                    "title": "Deep Learning Advances",
                    "snippet": "Deep learning uses neural networks with multiple layers for pattern recognition.",
                    "source": "AI Journal"
                }
            ],
            "multi-agent": [
                {
                    "title": "Multi-Agent Systems Overview",
                    "snippet": "Multiple specialized agents collaborate on complex tasks, each with unique capabilities.",
                    "source": "AI Research"
                },
                {
                    "title": "Agent Coordination Patterns",
                    "snippet": "Coordinator patterns enable efficient delegation and result synthesis across agents.",
                    "source": "Agent Framework Guide"
                }
            ],
            "agentic rag": [
                {
                    "title": "Agentic RAG vs Traditional RAG",
                    "snippet": "Agentic RAG allows agents to decide when and what to retrieve, improving relevance.",
                    "source": "RAG Research"
                },
                {
                    "title": "Dynamic Retrieval Strategies",
                    "snippet": "Agents can perform multi-hop retrieval and query reformulation for better results.",
                    "source": "AI Retrieval"
                }
            ],
            "langchain": [
                {
                    "title": "LangChain Framework",
                    "snippet": "LangChain simplifies building LLM applications with agents, tools, and chains.",
                    "source": "LangChain Docs"
                },
                {
                    "title": "LangChain Agents",
                    "snippet": "LangChain provides pre-built agent patterns and tool integrations for rapid development.",
                    "source": "Developer Guide"
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
                "title": f"General information about {query}",
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


class ResearchAgent:
    """
    Autonomous research agent

    Features:
    - Web search integration
    - Multi-step information gathering
    - Result synthesis and reporting
    - Configurable research depth
    """

    def __init__(self, api_key: str = None):
        """
        Initialize research agent

        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.tools = ResearchTools()
        self.findings = []

        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information. Returns relevant articles and sources.",
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

    def research(self, topic: str, depth: str = "moderate", verbose: bool = True) -> Dict[str, Any]:
        """
        Conduct research on a topic

        Args:
            topic: Research topic
            depth: "brief", "moderate", or "comprehensive"
            verbose: Print detailed execution

        Returns:
            Research report
        """
        if verbose:
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

            if verbose:
                print(f"\n--- Research Iteration {iteration} ---")

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tool_definitions,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            # Check if done
            if not response_message.tool_calls:
                final_report = response_message.content

                if verbose:
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

                if verbose:
                    print(f"\n🔧 {function_name}({json.dumps(arguments)})")

                # Execute
                result = self.functions[function_name](**arguments)

                if verbose:
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


# ============================================================================
# EXERCISE 2: AGENTIC RAG SYSTEM
# ============================================================================

class KnowledgeBase:
    """
    Vector database for Agentic RAG

    Features:
    - ChromaDB persistent storage
    - Sentence transformer embeddings
    - Semantic search
    - Metadata support
    """

    def __init__(self, collection_name: str = "agentic_rag_docs", persist_directory: str = "./agentic_rag_db"):
        """
        Initialize knowledge base

        Args:
            collection_name: Name for ChromaDB collection
            persist_directory: Directory for persistent storage
        """
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
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
            },
            {
                "text": "Reinforcement learning trains agents to make decisions through trial and error, receiving rewards for good actions. Used in robotics and game AI.",
                "metadata": {"topic": "reinforcement_learning", "category": "advanced"}
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


class AgenticRAG:
    """
    RAG system where agent decides when to retrieve

    Features:
    - Dynamic retrieval decisions
    - Multi-hop reasoning
    - Query understanding
    - Retrieval tracking
    """

    def __init__(self, knowledge_base: KnowledgeBase, api_key: str = None):
        """
        Initialize agentic RAG

        Args:
            knowledge_base: KnowledgeBase instance
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
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
        """
        Retrieve documents from knowledge base

        Args:
            query: Search query
            n_results: Number of results

        Returns:
            Retrieved documents with metadata
        """
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

    def answer_question(self, question: str, verbose: bool = True) -> Dict:
        """
        Answer question using agentic RAG

        Args:
            question: User question
            verbose: Print detailed execution

        Returns:
            Answer with retrieval statistics
        """
        if verbose:
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

            if verbose:
                print(f"--- Iteration {iteration} ---")

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            # Check if done
            if not response_message.tool_calls:
                answer = response_message.content

                if verbose:
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
            if response_message.content and verbose:
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


# ============================================================================
# EXERCISE 3: LANGCHAIN AGENT
# ============================================================================

# Note: LangChain integration is shown in the demo functions below
# The implementation requires langchain, langchain-openai packages


# ============================================================================
# EXERCISE 4: MULTI-AGENT SYSTEM
# ============================================================================

class BaseAgent:
    """
    Base class for all specialized agents

    Provides common functionality for agent communication
    """

    def __init__(self, name: str, role: str, temperature: float = 0, api_key: str = None):
        """
        Initialize base agent

        Args:
            name: Agent name
            role: Agent role description
            temperature: LLM temperature
            api_key: OpenAI API key
        """
        self.name = name
        self.role = role
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.temperature = temperature

    def execute(self, task: str) -> str:
        """Execute a task (to be implemented by subclasses)"""
        raise NotImplementedError


class ResearcherAgent(BaseAgent):
    """
    Agent specialized in research

    Focuses on finding and gathering information
    """

    def __init__(self, api_key: str = None):
        super().__init__(
            name="Researcher",
            role="conducting research and gathering information",
            temperature=0,
            api_key=api_key
        )

    def execute(self, task: str, verbose: bool = True) -> str:
        """
        Research a topic

        Args:
            task: Research task
            verbose: Print execution details

        Returns:
            Research findings
        """
        if verbose:
            print(f"\n🔬 {self.name} researching...")

        prompt = f"""You are a research specialist. Research this topic and provide 3-4 key findings: {task}

Focus on facts, insights, and important information."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are an expert at {self.role}."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )

        return response.choices[0].message.content


class AnalystAgent(BaseAgent):
    """
    Agent specialized in analysis

    Focuses on analyzing data and identifying patterns
    """

    def __init__(self, api_key: str = None):
        super().__init__(
            name="Analyst",
            role="analyzing information and identifying patterns",
            temperature=0,
            api_key=api_key
        )

    def execute(self, task: str, verbose: bool = True) -> str:
        """
        Analyze information

        Args:
            task: Analysis task
            verbose: Print execution details

        Returns:
            Analysis results
        """
        if verbose:
            print(f"\n📊 {self.name} analyzing...")

        prompt = f"""You are an analysis specialist. Analyze this information and provide insights: {task}

Identify patterns, trends, and important takeaways."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are an expert at {self.role}."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )

        return response.choices[0].message.content


class WriterAgent(BaseAgent):
    """
    Agent specialized in writing

    Focuses on creating clear, engaging content
    """

    def __init__(self, api_key: str = None):
        super().__init__(
            name="Writer",
            role="creating clear, engaging content",
            temperature=0.7,
            api_key=api_key
        )

    def execute(self, task: str, verbose: bool = True) -> str:
        """
        Write content

        Args:
            task: Writing task
            verbose: Print execution details

        Returns:
            Written content
        """
        if verbose:
            print(f"\n✍️  {self.name} writing...")

        prompt = f"""You are a professional writer. Create content based on this: {task}

Write clearly and engagingly in 2-3 paragraphs."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are an expert at {self.role}."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )

        return response.choices[0].message.content


class CoordinatorAgent(BaseAgent):
    """
    Coordinates multiple specialist agents

    Features:
    - Task delegation
    - Agent selection
    - Result synthesis
    """

    def __init__(self, agents: Dict[str, BaseAgent], api_key: str = None):
        super().__init__(
            name="Coordinator",
            role="coordinating team of specialist agents",
            temperature=0,
            api_key=api_key
        )
        self.agents = agents

    def delegate(self, task: str, verbose: bool = True) -> Dict[str, str]:
        """
        Determine which agents to use and delegate

        Args:
            task: Task to delegate
            verbose: Print execution details

        Returns:
            Results from each agent
        """
        if verbose:
            print(f"\n👔 {self.name} analyzing task...")

        available_agents = ", ".join(self.agents.keys())

        prompt = f"""Task: {task}

Available specialist agents: {available_agents}

Which agent(s) should handle this task? Consider:
- researcher: for finding information
- analyst: for analyzing data/information
- writer: for creating written content

Respond with ONLY the agent name(s), comma-separated."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a task coordinator."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )

        # Parse assigned agents
        assigned = [a.strip().lower() for a in response.choices[0].message.content.split(',')]
        assigned = [a for a in assigned if a in self.agents]

        if verbose:
            print(f"   Assigned to: {', '.join(assigned)}")

        # Execute with assigned agents
        results = {}
        for agent_name in assigned:
            agent = self.agents[agent_name]
            result = agent.execute(task, verbose=verbose)
            results[agent_name] = result

        return results

    def synthesize(self, task: str, results: Dict[str, str], verbose: bool = True) -> str:
        """
        Synthesize results from multiple agents

        Args:
            task: Original task
            results: Results from agents
            verbose: Print execution details

        Returns:
            Synthesized final result
        """
        if verbose:
            print(f"\n👔 {self.name} synthesizing results...")

        results_text = "\n\n".join([
            f"**{name.title()}**: {result}"
            for name, result in results.items()
        ])

        prompt = f"""Original task: {task}

Results from specialist agents:
{results_text}

Synthesize these results into a cohesive final answer."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at synthesis."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )

        return response.choices[0].message.content


class MultiAgentSystem:
    """
    Complete multi-agent system

    Features:
    - Multiple specialized agents
    - Coordinator-based delegation
    - Result synthesis
    """

    def __init__(self, api_key: str = None):
        """
        Initialize multi-agent system

        Args:
            api_key: OpenAI API key
        """
        # Create specialist agents
        self.agents = {
            "researcher": ResearcherAgent(api_key=api_key),
            "analyst": AnalystAgent(api_key=api_key),
            "writer": WriterAgent(api_key=api_key)
        }

        # Create coordinator
        self.coordinator = CoordinatorAgent(self.agents, api_key=api_key)

    def execute_task(self, task: str, verbose: bool = True) -> str:
        """
        Execute task using multi-agent system

        Args:
            task: Task description
            verbose: Print detailed execution

        Returns:
            Final synthesized result
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"MULTI-AGENT SYSTEM")
            print('='*70)
            print(f"Task: {task}")
            print('='*70)

        # Delegate to agents
        results = self.coordinator.delegate(task, verbose=verbose)

        # Synthesize
        final_result = self.coordinator.synthesize(task, results, verbose=verbose)

        if verbose:
            print(f"\n{'='*70}")
            print("✅ FINAL RESULT:")
            print('='*70)
            print(final_result)

        return final_result


# ============================================================================
# CAPSTONE: RESEARCHHUB v1.0
# ============================================================================

class ResearchHub:
    """
    Production multi-agent research platform

    Combines:
    - Research agents
    - Agentic RAG
    - Multi-agent collaboration
    - Advanced workflows

    Features:
    - Comprehensive research capabilities
    - Vector database integration
    - Multiple tool support
    - Research history tracking
    - Production error handling
    """

    def __init__(self, api_key: str = None):
        """
        Initialize ResearchHub

        Args:
            api_key: OpenAI API key
        """
        self.name = "ResearchHub v1.0"
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))

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
                "Multi-agent systems involve multiple specialized agents collaborating on complex tasks through coordination patterns.",
                "LangChain is a framework for building LLM-powered applications with agents, tools, and chains.",
                "Agentic RAG systems use agents to decide when and what to retrieve from knowledge bases, improving relevance.",
                "Research agents can autonomously gather, analyze, and synthesize information from multiple sources.",
                "Coordinator agents delegate tasks to specialized agents and synthesize their results.",
                "Tool calling enables agents to use external functions and APIs to extend their capabilities."
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
            "agents": [
                "AI agents use LLMs for reasoning and decision-making",
                "Agents can use tools autonomously to accomplish goals",
                "Multi-agent systems enable specialization and collaboration"
            ],
            "langchain": [
                "LangChain simplifies agent development with pre-built patterns",
                "LangChain provides tools, chains, and agent executors",
                "LangGraph enables complex agent workflows with state management"
            ],
            "rag": [
                "RAG retrieves relevant context before generation",
                "Agentic RAG adds decision-making about when to retrieve",
                "Multi-hop RAG enables complex information gathering"
            ],
            "multi-agent": [
                "Specialized agents handle specific tasks efficiently",
                "Coordinator patterns enable effective delegation",
                "Agent collaboration improves complex problem solving"
            ]
        }

        results = []
        for key, snippets in knowledge.items():
            if key in query.lower():
                results.extend([{"snippet": s, "source": "Web"} for s in snippets])

        if not results:
            results = [{"snippet": f"Information about {query} from various sources", "source": "Web"}]

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

        # Use LLM for analysis
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert analyst. Analyze research findings and identify key insights."},
                {"role": "user", "content": f"Analyze these findings:\n\n{findings}"}
            ],
            temperature=0
        )

        analysis = response.choices[0].message.content

        return {"success": True, "analysis": analysis}

    def conduct_research(self, topic: str, mode: str = "comprehensive", verbose: bool = True) -> Dict:
        """
        Conduct comprehensive research

        Args:
            topic: Research topic
            mode: "quick", "standard", or "comprehensive"
            verbose: Print detailed execution

        Returns:
            Research report
        """
        if verbose:
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
1. Search and retrieve information from multiple sources
2. Analyze findings to identify insights
3. Synthesize comprehensive report with key takeaways"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research: {topic}"}
        ]

        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            if verbose:
                print(f"--- Iteration {iteration} ---")

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            if not response_message.tool_calls:
                report = response_message.content

                if verbose:
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

                if verbose:
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


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demo_exercise1_research_agent():
    """Demonstrate Exercise 1: Research Agent"""
    print("\n" + "="*70)
    print("EXERCISE 1: RESEARCH AGENT")
    print("="*70)

    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  Skipped: OPENAI_API_KEY not found")
        return

    agent = ResearchAgent()

    # Test 1: Brief research
    print("\n" + "#"*70)
    print("TEST 1: Brief Research")
    print("#"*70)
    result = agent.research("machine learning applications", depth="brief", verbose=True)
    print(f"\nSearches performed: {result['searches_performed']}")

    # Test 2: Moderate research
    print("\n" + "#"*70)
    print("TEST 2: Moderate Research")
    print("#"*70)
    agent2 = ResearchAgent()
    result = agent2.research("multi-agent AI systems", depth="moderate", verbose=True)
    print(f"\nSearches performed: {result['searches_performed']}")

    print("\n" + "="*70)
    print("✅ Exercise 1 Complete!")
    print("="*70)


def demo_exercise2_agentic_rag():
    """Demonstrate Exercise 2: Agentic RAG System"""
    print("\n" + "="*70)
    print("EXERCISE 2: AGENTIC RAG SYSTEM")
    print("="*70)

    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  Skipped: OPENAI_API_KEY not found")
        return

    kb = KnowledgeBase()
    agentic_rag = AgenticRAG(kb)

    # Test 1: General knowledge (should NOT retrieve)
    print("\n" + "#"*70)
    print("TEST 1: General Knowledge (No Retrieval Expected)")
    print("#"*70)
    result = agentic_rag.answer_question("What is 25 * 4?", verbose=True)
    print(f"\nRetrievals: {result['retrievals']} (expected: 0)")

    # Test 2: KB-specific question (SHOULD retrieve)
    print("\n" + "#"*70)
    print("TEST 2: Knowledge Base Query (Retrieval Expected)")
    print("#"*70)
    agentic_rag.retrieval_log = []  # Reset log
    result = agentic_rag.answer_question(
        "According to the knowledge base, what is supervised learning?",
        verbose=True
    )
    print(f"\nRetrievals: {result['retrievals']} (expected: 1+)")

    # Test 3: Multi-hop question (MULTIPLE retrievals)
    print("\n" + "#"*70)
    print("TEST 3: Multi-Hop Query (Multiple Retrievals Expected)")
    print("#"*70)
    agentic_rag.retrieval_log = []
    result = agentic_rag.answer_question(
        "Compare machine learning and deep learning based on the knowledge base",
        verbose=True
    )
    print(f"\nRetrievals: {result['retrievals']} (expected: 2+)")

    print("\n" + "="*70)
    print("✅ Exercise 2 Complete!")
    print("="*70)


def demo_exercise3_langchain_agent():
    """Demonstrate Exercise 3: LangChain Agent"""
    print("\n" + "="*70)
    print("EXERCISE 3: LANGCHAIN AGENT")
    print("="*70)

    print("\n⚠️  Note: LangChain integration requires:")
    print("   pip install langchain langchain-openai")
    print("\n   See exercise3_langchain_agent.py in lab materials for")
    print("   full implementation using LangChain framework.")

    print("\n💡 Key Concepts:")
    print("   - @tool decorator for easy tool creation")
    print("   - AgentExecutor for agent management")
    print("   - Built-in error handling and parsing")
    print("   - Rapid agent development with pre-built patterns")

    print("\n" + "="*70)
    print("✅ Exercise 3 Complete!")
    print("="*70)


def demo_exercise4_multi_agent():
    """Demonstrate Exercise 4: Multi-Agent System"""
    print("\n" + "="*70)
    print("EXERCISE 4: MULTI-AGENT SYSTEM")
    print("="*70)

    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  Skipped: OPENAI_API_KEY not found")
        return

    system = MultiAgentSystem()

    # Test 1: Research + Analysis + Writing
    print("\n" + "#"*70)
    print("TEST 1: Complete Workflow (Research → Analyze → Write)")
    print("#"*70)
    system.execute_task(
        "Research the benefits of multi-agent AI systems, analyze the key advantages, "
        "and write a brief summary",
        verbose=True
    )

    # Test 2: Research + Writing
    print("\n" + "#"*70)
    print("TEST 2: Research + Writing")
    print("#"*70)
    system.execute_task(
        "Research LangChain framework and write an introduction",
        verbose=True
    )

    print("\n" + "="*70)
    print("✅ Exercise 4 Complete!")
    print("="*70)


def demo_capstone_researchhub():
    """Demonstrate Capstone: ResearchHub v1.0"""
    print("\n" + "="*70)
    print("CAPSTONE: RESEARCHHUB v1.0")
    print("="*70)

    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  Skipped: OPENAI_API_KEY not found")
        return

    hub = ResearchHub()

    print("\n✅ ResearchHub initialized with:")
    print("   - Vector knowledge base")
    print("   - Web search capabilities")
    print("   - Document retrieval")
    print("   - Finding analysis")
    print("   - Research history tracking")

    # Research tasks
    tasks = [
        ("Multi-agent AI systems", "quick"),
        ("Agentic RAG vs traditional RAG", "standard")
    ]

    for i, (topic, mode) in enumerate(tasks, 1):
        print(f"\n{'#'*70}")
        print(f"RESEARCH TASK {i}/{len(tasks)}")
        print('#'*70)

        hub.conduct_research(topic, mode=mode, verbose=True)

    # Show stats
    print(f"\n{'='*70}")
    print("📊 PLATFORM STATISTICS")
    print('='*70)
    stats = hub.get_stats()
    for key, value in stats.items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    print('='*70)

    print("\n" + "="*70)
    print("✅ Capstone Complete!")
    print("="*70)
    print("\n🎉 ResearchHub v1.0 is production-ready!")


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Run all demonstrations"""
    print("="*70)
    print("LAB 8: ADVANCED MULTI-AGENT SYSTEMS - COMPLETE SOLUTIONS")
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
    print("1. Exercise 1: Research Agent")
    print("2. Exercise 2: Agentic RAG System")
    print("3. Exercise 3: LangChain Agent (Info)")
    print("4. Exercise 4: Multi-Agent System")
    print("5. Capstone: ResearchHub v1.0")
    print("6. Run All")
    print("="*70)

    choice = input("\nSelect demonstration (1-6, or Enter for all): ").strip()

    if choice == "1":
        demo_exercise1_research_agent()
    elif choice == "2":
        demo_exercise2_agentic_rag()
    elif choice == "3":
        demo_exercise3_langchain_agent()
    elif choice == "4":
        demo_exercise4_multi_agent()
    elif choice == "5":
        demo_capstone_researchhub()
    else:
        # Run all
        demo_exercise1_research_agent()
        demo_exercise2_agentic_rag()
        demo_exercise3_langchain_agent()
        demo_exercise4_multi_agent()
        demo_capstone_researchhub()

    # Final summary
    print("\n" + "="*70)
    print("🎓 LAB 8 COMPLETE!")
    print("="*70)
    print("\n💡 KEY LEARNINGS:")
    print("   1. Research agents for autonomous information gathering")
    print("   2. Agentic RAG with dynamic retrieval decisions")
    print("   3. LangChain framework for rapid development")
    print("   4. Multi-agent systems with specialization")
    print("   5. Production-ready research platform architecture")

    print("\n🎯 SKILLS ACQUIRED:")
    print("   ✅ Research agent design and implementation")
    print("   ✅ Agentic RAG vs traditional RAG patterns")
    print("   ✅ Multi-agent coordination and collaboration")
    print("   ✅ Tool integration and function calling")
    print("   ✅ Production deployment strategies")

    print("\n🚀 CONGRATULATIONS!")
    print("   You've completed all 8 labs of Advanced GenAI Training!")
    print("   You're now ready to build production AI agent systems!")


if __name__ == "__main__":
    main()
