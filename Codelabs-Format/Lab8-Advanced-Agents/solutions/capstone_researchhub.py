"""
Lab 8 - Capstone: ResearchHub v1.0
Solution for production-ready multi-agent research platform

Combines all concepts:
- Research agents
- Agentic RAG
- Multi-agent collaboration
- Advanced workflows

Learning Objectives:
- Build production-ready research platform
- Integrate multiple tools and capabilities
- Implement comprehensive error handling
- Track research history and analytics
"""

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
