"""
Lab 8 - Exercise 1: Research Agent
Solution for building autonomous research agent with web search capabilities

Learning Objectives:
- Build research agents that gather information
- Implement multi-step research workflows
- Synthesize findings into coherent reports
- Configure research depth
"""

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

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            Search results with articles, snippets, and sources
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
