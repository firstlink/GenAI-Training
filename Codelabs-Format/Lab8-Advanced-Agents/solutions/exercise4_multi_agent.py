"""
Lab 8 - Exercise 4: Multi-Agent System
Solution for building collaborative multi-agent system with specialized agents

Learning Objectives:
- Create specialized agents with different roles
- Implement coordinator pattern for delegation
- Build agent-to-agent collaboration
- Synthesize results from multiple agents
"""

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
