"""
Lab 8 - Exercise 3: LangChain Framework Agent
Solution for building agents using LangChain framework

Learning Objectives:
- Use LangChain framework for rapid agent development
- Implement tool decorator pattern
- Create AgentExecutor for agent management
- Build agents with pre-built patterns

Note: Requires langchain and langchain-openai packages
Install: pip install langchain langchain-openai
"""

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
