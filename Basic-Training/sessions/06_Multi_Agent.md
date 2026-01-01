# Session 6: Multi-Agent Orchestration

**Duration**: 90 minutes  
**Difficulty**: Advanced  
**Colab Notebook**: [06_Multi_Agent.ipynb](../notebooks/06_Multi_Agent.ipynb)

## Learning Objectives
- 🎯 Design multi-agent architectures
- 🎯 Implement router agents  
- 🎯 Build specialist agents
- 🎯 Orchestrate agent workflows
- 🎯 Handle agent-to-agent communication

## Capstone: SupportGenie v0.6 - Multi-Agent System

Specialized agents:
- **Router Agent**: Classifies and routes queries
- **Support Agent**: General inquiries
- **Technical Agent**: Technical issues
- **Sales Agent**: Orders and billing
- **Escalation Agent**: Complex cases

## Part 1: Why Multi-Agent Systems?

### Single Agent Limitations
- Tries to do everything
- Generic responses
- Hard to maintain
- Doesn't scale well

### Multi-Agent Benefits
- Specialized expertise
- Better accuracy
- Easier to maintain
- Scalable architecture

## Part 2: Agent Types

### 1. Router Agent
```python
class RouterAgent:
    """Classifies queries and routes to specialists"""
    
    def classify(self, query):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": """Classify customer queries:
                - "support": General inquiries, policies
                - "technical": Product issues, troubleshooting
                - "sales": Orders, billing, payments
                - "escalation": Complaints, refunds, urgent
                
                Return ONLY the classification."""
            }, {
                "role": "user",
                "content": query
            }]
        )
        return response.choices[0].message.content.strip()
```

### 2. Specialist Agents
```python
class TechnicalAgent:
    """Handles technical issues"""
    
    SYSTEM_MESSAGE = """You are a technical support specialist.
    - Expert in troubleshooting
    - Provide step-by-step solutions
    - Ask diagnostic questions
    - Escalate if hardware issue"""

class SalesAgent:
    """Handles orders and billing"""
    
    SYSTEM_MESSAGE = """You are a sales support specialist.
    - Help with orders and payments
    - Answer pricing questions
    - Process refunds
    - Upsell when appropriate"""
```

## Part 3: Orchestrator Pattern

```python
class AgentOrchestrator:
    def __init__(self):
        self.router = RouterAgent()
        self.agents = {
            "support": SupportAgent(),
            "technical": TechnicalAgent(),
            "sales": SalesAgent(),
            "escalation": EscalationAgent()
        }
    
    def handle_query(self, query, context=None):
        # Route to appropriate agent
        agent_type = self.router.classify(query)
        agent = self.agents[agent_type]
        
        # Execute with specialist
        response = agent.process(query, context)
        
        # Check if escalation needed
        if response.needs_escalation:
            response = self.agents["escalation"].process(query, context)
        
        return response
```

## Part 4: Sequential Workflow

Agents execute in sequence, passing results forward:

```python
class SequentialWorkflow:
    def __init__(self, agents):
        self.agents = agents
    
    def execute(self, initial_input):
        state = {"input": initial_input}
        
        for agent in self.agents:
            result = agent.execute(state)
            state.update(result)
        
        return state

# Example: Research → Analyze → Write
workflow = SequentialWorkflow([
    ResearchAgent(),
    AnalysisAgent(),
    WriterAgent()
])

result = workflow.execute("Write article about AI")
```

## Part 5: Parallel Execution

Multiple agents run simultaneously:

```python
from concurrent.futures import ThreadPoolExecutor

class ParallelWorkflow:
    def execute_parallel(self, query, agents):
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = [
                executor.submit(agent.process, query) 
                for agent in agents
            ]
            results = [f.result() for f in futures]
        
        # Synthesize results
        return self.synthesize(results)
```

## Part 6: Hierarchical Multi-Agent

```
Orchestrator Agent
    ↓
┌───┴────┬─────────┬──────────┐
↓        ↓         ↓          ↓
Support  Technical Sales  Escalation
Agent    Agent     Agent  Agent
```

## Part 7: SupportGenie v0.6

```python
class SupportGenieV6:
    """Multi-agent customer support system"""
    
    def __init__(self):
        self.setup_agents()
    
    def setup_agents(self):
        self.router = RouterAgent()
        self.support_agent = SupportAgent()
        self.tech_agent = TechnicalAgent()
        self.sales_agent = SalesAgent()
        self.escalation_agent = EscalationAgent()
    
    def handle_query(self, query, customer_profile):
        # Classify query
        agent_type = self.router.classify(query)
        
        # Route to specialist
        if agent_type == "support":
            response = self.support_agent.process(query, customer_profile)
        elif agent_type == "technical":
            response = self.tech_agent.process(query, customer_profile)
        elif agent_type == "sales":
            response = self.sales_agent.process(query, customer_profile)
        else:
            response = self.escalation_agent.process(query, customer_profile)
        
        return response
```

## Exercises
1. Build a research multi-agent system
2. Create parallel agent execution
3. Implement agent voting system
4. Design hierarchical agents

## Key Takeaways
✅ Multi-agent systems scale better  
✅ Router agents enable specialization  
✅ Orchestrators coordinate workflows  
✅ Agents can run sequentially or parallel  
✅ Specialization improves accuracy

**Session 6 Complete!** 🎉  
**Next**: [Session 7: Evaluation →](07_Evaluation.md)
