# Capstone Project: AI-Powered Customer Intelligence Platform

## Project Overview

Throughout this course, you'll build a **complete, production-ready AI customer support and intelligence platform** called **"SupportGenie"**. Each session adds new capabilities, culminating in a deployable application.

---

## What You'll Build

### Final Application: SupportGenie

**SupportGenie** is an AI-powered customer support platform that:
- ✅ Answers customer questions using company knowledge base (RAG)
- ✅ Performs actions like creating tickets, looking up orders (Function Calling)
- ✅ Remembers conversation context and customer history (Memory)
- ✅ Routes complex issues to specialized agents (Multi-Agent)
- ✅ Monitors quality and continuously improves (Evaluation)
- ✅ Runs 24/7 in production with monitoring (Deployment)

### Tech Stack
- **LLMs**: OpenAI GPT-3.5/4, Claude (optional)
- **Framework**: LangChain
- **Vector DB**: ChromaDB
- **Backend**: FastAPI
- **Frontend**: Gradio
- **Deployment**: Docker + Railway/Render
- **Monitoring**: Custom dashboard

---

## Session-by-Session Build Plan

### 🎯 Session 0: Setup
**What You Build**: Development environment

**Tasks**:
- Set up API keys
- Create project structure
- Test basic API calls

**Deliverable**: Working dev environment

---

### 🎯 Session 1: LLM Fundamentals
**What You Build**: Basic chatbot

**Features**:
- Simple Q&A interface
- Token management
- Cost tracking
- Streaming responses

**Code**:
```python
# supportgenie/chatbot.py
class BasicChatbot:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def chat(self, message):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful customer support assistant."},
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content
```

**Deliverable**: Working chatbot with streaming UI

---

### 🎯 Session 2: Prompt Engineering
**What You Build**: Smart response system with templates

**Features**:
- System prompts for different scenarios
- Few-shot examples for consistent format
- Template system for common queries
- Tone and style control

**Enhancements**:
- Professional tone for complaints
- Friendly tone for general questions
- Technical tone for product inquiries
- Empathetic responses for issues

**Code**:
```python
# supportgenie/prompts.py
class PromptTemplates:
    SYSTEM_PROMPT = """You are SupportGenie, a customer support AI assistant.

    Guidelines:
    - Be professional and empathetic
    - Keep responses under 100 words
    - Always offer to escalate if needed
    - Never make up information

    Response format:
    1. Acknowledge the issue
    2. Provide solution or information
    3. Ask if anything else is needed
    """

    def format_response(self, query, context=None):
        # Template logic
        pass
```

**Deliverable**: Chatbot with professional, consistent responses

---

### 🎯 Session 3: RAG System
**What You Build**: Knowledge base integration

**Features**:
- Load company knowledge base (policies, FAQs, product docs)
- Semantic search for relevant information
- Source attribution
- Confidence scoring

**Knowledge Base**:
- Return policy
- Shipping information
- Product specifications
- Troubleshooting guides
- Account management

**Code**:
```python
# supportgenie/knowledge_base.py
class KnowledgeBase:
    def __init__(self):
        self.vector_db = chromadb.Client()
        self.collection = self.vector_db.get_collection("support_docs")

    def search(self, query, top_k=3):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results

    def answer_with_sources(self, question):
        # RAG pipeline
        context = self.search(question)
        answer = self.generate_answer(question, context)
        return {
            "answer": answer,
            "sources": context['metadatas']
        }
```

**Deliverable**: Chatbot that answers from knowledge base with citations

---

### 🎯 Session 4: Function Calling
**What You Build**: Action-taking agent

**Capabilities**:
- Create support tickets
- Look up order status
- Check account information
- Update customer preferences
- Schedule callbacks

**Functions**:
```python
# supportgenie/tools.py
class SupportTools:
    def create_ticket(self, subject, description, priority="medium"):
        """Create a support ticket in the system"""
        ticket_id = generate_ticket_id()
        save_to_database(ticket_id, subject, description, priority)
        return f"Ticket #{ticket_id} created successfully"

    def lookup_order(self, order_id):
        """Look up order status"""
        order = fetch_order(order_id)
        return {
            "status": order.status,
            "tracking": order.tracking_number,
            "eta": order.estimated_delivery
        }

    def check_account(self, customer_email):
        """Check customer account details"""
        account = fetch_account(customer_email)
        return {
            "member_since": account.created_date,
            "orders": len(account.orders),
            "rewards_points": account.points
        }
```

**Deliverable**: Agent that can perform actions, not just answer questions

---

### 🎯 Session 5: Memory & Context
**What You Build**: Personalized support with memory

**Memory Types**:
1. **Conversation Memory**: Recent chat history
2. **Customer Memory**: Past interactions, preferences
3. **Context Memory**: Current issue tracking

**Features**:
- Remember customer details from earlier in conversation
- Reference past tickets/orders
- Personalized greetings
- Context-aware responses

**Code**:
```python
# supportgenie/memory.py
class CustomerMemory:
    def __init__(self):
        self.conversation_history = []
        self.customer_profile = {}
        self.current_context = {}

    def add_message(self, role, content):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })

    def get_customer_context(self, customer_id):
        # Fetch from database
        past_tickets = fetch_tickets(customer_id)
        past_orders = fetch_orders(customer_id)
        preferences = fetch_preferences(customer_id)

        return {
            "past_tickets": past_tickets,
            "past_orders": past_orders,
            "preferences": preferences
        }
```

**Example Interaction**:
```
Customer: Hi, I need help with my recent order
Agent: Hello! I see you ordered the Premium Widget on Nov 25th.
       How can I help with that order?

Customer: It hasn't arrived yet
Agent: I'm sorry about the delay. Let me check the tracking...
       Your order is currently in transit and should arrive by Dec 3rd.
       Would you like me to create a ticket to follow up?

Customer: Yes please
Agent: I've created ticket #12345 for you. Based on your past orders,
       I've marked this as high priority. Is there anything else?
```

**Deliverable**: Personalized agent with full context awareness

---

### 🎯 Session 6: Multi-Agent System
**What You Build**: Specialized agent orchestration

**Agent Types**:
1. **Router Agent**: Directs queries to right specialist
2. **Support Agent**: Handles general inquiries
3. **Technical Agent**: Solves technical issues
4. **Sales Agent**: Handles orders and billing
5. **Escalation Agent**: Manages complex cases

**Architecture**:
```
Customer Query
     ↓
Router Agent (classifies intent)
     ↓
   ┌─────┴─────┬─────────┬──────────┐
   ↓           ↓         ↓          ↓
Support    Technical   Sales   Escalation
Agent       Agent      Agent     Agent
```

**Code**:
```python
# supportgenie/multi_agent.py
class AgentOrchestrator:
    def __init__(self):
        self.router = RouterAgent()
        self.agents = {
            "support": SupportAgent(),
            "technical": TechnicalAgent(),
            "sales": SalesAgent(),
            "escalation": EscalationAgent()
        }

    def handle_query(self, query, context):
        # Route to appropriate agent
        agent_type = self.router.classify(query)
        agent = self.agents[agent_type]

        # Execute with agent
        response = agent.process(query, context)

        # Check if escalation needed
        if response.needs_escalation:
            response = self.agents["escalation"].process(query, context)

        return response
```

**Deliverable**: Multi-agent system that routes queries intelligently

---

### 🎯 Session 7: Evaluation & Monitoring
**What You Build**: Quality assurance system

**Metrics Tracked**:
- Response accuracy
- Customer satisfaction
- Resolution rate
- Response time
- Cost per conversation
- Common issues

**Features**:
- Automated testing suite
- Quality scoring
- A/B testing framework
- Performance dashboard
- Alert system

**Code**:
```python
# supportgenie/evaluation.py
class EvaluationFramework:
    def __init__(self):
        self.test_cases = load_test_cases()
        self.metrics = MetricsTracker()

    def evaluate_response(self, query, response, expected):
        scores = {
            "accuracy": self.check_accuracy(response, expected),
            "relevance": self.check_relevance(response, query),
            "tone": self.check_tone(response),
            "completeness": self.check_completeness(response)
        }
        return scores

    def run_test_suite(self):
        results = []
        for test in self.test_cases:
            response = self.agent.handle(test.query)
            score = self.evaluate_response(
                test.query,
                response,
                test.expected
            )
            results.append(score)
        return summarize_results(results)
```

**Dashboard Shows**:
- Real-time conversation metrics
- Quality scores over time
- Cost trends
- Common failure modes
- User satisfaction ratings

**Deliverable**: Monitored, evaluated system with quality metrics

---

### 🎯 Session 8: Production Deployment
**What You Build**: Production-ready application

**Components**:
1. **FastAPI Backend**: REST API endpoints
2. **Gradio Frontend**: User interface
3. **Database**: PostgreSQL for persistence
4. **Cache**: Redis for performance
5. **Monitoring**: Logging and alerts
6. **Deployment**: Docker + cloud hosting

**Architecture**:
```
┌─────────────┐
│   Gradio    │  (Frontend)
│     UI      │
└──────┬──────┘
       │
┌──────▼──────┐
│   FastAPI   │  (Backend API)
│   Server    │
└──────┬──────┘
       │
    ┌──┴───┬────────┬─────────┐
    ↓      ↓        ↓         ↓
┌───────┐ ┌────┐ ┌─────┐ ┌────────┐
│ LLM   │ │RAG │ │Tools│ │Memory  │
│ API   │ │ DB │ │     │ │  DB    │
└───────┘ └────┘ └─────┘ └────────┘
```

**Code Structure**:
```
supportgenie/
├── api/
│   ├── main.py              # FastAPI app
│   ├── routes.py            # API endpoints
│   └── middleware.py        # Auth, logging
├── agents/
│   ├── router.py            # Router agent
│   ├── support.py           # Support agent
│   ├── technical.py         # Technical agent
│   └── orchestrator.py      # Multi-agent system
├── knowledge/
│   ├── loader.py            # Document loading
│   ├── vector_store.py      # ChromaDB interface
│   └── retriever.py         # RAG pipeline
├── tools/
│   ├── tickets.py           # Ticket management
│   ├── orders.py            # Order lookup
│   └── accounts.py          # Account management
├── memory/
│   ├── conversation.py      # Chat history
│   ├── customer.py          # Customer profiles
│   └── database.py          # PostgreSQL interface
├── evaluation/
│   ├── metrics.py           # Quality metrics
│   ├── tests.py             # Test suite
│   └── dashboard.py         # Monitoring dashboard
├── ui/
│   └── gradio_app.py        # Frontend interface
├── config/
│   ├── settings.py          # Configuration
│   └── prompts.py           # Prompt templates
├── utils/
│   ├── logger.py            # Logging
│   ├── cache.py             # Redis cache
│   └── security.py          # Input validation
├── tests/
│   ├── test_agents.py
│   ├── test_rag.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

**Deployment**:
```bash
# Build Docker image
docker build -t supportgenie:latest .

# Run with docker-compose
docker-compose up -d

# Deploy to Railway
railway up

# Or deploy to Render
render deploy
```

**Deliverable**: Fully deployed, production-ready application

---

## Project Milestones

### Milestone 1: Basic Chatbot (Session 1-2)
**Capabilities**:
- Answer general questions
- Professional responses
- Token management

### Milestone 2: Knowledge Integration (Session 3)
**Capabilities**:
- Answer from company knowledge
- Provide sources
- Confidence scoring

### Milestone 3: Action-Taking Agent (Session 4-5)
**Capabilities**:
- Create tickets
- Look up information
- Remember context

### Milestone 4: Advanced System (Session 6-7)
**Capabilities**:
- Multi-agent routing
- Quality monitoring
- Performance optimization

### Milestone 5: Production Launch (Session 8)
**Capabilities**:
- Deployed application
- Monitoring dashboard
- Scalable architecture

---

## Sample Data & Resources

### Knowledge Base Documents (Provided)
- `data/policies/return_policy.md`
- `data/policies/shipping_info.md`
- `data/policies/warranty.md`
- `data/faqs/account_management.md`
- `data/faqs/product_questions.md`
- `data/products/catalog.json`
- `data/troubleshooting/common_issues.md`

### Test Datasets (Provided)
- `data/test_conversations/support_queries.json`
- `data/test_conversations/technical_issues.json`
- `data/test_conversations/sales_questions.json`
- `data/evaluation/golden_responses.json`

### Mock APIs (Provided)
- `mock_apis/ticket_system.py`
- `mock_apis/order_system.py`
- `mock_apis/customer_database.py`

---

## Success Criteria

### Technical Requirements
- ✅ Responds in < 2 seconds
- ✅ 90%+ accuracy on test suite
- ✅ Handles 100+ concurrent users
- ✅ < $0.10 per conversation cost
- ✅ 99.9% uptime

### User Experience
- ✅ Natural conversation flow
- ✅ Accurate information with sources
- ✅ Can complete common tasks
- ✅ Personalized interactions
- ✅ Professional tone

### Production Readiness
- ✅ Deployed and accessible
- ✅ Monitored with alerts
- ✅ Secure (auth, input validation)
- ✅ Scalable architecture
- ✅ Documented code

---

## Extension Ideas

After completing the core project, students can:

1. **Add More Agents**: Refunds agent, shipping agent, loyalty program agent
2. **Multi-Language**: Support multiple languages
3. **Voice Interface**: Add speech-to-text/text-to-speech
4. **Analytics Dashboard**: Customer insights and trends
5. **Mobile App**: React Native or Flutter frontend
6. **Slack/Discord Integration**: Deploy as bot
7. **Email Support**: Process and respond to emails
8. **Sentiment Analysis**: Detect frustrated customers
9. **Predictive Support**: Proactive issue detection
10. **Knowledge Graph**: More sophisticated reasoning

---

## Portfolio Showcase

### What to Include:
1. **Live Demo**: Deployed application URL
2. **GitHub Repo**: Well-documented code
3. **Architecture Diagram**: System design
4. **Demo Video**: 3-5 min walkthrough
5. **Performance Metrics**: Evaluation results
6. **Case Study**: Technical write-up

### Portfolio Template:
```markdown
# SupportGenie: AI-Powered Customer Intelligence Platform

## Overview
[Brief description]

## Demo
🔗 Live Demo: https://supportgenie.railway.app
🎥 Video Demo: [YouTube link]
💻 GitHub: https://github.com/yourusername/supportgenie

## Features
- RAG-powered knowledge base
- Multi-agent system
- Real-time monitoring
- Production deployment

## Tech Stack
OpenAI GPT-4, LangChain, ChromaDB, FastAPI, Docker

## Metrics
- 92% accuracy on test suite
- < 1.5s average response time
- $0.08 per conversation cost

## Challenges & Solutions
[Technical challenges you overcame]

## Future Enhancements
[Next features to add]
```

---

## Grading Rubric

### Session Checkpoints (10 points each)
- Code runs successfully: 5 points
- Meets feature requirements: 3 points
- Code quality and documentation: 2 points

### Final Project (100 points)
- **Functionality (40 points)**:
  - All features working: 30 points
  - Code quality: 10 points

- **Performance (20 points)**:
  - Response time: 10 points
  - Accuracy: 10 points

- **Production Readiness (20 points)**:
  - Deployment: 10 points
  - Monitoring: 5 points
  - Security: 5 points

- **Documentation (10 points)**:
  - Code documentation: 5 points
  - README and guides: 5 points

- **Presentation (10 points)**:
  - Demo video: 5 points
  - Technical write-up: 5 points

**Total**: 180 points (80 from checkpoints + 100 from final)

---

## Getting Started

### Initial Setup
```bash
# Clone starter template
git clone https://github.com/course/supportgenie-starter
cd supportgenie-starter

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your API keys to .env

# Run initial tests
pytest tests/

# Start development server
python -m uvicorn api.main:app --reload
```

### First Task (Session 1)
Build the basic chatbot in `supportgenie/chatbot.py` following the session guide.

---

**Ready to build SupportGenie? Let's start with Session 0!** 🚀
