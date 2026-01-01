# 🚀 Advanced GenAI Training - Interactive Codelabs

> **Master Production-Ready AI Systems Through Hands-On Labs**

Welcome to the interactive, hands-on version of the Advanced GenAI Training course! This collection of codelabs will guide you step-by-step from AI beginner to enterprise AI engineer.

---

## 📖 About These Codelabs

Each codelab is designed following **Google Codelabs best practices** with:
- ✅ Clear learning objectives
- ✅ Estimated completion time
- ✅ Progressive difficulty
- ✅ Hands-on code examples
- ✅ Interactive exercises
- ✅ Visual diagrams and callouts
- ✅ Step-by-step navigation

---

## 🎯 Learning Path

```
┌─────────────────────────────────────────────────────────┐
│           START YOUR AI ENGINEERING JOURNEY              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  WEEK 1: FOUNDATIONS                                     │
│  ├─ Lab 1: LLM Fundamentals & API Usage                 │
│  └─ Lab 2: Prompt Engineering                           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  WEEK 2: RETRIEVAL-AUGMENTED GENERATION                 │
│  ├─ Lab 3: Document Processing & Embeddings             │
│  ├─ Lab 4: Semantic Search & Retrieval                  │
│  └─ Lab 5: Complete RAG Pipeline                        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  WEEK 3: AI AGENTS                                       │
│  ├─ Lab 6: AI Agents & Tool Calling                     │
│  ├─ Lab 7: Agent Memory & Planning                      │
│  └─ Lab 8: Advanced Multi-Agent Systems                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  WEEK 4: PRODUCTION & DEPLOYMENT                         │
│  └─ Guardrails, Safety & Best Practices                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🧪 Available Codelabs

### 🟢 Week 1: Foundations

#### Lab 1: LLM Fundamentals & API Usage
**Duration:** 90-120 minutes total | **Difficulty:** Beginner

Learn the foundations of Large Language Models and make your first API calls.

**📚 Learning (30 min)** - Theory and concepts
**🛠️ Hands-On Lab (60-90 min)** - Practical coding exercises

**What You'll Learn:**
- How LLMs work under the hood
- Understanding tokens and costs
- Making API calls (OpenAI, Claude, Gemini)
- Key parameters: temperature, top_p, max_tokens
- Streaming responses
- Building your first chatbot

**What You'll Build:**
- Token counter and cost calculator
- Temperature experiments
- SimpleChatbot class
- **SupportGenie v0.1** - Professional AI chatbot

**📚 Start with:** [Learning Material →](Lab1-LLM-Fundamentals/learning.md)
**🛠️ Then practice:** [Hands-On Lab →](Lab1-LLM-Fundamentals/lab.md)

---

#### Lab 2: Prompt Engineering
**Duration:** 100-140 minutes total | **Difficulty:** Beginner-Intermediate

Master the art of prompt engineering to get exceptional results from LLMs.

**📚 Learning (40 min)** - Theory and concepts
**🛠️ Hands-On Lab (60-90 min)** - Practical coding exercises

**What You'll Learn:**
- Anatomy of a perfect prompt (7-part structure)
- System messages for behavior control
- Few-shot learning techniques
- Chain-of-thought prompting
- Reusable prompt templates
- Edge case handling strategies
- Tone and style control

**What You'll Build:**
- Prompt quality experiments
- Few-shot classifiers (sentiment, intent)
- Chain-of-thought calculators
- Reusable PromptTemplate class
- Edge case handlers
- **SupportGenie v0.2** - Enhanced with advanced prompting

**📚 Start with:** [Learning Material →](Lab2-Prompt-Engineering/learning.md)
**🛠️ Then practice:** [Hands-On Lab →](Lab2-Prompt-Engineering/lab.md)

---

### 🟡 Week 2: RAG Systems

#### Lab 3: Document Processing & Embeddings
**Duration:** 95-125 minutes total | **Difficulty:** Intermediate

Process documents and generate semantic embeddings for intelligent search.

**📚 Learning (35 min)** - Theory and concepts
**🛠️ Hands-On Lab (60-90 min)** - Practical coding exercises

**What You'll Learn:**
- Why and how to chunk documents
- Text chunking strategies (fixed, sentence, paragraph, recursive)
- Understanding embeddings and semantic similarity
- Embedding models (HuggingFace, OpenAI)
- Vector databases with ChromaDB
- Cosine similarity calculations

**What You'll Build:**
- Document loaders (TXT, PDF)
- Multiple chunking implementations
- Embedding generation pipeline
- Vector database with ChromaDB
- Semantic similarity comparison tools
- **DocumentProcessor** - Complete reusable processing system

**📚 Start with:** [Learning Material →](Lab3-Document-Processing/learning.md)
**🛠️ Then practice:** [Hands-On Lab →](Lab3-Document-Processing/lab.md)

---

#### Lab 4: Semantic Search & Retrieval
**Duration:** 90-120 minutes total | **Difficulty:** Intermediate

Build semantic search engines that understand meaning, not just keywords.

**📚 Learning (30 min)** - Theory and concepts
**🛠️ Hands-On Lab (60-90 min)** - Practical coding exercises

**What You'll Learn:**
- Semantic search vs. traditional keyword search
- Distance metrics (L2, cosine similarity, dot product)
- Top-K retrieval strategies and optimization
- Metadata filtering for advanced queries
- Hybrid search combining semantic + keyword (BM25)
- Search strategy selection for different query types
- Reciprocal Rank Fusion (RRF)

**What You'll Build:**
- Semantic search engine with ChromaDB
- Top-K comparison and analysis tools
- Metadata filtering system
- BM25 keyword search implementation
- Hybrid search with configurable weights
- **ProductionSearchSystem** - Complete multi-strategy search system

**📚 Start with:** [Learning Material →](Lab4-Semantic-Search/learning.md)
**🛠️ Then practice:** [Hands-On Lab →](Lab4-Semantic-Search/lab.md)

---

#### Lab 5: Complete RAG Pipeline
**Duration:** 110-125 minutes total | **Difficulty:** Intermediate

Build end-to-end RAG systems that combine retrieval with generation.

**📚 Learning (35 min)** - Theory and concepts
**🛠️ Hands-On Lab (75-90 min)** - Practical coding exercises

**What You'll Learn:**
- What RAG is and why it matters
- Complete RAG architecture (Retrieve → Augment → Generate)
- The three steps of RAG in detail
- Prompt engineering for RAG systems
- RAG vs. non-RAG comparison
- RAG evaluation metrics and strategies
- Advanced RAG techniques (re-ranking, query rewriting, HyDE)

**What You'll Build:**
- Complete RAG pipeline with OpenAI
- RAG vs. non-RAG comparison tools
- Multiple prompt template strategies
- Multi-LLM support (OpenAI, Claude, Bedrock)
- RAG evaluation framework
- **SupportGenie v3.0** - Production RAG-powered support system

**📚 Start with:** [Learning Material →](Lab5-RAG-Pipeline/learning.md)
**🛠️ Then practice:** [Hands-On Lab →](Lab5-RAG-Pipeline/lab.md)

---

### 🔴 Week 3: AI Agents

#### Lab 6: AI Agents & Tool Calling
**Duration:** 130-160 minutes total | **Difficulty:** Intermediate-Advanced

Build autonomous AI agents that can use tools and make decisions.

**📚 Learning (40 min)** - Theory and concepts
**🛠️ Hands-On Lab (90-120 min)** - Practical coding exercises

**What You'll Learn:**
- Understanding AI agents vs chatbots
- Agent execution loop (Observe → Think → Act → Evaluate)
- Tool/function calling with OpenAI and Claude
- Multiple tool coordination and selection
- Conditional workflows and branching logic
- Agent design patterns (ReAct, Tool-Calling Loop, Plan-and-Execute)
- Production-ready error handling and fallbacks
- When to use agents vs traditional approaches

**What You'll Build:**
- Basic calculator agents (OpenAI + Claude)
- Multi-tool assistant with 5+ tools
- Conditional workflow agent with smart decision-making
- Resilient agent system with error handling
- **AgentHub v1.0** - Production multi-agent platform with routing

**📚 Start with:** [Learning Material →](Lab6-AI-Agents/learning.md)
**🛠️ Then practice:** [Hands-On Lab →](Lab6-AI-Agents/lab.md)

---

#### Lab 7: Agent Memory & Planning
**Duration:** 140-170 minutes total | **Difficulty:** Advanced

Implement memory systems and planning capabilities for agents.

**📚 Learning (40 min)** - Theory and concepts
**🛠️ Hands-On Lab (100-130 min)** - Practical coding exercises

**What You'll Learn:**
- Three types of agent memory (short-term, working, long-term)
- Conversation history management
- Working memory for task tracking
- Long-term memory with vector databases (ChromaDB)
- ReAct pattern (Reasoning + Acting) framework
- Thought → Action → Observation loops
- Task planning and decomposition
- Plan-then-execute pattern
- Self-reflection and error correction
- Dynamic replanning strategies

**What You'll Build:**
- Memory agent with complete memory systems
- ReAct agent with transparent reasoning
- Planning agent that strategizes before executing
- Reflective agent with self-evaluation
- **IntelliAgent v1.0** - Complete intelligent agent with memory + planning

**📚 Start with:** [Learning Material →](Lab7-Agent-Memory/learning.md)
**🛠️ Then practice:** [Hands-On Lab →](Lab7-Agent-Memory/lab.md)

---

#### Lab 8: Advanced Multi-Agent Systems
**Duration:** 165-195 minutes total | **Difficulty:** Advanced

Build enterprise-scale multi-agent systems with coordination and frameworks.

**📚 Learning (45 min)** - Theory and concepts
**🛠️ Hands-On Lab (120-150 min)** - Practical coding exercises

**What You'll Learn:**
- Research agent architecture and autonomous information gathering
- Agentic RAG vs traditional RAG (dynamic retrieval decisions)
- LangChain framework (@tool decorator, AgentExecutor, chains)
- LangGraph for complex multi-step workflows
- Multi-agent coordination patterns (hierarchical, sequential, parallel)
- Agent-to-agent communication and task delegation
- Production deployment strategies and monitoring
- When to use agents vs traditional approaches

**What You'll Build:**
- Autonomous research agent with web search capabilities
- Agentic RAG system with dynamic retrieval strategy
- LangChain-powered agents with streamlined tool integration
- Multi-agent system with specialized workers and coordinator
- **ResearchHub v1.0** - Production multi-agent research platform

**📚 Start with:** [Learning Material →](Lab8-Advanced-Agents/learning.md)
**🛠️ Then practice:** [Hands-On Lab →](Lab8-Advanced-Agents/lab.md)

---

## 🛠️ Prerequisites

Before starting these codelabs, ensure you have:

### Required
- ✅ **Python 3.8+** installed
- ✅ **Basic Python knowledge** (variables, functions, classes)
- ✅ **API Keys** from OpenAI and/or Anthropic
- ✅ **Internet connection** for API calls

### Recommended
- 💡 **Jupyter Notebook** or Google Colab access
- 💡 **VS Code** or similar code editor
- 💡 **Git** for cloning repositories
- 💡 **Virtual environment** knowledge

---

## 📦 Setup Instructions

### Step 1: Clone or Download Materials
```bash
# Option 1: Clone repository
git clone <repository-url>
cd AdvancedTraining/Codelabs-Format

# Option 2: Download and extract ZIP
```

### Step 2: Set Up Python Environment
```bash
# Create virtual environment
python -m venv genai-env

# Activate (Mac/Linux)
source genai-env/bin/activate

# Activate (Windows)
genai-env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install openai anthropic google-generativeai tiktoken
pip install langchain chromadb sentence-transformers
pip install jupyter notebook
```

### Step 4: Configure API Keys
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
GOOGLE_API_KEY=your-gemini-key-here
```

---

## 🎯 How to Use These Codelabs

### For Self-Paced Learning
1. **Start with Lab 1** - Follow the sequential order
2. **Complete each section** - Don't skip ahead
3. **Run all code examples** - Type them out, don't just copy
4. **Build the capstone projects** - Apply what you've learned
5. **Review and experiment** - Modify code to deepen understanding

### For Workshops/Classes
- Each lab is **self-contained** and can be completed in one session
- **Estimated times** help with scheduling
- **Prerequisites** listed for each lab
- **Checkpoints** throughout for group discussion
- **Capstone projects** work as assignments

### For Reference
- Each codelab includes **complete code examples**
- **Troubleshooting sections** for common issues
- **Best practices** highlighted throughout
- **Links to documentation** for deep dives

---

## 📚 Additional Resources

### Documentation
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Anthropic Claude Docs](https://docs.anthropic.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

### Community
- GitHub Discussions (for questions)
- Discord/Slack (for live help)
- Office Hours (scheduled sessions)

### Supplementary Materials
- Video walkthroughs (coming soon)
- Presentation slides (PDF format)
- Code repositories with solutions
- Additional practice exercises

---

## 🏆 Completion Tracking

Track your progress through the codelabs:

- [ ] Lab 1: LLM Fundamentals & API Usage
- [ ] Lab 2: Prompt Engineering
- [ ] Lab 3: Document Processing & Embeddings
- [ ] Lab 4: Semantic Search & Retrieval
- [ ] Lab 5: Complete RAG Pipeline
- [ ] Lab 6: AI Agents & Tool Calling
- [ ] Lab 7: Agent Memory & Planning
- [ ] Lab 8: Advanced Multi-Agent Systems

---

## 💡 Tips for Success

### ⚡️ Learning Tips
- **Practice daily** - Even 30 minutes makes a difference
- **Build projects** - Apply concepts to real problems
- **Read error messages** - They're teaching opportunities
- **Experiment** - Modify code to see what happens
- **Ask questions** - Use discussion forums

### 🚨 Common Pitfalls to Avoid
- ❌ Skipping the setup steps
- ❌ Rushing through without understanding
- ❌ Not running the code examples
- ❌ Ignoring best practices
- ❌ Hardcoding API keys in code

### ✅ Best Practices
- ✅ Use version control (Git)
- ✅ Keep API keys in environment variables
- ✅ Test code in small increments
- ✅ Document your own learning
- ✅ Build a portfolio of projects

---

## 🔄 Updates and Versions

**Current Version:** v1.1 (Codelabs Format)
**Last Updated:** December 2025
**Format:** Google Codelabs-inspired interactive tutorials

### Changelog
- **v1.1** - Initial Codelabs format release
- Added interactive step-by-step structure
- Included visual diagrams and callouts
- Enhanced code examples with explanations

---

## 🤝 Contributing

Found an issue or want to improve these codelabs?
- Report bugs via GitHub Issues
- Suggest improvements via Pull Requests
- Share your projects built with these materials

---

## 📄 License

These materials are provided for educational purposes.
Please check the repository LICENSE file for details.

---

## 🚀 Ready to Start?

**Begin your AI engineering journey with Lab 1!**

👉 [Start Lab 1: LLM Fundamentals & API Usage →](Lab1-LLM-Fundamentals/codelab.md)

---

**Questions?** Check the FAQ or join our community discussions.

**Need help?** Each lab includes troubleshooting sections and additional resources.

**Want to go deeper?** Explore the advanced topics and optional challenges in each lab.

---

*Transform from AI beginner to enterprise AI engineer in 4 weeks.*
*Let's build the future together! 🌟*
