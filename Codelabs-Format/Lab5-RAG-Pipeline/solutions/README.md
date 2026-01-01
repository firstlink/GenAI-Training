# Lab 5 Solutions: Complete RAG Pipeline

## 📚 Overview

Complete solutions for Lab 5 covering the full RAG (Retrieval-Augmented Generation) pipeline, multi-LLM integration, prompt engineering, and production deployment. Build a complete RAG-powered support system from scratch.

---

## 📁 Files Included

### Core Exercises
- **`exercise1_basic_rag.py`** - Basic RAG pipeline implementation
- **`exercise2_rag_comparison.py`** - RAG vs non-RAG comparison
- **`exercise3_prompt_strategies.py`** - Advanced prompt engineering
- **`exercise4_multi_llm.py`** - Multi-LLM provider support

### Capstone Project
- **`capstone_support_genie_v3.py`** - Production RAG support system

### Environment Setup
- **`.env.example`** - Example environment configuration

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install required libraries
pip install sentence-transformers chromadb openai anthropic python-dotenv numpy
```

### Environment Setup

Create a `.env` file in the solutions directory:

```bash
# Required: At least one LLM provider
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here  # Optional
```

### Run Exercises

```bash
# Exercise 1: Basic RAG Pipeline
python exercise1_basic_rag.py

# Exercise 2: RAG Comparison
python exercise2_rag_comparison.py

# Exercise 3: Prompt Strategies
python exercise3_prompt_strategies.py

# Exercise 4: Multi-LLM Support
python exercise4_multi_llm.py

# Capstone: SupportGenie v3.0
python capstone_support_genie_v3.py
```

---

## 📖 Exercise Guide

### Exercise 1: Basic RAG Pipeline (20 min)

**What you'll learn:**
- Build complete RAG pipeline (Retrieve → Augment → Generate)
- Integrate ChromaDB with LLMs
- Create effective RAG prompts
- Understand RAG workflow

**Key Implementation:**
```python
from exercise1_basic_rag import BasicRAGPipeline, setup_sample_knowledge_base

# Setup
client, collection, embedding_model = setup_sample_knowledge_base()

# Create RAG pipeline
rag = BasicRAGPipeline(collection, embedding_model, openai_api_key)

# Query with RAG
response = rag.query("What is machine learning?", n_results=3)
```

**RAG Pipeline Steps:**
1. **Retrieve**: Semantic search in vector database
2. **Augment**: Combine retrieved context with query in prompt
3. **Generate**: LLM generates answer based on context

**Sample Output:**
```
RAG ANSWER:
Machine learning is a subset of AI that enables computers to learn from data
without explicit programming [1]. It uses statistical techniques to give
computers the ability to learn and improve from experience [1].

Metadata:
  Model: gpt-4o-mini
  Tokens: 245
  Chunks: 3
  Avg relevance: 0.3421
  Time: 1.23s
```

**Key Features:**
- ✅ Semantic retrieval with embeddings
- ✅ Contextual prompt creation
- ✅ Source citations [1], [2], [3]
- ✅ Performance tracking

---

### Exercise 2: RAG vs Non-RAG Comparison (15 min)

**What you'll learn:**
- Compare RAG and non-RAG responses
- Understand when RAG adds value
- Analyze response quality differences
- Optimize for use cases

**Usage:**
```python
from exercise2_rag_comparison import RAGComparator

comparator = RAGComparator(rag_pipeline, openai_api_key)

# Side-by-side comparison
comparison = comparator.compare("What is deep learning?")

# Batch comparison
results = comparator.batch_compare([
    "What is machine learning?",
    "Explain neural networks",
    "How does NLP work?"
])
```

**Comparison Output:**
```
WITHOUT RAG (LLM Only):
Deep learning is a subset of machine learning that uses neural networks...
[Generic, general knowledge response]
Tokens: 180

WITH RAG (LLM + Context):
According to the context, deep learning uses neural networks with multiple
layers [1]. These networks can automatically learn hierarchical
representations [1]...
Tokens: 285 | Chunks: 3 | Citations: ✅
```

**When RAG Helps Most:**
- ✅ Domain-specific queries
- ✅ Recent or proprietary information
- ✅ Need for source citations
- ✅ Grounding in specific documents

**When RAG Helps Less:**
- ⚠️ General knowledge questions
- ⚠️ Topics outside knowledge base
- ⚠️ Speed-critical applications

---

### Exercise 3: Advanced Prompt Strategies (15 min)

**What you'll learn:**
- Create multiple prompt templates
- Compare prompt effectiveness
- Format context with metadata
- Optimize for different use cases

**Available Templates:**
```python
from exercise3_prompt_strategies import RAGPromptTemplates

# 6 different template styles
templates = {
    "basic": RAGPromptTemplates.basic_template,
    "detailed": RAGPromptTemplates.detailed_template,
    "structured": RAGPromptTemplates.structured_template,
    "conversational": RAGPromptTemplates.conversational_template,
    "chain_of_thought": RAGPromptTemplates.chain_of_thought_template,
    "expert": RAGPromptTemplates.expert_template
}
```

**Template Comparison:**

| Template | Use Case | Citations | Format |
|----------|----------|-----------|--------|
| **Basic** | Quick prototypes | ❌ | Simple |
| **Detailed** | Production | ✅ | Structured |
| **Structured** | APIs | ✅ | Fixed format |
| **Conversational** | Customer-facing | ✅ | Friendly |
| **Chain-of-Thought** | Complex reasoning | ✅ | Step-by-step |
| **Expert** | Domain expertise | ✅ | Authoritative |

**Custom Template Example:**
```python
custom_prompt = create_custom_template(
    query="Explain neural networks",
    context_chunks=chunks,
    style="professional",  # or "casual", "academic"
    include_metadata=True
)
```

**Best Practices:**
- ✓ Use "detailed" template for production
- ✓ Include explicit citation instructions
- ✓ Match style to audience
- ✓ Add metadata when it provides value
- ✓ Test multiple templates for your use case

---

### Exercise 4: Multi-LLM Support (15 min)

**What you'll learn:**
- Integrate multiple LLM providers (OpenAI, Claude)
- Create unified RAG interface
- Compare provider performance
- Implement fallback strategies

**Usage:**
```python
from exercise4_multi_llm import UnifiedRAGPipeline

# Initialize with auto-detection
unified_rag = UnifiedRAGPipeline(collection, embedding_model)

# Query with OpenAI
result = unified_rag.query(
    "What is machine learning?",
    provider="openai",
    model="gpt-4o-mini"
)

# Query with Claude
result = unified_rag.query(
    "What is machine learning?",
    provider="claude",
    model="claude-3-5-haiku-20241022"
)
```

**Provider Comparison:**

| Provider | Model | Speed | Cost/1M tokens | Best For |
|----------|-------|-------|----------------|----------|
| OpenAI | gpt-4o-mini | Fast | $0.15-0.60 | High volume |
| OpenAI | gpt-4o | Medium | $2.50-10.00 | Complex tasks |
| Claude | haiku | Fast | $0.80-4.00 | Instruction following |
| Claude | sonnet | Medium | $3.00-15.00 | Reasoning |

**Performance Metrics:**
```python
from exercise4_multi_llm import LLMComparator

comparator = LLMComparator(unified_rag)
results = comparator.compare_providers(
    "Explain deep learning",
    providers=["openai", "claude"]
)
```

**Production Tips:**
- ✓ Implement fallback providers
- ✓ Use faster models for simple queries
- ✓ Monitor costs and latency
- ✓ Cache common responses
- ✓ Load balance across providers

---

### 🏆 Capstone: SupportGenie v3.0

**What you'll build:**
Production-ready RAG-powered customer support system with full capabilities.

**Features:**
✅ Complete RAG pipeline (Retrieve → Augment → Generate)
✅ Multi-LLM support (OpenAI, Claude)
✅ Conversation history management
✅ Source citations in all responses
✅ Performance metrics tracking
✅ Error handling and fallbacks
✅ Interactive chat mode
✅ Export conversation history

**Usage:**
```python
from capstone_support_genie_v3 import SupportGenieV3, setup_support_knowledge_base

# Setup
client, collection, embedding_model = setup_support_knowledge_base()

# Initialize SupportGenie
genie = SupportGenieV3(
    collection=collection,
    embedding_model=embedding_model,
    llm_provider="openai"  # or "claude"
)

# Chat
response = genie.chat("How do I reset my password?")
genie.display_response(response)

# Interactive mode
genie.interactive_mode()
```

**Interactive Commands:**
- Regular text: Ask a question
- `:history` - Show conversation history
- `:stats` - Show usage statistics
- `:clear` - Clear conversation history
- `:quit` - Exit

**Sample Interaction:**
```
👤 You: How do I reset my password?

🤖 SupportGenie:
──────────────────────────────────────────────────────────────────────
To reset your password, click on the 'Forgot Password' link on the
login page [1]. Enter your email address, and we'll send you a password
reset link [1]. Please note that the link expires after 24 hours for
security reasons [1].

📊 Response Metadata:
   Provider: openai
   Model: gpt-4o-mini
   Chunks retrieved: 3
   Avg relevance: 0.2841
   Tokens used: 187
   Latency: 1243ms
   Citations: ✅ Yes

📚 Knowledge Base Sources:
   • Topic: password
──────────────────────────────────────────────────────────────────────
```

**Knowledge Base:**
- 10 support documents
- Topics: refund, password, pricing, shipping, support, billing, security
- Categories: billing, account, orders, general, technical

**Architecture:**
```
User Query
    ↓
[Retrieve Context] → ChromaDB (semantic search)
    ↓
[Create Prompt] → RAG template with context
    ↓
[Generate Response] → OpenAI/Claude LLM
    ↓
[Track Metrics] → Tokens, latency, relevance
    ↓
[Display Answer] → With citations and metadata
```

---

## 💡 Key Concepts

### RAG Pipeline

**The Three Steps:**

1. **Retrieve**
   - Embed user query
   - Search vector database
   - Retrieve top-K relevant chunks

2. **Augment**
   - Combine query + context in prompt
   - Add instructions for citations
   - Format with metadata

3. **Generate**
   - Send prompt to LLM
   - Get contextually-grounded answer
   - Extract and display sources

**Why RAG?**
- ✅ Grounds LLM in factual context
- ✅ Reduces hallucinations
- ✅ Enables source citations
- ✅ Works with proprietary data
- ✅ No model retraining needed
- ✅ Dynamically updatable knowledge

### Prompt Engineering for RAG

**Essential Elements:**
1. **System Instructions**: Define role and behavior
2. **Context Section**: Formatted retrieved chunks
3. **User Question**: The actual query
4. **Response Guidelines**: Citation format, structure
5. **Tone/Style**: Professional, casual, academic

**Citation Best Practices:**
```
Bad:  "Deep learning uses neural networks."
Good: "Deep learning uses neural networks [1]."
Best: "According to the context, deep learning uses neural networks
       with multiple layers [1] that can learn hierarchical
       representations [1]."
```

### RAG vs Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Cost** | Low (inference only) | High (training + inference) |
| **Speed** | Medium (retrieval overhead) | Fast |
| **Updates** | Instant (update DB) | Slow (retrain) |
| **Use Case** | Dynamic knowledge | Static behavior |
| **Complexity** | Medium | High |
| **Citations** | Easy (built-in) | Difficult |

---

## 🎯 Best Practices

### Retrieval Optimization

✅ **Use appropriate n_results:**
- Simple queries: 2-3 chunks
- Complex queries: 5-7 chunks
- Comprehensive: 8-10 chunks

✅ **Monitor relevance scores:**
- < 0.5: Excellent match
- 0.5-1.0: Good match
- > 1.0: Weak match (consider excluding)

✅ **Implement metadata filtering:**
```python
response = genie.chat(
    "What's your refund policy?",
    metadata_filter={"category": "billing"}
)
```

### Prompt Optimization

✅ **Be explicit about citations:**
- Include citation format in instructions
- Request specific section references
- Penalize non-cited claims

✅ **Structure prompts clearly:**
- Separate context from question
- Use clear section headers
- Include formatting guidelines

✅ **Match temperature to use case:**
- Factual answers: 0.0-0.3
- Creative responses: 0.7-0.9
- Balanced: 0.5

### Performance Optimization

✅ **Cache embeddings:**
```python
# Cache query embeddings for common questions
query_cache = {}
if query in query_cache:
    embedding = query_cache[query]
else:
    embedding = model.encode(query)
    query_cache[query] = embedding
```

✅ **Use faster models for simple queries:**
- Query classification → route to appropriate model
- Simple lookup → gpt-4o-mini
- Complex reasoning → gpt-4o or claude-sonnet

✅ **Batch processing:**
```python
# Process multiple queries in parallel
with ThreadPoolExecutor() as executor:
    responses = executor.map(genie.chat, queries)
```

### Error Handling

✅ **Graceful degradation:**
```python
try:
    response = genie.chat(query)
except RetrievalError:
    # Fall back to non-RAG
    response = llm.generate(query)
except GenerationError:
    # Use cached response or fallback LLM
    response = fallback_llm.generate(query)
```

✅ **Monitor and alert:**
- Track failure rates
- Monitor latency
- Alert on degraded performance

---

## 🔧 Troubleshooting

**Issue:** "OPENAI_API_KEY not found"
**Solution:** Create `.env` file with your API key:
```bash
OPENAI_API_KEY=sk-your-key-here
```

**Issue:** "No retrieval results found"
**Solution:**
- Check if knowledge base is populated
- Verify query is relevant to documents
- Lower n_results or adjust threshold

**Issue:** "Responses don't include citations"
**Solution:**
- Use "detailed" template with explicit citation instructions
- Lower temperature (0.1-0.3)
- Add examples in prompt

**Issue:** "Slow response times"
**Solution:**
- Reduce n_results (fewer chunks)
- Use faster model (gpt-4o-mini)
- Cache common queries
- Consider async processing

**Issue:** "High token usage"
**Solution:**
- Reduce chunk size in knowledge base
- Decrease n_results
- Use more concise prompts
- Implement smart chunk selection

---

## 📊 Performance Benchmarks

### Typical Latencies (with 3 chunks):

| Component | Time | % of Total |
|-----------|------|-----------|
| Retrieval | 50-100ms | 5-10% |
| Prompt creation | 5-10ms | < 1% |
| LLM generation | 1000-2000ms | 85-90% |
| Post-processing | 5-10ms | < 1% |
| **Total** | **1100-2200ms** | **100%** |

### Token Usage (per query):

| Component | Tokens | % of Total |
|-----------|--------|-----------|
| System prompt | 50-100 | 15-20% |
| Context (3 chunks) | 200-400 | 50-60% |
| User question | 10-30 | 5-10% |
| Response | 100-200 | 20-30% |
| **Total** | **400-700** | **100%** |

### Scaling Considerations:

- **100 queries/day**: $0.03-0.07 (gpt-4o-mini)
- **1,000 queries/day**: $0.30-0.70
- **10,000 queries/day**: $3.00-7.00
- **100,000 queries/day**: $30-70

---

## 🧪 Testing

### Validate All Solutions:
```bash
cd solutions/
for file in exercise*.py capstone*.py; do
    python3 -m py_compile "$file" && echo "✅ $file"
done
```

### Run Test Suite:
```bash
# Test basic RAG
python exercise1_basic_rag.py

# Test comparison
python exercise2_rag_comparison.py

# Test prompts
python exercise3_prompt_strategies.py

# Test multi-LLM
python exercise4_multi_llm.py

# Test capstone
python capstone_support_genie_v3.py
```

---

## 📚 Next Steps

After completing Lab 5:
1. **Lab 6:** AI Agents & Tool Calling
2. **Lab 7:** Advanced RAG Techniques
3. Deploy RAG system to production
4. Build domain-specific RAG applications

---

## 🎓 What You've Learned

✅ Complete RAG pipeline implementation
✅ Retrieval optimization strategies
✅ Advanced prompt engineering for RAG
✅ Multi-LLM provider integration
✅ Production deployment patterns
✅ Performance monitoring and optimization
✅ Error handling and fallbacks
✅ Conversation management

---

## 🌟 Advanced Topics

### Query Rewriting

Improve retrieval by rewriting user queries:
```python
def rewrite_query(original_query, llm):
    prompt = f"""Rewrite this query for better document retrieval.
    Make it more specific and include key terms.

    Original: {original_query}
    Rewritten:"""

    return llm.generate(prompt)
```

### Multi-Query RAG

Retrieve with multiple query variations:
```python
def multi_query_rag(query, llm, rag_pipeline):
    # Generate variations
    variations = llm.generate_variations(query, n=3)

    # Retrieve for each
    all_docs = []
    for var in variations:
        docs = rag_pipeline.retrieve(var)
        all_docs.extend(docs)

    # Deduplicate and use top results
    unique_docs = deduplicate(all_docs)
    return rag_pipeline.generate(query, unique_docs)
```

### Hybrid Search

Combine semantic and keyword search:
```python
from rank_bm25 import BM25Okapi

def hybrid_search(query, documents, embeddings, alpha=0.5):
    # Semantic search
    semantic_scores = cosine_similarity(query_emb, embeddings)

    # Keyword search (BM25)
    bm25 = BM25Okapi(tokenized_docs)
    keyword_scores = bm25.get_scores(query_tokens)

    # Combine scores
    hybrid_scores = alpha * semantic_scores + (1-alpha) * keyword_scores

    return top_k(hybrid_scores)
```

---

**Ready for Production! 🚀**

*You now have the skills to build and deploy complete RAG systems for real-world applications.*
