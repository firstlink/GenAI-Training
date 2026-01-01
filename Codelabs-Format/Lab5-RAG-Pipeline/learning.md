# Lab 5: Complete RAG Pipeline

## 📚 Learning Material

**Duration:** 35 minutes
**Difficulty:** Intermediate
**Prerequisites:** Labs 1-4 completed

---

## 🎯 Learning Objectives

By the end of this learning module, you will understand:
- ✅ What RAG (Retrieval-Augmented Generation) is and why it matters
- ✅ The complete RAG pipeline architecture
- ✅ How to combine retrieval with generation effectively
- ✅ Prompt engineering for RAG systems
- ✅ RAG vs. non-RAG tradeoffs
- ✅ RAG evaluation metrics and strategies
- ✅ Common RAG challenges and solutions

---

## 📖 Table of Contents

1. [Introduction: The RAG Revolution](#1-introduction-the-rag-revolution)
2. [RAG Architecture](#2-rag-architecture)
3. [The Three Steps of RAG](#3-the-three-steps-of-rag)
4. [Prompt Engineering for RAG](#4-prompt-engineering-for-rag)
5. [RAG vs. Non-RAG](#5-rag-vs-non-rag)
6. [RAG Evaluation](#6-rag-evaluation)
7. [Advanced RAG Techniques](#7-advanced-rag-techniques)
8. [Review & Key Takeaways](#8-review--key-takeaways)

---

## 1. Introduction: The RAG Revolution

### The Problem with LLMs Alone

**LLM Knowledge Limitations:**

```
Question: "What's our company's return policy for laptops?"

LLM Without RAG:
❌ "I don't have access to your specific company policies."
❌ Or worse: Hallucinates a policy that doesn't exist!

LLM With RAG:
✅ Retrieves actual company policy document
✅ "According to your policy, laptops can be returned within
    30 days if unopened, or 14 days if opened, with original
    packaging and receipt."
```

### What is RAG?

**RAG = Retrieval-Augmented Generation**

A technique that enhances LLM responses by:
1. **Retrieving** relevant information from your documents
2. **Augmenting** the prompt with that information
3. **Generating** answers grounded in your data

```
┌────────────────────────────────────────────────────────┐
│  TRADITIONAL LLM (No RAG)                              │
├────────────────────────────────────────────────────────┤
│                                                        │
│  User Question → LLM → Answer (from training data)    │
│                                                        │
│  Limitations:                                          │
│  - Training data cutoff                                │
│  - No company-specific knowledge                       │
│  - Can hallucinate                                     │
│                                                        │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  RAG-ENHANCED LLM                                      │
├────────────────────────────────────────────────────────┤
│                                                        │
│  User Question → Search Vector DB → Retrieve Docs     │
│                          ↓                             │
│                  Combine Question + Docs              │
│                          ↓                             │
│                      LLM with Context                  │
│                          ↓                             │
│                  Answer (grounded in your data)        │
│                                                        │
│  Benefits:                                             │
│  ✓ Uses YOUR documents                                │
│  ✓ Always up-to-date (as docs update)                 │
│  ✓ Reduces hallucinations                             │
│  ✓ Can cite sources                                   │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Why RAG Matters

**1. Current Information**
```
Without RAG: "I was trained on data up to April 2024..."
With RAG: Uses your documents updated today!
```

**2. Domain-Specific Knowledge**
```
Without RAG: General medical advice
With RAG: Your hospital's specific protocols and procedures
```

**3. Reduced Hallucinations**
```
Without RAG: Makes up plausible-sounding but wrong facts
With RAG: Constrained to information in retrieved documents
```

**4. Verifiable Answers**
```
Without RAG: No way to verify source
With RAG: "According to document X, section Y..."
```

**5. Privacy & Control**
```
Without RAG: All knowledge is baked into model
With RAG: You control what documents are accessible
```

**⏱️ Duration so far:** 5 minutes

---

## 2. RAG Architecture

### The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  OFFLINE PREPARATION (Done Once)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Your Documents → Chunk → Embed → Store in Vector DB       │
│                                                             │
│  [Lab 3: Document Processing]                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  ONLINE QUERY (Every User Question)                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: USER QUESTION                                     │
│  "What is machine learning?"                                │
│                    ↓                                        │
│  Step 2: RETRIEVE (Lab 4: Semantic Search)                 │
│  ┌─────────────────────────────────────┐                  │
│  │ Embed question                       │                  │
│  │ Search vector database               │                  │
│  │ Get top-K similar chunks             │                  │
│  └─────────────────────────────────────┘                  │
│                    ↓                                        │
│  Retrieved Context:                                         │
│  [1] "Machine learning is a subset of AI..."               │
│  [2] "ML algorithms learn from data..."                    │
│  [3] "Common ML types include supervised..."               │
│                    ↓                                        │
│  Step 3: AUGMENT (Combine question + context)              │
│  ┌─────────────────────────────────────┐                  │
│  │ Create prompt:                       │                  │
│  │ "Answer using this context:          │                  │
│  │  [1] Machine learning is...          │                  │
│  │  [2] ML algorithms learn...          │                  │
│  │  [3] Common ML types...              │                  │
│  │                                       │                  │
│  │  Question: What is machine learning?"│                  │
│  └─────────────────────────────────────┘                  │
│                    ↓                                        │
│  Step 4: GENERATE (LLM creates answer)                     │
│  ┌─────────────────────────────────────┐                  │
│  │ Send to LLM (GPT-4, Claude, etc.)   │                  │
│  │ LLM reads context and question       │                  │
│  │ Generates grounded answer            │                  │
│  └─────────────────────────────────────┘                  │
│                    ↓                                        │
│  Step 5: RESPONSE                                           │
│  "Based on the provided context, machine learning is       │
│   a subset of AI that allows algorithms to learn from      │
│   data without explicit programming [1][2]..."             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Components Breakdown

**1. Vector Database** (ChromaDB, Pinecone, Weaviate)
- Stores: Document chunks + embeddings + metadata
- Provides: Fast semantic search

**2. Embedding Model** (Sentence-Transformers, OpenAI)
- Converts: Text → numerical vectors
- Same model for documents and queries!

**3. Retrieval System** (Semantic, Keyword, Hybrid)
- Finds: Most relevant chunks for query
- Returns: Top-K results with scores

**4. Prompt Constructor**
- Combines: Retrieved context + user question
- Formats: Clear instructions for LLM

**5. LLM** (GPT-4, Claude, Llama, etc.)
- Reads: Context and question
- Generates: Answer based on provided information

**⏱️ Duration so far:** 10 minutes

---

## 3. The Three Steps of RAG

### Step 1: Retrieve

**Goal:** Find the most relevant information for the user's question.

```python
# Simplified example
query = "What is our return policy?"

# 1. Convert query to embedding
query_embedding = embedding_model.encode(query)

# 2. Search vector database
results = vector_db.search(query_embedding, top_k=3)

# Results:
# [1] "Our return policy allows 30-day returns..." (distance: 0.15)
# [2] "To initiate a return, contact support..." (distance: 0.28)
# [3] "Refunds are processed within 5-7 days..." (distance: 0.35)
```

**Key Decisions:**
- **How many chunks to retrieve (K)?** Typically 3-5 for balance
- **Which search method?** Semantic, keyword, or hybrid
- **Filters?** Limit to specific document types, dates, etc.

### Step 2: Augment

**Goal:** Combine retrieved information with the user's question in a clear prompt.

**Bad Augmentation:**
```
Context: [chunk1] [chunk2] [chunk3]
Question: What is our return policy?
```
❌ No structure, LLM might ignore context

**Good Augmentation:**
```
You are a helpful AI assistant. Answer the user's question using ONLY
the context provided below.

CONTEXT:
[1] Our return policy allows customers to return unused items within
    30 days of purchase...

[2] To initiate a return, contact support@company.com with your order
    number...

[3] Refunds are processed within 5-7 business days after we receive
    the returned item...

QUESTION: What is our return policy?

INSTRUCTIONS:
- Base your answer ONLY on the context above
- Cite which context section(s) you used (e.g., [1], [2])
- If the context doesn't fully answer the question, say so
- Be concise and accurate

ANSWER:
```
✅ Clear structure, explicit instructions, encourages citation

**Augmentation Template Structure:**

```
┌──────────────────────────────────────────┐
│  1. SYSTEM ROLE                          │
│     "You are a helpful assistant..."     │
├──────────────────────────────────────────┤
│  2. CONTEXT                              │
│     "Use this information:               │
│      [1] chunk 1                         │
│      [2] chunk 2                         │
│      [3] chunk 3"                        │
├──────────────────────────────────────────┤
│  3. USER QUESTION                        │
│     "Question: {user_query}"             │
├──────────────────────────────────────────┤
│  4. INSTRUCTIONS                         │
│     "- Only use context                  │
│      - Cite sources                      │
│      - Be concise"                       │
├──────────────────────────────────────────┤
│  5. OUTPUT INDICATOR                     │
│     "Answer:"                            │
└──────────────────────────────────────────┘
```

### Step 3: Generate

**Goal:** LLM reads context and generates accurate answer.

**LLM receives:**
```
[Complete prompt with context and question]
```

**LLM generates:**
```
Based on the provided context, our return policy allows customers to
return unused items within 30 days of purchase [1]. To initiate a
return, you should contact support@company.com with your order number
[2]. Once we receive your returned item, refunds are processed within
5-7 business days [3].
```

**Key Parameters:**
- **Temperature:** Low (0.1-0.3) for factual consistency
- **Max tokens:** Enough for complete answer (300-500)
- **Model:** GPT-4, Claude 3.5, or similar for best quality

**⏱️ Duration so far:** 18 minutes

---

## 4. Prompt Engineering for RAG

### RAG Prompt Principles

**1. Explicit Context Boundaries**

```
❌ Bad:
"Here's some info: [context]. Question: [query]"

✅ Good:
"Context information is below.
---------------------
{context}
---------------------
Given the context information above, answer the question: {query}"
```

**2. Clear Instructions**

```
✅ Essential Instructions:
- "Use ONLY the information from the context"
- "If the context doesn't contain the answer, say 'I don't have enough information'"
- "Cite which context section you used (e.g., [1], [2])"
- "Be concise and accurate"
- "Do not make assumptions beyond the context"
```

**3. Encourage Source Citation**

```
✅ Prompt Addition:
"When answering, cite which context sections you used.
Format: [1] for first section, [2] for second, etc."

Example Output:
"According to the documentation, the return window is 30 days [1].
You can initiate returns through the customer portal [2]."
```

**4. Handle Insufficient Context**

```
✅ Prompt Addition:
"If the context doesn't provide enough information to fully answer
the question, explicitly state what information is missing."

Example Output:
"Based on the context, I can confirm returns are accepted within
30 days [1], but I don't have information about the condition
requirements for returned items."
```

### RAG Prompt Template (Production-Ready)

```python
RAG_PROMPT_TEMPLATE = """
You are an AI assistant helping users find information from documents.

**Context Information:**
The following context sections are provided to help answer the question.
Each section is numbered for reference.

{context_sections}

**User Question:**
{user_question}

**Instructions:**
1. Answer the question using ONLY the information from the context above
2. Cite the context section(s) you used (e.g., [1], [2], [3])
3. If multiple sections are relevant, synthesize the information
4. If the context doesn't provide enough information, say so explicitly
5. Do NOT use your general knowledge - stick to the provided context
6. Be concise, accurate, and helpful

**Answer:**
"""
```

### Context Formatting Strategies

**Strategy 1: Numbered Sections**
```
[1] Machine learning is a subset of artificial intelligence...
[2] Common ML algorithms include decision trees, neural networks...
[3] ML applications range from image recognition to...
```

**Strategy 2: Source Attribution**
```
Source: "ML_Guide.pdf" (Page 3)
Machine learning is a subset of artificial intelligence...

Source: "ML_Guide.pdf" (Page 7)
Common ML algorithms include decision trees...
```

**Strategy 3: Metadata-Rich**
```
[Document: Employee_Handbook.pdf | Section: Benefits | Last Updated: 2024-01-15]
Employees are eligible for health insurance after 90 days...

[Document: Employee_Handbook.pdf | Section: PTO | Last Updated: 2024-01-15]
Full-time employees accrue 15 days of PTO per year...
```

**⏱️ Duration so far:** 25 minutes

---

## 5. RAG vs. Non-RAG

### When to Use RAG

```
┌──────────────────────────┬────────────┬──────────────┐
│  Scenario                │  Use RAG?  │  Why         │
├──────────────────────────┼────────────┼──────────────┤
│ Company-specific info    │  ✓ YES     │  Not in      │
│ (policies, procedures)   │            │  training    │
├──────────────────────────┼────────────┼──────────────┤
│ Recent information       │  ✓ YES     │  After       │
│ (last month's data)      │            │  cutoff      │
├──────────────────────────┼────────────┼──────────────┤
│ General knowledge        │  ✗ NO      │  LLM already │
│ ("What is Python?")      │            │  knows       │
├──────────────────────────┼────────────┼──────────────┤
│ Legal documents          │  ✓ YES     │  Need exact  │
│ (contracts, terms)       │            │  language    │
├──────────────────────────┼────────────┼──────────────┤
│ Creative writing         │  ✗ NO      │  No factual  │
│ (stories, poems)         │            │  grounding   │
├──────────────────────────┼────────────┼──────────────┤
│ Technical documentation  │  ✓ YES     │  Specific    │
│ (API docs, manuals)      │            │  versions    │
├──────────────────────────┼────────────┼──────────────┤
│ Math calculations        │  ✗ NO      │  Logic-based │
│ (no document needed)     │            │  reasoning   │
└──────────────────────────┴────────────┴──────────────┘
```

### RAG Advantages

✅ **Grounded in Facts**
- Answers come from your documents
- Reduces hallucinations significantly

✅ **Always Current**
- Update documents → answers update automatically
- No need to retrain the model

✅ **Source Attribution**
- Can cite where information came from
- Verifiable and trustworthy

✅ **Domain-Specific**
- Works with specialized knowledge
- Handles company jargon and terminology

✅ **Cost-Effective Updates**
- Change documents, not the model
- No expensive retraining

### RAG Limitations

❌ **Retrieval Quality Dependent**
- Bad retrieval → bad answers
- "Garbage in, garbage out"

❌ **Context Window Limits**
- Can only include limited chunks
- Might miss important information split across many documents

❌ **Slower than Non-RAG**
- Extra retrieval step adds latency
- ~100-500ms additional time

❌ **Requires Infrastructure**
- Need vector database
- Need embedding model
- More complex system

❌ **Not Good for Reasoning**
- Works for facts, not complex logic
- Multi-step reasoning can be challenging

### Side-by-Side Example

**Question:** "What's the return policy for opened electronics?"

**Without RAG:**
```
LLM Response:
"Typically, most retailers have a 14-30 day return window for
electronics. Opened items may have restocking fees of 10-15%.
You should check with the specific retailer for their policy."

Issues:
❌ Generic answer
❌ No specific information about YOUR policy
❌ Makes assumptions ("typically", "may have")
```

**With RAG:**
```
Retrieved Context:
[1] "Electronics can be returned within 30 days if unopened..."
[2] "Opened electronics are accepted for return within 14 days..."
[3] "All returns require original packaging and receipt..."

RAG Response:
"Based on our return policy, opened electronics can be returned
within 14 days [2]. All returns require the original packaging
and receipt [3]. Note that unopened electronics have a longer
return window of 30 days [1]."

Benefits:
✅ Specific to YOUR company
✅ Accurate time frames
✅ Cites sources
✅ No assumptions
```

**⏱️ Duration so far:** 30 minutes

---

## 6. RAG Evaluation

### How to Measure RAG Quality

**Three Components to Evaluate:**

1. **Retrieval Quality** - Did we get the right documents?
2. **Generation Quality** - Is the answer good given the context?
3. **End-to-End Quality** - Is the final answer correct and helpful?

### Retrieval Metrics

**1. Precision@K**
```
Precision = (Relevant chunks retrieved) / (Total chunks retrieved)

Example:
Retrieved 5 chunks, 3 are relevant
Precision@5 = 3/5 = 0.6 or 60%
```

**2. Recall@K**
```
Recall = (Relevant chunks retrieved) / (Total relevant chunks in DB)

Example:
Database has 10 relevant chunks, retrieved 3
Recall@10 = 3/10 = 0.3 or 30%
```

**3. MRR (Mean Reciprocal Rank)**
```
MRR = 1 / (Rank of first relevant result)

Example:
First relevant chunk is at position 2
MRR = 1/2 = 0.5
```

### Generation Metrics

**1. Faithfulness**
- Does the answer stick to the provided context?
- Metric: % of claims that can be traced to context

**2. Answer Relevance**
- Does the answer address the question?
- Metric: Human rating or automated similarity score

**3. Context Utilization**
- Does the answer use the retrieved context?
- Metric: Citation count, context overlap

### Simple Evaluation Framework

```python
def evaluate_rag_answer(question, answer, context, expected_info):
    """
    Simple RAG evaluation

    Returns:
        dict: Evaluation scores
    """
    scores = {}

    # 1. Citation check
    citations = count_citations(answer)  # [1], [2], etc.
    scores['uses_citations'] = citations > 0

    # 2. Context usage
    scores['uses_context'] = any(chunk_text in answer
                                  for chunk_text in context)

    # 3. Answers question
    scores['answers_question'] = expected_info.lower() in answer.lower()

    # 4. No hallucination
    scores['no_extra_info'] = not has_information_not_in_context(
        answer, context
    )

    return scores
```

### Human Evaluation Criteria

When manually evaluating RAG systems:

```
Rate each answer on a scale of 1-5:

1. ACCURACY
   5 = Completely accurate based on context
   1 = Contains incorrect information

2. COMPLETENESS
   5 = Fully answers the question
   1 = Incomplete or missing key information

3. CONCISENESS
   5 = Appropriately concise
   1 = Too verbose or too brief

4. SOURCE USAGE
   5 = Properly cites all sources
   1 = No citations or incorrect citations

5. COHERENCE
   5 = Clear and well-structured
   1 = Confusing or poorly organized
```

**⏱️ Duration so far:** 35 minutes

---

## 7. Advanced RAG Techniques

### 1. Re-Ranking

**Problem:** Initial retrieval isn't always perfect.

**Solution:** Retrieve more chunks (e.g., 20), then re-rank to get best 3-5.

```
Query → Retrieve 20 chunks → Re-rank with cross-encoder → Top 3 → LLM
```

**Benefits:**
- Better final selection
- Catches edge cases

### 2. Query Rewriting

**Problem:** User queries aren't always optimal for retrieval.

**Solution:** Rewrite query before searching.

```
User: "How do I get my money back?"
Rewritten: "return policy refund process customer money back"

→ Better retrieval results
```

### 3. Hypothetical Document Embeddings (HyDE)

**Problem:** Query and document embeddings might not align.

**Solution:** Generate a hypothetical answer, embed it, search with that.

```
Query → LLM generates hypothetical answer → Embed → Search → Real context → Final answer
```

### 4. Multi-Query RAG

**Problem:** Single query might miss relevant information.

**Solution:** Generate multiple query variations, retrieve for each, combine.

```
Original: "What is machine learning?"

Variations:
- "Define machine learning"
- "Machine learning explanation"
- "What does ML mean?"

→ Retrieve for all → Combine unique results → Generate answer
```

### 5. RAG with Memory

**Problem:** No conversation context.

**Solution:** Include conversation history in retrieval.

```
User: "What's your return policy?"
Assistant: [Answer about returns]

User: "What about electronics?" ← Needs context!

Solution: Combine current + previous messages for retrieval
```

---

## 8. Review & Key Takeaways

### 🎯 What You Learned

✅ **RAG Concept** - Retrieval + Augmentation + Generation
✅ **Pipeline Architecture** - Offline prep + online query flow
✅ **Three Steps** - Retrieve, Augment, Generate
✅ **Prompt Engineering** - Structure, instructions, citations
✅ **RAG vs. Non-RAG** - When to use each approach
✅ **Evaluation** - Metrics for retrieval and generation quality
✅ **Advanced Techniques** - Re-ranking, query rewriting, HyDE

### 💡 Key Concepts

**1. RAG Solves LLM Limitations**
```
✓ Current information (not limited by training cutoff)
✓ Domain-specific knowledge (your documents)
✓ Reduced hallucinations (grounded in facts)
✓ Source attribution (verifiable)
```

**2. Quality Depends on All Three Steps**
```
Bad Retrieval → Bad Answer (even with good LLM)
Good Retrieval + Bad Prompt → Mediocre Answer
Good Retrieval + Good Prompt + Good LLM → Excellent Answer
```

**3. Prompt Engineering is Critical**
```
- Clear context boundaries
- Explicit instructions
- Source citation encouragement
- Handling insufficient context
```

**4. Evaluation is Multi-Faceted**
```
Retrieval: Did we get the right chunks?
Generation: Is the answer faithful to context?
End-to-End: Is the user satisfied?
```

### 🧠 Knowledge Check

<details>
<summary><strong>Question 1:</strong> What are the three steps of RAG?</summary>

**Answer:**
1. **Retrieve** - Search vector database for relevant chunks
2. **Augment** - Combine retrieved context with user question in a prompt
3. **Generate** - LLM creates answer based on provided context
</details>

<details>
<summary><strong>Question 2:</strong> Why use RAG instead of just asking the LLM directly?</summary>

**Answer:**
RAG provides:
- Access to current/updated information (not limited by training cutoff)
- Company/domain-specific knowledge not in LLM's training
- Reduced hallucinations (answers grounded in real documents)
- Source attribution (can cite where info came from)
- Cost-effective updates (change docs, not the model)
</details>

<details>
<summary><strong>Question 3:</strong> What should a good RAG prompt include?</summary>

**Answer:**
A good RAG prompt should include:
- Clear context boundaries (separating provided info from question)
- Explicit instructions (use only context, cite sources)
- Numbered or labeled context sections
- The user's question
- Guidelines for handling insufficient context
- Output format guidance
</details>

<details>
<summary><strong>Question 4:</strong> When should you NOT use RAG?</summary>

**Answer:**
Don't use RAG for:
- General knowledge questions LLM already knows well
- Creative tasks (stories, poems) that don't need factual grounding
- Pure reasoning/logic tasks that don't require documents
- Real-time calculations
- Simple queries where retrieval overhead isn't worth it
</details>

<details>
<summary><strong>Question 5:</strong> What's the difference between retrieval precision and recall?</summary>

**Answer:**
- **Precision**: Of the chunks retrieved, how many are actually relevant?
  (Relevant retrieved / Total retrieved)
- **Recall**: Of all relevant chunks in the database, how many did we retrieve?
  (Relevant retrieved / Total relevant in DB)

High precision = few irrelevant results
High recall = found most of the relevant information
</details>

### 🚀 Ready for Hands-On Practice?

You now understand:
- ✅ The complete RAG pipeline architecture
- ✅ How to combine retrieval with generation
- ✅ Prompt engineering for RAG
- ✅ RAG evaluation strategies
- ✅ Advanced RAG techniques

**Next step**: [Hands-On Lab →](lab.md)

In the lab, you'll:
1. Build a complete RAG pipeline
2. Integrate with OpenAI, Claude, or Bedrock
3. Compare RAG vs. non-RAG responses
4. Implement different prompt strategies
5. Create evaluation tools
6. Build a production-ready RAG system

---

### 📚 Additional Resources

**Want to dive deeper?**
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [OpenAI RAG Guide](https://platform.openai.com/docs/guides/retrieval-augmented-generation)
- [Anthropic RAG Best Practices](https://docs.anthropic.com/claude/docs/retrieval-augmented-generation)
- [RAG Evaluation Paper (RAGAS)](https://arxiv.org/abs/2309.15217)

---

**Learning Material Complete!** ✅
[← Back to README](../README.md) | [Start Hands-On Lab →](lab.md)
