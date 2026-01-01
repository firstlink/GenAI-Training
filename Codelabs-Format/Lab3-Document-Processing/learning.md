# Lab 3: Document Processing & Embeddings

## 📚 Learning Material

**Duration:** 35 minutes
**Difficulty:** Beginner to Intermediate
**Prerequisites:** Lab 1 & Lab 2 completed

---

## 🎯 Learning Objectives

By the end of this learning module, you will understand:
- ✅ Why we need to chunk documents for RAG systems
- ✅ Different text chunking strategies and their tradeoffs
- ✅ What embeddings are and how they work
- ✅ How to choose embedding models
- ✅ Vector databases and semantic similarity
- ✅ The complete document processing pipeline

---

## 📖 Table of Contents

1. [Introduction: The RAG Pipeline](#1-introduction-the-rag-pipeline)
2. [Why Chunk Documents?](#2-why-chunk-documents)
3. [Text Chunking Strategies](#3-text-chunking-strategies)
4. [Understanding Embeddings](#4-understanding-embeddings)
5. [Embedding Models](#5-embedding-models)
6. [Vector Databases](#6-vector-databases)
7. [Semantic Similarity](#7-semantic-similarity)
8. [The Complete Pipeline](#8-the-complete-pipeline)
9. [Review & Key Takeaways](#9-review--key-takeaways)

---

## 1. Introduction: The RAG Pipeline

### What is RAG?

**RAG** = **Retrieval-Augmented Generation**

It's a technique to give LLMs access to external knowledge by:
1. **Storing** documents in a searchable format
2. **Retrieving** relevant content based on user questions
3. **Augmenting** the LLM prompt with that content
4. **Generating** answers using both the LLM's knowledge and your documents

### The Problem RAG Solves

```
❌ WITHOUT RAG:
User: "What's our return policy for laptops?"
LLM: "I don't have information about your specific return policy."

✅ WITH RAG:
User: "What's our return policy for laptops?"
System: [Retrieves company policy document]
LLM: "According to your policy, laptops can be returned within 30
     days if unopened, or 14 days if opened, with original packaging..."
```

### The RAG Pipeline (High-Level)

```
┌─────────────────────────────────────────────────────────┐
│                    RAG PIPELINE                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  OFFLINE (Once):                                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐          │
│  │Documents │ → │  Chunk   │ → │ Embeddings│          │
│  │          │   │   Text   │   │  + Store  │          │
│  └──────────┘   └──────────┘   └──────────┘          │
│                                       ↓                 │
│                                 ┌──────────┐           │
│                                 │  Vector  │           │
│                                 │    DB    │           │
│                                 └──────────┘           │
│                                       ↑                 │
│  ONLINE (Every Query):                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐          │
│  │   User   │ → │ Search   │ → │ Generate │          │
│  │ Question │   │  Vector  │   │  Answer  │          │
│  └──────────┘   │    DB    │   │with LLM  │          │
│                 └──────────┘   └──────────┘          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Lab 3 Focus**: The OFFLINE part - processing documents and creating embeddings.

**⏱️ Duration so far:** 5 minutes

---

## 2. Why Chunk Documents?

### The Challenge

Imagine you have a 100-page company handbook. You can't send the entire document to the LLM because:

1. **Context Limits**: LLMs have maximum context windows (e.g., 4K, 16K, 128K tokens)
2. **Cost**: More tokens = more money
3. **Precision**: LLMs perform better with focused, relevant context
4. **Noise**: Irrelevant information can confuse the model

### The Solution: Chunking

**Chunking** = Breaking large documents into smaller, meaningful pieces.

```
┌─────────────────────────────────────────────┐
│        ORIGINAL DOCUMENT (10,000 words)     │
│  "Our company was founded in 1995...        │
│   ...return policy states that...           │
│   ...warranty covers manufacturing..."      │
└─────────────────────────────────────────────┘
                    ↓ CHUNK
┌─────────────────────────────────────────────┐
│ CHUNK 1: Company History (500 words)        │
│ CHUNK 2: Return Policy (300 words)          │
│ CHUNK 3: Warranty Information (400 words)   │
│ CHUNK 4: Contact Information (200 words)    │
│ ...                                         │
└─────────────────────────────────────────────┘
```

### Benefits of Chunking

✅ **Relevance**: Retrieve only the chunks that matter for the question
✅ **Efficiency**: Send less data to the LLM
✅ **Cost Savings**: Fewer tokens = lower API costs
✅ **Better Answers**: Focused context = more accurate responses

### Real-World Example

```
Question: "What's your laptop return policy?"

Without Chunking:
- Send entire 100-page handbook to LLM
- Cost: ~$0.50 per query
- Risk: LLM might miss the relevant section

With Chunking:
- Retrieve only "Return Policy" chunk (300 words)
- Cost: ~$0.05 per query
- Result: LLM focuses on exactly what matters
```

**⏱️ Duration so far:** 10 minutes

---

## 3. Text Chunking Strategies

### Strategy 1: Fixed-Size Character Chunking

**Simple approach**: Split text every N characters.

```python
# Example: 200 characters per chunk
text = "Artificial Intelligence is transforming industries..."
chunks = [text[0:200], text[200:400], text[400:600], ...]
```

**Pros:**
✅ Simple to implement
✅ Predictable chunk sizes
✅ Fast

**Cons:**
❌ Might split mid-sentence
❌ Might split mid-word
❌ Loses semantic boundaries

**Example of Bad Split:**
```
Chunk 1: "Our return policy allows customers to return produc"
Chunk 2: "ts within 30 days of purchase with original packaging."
```

### Strategy 2: Sentence-Based Chunking

**Better approach**: Split on sentence boundaries.

```
Chunk 1: "Our return policy is customer-friendly. Items can
          be returned within 30 days."

Chunk 2: "To initiate a return, contact customer service.
          Refunds are processed within 5-7 business days."
```

**Pros:**
✅ Maintains sentence integrity
✅ More semantic coherence
✅ Better for reading comprehension

**Cons:**
❌ Variable chunk sizes
❌ Sentences might be very long or very short

### Strategy 3: Paragraph-Based Chunking

**Even better**: Split on paragraph boundaries.

```
Chunk 1: [Entire paragraph about return policy]
Chunk 2: [Entire paragraph about shipping]
Chunk 3: [Entire paragraph about warranty]
```

**Pros:**
✅ Maintains topical coherence
✅ Natural semantic boundaries
✅ Good for documents with clear structure

**Cons:**
❌ Paragraphs can vary greatly in size
❌ Might exceed optimal chunk size

### Strategy 4: Recursive Character Text Splitting (BEST)

**LangChain's approach**: Try to split on paragraphs, then sentences, then words.

```python
Separators (in order of preference):
1. "\n\n"    # Paragraph breaks (best)
2. "\n"      # Line breaks (good)
3. ". "      # Sentence breaks (okay)
4. " "       # Word breaks (acceptable)
5. ""        # Character breaks (last resort)
```

**How it works:**
```
1. Try to split on "\n\n" (paragraphs)
   - If chunk is still too big → continue

2. Try to split on "\n" (lines)
   - If chunk is still too big → continue

3. Try to split on ". " (sentences)
   - If chunk is still too big → continue

4. Split on " " (words)
```

**Pros:**
✅ Maintains semantic meaning
✅ Respects document structure
✅ Configurable chunk size
✅ Industry standard

**This is what we'll use in the lab!**

### Chunk Overlap: The Secret Sauce

**Problem**: Information at chunk boundaries might get lost.

```
Chunk 1: "...the warranty covers parts and labor."
Chunk 2: "Warranty claims must be filed within 90 days..."
```

If someone asks "How long is the warranty coverage?", information is split across chunks.

**Solution**: Overlap chunks

```
Chunk 1: "...the warranty covers parts and labor.
          Warranty claims must be filed..."

Chunk 2: "Warranty claims must be filed within 90 days.
          To file a claim, contact support..."
```

**Typical Settings:**
- **Chunk Size**: 500-1000 characters (or 100-200 tokens)
- **Overlap**: 10-20% of chunk size (50-200 characters)

### Chunking Strategy Comparison

```
┌──────────────────┬─────────┬─────────────┬──────────────┐
│   Strategy       │  Speed  │  Quality    │  Use Case    │
├──────────────────┼─────────┼─────────────┼──────────────┤
│ Fixed-size       │  ★★★★★  │  ★☆☆☆☆      │  Quick tests │
│ Sentence-based   │  ★★★★☆  │  ★★★☆☆      │  Simple docs │
│ Paragraph-based  │  ★★★☆☆  │  ★★★★☆      │  Structured  │
│ Recursive (Best) │  ★★★☆☆  │  ★★★★★      │  Production  │
└──────────────────┴─────────┴─────────────┴──────────────┘
```

**⏱️ Duration so far:** 18 minutes

---

## 4. Understanding Embeddings

### What Are Embeddings?

**Embeddings** = Numerical representations of text that capture semantic meaning.

Instead of storing text as words, we convert it to numbers (vectors) that represent its *meaning*.

```
Text: "I love pizza"
Embedding: [0.23, -0.45, 0.78, 0.12, -0.34, ...]
           (typically 384 to 1536 dimensions)
```

### Why Numbers?

Because computers can:
- **Compare** numbers (which texts are similar?)
- **Search** numbers efficiently (find relevant documents)
- **Calculate** distances (how related are two concepts?)

### Visual Intuition (2D Simplification)

Imagine text mapped to a 2D space:

```
                 Animals
                    ↑
            cat •   |   • dog
                    |
        pizza •     |     • burger
                    |
    ←───────────────┼───────────────→
       Food         |        Sports
                    |
         • apple    |    • basketball
                    |
                    ↓
                 Health
```

**Key Insight**: Similar meanings are close together in vector space!

### Real Embeddings (384D Example)

In reality, embeddings have hundreds of dimensions:

```
"The cat sat on the mat"
→ [0.23, -0.45, 0.12, 0.89, -0.34, 0.56, ...]
   (384 numbers total)

"A feline rested on the rug"
→ [0.25, -0.43, 0.15, 0.87, -0.32, 0.54, ...]
   (Very similar numbers because similar meaning!)

"I love pizza"
→ [-0.78, 0.92, -0.45, 0.23, 0.67, -0.12, ...]
   (Very different numbers because different meaning)
```

### How Embeddings Capture Meaning

**Example 1: Synonyms**
```
"happy"     → [0.5, 0.8, -0.2, ...]
"joyful"    → [0.5, 0.8, -0.2, ...]  (very similar!)
"sad"       → [-0.5, -0.7, 0.3, ...]  (very different!)
```

**Example 2: Context**
```
"bank" (financial institution)
→ [0.2, 0.5, -0.3, 0.1, ...]

"bank" (river bank)
→ [-0.3, 0.1, 0.4, -0.2, ...]
```

The embedding model uses context to determine meaning!

### Embeddings vs. Keywords

**Old Way (Keyword Search):**
```
User: "How do I get my money back?"
Keyword Search: Looks for documents with "money" and "back"
Misses: "refund policy" (no keywords match!)
```

**New Way (Semantic Search with Embeddings):**
```
User: "How do I get my money back?"
Embedding: [0.5, -0.3, 0.7, ...]

Document: "Our refund policy allows returns..."
Embedding: [0.5, -0.3, 0.7, ...]  (very similar!)

Match found! ✓
```

### Properties of Good Embeddings

✅ **Semantic Similarity**: Similar meanings → similar vectors
✅ **Dimensionality**: Enough dimensions to capture nuance (384-1536)
✅ **Normalized**: Typically unit length for easy comparison
✅ **Dense**: Every dimension contributes to meaning
✅ **Context-Aware**: Same word in different contexts → different embeddings

**⏱️ Duration so far:** 25 minutes

---

## 5. Embedding Models

### What is an Embedding Model?

A **neural network** trained to convert text into meaningful vectors.

```
┌────────────────────────────────────────┐
│        EMBEDDING MODEL                 │
│                                        │
│  Input: "I love pizza"                │
│         ↓                              │
│  [Neural Network Processing]          │
│         ↓                              │
│  Output: [0.23, -0.45, 0.78, ...]     │
└────────────────────────────────────────┘
```

### Popular Embedding Models

#### 1. **sentence-transformers/all-MiniLM-L6-v2**
```
Dimensions: 384
Speed: ★★★★★ (Very Fast)
Quality: ★★★★☆ (Good)
Size: 80 MB
Use Case: Beginner-friendly, production-ready
```

**Best for:** Lab exercises, small to medium applications

#### 2. **sentence-transformers/all-mpnet-base-v2**
```
Dimensions: 768
Speed: ★★★☆☆ (Medium)
Quality: ★★★★★ (Excellent)
Size: 420 MB
Use Case: Higher quality requirements
```

**Best for:** Production systems where quality matters more than speed

#### 3. **text-embedding-3-small** (OpenAI)
```
Dimensions: 1536
Speed: ★★★★☆ (Fast via API)
Quality: ★★★★★ (Excellent)
Cost: $0.02 per 1M tokens
Use Case: Enterprise applications
```

**Best for:** Production with budget for API calls

#### 4. **text-embedding-3-large** (OpenAI)
```
Dimensions: 3072
Speed: ★★★☆☆ (Medium via API)
Quality: ★★★★★ (State-of-the-art)
Cost: $0.13 per 1M tokens
Use Case: Highest quality needs
```

**Best for:** Maximum accuracy requirements

### Model Comparison

```
┌──────────────────────────┬───────┬────────┬──────────┐
│ Model                    │ Dims  │ Speed  │ Quality  │
├──────────────────────────┼───────┼────────┼──────────┤
│ all-MiniLM-L6-v2        │  384  │ Fastest│ Good     │
│ all-mpnet-base-v2       │  768  │ Medium │ Better   │
│ text-embedding-3-small  │ 1536  │ Fast*  │ Best     │
│ text-embedding-3-large  │ 3072  │ Medium*│ Best++   │
└──────────────────────────┴───────┴────────┴──────────┘
* Requires API call
```

### Choosing the Right Model

**Start with:** `all-MiniLM-L6-v2`
- Free
- Fast
- Good enough for most use cases
- Easy to run locally
- **This is what we use in Lab 3!**

**Upgrade to:** `all-mpnet-base-v2`
- When you need better quality
- Still free and local
- 2x slower, 2x larger

**Upgrade to:** OpenAI embeddings
- When quality is critical
- When you're already using OpenAI for generation
- Budget for API costs

### Key Consideration: Consistency

⚠️ **IMPORTANT**: Use the **same embedding model** for:
1. Creating embeddings (offline)
2. Searching embeddings (online)

```
❌ BAD:
Documents embedded with: all-MiniLM-L6-v2 (384D)
Query embedded with: all-mpnet-base-v2 (768D)
→ Dimension mismatch! Won't work!

✅ GOOD:
Documents embedded with: all-MiniLM-L6-v2 (384D)
Query embedded with: all-MiniLM-L6-v2 (384D)
→ Perfect match! Works great!
```

**⏱️ Duration so far:** 30 minutes

---

## 6. Vector Databases

### What is a Vector Database?

A **specialized database** designed to:
1. Store high-dimensional vectors (embeddings)
2. Perform fast similarity searches
3. Handle millions of vectors efficiently

```
Traditional Database:
┌──────┬──────────┬────────┐
│ ID   │ Name     │ Email  │
├──────┼──────────┼────────┤
│ 1    │ John     │ j@...  │
│ 2    │ Sarah    │ s@...  │
└──────┴──────────┴────────┘

Vector Database:
┌──────┬──────────────────────┬─────────────────────────┐
│ ID   │ Text                 │ Embedding (384D)        │
├──────┼──────────────────────┼─────────────────────────┤
│ 1    │ "Return policy..."   │ [0.2, -0.5, 0.7, ...]   │
│ 2    │ "Shipping info..."   │ [-0.3, 0.8, -0.1, ...]  │
└──────┴──────────────────────┴─────────────────────────┘
```

### Why Not Use Regular Databases?

**Problem**: Finding similar vectors in regular databases is SLOW.

```
Finding similar text in PostgreSQL:
SELECT * FROM documents
WHERE CONTAINS(text, 'return policy');
→ Only finds exact keyword matches

Finding similar vectors in Vector DB:
query_vector = [0.2, -0.5, 0.7, ...]
results = db.search(query_vector, top_k=5)
→ Finds semantically similar documents (fast!)
```

### Popular Vector Databases

#### 1. **ChromaDB** (What we use in Lab 3)
```
Type: Embedded / Server
Setup: pip install chromadb
Speed: ★★★★★
Ease: ★★★★★ (Easiest!)
Scale: Up to millions of vectors
Best for: Development, small-medium production
```

#### 2. **Pinecone**
```
Type: Cloud-only
Setup: API key required
Speed: ★★★★★
Ease: ★★★★☆
Scale: Billions of vectors
Best for: Large-scale production
```

#### 3. **Weaviate**
```
Type: Self-hosted / Cloud
Setup: Docker required
Speed: ★★★★☆
Ease: ★★★☆☆
Scale: Hundreds of millions
Best for: Enterprise, self-hosted
```

#### 4. **FAISS** (Facebook AI)
```
Type: Library (not a database)
Setup: pip install faiss-cpu
Speed: ★★★★★
Ease: ★★☆☆☆
Scale: Billions of vectors
Best for: Research, advanced users
```

### ChromaDB: What We'll Use

**Why ChromaDB?**
✅ **Easy**: Works out of the box, no setup
✅ **Persistent**: Saves to disk automatically
✅ **Fast**: Optimized for similarity search
✅ **Metadata**: Can store metadata with each vector
✅ **Free**: Open source, no API costs

**Basic ChromaDB Operations:**

```python
# 1. Create/connect
client = chromadb.PersistentClient(path="./my_db")
collection = client.get_or_create_collection("my_docs")

# 2. Add documents
collection.add(
    documents=["text chunk 1", "text chunk 2"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    ids=["id1", "id2"],
    metadatas=[{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]
)

# 3. Search (we'll learn this in Lab 4)
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=5
)
```

**⏱️ Duration so far:** 33 minutes

---

## 7. Semantic Similarity

### How Do We Measure Similarity?

**Cosine Similarity** = The standard way to compare embeddings.

**Formula:**
```
similarity = (A · B) / (||A|| × ||B||)

Where:
· = dot product
||A|| = magnitude of vector A
```

**Don't worry about the math!** Libraries calculate this for you.

### Cosine Similarity Range

```
┌────────────────────────────────────────┐
│  Similarity Score Interpretation       │
├────────────┬───────────────────────────┤
│  1.0       │ Identical (same text)     │
│  0.9-1.0   │ Extremely similar         │
│  0.8-0.9   │ Very similar              │
│  0.7-0.8   │ Similar                   │
│  0.5-0.7   │ Somewhat related          │
│  0.0-0.5   │ Barely related            │
│  < 0.0     │ Opposite meanings         │
└────────────┴───────────────────────────┘
```

### Real Example

```python
from sklearn.metrics.pairwise import cosine_similarity

text1 = "I love programming in Python"
text2 = "Python is my favorite coding language"
text3 = "I enjoy eating pizza"

# After embedding:
embedding1 = [0.5, 0.8, -0.2, 0.3, ...]
embedding2 = [0.5, 0.8, -0.1, 0.3, ...]
embedding3 = [-0.3, 0.1, 0.7, -0.5, ...]

similarity_1_2 = cosine_similarity([embedding1], [embedding2])
# Result: 0.92 (very similar!)

similarity_1_3 = cosine_similarity([embedding1], [embedding3])
# Result: 0.15 (not related)
```

### Why Cosine Similarity?

**Advantages:**
✅ Normalized (-1 to 1 range)
✅ Works well for high dimensions
✅ Fast to compute
✅ Industry standard

**Alternatives:**
- **Euclidean Distance**: Measures straight-line distance
- **Dot Product**: Simple but not normalized
- **Manhattan Distance**: Sum of absolute differences

**For RAG systems, use Cosine Similarity** (it's the default in ChromaDB).

**⏱️ Duration so far:** 35 minutes

---

## 8. The Complete Pipeline

### End-to-End Flow

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: Load Document                                  │
│  ┌──────────────┐                                       │
│  │ PDF / TXT /  │ → Read file into memory               │
│  │ DOCX / HTML  │                                       │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 2: Chunk Text                                     │
│  ┌──────────────────────────────────────┐              │
│  │ "Lorem ipsum dolor sit amet...       │              │
│  │  consectetur adipiscing elit..."     │              │
│  └──────────────────────────────────────┘              │
│                    ↓                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Chunk 1  │  │ Chunk 2  │  │ Chunk 3  │            │
│  │ (500 chr)│  │ (500 chr)│  │ (500 chr)│            │
│  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 3: Generate Embeddings                            │
│  ┌──────────────────────────────────────┐              │
│  │    Embedding Model (384D)            │              │
│  │    all-MiniLM-L6-v2                  │              │
│  └──────────────────────────────────────┘              │
│                    ↓                                    │
│  Chunk 1 → [0.2, -0.5, 0.7, 0.1, ...]                  │
│  Chunk 2 → [-0.3, 0.8, -0.1, 0.4, ...]                 │
│  Chunk 3 → [0.6, -0.2, 0.9, -0.3, ...]                 │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 4: Store in Vector Database                      │
│  ┌─────────────────────────────────────┐               │
│  │        ChromaDB Collection          │               │
│  ├──────┬──────────────┬──────────────┤               │
│  │ ID   │ Text         │ Embedding    │               │
│  ├──────┼──────────────┼──────────────┤               │
│  │ ch_1 │ "Lorem..."   │ [0.2, -0.5...]│              │
│  │ ch_2 │ "Ipsum..."   │ [-0.3, 0.8...]│              │
│  │ ch_3 │ "Dolor..."   │ [0.6, -0.2...]│              │
│  └──────┴──────────────┴──────────────┘               │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 5: Query (Lab 4)                                  │
│  User Question → Embedding → Search → Retrieve Chunks   │
└─────────────────────────────────────────────────────────┘
```

### Code Example (Complete Pipeline)

```python
# Complete document processing pipeline

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# STEP 1: Load document
with open('document.txt', 'r') as f:
    document = f.read()

# STEP 2: Chunk text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(document)

# STEP 3: Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# STEP 4: Store in vector database
client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("docs")

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[{"index": i} for i in range(len(chunks))]
)

print(f"✓ Processed {len(chunks)} chunks and stored in ChromaDB")
```

**That's it!** In just ~15 lines of code, you have a working document processing pipeline.

---

## 9. Review & Key Takeaways

### 🎯 What You Learned

✅ **RAG Pipeline**: Retrieval-Augmented Generation gives LLMs external knowledge
✅ **Chunking**: Breaking documents into smaller pieces for better retrieval
✅ **Embeddings**: Numerical representations that capture semantic meaning
✅ **Vector Databases**: Specialized storage for fast similarity search
✅ **Similarity**: Cosine similarity measures how related chunks are

### 💡 Key Concepts

**1. Why Chunk?**
- LLMs have context limits
- Focused context = better answers
- Lower costs

**2. Best Chunking Strategy**
- Recursive Character Text Splitter
- Chunk size: 500-1000 characters
- Overlap: 10-20%

**3. Embeddings Capture Meaning**
- Similar meaning → similar vectors
- "refund" and "money back" will be close in vector space
- Enables semantic search (not just keyword matching)

**4. Vector Databases Are Essential**
- Regular databases can't efficiently search vectors
- ChromaDB makes it easy
- Stores documents, embeddings, and metadata

**5. The Complete Pipeline**
```
Load → Chunk → Embed → Store → Query (next lab!)
```

### 🧠 Knowledge Check

<details>
<summary><strong>Question 1:</strong> Why do we chunk documents instead of storing them whole?</summary>

**Answer:**
1. LLMs have context window limits
2. Focused context produces better answers
3. Reduces token costs
4. Improves retrieval precision
</details>

<details>
<summary><strong>Question 2:</strong> What is an embedding?</summary>

**Answer:**
A numerical (vector) representation of text that captures semantic meaning. Similar meanings produce similar vectors, enabling semantic search.
</details>

<details>
<summary><strong>Question 3:</strong> What's the difference between keyword search and semantic search?</summary>

**Answer:**
- **Keyword search**: Matches exact words ("refund" won't match "money back")
- **Semantic search**: Matches meaning (finds "refund policy" when you ask about "getting money back")
</details>

<details>
<summary><strong>Question 4:</strong> What is cosine similarity and what does a score of 0.9 mean?</summary>

**Answer:**
Cosine similarity measures how similar two embeddings are, ranging from -1 to 1. A score of 0.9 means the texts are extremely similar in meaning.
</details>

<details>
<summary><strong>Question 5:</strong> Why must you use the same embedding model for documents and queries?</summary>

**Answer:**
Different models produce embeddings with different dimensions and meanings. Using different models would be like trying to compare temperatures in Celsius and Fahrenheit without conversion - the numbers won't match up correctly.
</details>

### 🚀 Ready for Hands-On Practice?

You now understand:
- ✅ The theory behind document processing
- ✅ Why and how to chunk text
- ✅ What embeddings are and how they work
- ✅ How vector databases enable semantic search

**Next step**: [Hands-On Lab →](lab.md)

In the lab, you'll:
1. Load and chunk real documents
2. Generate embeddings using HuggingFace models
3. Store them in ChromaDB
4. Visualize and compare embeddings
5. Build the complete document processing pipeline

---

### 📚 Additional Resources

**Want to dive deeper?**
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Understanding Embeddings (Visual)](https://jalammar.github.io/illustrated-word2vec/)

---

**Learning Material Complete!** ✅
[← Back to README](../README.md) | [Start Hands-On Lab →](lab.md)
