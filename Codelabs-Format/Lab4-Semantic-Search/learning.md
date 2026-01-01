# Lab 4: Semantic Search & Retrieval

## 📚 Learning Material

**Duration:** 30 minutes
**Difficulty:** Intermediate
**Prerequisites:** Lab 3 completed (document processing & embeddings)

---

## 🎯 Learning Objectives

By the end of this learning module, you will understand:
- ✅ Semantic search vs. traditional keyword search
- ✅ How to query vector databases
- ✅ Distance metrics and similarity scoring
- ✅ Top-K retrieval strategies
- ✅ Metadata filtering for advanced queries
- ✅ Hybrid search combining semantic + keyword approaches
- ✅ When to use which search method

---

## 📖 Table of Contents

1. [Introduction: Search Paradigms](#1-introduction-search-paradigms)
2. [How Semantic Search Works](#2-how-semantic-search-works)
3. [Distance Metrics Explained](#3-distance-metrics-explained)
4. [Top-K Retrieval](#4-top-k-retrieval)
5. [Advanced Querying](#5-advanced-querying)
6. [Hybrid Search](#6-hybrid-search)
7. [Search Strategy Selection](#7-search-strategy-selection)
8. [Review & Key Takeaways](#8-review--key-takeaways)

---

## 1. Introduction: Search Paradigms

### The Evolution of Search

```
┌────────────────────────────────────────────────────────┐
│  SEARCH EVOLUTION                                      │
├────────────────────────────────────────────────────────┤
│                                                        │
│  1990s: EXACT MATCH                                   │
│  "Find 'artificial intelligence'"                     │
│  → Only matches those exact words                     │
│                                                        │
│  2000s: KEYWORD SEARCH (Google, BM25)                 │
│  "artificial intelligence"                            │
│  → Matches: AI, A.I., artificial, intelligence        │
│  → Uses TF-IDF, ranking algorithms                    │
│                                                        │
│  2020s: SEMANTIC SEARCH (Vector-based)                │
│  "teaching computers to think"                        │
│  → Matches: machine learning, AI, neural networks     │
│  → Understands MEANING, not just words                │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Traditional Keyword Search

**How it works:**
1. User enters query: "machine learning"
2. System looks for documents containing those words
3. Returns documents with "machine" AND "learning"
4. Ranks by frequency (TF-IDF) or other metrics

**Limitations:**
```
Query: "How do I get my money back?"

Keyword Search Results:
❌ Misses: "Refund Policy" (no keywords match!)
❌ Misses: "Return Process" (different words)
❌ Finds: "We don't give money back" (has keywords but wrong meaning!)
```

### Semantic Search

**How it works:**
1. User enters query: "How do I get my money back?"
2. System converts query → embedding vector
3. Compares query vector with all document vectors
4. Returns most similar documents by meaning

**Advantages:**
```
Query: "How do I get my money back?"

Semantic Search Results:
✅ Finds: "Refund Policy" (similar meaning!)
✅ Finds: "Return Process" (conceptually related)
✅ Finds: "Getting Your Money Refunded" (exact match semantically)
```

### Side-by-Side Comparison

```
┌──────────────────────┬──────────────────────┬──────────────────┐
│   Feature            │  Keyword Search      │ Semantic Search  │
├──────────────────────┼──────────────────────┼──────────────────┤
│ Exact matches        │  ★★★★★               │  ★★★☆☆           │
│ Synonyms             │  ★☆☆☆☆               │  ★★★★★           │
│ Related concepts     │  ★☆☆☆☆               │  ★★★★★           │
│ Acronyms             │  ★★★★★               │  ★★☆☆☆           │
│ Speed                │  ★★★★★               │  ★★★★☆           │
│ Understands context  │  ★☆☆☆☆               │  ★★★★★           │
│ Handles typos        │  ★☆☆☆☆               │  ★★★☆☆           │
└──────────────────────┴──────────────────────┴──────────────────┘
```

**⏱️ Duration so far:** 5 minutes

---

## 2. How Semantic Search Works

### The Search Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  SEMANTIC SEARCH PIPELINE                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  USER INPUT:                                            │
│  "What is deep learning?"                               │
│         ↓                                               │
│  ┌──────────────────────────┐                          │
│  │  EMBEDDING MODEL         │                          │
│  │  (same as documents!)    │                          │
│  └──────────────────────────┘                          │
│         ↓                                               │
│  QUERY EMBEDDING:                                       │
│  [0.23, -0.45, 0.78, 0.12, ...]  (384 dimensions)      │
│         ↓                                               │
│  ┌──────────────────────────┐                          │
│  │  VECTOR DATABASE         │                          │
│  │  (ChromaDB)              │                          │
│  └──────────────────────────┘                          │
│         ↓                                               │
│  SIMILARITY CALCULATION:                                │
│  Compare query embedding with all document embeddings   │
│         ↓                                               │
│  RANKED RESULTS:                                        │
│  1. "Deep learning uses neural networks..." (0.12)      │
│  2. "Neural networks with multiple layers..." (0.18)    │
│  3. "Machine learning subset..." (0.25)                 │
│     (Lower distance = more similar)                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Key Requirements

**1. Same Embedding Model**

⚠️ **CRITICAL**: Must use the SAME model for documents and queries!

```
Documents embedded with:
  all-MiniLM-L6-v2 (384 dimensions)

Query MUST be embedded with:
  all-MiniLM-L6-v2 (384 dimensions)

❌ WRONG: Query with all-mpnet-base-v2 (768 dimensions)
→ Dimension mismatch! Won't work!
```

**2. Vector Database**

The database stores:
- ✅ Document chunks (text)
- ✅ Embeddings (vectors)
- ✅ Metadata (source, chunk index, etc.)

And provides:
- ✅ Fast similarity search
- ✅ Ranking by distance/similarity
- ✅ Filtering capabilities

**3. Similarity Calculation**

How similar is the query to each document?

```python
query_embedding = [0.5, 0.8, -0.2, ...]
doc1_embedding = [0.5, 0.8, -0.2, ...]  # Very similar!
doc2_embedding = [-0.3, 0.1, 0.7, ...]  # Different

similarity(query, doc1) = 0.99  # High!
similarity(query, doc2) = 0.35  # Low
```

### Real-World Example

```
Question: "How do neural networks learn?"

Step 1: Embed the question
→ [0.23, -0.11, 0.56, 0.78, -0.34, ...]

Step 2: Compare with documents in database
Document 1: "Neural networks adjust weights..."
  Embedding: [0.25, -0.10, 0.58, 0.76, -0.32, ...]
  Distance: 0.08 ← Very close!

Document 2: "Computer vision applications..."
  Embedding: [-0.12, 0.45, -0.23, 0.11, 0.67, ...]
  Distance: 1.45 ← Far away

Document 3: "Training with backpropagation..."
  Embedding: [0.22, -0.13, 0.54, 0.79, -0.35, ...]
  Distance: 0.12 ← Close!

Step 3: Return top results (lowest distances)
1. Document 1 (distance: 0.08)
2. Document 3 (distance: 0.12)
```

**⏱️ Duration so far:** 10 minutes

---

## 3. Distance Metrics Explained

### What is Distance?

**Distance** = How "far apart" two vectors are in multi-dimensional space.

In 2D (easy to visualize):
```
     Y
     ↑
  B  •
     │
     │   • A
     │
  ───┼────────→ X

Distance(A, B) = √[(x₂-x₁)² + (y₂-y₁)²]
```

In 384D (what we actually use):
- Same concept, just 384 dimensions instead of 2!
- Can't visualize, but math is the same

### Common Distance Metrics

#### 1. Euclidean Distance (L2)

**What it is:** Straight-line distance between two points.

```
Formula: √[Σ(a₁ - b₁)² + (a₂ - b₂)² + ... + (a₃₈₄ - b₃₈₄)²]

Properties:
- Range: 0 to ∞
- 0 = identical vectors
- Larger = more different
```

**Example:**
```python
vector1 = [0.5, 0.8, -0.2]
vector2 = [0.5, 0.8, -0.2]  # Identical
euclidean_distance = 0.0

vector3 = [1.0, 0.0, 0.5]   # Different
euclidean_distance = 1.24
```

**ChromaDB uses L2 (Euclidean) distance by default!**

#### 2. Cosine Similarity

**What it is:** Measures the angle between two vectors.

```
Formula: (A · B) / (||A|| × ||B||)

Properties:
- Range: -1 to 1
- 1 = same direction (identical meaning)
- 0 = perpendicular (unrelated)
- -1 = opposite direction (opposite meaning)
```

**Visual:**
```
Same direction (similar):
  A →→→→→
  B →→→→

Cosine similarity = 1.0

Different directions (dissimilar):
  A →→→→→
        ↓
        ↓ B
        ↓

Cosine similarity = 0.0
```

**Often converted to distance:**
```python
cosine_distance = 1 - cosine_similarity

# If similarity = 0.9 (very similar)
# Then distance = 1 - 0.9 = 0.1 (very close)
```

#### 3. Dot Product

**What it is:** Simple multiplication and sum.

```
Formula: A₁×B₁ + A₂×B₂ + ... + A₃₈₄×B₃₈₄

Properties:
- Range: -∞ to ∞
- Higher = more similar (if vectors normalized)
- Fast to compute
```

### Which Metric to Use?

```
┌─────────────────┬──────────────────────────────────────┐
│ Metric          │ When to Use                          │
├─────────────────┼──────────────────────────────────────┤
│ Euclidean (L2)  │ Default for ChromaDB                 │
│                 │ Good general-purpose metric          │
│                 │ Considers both direction and length  │
├─────────────────┼──────────────────────────────────────┤
│ Cosine          │ When vector magnitude doesn't matter │
│                 │ Good for text (sentence length ≠     │
│                 │   importance)                        │
│                 │ Often better for semantic search    │
├─────────────────┼──────────────────────────────────────┤
│ Dot Product     │ When vectors are normalized          │
│                 │ Fastest to compute                   │
│                 │ Equivalent to cosine if normalized   │
└─────────────────┴──────────────────────────────────────┘
```

### Interpreting Distance Scores

**For L2 (Euclidean) Distance:**
```
┌──────────────┬────────────────────────────┐
│  Distance    │  Interpretation            │
├──────────────┼────────────────────────────┤
│  0.0 - 0.3   │  Extremely similar         │
│  0.3 - 0.6   │  Very similar              │
│  0.6 - 1.0   │  Similar                   │
│  1.0 - 1.5   │  Somewhat related          │
│  > 1.5       │  Not very related          │
└──────────────┴────────────────────────────┘
```

**For Cosine Similarity:**
```
┌──────────────┬────────────────────────────┐
│  Similarity  │  Interpretation            │
├──────────────┼────────────────────────────┤
│  0.9 - 1.0   │  Extremely similar         │
│  0.8 - 0.9   │  Very similar              │
│  0.7 - 0.8   │  Similar                   │
│  0.5 - 0.7   │  Somewhat related          │
│  < 0.5       │  Not very related          │
└──────────────┴────────────────────────────┘
```

**⏱️ Duration so far:** 15 minutes

---

## 4. Top-K Retrieval

### What is Top-K?

**Top-K retrieval** = Returning the K most similar results.

```
Query: "What is machine learning?"

K=1 (Top-1): Return only the BEST match
K=3 (Top-3): Return the 3 BEST matches
K=10 (Top-10): Return the 10 BEST matches
```

### How to Choose K

**Too Small (K=1):**
```
❌ Might miss relevant information
❌ No diversity in results
❌ If best match is poor, you're stuck with it
```

**Too Large (K=20):**
```
❌ Includes less relevant results
❌ More noise for the LLM to process
❌ Higher costs (more tokens)
❌ Slower processing
```

**Just Right (K=3-5):**
```
✅ Multiple perspectives
✅ Captures main relevant content
✅ Reasonable token count
✅ Good balance of precision and recall
```

### Recommended K Values

```
┌──────────────────────┬────────────┬─────────────────┐
│  Use Case            │  K Value   │  Why            │
├──────────────────────┼────────────┼─────────────────┤
│ Quick answers        │  1-2       │ Fast, focused   │
│ Standard RAG         │  3-5       │ Balanced        │
│ Comprehensive        │  5-10      │ More context    │
│ Research/Analysis    │  10-20     │ Thorough        │
│ Reranking pipeline   │  20-50     │ Filter later    │
└──────────────────────┴────────────┴─────────────────┘
```

### K vs. Context Window

**Important consideration:**

```
LLM Context Window: 4096 tokens
Chunk size: ~200 tokens
System prompt: ~100 tokens
User query: ~50 tokens

Available for context: 4096 - 100 - 50 = 3946 tokens

Maximum K: 3946 / 200 ≈ 19 chunks

Practical K (with room for response): 3-10 chunks
```

### Example: Different K Values

```
Query: "How do neural networks learn?"

K=1:
[1] "Neural networks learn by adjusting weights through
     backpropagation based on training data..."
→ Single focused answer

K=3:
[1] "Neural networks learn by adjusting weights..."
[2] "The learning process involves forward and backward
     passes through the network..."
[3] "Training data is used to optimize the network's
     parameters using gradient descent..."
→ Multiple perspectives, richer context

K=10:
[1-3] Directly relevant
[4-6] Related concepts
[7-10] Loosely related or redundant
→ More comprehensive but potentially noisy
```

**⏱️ Duration so far:** 20 minutes

---

## 5. Advanced Querying

### Metadata Filtering

**Metadata** = Additional information stored with each chunk.

```
Chunk: "Our return policy allows 30-day returns..."
Metadata: {
  "source": "policy_2024.pdf",
  "chunk_index": 3,
  "document_type": "policy",
  "last_updated": "2024-01-15"
}
```

### Why Use Metadata Filtering?

**Scenario 1: Multi-Source Database**
```
Question: "What's the return policy?"

Without filtering:
→ Returns results from ALL sources (policies, emails, blogs, etc.)

With filtering:
→ Only search in source="policy_2024.pdf"
→ More accurate, relevant results
```

**Scenario 2: Time-Sensitive Information**
```
Question: "What are the current shipping rates?"

Filter: last_updated >= "2024-01-01"
→ Only get recent information, not outdated rates
```

### Filter Examples

```python
# Filter by source
filter = {"source": {"$eq": "employee_handbook.pdf"}}

# Filter by date
filter = {"last_updated": {"$gte": "2024-01-01"}}

# Filter by document type
filter = {"document_type": {"$in": ["policy", "guideline"]}}

# Combine filters
filter = {
  "$and": [
    {"source": {"$eq": "handbook.pdf"}},
    {"section": {"$eq": "benefits"}}
  ]
}
```

### ChromaDB Filter Operators

```
┌──────────────┬────────────────────────────────────┐
│  Operator    │  Meaning                           │
├──────────────┼────────────────────────────────────┤
│  $eq         │  Equal to                          │
│  $ne         │  Not equal to                      │
│  $gt         │  Greater than                      │
│  $gte        │  Greater than or equal             │
│  $lt         │  Less than                         │
│  $lte        │  Less than or equal                │
│  $in         │  In list                           │
│  $nin        │  Not in list                       │
│  $and        │  All conditions must match         │
│  $or         │  Any condition must match          │
└──────────────┴────────────────────────────────────┘
```

### Combining Filters with Search

```
Query: "vacation policy"
Filter: {
  "$and": [
    {"source": "employee_handbook.pdf"},
    {"last_updated": {"$gte": "2024-01-01"}},
    {"section": "benefits"}
  ]
}

Result:
1. Find all chunks matching the filter
2. Calculate similarity with query
3. Return top-k most similar chunks
→ Only from employee handbook, recent, in benefits section
```

**⏱️ Duration so far:** 25 minutes

---

## 6. Hybrid Search

### The Problem with Single Methods

**Semantic Search Alone:**
```
Query: "NLP applications"

Issue: "NLP" is an acronym
- Semantic search might not find exact "NLP" mentions
- Might return "natural language processing" (good!)
- But could miss specific "NLP" technical discussions
```

**Keyword Search Alone:**
```
Query: "teaching computers to understand language"

Issue: No exact keyword matches
- Document says "natural language processing"
- Keyword search misses it (different words!)
- But meaning is identical
```

### Solution: Hybrid Search

**Hybrid Search** = Semantic Search + Keyword Search combined!

```
┌─────────────────────────────────────────────────────────┐
│  HYBRID SEARCH ARCHITECTURE                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Query: "NLP applications"                              │
│         ↓                     ↓                         │
│  ┌──────────────┐      ┌──────────────┐               │
│  │  SEMANTIC    │      │   KEYWORD    │               │
│  │   SEARCH     │      │   SEARCH     │               │
│  │  (Vectors)   │      │   (BM25)     │               │
│  └──────────────┘      └──────────────┘               │
│         ↓                     ↓                         │
│    Results A             Results B                      │
│    (semantic)            (keyword)                      │
│         ↓                     ↓                         │
│         └─────────┬───────────┘                        │
│                   ↓                                     │
│          ┌─────────────────┐                           │
│          │  MERGE & RANK   │                           │
│          │   (Weighted)    │                           │
│          └─────────────────┘                           │
│                   ↓                                     │
│          Final Ranked Results                          │
│          (best of both!)                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### BM25: The Keyword Search Algorithm

**BM25** = Best Match 25, a ranking function used by search engines.

**How it works:**
1. **Term Frequency (TF)**: How often does the term appear in the document?
2. **Inverse Document Frequency (IDF)**: How rare is the term overall?
3. **Document Length**: Normalize by document length

**Formula (simplified):**
```
score(query, doc) = Σ IDF(term) × TF(term, doc) × boost_factors

For each term in query:
  - Rare terms (high IDF) = more important
  - Frequent in doc (high TF) = more relevant
  - Shorter docs get slight boost
```

**Example:**
```
Query: "machine learning algorithms"

Document 1: "Machine learning algorithms are used for..."
  - "machine" appears 3 times (common word, low IDF)
  - "learning" appears 2 times (common word, low IDF)
  - "algorithms" appears 5 times (important word, high IDF)
  BM25 score: 8.5

Document 2: "Algorithms for sorting data include..."
  - "algorithms" appears 2 times
  - "machine" appears 0 times
  - "learning" appears 0 times
  BM25 score: 3.2

Document 1 wins!
```

### Combining Scores

**Method 1: Weighted Average**
```python
final_score = (semantic_weight × semantic_score) +
              (keyword_weight × keyword_score)

# Example:
semantic_score = 0.85  # High similarity
keyword_score = 0.30   # Low keyword match

# 70% semantic, 30% keyword
final_score = (0.7 × 0.85) + (0.3 × 0.30)
            = 0.595 + 0.09
            = 0.685
```

**Method 2: Reciprocal Rank Fusion (RRF)**
```python
# Instead of combining scores, combine rankings

Semantic ranking:   [doc3, doc1, doc5, doc2, doc4]
Keyword ranking:    [doc1, doc3, doc2, doc5, doc4]

RRF score for each doc = Σ(1 / (k + rank))
where k = constant (usually 60)

doc1: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
doc3: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
doc5: 1/(60+3) + 1/(60+4) = 0.0159 + 0.0156 = 0.0315

Final ranking: [doc1, doc3, doc5, doc2, doc4]
```

**RRF Advantages:**
✅ No need to normalize scores
✅ Works with different scoring systems
✅ Resistant to outliers
✅ Industry-proven method

### When to Use Hybrid Search

```
┌──────────────────────┬─────────────────┬──────────────────┐
│  Query Type          │  Best Method    │  Why             │
├──────────────────────┼─────────────────┼──────────────────┤
│ "What is ML?"        │  Semantic       │  Conceptual      │
│ "NLP"                │  Keyword        │  Acronym         │
│ "neural networks"    │  Hybrid         │  Exact + similar │
│ "teaching computers" │  Semantic       │  No exact match  │
│ "API key"            │  Keyword        │  Technical term  │
│ "refund process"     │  Hybrid         │  Both useful     │
└──────────────────────┴─────────────────┴──────────────────┘
```

**⏱️ Duration so far:** 30 minutes

---

## 7. Search Strategy Selection

### Decision Tree

```
                    START
                      │
                      ↓
            Is query an acronym?
                 /        \
               YES         NO
                │           │
                ↓           ↓
          Use Keyword    Is query conceptual?
              or            /        \
           Hybrid         YES         NO
                           │           │
                           ↓           ↓
                      Use Semantic  Exact match needed?
                                      /        \
                                    YES         NO
                                     │           │
                                     ↓           ↓
                                Use Hybrid  Use Semantic
```

### Strategy Guide

**1. Semantic Search (Vector-only)**

**Use when:**
- ✅ Queries are natural language questions
- ✅ Synonyms and paraphrasing are common
- ✅ Conceptual understanding is key
- ✅ No specific technical terms required

**Examples:**
- "How do I return a product?"
- "What is the vacation policy?"
- "Explain machine learning"

---

**2. Keyword Search (BM25-only)**

**Use when:**
- ✅ Exact term matching is critical
- ✅ Queries contain acronyms or codes
- ✅ Technical jargon that shouldn't be paraphrased
- ✅ Speed is paramount

**Examples:**
- "API-KEY-123"
- "NLP transformer"
- "HTTP 404 error"

---

**3. Hybrid Search (Combined)**

**Use when:**
- ✅ General-purpose search
- ✅ Mix of exact and conceptual matching
- ✅ Production systems (best overall performance)
- ✅ User queries are unpredictable

**Examples:**
- "NLP sentiment analysis" (acronym + concept)
- "return policy for laptops" (exact product + concept)
- "API authentication methods" (technical + general)

**Recommended weight:** 60% semantic, 40% keyword

### Real-World Recommendations

**Customer Support Chatbot:**
```
Strategy: Hybrid (70% semantic, 30% keyword)
Why: Users ask natural questions but mention specific product names/codes
```

**Technical Documentation Search:**
```
Strategy: Hybrid (40% semantic, 60% keyword)
Why: Developers search for exact function names but also concepts
```

**Research Paper Search:**
```
Strategy: Semantic (100%)
Why: Conceptual understanding is key, synonyms are common
```

**E-commerce Product Search:**
```
Strategy: Keyword (80%), Semantic (20%)
Why: Users search by brand names, model numbers, but also descriptions
```

---

## 8. Review & Key Takeaways

### 🎯 What You Learned

✅ **Search Paradigms**: Traditional keyword vs. modern semantic search
✅ **Distance Metrics**: L2 (Euclidean), Cosine similarity, Dot product
✅ **Top-K Retrieval**: Choosing the right number of results
✅ **Advanced Filtering**: Using metadata to refine searches
✅ **Hybrid Search**: Combining semantic + keyword for best results
✅ **Strategy Selection**: When to use which approach

### 💡 Key Concepts

**1. Semantic Search Captures Meaning**
```
"refund" ≈ "money back" ≈ "return my purchase"
All have similar embeddings despite different words!
```

**2. Same Embedding Model Required**
```
Documents: all-MiniLM-L6-v2 (384D)
Queries:   all-MiniLM-L6-v2 (384D) ✓

Documents: all-MiniLM-L6-v2 (384D)
Queries:   all-mpnet-base-v2 (768D) ✗
```

**3. Lower Distance = More Similar**
```
Distance 0.08: Extremely relevant
Distance 0.85: Somewhat relevant
Distance 2.50: Not relevant
```

**4. Top-K Balance**
```
Too few (K=1): Might miss important info
Sweet spot (K=3-5): Best for RAG
Too many (K=20): Noise and high cost
```

**5. Hybrid Search is Often Best**
```
Semantic: Understands meaning
Keyword: Finds exact matches
Hybrid: Best of both worlds!
```

### 🧠 Knowledge Check

<details>
<summary><strong>Question 1:</strong> What's the main difference between semantic and keyword search?</summary>

**Answer:**
- **Keyword search** looks for exact word matches
- **Semantic search** understands meaning and finds conceptually similar content
- Semantic search uses embeddings to find "refund policy" when you search for "how to get money back"
</details>

<details>
<summary><strong>Question 2:</strong> Why must you use the same embedding model for documents and queries?</summary>

**Answer:**
Different models produce embeddings with different dimensions and different semantic spaces. Using different models is like trying to match GPS coordinates from Earth with coordinates from Mars - they won't align correctly!
</details>

<details>
<summary><strong>Question 3:</strong> What does a distance score of 0.5 mean in L2 distance?</summary>

**Answer:**
With L2 (Euclidean) distance, 0.5 typically indicates a somewhat similar/related document. Lower is better (0.0 = identical). Generally:
- 0.0-0.3: Very similar
- 0.3-0.6: Similar  ← 0.5 is here
- 0.6-1.0: Somewhat related
- >1.0: Less related
</details>

<details>
<summary><strong>Question 4:</strong> When should you use hybrid search instead of pure semantic search?</summary>

**Answer:**
Use hybrid search when:
- Queries contain both conceptual questions and specific terms/acronyms
- You want the benefits of both semantic understanding AND exact matching
- In production systems where query types are unpredictable
- When searching technical documentation with specific function/API names

Example: "NLP sentiment analysis" benefits from keyword matching "NLP" and semantic understanding of "sentiment analysis"
</details>

<details>
<summary><strong>Question 5:</strong> What's the recommended K value for standard RAG applications?</summary>

**Answer:**
K=3-5 is recommended for standard RAG because:
- Provides multiple perspectives
- Doesn't overwhelm the LLM with too much context
- Balances precision and recall
- Fits well within typical context windows
- Cost-effective (fewer tokens)
</details>

### 🚀 Ready for Hands-On Practice?

You now understand:
- ✅ How semantic search finds meaning, not just keywords
- ✅ Distance metrics and what scores mean
- ✅ Top-K retrieval strategies
- ✅ Metadata filtering for advanced queries
- ✅ Hybrid search combining both approaches
- ✅ When to use which search method

**Next step**: [Hands-On Lab →](lab.md)

In the lab, you'll:
1. Query your vector database from Lab 3
2. Implement semantic search
3. Compare different K values
4. Add metadata filtering
5. Build hybrid search with BM25
6. Create a complete search system

---

### 📚 Additional Resources

**Want to dive deeper?**
- [ChromaDB Querying Documentation](https://docs.trychroma.com/usage-guide#querying-a-collection)
- [BM25 Algorithm Explained](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Vector Search Best Practices](https://www.pinecone.io/learn/vector-search/)

---

**Learning Material Complete!** ✅
[← Back to README](../README.md) | [Start Hands-On Lab →](lab.md)
