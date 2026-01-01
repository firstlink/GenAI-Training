# Lab 4 Solutions: Semantic Search & Retrieval

## 📚 Overview

Complete solutions for Lab 4 covering semantic search, Top-K retrieval optimization, metadata filtering, BM25 keyword search, and hybrid search strategies. Build production-ready search systems combining multiple retrieval methods.

---

## 📁 Files Included

### Core Exercises
- **`exercise1_basic_search.py`** - Semantic search fundamentals with ChromaDB
- **`exercise2_topk_retrieval.py`** - Top-K optimization strategies
- **`exercise3_metadata_filtering.py`** - Filter search by metadata attributes
- **`exercise4_hybrid_search.py`** - BM25 keyword search and hybrid retrieval

### Capstone Project
- **`capstone_production_search.py`** - Complete production search system

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install sentence-transformers chromadb numpy
```

### Run Exercises
```bash
python exercise1_basic_search.py
python exercise2_topk_retrieval.py
python exercise3_metadata_filtering.py
python exercise4_hybrid_search.py
```

### Run Capstone
```bash
python capstone_production_search.py
```

---

## 📖 Exercise Guide

### Exercise 1: Basic Semantic Search (15 min)
**What you'll learn:**
- Connect to ChromaDB vector database
- Perform semantic search with embeddings
- Understand distance metrics
- Retrieve and rank results

**Key features:**
```python
# Setup sample database
client, collection, embedding_model = setup_sample_database()

# Semantic search
search_engine = SemanticSearchEngine(collection, embedding_model)
results = search_engine.search("neural networks for learning", n_results=5)
```

**Distance Interpretation:**
```
< 0.5   → Very similar (excellent match)
0.5-1.0 → Similar (good match)
1.0-1.5 → Somewhat related
> 1.5   → Not very related
```

**How it works:**
1. Query text → Embedding (384D vector)
2. Compare query embedding to all stored embeddings
3. Calculate distance (L2/Euclidean by default)
4. Return top-K results sorted by distance

---

### Exercise 2: Top-K Retrieval Strategies (20 min)
**What you'll learn:**
- Understand the Top-K parameter
- Compare different K values
- Optimize retrieval performance
- Balance precision vs recall

**Usage:**
```python
optimizer = TopKOptimizer(collection, embedding_model)

# Compare different K values
optimizer.compare_topk(query, k_values=[1, 3, 5, 10])

# Analyze distance distribution
optimizer.analyze_distance_distribution(query, k=10)
```

**K Selection Guidelines:**

| K Value | Use Case | Example Query |
|---------|----------|---------------|
| 1-3 | Precise answers, direct questions | "What is deep learning?" |
| 3-5 | General search, balanced results | "Machine learning techniques" |
| 5-10 | Exploration, broad topics | "AI applications" |
| > 10 | Comprehensive research | "Everything about neural networks" |

**Quality Breakdown:**
- **Excellent** (< 0.5): Highly relevant results
- **Good** (0.5-1.0): Relevant results
- **Fair** (1.0-1.5): Somewhat relevant
- **Poor** (>= 1.5): Low relevance

---

### Exercise 3: Metadata Filtering (15 min)
**What you'll learn:**
- Filter search results by metadata
- Combine semantic search with metadata
- Build faceted search
- Route queries intelligently

**Usage:**
```python
search = MetadataFilteredSearch(collection, embedding_model)

# Filter by topic
results = search.search_by_topic("learning from data", "ML", n_results=3)

# Filter by difficulty
results = search.search_by_difficulty("machine learning", "beginner", n_results=3)

# Combined filters
combined_filter = {
    "$and": [
        {"topic": "NLP"},
        {"difficulty": "advanced"}
    ]
}
results = search.search_with_filter("neural networks", combined_filter, n_results=3)
```

**Benefits:**
- Narrows search scope
- Improves precision for specific domains
- Supports complex filter combinations
- Enables topic routing and faceted search

---

### Exercise 4: Hybrid Search with BM25 (25 min)
**What you'll learn:**
- Understand BM25 keyword search algorithm
- Implement keyword search
- Combine semantic + keyword search
- Use Reciprocal Rank Fusion (RRF)

**BM25 Implementation:**
```python
# Initialize BM25
bm25 = BM25(documents, k1=1.5, b=0.75)

# Search with BM25
results = bm25.search("neural networks layers", top_k=5)
```

**Hybrid Search:**
```python
hybrid = HybridSearch(collection, embedding_model)

# Combines semantic + keyword search with RRF
results = hybrid.hybrid_search("neural networks for learning", n_results=5)
```

**BM25 Formula:**
```
score(D,Q) = Σ IDF(qi) · (f(qi,D) · (k1+1)) / (f(qi,D) + k1·(1-b+b·|D|/avgdl))

where:
  - IDF(qi) = inverse document frequency of query term qi
  - f(qi,D) = frequency of qi in document D
  - |D| = length of document D
  - avgdl = average document length
  - k1, b = tuning parameters (typically k1=1.5, b=0.75)
```

**Reciprocal Rank Fusion (RRF):**
```
RRF(d) = Σ 1/(k + rank(d))

where:
  - k = constant (typically 60)
  - rank(d) = rank of document d in result list

Sum over all retrieval methods (semantic + keyword)
```

**When to Use Each Method:**

**🧠 Semantic Search:**
- ✅ Conceptual queries ("how does learning work?")
- ✅ Synonym matching ("car" finds "automobile")
- ✅ Paraphrase detection
- ✅ Cross-lingual search

**🔤 Keyword Search (BM25):**
- ✅ Exact term matching ("Python 3.11")
- ✅ Technical terms ("transformers architecture")
- ✅ Proper nouns ("TensorFlow")
- ✅ Code/formula search

**🔀 Hybrid Search:**
- ✅ Best of both worlds
- ✅ Production search systems
- ✅ When precision matters
- ✅ Diverse query types

---

### 🏆 Capstone: Production Search System

**What you'll build:**
Complete production-ready search system combining all Lab 4 concepts.

**Features:**
✅ Multi-strategy search (semantic, keyword, hybrid)
✅ Automatic strategy selection
✅ Metadata filtering
✅ Reciprocal Rank Fusion
✅ Performance monitoring
✅ Query analytics
✅ Configurable parameters

**Usage:**
```python
# Initialize system
search_system = ProductionSearchSystem(
    embedding_model='all-MiniLM-L6-v2',
    persist_directory="./production_search_db",
    default_strategy='hybrid'
)

# Create knowledge base
search_system.create_knowledge_base(
    documents=documents,
    metadatas=metadatas,
    collection_name="ai_knowledge"
)

# Search with different strategies
result = search_system.search(
    query="neural networks for images",
    collection_name="ai_knowledge",
    strategy="hybrid",  # or 'semantic', 'keyword', 'auto'
    n_results=5,
    metadata_filter={"difficulty": "beginner"}
)

# Display results
search_system.display_results(result)

# Get statistics
stats = search_system.get_statistics()
```

**Automatic Strategy Selection:**
The system intelligently selects the best strategy based on query type:
- **Quoted phrases** → Keyword search
- **Technical terms/versions** → Keyword search
- **Question words** (what, why, how) → Semantic search
- **General queries** → Hybrid search

**Search Pipeline:**
```
Query → Strategy Selection → [Semantic + Keyword] → RRF Fusion → Ranking → Results
```

---

## 💡 Key Concepts

### Semantic Search vs Keyword Search

| Aspect | Semantic Search | Keyword Search (BM25) |
|--------|----------------|---------------------|
| **Method** | Vector similarity | Term frequency |
| **Matching** | Conceptual | Exact terms |
| **Synonyms** | Handles well | Misses synonyms |
| **Speed** | Fast (with index) | Very fast |
| **Use Case** | Conceptual queries | Technical terms |

### Top-K Optimization

**Key Principles:**
1. **Larger K** = More results, lower precision
2. **Smaller K** = Fewer results, higher precision
3. Monitor distance distribution
4. Adjust K based on use case
5. Consider filtering low-quality results

### Metadata Filtering Strategies

**Common Filter Patterns:**
```python
# Single field
{"topic": "ML"}

# Multiple fields (AND)
{"$and": [{"topic": "NLP"}, {"difficulty": "advanced"}]}

# Multiple fields (OR)
{"$or": [{"topic": "ML"}, {"topic": "DL"}]}

# Nested conditions
{"$and": [
    {"topic": "NLP"},
    {"$or": [{"difficulty": "intermediate"}, {"difficulty": "advanced"}]}
]}
```

### Hybrid Search Benefits

1. **Better Recall**: Finds more relevant results
2. **Better Precision**: Reduces false positives
3. **Robust**: Works well across query types
4. **Production-ready**: Used by major search engines

---

## 🎯 Best Practices

### Search Strategy Selection

✅ Use **semantic search** for:
- Natural language questions
- Conceptual understanding
- Cross-language queries
- Paraphrase detection

✅ Use **keyword search** for:
- Exact term matching
- Technical documentation
- Code search
- Named entities

✅ Use **hybrid search** for:
- Production systems
- Unknown query types
- Maximum accuracy
- General-purpose search

### Top-K Configuration

✅ Start with K=5 for most use cases
✅ Increase K for exploratory search
✅ Decrease K for precise answers
✅ Monitor distance metrics
✅ Filter results below quality threshold

### Metadata Design

✅ Use descriptive metadata keys
✅ Include categorical attributes (topic, category)
✅ Include difficulty/priority levels
✅ Add temporal information (date, version)
✅ Keep metadata consistent across documents

### Performance Optimization

✅ Use smaller embedding models (384 dims) for speed
✅ Cache frequently used embeddings
✅ Limit n_results for initial retrieval
✅ Use metadata filtering to reduce search space
✅ Monitor query times and optimize

---

## 🔧 Troubleshooting

**Issue:** "ModuleNotFoundError: No module named 'sentence_transformers'"
**Solution:** `pip install sentence-transformers`

**Issue:** "Collection not found"
**Solution:** Ensure you've created the collection with `create_collection()` or `setup_sample_database()`

**Issue:** "Slow search performance"
**Solution:**
- Use smaller embedding model
- Reduce n_results in initial retrieval
- Apply metadata filters to narrow scope
- Consider using keyword search for exact matches

**Issue:** "Poor search results"
**Solution:**
- Try different search strategies
- Adjust Top-K value
- Use hybrid search for better accuracy
- Check query formulation
- Ensure documents are properly indexed

**Issue:** "Empty results with metadata filter"
**Solution:**
- Verify filter syntax
- Check metadata values in collection
- Ensure metadata was added during indexing

---

## 📊 Performance Tips

### Speed Optimization
- Use `all-MiniLM-L6-v2` (384 dims) instead of larger models
- Apply metadata filters early
- Limit initial retrieval to 2x final n_results
- Cache embeddings for repeated queries
- Use keyword search for exact matches

### Quality Optimization
- Use hybrid search for best results
- Tune BM25 parameters (k1, b) for your domain
- Adjust RRF k value (default 60)
- Filter results by distance threshold
- Include rich metadata for filtering

### Scalability
- Use ChromaDB persistent storage
- Batch process document additions
- Index metadata fields
- Monitor database size
- Consider sharding for large collections

---

## 🧪 Testing

Validate all solutions:
```bash
cd solutions/
for file in exercise*.py capstone*.py; do
    python3 -m py_compile "$file" && echo "✅ $file"
done
```

Test each exercise:
```bash
for file in exercise*.py; do
    echo "Testing $file..."
    python3 "$file"
done
```

---

## 📚 Next Steps

After completing Lab 4:
1. **Lab 5:** Complete RAG Pipeline
2. **Lab 6:** Advanced RAG Techniques
3. Build production search applications
4. Integrate with LLM systems

---

## 🎓 What You've Learned

✅ Semantic search with vector databases
✅ Top-K retrieval optimization
✅ Metadata filtering strategies
✅ BM25 keyword search algorithm
✅ Hybrid search with RRF
✅ Production search system architecture
✅ Query routing and strategy selection
✅ Performance monitoring and optimization

---

## 📈 Comparison: Search Methods

### Example Query: "neural networks for image recognition"

**Semantic Search Results:**
1. "Deep learning uses neural networks..." (distance: 0.45)
2. "Computer vision allows machines..." (distance: 0.68)
3. "Convolutional neural networks..." (distance: 0.72)

**Keyword Search Results:**
1. "Convolutional neural networks (CNNs)..." (BM25: 3.24)
2. "Deep learning uses neural networks..." (BM25: 2.87)
3. "Recurrent neural networks process..." (BM25: 2.15)

**Hybrid Search Results (Best):**
1. "Convolutional neural networks (CNNs)..." (RRF: 0.048)
2. "Deep learning uses neural networks..." (RRF: 0.042)
3. "Computer vision allows machines..." (RRF: 0.031)

**Analysis:**
- Semantic: Found conceptually related documents
- Keyword: Prioritized exact term matches
- Hybrid: Combined both for optimal ranking

---

## 🌟 Advanced Topics

### Custom Distance Metrics
ChromaDB supports multiple distance functions:
- **L2 (Euclidean)**: Default, good for most cases
- **Cosine**: Angle-based similarity
- **Inner Product**: Dot product similarity

### Query Expansion
Improve recall by expanding queries:
```python
# Add synonyms
original_query = "car"
expanded_query = "car automobile vehicle"

# Add related terms
query = "machine learning"
expanded = "machine learning ML supervised unsupervised"
```

### Reranking
Improve precision with two-stage retrieval:
1. **Stage 1**: Fast retrieval (get top 100)
2. **Stage 2**: Precise reranking (return top 10)

### Custom RRF Parameters
Tune RRF k value for your use case:
```python
# More aggressive fusion (recent results prioritized)
rrf_k = 30

# More balanced fusion (default)
rrf_k = 60

# More conservative fusion (all ranks considered equally)
rrf_k = 100
```

---

## 📊 Performance Benchmarks

### Typical Query Times (8 documents, all-MiniLM-L6-v2):
- **Semantic search**: ~5-10ms
- **Keyword search (BM25)**: ~1-3ms
- **Hybrid search**: ~10-20ms

### Scaling (approximate):
- **100 documents**: < 50ms
- **1,000 documents**: < 100ms
- **10,000 documents**: < 500ms
- **100,000+ documents**: Consider sharding

---

**Ready for Production! 🚀**

*You now have the skills to build advanced search systems combining semantic and keyword retrieval for maximum accuracy.*
