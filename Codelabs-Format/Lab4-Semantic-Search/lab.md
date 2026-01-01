# Lab 4: Semantic Search & Retrieval

## 🛠️ Hands-On Lab

**Duration:** 60-90 minutes
**Difficulty:** Intermediate
**Prerequisites:** Lab 3 completed (ChromaDB with stored documents)

---

## What You'll Build

By the end of this lab, you'll have:
- ✅ Semantic search engine querying vector databases
- ✅ Top-K retrieval with configurable parameters
- ✅ Metadata filtering system
- ✅ Hybrid search combining semantic + keyword (BM25)
- ✅ Search comparison and evaluation tools
- ✅ **Capstone**: Production-ready search system with multiple strategies

---

## 📋 Setup

### Step 1: Verify Lab 3 Completion

You need the ChromaDB database from Lab 3:

```bash
# Check if ChromaDB exists
ls -la ./chroma_db

# Should see chroma.sqlite3 and other files
```

### Step 2: Install Additional Libraries

```bash
pip install rank-bm25
pip install matplotlib
```

### Step 3: Create Lab File

```bash
touch lab4_semantic_search.py
```

✅ **Checkpoint**: ChromaDB from Lab 3 exists and libraries installed.

---

## Exercise 1: Basic Semantic Search (15 min)

**Objective:** Query the vector database and retrieve relevant chunks.

### Task 1A: Connect to Database

```python
# lab4_semantic_search.py

from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

# Load the same embedding model from Lab 3
print("Loading embedding model...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print(f"✓ Model loaded (dimension: {embedding_model.get_sentence_embedding_dimension()})")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Get the collection from Lab 3
# Note: Adjust collection name if you used a different name
try:
    collection = client.get_collection(name="lab3_documents")
    print(f"✓ Connected to collection: {collection.name}")
    print(f"  Documents in collection: {collection.count()}")
except Exception as e:
    print(f"Error: {e}")
    print("Hint: Check collection name from Lab 3 or run Lab 3 first")
```

**Expected Output:**
```
✓ Model loaded (dimension: 384)
✓ Connected to collection: lab3_documents
  Documents in collection: 5
```

### Task 1B: Simple Search Function

```python
def semantic_search(query, n_results=3):
    """
    Perform semantic search on the vector database

    Args:
        query: Search query string
        n_results: Number of results to return

    Returns:
        dict: Search results from ChromaDB
    """
    # Convert query to embedding
    query_embedding = embedding_model.encode([query])

    # Query the collection
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["documents", "distances", "metadatas"]
    )

    return results

# Test the search
query = "What is deep learning?"
results = semantic_search(query, n_results=3)

print(f"\n{'='*60}")
print(f"QUERY: {query}")
print('='*60)

for i in range(len(results['documents'][0])):
    distance = results['distances'][0][i]
    document = results['documents'][0][i]

    print(f"\nResult {i+1} (Distance: {distance:.4f}):")
    print(f"{document}")
```

### Task 1C: Interpret Distance Scores

```python
def interpret_distance(distance):
    """
    Provide human-readable interpretation of L2 distance

    Args:
        distance: L2 (Euclidean) distance score

    Returns:
        str: Interpretation
    """
    if distance < 0.3:
        return "Extremely similar ★★★★★"
    elif distance < 0.6:
        return "Very similar ★★★★☆"
    elif distance < 1.0:
        return "Similar ★★★☆☆"
    elif distance < 1.5:
        return "Somewhat related ★★☆☆☆"
    else:
        return "Not very related ★☆☆☆☆"

# Enhanced display
def display_results(query, results):
    """Display search results with interpretations"""

    print(f"\n{'='*70}")
    print(f"SEARCH: {query}")
    print('='*70)

    for i in range(len(results['documents'][0])):
        distance = results['distances'][0][i]
        document = results['documents'][0][i]
        metadata = results['metadatas'][0][i] if results['metadatas'] else {}

        interpretation = interpret_distance(distance)

        print(f"\n[Result {i+1}] {interpretation}")
        print(f"Distance: {distance:.4f}")
        if metadata:
            print(f"Metadata: {metadata}")
        print(f"Text: {document[:200]}...")
        print("-" * 70)

# Test with different queries
test_queries = [
    "What is machine learning?",
    "How do neural networks work?",
    "Tell me about computer vision",
    "What is AI ethics?"
]

for query in test_queries:
    results = semantic_search(query, n_results=2)
    display_results(query, results)
```

✅ **Checkpoint**: You should see relevant results for each query with distance scores.

---

## Exercise 2: Top-K Retrieval Strategies (15 min)

**Objective:** Understand how different K values affect results.

### Task 2A: Compare Different K Values

```python
def compare_k_values(query, k_values=[1, 3, 5, 10]):
    """
    Compare search results with different K values

    Args:
        query: Search query
        k_values: List of K values to test
    """
    print(f"\n{'='*70}")
    print(f"TOP-K COMPARISON: '{query}'")
    print('='*70)

    for k in k_values:
        results = semantic_search(query, n_results=k)
        distances = results['distances'][0]

        print(f"\nK={k}:")
        print(f"  Results returned: {len(distances)}")
        print(f"  Best distance: {min(distances):.4f} ({interpret_distance(min(distances))})")
        print(f"  Average distance: {np.mean(distances):.4f}")
        print(f"  Worst distance: {max(distances):.4f} ({interpret_distance(max(distances))})")

# Test
compare_k_values("What is machine learning?")
compare_k_values("Tell me about reinforcement learning")
```

### Task 2B: Quality vs. Quantity Analysis

```python
def analyze_retrieval_quality(query, max_k=10):
    """
    Analyze how result quality degrades with higher K

    Args:
        query: Search query
        max_k: Maximum K to test
    """
    results = semantic_search(query, n_results=max_k)
    distances = results['distances'][0]

    print(f"\n{'='*70}")
    print(f"RETRIEVAL QUALITY ANALYSIS: '{query}'")
    print('='*70)

    # Calculate quality metrics at different K
    for k in [1, 3, 5, max_k]:
        if k <= len(distances):
            top_k_distances = distances[:k]
            avg_distance = np.mean(top_k_distances)
            relevant_count = sum(1 for d in top_k_distances if d < 1.0)

            print(f"\nTop-{k} Results:")
            print(f"  Average distance: {avg_distance:.4f}")
            print(f"  Relevant (distance < 1.0): {relevant_count}/{k}")
            print(f"  Quality score: {relevant_count/k * 100:.1f}%")

# Test
analyze_retrieval_quality("What is deep learning?", max_k=10)
```

### Task 2C: Find Optimal K

```python
def find_optimal_k(query, distance_threshold=1.0, max_k=20):
    """
    Find optimal K based on distance threshold

    Args:
        query: Search query
        distance_threshold: Maximum acceptable distance
        max_k: Maximum K to consider

    Returns:
        int: Recommended K value
    """
    results = semantic_search(query, n_results=max_k)
    distances = results['distances'][0]

    # Find where quality drops below threshold
    optimal_k = 0
    for i, distance in enumerate(distances):
        if distance <= distance_threshold:
            optimal_k = i + 1
        else:
            break

    print(f"\nQuery: '{query}'")
    print(f"Distance threshold: {distance_threshold}")
    print(f"Recommended K: {optimal_k}")
    print(f"\nRationale:")
    for i in range(min(optimal_k + 2, len(distances))):
        status = "✓ Include" if distances[i] <= distance_threshold else "✗ Exclude"
        print(f"  Result {i+1}: {distances[i]:.4f} {status}")

    return optimal_k

# Test with different queries
find_optimal_k("What is machine learning?", distance_threshold=0.8)
find_optimal_k("Tell me about NLP", distance_threshold=1.0)
```

✅ **Checkpoint**: You should see how different K values affect result quality.

---

## Exercise 3: Metadata Filtering (15 min)

**Objective:** Use metadata to refine searches.

### Task 3A: Add Metadata to New Documents

First, let's add documents with rich metadata:

```python
def add_documents_with_metadata():
    """Add sample documents with detailed metadata"""

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # Sample documents with different sources
    documents = {
        "policy.txt": "Our return policy allows customers to return unused items within 30 days of purchase. Items must be in original packaging. Refunds are processed within 5-7 business days.",
        "faq.txt": "Q: How long does shipping take? A: Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days.",
        "guide.txt": "To set up your device, first charge it fully. Then download the companion app from the app store. Follow the in-app instructions to pair your device."
    }

    # Chunk and add each document
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    for source, content in documents.items():
        chunks = splitter.split_text(content)
        embeddings = embedding_model.encode(chunks)

        # Get current count for ID generation
        current_count = collection.count()

        # Create IDs and metadata
        ids = [f"doc_{current_count + i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": source,
                "document_type": source.split('.')[0],
                "chunk_index": i,
                "chunk_size": len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]

        # Add to collection
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=metadatas
        )

        print(f"✓ Added {len(chunks)} chunks from {source}")

    print(f"\nTotal documents in collection: {collection.count()}")

# Add documents
# add_documents_with_metadata()  # Uncomment to add more documents
```

### Task 3B: Search with Filters

```python
def search_with_filter(query, where_filter, n_results=3):
    """
    Search with metadata filtering

    Args:
        query: Search query
        where_filter: ChromaDB filter dictionary
        n_results: Number of results

    Returns:
        dict: Filtered search results
    """
    query_embedding = embedding_model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        where=where_filter,
        include=["documents", "distances", "metadatas"]
    )

    return results

# Example filters
print("\n" + "="*70)
print("FILTERED SEARCH EXAMPLES")
print("="*70)

# Filter 1: Search only in policy documents
query = "How do I return an item?"
filter1 = {"document_type": {"$eq": "policy"}}
results1 = search_with_filter(query, filter1, n_results=2)

print(f"\n1. Query: '{query}'")
print(f"   Filter: document_type = 'policy'")
for i, (doc, meta) in enumerate(zip(results1['documents'][0], results1['metadatas'][0])):
    print(f"\n   Result {i+1}:")
    print(f"   Source: {meta.get('source', 'unknown')}")
    print(f"   Text: {doc[:100]}...")

# Filter 2: Search only in FAQ
query2 = "shipping time"
filter2 = {"document_type": {"$eq": "faq"}}
results2 = search_with_filter(query2, filter2, n_results=2)

print(f"\n2. Query: '{query2}'")
print(f"   Filter: document_type = 'faq'")
for i, (doc, meta) in enumerate(zip(results2['documents'][0], results2['metadatas'][0])):
    print(f"\n   Result {i+1}:")
    print(f"   Source: {meta.get('source', 'unknown')}")
    print(f"   Text: {doc[:100]}...")
```

### Task 3C: Advanced Filter Combinations

```python
def demonstrate_advanced_filters():
    """Show different filter combinations"""

    print(f"\n{'='*70}")
    print("ADVANCED FILTERING")
    print('='*70)

    # Filter 1: Multiple sources
    print("\n1. Search in multiple document types:")
    filter_multi = {"document_type": {"$in": ["policy", "faq"]}}
    results = search_with_filter("How long?", filter_multi, n_results=3)
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"   - {meta['document_type']}: {doc[:60]}...")

    # Filter 2: Chunk size range
    print("\n2. Search only in larger chunks:")
    filter_size = {"chunk_size": {"$gte": 100}}
    results = search_with_filter("return policy", filter_size, n_results=2)
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"   - Size {meta['chunk_size']}: {doc[:60]}...")

    # Filter 3: Combined filters
    print("\n3. Combined filters (AND logic):")
    filter_combined = {
        "$and": [
            {"document_type": {"$eq": "policy"}},
            {"chunk_size": {"$gte": 50}}
        ]
    }
    results = search_with_filter("refund", filter_combined, n_results=2)
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"   - {meta['document_type']} (size: {meta['chunk_size']}): {doc[:50]}...")

demonstrate_advanced_filters()
```

✅ **Checkpoint**: Filtered searches should return only results matching the metadata criteria.

---

## Exercise 4: Hybrid Search with BM25 (25 min)

**Objective:** Combine semantic search with keyword search for better results.

### Task 4A: Implement BM25 Keyword Search

```python
from rank_bm25 import BM25Okapi

class HybridSearcher:
    """
    Hybrid search combining semantic (vector) and keyword (BM25) search
    """

    def __init__(self, collection, embedding_model):
        """
        Initialize hybrid searcher

        Args:
            collection: ChromaDB collection
            embedding_model: SentenceTransformer model
        """
        self.collection = collection
        self.embedding_model = embedding_model

        # Get all documents for BM25 indexing
        all_data = collection.get(include=["documents"])
        self.documents = all_data['documents']
        self.doc_ids = all_data['ids']

        # Create BM25 index
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        print(f"✓ Hybrid searcher initialized")
        print(f"  Indexed {len(self.documents)} documents for BM25")

    def keyword_search(self, query, top_k=10):
        """
        Perform BM25 keyword search

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            list: Results with BM25 scores
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only non-zero scores
                results.append({
                    'document': self.documents[idx],
                    'score': scores[idx],
                    'id': self.doc_ids[idx],
                    'index': idx
                })

        return results

    def semantic_search(self, query, top_k=10):
        """
        Perform semantic vector search

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            list: Results with similarity scores
        """
        query_embedding = self.embedding_model.encode([query])

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["documents", "distances"]
        )

        semantic_results = []
        for doc, distance in zip(results['documents'][0], results['distances'][0]):
            # Convert distance to similarity score
            similarity = 1 / (1 + distance)
            semantic_results.append({
                'document': doc,
                'score': similarity,
                'distance': distance
            })

        return semantic_results

    def hybrid_search(self, query, top_k=5, semantic_weight=0.5, keyword_weight=0.5):
        """
        Combine semantic and keyword search

        Args:
            query: Search query
            top_k: Number of final results
            semantic_weight: Weight for semantic scores (0-1)
            keyword_weight: Weight for keyword scores (0-1)

        Returns:
            list: Ranked hybrid results
        """
        # Get results from both methods
        semantic_results = self.semantic_search(query, top_k=10)
        keyword_results = self.keyword_search(query, top_k=10)

        # Normalize scores to 0-1 range
        if semantic_results:
            max_sem = max(r['score'] for r in semantic_results)
            min_sem = min(r['score'] for r in semantic_results)
            sem_range = max_sem - min_sem if max_sem != min_sem else 1

            for r in semantic_results:
                r['normalized_score'] = (r['score'] - min_sem) / sem_range

        if keyword_results:
            max_kw = max(r['score'] for r in keyword_results)
            if max_kw > 0:
                for r in keyword_results:
                    r['normalized_score'] = r['score'] / max_kw

        # Combine scores
        combined_scores = {}

        for result in semantic_results:
            doc = result['document']
            combined_scores[doc] = semantic_weight * result['normalized_score']

        for result in keyword_results:
            doc = result['document']
            if doc in combined_scores:
                combined_scores[doc] += keyword_weight * result['normalized_score']
            else:
                combined_scores[doc] = keyword_weight * result['normalized_score']

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [{'document': doc, 'hybrid_score': score} for doc, score in sorted_results]

# Initialize hybrid searcher
hybrid_searcher = HybridSearcher(collection, embedding_model)
```

### Task 4B: Compare Search Methods

```python
def compare_search_methods(query):
    """
    Compare semantic, keyword, and hybrid search side-by-side

    Args:
        query: Search query
    """
    print(f"\n{'='*70}")
    print(f"SEARCH METHOD COMPARISON")
    print(f"Query: '{query}'")
    print('='*70)

    # Semantic search
    print("\n1. SEMANTIC SEARCH (Vector-based):")
    print("-" * 70)
    semantic_results = hybrid_searcher.semantic_search(query, top_k=3)
    for i, result in enumerate(semantic_results):
        print(f"[{i+1}] Similarity: {result['score']:.4f} | Distance: {result['distance']:.4f}")
        print(f"    {result['document'][:80]}...")

    # Keyword search
    print("\n2. KEYWORD SEARCH (BM25):")
    print("-" * 70)
    keyword_results = hybrid_searcher.keyword_search(query, top_k=3)
    if keyword_results:
        for i, result in enumerate(keyword_results):
            print(f"[{i+1}] BM25 Score: {result['score']:.4f}")
            print(f"    {result['document'][:80]}...")
    else:
        print("No keyword matches found.")

    # Hybrid search
    print("\n3. HYBRID SEARCH (50/50 blend):")
    print("-" * 70)
    hybrid_results = hybrid_searcher.hybrid_search(
        query,
        top_k=3,
        semantic_weight=0.5,
        keyword_weight=0.5
    )
    for i, result in enumerate(hybrid_results):
        print(f"[{i+1}] Hybrid Score: {result['hybrid_score']:.4f}")
        print(f"    {result['document'][:80]}...")

# Test with different query types
test_queries = [
    "What is deep learning?",          # Conceptual
    "NLP",                             # Acronym
    "neural networks layers",           # Mix
    "teaching computers to understand" # Paraphrase
]

for query in test_queries:
    compare_search_methods(query)
```

### Task 4C: Tune Hybrid Search Weights

```python
def tune_hybrid_weights(query):
    """
    Test different weight combinations

    Args:
        query: Search query
    """
    print(f"\n{'='*70}")
    print(f"HYBRID SEARCH WEIGHT TUNING")
    print(f"Query: '{query}'")
    print('='*70)

    weight_combinations = [
        (1.0, 0.0, "100% Semantic"),
        (0.7, 0.3, "70% Semantic, 30% Keyword"),
        (0.5, 0.5, "50/50 Balanced"),
        (0.3, 0.7, "30% Semantic, 70% Keyword"),
        (0.0, 1.0, "100% Keyword"),
    ]

    for sem_w, kw_w, description in weight_combinations:
        print(f"\n{description}:")
        print("-" * 70)

        results = hybrid_searcher.hybrid_search(
            query,
            top_k=3,
            semantic_weight=sem_w,
            keyword_weight=kw_w
        )

        for i, result in enumerate(results):
            print(f"  [{i+1}] Score: {result['hybrid_score']:.4f}")
            print(f"      {result['document'][:70]}...")

# Test
tune_hybrid_weights("deep learning neural networks")
tune_hybrid_weights("machine learning")
```

✅ **Checkpoint**: Hybrid search should combine benefits of both semantic and keyword approaches.

---

## 🎯 Capstone Project: Production Search System (30 min)

**Objective:** Build a complete, production-ready search system with multiple strategies.

### Requirements

Your system must:
1. ✅ Support semantic, keyword, and hybrid search
2. ✅ Allow configurable Top-K values
3. ✅ Support metadata filtering
4. ✅ Provide search quality metrics
5. ✅ Handle edge cases gracefully
6. ✅ Include a user-friendly interface

### Complete Implementation

```python
# production_search_system.py

class ProductionSearchSystem:
    """
    Production-ready search system with multiple strategies
    """

    def __init__(self, collection, embedding_model):
        """Initialize the search system"""
        self.collection = collection
        self.embedding_model = embedding_model

        # Initialize hybrid searcher
        self.hybrid_searcher = HybridSearcher(collection, embedding_model)

        print("✓ Production Search System initialized")
        print(f"  Collection size: {collection.count()} documents")

    def search(
        self,
        query,
        method="hybrid",
        top_k=5,
        semantic_weight=0.6,
        keyword_weight=0.4,
        metadata_filter=None,
        distance_threshold=None
    ):
        """
        Unified search interface

        Args:
            query: Search query
            method: 'semantic', 'keyword', or 'hybrid'
            top_k: Number of results
            semantic_weight: Weight for semantic scores (hybrid only)
            keyword_weight: Weight for keyword scores (hybrid only)
            metadata_filter: Optional metadata filter
            distance_threshold: Optional max distance for results

        Returns:
            dict: Search results with metrics
        """
        start_time = time.time()

        # Perform search based on method
        if method == "semantic":
            results = self._semantic_search_filtered(
                query, top_k, metadata_filter, distance_threshold
            )
        elif method == "keyword":
            results = self.hybrid_searcher.keyword_search(query, top_k)
        elif method == "hybrid":
            # Apply filter in semantic component
            raw_results = self.hybrid_searcher.hybrid_search(
                query, top_k*2, semantic_weight, keyword_weight
            )
            results = raw_results[:top_k]
        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed_time = time.time() - start_time

        # Calculate metrics
        metrics = self._calculate_metrics(results, method)

        return {
            'query': query,
            'method': method,
            'results': results,
            'metrics': {
                **metrics,
                'search_time_ms': elapsed_time * 1000,
                'results_returned': len(results)
            }
        }

    def _semantic_search_filtered(self, query, top_k, metadata_filter, distance_threshold):
        """Semantic search with optional filtering"""
        query_embedding = self.embedding_model.encode([query])

        search_params = {
            'query_embeddings': query_embedding.tolist(),
            'n_results': top_k,
            'include': ['documents', 'distances', 'metadatas']
        }

        if metadata_filter:
            search_params['where'] = metadata_filter

        results = self.collection.query(**search_params)

        # Format results
        formatted = []
        for i in range(len(results['documents'][0])):
            distance = results['distances'][0][i]

            # Apply distance threshold if specified
            if distance_threshold and distance > distance_threshold:
                continue

            formatted.append({
                'document': results['documents'][0][i],
                'score': 1 / (1 + distance),
                'distance': distance,
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
            })

        return formatted

    def _calculate_metrics(self, results, method):
        """Calculate quality metrics for results"""
        if not results:
            return {
                'avg_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0
            }

        if method == "semantic":
            scores = [r.get('score', 0) for r in results]
        elif method == "keyword":
            scores = [r.get('score', 0) for r in results]
        elif method == "hybrid":
            scores = [r.get('hybrid_score', 0) for r in results]
        else:
            scores = [0]

        return {
            'avg_score': np.mean(scores) if scores else 0.0,
            'min_score': min(scores) if scores else 0.0,
            'max_score': max(scores) if scores else 0.0,
            'std_score': np.std(scores) if scores else 0.0
        }

    def interactive_search(self):
        """Interactive search interface"""
        print("\n" + "="*70)
        print("PRODUCTION SEARCH SYSTEM - Interactive Mode")
        print("="*70)
        print("\nCommands:")
        print("  Type your query to search")
        print("  ':method <semantic|keyword|hybrid>' - Change search method")
        print("  ':k <number>' - Set number of results")
        print("  ':quit' - Exit")
        print("\nCurrent settings: method=hybrid, k=3")
        print("-"*70)

        method = "hybrid"
        top_k = 3

        while True:
            query = input("\nQuery> ").strip()

            if not query:
                continue

            # Handle commands
            if query.startswith(':'):
                if query.lower() in [':quit', ':exit', ':q']:
                    print("Goodbye!")
                    break
                elif query.startswith(':method '):
                    method = query.split()[1]
                    print(f"✓ Method set to: {method}")
                    continue
                elif query.startswith(':k '):
                    top_k = int(query.split()[1])
                    print(f"✓ K set to: {top_k}")
                    continue
                else:
                    print("Unknown command")
                    continue

            # Perform search
            result = self.search(query, method=method, top_k=top_k)

            # Display results
            print(f"\n{'='*70}")
            print(f"Results ({result['method']}):")
            print(f"Search time: {result['metrics']['search_time_ms']:.2f}ms")
            print(f"Avg score: {result['metrics']['avg_score']:.4f}")
            print('-'*70)

            for i, res in enumerate(result['results']):
                score_key = 'hybrid_score' if method == 'hybrid' else 'score'
                score = res.get(score_key, res.get('score', 0))

                print(f"\n[{i+1}] Score: {score:.4f}")
                print(f"{res['document'][:150]}...")

    def batch_evaluate(self, test_queries):
        """
        Evaluate search quality on a batch of queries

        Args:
            test_queries: List of test queries

        Returns:
            dict: Evaluation metrics
        """
        print(f"\n{'='*70}")
        print(f"BATCH EVALUATION: {len(test_queries)} queries")
        print('='*70)

        methods = ['semantic', 'keyword', 'hybrid']
        results = {method: [] for method in methods}

        for query in test_queries:
            for method in methods:
                result = self.search(query, method=method, top_k=3)
                results[method].append(result['metrics'])

        # Aggregate metrics
        print(f"\n{'Method':<15} {'Avg Score':<12} {'Avg Time (ms)':<15}")
        print('-'*70)

        for method in methods:
            avg_score = np.mean([r['avg_score'] for r in results[method]])
            avg_time = np.mean([r['search_time_ms'] for r in results[method]])
            print(f"{method:<15} {avg_score:<12.4f} {avg_time:<15.2f}")

        return results


# Initialize the system
import time

search_system = ProductionSearchSystem(collection, embedding_model)

# Test batch evaluation
test_queries = [
    "What is machine learning?",
    "deep learning neural networks",
    "computer vision applications",
    "NLP transformers",
    "reinforcement learning agents"
]

evaluation_results = search_system.batch_evaluate(test_queries)

# Run interactive search (optional)
# search_system.interactive_search()
```

### Your Tasks

1. **Run the production system** and test all search methods
2. **Add error handling** for edge cases (empty queries, no results)
3. **Implement result caching** to speed up repeated queries
4. **Add search history tracking**
5. **Create a confidence score** that combines distance and score

✅ **Checkpoint**: Production system should handle all search methods with quality metrics.

---

## 🏆 Extension Challenges

### Challenge 1: Reciprocal Rank Fusion (RRF)

Implement RRF for combining rankings:

```python
def reciprocal_rank_fusion(semantic_results, keyword_results, k=60, top_n=5):
    """
    Combine rankings using RRF

    Args:
        semantic_results: Results from semantic search
        keyword_results: Results from keyword search
        k: RRF constant (typically 60)
        top_n: Number of results to return

    Returns:
        list: Fused results
    """
    scores = {}

    # Add semantic rankings
    for rank, result in enumerate(semantic_results):
        doc = result['document']
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)

    # Add keyword rankings
    for rank, result in enumerate(keyword_results):
        doc = result['document']
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)

    # Sort by RRF score
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return [{'document': doc, 'rrf_score': score} for doc, score in sorted_results]
```

### Challenge 2: Query Expansion

Expand user queries with synonyms:

```python
def expand_query(query):
    """
    Expand query with synonyms for better recall

    Args:
        query: Original query

    Returns:
        str: Expanded query
    """
    # Simple expansion (in production, use WordNet or similar)
    expansions = {
        'ML': 'machine learning',
        'AI': 'artificial intelligence',
        'DL': 'deep learning',
        'NLP': 'natural language processing'
    }

    expanded = query
    for abbrev, full in expansions.items():
        if abbrev in query:
            expanded += f" {full}"

    return expanded
```

### Challenge 3: Result Re-ranking

Implement cross-encoder for re-ranking:

```python
from sentence_transformers import CrossEncoder

def rerank_results(query, results, top_k=5):
    """
    Re-rank results using cross-encoder

    Args:
        query: Search query
        results: Initial search results
        top_k: Number of final results

    Returns:
        list: Re-ranked results
    """
    # Load cross-encoder model
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Create query-document pairs
    pairs = [[query, r['document']] for r in results]

    # Score pairs
    scores = cross_encoder.predict(pairs)

    # Re-rank
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

    return [{'document': r['document'], 'rerank_score': float(s)}
            for r, s in ranked[:top_k]]
```

---

## 📝 Key Takeaways

After completing this lab, you should understand:

✅ **Semantic Search** - Finding by meaning, not just keywords
✅ **Distance Metrics** - L2, cosine similarity, and interpretation
✅ **Top-K Retrieval** - Balancing quality vs. quantity
✅ **Metadata Filtering** - Refining searches with filters
✅ **Hybrid Search** - Best of semantic + keyword
✅ **Production Considerations** - Speed, quality, edge cases

---

## 🔍 Troubleshooting

**Issue**: "Collection not found"

**Solution**: Run Lab 3 first to create the ChromaDB collection, or adjust the collection name.

---

**Issue**: "BM25 returns no results"

**Solution**: BM25 requires exact word matches. For conceptual queries, use semantic or hybrid search.

---

**Issue**: "Hybrid search seems worse than semantic alone"

**Solution**: Adjust weights based on query type. Try 70% semantic, 30% keyword for conceptual queries.

---

## 🎓 What's Next?

You've completed Lab 4! You now have:
- ✅ Working semantic search system
- ✅ Hybrid search with BM25
- ✅ Production-ready search infrastructure
- ✅ Quality metrics and evaluation tools

**Next Lab**: Lab 5 - Complete RAG Pipeline
Combine retrieval (Lab 4) with generation (LLM) to create a full RAG system!

---

**Lab 4 Complete!** 🎉
[← Back to Learning Material](learning.md) | [Next: Lab 5 →](../Lab5-RAG-Pipeline/learning.md)
