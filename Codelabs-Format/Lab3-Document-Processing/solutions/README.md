# Lab 3 Solutions: Document Processing & Embeddings

## 📚 Overview

Complete solutions for Lab 3 covering document loading, text chunking, embedding generation, and vector databases. Build production-ready document processing pipelines.

---

## 📁 Files Included

### Sample Data
- **`sample_document.txt`** - Sample AI article for testing

### Core Exercises
- **`exercise1_document_loading.py`** - Load TXT and PDF files
- **`exercise2_text_chunking.py`** - 4 chunking strategies
- **`exercise3_embeddings.py`** - Generate embeddings with sentence-transformers
- **`exercise4_similarity.py`** - Calculate semantic similarity
- **`exercise5_vector_database.py`** - ChromaDB vector database

### Capstone Project
- **`capstone_document_processor.py`** - Complete processing system

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install sentence-transformers chromadb pypdf langchain scikit-learn numpy
```

### Run Exercises
```bash
python exercise1_document_loading.py
python exercise2_text_chunking.py
python exercise3_embeddings.py
python exercise4_similarity.py
python exercise5_vector_database.py
```

### Run Capstone
```bash
python capstone_document_processor.py
```

---

## 📖 Exercise Guide

### Exercise 1: Document Loading (10 min)
**What you'll learn:**
- Load text files with proper encoding
- Extract text from PDFs
- Build DocumentLoader class
- Handle file errors

**Key features:**
```python
loader = DocumentLoader()
doc = loader.load('sample_document.txt')
# Returns: {file_name, file_type, content, length, word_count}
```

---

### Exercise 2: Text Chunking (20 min)
**What you'll learn:**
- Why chunking is necessary
- 4 chunking strategies
- When to use each strategy

**Strategies:**
1. **Fixed-size** - Simple, consistent chunks
2. **Sentence** - Preserves semantic units
3. **Paragraph** - Natural document structure
4. **Recursive** - Hierarchical splitting (best)

**Usage:**
```python
chunker = TextChunker(strategy='recursive', max_chunk_size=500)
chunks = chunker.chunk(text)
```

**Recommendations:**
- **Fixed-size** → Simple documents, consistent sizes needed
- **Sentence** → Q&A, chatbots, semantic search
- **Paragraph** → Articles, documentation
- **Recursive** → Complex documents, best quality

---

### Exercise 3: Embeddings (20 min)
**What you'll learn:**
- Generate embeddings with sentence-transformers
- Compare embedding models
- Batch processing
- Embedding statistics

**Models covered:**
- `all-MiniLM-L6-v2` - Fast, 384 dims, good quality ⭐
- `all-mpnet-base-v2` - Best quality, 768 dims, slower

**Usage:**
```python
generator = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
embeddings = generator.encode(texts, show_progress=True)
stats = generator.get_embedding_stats(embeddings)
```

---

### Exercise 4: Semantic Similarity (15 min)
**What you'll learn:**
- Calculate cosine similarity
- Find most similar texts
- Build similarity search

**Usage:**
```python
calculator = SimilarityCalculator()
similarity = calculator.calculate_similarity(text1, text2)
results = calculator.find_similar(query, corpus, top_k=5)
```

**Output:**
```
Query: "neural networks for learning"
Top Result: "Deep learning uses neural networks"
Score: 0.8542
```

---

### Exercise 5: Vector Database (20 min)
**What you'll learn:**
- Set up ChromaDB
- Store documents with embeddings
- Query by similarity
- Manage collections

**Usage:**
```python
db = VectorDatabase()
collection = db.create_collection("ai_docs")
db.add_documents(collection, documents, metadatas)
results = db.query(collection, "neural networks", n_results=5)
```

**Features:**
- Persistent storage
- Automatic embedding
- Metadata filtering
- Distance-based ranking

---

### 🏆 Capstone: Complete Document Processor

**What you'll build:**
Production-ready document processing system combining all Lab 3 concepts.

**Features:**
✅ Multi-format loading (TXT, PDF, MD)
✅ 4 chunking strategies
✅ Automatic embedding generation
✅ Vector database storage (ChromaDB)
✅ Semantic search
✅ Statistics tracking

**Usage:**
```python
processor = DocumentProcessor(
    embedding_model='all-MiniLM-L6-v2',
    chunking_strategy='recursive',
    chunk_size=300
)

# Process document
result = processor.process_document(
    file_path='sample_document.txt',
    collection_name='knowledge_base',
    metadata={"category": "AI"}
)

# Search
results = processor.search(
    query="neural networks",
    collection_name='knowledge_base',
    n_results=5
)
```

**Pipeline:**
```
Document → Load → Chunk → Embed → Store → Search
```

---

## 💡 Key Concepts

### Why Chunking?
- LLMs have context limits (4K-128K tokens)
- Better precision with smaller chunks
- Maintains context with overlap
- Enables semantic search

### Chunking Strategy Selection

| Use Case | Best Strategy | Chunk Size | Overlap |
|----------|--------------|------------|---------|
| Q&A | Sentence | 2-3 sentences | 1 sentence |
| Chatbots | Recursive | 300-500 chars | 20% |
| Search | Paragraph | Natural | None |
| General | Recursive | 500 chars | 10-20% |

### Embedding Models

| Model | Dimensions | Speed | Quality | Use For |
|-------|-----------|-------|---------|---------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | Production ⭐ |
| all-mpnet-base-v2 | 768 | Medium | Best | High accuracy |
| paraphrase-MiniLM | 384 | Fast | Good | Paraphrase detection |

### Similarity Scores

```
> 0.8 → Very similar
0.6-0.8 → Related
0.4-0.6 → Somewhat related
< 0.4 → Not related
```

---

## 🎯 Best Practices

### Chunking
✅ Use 10-20% overlap for context
✅ Recursive chunking for best quality
✅ Test different chunk sizes
✅ Keep chunks under 512 tokens

### Embeddings
✅ Use all-MiniLM-L6-v2 for production
✅ Batch process for efficiency
✅ Cache embeddings when possible
✅ Normalize vectors for cosine similarity

### Vector Database
✅ Use meaningful collection names
✅ Add metadata for filtering
✅ Set appropriate n_results
✅ Monitor database size

---

## 🔧 Troubleshooting

**Issue:** "ModuleNotFoundError: No module named 'sentence_transformers'"
**Solution:** `pip install sentence-transformers`

**Issue:** "PDF text extraction fails"
**Solution:** `pip install pypdf` and ensure PDF is text-based (not scanned image)

**Issue:** "ChromaDB database locked"
**Solution:** Close other processes using the database

**Issue:** "Out of memory"
**Solution:** Reduce batch_size when encoding embeddings

---

## 📊 Performance Tips

### Speed Optimization
- Use smaller embedding models (384 dims vs 768)
- Batch process documents
- Cache frequently used embeddings
- Use GPU if available

### Quality Optimization
- Use larger chunks for more context
- Add 10-20% overlap
- Use better embedding models
- Include metadata for filtering

---

## 🧪 Testing

Validate all solutions:
```bash
cd solutions/
for file in *.py; do
    python3 -m py_compile "$file" && echo "✅ $file"
done
```

---

## 📚 Next Steps

After completing Lab 3:
1. **Lab 4:** Semantic Search & Retrieval
2. **Lab 5:** Complete RAG Pipeline
3. Build knowledge bases for your projects

---

## 🎓 What You've Learned

✅ Document loading from multiple formats
✅ Text chunking with 4 strategies
✅ Embedding generation with sentence-transformers
✅ Semantic similarity calculation
✅ Vector database (ChromaDB) setup and querying
✅ Complete document processing pipeline

---

**Ready for Production! 🚀**

*You now have the skills to build RAG systems and semantic search applications.*
