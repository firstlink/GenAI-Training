# Session 3: Building RAG Systems
## Retrieval-Augmented Generation

**Duration**: 90 minutes
**Difficulty**: Intermediate
**Colab Notebook**: [03_RAG_Systems.ipynb](../notebooks/03_RAG_Systems.ipynb)

---

## Learning Objectives

By the end of this session, you will:
- 🎯 Understand what RAG is and why it's essential
- 🎯 Load and process documents for RAG pipelines
- 🎯 Implement effective chunking strategies
- 🎯 Generate and store embeddings in vector databases
- 🎯 Build semantic search capabilities
- 🎯 Create an end-to-end RAG application
- 🎯 Evaluate and optimize RAG performance

---

## What is RAG?

### The Problem with Plain LLMs

Large Language Models have limitations:

❌ **Knowledge Cutoff**: Training data has a cutoff date (e.g., April 2023)
❌ **No Private Data**: Can't access your company's documents
❌ **Hallucinations**: May generate plausible but incorrect information
❌ **No Source Attribution**: Can't cite where information comes from

### The RAG Solution

**Retrieval-Augmented Generation** solves these problems by:

1. **Storing your documents** in a searchable format (vector database)
2. **Retrieving relevant information** when a question is asked
3. **Augmenting the LLM prompt** with retrieved context
4. **Generating grounded answers** based on your actual documents

### RAG Architecture

```
User Question: "What is our refund policy?"
        ↓
  [1. EMBED QUESTION]
        ↓
  Vector: [0.23, -0.45, 0.67, ...]
        ↓
  [2. SEARCH VECTOR DB]
        ↓
  Top 3 relevant chunks:
  - Chunk 1: "Our refund policy allows..."
  - Chunk 2: "To request a refund..."
  - Chunk 3: "Refunds are processed within..."
        ↓
  [3. AUGMENT PROMPT]
        ↓
  Prompt: "Based on these documents: [...chunks...]
          Answer: What is our refund policy?"
        ↓
  [4. GENERATE RESPONSE]
        ↓
  Answer: "According to our policy, refunds are..."
```

---

## Part 1: Document Loading and Processing

### Step 1.1: Prepare Sample Documents

We'll work with company documentation as an example:

```python
# Create sample documents
documents = [
    {
        "content": """
        Product Return Policy

        Our company offers a 30-day return policy for all products.
        To be eligible for a return, items must be unused and in their
        original packaging. Customers can initiate a return by contacting
        customer service at support@example.com or calling 1-800-RETURNS.

        Refunds are processed within 5-7 business days after we receive
        the returned item. The refund will be issued to the original
        payment method. Shipping costs are non-refundable unless the
        return is due to our error.
        """,
        "metadata": {"source": "return_policy.txt", "department": "customer_service"}
    },
    {
        "content": """
        Shipping Information

        We offer three shipping options:
        - Standard Shipping: 5-7 business days ($5.99)
        - Express Shipping: 2-3 business days ($12.99)
        - Overnight Shipping: 1 business day ($24.99)

        All orders over $50 qualify for free standard shipping.
        International shipping is available to select countries.
        Tracking information is provided via email once the order ships.
        """,
        "metadata": {"source": "shipping_info.txt", "department": "logistics"}
    },
    {
        "content": """
        Customer Support Hours

        Our customer support team is available:
        - Monday to Friday: 9 AM - 9 PM EST
        - Saturday: 10 AM - 6 PM EST
        - Sunday: 12 PM - 5 PM EST

        Contact methods:
        - Phone: 1-800-SUPPORT
        - Email: support@example.com
        - Live Chat: Available on our website during business hours

        Average response time for emails is 24 hours on business days.
        """,
        "metadata": {"source": "support_hours.txt", "department": "customer_service"}
    },
    {
        "content": """
        Product Warranty

        All products come with a standard 1-year manufacturer's warranty
        covering defects in materials and workmanship. Extended warranty
        plans are available for purchase at checkout.

        Warranty claims can be submitted through our website or by
        contacting customer service. Proof of purchase is required for
        all warranty claims. The warranty does not cover damage from
        misuse, accidents, or normal wear and tear.
        """,
        "metadata": {"source": "warranty_info.txt", "department": "product"}
    }
]

# Save documents to text files
import os

os.makedirs('sample_docs', exist_ok=True)

for doc in documents:
    source = doc['metadata']['source']
    with open(f'sample_docs/{source}', 'w') as f:
        f.write(doc['content'])

print(f"✅ Created {len(documents)} sample documents")
```

### Step 1.2: Load Documents from Files

```python
from pathlib import Path

def load_documents_from_directory(directory_path):
    """Load all text documents from a directory"""
    documents = []

    directory = Path(directory_path)
    for file_path in directory.glob('*.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        documents.append({
            'content': content,
            'metadata': {
                'source': file_path.name,
                'path': str(file_path)
            }
        })

    return documents

# Load the documents
loaded_docs = load_documents_from_directory('sample_docs')
print(f"✅ Loaded {len(loaded_docs)} documents")

for doc in loaded_docs:
    print(f"  - {doc['metadata']['source']}: {len(doc['content'])} characters")
```

### Step 1.3: Load PDFs (Optional)

```python
!pip install pypdf

from pypdf import PdfReader

def load_pdf_document(pdf_path):
    """Extract text from PDF file"""
    reader = PdfReader(pdf_path)
    text = ""

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text += f"\n--- Page {page_num + 1} ---\n{page_text}"

    return {
        'content': text,
        'metadata': {
            'source': pdf_path,
            'num_pages': len(reader.pages)
        }
    }

# Example usage:
# pdf_doc = load_pdf_document('my_document.pdf')
```

---

## Part 2: Text Chunking Strategies

### Why Chunk Documents?

Documents must be split into smaller chunks because:

1. **LLM Context Limits**: Models have maximum context length
2. **Embedding Quality**: Embeddings work better on focused content
3. **Retrieval Precision**: Smaller chunks mean more precise matches
4. **Cost Optimization**: Only process relevant sections

### Step 2.1: Simple Character-Based Chunking

```python
def simple_chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into chunks of specified size with overlap

    Args:
        text: The text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks

# Test on a single document
test_doc = loaded_docs[0]
simple_chunks = simple_chunk_text(test_doc['content'], chunk_size=300, overlap=50)

print(f"Created {len(simple_chunks)} chunks from document")
print("\nFirst chunk:")
print(simple_chunks[0])
print("\nSecond chunk (note the overlap):")
print(simple_chunks[1])
```

### Step 2.2: Intelligent Chunking with LangChain

```python
!pip install langchain langchain-community

from langchain.text_splitter import RecursiveCharacterTextSplitter

def intelligent_chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    """
    Split documents using LangChain's intelligent text splitter
    This preserves sentence and paragraph boundaries
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Try to split on paragraphs, then sentences
    )

    all_chunks = []

    for doc in documents:
        chunks = text_splitter.split_text(doc['content'])

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'content': chunk,
                'metadata': {
                    **doc['metadata'],
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            })

    return all_chunks

# Chunk all documents
chunked_docs = intelligent_chunk_documents(loaded_docs, chunk_size=400, chunk_overlap=100)

print(f"✅ Created {len(chunked_docs)} chunks from {len(loaded_docs)} documents")
print("\nChunk distribution:")
for doc in loaded_docs:
    source = doc['metadata']['source']
    count = sum(1 for c in chunked_docs if c['metadata']['source'] == source)
    print(f"  - {source}: {count} chunks")
```

### Step 2.3: Comparing Chunking Strategies

```python
# Visualize chunk content
print("=" * 60)
print("CHUNK EXAMPLE")
print("=" * 60)
print(f"Source: {chunked_docs[0]['metadata']['source']}")
print(f"Chunk {chunked_docs[0]['metadata']['chunk_index'] + 1} of {chunked_docs[0]['metadata']['total_chunks']}")
print("-" * 60)
print(chunked_docs[0]['content'])
print("=" * 60)
```

---

## Part 3: Generate Embeddings

### Step 3.1: Understanding Embeddings

**What are embeddings?**
- Numerical representations of text (vectors)
- Similar meaning → Similar vectors
- Example: "dog" and "puppy" have similar embeddings

**Embedding dimensions:**
- Typical range: 384 to 1536 dimensions
- Higher dimensions = more information (but slower/costlier)

### Step 3.2: Using Sentence Transformers

```python
!pip install sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
# Options:
# - 'all-MiniLM-L6-v2': Fast, 384 dimensions, good for beginners
# - 'all-mpnet-base-v2': Higher quality, 768 dimensions
model_name = 'all-MiniLM-L6-v2'
embedding_model = SentenceTransformer(model_name)

print(f"✅ Loaded model: {model_name}")
print(f"   Embedding dimension: {embedding_model.get_sentence_embedding_dimension()}")
```

### Step 3.3: Generate Embeddings for All Chunks

```python
# Extract just the text content
chunk_texts = [chunk['content'] for chunk in chunked_docs]

# Generate embeddings (batched for efficiency)
print("Generating embeddings...")
embeddings = embedding_model.encode(
    chunk_texts,
    show_progress_bar=True,
    batch_size=32
)

print(f"✅ Generated {len(embeddings)} embeddings")
print(f"   Shape: {embeddings.shape}")
print(f"   Each embedding: {embeddings[0].shape}")

# Example embedding values (first 10 dimensions)
print(f"\nFirst embedding (first 10 values):")
print(embeddings[0][:10])
```

### Step 3.4: Visualize Embedding Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(text1, text2):
    """Calculate cosine similarity between two texts"""
    emb1 = embedding_model.encode([text1])
    emb2 = embedding_model.encode([text2])
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity

# Test similarity
examples = [
    ("What is the return policy?", "How do I return a product?"),
    ("What is the return policy?", "What are the shipping options?"),
    ("refund and returns", "Product Return Policy")
]

print("Similarity Examples:")
print("-" * 60)
for text1, text2 in examples:
    sim = calculate_similarity(text1, text2)
    print(f"'{text1[:30]}...' vs")
    print(f"'{text2[:30]}...'")
    print(f"Similarity: {sim:.4f}\n")
```

---

## Part 4: Store in Vector Database

### Step 4.1: Using ChromaDB

```python
!pip install chromadb

import chromadb
from chromadb.config import Settings

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
collection_name = "company_knowledge_base"

# Delete if exists (for fresh start)
try:
    client.delete_collection(collection_name)
except:
    pass

collection = client.create_collection(
    name=collection_name,
    metadata={"description": "Company documentation for RAG"}
)

print(f"✅ Created collection: {collection_name}")
```

### Step 4.2: Add Documents with Embeddings

```python
# Prepare data for ChromaDB
ids = [f"chunk_{i}" for i in range(len(chunked_docs))]
documents = [chunk['content'] for chunk in chunked_docs]
metadatas = [chunk['metadata'] for chunk in chunked_docs]
embeddings_list = [emb.tolist() for emb in embeddings]

# Add to collection
collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings_list,
    metadatas=metadatas
)

print(f"✅ Added {collection.count()} documents to ChromaDB")
```

### Step 4.3: Verify Storage

```python
# Retrieve a specific chunk
result = collection.get(
    ids=["chunk_0"],
    include=["documents", "metadatas", "embeddings"]
)

print("Retrieved chunk_0:")
print(f"Document: {result['documents'][0][:100]}...")
print(f"Metadata: {result['metadatas'][0]}")
print(f"Embedding dimensions: {len(result['embeddings'][0])}")
```

---

## Part 5: Semantic Search

### Step 5.1: Basic Query

```python
def search_knowledge_base(query, n_results=3):
    """Search the knowledge base for relevant chunks"""

    # Generate embedding for query
    query_embedding = embedding_model.encode([query])[0]

    # Search vector database
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    return results

# Test query
query = "What is the return policy?"
results = search_knowledge_base(query, n_results=3)

print(f"Query: '{query}'")
print("\nTop 3 Results:")
print("=" * 60)

for i in range(len(results['documents'][0])):
    print(f"\nResult {i+1}:")
    print(f"Source: {results['metadatas'][0][i]['source']}")
    print(f"Distance: {results['distances'][0][i]:.4f}")
    print(f"Content: {results['documents'][0][i][:200]}...")
    print("-" * 60)
```

### Step 5.2: Multiple Query Examples

```python
test_queries = [
    "How long does shipping take?",
    "What are customer support hours?",
    "How do warranties work?",
    "Can I get free shipping?"
]

for query in test_queries:
    print(f"\n🔍 Query: {query}")
    results = search_knowledge_base(query, n_results=2)

    for i in range(len(results['documents'][0])):
        print(f"  ✓ {results['metadatas'][0][i]['source']}")
        print(f"    {results['documents'][0][i][:80]}...")
```

---

## Part 6: Build Complete RAG Pipeline

### Step 6.1: RAG with OpenAI

```python
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

def rag_query(question, n_results=3):
    """
    Complete RAG pipeline:
    1. Search for relevant documents
    2. Format context
    3. Generate answer with LLM
    """

    # Step 1: Retrieve relevant chunks
    search_results = search_knowledge_base(question, n_results=n_results)

    # Step 2: Format context
    context_parts = []
    for i, doc in enumerate(search_results['documents'][0]):
        source = search_results['metadatas'][0][i]['source']
        context_parts.append(f"[Source: {source}]\n{doc}")

    context = "\n\n".join(context_parts)

    # Step 3: Create prompt
    prompt = f"""Answer the question based on the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    # Step 4: Generate response
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful customer service assistant. Answer questions based only on the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Lower temperature for more factual responses
        max_tokens=200
    )

    answer = response.choices[0].message.content

    return {
        'question': question,
        'answer': answer,
        'sources': [meta['source'] for meta in search_results['metadatas'][0]],
        'context_chunks': search_results['documents'][0]
    }

# Test the RAG system
question = "What is your return policy and how long do I have?"
result = rag_query(question)

print("=" * 70)
print(f"Question: {result['question']}")
print("=" * 70)
print(f"\nAnswer:\n{result['answer']}")
print(f"\nSources: {', '.join(result['sources'])}")
print("=" * 70)
```

### Step 6.2: Interactive RAG Demo

```python
def interactive_rag():
    """Interactive Q&A session"""
    print("🤖 RAG Assistant Ready!")
    print("Ask questions about our company policies. Type 'quit' to exit.\n")

    while True:
        question = input("You: ")

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        result = rag_query(question)
        print(f"\n🤖 Assistant: {result['answer']}")
        print(f"📚 Sources: {', '.join(result['sources'])}\n")

# Run interactive demo
# interactive_rag()  # Uncomment to run
```

---

## Part 7: Advanced RAG Techniques

### Step 7.1: Reranking Results

```python
def rerank_results(query, initial_results, top_k=3):
    """
    Rerank results using cross-encoder for better accuracy
    """
    !pip install sentence-transformers

    from sentence_transformers import CrossEncoder

    # Load cross-encoder model (better for reranking)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Create query-document pairs
    pairs = [[query, doc] for doc in initial_results['documents'][0]]

    # Score all pairs
    scores = cross_encoder.predict(pairs)

    # Sort by score
    scored_results = list(zip(
        initial_results['documents'][0],
        initial_results['metadatas'][0],
        scores
    ))
    scored_results.sort(key=lambda x: x[2], reverse=True)

    # Return top-k
    return scored_results[:top_k]

# Example usage
query = "How can I contact customer support?"
initial_results = search_knowledge_base(query, n_results=5)
reranked = rerank_results(query, initial_results, top_k=3)

print(f"Query: {query}\n")
for i, (doc, meta, score) in enumerate(reranked):
    print(f"{i+1}. Score: {score:.4f} | Source: {meta['source']}")
    print(f"   {doc[:100]}...\n")
```

### Step 7.2: Query Expansion

```python
def expand_query(original_query):
    """Generate multiple query variations for better retrieval"""

    expansion_prompt = f"""Generate 2 alternative phrasings of this question:
Original: {original_query}

Alternative 1:
Alternative 2:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": expansion_prompt}],
        temperature=0.7,
        max_tokens=100
    )

    variations = [original_query]  # Include original
    variations.extend(response.choices[0].message.content.strip().split('\n'))

    return variations

# Test query expansion
query = "How do returns work?"
expanded = expand_query(query)

print("Original query:", query)
print("\nExpanded queries:")
for i, var in enumerate(expanded):
    print(f"{i+1}. {var}")
```

---

## Part 8: Evaluation and Optimization

### Step 8.1: Create Test Questions

```python
test_cases = [
    {
        "question": "How long do I have to return a product?",
        "expected_answer_contains": ["30-day", "30 day"],
        "expected_source": "return_policy.txt"
    },
    {
        "question": "What are the customer service hours on Saturday?",
        "expected_answer_contains": ["10 AM", "6 PM"],
        "expected_source": "support_hours.txt"
    },
    {
        "question": "How much is overnight shipping?",
        "expected_answer_contains": ["24.99", "$24.99"],
        "expected_source": "shipping_info.txt"
    }
]
```

### Step 8.2: Evaluate RAG Performance

```python
def evaluate_rag(test_cases):
    """Evaluate RAG system on test cases"""
    results = []

    for test in test_cases:
        result = rag_query(test['question'])

        # Check if answer contains expected content
        answer_correct = any(
            exp.lower() in result['answer'].lower()
            for exp in test['expected_answer_contains']
        )

        # Check if correct source was retrieved
        source_correct = test['expected_source'] in result['sources']

        results.append({
            'question': test['question'],
            'answer': result['answer'],
            'sources': result['sources'],
            'answer_correct': answer_correct,
            'source_correct': source_correct
        })

    # Calculate metrics
    answer_accuracy = sum(r['answer_correct'] for r in results) / len(results)
    source_accuracy = sum(r['source_correct'] for r in results) / len(results)

    return results, answer_accuracy, source_accuracy

# Run evaluation
eval_results, answer_acc, source_acc = evaluate_rag(test_cases)

print("EVALUATION RESULTS")
print("=" * 60)
print(f"Answer Accuracy: {answer_acc:.1%}")
print(f"Source Accuracy: {source_acc:.1%}\n")

for i, result in enumerate(eval_results):
    print(f"Test {i+1}: {result['question']}")
    print(f"  ✓ Answer: {'✅' if result['answer_correct'] else '❌'}")
    print(f"  ✓ Source: {'✅' if result['source_correct'] else '❌'}")
    print()
```

---

## Exercises

### Exercise 1: Custom Documents
Replace the sample documents with your own content:
- Create 5-10 text files on a topic of your choice
- Load and chunk them
- Build a RAG system for your documents
- Test with relevant questions

### Exercise 2: Optimize Chunk Size
Experiment with different chunk sizes (200, 400, 800):
- Create multiple collections with different chunk sizes
- Compare retrieval quality
- Which size works best for your documents?

### Exercise 3: Multi-Query RAG
Implement a RAG system that:
- Takes a user question
- Expands it into 3 variations
- Retrieves results for all variations
- Combines and deduplicates results
- Generates final answer

### Exercise 4: Add Citations
Modify the RAG system to include specific citations:
- Format answer with numbered citations
- List source documents at the end
- Example: "Our return policy is 30 days [1]. Contact support at..."

---

## Key Takeaways

✅ **RAG grounds LLM responses in your data** - Prevents hallucinations

✅ **Chunking strategy matters** - Balance between context and precision

✅ **Vector search enables semantic retrieval** - Finds meaning, not just keywords

✅ **Quality of retrieval affects quality of generation** - Garbage in, garbage out

✅ **Evaluation is essential** - Test your RAG system systematically

✅ **Advanced techniques improve accuracy** - Reranking, query expansion, hybrid search

---

## Next Session Preview

In **Session 4: Function Calling & Tool Use**, you'll learn:
- How to give LLMs access to external tools
- Defining function schemas
- Implementing tool calling with OpenAI and Claude
- Building multi-tool agents
- Handling errors and edge cases

---

**Session 3 Complete!** 🎉

You now know how to build production-ready RAG systems!

**[Continue to Session 4: Function Calling →](04_Function_Calling.md)**
