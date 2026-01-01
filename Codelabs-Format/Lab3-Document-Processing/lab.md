# Lab 3: Document Processing & Embeddings

## 🛠️ Hands-On Lab

**Duration:** 60-90 minutes
**Difficulty:** Beginner to Intermediate
**Prerequisites:** Lab 1 & Lab 2 completed

---

## What You'll Build

By the end of this lab, you'll have:
- ✅ Document loader for TXT and PDF files
- ✅ Text chunking system with multiple strategies
- ✅ Embedding generation pipeline
- ✅ Vector database (ChromaDB) with stored documents
- ✅ Similarity comparison tools
- ✅ **Capstone**: Complete document processing system for SupportGenie

---

## 📋 Setup

### Step 1: Install Required Libraries

```bash
pip install sentence-transformers
pip install chromadb
pip install pypdf
pip install langchain
pip install langchain-community
pip install scikit-learn
pip install numpy
```

### Step 2: Create Sample Document

Create a file called `sample_document.txt`:

```text
Artificial Intelligence (AI) is transforming industries worldwide. Machine learning, a subset of AI, enables computers to learn from data without explicit programming.

Deep learning is a specialized form of machine learning that uses neural networks with multiple layers. These networks can process complex patterns in images, text, and audio.

Natural Language Processing (NLP) is another important branch of AI that focuses on the interaction between computers and human language. Applications include chatbots, translation services, and sentiment analysis.

Computer vision enables machines to interpret and understand visual information from the world. Self-driving cars and facial recognition systems rely heavily on computer vision technologies.

Reinforcement learning is a type of machine learning where agents learn to make decisions by interacting with an environment. It has been successfully applied to game playing, robotics, and resource management.

AI ethics is becoming increasingly important as AI systems are deployed in critical areas like healthcare, criminal justice, and employment. Issues include bias in algorithms, privacy concerns, and the need for transparency and accountability.

The future of AI holds tremendous potential for solving complex problems in climate change, disease diagnosis, and scientific research. However, it also raises important questions about job displacement and the role of humans in an AI-driven world.
```

### Step 3: Create Lab File

```bash
touch lab3_document_processing.py
```

✅ **Checkpoint**: Files created successfully.

---

## Exercise 1: Document Loading (10 min)

**Objective:** Load documents from different file formats.

### Task 1A: Load Text File

```python
# lab3_document_processing.py

def load_text_file(file_path):
    """
    Load a text file and return its content

    Args:
        file_path: Path to the text file

    Returns:
        str: File content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Test the function
document_path = 'sample_document.txt'
document_content = load_text_file(document_path)

if document_content:
    print("✓ Document loaded successfully!")
    print(f"  Length: {len(document_content)} characters")
    print(f"  Words: ~{len(document_content.split())} words")
    print(f"\nFirst 200 characters:")
    print(document_content[:200])
```

**Expected Output:**
```
✓ Document loaded successfully!
  Length: 1753 characters
  Words: ~247 words

First 200 characters:
Artificial Intelligence (AI) is transforming industries worldwide. Machine learning, a subset of AI, enables computers to learn from data without explicit programming...
```

### Task 1B: Load PDF File (Optional)

```python
from pypdf import PdfReader

def load_pdf_file(file_path):
    """
    Load a PDF file and extract text from all pages

    Args:
        file_path: Path to the PDF file

    Returns:
        str: Extracted text from all pages
    """
    try:
        reader = PdfReader(file_path)
        text = ""

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            text += page_text
            print(f"  Extracted page {page_num + 1}: {len(page_text)} characters")

        return text
    except FileNotFoundError:
        print(f"Error: PDF file {file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

# Test with your own PDF (optional)
# pdf_content = load_pdf_file('sample.pdf')
```

### Task 1C: Document Statistics

```python
def analyze_document(content):
    """Print useful statistics about a document"""

    lines = content.split('\n')
    words = content.split()
    paragraphs = [p for p in content.split('\n\n') if p.strip()]

    print("\n=== DOCUMENT STATISTICS ===")
    print(f"Characters: {len(content):,}")
    print(f"Words: {len(words):,}")
    print(f"Lines: {len(lines):,}")
    print(f"Paragraphs: {len(paragraphs):,}")
    print(f"Avg words per paragraph: {len(words) / len(paragraphs):.1f}")
    print(f"Avg characters per word: {len(content) / len(words):.1f}")

# Analyze the document
analyze_document(document_content)
```

✅ **Checkpoint**: You should see detailed document statistics.

---

## Exercise 2: Text Chunking Strategies (20 min)

**Objective:** Implement and compare different chunking strategies.

### Task 2A: Fixed-Size Character Chunking

```python
def chunk_by_characters(text, chunk_size=200, overlap=50):
    """
    Split text into fixed-size chunks with overlap

    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        list: Text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # Move start forward by (chunk_size - overlap)
        start += (chunk_size - overlap)

    return chunks

# Test character chunking
char_chunks = chunk_by_characters(
    document_content,
    chunk_size=300,
    overlap=50
)

print("\n=== CHARACTER-BASED CHUNKING ===")
print(f"Created {len(char_chunks)} chunks\n")

for i, chunk in enumerate(char_chunks[:3]):  # Show first 3
    print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk)
    print()
```

**Observe:** Some chunks might be split mid-sentence or mid-word.

### Task 2B: Sentence-Based Chunking (Better)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_smartly(text, chunk_size=500, chunk_overlap=100):
    """
    Split text using LangChain's intelligent splitter
    Tries to split on paragraphs, then sentences, then words

    Args:
        text: Text to chunk
        chunk_size: Target size for each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        list: Text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    return chunks

# Test smart chunking
smart_chunks = chunk_smartly(
    document_content,
    chunk_size=300,
    chunk_overlap=50
)

print("\n=== SMART (RECURSIVE) CHUNKING ===")
print(f"Created {len(smart_chunks)} chunks\n")

for i, chunk in enumerate(smart_chunks):
    print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk)
    print()
```

**Observe:** Chunks respect sentence boundaries and maintain coherence.

### Task 2C: Compare Chunking Strategies

```python
def compare_chunking_strategies(text):
    """Compare different chunking approaches"""

    # Strategy 1: Fixed character
    char_chunks = chunk_by_characters(text, chunk_size=300, overlap=50)

    # Strategy 2: Smart recursive
    smart_chunks = chunk_smartly(text, chunk_size=300, chunk_overlap=50)

    print("\n=== CHUNKING STRATEGY COMPARISON ===")
    print(f"\nFixed Character Chunking:")
    print(f"  Chunks created: {len(char_chunks)}")
    print(f"  Avg chunk size: {sum(len(c) for c in char_chunks) / len(char_chunks):.1f}")
    print(f"  Sample split: '{char_chunks[0][-30:]}'")
    print(f"                '{char_chunks[1][:30]}'")

    print(f"\nSmart Recursive Chunking:")
    print(f"  Chunks created: {len(smart_chunks)}")
    print(f"  Avg chunk size: {sum(len(c) for c in smart_chunks) / len(smart_chunks):.1f}")
    print(f"  Sample split: '{smart_chunks[0][-30:]}'")
    print(f"                '{smart_chunks[1][:30]}'")

    # Check if splits respect sentences
    char_breaks_sentence = not char_chunks[0].endswith(('.', '!', '?'))
    smart_breaks_sentence = not smart_chunks[0].endswith(('.', '!', '?'))

    print(f"\nBreaks mid-sentence?")
    print(f"  Fixed: {char_breaks_sentence}")
    print(f"  Smart: {smart_breaks_sentence}")

compare_chunking_strategies(document_content)
```

**Question:** Which strategy produces better chunks? Why?

✅ **Checkpoint**: You should see that smart chunking respects sentence boundaries.

---

## Exercise 3: Generate Embeddings (20 min)

**Objective:** Convert text chunks into embeddings using HuggingFace models.

### Task 3A: Load Embedding Model

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the embedding model (this will download ~80MB on first run)
print("Loading embedding model...")
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embedding_model = SentenceTransformer(model_name)

print(f"✓ Loaded: {model_name}")
print(f"  Embedding dimension: {embedding_model.get_sentence_embedding_dimension()}")
```

**Expected Output:**
```
✓ Loaded: sentence-transformers/all-MiniLM-L6-v2
  Embedding dimension: 384
```

### Task 3B: Generate Embeddings for Chunks

```python
def generate_embeddings(chunks, model):
    """
    Generate embeddings for a list of text chunks

    Args:
        chunks: List of text strings
        model: SentenceTransformer model

    Returns:
        numpy array: Embeddings (num_chunks × embedding_dim)
    """
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"✓ Generated {len(embeddings)} embeddings")
    return embeddings

# Generate embeddings for our smart chunks
embeddings = generate_embeddings(smart_chunks, embedding_model)

print(f"\nEmbedding shape: {embeddings.shape}")
print(f"  {embeddings.shape[0]} chunks")
print(f"  {embeddings.shape[1]} dimensions each")
print(f"\nFirst embedding (first 10 values):")
print(embeddings[0][:10])
```

### Task 3C: Visualize Embedding Statistics

```python
def visualize_embedding_stats(embeddings):
    """Print statistics about embeddings"""

    embeddings_array = np.array(embeddings)

    print("\n=== EMBEDDING STATISTICS ===")
    print(f"Shape: {embeddings_array.shape}")
    print(f"  {embeddings_array.shape[0]} vectors")
    print(f"  {embeddings_array.shape[1]} dimensions")
    print(f"\nValue ranges:")
    print(f"  Min: {embeddings_array.min():.4f}")
    print(f"  Max: {embeddings_array.max():.4f}")
    print(f"  Mean: {embeddings_array.mean():.4f}")
    print(f"  Std Dev: {embeddings_array.std():.4f}")

    # Check if normalized (unit length)
    norms = np.linalg.norm(embeddings_array, axis=1)
    print(f"\nVector norms (should be ~1.0 if normalized):")
    print(f"  Min norm: {norms.min():.4f}")
    print(f"  Max norm: {norms.max():.4f}")
    print(f"  Mean norm: {norms.mean():.4f}")

visualize_embedding_stats(embeddings)
```

### Task 3D: Visualize Individual Embeddings

```python
import matplotlib.pyplot as plt

def plot_embedding_distribution(embedding, chunk_text, max_chars=50):
    """Visualize a single embedding as a histogram"""

    plt.figure(figsize=(12, 4))

    # Plot histogram
    plt.subplot(1, 2, 1)
    plt.hist(embedding, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Embedding Value Distribution\n"{chunk_text[:max_chars]}..."')
    plt.grid(True, alpha=0.3)

    # Plot values
    plt.subplot(1, 2, 2)
    plt.plot(embedding, linewidth=0.5)
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.title(f'Embedding Values (384 dimensions)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('embedding_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'embedding_visualization.png'")
    # plt.show()  # Uncomment to display

# Visualize first embedding
plot_embedding_distribution(embeddings[0], smart_chunks[0])
```

✅ **Checkpoint**: You should have embeddings with 384 dimensions, values roughly between -1 and 1.

---

## Exercise 4: Semantic Similarity (15 min)

**Objective:** Understand how embeddings capture semantic meaning.

### Task 4A: Compare Chunk Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

def compare_chunks(embeddings, chunks, idx1, idx2):
    """
    Calculate and display similarity between two chunks

    Args:
        embeddings: Array of embeddings
        chunks: List of text chunks
        idx1, idx2: Indices of chunks to compare

    Returns:
        float: Cosine similarity score
    """
    similarity = cosine_similarity(
        [embeddings[idx1]],
        [embeddings[idx2]]
    )[0][0]

    print(f"\n=== COMPARING CHUNKS {idx1} and {idx2} ===")
    print(f"\nChunk {idx1} ({len(chunks[idx1])} chars):")
    print(f'"{chunks[idx1]}"')
    print(f"\nChunk {idx2} ({len(chunks[idx2])} chars):")
    print(f'"{chunks[idx2]}"')
    print(f"\n{'='*50}")
    print(f"Cosine Similarity: {similarity:.4f}")

    # Interpret the score
    if similarity > 0.9:
        interpretation = "Extremely similar (nearly identical)"
    elif similarity > 0.8:
        interpretation = "Very similar"
    elif similarity > 0.7:
        interpretation = "Similar"
    elif similarity > 0.5:
        interpretation = "Somewhat related"
    else:
        interpretation = "Not very related"

    print(f"Interpretation: {interpretation}")
    print('='*50)

    return similarity

# Compare adjacent chunks (should be similar due to overlap)
similarity_adjacent = compare_chunks(embeddings, smart_chunks, 0, 1)

# Compare distant chunks (should be less similar)
similarity_distant = compare_chunks(
    embeddings,
    smart_chunks,
    0,
    len(smart_chunks) - 1
)
```

### Task 4B: Create Similarity Matrix

```python
def create_similarity_matrix(embeddings):
    """
    Create a matrix showing similarity between all chunks

    Returns:
        numpy array: Similarity matrix
    """
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

def visualize_similarity_matrix(similarity_matrix):
    """Visualize the similarity matrix as a heatmap"""

    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Cosine Similarity')
    plt.xlabel('Chunk Index')
    plt.ylabel('Chunk Index')
    plt.title('Chunk Similarity Matrix\n(Lighter = More Similar)')

    # Add value annotations for small matrices
    if len(similarity_matrix) <= 10:
        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix)):
                plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center',
                        color='black' if similarity_matrix[i, j] < 0.7 else 'white',
                        fontsize=8)

    plt.tight_layout()
    plt.savefig('similarity_matrix.png', dpi=150, bbox_inches='tight')
    print("✓ Saved similarity matrix to 'similarity_matrix.png'")
    # plt.show()  # Uncomment to display

# Create and visualize similarity matrix
sim_matrix = create_similarity_matrix(embeddings)
print(f"\nSimilarity Matrix Shape: {sim_matrix.shape}")
visualize_similarity_matrix(sim_matrix)

# Find most similar chunk pairs
print("\n=== MOST SIMILAR CHUNK PAIRS ===")
for i in range(len(sim_matrix)):
    for j in range(i + 1, len(sim_matrix)):
        if sim_matrix[i, j] > 0.8:  # High similarity threshold
            print(f"Chunks {i} and {j}: {sim_matrix[i, j]:.4f}")
```

### Task 4C: Test Semantic Search (Manual)

```python
def find_similar_chunks(query, chunks, embeddings, model, top_k=3):
    """
    Find chunks most similar to a query (manual semantic search)

    Args:
        query: Search query text
        chunks: List of text chunks
        embeddings: Pre-computed chunk embeddings
        model: SentenceTransformer model
        top_k: Number of results to return

    Returns:
        list: Top-k most similar chunks with scores
    """
    # Generate query embedding
    query_embedding = model.encode([query])

    # Calculate similarity with all chunks
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'chunk_index': idx,
            'similarity': similarities[idx],
            'text': chunks[idx]
        })

    return results

# Test semantic search
test_queries = [
    "What is deep learning?",
    "Tell me about computer vision",
    "What are AI ethics concerns?"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print('='*60)

    results = find_similar_chunks(
        query,
        smart_chunks,
        embeddings,
        embedding_model,
        top_k=2
    )

    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Similarity: {result['similarity']:.4f}):")
        print(f"Chunk {result['chunk_index']}: {result['text']}")
```

**Observe:** The most similar chunks should actually relate to the query, even if they don't contain the exact words!

✅ **Checkpoint**: Semantic search should retrieve relevant chunks based on meaning, not just keywords.

---

## Exercise 5: Vector Database with ChromaDB (20 min)

**Objective:** Store embeddings in a persistent vector database.

### Task 5A: Initialize ChromaDB

```python
import chromadb
from chromadb.config import Settings

def initialize_chromadb(persist_directory="./chroma_db"):
    """
    Initialize ChromaDB with persistent storage

    Args:
        persist_directory: Where to save the database

    Returns:
        tuple: (client, collection)
    """
    # Create persistent client
    client = chromadb.PersistentClient(path=persist_directory)

    # Delete existing collection if it exists (for fresh start)
    try:
        client.delete_collection("lab3_documents")
        print("Deleted existing collection")
    except:
        pass

    # Create new collection
    collection = client.create_collection(
        name="lab3_documents",
        metadata={"description": "Lab 3 - Document Processing & Embeddings"}
    )

    print(f"✓ Initialized ChromaDB at '{persist_directory}'")
    print(f"✓ Created collection: 'lab3_documents'")

    return client, collection

# Initialize the database
chroma_client, collection = initialize_chromadb()
```

### Task 5B: Add Documents to ChromaDB

```python
def add_documents_to_chroma(collection, chunks, embeddings, source_file="sample_document.txt"):
    """
    Add chunks and embeddings to ChromaDB collection

    Args:
        collection: ChromaDB collection
        chunks: List of text chunks
        embeddings: Numpy array of embeddings
        source_file: Source filename for metadata
    """
    # Convert numpy embeddings to list
    embeddings_list = embeddings.tolist()

    # Create unique IDs
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Create metadata for each chunk
    metadatas = [
        {
            "chunk_index": i,
            "source": source_file,
            "chunk_size": len(chunk),
            "word_count": len(chunk.split())
        }
        for i, chunk in enumerate(chunks)
    ]

    # Add to collection
    collection.add(
        documents=chunks,
        embeddings=embeddings_list,
        ids=ids,
        metadatas=metadatas
    )

    print(f"\n✓ Added {len(chunks)} documents to ChromaDB")
    print(f"  Collection now contains: {collection.count()} items")

# Add our documents
add_documents_to_chroma(collection, smart_chunks, embeddings)
```

### Task 5C: Verify Storage and Query

```python
def verify_chroma_storage(collection, chunk_id="chunk_0"):
    """Verify that data was stored correctly"""

    # Get a specific chunk
    result = collection.get(
        ids=[chunk_id],
        include=["documents", "metadatas", "embeddings"]
    )

    print(f"\n=== VERIFYING STORAGE (chunk_id: {chunk_id}) ===")
    print(f"\nDocument text:")
    print(f'"{result["documents"][0]}"')
    print(f"\nMetadata:")
    for key, value in result["metadatas"][0].items():
        print(f"  {key}: {value}")
    print(f"\nEmbedding (first 10 values):")
    print(result["embeddings"][0][:10])
    print(f"  ... ({len(result['embeddings'][0])} values total)")

verify_chroma_storage(collection, "chunk_0")

# Query the database
def query_chroma(collection, query_text, n_results=3):
    """
    Query ChromaDB using natural language

    Args:
        collection: ChromaDB collection
        query_text: Search query
        n_results: Number of results to return

    Returns:
        dict: Query results
    """
    # Generate embedding for query
    query_embedding = embedding_model.encode([query_text])

    # Query the collection
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    return results

# Test querying
test_query = "What is machine learning?"
print(f"\n{'='*60}")
print(f"QUERY: {test_query}")
print('='*60)

results = query_chroma(collection, test_query, n_results=3)

for i in range(len(results['documents'][0])):
    distance = results['distances'][0][i]
    similarity = 1 - distance  # ChromaDB uses distance, convert to similarity

    print(f"\nResult {i+1} (Similarity: {similarity:.4f}):")
    print(f"Metadata: {results['metadatas'][0][i]}")
    print(f"Text: {results['documents'][0][i]}")
```

### Task 5D: Persistence Check

```python
def test_persistence():
    """Test that ChromaDB persists data across sessions"""

    print("\n=== TESTING PERSISTENCE ===")

    # Create a new client (simulates restarting the program)
    new_client = chromadb.PersistentClient(path="./chroma_db")

    # Get existing collection
    reloaded_collection = new_client.get_collection("lab3_documents")

    print(f"✓ Reloaded collection: {reloaded_collection.name}")
    print(f"  Count: {reloaded_collection.count()} documents")

    # Verify data is still there
    result = reloaded_collection.get(ids=["chunk_0"], include=["documents"])
    print(f"  First chunk still accessible: {result['documents'][0][:50]}...")

    return reloaded_collection

persisted_collection = test_persistence()
```

✅ **Checkpoint**: ChromaDB should persist data to disk. You can close and reopen the collection without losing data.

---

## 🎯 Capstone Project: Complete Document Processing Pipeline (30 min)

**Objective:** Build a complete, reusable document processing system.

### Requirements

Your system must:
1. ✅ Load documents from files
2. ✅ Chunk text intelligently
3. ✅ Generate embeddings
4. ✅ Store in ChromaDB
5. ✅ Support querying
6. ✅ Handle errors gracefully
7. ✅ Be reusable for different documents

### Starter Code

```python
# document_processor.py

import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from pypdf import PdfReader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DocumentProcessor:
    """
    Complete document processing pipeline for RAG systems
    """

    def __init__(
        self,
        embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
        chroma_db_path='./chroma_db',
        collection_name='documents'
    ):
        """
        Initialize the document processor

        Args:
            embedding_model_name: HuggingFace model name
            chroma_db_path: Path to ChromaDB storage
            collection_name: Name of ChromaDB collection
        """
        print("Initializing Document Processor...")

        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print(f"✓ Loaded embedding model: {embedding_model_name}")

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            print(f"✓ Loaded existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(collection_name)
            print(f"✓ Created new collection: {collection_name}")

        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        print(f"✓ Document Processor ready!")
        print(f"  Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        print(f"  Current collection size: {self.collection.count()}")

    def load_text_file(self, file_path):
        """Load a text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_pdf_file(self, file_path):
        """Load a PDF file"""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def load_document(self, file_path):
        """
        Load a document (auto-detects file type)

        Args:
            file_path: Path to document

        Returns:
            str: Document content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            content = self.load_pdf_file(file_path)
        elif ext in ['.txt', '.md']:
            content = self.load_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        print(f"✓ Loaded {file_path} ({len(content)} characters)")
        return content

    def chunk_document(self, content):
        """
        Split document into chunks

        Args:
            content: Document text

        Returns:
            list: Text chunks
        """
        chunks = self.text_splitter.split_text(content)
        print(f"✓ Created {len(chunks)} chunks")
        return chunks

    def generate_embeddings(self, chunks):
        """
        Generate embeddings for chunks

        Args:
            chunks: List of text chunks

        Returns:
            numpy array: Embeddings
        """
        embeddings = self.embedding_model.encode(
            chunks,
            show_progress_bar=True
        )
        print(f"✓ Generated {len(embeddings)} embeddings")
        return embeddings

    def add_to_database(self, chunks, embeddings, source_file):
        """
        Add chunks to ChromaDB

        Args:
            chunks: List of text chunks
            embeddings: Numpy array of embeddings
            source_file: Source filename
        """
        # Get current count for ID generation
        current_count = self.collection.count()

        # Create IDs
        ids = [f"doc_{current_count + i}" for i in range(len(chunks))]

        # Create metadata
        metadatas = [
            {
                "source": source_file,
                "chunk_index": i,
                "chunk_size": len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]

        # Add to collection
        self.collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=metadatas
        )

        print(f"✓ Added {len(chunks)} chunks to database")
        print(f"  Total documents in collection: {self.collection.count()}")

    def process_document(self, file_path):
        """
        Complete pipeline: load → chunk → embed → store

        Args:
            file_path: Path to document

        Returns:
            dict: Processing statistics
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING: {file_path}")
        print('='*60)

        # Load
        content = self.load_document(file_path)

        # Chunk
        chunks = self.chunk_document(content)

        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)

        # Store
        self.add_to_database(chunks, embeddings, file_path)

        stats = {
            'file': file_path,
            'characters': len(content),
            'chunks': len(chunks),
            'embeddings_shape': embeddings.shape
        }

        print(f"\n✓ PROCESSING COMPLETE!")
        print(f"  {stats['chunks']} chunks stored in database")

        return stats

    def search(self, query, n_results=5):
        """
        Search for relevant chunks

        Args:
            query: Search query
            n_results: Number of results

        Returns:
            dict: Search results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])

        # Query database
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        return results

    def search_and_display(self, query, n_results=3):
        """Search and display results in readable format"""

        print(f"\n{'='*60}")
        print(f"SEARCH: {query}")
        print('='*60)

        results = self.search(query, n_results)

        for i in range(len(results['documents'][0])):
            distance = results['distances'][0][i]
            similarity = 1 - distance

            print(f"\nResult {i+1} (Similarity: {similarity:.4f})")
            print(f"Source: {results['metadatas'][0][i]['source']}")
            print(f"Chunk: {results['metadatas'][0][i]['chunk_index']}")
            print(f"Text: {results['documents'][0][i]}")
            print("-" * 60)

    def get_stats(self):
        """Get database statistics"""
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.collection.name,
            'embedding_dimension': self.embedding_model.get_sentence_embedding_dimension()
        }


# ============================================
# USAGE EXAMPLE
# ============================================

def main():
    """Test the complete document processor"""

    # Initialize processor
    processor = DocumentProcessor(
        collection_name='lab3_complete',
        chroma_db_path='./chroma_db_complete'
    )

    # Process sample document
    stats = processor.process_document('sample_document.txt')

    # Test searches
    queries = [
        "What is machine learning?",
        "Tell me about AI ethics",
        "How does computer vision work?"
    ]

    for query in queries:
        processor.search_and_display(query, n_results=2)

    # Show stats
    print(f"\n{'='*60}")
    print("DATABASE STATISTICS")
    print('='*60)
    stats = processor.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
```

### Your Tasks

1. **Run the complete pipeline** and verify it works
2. **Add error handling** for edge cases
3. **Process multiple documents** (create another text file)
4. **Add a method** to delete documents by source file
5. **Add a method** to get all unique sources in the database

### Bonus Challenges

```python
# TODO 1: Add support for DOCX files
def load_docx_file(self, file_path):
    """Load a DOCX file"""
    # Hint: pip install python-docx
    pass

# TODO 2: Add duplicate detection
def is_duplicate(self, chunk):
    """Check if chunk already exists in database"""
    pass

# TODO 3: Add batch processing
def process_directory(self, directory_path):
    """Process all documents in a directory"""
    pass

# TODO 4: Add re-ranking
def rerank_results(self, query, results):
    """Re-rank results using a different model"""
    pass
```

✅ **Checkpoint**: Your document processor should successfully load, chunk, embed, and store documents, then allow searching.

---

## 🏆 Extension Challenges

### Challenge 1: Compare Embedding Models

Test different models and compare results:

```python
models_to_compare = [
    'sentence-transformers/all-MiniLM-L6-v2',      # 384D
    'sentence-transformers/all-mpnet-base-v2',     # 768D
    'sentence-transformers/paraphrase-MiniLM-L3-v2' # 384D
]

# TODO: Compare speed, quality, and embedding dimensions
```

### Challenge 2: Optimal Chunk Size Finder

Find the best chunk size for your documents:

```python
def find_optimal_chunk_size(document, sizes=[200, 500, 1000]):
    """
    Test different chunk sizes and compare results

    Metrics to consider:
    - Number of chunks created
    - Average chunk coherence
    - Query performance
    """
    pass
```

### Challenge 3: Multi-Document Search

Build a system that can search across multiple documents:

```python
# Process multiple documents
documents = [
    'tech_manual.pdf',
    'user_guide.txt',
    'faq.md'
]

# Search should return results from all documents
# with source attribution
```

### Challenge 4: Semantic Clustering

Group similar chunks together:

```python
from sklearn.cluster import KMeans

def cluster_chunks(embeddings, n_clusters=5):
    """
    Cluster chunks by semantic similarity

    Returns:
        List of cluster assignments
    """
    pass
```

---

## 📝 Key Takeaways

After completing this lab, you should understand:

✅ **Document Loading** - How to load TXT, PDF, and other formats
✅ **Text Chunking** - Why and how to split documents intelligently
✅ **Embeddings** - Converting text to numerical vectors that capture meaning
✅ **Vector Databases** - Storing and querying embeddings efficiently
✅ **Semantic Search** - Finding relevant content by meaning, not keywords
✅ **Complete Pipeline** - All steps from document to searchable database

---

## 🔍 Troubleshooting

**Issue**: "Model download is slow"

**Solution**: First download is ~80MB and may take time. It's cached for future use.

---

**Issue**: "ChromaDB not persisting"

**Solution**: Ensure you're using `PersistentClient`, not `Client`:
```python
# ✓ Correct
client = chromadb.PersistentClient(path="./chroma_db")

# ✗ Wrong (in-memory only)
client = chromadb.Client()
```

---

**Issue**: "Embeddings dimension mismatch"

**Solution**: Ensure same model for embedding and querying:
```python
# ✓ Same model everywhere
model = SentenceTransformer('all-MiniLM-L6-v2')
```

---

**Issue**: "Out of memory with large documents"

**Solution**: Process in batches:
```python
# Embed in batches of 100
for i in range(0, len(chunks), 100):
    batch = chunks[i:i+100]
    embeddings = model.encode(batch)
```

---

## 🎓 What's Next?

You've completed Lab 3! You now have:
- ✅ Working document processing pipeline
- ✅ Vector database with semantic search
- ✅ Foundation for building RAG systems

**Next Lab**: Lab 4 - Semantic Search & Retrieval
Learn advanced querying techniques and build a production-ready search system!

---

## 📚 Additional Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Understanding Embeddings](https://jalammar.github.io/illustrated-word2vec/)

---

**Lab 3 Complete!** 🎉
[← Back to Learning Material](learning.md) | [Next: Lab 4 →](../Lab4-Semantic-Search/learning.md)
