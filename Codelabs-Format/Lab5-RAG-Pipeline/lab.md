# Lab 5: Complete RAG Pipeline

## 🛠️ Hands-On Lab

**Duration:** 75-90 minutes
**Difficulty:** Intermediate
**Prerequisites:** Labs 1-4 completed, API keys for OpenAI/Claude

---

## What You'll Build

By the end of this lab, you'll have:
- ✅ Complete RAG pipeline (Retrieve → Augment → Generate)
- ✅ Integration with multiple LLMs (OpenAI, Claude, Bedrock)
- ✅ RAG vs. non-RAG comparison tools
- ✅ Multiple prompt strategies
- ✅ RAG evaluation framework
- ✅ **Capstone**: Production SupportGenie v3.0 with full RAG

---

## 📋 Setup

### Step 1: Install Required Libraries

```bash
pip install openai
pip install anthropic
pip install boto3  # For AWS Bedrock (optional)
pip install python-dotenv
```

### Step 2: Create .env File

Create a `.env` file in your project directory:

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Optional: AWS Bedrock
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_REGION=us-east-1
```

### Step 3: Verify Setup

```python
# lab5_rag_pipeline.py

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify API keys
if os.getenv('OPENAI_API_KEY'):
    print("✓ OpenAI API key loaded")
else:
    print("✗ OpenAI API key not found")

if os.getenv('ANTHROPIC_API_KEY'):
    print("✓ Anthropic API key loaded")
else:
    print("✗ Anthropic API key not found (optional)")

print("\nNote: You need at least one API key to complete this lab")
```

### Step 4: Import Libraries

```python
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
from anthropic import Anthropic
import numpy as np
import time
```

✅ **Checkpoint**: API keys loaded successfully.

---

## Exercise 1: Basic RAG Pipeline (20 min)

**Objective:** Build your first complete RAG pipeline.

### Task 1A: Initialize Components

```python
# Initialize embedding model and vector database (from Lab 3)
print("Initializing components...")

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")

# Use collection from Lab 3
try:
    collection = client.get_collection(name="lab3_documents")
    print(f"✓ Loaded collection: {collection.count()} documents")
except:
    print("✗ Collection not found. Please complete Lab 3 first.")
    exit()

# Initialize LLM client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
print("✓ OpenAI client initialized")
```

### Task 1B: Build Retrieval Function

```python
def retrieve_context(query, n_results=3):
    """
    Retrieve relevant chunks for a query

    Args:
        query: User's question
        n_results: Number of chunks to retrieve

    Returns:
        tuple: (documents, distances)
    """
    # Embed the query
    query_embedding = embedding_model.encode([query])

    # Search vector database
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["documents", "distances"]
    )

    return results['documents'][0], results['distances'][0]

# Test retrieval
query = "What is deep learning?"
docs, distances = retrieve_context(query, n_results=3)

print(f"\nQuery: {query}")
print(f"\nRetrieved {len(docs)} chunks:")
for i, (doc, dist) in enumerate(zip(docs, distances)):
    print(f"\n[{i+1}] Distance: {dist:.4f}")
    print(f"    {doc[:100]}...")
```

### Task 1C: Create RAG Prompt

```python
def create_rag_prompt(query, context_chunks):
    """
    Create a prompt with retrieved context

    Args:
        query: User's question
        context_chunks: List of retrieved text chunks

    Returns:
        str: Complete RAG prompt
    """
    # Format context sections
    context_text = ""
    for i, chunk in enumerate(context_chunks):
        context_text += f"[{i+1}] {chunk}\n\n"

    # Create structured prompt
    prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

CONTEXT INFORMATION:
{context_text}

USER QUESTION:
{query}

INSTRUCTIONS:
- Answer using ONLY the information from the context above
- Cite which context section(s) you used (e.g., [1], [2])
- If the context doesn't contain enough information, say so clearly
- Be concise and accurate

ANSWER:"""

    return prompt

# Test prompt creation
rag_prompt = create_rag_prompt(query, docs)
print("\n" + "="*70)
print("RAG PROMPT:")
print("="*70)
print(rag_prompt[:500] + "...")
```

### Task 1D: Complete RAG Pipeline

```python
def rag_pipeline_openai(query, n_results=3, model="gpt-4o-mini"):
    """
    Complete RAG pipeline using OpenAI

    Args:
        query: User's question
        n_results: Number of context chunks to retrieve
        model: OpenAI model to use

    Returns:
        tuple: (answer, context_chunks, metadata)
    """
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"RAG PIPELINE: {query}")
    print('='*70)

    # Step 1: Retrieve
    print("\n[Step 1/3] Retrieving relevant context...")
    context_chunks, distances = retrieve_context(query, n_results)

    print(f"Retrieved {len(context_chunks)} chunks:")
    for i, dist in enumerate(distances):
        print(f"  [{i+1}] Distance: {dist:.4f}")

    # Step 2: Augment (create prompt)
    print("\n[Step 2/3] Creating RAG prompt...")
    prompt = create_rag_prompt(query, context_chunks)

    # Step 3: Generate
    print(f"\n[Step 3/3] Generating answer with {model}...")
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Low temperature for factual responses
        max_tokens=500
    )

    answer = response.choices[0].message.content
    elapsed_time = time.time() - start_time

    # Metadata
    metadata = {
        'model': model,
        'n_retrieved': len(context_chunks),
        'avg_distance': np.mean(distances),
        'time_seconds': elapsed_time,
        'tokens_used': response.usage.total_tokens
    }

    print(f"\n{'='*70}")
    print("RAG ANSWER:")
    print('='*70)
    print(answer)
    print(f"\n{'='*70}")
    print(f"Metadata: {metadata}")
    print('='*70)

    return answer, context_chunks, metadata

# Test the complete pipeline
answer, context, metadata = rag_pipeline_openai("What is machine learning?")
```

✅ **Checkpoint**: You should see a complete RAG response with retrieved context and generated answer.

---

## Exercise 2: RAG vs. Non-RAG Comparison (15 min)

**Objective:** Understand the difference RAG makes.

### Task 2A: Non-RAG Baseline

```python
def generate_without_rag(query, model="gpt-4o-mini"):
    """
    Generate answer WITHOUT retrieval (baseline)

    Args:
        query: User's question
        model: OpenAI model

    Returns:
        tuple: (answer, metadata)
    """
    start_time = time.time()

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": query}
        ],
        temperature=0.3,
        max_tokens=500
    )

    answer = response.choices[0].message.content
    elapsed_time = time.time() - start_time

    metadata = {
        'model': model,
        'time_seconds': elapsed_time,
        'tokens_used': response.usage.total_tokens
    }

    return answer, metadata

# Test non-RAG
no_rag_answer, no_rag_meta = generate_without_rag("What is machine learning?")
print("\n" + "="*70)
print("NON-RAG ANSWER (No Context):")
print("="*70)
print(no_rag_answer)
```

### Task 2B: Side-by-Side Comparison

```python
def compare_rag_vs_no_rag(query):
    """
    Compare RAG and non-RAG responses side-by-side

    Args:
        query: User's question
    """
    print(f"\n{'#'*70}")
    print(f"COMPARISON: RAG vs NON-RAG")
    print(f"{'#'*70}")
    print(f"Query: {query}\n")

    # Non-RAG
    print("="*70)
    print("WITHOUT RAG (LLM Only - No Retrieved Context)")
    print("="*70)
    no_rag_answer, no_rag_meta = generate_without_rag(query)
    print(no_rag_answer)
    print(f"\nTokens used: {no_rag_meta['tokens_used']}")

    # RAG
    print("\n" + "="*70)
    print("WITH RAG (LLM + Retrieved Context)")
    print("="*70)
    rag_answer, context, rag_meta = rag_pipeline_openai(query)

    # Analysis
    print(f"\n{'#'*70}")
    print("ANALYSIS:")
    print(f"{'#'*70}")
    print(f"Non-RAG tokens: {no_rag_meta['tokens_used']}")
    print(f"RAG tokens: {rag_meta['tokens_used']}")
    print(f"Token difference: {rag_meta['tokens_used'] - no_rag_meta['tokens_used']}")
    print(f"\nRAG retrieved {rag_meta['n_retrieved']} chunks")
    print(f"Average chunk relevance: {rag_meta['avg_distance']:.4f}")

    print("\nKey Differences to Observe:")
    print("  - Is RAG answer more specific to our documents?")
    print("  - Does RAG answer cite sources?")
    print("  - Is non-RAG answer more general?")
    print(f"{'#'*70}\n")

# Test with different queries
test_queries = [
    "What is machine learning?",
    "Tell me about deep learning",
    "What is computer vision?"
]

for query in test_queries:
    compare_rag_vs_no_rag(query)
    print("\n" + "="*70 + "\n")
```

### Task 2C: When RAG Helps Most

```python
def demonstrate_rag_value(specific_query, general_query):
    """
    Show when RAG provides most value

    Args:
        specific_query: Question about specific content in your documents
        general_query: General knowledge question
    """
    print(f"\n{'='*70}")
    print("WHEN DOES RAG HELP MOST?")
    print('='*70)

    print("\n--- Test 1: Specific Query (RAG should excel) ---")
    print(f"Query: {specific_query}")
    rag_answer1, _, _ = rag_pipeline_openai(specific_query, n_results=2)

    print("\n--- Test 2: General Query (RAG may not add much) ---")
    print(f"Query: {general_query}")
    rag_answer2, _, _ = rag_pipeline_openai(general_query, n_results=2)

    print("\n" + "="*70)
    print("INSIGHT:")
    print("RAG adds most value when:")
    print("  ✓ Question is about specific document content")
    print("  ✓ Information is not in LLM's training data")
    print("  ✓ Need to cite sources")
    print("  ✓ Domain-specific or recent information")
    print("="*70)

# Test
demonstrate_rag_value(
    specific_query="What specific applications of computer vision are mentioned?",
    general_query="What is Python programming?"
)
```

✅ **Checkpoint**: You should see clear differences between RAG and non-RAG responses.

---

## Exercise 3: Advanced Prompt Strategies (15 min)

**Objective:** Experiment with different RAG prompt formats.

### Task 3A: Multiple Prompt Templates

```python
class RAGPromptTemplates:
    """Collection of different RAG prompt strategies"""

    @staticmethod
    def basic_template(query, context_chunks):
        """Simple, straightforward prompt"""
        context = "\n\n".join([f"[{i+1}] {chunk}"
                               for i, chunk in enumerate(context_chunks)])

        return f"""Answer the question using the context below.

Context:
{context}

Question: {query}

Answer:"""

    @staticmethod
    def detailed_template(query, context_chunks):
        """Detailed with explicit instructions"""
        context = "\n\n".join([f"[{i+1}] {chunk}"
                               for i, chunk in enumerate(context_chunks)])

        return f"""You are a helpful AI assistant. Answer the user's question based ONLY on the provided context.

CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
1. Use ONLY information from the context above
2. Cite sources using [1], [2], [3] format
3. If context is insufficient, state what's missing
4. Be concise and accurate
5. Do not use your general knowledge

ANSWER:"""

    @staticmethod
    def structured_template(query, context_chunks):
        """Structured output format"""
        context = "\n\n".join([f"[{i+1}] {chunk}"
                               for i, chunk in enumerate(context_chunks)])

        return f"""Answer the question using the context provided. Structure your response.

CONTEXT:
{context}

QUESTION:
{query}

Provide your answer in this format:
**Summary:** [One sentence answer]
**Details:** [Detailed explanation with citations]
**Sources:** [List which context sections were used]

ANSWER:"""

    @staticmethod
    def conversational_template(query, context_chunks):
        """Friendly, conversational tone"""
        context = "\n\n".join([f"[{i+1}] {chunk}"
                               for i, chunk in enumerate(context_chunks)])

        return f"""You're a friendly AI assistant helping users understand information from documents.

Here's what I found in our documents:
{context}

The user asked: "{query}"

Based on what I found, here's what I can tell you:"""

# Test different templates
def compare_prompt_templates(query):
    """Compare different prompt templates"""

    print(f"\n{'='*70}")
    print(f"PROMPT TEMPLATE COMPARISON")
    print(f"Query: {query}")
    print('='*70)

    # Get context
    context_chunks, _ = retrieve_context(query, n_results=3)

    templates = {
        "Basic": RAGPromptTemplates.basic_template,
        "Detailed": RAGPromptTemplates.detailed_template,
        "Structured": RAGPromptTemplates.structured_template,
        "Conversational": RAGPromptTemplates.conversational_template
    }

    results = {}

    for name, template_func in templates.items():
        print(f"\n--- Template: {name} ---")
        prompt = template_func(query, context_chunks)

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )

        answer = response.choices[0].message.content
        results[name] = answer

        print(answer)
        print()

    return results

# Test
template_results = compare_prompt_templates("What is deep learning?")
```

### Task 3B: Context Formatting Strategies

```python
def format_context_with_metadata(chunks, metadatas=None):
    """
    Format context with rich metadata

    Args:
        chunks: List of text chunks
        metadatas: Optional list of metadata dicts

    Returns:
        str: Formatted context
    """
    formatted = ""

    for i, chunk in enumerate(chunks):
        formatted += f"\n{'='*60}\n"
        formatted += f"CONTEXT SECTION [{i+1}]\n"

        if metadatas and i < len(metadatas):
            meta = metadatas[i]
            if 'source' in meta:
                formatted += f"Source: {meta['source']}\n"
            if 'chunk_index' in meta:
                formatted += f"Section: {meta['chunk_index']}\n"

        formatted += f"{'='*60}\n"
        formatted += chunk + "\n"

    return formatted

# Test rich context formatting
query = "What is machine learning?"
chunks, _ = retrieve_context(query, n_results=3)

# Get metadata if available
results_with_meta = collection.query(
    query_embeddings=embedding_model.encode([query]).tolist(),
    n_results=3,
    include=["documents", "metadatas"]
)

rich_context = format_context_with_metadata(
    results_with_meta['documents'][0],
    results_with_meta['metadatas'][0]
)

print("Rich Context Format:")
print(rich_context)
```

✅ **Checkpoint**: Different prompt templates should produce different response styles.

---

## Exercise 4: Multi-LLM Support (15 min)

**Objective:** Support multiple LLM providers.

### Task 4A: Claude Integration

```python
def rag_pipeline_claude(query, n_results=3, model="claude-3-5-haiku-20241022"):
    """
    RAG pipeline using Claude (Anthropic)

    Args:
        query: User's question
        n_results: Number of chunks to retrieve
        model: Claude model to use

    Returns:
        tuple: (answer, context_chunks, metadata)
    """
    # Check if API key is available
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("Anthropic API key not found. Skipping Claude example.")
        return None, None, None

    anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"RAG with CLAUDE: {query}")
    print('='*70)

    # Retrieve
    print("\n[Step 1/3] Retrieving context...")
    context_chunks, distances = retrieve_context(query, n_results)

    # Create prompt
    print("[Step 2/3] Creating prompt...")
    prompt = create_rag_prompt(query, context_chunks)

    # Generate with Claude
    print(f"[Step 3/3] Generating with Claude {model}...")
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0.3,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.content[0].text
    elapsed_time = time.time() - start_time

    metadata = {
        'model': model,
        'n_retrieved': len(context_chunks),
        'avg_distance': np.mean(distances),
        'time_seconds': elapsed_time,
        'tokens_used': response.usage.input_tokens + response.usage.output_tokens
    }

    print(f"\n{'='*70}")
    print("CLAUDE ANSWER:")
    print('='*70)
    print(answer)
    print(f"\nMetadata: {metadata}")

    return answer, context_chunks, metadata

# Test Claude
# claude_answer, claude_context, claude_meta = rag_pipeline_claude("What is deep learning?")
```

### Task 4B: Unified RAG Interface

```python
class UnifiedRAGPipeline:
    """
    Unified RAG interface supporting multiple LLM providers
    """

    def __init__(self, collection, embedding_model):
        """Initialize with vector database and embedding model"""
        self.collection = collection
        self.embedding_model = embedding_model

        # Initialize LLM clients
        self.openai_client = None
        self.anthropic_client = None

        if os.getenv('OPENAI_API_KEY'):
            self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            print("✓ OpenAI client initialized")

        if os.getenv('ANTHROPIC_API_KEY'):
            self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            print("✓ Claude client initialized")

    def retrieve(self, query, n_results=3, metadata_filter=None):
        """Retrieve relevant context"""
        query_embedding = self.embedding_model.encode([query])

        search_params = {
            'query_embeddings': query_embedding.tolist(),
            'n_results': n_results,
            'include': ['documents', 'distances', 'metadatas']
        }

        if metadata_filter:
            search_params['where'] = metadata_filter

        results = self.collection.query(**search_params)

        return {
            'documents': results['documents'][0],
            'distances': results['distances'][0],
            'metadatas': results['metadatas'][0] if results['metadatas'] else []
        }

    def generate_openai(self, prompt, model="gpt-4o-mini", temperature=0.3):
        """Generate with OpenAI"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=500
        )

        return {
            'answer': response.choices[0].message.content,
            'tokens': response.usage.total_tokens,
            'model': model
        }

    def generate_claude(self, prompt, model="claude-3-5-haiku-20241022", temperature=0.3):
        """Generate with Claude"""
        if not self.anthropic_client:
            raise ValueError("Claude client not initialized")

        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=500,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            'answer': response.content[0].text,
            'tokens': response.usage.input_tokens + response.usage.output_tokens,
            'model': model
        }

    def query(
        self,
        question,
        provider="openai",
        model=None,
        n_results=3,
        template="detailed",
        temperature=0.3
    ):
        """
        Complete RAG query

        Args:
            question: User's question
            provider: 'openai' or 'claude'
            model: Specific model (or use default)
            n_results: Number of chunks to retrieve
            template: Prompt template to use
            temperature: LLM temperature

        Returns:
            dict: Complete response with answer and metadata
        """
        start_time = time.time()

        # Step 1: Retrieve
        retrieval_results = self.retrieve(question, n_results)

        # Step 2: Create prompt
        template_func = getattr(RAGPromptTemplates, f"{template}_template")
        prompt = template_func(question, retrieval_results['documents'])

        # Step 3: Generate
        if provider == "openai":
            if model is None:
                model = "gpt-4o-mini"
            generation_result = self.generate_openai(prompt, model, temperature)

        elif provider == "claude":
            if model is None:
                model = "claude-3-5-haiku-20241022"
            generation_result = self.generate_claude(prompt, model, temperature)

        else:
            raise ValueError(f"Unknown provider: {provider}")

        elapsed_time = time.time() - start_time

        return {
            'question': question,
            'answer': generation_result['answer'],
            'provider': provider,
            'model': generation_result['model'],
            'retrieved_chunks': retrieval_results['documents'],
            'chunk_distances': retrieval_results['distances'],
            'tokens_used': generation_result['tokens'],
            'time_seconds': elapsed_time
        }

# Initialize unified pipeline
unified_rag = UnifiedRAGPipeline(collection, embedding_model)

# Test with OpenAI
result_openai = unified_rag.query(
    "What is machine learning?",
    provider="openai",
    template="detailed"
)

print(f"\n{'='*70}")
print(f"Provider: {result_openai['provider']}")
print(f"Model: {result_openai['model']}")
print(f"Time: {result_openai['time_seconds']:.2f}s")
print(f"Tokens: {result_openai['tokens_used']}")
print(f"\nAnswer:\n{result_openai['answer']}")

# Test with Claude (if available)
# result_claude = unified_rag.query(
#     "What is machine learning?",
#     provider="claude",
#     template="conversational"
# )
```

✅ **Checkpoint**: Unified interface should work with multiple LLM providers.

---

## 🎯 Capstone Project: SupportGenie v3.0 with RAG (30 min)

**Objective:** Build a production-ready RAG-powered support system.

### Requirements

Your system must:
1. ✅ Support semantic search with ChromaDB
2. ✅ Integrate with multiple LLMs (OpenAI, Claude)
3. ✅ Handle conversation history
4. ✅ Provide source citations
5. ✅ Track metrics (latency, tokens, relevance)
6. ✅ Gracefully handle errors

### Complete Implementation

```python
# support_genie_v3.py

class SupportGenieV3:
    """
    Production RAG-powered support system
    Version 3.0 - Complete with retrieval-augmented generation
    """

    SYSTEM_PROMPT = """You are SupportGenie, an AI customer support assistant.

CAPABILITIES:
- Answer questions using provided knowledge base context
- Cite sources for all information
- Admit when information is not available

RESPONSE GUIDELINES:
1. Use ONLY the provided context to answer
2. Cite context sections used: [1], [2], etc.
3. If context is insufficient, say so clearly
4. Be professional, helpful, and concise
5. Structure answers clearly

TONE: Professional, helpful, and empathetic"""

    def __init__(self, collection, embedding_model, llm_provider="openai"):
        """
        Initialize SupportGenie v3.0

        Args:
            collection: ChromaDB collection
            embedding_model: SentenceTransformer model
            llm_provider: 'openai' or 'claude'
        """
        self.collection = collection
        self.embedding_model = embedding_model
        self.llm_provider = llm_provider
        self.conversation_history = []

        # Initialize LLM client
        if llm_provider == "openai":
            self.llm_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.model = "gpt-4o-mini"
        elif llm_provider == "claude":
            self.llm_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            self.model = "claude-3-5-haiku-20241022"
        else:
            raise ValueError(f"Unknown provider: {llm_provider}")

        print(f"✓ SupportGenie v3.0 initialized")
        print(f"  LLM Provider: {llm_provider}")
        print(f"  Model: {self.model}")
        print(f"  Knowledge Base: {collection.count()} documents")

    def retrieve_context(self, query, n_results=3):
        """Retrieve relevant context from knowledge base"""
        query_embedding = self.embedding_model.encode([query])

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=['documents', 'distances', 'metadatas']
        )

        return {
            'documents': results['documents'][0],
            'distances': results['distances'][0],
            'metadatas': results['metadatas'][0] if results['metadatas'] else []
        }

    def create_rag_prompt(self, query, context_data):
        """Create RAG prompt with context"""

        # Format context
        context_text = ""
        for i, (doc, meta) in enumerate(zip(
            context_data['documents'],
            context_data['metadatas']
        )):
            context_text += f"\n[{i+1}] "
            if meta and 'source' in meta:
                context_text += f"(Source: {meta['source']}) "
            context_text += doc + "\n"

        # Build prompt
        prompt = f"""{self.SYSTEM_PROMPT}

KNOWLEDGE BASE CONTEXT:
{context_text}

CUSTOMER QUESTION:
{query}

YOUR RESPONSE:"""

        return prompt

    def generate_response(self, prompt):
        """Generate response using configured LLM"""

        if self.llm_provider == "openai":
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return {
                'answer': response.choices[0].message.content,
                'tokens': response.usage.total_tokens
            }

        elif self.llm_provider == "claude":
            response = self.llm_client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return {
                'answer': response.content[0].text,
                'tokens': response.usage.input_tokens + response.usage.output_tokens
            }

    def chat(self, user_message, n_results=3):
        """
        Main chat interface with RAG

        Args:
            user_message: Customer's question
            n_results: Number of context chunks to retrieve

        Returns:
            dict: Response with answer and metadata
        """
        start_time = time.time()

        try:
            # Step 1: Retrieve context
            context_data = self.retrieve_context(user_message, n_results)

            # Step 2: Create RAG prompt
            prompt = self.create_rag_prompt(user_message, context_data)

            # Step 3: Generate response
            generation_result = self.generate_response(prompt)

            # Calculate metrics
            elapsed_time = time.time() - start_time
            avg_relevance = np.mean(context_data['distances'])

            # Store in history
            interaction = {
                'user_message': user_message,
                'assistant_response': generation_result['answer'],
                'context_used': context_data['documents'],
                'relevance_scores': context_data['distances'],
                'timestamp': time.time()
            }
            self.conversation_history.append(interaction)

            return {
                'success': True,
                'answer': generation_result['answer'],
                'metadata': {
                    'retrieved_chunks': len(context_data['documents']),
                    'avg_relevance': avg_relevance,
                    'tokens_used': generation_result['tokens'],
                    'latency_ms': elapsed_time * 1000,
                    'provider': self.llm_provider,
                    'model': self.model
                },
                'sources': context_data['metadatas']
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'answer': "I'm sorry, I encountered an error processing your request. Please try again."
            }

    def display_response(self, response):
        """Display response in a user-friendly format"""

        print(f"\n{'='*70}")
        if response['success']:
            print("SupportGenie:")
            print('-'*70)
            print(response['answer'])
            print()
            print("Metadata:")
            for key, value in response['metadata'].items():
                print(f"  {key}: {value}")

            if response['sources']:
                print("\nSources Used:")
                for i, source in enumerate(response['sources']):
                    if source and 'source' in source:
                        print(f"  [{i+1}] {source['source']}")
        else:
            print(f"Error: {response['error']}")
        print('='*70)

    def interactive_mode(self):
        """Interactive chat mode"""

        print("\n" + "="*70)
        print("SUPPORTGENIE V3.0 - RAG-Powered Support")
        print("="*70)
        print("\nCommands:")
        print("  Type your question to chat")
        print("  ':history' - Show conversation history")
        print("  ':stats' - Show usage statistics")
        print("  ':quit' - Exit")
        print("\n" + "-"*70)

        while True:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in [':quit', ':exit', ':q']:
                print("\nThank you for using SupportGenie!")
                break

            elif user_input.lower() == ':history':
                self.show_history()
                continue

            elif user_input.lower() == ':stats':
                self.show_stats()
                continue

            # Process query
            response = self.chat(user_input)
            self.display_response(response)

    def show_history(self):
        """Display conversation history"""
        print(f"\n{'='*70}")
        print(f"CONVERSATION HISTORY ({len(self.conversation_history)} interactions)")
        print('='*70)

        for i, interaction in enumerate(self.conversation_history):
            print(f"\n[{i+1}]")
            print(f"User: {interaction['user_message']}")
            print(f"Assistant: {interaction['assistant_response'][:100]}...")

    def show_stats(self):
        """Display usage statistics"""
        if not self.conversation_history:
            print("\nNo interactions yet.")
            return

        total_tokens = sum(
            [interaction.get('tokens_used', 0)
             for interaction in self.conversation_history]
        )

        avg_chunks = np.mean([
            len(interaction['context_used'])
            for interaction in self.conversation_history
        ])

        print(f"\n{'='*70}")
        print("USAGE STATISTICS")
        print('='*70)
        print(f"Total interactions: {len(self.conversation_history)}")
        print(f"Average chunks retrieved: {avg_chunks:.1f}")
        print(f"LLM Provider: {self.llm_provider}")
        print(f"Model: {self.model}")
        print('='*70)


# Initialize SupportGenie v3.0
genie = SupportGenieV3(
    collection=collection,
    embedding_model=embedding_model,
    llm_provider="openai"  # or "claude"
)

# Test queries
test_queries = [
    "What is machine learning?",
    "Tell me about deep learning",
    "What are some applications of AI?"
]

print("\n" + "="*70)
print("TESTING SUPPORTGENIE V3.0")
print("="*70)

for query in test_queries:
    print(f"\nQuery: {query}")
    response = genie.chat(query)
    genie.display_response(response)

# Run interactive mode (optional)
# genie.interactive_mode()
```

### Your Tasks

1. **Test SupportGenie v3.0** with various queries
2. **Add error handling** for edge cases (no API key, empty query, etc.)
3. **Implement conversation context** - use previous messages in retrieval
4. **Add confidence scoring** - indicate how confident the answer is
5. **Create a feedback system** - allow users to rate responses

✅ **Checkpoint**: SupportGenie v3.0 should provide RAG-powered responses with source citations.

---

## 🏆 Extension Challenges

### Challenge 1: Query Rewriting

Improve retrieval by rewriting user queries:

```python
def rewrite_query(original_query):
    """
    Rewrite query for better retrieval

    Args:
        original_query: User's original question

    Returns:
        str: Rewritten query
    """
    rewrite_prompt = f"""Rewrite this question to be more effective for document search.
Make it more specific and include key terms.

Original: {original_query}
Rewritten:"""

    # Use LLM to rewrite
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": rewrite_prompt}],
        temperature=0.5,
        max_tokens=100
    )

    return response.choices[0].message.content.strip()
```

### Challenge 2: Multi-Query RAG

Retrieve with multiple query variations:

```python
def multi_query_rag(original_query, n_variations=3):
    """
    Generate multiple query variations and retrieve for each

    Args:
        original_query: User's question
        n_variations: Number of variations to generate

    Returns:
        list: Combined unique results
    """
    # Generate variations
    variations = generate_query_variations(original_query, n_variations)

    # Retrieve for each
    all_results = []
    for variation in variations:
        docs, _ = retrieve_context(variation, n_results=5)
        all_results.extend(docs)

    # Deduplicate and return top results
    unique_results = list(set(all_results))
    return unique_results[:5]
```

### Challenge 3: RAG Evaluation Framework

Build comprehensive evaluation:

```python
def evaluate_rag_system(test_cases):
    """
    Evaluate RAG system on test cases

    Args:
        test_cases: List of (question, expected_info, expected_sources)

    Returns:
        dict: Evaluation metrics
    """
    results = {
        'accuracy': [],
        'citation_accuracy': [],
        'latency': []
    }

    for question, expected_info, expected_sources in test_cases:
        # Run RAG
        answer, context, metadata = rag_pipeline_openai(question)

        # Check accuracy (simple keyword matching)
        accuracy = 1.0 if expected_info.lower() in answer.lower() else 0.0
        results['accuracy'].append(accuracy)

        # Check citations
        has_citations = '[1]' in answer or '[2]' in answer or '[3]' in answer
        results['citation_accuracy'].append(1.0 if has_citations else 0.0)

        # Track latency
        results['latency'].append(metadata['time_seconds'])

    return {
        'avg_accuracy': np.mean(results['accuracy']),
        'citation_rate': np.mean(results['citation_accuracy']),
        'avg_latency': np.mean(results['latency'])
    }
```

---

## 📝 Key Takeaways

After completing this lab, you should understand:

✅ **RAG Pipeline** - Retrieve → Augment → Generate flow
✅ **Prompt Engineering** - How to structure RAG prompts effectively
✅ **Multi-LLM Integration** - Support OpenAI, Claude, Bedrock
✅ **RAG vs. Non-RAG** - When RAG adds value
✅ **Source Citations** - How to make LLMs cite sources
✅ **Production Considerations** - Error handling, metrics, latency

---

## 🎓 What's Next?

You've completed Lab 5! You now have:
- ✅ Complete RAG pipeline
- ✅ Multi-LLM support
- ✅ Production-ready SupportGenie v3.0
- ✅ RAG evaluation tools

**Next Lab**: Lab 6 - AI Agents & Tool Calling
Learn how to give your RAG system the ability to take actions and use tools!

---

**Lab 5 Complete!** 🎉
[← Back to Learning Material](learning.md) | [Next: Lab 6 →](../Lab6-AI-Agents/learning.md)
