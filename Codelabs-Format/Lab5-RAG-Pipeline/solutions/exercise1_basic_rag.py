"""
Lab 5 - Exercise 1: Basic RAG Pipeline
Solution for building a complete Retrieval-Augmented Generation pipeline

Learning Objectives:
- Build complete RAG pipeline (Retrieve → Augment → Generate)
- Integrate vector database with LLM
- Create effective RAG prompts
- Understand RAG workflow
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import numpy as np
import time
from typing import List, Tuple, Dict, Optional

# Load environment variables
load_dotenv()


def setup_sample_knowledge_base():
    """
    Create sample knowledge base for testing RAG

    Returns:
        tuple: (client, collection, embedding_model)
    """
    print("\n📚 Setting up sample knowledge base...")

    # Initialize components
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./lab5_rag_db")

    # Create collection
    try:
        client.delete_collection("ai_knowledge")
    except:
        pass

    collection = client.create_collection("ai_knowledge")

    # Sample documents about AI topics
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses statistical techniques to give computers the ability to learn and improve from experience.",

        "Deep learning is a specialized form of machine learning that uses artificial neural networks with multiple layers. These deep neural networks can automatically learn hierarchical representations of data, making them particularly effective for complex tasks like image recognition and natural language processing.",

        "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language. NLP techniques enable machines to read, understand, and derive meaning from human languages. Applications include chatbots, translation, and sentiment analysis.",

        "Computer vision is an AI field that trains computers to interpret and understand visual information from the world. It involves methods for acquiring, processing, and analyzing images and videos. Common applications include facial recognition, object detection, and autonomous vehicles.",

        "Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties for its actions and learns to maximize cumulative reward over time. This approach is used in robotics, game playing, and autonomous systems.",

        "Supervised learning is a machine learning approach where models are trained on labeled data. The algorithm learns to map inputs to outputs based on example input-output pairs. Common applications include classification tasks like spam detection and regression tasks like price prediction.",

        "Unsupervised learning is a machine learning technique that finds patterns in data without predefined labels. It discovers hidden structures in unlabeled data through clustering, dimensionality reduction, and anomaly detection. This is useful for exploratory data analysis and pattern discovery.",

        "Transfer learning is a technique where knowledge gained from solving one problem is applied to a different but related problem. Pre-trained models can be fine-tuned on new tasks with less data and computational resources. This approach has revolutionized fields like computer vision and NLP.",

        "Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of interconnected nodes (neurons) organized in layers that process information. Neural networks can learn complex patterns and are the foundation of deep learning.",

        "Transformers are a neural network architecture that has become dominant in NLP. They use self-attention mechanisms to process sequential data in parallel, making them more efficient than recurrent networks. Models like GPT and BERT are based on transformer architecture."
    ]

    # Metadata for each document
    metadatas = [
        {"topic": "ML", "difficulty": "beginner", "category": "fundamentals"},
        {"topic": "DL", "difficulty": "intermediate", "category": "fundamentals"},
        {"topic": "NLP", "difficulty": "intermediate", "category": "application"},
        {"topic": "CV", "difficulty": "intermediate", "category": "application"},
        {"topic": "RL", "difficulty": "advanced", "category": "fundamentals"},
        {"topic": "ML", "difficulty": "beginner", "category": "fundamentals"},
        {"topic": "ML", "difficulty": "intermediate", "category": "fundamentals"},
        {"topic": "ML", "difficulty": "intermediate", "category": "technique"},
        {"topic": "DL", "difficulty": "intermediate", "category": "architecture"},
        {"topic": "NLP", "difficulty": "advanced", "category": "architecture"}
    ]

    # Generate embeddings and add to collection
    embeddings = embedding_model.encode(documents)

    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )

    print(f"   ✅ Created knowledge base with {len(documents)} documents")

    return client, collection, embedding_model


class BasicRAGPipeline:
    """
    Basic RAG (Retrieval-Augmented Generation) Pipeline

    Implements the three-step RAG process:
    1. Retrieve: Find relevant context from vector database
    2. Augment: Combine query with retrieved context in prompt
    3. Generate: Use LLM to generate answer based on context
    """

    def __init__(self, collection, embedding_model, openai_api_key: str):
        """
        Initialize RAG pipeline

        Args:
            collection: ChromaDB collection
            embedding_model: SentenceTransformer model
            openai_api_key: OpenAI API key
        """
        self.collection = collection
        self.embedding_model = embedding_model
        self.openai_client = OpenAI(api_key=openai_api_key)

        print("✅ RAG Pipeline initialized")
        print(f"   Knowledge base: {collection.count()} documents")

    def retrieve(self, query: str, n_results: int = 3) -> Dict:
        """
        Step 1: Retrieve relevant context from vector database

        Args:
            query: User's question
            n_results: Number of chunks to retrieve

        Returns:
            dict: Retrieved documents, distances, and metadata
        """
        # Encode query to embedding
        query_embedding = self.embedding_model.encode([query])

        # Search vector database
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=["documents", "distances", "metadatas"]
        )

        return {
            'documents': results['documents'][0],
            'distances': results['distances'][0],
            'metadatas': results['metadatas'][0] if results['metadatas'] else []
        }

    def create_rag_prompt(self, query: str, context_chunks: List[str]) -> str:
        """
        Step 2: Augment - Create prompt with retrieved context

        Args:
            query: User's question
            context_chunks: Retrieved text chunks

        Returns:
            str: Complete RAG prompt
        """
        # Format context sections
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_text += f"[{i}] {chunk}\n\n"

        # Create structured RAG prompt
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

    def generate(self, prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.3) -> Dict:
        """
        Step 3: Generate answer using LLM

        Args:
            prompt: Complete RAG prompt
            model: OpenAI model to use
            temperature: LLM temperature

        Returns:
            dict: Generated answer and metadata
        """
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

    def query(
        self,
        question: str,
        n_results: int = 3,
        model: str = "gpt-4o-mini",
        verbose: bool = True
    ) -> Dict:
        """
        Complete RAG pipeline execution

        Args:
            question: User's question
            n_results: Number of context chunks to retrieve
            model: OpenAI model to use
            verbose: Print progress information

        Returns:
            dict: Complete response with answer and metadata
        """
        start_time = time.time()

        if verbose:
            print(f"\n{'='*70}")
            print(f"RAG PIPELINE: {question}")
            print('='*70)

        # Step 1: Retrieve
        if verbose:
            print("\n[Step 1/3] Retrieving relevant context...")

        retrieval_results = self.retrieve(question, n_results)

        if verbose:
            print(f"   Retrieved {len(retrieval_results['documents'])} chunks:")
            for i, dist in enumerate(retrieval_results['distances'], 1):
                print(f"      [{i}] Distance: {dist:.4f}")

        # Step 2: Augment (create prompt)
        if verbose:
            print("\n[Step 2/3] Creating RAG prompt...")

        prompt = self.create_rag_prompt(question, retrieval_results['documents'])

        # Step 3: Generate
        if verbose:
            print(f"\n[Step 3/3] Generating answer with {model}...")

        generation_result = self.generate(prompt, model)

        elapsed_time = time.time() - start_time

        # Compile complete response
        response = {
            'success': True,
            'question': question,
            'answer': generation_result['answer'],
            'retrieved_chunks': retrieval_results['documents'],
            'chunk_distances': retrieval_results['distances'],
            'chunk_metadatas': retrieval_results['metadatas'],
            'model': generation_result['model'],
            'tokens_used': generation_result['tokens'],
            'avg_distance': np.mean(retrieval_results['distances']),
            'time_seconds': elapsed_time
        }

        if verbose:
            print(f"\n{'='*70}")
            print("ANSWER:")
            print('='*70)
            print(generation_result['answer'])
            print(f"\n{'='*70}")
            print("METADATA:")
            print('='*70)
            print(f"   Model: {response['model']}")
            print(f"   Tokens used: {response['tokens_used']}")
            print(f"   Chunks retrieved: {len(response['retrieved_chunks'])}")
            print(f"   Avg relevance: {response['avg_distance']:.4f}")
            print(f"   Time: {response['time_seconds']:.2f}s")
            print('='*70)

        return response

    def display_context(self, retrieval_results: Dict):
        """Display retrieved context chunks"""
        print("\n📚 RETRIEVED CONTEXT:")
        print("="*70)

        for i, (doc, dist, meta) in enumerate(zip(
            retrieval_results['documents'],
            retrieval_results['distances'],
            retrieval_results['metadatas']
        ), 1):
            print(f"\n[{i}] Distance: {dist:.4f}")
            if meta:
                print(f"    Topic: {meta.get('topic', 'N/A')}")
                print(f"    Difficulty: {meta.get('difficulty', 'N/A')}")
            print(f"    Text: {doc[:100]}...")


def main():
    """Demonstrate basic RAG pipeline"""
    print("="*70)
    print("EXERCISE 1: BASIC RAG PIPELINE")
    print("="*70)

    # Check for API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("\n❌ Error: OPENAI_API_KEY not found in environment")
        print("   Please set your API key in .env file:")
        print("   OPENAI_API_KEY=sk-your-key-here")
        return

    print("\n✅ OpenAI API key loaded")

    # Task 1A: Setup knowledge base
    print("\n" + "="*70)
    print("TASK 1A: INITIALIZE COMPONENTS")
    print("="*70)

    client, collection, embedding_model = setup_sample_knowledge_base()

    # Task 1B: Initialize RAG pipeline
    print("\n" + "="*70)
    print("TASK 1B: BUILD RAG PIPELINE")
    print("="*70)

    rag = BasicRAGPipeline(collection, embedding_model, openai_api_key)

    # Task 1C: Test retrieval
    print("\n" + "="*70)
    print("TASK 1C: TEST RETRIEVAL")
    print("="*70)

    test_query = "What is deep learning?"
    print(f"\n📝 Query: '{test_query}'")

    retrieval_results = rag.retrieve(test_query, n_results=3)
    rag.display_context(retrieval_results)

    # Task 1D: Test complete RAG pipeline
    print("\n" + "="*70)
    print("TASK 1D: COMPLETE RAG PIPELINE")
    print("="*70)

    test_queries = [
        "What is machine learning?",
        "Explain natural language processing",
        "How does reinforcement learning work?"
    ]

    for query in test_queries:
        response = rag.query(query, n_results=3, verbose=True)
        print("\n" + "-"*70 + "\n")

    # Summary
    print("\n" + "="*70)
    print("✅ EXERCISE 1 COMPLETE!")
    print("="*70)

    print("\n💡 KEY CONCEPTS:")
    print("   1. RAG = Retrieve → Augment → Generate")
    print("   2. Retrieval uses semantic search with embeddings")
    print("   3. Augmentation combines context with query in prompt")
    print("   4. Generation uses LLM to answer based on context")
    print("   5. Source citations improve trustworthiness")

    print("\n🎯 RAG ADVANTAGES:")
    print("   ✅ Grounds answers in provided context")
    print("   ✅ Reduces hallucinations")
    print("   ✅ Enables source citations")
    print("   ✅ Works with domain-specific knowledge")
    print("   ✅ No model retraining needed")

    print("\n📊 PIPELINE METRICS:")
    print("   - Retrieval quality: Average distance (lower = better)")
    print("   - Token usage: Context + query + response tokens")
    print("   - Latency: Retrieval time + generation time")
    print("   - Citation accuracy: Does response cite sources?")


if __name__ == "__main__":
    main()
