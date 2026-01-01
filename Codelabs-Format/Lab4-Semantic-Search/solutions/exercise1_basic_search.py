"""
Lab 4 - Exercise 1: Basic Semantic Search
Solution for querying vector databases

Learning Objectives:
- Connect to ChromaDB
- Perform semantic search
- Understand distance metrics
- Retrieve and rank results
"""

from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np


def setup_sample_database():
    """Create a sample database for testing"""
    # Initialize
    client = chromadb.PersistentClient(path="./lab4_demo_db")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create collection
    try:
        client.delete_collection("ai_knowledge")
    except:
        pass

    collection = client.create_collection("ai_knowledge")

    # Sample documents
    documents = [
        "Machine learning enables computers to learn from data without explicit programming",
        "Deep learning uses neural networks with multiple layers to process information",
        "Natural language processing helps computers understand and generate human language",
        "Computer vision allows machines to interpret and analyze visual information",
        "Reinforcement learning trains agents through rewards and penalties",
        "Transformers are a neural network architecture used in modern NLP models",
        "Supervised learning uses labeled data to train predictive models",
        "Unsupervised learning finds patterns in unlabeled data"
    ]

    metadatas = [
        {"topic": "ML", "difficulty": "beginner"},
        {"topic": "DL", "difficulty": "intermediate"},
        {"topic": "NLP", "difficulty": "intermediate"},
        {"topic": "CV", "difficulty": "intermediate"},
        {"topic": "RL", "difficulty": "advanced"},
        {"topic": "NLP", "difficulty": "advanced"},
        {"topic": "ML", "difficulty": "beginner"},
        {"topic": "ML", "difficulty": "intermediate"}
    ]

    # Generate embeddings and add
    embeddings = embedding_model.encode(documents)

    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )

    print(f"✅ Created sample database with {len(documents)} documents")
    return client, collection, embedding_model


def semantic_search(collection, embedding_model, query, n_results=3):
    """
    Perform semantic search on the vector database

    Args:
        collection: ChromaDB collection
        embedding_model: Sentence transformer model
        query: Search query string
        n_results: Number of results to return

    Returns:
        dict: Search results
    """
    # Generate query embedding
    query_embedding = embedding_model.encode(query)

    # Search
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )

    return results


class SemanticSearchEngine:
    """
    Basic semantic search engine

    Features:
    - Vector similarity search
    - Configurable top-K
    - Result formatting
    """

    def __init__(self, collection, embedding_model):
        """Initialize search engine"""
        self.collection = collection
        self.embedding_model = embedding_model

    def search(self, query, n_results=5):
        """
        Search for similar documents

        Args:
            query: Search query
            n_results: Number of results

        Returns:
            list: Formatted results
        """
        # Generate embedding
        query_embedding = self.embedding_model.encode(query)

        # Query database
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'rank': i + 1,
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'id': results['ids'][0][i]
            })

        return formatted_results

    def display_results(self, results, query):
        """Pretty print search results"""
        print(f"\n🔍 Query: '{query}'")
        print(f"📊 Found {len(results)} results")
        print("=" * 70)

        for result in results:
            print(f"\n#{result['rank']} - Distance: {result['distance']:.4f}")
            print(f"Topic: {result['metadata'].get('topic', 'N/A')}")
            print(f"Text: {result['document']}")
            print("-" * 70)


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run all demonstrations"""
    print("=" * 70)
    print("EXERCISE 1: BASIC SEMANTIC SEARCH")
    print("=" * 70)

    # Setup sample database
    print("\n📚 Setting up sample database...")
    client, collection, embedding_model = setup_sample_database()

    # Task 1A: Connect to Database
    print("\n\n📡 TASK 1A: CONNECT TO DATABASE")
    print("=" * 70)

    print(f"✅ Model: all-MiniLM-L6-v2")
    print(f"✅ Embedding dimensions: {embedding_model.get_sentence_embedding_dimension()}")
    print(f"✅ Collection: {collection.name}")
    print(f"✅ Documents: {collection.count()}")

    # Task 1B: Simple Search Function
    print("\n\n🔍 TASK 1B: SIMPLE SEARCH FUNCTION")
    print("=" * 70)

    query = "neural networks for learning"
    print(f"\n📝 Query: '{query}'")

    results = semantic_search(collection, embedding_model, query, n_results=3)

    print(f"\n🏆 Top 3 Results:")
    for i in range(len(results['documents'][0])):
        print(f"\n{i+1}. Distance: {results['distances'][0][i]:.4f}")
        print(f"   Topic: {results['metadatas'][0][i]['topic']}")
        print(f"   Text: {results['documents'][0][i]}")

    # Task 1C: Search Engine Class
    print("\n\n🏭 TASK 1C: SEARCH ENGINE CLASS")
    print("=" * 70)

    search_engine = SemanticSearchEngine(collection, embedding_model)

    # Test multiple queries
    test_queries = [
        "understanding human language",
        "training models with labeled data",
        "computer vision for images"
    ]

    for query in test_queries:
        results = search_engine.search(query, n_results=2)
        search_engine.display_results(results, query)

    # Task 1D: Understanding Distance Metrics
    print("\n\n📏 TASK 1D: UNDERSTANDING DISTANCE METRICS")
    print("=" * 70)

    print("\n📊 Distance Interpretation:")
    print("   < 0.5  → Very similar (excellent match)")
    print("   0.5-1.0 → Similar (good match)")
    print("   1.0-1.5 → Somewhat related")
    print("   > 1.5   → Not very related")

    # Test with different queries
    print("\n\n🧪 Testing with varied queries:")

    test_cases = [
        ("neural networks with layers", "Should match: Deep learning"),
        ("rewards and penalties", "Should match: Reinforcement learning"),
        ("weather forecast", "Should NOT match well (off-topic)")
    ]

    for query, expected in test_cases:
        print(f"\n📝 Query: '{query}'")
        print(f"   Expected: {expected}")

        results = search_engine.search(query, n_results=1)
        top_result = results[0]

        print(f"   Got: {top_result['document'][:60]}...")
        print(f"   Distance: {top_result['distance']:.4f}")

    # Key takeaways
    print("\n\n" + "=" * 70)
    print("✅ EXERCISE 1 COMPLETE!")
    print("=" * 70)

    print("\n💡 KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. Semantic search finds meaning, not just keywords")
    print("2. ChromaDB handles embedding storage and search")
    print("3. Lower distance = more similar")
    print("4. Query embedding is compared to all stored embeddings")
    print("5. Top-K returns K most similar results")
    print("6. Works even with different wording (synonyms)")

    print("\n📖 HOW IT WORKS:")
    print("=" * 70)
    print("1. Query text → Embedding (384D vector)")
    print("2. Compare query embedding to all stored embeddings")
    print("3. Calculate distance (L2/Euclidean by default)")
    print("4. Return top-K results sorted by distance")

    print("\n🎯 ADVANTAGES OVER KEYWORD SEARCH:")
    print("=" * 70)
    print("✅ Finds semantically similar content")
    print("✅ Handles synonyms naturally")
    print("✅ No exact keyword match needed")
    print("✅ Understands context and meaning")
    print("✅ Works across languages (with multilingual models)")


if __name__ == "__main__":
    main()
