"""
Lab 3 - Exercise 5: Vector Database with ChromaDB
Solution for storing and querying embeddings

Learning Objectives:
- Set up ChromaDB vector database
- Store documents with embeddings
- Query by similarity
- Manage collections
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorDatabase:
    """
    Vector database wrapper for ChromaDB

    Features:
    - Document storage with auto-embedding
    - Similarity search
    - Collection management
    """

    def __init__(self, persist_directory="./chroma_db"):
        """Initialize ChromaDB client"""
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_collection(self, name, reset=False):
        """Create or get a collection"""
        if reset:
            try:
                self.client.delete_collection(name)
            except:
                pass

        collection = self.client.get_or_create_collection(
            name=name,
            metadata={"description": f"Collection: {name}"}
        )

        return collection

    def add_documents(self, collection, documents, metadatas=None, ids=None):
        """
        Add documents to collection

        Args:
            collection: ChromaDB collection
            documents: List of text documents
            metadatas: Optional metadata for each document
            ids: Optional IDs for documents
        """
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()

        # Add to collection
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        return len(documents)

    def query(self, collection, query_text, n_results=5):
        """
        Query collection for similar documents

        Args:
            collection: ChromaDB collection
            query_text: Query string
            n_results: Number of results to return

        Returns:
            dict: Query results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query_text).tolist()

        # Query collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return results


def main():
    """Run demonstrations"""
    print("=" * 70)
    print("EXERCISE 5: VECTOR DATABASE WITH CHROMADB")
    print("=" * 70)

    # Task 5A: Initialize Database
    print("\n💾 TASK 5A: INITIALIZE DATABASE")
    print("=" * 70)

    db = VectorDatabase(persist_directory="./demo_chroma_db")
    print("✅ Vector database initialized!")

    # Task 5B: Create Collection and Add Documents
    print("\n\n📚 TASK 5B: CREATE COLLECTION & ADD DOCUMENTS")
    print("=" * 70)

    collection = db.create_collection("ai_docs", reset=True)

    documents = [
        "Machine learning enables computers to learn from data",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing handles human language",
        "Computer vision analyzes and interprets images",
        "Reinforcement learning trains agents through rewards"
    ]

    metadatas = [
        {"topic": "ML", "category": "fundamentals"},
        {"topic": "DL", "category": "advanced"},
        {"topic": "NLP", "category": "language"},
        {"topic": "CV", "category": "vision"},
        {"topic": "RL", "category": "advanced"}
    ]

    num_added = db.add_documents(collection, documents, metadatas)
    print(f"✅ Added {num_added} documents to collection!")

    # Task 5C: Query the Database
    print("\n\n🔍 TASK 5C: QUERY DATABASE")
    print("=" * 70)

    queries = [
        "neural networks for learning",
        "processing text and language",
        "visual recognition systems"
    ]

    for query in queries:
        print(f"\n📝 Query: '{query}'")

        results = db.query(collection, query, n_results=2)

        print(f"\n🏆 Top 2 Results:")
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            print(f"\n{i}. Distance: {distance:.4f}")
            print(f"   Topic: {metadata['topic']}")
            print(f"   Text: '{doc}'")

    print("\n\n" + "=" * 70)
    print("✅ EXERCISE 5 COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
