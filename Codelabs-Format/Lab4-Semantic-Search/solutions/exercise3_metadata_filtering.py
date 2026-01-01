"""
Lab 4 - Exercise 3: Metadata Filtering
Solution for filtering search results by metadata

Learning Objectives:
- Filter by metadata attributes
- Combine semantic search with metadata
- Build faceted search
- Route queries intelligently
"""

from sentence_transformers import SentenceTransformer
import chromadb
from exercise1_basic_search import setup_sample_database


class MetadataFilteredSearch:
    """
    Search with metadata filtering

    Features:
    - Filter by any metadata field
    - Combine multiple filters
    - Topic-based routing
    """

    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model

    def search_with_filter(self, query, metadata_filter, n_results=5):
        """
        Search with metadata filtering

        Args:
            query: Search query
            metadata_filter: Dict of metadata filters
            n_results: Number of results

        Returns:
            list: Filtered results
        """
        query_embedding = self.embedding_model.encode(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=metadata_filter
        )

        return results

    def search_by_topic(self, query, topic, n_results=3):
        """Search within a specific topic"""
        return self.search_with_filter(
            query,
            metadata_filter={"topic": topic},
            n_results=n_results
        )

    def search_by_difficulty(self, query, difficulty, n_results=3):
        """Search by difficulty level"""
        return self.search_with_filter(
            query,
            metadata_filter={"difficulty": difficulty},
            n_results=n_results
        )

    def display_results(self, results, title="Results"):
        """Display search results"""
        print(f"\n{title}")
        print("-" * 70)

        if not results['documents'][0]:
            print("  No results found")
            return

        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]

            print(f"\n{i+1}. Distance: {dist:.4f}")
            print(f"   Topic: {meta.get('topic', 'N/A')}")
            print(f"   Difficulty: {meta.get('difficulty', 'N/A')}")
            print(f"   Text: {doc[:80]}...")


def main():
    """Run demonstrations"""
    print("=" * 70)
    print("EXERCISE 3: METADATA FILTERING")
    print("=" * 70)

    # Setup
    client, collection, embedding_model = setup_sample_database()
    search = MetadataFilteredSearch(collection, embedding_model)

    # Task 3A: Filter by Topic
    print("\n\n🏷️  TASK 3A: FILTER BY TOPIC")
    print("=" * 70)

    query = "learning from data"
    print(f"\n📝 Query: '{query}'")

    # Search in ML topic
    results_ml = search.search_by_topic(query, "ML", n_results=3)
    search.display_results(results_ml, "🔍 Results filtered by Topic: ML")

    # Search in NLP topic
    results_nlp = search.search_by_topic(query, "NLP", n_results=3)
    search.display_results(results_nlp, "🔍 Results filtered by Topic: NLP")

    # Task 3B: Filter by Difficulty
    print("\n\n📊 TASK 3B: FILTER BY DIFFICULTY")
    print("=" * 70)

    query = "machine learning concepts"

    for difficulty in ["beginner", "intermediate", "advanced"]:
        results = search.search_by_difficulty(query, difficulty, n_results=2)
        search.display_results(results, f"🎯 Difficulty: {difficulty.upper()}")

    # Task 3C: Combined Filters
    print("\n\n🔀 TASK 3C: COMBINED FILTERS")
    print("=" * 70)

    print("\n📝 Query: 'neural networks'")
    print("   Filters: topic=NLP AND difficulty=advanced")

    combined_filter = {
        "$and": [
            {"topic": "NLP"},
            {"difficulty": "advanced"}
        ]
    }

    results = search.search_with_filter(
        "neural networks",
        metadata_filter=combined_filter,
        n_results=3
    )

    search.display_results(results, "🔍 Combined Filter Results")

    print("\n\n" + "=" * 70)
    print("✅ EXERCISE 3 COMPLETE!")
    print("=" * 70)

    print("\n💡 KEY TAKEAWAYS:")
    print("1. Metadata filtering narrows search scope")
    print("2. Combine semantic similarity + metadata")
    print("3. Use for topic routing and faceted search")
    print("4. Improves precision for specific domains")
    print("5. Supports complex filter combinations")


if __name__ == "__main__":
    main()
