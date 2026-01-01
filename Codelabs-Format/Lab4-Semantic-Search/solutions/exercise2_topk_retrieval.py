"""
Lab 4 - Exercise 2: Top-K Retrieval Strategies
Solution for optimizing result retrieval

Learning Objectives:
- Understand Top-K parameter
- Compare different K values
- Optimize retrieval performance
- Balance precision and recall
"""

from sentence_transformers import SentenceTransformer
import chromadb
from exercise1_basic_search import setup_sample_database


class TopKOptimizer:
    """
    Optimize Top-K retrieval strategies

    Features:
    - Compare different K values
    - Analyze result quality
    - Performance metrics
    """

    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model

    def compare_topk(self, query, k_values=[1, 3, 5, 10]):
        """
        Compare results for different K values

        Args:
            query: Search query
            k_values: List of K values to test

        Returns:
            dict: Results for each K
        """
        print(f"\n🔍 Query: '{query}'")
        print("=" * 70)

        query_embedding = self.embedding_model.encode(query)
        results_by_k = {}

        for k in k_values:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(k, self.collection.count())
            )

            print(f"\n📊 Top-{k} Results:")
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                distance = results['distances'][0][i]
                print(f"  {i+1}. Distance: {distance:.4f} - {doc[:60]}...")

            results_by_k[k] = results

        return results_by_k

    def analyze_distance_distribution(self, query, k=10):
        """Analyze how distance changes across results"""
        query_embedding = self.embedding_model.encode(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, self.collection.count())
        )

        distances = results['distances'][0]

        print(f"\n📈 Distance Distribution (Top-{k}):")
        print(f"   Min distance: {min(distances):.4f}")
        print(f"   Max distance: {max(distances):.4f}")
        print(f"   Range: {max(distances) - min(distances):.4f}")

        # Quality zones
        excellent = sum(1 for d in distances if d < 0.5)
        good = sum(1 for d in distances if 0.5 <= d < 1.0)
        fair = sum(1 for d in distances if 1.0 <= d < 1.5)
        poor = sum(1 for d in distances if d >= 1.5)

        print(f"\n🎯 Quality Breakdown:")
        print(f"   Excellent (< 0.5):    {excellent} results")
        print(f"   Good (0.5-1.0):       {good} results")
        print(f"   Fair (1.0-1.5):       {fair} results")
        print(f"   Poor (>= 1.5):        {poor} results")


def main():
    """Run demonstrations"""
    print("=" * 70)
    print("EXERCISE 2: TOP-K RETRIEVAL STRATEGIES")
    print("=" * 70)

    # Setup
    client, collection, embedding_model = setup_sample_database()

    # Task 2A: Compare K Values
    print("\n\n⚖️  TASK 2A: COMPARE K VALUES")
    print("=" * 70)

    optimizer = TopKOptimizer(collection, embedding_model)

    query = "neural networks for machine learning"
    optimizer.compare_topk(query, k_values=[1, 3, 5])

    # Task 2B: Distance Distribution
    print("\n\n📊 TASK 2B: DISTANCE DISTRIBUTION")
    print("=" * 70)

    optimizer.analyze_distance_distribution(query, k=8)

    # Task 2C: Optimal K Selection
    print("\n\n🎯 TASK 2C: OPTIMAL K SELECTION")
    print("=" * 70)

    print("\n💡 Guidelines for selecting K:")
    print("\nK = 1-3:")
    print("  Use for: Precise answers, direct questions")
    print("  Example: 'What is deep learning?'")

    print("\nK = 3-5:")
    print("  Use for: General search, balanced results")
    print("  Example: 'Machine learning techniques'")

    print("\nK = 5-10:")
    print("  Use for: Exploration, broad topics")
    print("  Example: 'AI applications'")

    print("\nK > 10:")
    print("  Use for: Comprehensive research, ensuring coverage")
    print("  Example: 'Everything about neural networks'")

    print("\n\n" + "=" * 70)
    print("✅ EXERCISE 2 COMPLETE!")
    print("=" * 70)

    print("\n💡 KEY TAKEAWAYS:")
    print("1. Larger K = More results, lower precision")
    print("2. Smaller K = Fewer results, higher precision")
    print("3. Monitor distance distribution")
    print("4. Adjust K based on use case")
    print("5. Consider filtering low-quality results")


if __name__ == "__main__":
    main()
