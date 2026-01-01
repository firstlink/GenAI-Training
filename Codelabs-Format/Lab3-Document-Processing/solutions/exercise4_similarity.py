"""
Lab 3 - Exercise 4: Semantic Similarity
Solution for calculating similarity between embeddings

Learning Objectives:
- Calculate cosine similarity
- Find most similar texts
- Build similarity matrix
- Understand distance metrics
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    return dot_product / (magnitude1 * magnitude2)


def find_most_similar(query_embedding, embeddings, texts, top_k=3):
    """
    Find most similar texts to query

    Args:
        query_embedding: Query vector
        embeddings: Matrix of embeddings
        texts: Original texts
        top_k: Number of results to return

    Returns:
        list: Top-k similar texts with scores
    """
    # Calculate similarities
    similarities = []
    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((texts[i], sim))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


class SimilarityCalculator:
    """Calculate semantic similarity between texts"""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        emb1 = self.model.encode(text1)
        emb2 = self.model.encode(text2)
        return cosine_similarity(emb1, emb2)

    def find_similar(self, query, corpus, top_k=5):
        """Find similar texts in corpus"""
        query_emb = self.model.encode(query)
        corpus_emb = self.model.encode(corpus)

        # Use sentence-transformers util for efficiency
        similarities = util.cos_sim(query_emb, corpus_emb)[0]

        # Get top-k indices
        top_indices = np.argsort(-similarities.numpy())[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'text': corpus[idx],
                'score': float(similarities[idx])
            })

        return results


def main():
    """Run demonstrations"""
    print("=" * 70)
    print("EXERCISE 4: SEMANTIC SIMILARITY")
    print("=" * 70)

    # Task 4A: Cosine Similarity
    print("\n📐 TASK 4A: COSINE SIMILARITY")
    print("=" * 70)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    text1 = "Machine learning is a subset of AI"
    text2 = "AI includes machine learning"
    text3 = "The weather is nice today"

    emb1, emb2, emb3 = model.encode([text1, text2, text3])

    sim_1_2 = cosine_similarity(emb1, emb2)
    sim_1_3 = cosine_similarity(emb1, emb3)

    print(f"\nText 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Similarity: {sim_1_2:.4f} (similar topics)\n")

    print(f"Text 1: '{text1}'")
    print(f"Text 3: '{text3}'")
    print(f"Similarity: {sim_1_3:.4f} (different topics)")

    # Task 4B: Find Most Similar
    print("\n\n🔍 TASK 4B: FIND MOST SIMILAR")
    print("=" * 70)

    corpus = [
        "Deep learning uses neural networks",
        "NLP handles natural language",
        "Computer vision analyzes images",
        "ML algorithms learn from data",
        "The sun is shining brightly"
    ]

    query = "Neural networks for machine learning"

    print(f"\n📝 Query: '{query}'")
    print(f"\n📚 Searching in corpus of {len(corpus)} texts...")

    calculator = SimilarityCalculator()
    results = calculator.find_similar(query, corpus, top_k=3)

    print(f"\n🏆 Top 3 Similar Texts:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Text: '{result['text']}'")

    print("\n\n" + "=" * 70)
    print("✅ EXERCISE 4 COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
