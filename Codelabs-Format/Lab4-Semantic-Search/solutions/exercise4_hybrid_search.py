"""
Lab 4 - Exercise 4: Hybrid Search with BM25
Solution for combining semantic and keyword search

Learning Objectives:
- Understand BM25 keyword search
- Combine semantic + keyword search
- Implement ranking fusion
- Optimize hybrid retrieval
"""

from sentence_transformers import SentenceTransformer
import chromadb
from exercise1_basic_search import setup_sample_database
import numpy as np
from typing import List, Dict
import re
from collections import Counter
import math


class BM25:
    """
    BM25 (Best Matching 25) keyword search algorithm

    Features:
    - Term frequency-inverse document frequency
    - Document length normalization
    - Tunable parameters (k1, b)
    """

    def __init__(self, corpus: List[str], k1=1.5, b=0.75):
        """
        Initialize BM25

        Args:
            corpus: List of documents
            k1: Term frequency saturation parameter (default 1.5)
            b: Length normalization parameter (default 0.75)
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b

        # Tokenize documents
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]

        # Calculate statistics
        self.doc_count = len(corpus)
        self.avgdl = sum(len(doc) for doc in self.tokenized_corpus) / self.doc_count

        # Calculate IDF for all terms
        self.idf = self._calculate_idf()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def _calculate_idf(self) -> Dict[str, float]:
        """Calculate IDF for all terms in corpus"""
        idf = {}

        # Count documents containing each term
        for doc in self.tokenized_corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                idf[term] = idf.get(term, 0) + 1

        # Calculate IDF: log((N - df + 0.5) / (df + 0.5))
        for term, df in idf.items():
            idf[term] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

        return idf

    def score(self, query: str, doc_idx: int) -> float:
        """
        Calculate BM25 score for a document

        Args:
            query: Query string
            doc_idx: Document index

        Returns:
            float: BM25 score
        """
        score = 0.0
        doc_tokens = self.tokenized_corpus[doc_idx]
        doc_len = len(doc_tokens)

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Count term frequencies in document
        term_freqs = Counter(doc_tokens)

        for term in query_tokens:
            if term not in term_freqs:
                continue

            # Get term frequency and IDF
            tf = term_freqs[term]
            idf = self.idf.get(term, 0)

            # Calculate BM25 component for this term
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))

            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search corpus with BM25

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            list: Ranked results
        """
        # Score all documents
        scores = [(i, self.score(query, i)) for i in range(self.doc_count)]

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        results = []
        for idx, score in scores[:top_k]:
            results.append({
                'doc_idx': idx,
                'document': self.corpus[idx],
                'bm25_score': score
            })

        return results


class HybridSearch:
    """
    Hybrid search combining semantic and keyword search

    Features:
    - Semantic search (vector similarity)
    - Keyword search (BM25)
    - Reciprocal Rank Fusion (RRF)
    - Configurable weighting
    """

    def __init__(self, collection, embedding_model):
        """Initialize hybrid search"""
        self.collection = collection
        self.embedding_model = embedding_model

        # Get all documents for BM25
        all_docs = collection.get()
        self.documents = all_docs['documents']
        self.doc_ids = all_docs['ids']

        # Initialize BM25
        self.bm25 = BM25(self.documents)

    def semantic_search(self, query: str, n_results: int = 10) -> List[Dict]:
        """Semantic search using embeddings"""
        query_embedding = self.embedding_model.encode(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

        # Format results
        formatted = []
        for i in range(len(results['documents'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'semantic_distance': results['distances'][0][i],
                'metadata': results['metadatas'][0][i]
            })

        return formatted

    def keyword_search(self, query: str, n_results: int = 10) -> List[Dict]:
        """Keyword search using BM25"""
        results = self.bm25.search(query, top_k=n_results)

        # Add document IDs
        for result in results:
            result['id'] = self.doc_ids[result['doc_idx']]

        return results

    def reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        Combine results using Reciprocal Rank Fusion (RRF)

        RRF score = sum(1 / (k + rank)) for each result list

        Args:
            semantic_results: Semantic search results
            keyword_results: Keyword search results
            k: Constant (default 60)

        Returns:
            list: Fused and ranked results
        """
        # Calculate RRF scores
        rrf_scores = {}

        # Add semantic rankings
        for rank, result in enumerate(semantic_results, 1):
            doc_id = result['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))

        # Add keyword rankings
        for rank, result in enumerate(keyword_results, 1):
            doc_id = result['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))

        # Get full document info
        doc_info = {}
        for result in semantic_results + keyword_results:
            if result['id'] not in doc_info:
                doc_info[result['id']] = result

        # Create ranked list
        ranked = []
        for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            info = doc_info[doc_id]
            ranked.append({
                'id': doc_id,
                'document': info['document'],
                'rrf_score': score,
                'semantic_distance': info.get('semantic_distance', None),
                'bm25_score': info.get('bm25_score', None),
                'metadata': info.get('metadata', {})
            })

        return ranked

    def hybrid_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Perform hybrid search

        Args:
            query: Search query
            n_results: Number of final results

        Returns:
            list: Hybrid search results
        """
        # Get results from both methods (fetch more for fusion)
        semantic_results = self.semantic_search(query, n_results=10)
        keyword_results = self.keyword_search(query, n_results=10)

        # Fuse results
        fused_results = self.reciprocal_rank_fusion(
            semantic_results,
            keyword_results
        )

        return fused_results[:n_results]

    def display_results(self, results: List[Dict], title: str = "Results"):
        """Display search results"""
        print(f"\n{title}")
        print("=" * 70)

        if not results:
            print("  No results found")
            return

        for i, result in enumerate(results, 1):
            print(f"\n{i}. Document: {result['document'][:60]}...")

            if 'rrf_score' in result:
                print(f"   RRF Score: {result['rrf_score']:.4f}")

            if result.get('semantic_distance') is not None:
                print(f"   Semantic Distance: {result['semantic_distance']:.4f}")

            if result.get('bm25_score') is not None:
                print(f"   BM25 Score: {result['bm25_score']:.4f}")

            if result.get('metadata'):
                topic = result['metadata'].get('topic', 'N/A')
                print(f"   Topic: {topic}")


def main():
    """Run demonstrations"""
    print("=" * 70)
    print("EXERCISE 4: HYBRID SEARCH WITH BM25")
    print("=" * 70)

    # Setup
    client, collection, embedding_model = setup_sample_database()

    # Task 4A: BM25 Keyword Search
    print("\n\n🔤 TASK 4A: BM25 KEYWORD SEARCH")
    print("=" * 70)

    # Get all documents for BM25
    all_docs = collection.get()
    documents = all_docs['documents']

    bm25 = BM25(documents)

    query = "neural networks layers"
    print(f"\n📝 Query: '{query}'")

    bm25_results = bm25.search(query, top_k=3)

    print(f"\n🏆 Top 3 BM25 Results:")
    for i, result in enumerate(bm25_results, 1):
        print(f"\n{i}. BM25 Score: {result['bm25_score']:.4f}")
        print(f"   Text: {result['document'][:70]}...")

    # Task 4B: Compare Semantic vs Keyword
    print("\n\n⚖️  TASK 4B: SEMANTIC VS KEYWORD SEARCH")
    print("=" * 70)

    hybrid = HybridSearch(collection, embedding_model)

    test_query = "understanding language"
    print(f"\n📝 Query: '{test_query}'")

    # Semantic search
    semantic_results = hybrid.semantic_search(test_query, n_results=3)
    hybrid.display_results(semantic_results, "🧠 Semantic Search Results")

    # Keyword search
    keyword_results = hybrid.keyword_search(test_query, n_results=3)
    hybrid.display_results(keyword_results, "🔤 Keyword (BM25) Results")

    # Task 4C: Hybrid Search with RRF
    print("\n\n🔀 TASK 4C: HYBRID SEARCH (RRF)")
    print("=" * 70)

    print("\n💡 Reciprocal Rank Fusion combines rankings from both methods")

    queries = [
        "neural networks for machine learning",
        "natural language understanding",
        "computer vision images"
    ]

    for query in queries:
        print(f"\n📝 Query: '{query}'")

        hybrid_results = hybrid.hybrid_search(query, n_results=3)
        hybrid.display_results(hybrid_results, "🏆 Hybrid Search Results")

    # Task 4D: When to Use Each Method
    print("\n\n🎯 TASK 4D: WHEN TO USE EACH METHOD")
    print("=" * 70)

    print("\n🧠 Semantic Search:")
    print("   ✅ Conceptual queries (\"how does learning work?\")")
    print("   ✅ Synonym matching (\"car\" finds \"automobile\")")
    print("   ✅ Paraphrase detection")
    print("   ✅ Cross-lingual search")

    print("\n🔤 Keyword Search (BM25):")
    print("   ✅ Exact term matching (\"Python 3.11\")")
    print("   ✅ Technical terms (\"transformers architecture\")")
    print("   ✅ Proper nouns (\"TensorFlow\")")
    print("   ✅ Code/formula search")

    print("\n🔀 Hybrid Search:")
    print("   ✅ Best of both worlds")
    print("   ✅ Production search systems")
    print("   ✅ When precision matters")
    print("   ✅ Diverse query types")

    print("\n\n" + "=" * 70)
    print("✅ EXERCISE 4 COMPLETE!")
    print("=" * 70)

    print("\n💡 KEY TAKEAWAYS:")
    print("1. BM25 uses term frequency and inverse document frequency")
    print("2. Semantic search captures meaning, BM25 captures keywords")
    print("3. Reciprocal Rank Fusion combines rankings effectively")
    print("4. Hybrid search provides better results than either alone")
    print("5. Choose method based on query type and use case")

    print("\n📖 BM25 FORMULA:")
    print("   score(D,Q) = Σ IDF(qi) · (f(qi,D) · (k1+1)) / (f(qi,D) + k1·(1-b+b·|D|/avgdl))")
    print("   where:")
    print("     - IDF(qi) = inverse document frequency of query term qi")
    print("     - f(qi,D) = frequency of qi in document D")
    print("     - |D| = length of document D")
    print("     - avgdl = average document length")
    print("     - k1, b = tuning parameters (typically k1=1.5, b=0.75)")

    print("\n🔀 RECIPROCAL RANK FUSION:")
    print("   RRF(d) = Σ 1/(k + rank(d))")
    print("   where:")
    print("     - k = constant (typically 60)")
    print("     - rank(d) = rank of document d in result list")
    print("   Sum over all retrieval methods (semantic + keyword)")


if __name__ == "__main__":
    main()
