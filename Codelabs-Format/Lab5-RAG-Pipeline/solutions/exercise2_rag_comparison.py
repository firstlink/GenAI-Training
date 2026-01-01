"""
Lab 5 - Exercise 2: RAG vs Non-RAG Comparison
Solution for understanding when RAG adds value

Learning Objectives:
- Compare RAG and non-RAG responses
- Understand RAG value proposition
- Identify when RAG helps most
- Analyze differences in response quality
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import time
from typing import Dict, Tuple
from exercise1_basic_rag import setup_sample_knowledge_base, BasicRAGPipeline

# Load environment variables
load_dotenv()


class RAGComparator:
    """
    Compare RAG and non-RAG approaches

    Features:
    - Side-by-side comparison
    - Performance metrics
    - Quality analysis
    """

    def __init__(self, rag_pipeline: BasicRAGPipeline, openai_api_key: str):
        """
        Initialize comparator

        Args:
            rag_pipeline: Configured RAG pipeline
            openai_api_key: OpenAI API key
        """
        self.rag_pipeline = rag_pipeline
        self.openai_client = OpenAI(api_key=openai_api_key)

    def generate_without_rag(
        self,
        query: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3
    ) -> Dict:
        """
        Generate answer WITHOUT retrieval (baseline)

        Args:
            query: User's question
            model: OpenAI model
            temperature: LLM temperature

        Returns:
            dict: Answer and metadata
        """
        start_time = time.time()

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            temperature=temperature,
            max_tokens=500
        )

        elapsed_time = time.time() - start_time

        return {
            'answer': response.choices[0].message.content,
            'model': model,
            'tokens_used': response.usage.total_tokens,
            'time_seconds': elapsed_time
        }

    def compare(self, query: str, n_results: int = 3) -> Dict:
        """
        Compare RAG vs non-RAG for a query

        Args:
            query: User's question
            n_results: Number of chunks for RAG

        Returns:
            dict: Comparison results
        """
        print(f"\n{'#'*70}")
        print(f"COMPARISON: RAG vs NON-RAG")
        print(f"{'#'*70}")
        print(f"Query: {query}\n")

        # Get non-RAG response
        print("="*70)
        print("WITHOUT RAG (LLM Only - No Retrieved Context)")
        print("="*70)

        no_rag_result = self.generate_without_rag(query)
        print(f"\n{no_rag_result['answer']}")
        print(f"\n📊 Tokens used: {no_rag_result['tokens_used']}")
        print(f"⏱️  Time: {no_rag_result['time_seconds']:.2f}s")

        # Get RAG response
        print("\n" + "="*70)
        print("WITH RAG (LLM + Retrieved Context)")
        print("="*70)

        rag_result = self.rag_pipeline.query(query, n_results=n_results, verbose=False)

        print(f"\n{rag_result['answer']}")
        print(f"\n📊 Tokens used: {rag_result['tokens_used']}")
        print(f"📚 Chunks retrieved: {len(rag_result['retrieved_chunks'])}")
        print(f"🎯 Avg relevance: {rag_result['avg_distance']:.4f}")
        print(f"⏱️  Time: {rag_result['time_seconds']:.2f}s")

        # Analysis
        print(f"\n{'#'*70}")
        print("ANALYSIS")
        print(f"{'#'*70}")

        token_diff = rag_result['tokens_used'] - no_rag_result['tokens_used']
        time_diff = rag_result['time_seconds'] - no_rag_result['time_seconds']

        print(f"\n📊 Token Comparison:")
        print(f"   Non-RAG: {no_rag_result['tokens_used']} tokens")
        print(f"   RAG: {rag_result['tokens_used']} tokens")
        print(f"   Difference: +{token_diff} tokens ({(token_diff/no_rag_result['tokens_used']*100):.1f}% increase)")

        print(f"\n⏱️  Time Comparison:")
        print(f"   Non-RAG: {no_rag_result['time_seconds']:.2f}s")
        print(f"   RAG: {rag_result['time_seconds']:.2f}s")
        print(f"   Difference: +{time_diff:.2f}s")

        print("\n🔍 Key Differences to Observe:")
        print("   ✓ Is RAG answer more specific to our documents?")
        print("   ✓ Does RAG answer cite sources [1], [2], [3]?")
        print("   ✓ Is non-RAG answer more general/generic?")
        print("   ✓ Does RAG ground its claims in provided context?")

        # Check for citations
        has_citations = any(f'[{i}]' in rag_result['answer'] for i in range(1, 6))
        print(f"\n📖 RAG Citations: {'✅ Yes' if has_citations else '❌ No'}")

        print(f"\n{'#'*70}\n")

        return {
            'query': query,
            'no_rag': no_rag_result,
            'rag': rag_result,
            'token_difference': token_diff,
            'time_difference': time_diff,
            'rag_has_citations': has_citations
        }

    def batch_compare(self, queries: list) -> Dict:
        """
        Compare multiple queries

        Args:
            queries: List of questions

        Returns:
            dict: Aggregated comparison results
        """
        results = []

        for query in queries:
            result = self.compare(query)
            results.append(result)
            print("\n" + "="*70 + "\n")

        # Aggregate statistics
        avg_token_diff = sum(r['token_difference'] for r in results) / len(results)
        avg_time_diff = sum(r['time_difference'] for r in results) / len(results)
        citation_rate = sum(r['rag_has_citations'] for r in results) / len(results)

        print(f"\n{'#'*70}")
        print(f"AGGREGATE ANALYSIS ({len(queries)} queries)")
        print(f"{'#'*70}")
        print(f"\n📊 Average token increase: +{avg_token_diff:.0f} tokens")
        print(f"⏱️  Average time increase: +{avg_time_diff:.2f}s")
        print(f"📖 Citation rate: {citation_rate*100:.0f}%")

        return {
            'results': results,
            'avg_token_difference': avg_token_diff,
            'avg_time_difference': avg_time_diff,
            'citation_rate': citation_rate
        }


def demonstrate_rag_value(rag_pipeline: BasicRAGPipeline):
    """
    Show when RAG provides most value

    Args:
        rag_pipeline: Configured RAG pipeline
    """
    print(f"\n{'='*70}")
    print("WHEN DOES RAG HELP MOST?")
    print('='*70)

    # Test 1: Specific query (RAG should excel)
    print("\n--- Test 1: Specific Query (RAG should excel) ---")
    specific_query = "What specific types of machine learning are mentioned in the documents?"
    print(f"Query: {specific_query}")

    result1 = rag_pipeline.query(specific_query, n_results=3, verbose=False)
    print(f"\nRAG Answer:\n{result1['answer']}")
    print(f"\n✅ RAG retrieved {len(result1['retrieved_chunks'])} relevant chunks")

    # Test 2: General query (RAG may not add much)
    print("\n" + "="*70)
    print("--- Test 2: General Query (RAG may not add as much value) ---")
    general_query = "What is Python programming?"
    print(f"Query: {general_query}")

    result2 = rag_pipeline.query(general_query, n_results=3, verbose=False)
    print(f"\nRAG Answer:\n{result2['answer']}")
    print(f"\n⚠️  Avg distance: {result2['avg_distance']:.4f} (higher = less relevant)")

    # Insights
    print("\n" + "="*70)
    print("INSIGHTS:")
    print("="*70)

    print("\n🎯 RAG adds MOST value when:")
    print("   ✅ Question is about specific document content")
    print("   ✅ Information is not in LLM's training data")
    print("   ✅ Need to cite sources for trustworthiness")
    print("   ✅ Domain-specific or recent information")
    print("   ✅ Organization-specific knowledge")

    print("\n⚡ RAG adds LESS value when:")
    print("   ⚠️  Question is about general knowledge")
    print("   ⚠️  Retrieved context is not relevant")
    print("   ⚠️  Query is outside knowledge base scope")

    print("\n💡 BEST PRACTICES:")
    print("   1. Use RAG for domain-specific queries")
    print("   2. Monitor retrieval quality (avg distance)")
    print("   3. Fall back to non-RAG for general queries")
    print("   4. Combine RAG with query classification")


def analyze_response_quality(rag_result: Dict, no_rag_result: Dict):
    """
    Analyze quality differences between RAG and non-RAG

    Args:
        rag_result: RAG response
        no_rag_result: Non-RAG response
    """
    print("\n" + "="*70)
    print("RESPONSE QUALITY ANALYSIS")
    print("="*70)

    # Length comparison
    rag_length = len(rag_result['answer'])
    no_rag_length = len(no_rag_result['answer'])

    print(f"\n📏 Response Length:")
    print(f"   RAG: {rag_length} characters")
    print(f"   Non-RAG: {no_rag_length} characters")

    # Citation check
    has_citations = any(f'[{i}]' in rag_result['answer'] for i in range(1, 6))
    print(f"\n📖 Citations:")
    print(f"   RAG: {'✅ Includes citations' if has_citations else '❌ No citations'}")
    print(f"   Non-RAG: ❌ No citations (by design)")

    # Specificity indicators
    specific_words = ['specifically', 'particular', 'according to', 'states that', 'mentions']
    rag_specific_count = sum(1 for word in specific_words if word.lower() in rag_result['answer'].lower())
    no_rag_specific_count = sum(1 for word in specific_words if word.lower() in no_rag_result['answer'].lower())

    print(f"\n🎯 Specificity Indicators:")
    print(f"   RAG: {rag_specific_count} specific terms")
    print(f"   Non-RAG: {no_rag_specific_count} specific terms")

    # Generic indicators
    generic_words = ['generally', 'typically', 'usually', 'often', 'in general']
    rag_generic_count = sum(1 for word in generic_words if word.lower() in rag_result['answer'].lower())
    no_rag_generic_count = sum(1 for word in generic_words if word.lower() in no_rag_result['answer'].lower())

    print(f"\n🌐 Generic Indicators:")
    print(f"   RAG: {rag_generic_count} generic terms")
    print(f"   Non-RAG: {no_rag_generic_count} generic terms")


def main():
    """Demonstrate RAG vs non-RAG comparison"""
    print("="*70)
    print("EXERCISE 2: RAG vs NON-RAG COMPARISON")
    print("="*70)

    # Check for API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("\n❌ Error: OPENAI_API_KEY not found")
        return

    print("\n✅ OpenAI API key loaded")

    # Setup
    print("\n" + "="*70)
    print("SETUP")
    print("="*70)

    client, collection, embedding_model = setup_sample_knowledge_base()
    rag_pipeline = BasicRAGPipeline(collection, embedding_model, openai_api_key)

    # Task 2A: Initialize comparator
    print("\n" + "="*70)
    print("TASK 2A: NON-RAG BASELINE")
    print("="*70)

    comparator = RAGComparator(rag_pipeline, openai_api_key)
    print("✅ RAG Comparator initialized")

    # Task 2B: Side-by-side comparison
    print("\n" + "="*70)
    print("TASK 2B: SIDE-BY-SIDE COMPARISON")
    print("="*70)

    test_queries = [
        "What is machine learning?",
        "Explain deep learning",
        "How does natural language processing work?"
    ]

    batch_results = comparator.batch_compare(test_queries)

    # Task 2C: When RAG helps most
    print("\n" + "="*70)
    print("TASK 2C: WHEN RAG HELPS MOST")
    print("="*70)

    demonstrate_rag_value(rag_pipeline)

    # Additional analysis
    print("\n" + "="*70)
    print("DETAILED QUALITY ANALYSIS")
    print("="*70)

    sample_query = "What is reinforcement learning?"
    comparison = comparator.compare(sample_query)

    analyze_response_quality(
        comparison['rag'],
        comparison['no_rag']
    )

    # Summary
    print("\n" + "="*70)
    print("✅ EXERCISE 2 COMPLETE!")
    print("="*70)

    print("\n💡 KEY TAKEAWAYS:")
    print("   1. RAG grounds answers in provided context")
    print("   2. RAG enables source citations")
    print("   3. RAG uses more tokens (context overhead)")
    print("   4. RAG is slower (retrieval + generation)")
    print("   5. RAG excels for domain-specific queries")

    print("\n📊 WHEN TO USE:")
    print("   ✅ RAG: Domain knowledge, citations needed, specific info")
    print("   ⚡ Non-RAG: General knowledge, speed critical, broad queries")

    print("\n🎯 OPTIMIZATION STRATEGIES:")
    print("   • Use query classification to route queries")
    print("   • Monitor retrieval quality (avg distance)")
    print("   • Cache common RAG responses")
    print("   • Adjust n_results based on query complexity")


if __name__ == "__main__":
    main()
