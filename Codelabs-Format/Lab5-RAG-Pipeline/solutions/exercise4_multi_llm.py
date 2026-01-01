"""
Lab 5 - Exercise 4: Multi-LLM Support
Solution for integrating multiple LLM providers

Learning Objectives:
- Integrate Claude (Anthropic) API
- Create unified RAG interface
- Support multiple LLM providers
- Compare LLM performance
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import time
import numpy as np
from typing import Dict, Optional, List
from exercise1_basic_rag import setup_sample_knowledge_base
from exercise3_prompt_strategies import RAGPromptTemplates

# Load environment variables
load_dotenv()


class UnifiedRAGPipeline:
    """
    Unified RAG interface supporting multiple LLM providers

    Supported providers:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)

    Features:
    - Provider-agnostic interface
    - Automatic failover
    - Performance tracking
    """

    def __init__(self, collection, embedding_model):
        """
        Initialize unified RAG pipeline

        Args:
            collection: ChromaDB collection
            embedding_model: SentenceTransformer model
        """
        self.collection = collection
        self.embedding_model = embedding_model

        # Initialize LLM clients
        self.openai_client = None
        self.anthropic_client = None

        # Track available providers
        self.available_providers = []

        # Initialize OpenAI
        if os.getenv('OPENAI_API_KEY'):
            try:
                self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                self.available_providers.append('openai')
                print("✅ OpenAI client initialized")
            except Exception as e:
                print(f"⚠️  OpenAI initialization failed: {e}")

        # Initialize Anthropic (Claude)
        if os.getenv('ANTHROPIC_API_KEY'):
            try:
                self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
                self.available_providers.append('claude')
                print("✅ Claude client initialized")
            except Exception as e:
                print(f"⚠️  Claude initialization failed: {e}")

        if not self.available_providers:
            raise ValueError("No LLM providers available. Please configure API keys.")

        print(f"\n🔌 Available providers: {', '.join(self.available_providers)}")

    def retrieve(self, query: str, n_results: int = 3, metadata_filter: Optional[Dict] = None) -> Dict:
        """
        Retrieve relevant context from vector database

        Args:
            query: User's question
            n_results: Number of chunks to retrieve
            metadata_filter: Optional metadata filter

        Returns:
            dict: Retrieved documents, distances, and metadata
        """
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

    def generate_openai(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 500
    ) -> Dict:
        """
        Generate response using OpenAI

        Args:
            prompt: Complete RAG prompt
            model: OpenAI model name
            temperature: LLM temperature
            max_tokens: Maximum tokens in response

        Returns:
            dict: Generated answer and metadata
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return {
            'answer': response.choices[0].message.content,
            'tokens': response.usage.total_tokens,
            'model': model,
            'provider': 'openai'
        }

    def generate_claude(
        self,
        prompt: str,
        model: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.3,
        max_tokens: int = 500
    ) -> Dict:
        """
        Generate response using Claude (Anthropic)

        Args:
            prompt: Complete RAG prompt
            model: Claude model name
            temperature: LLM temperature
            max_tokens: Maximum tokens in response

        Returns:
            dict: Generated answer and metadata
        """
        if not self.anthropic_client:
            raise ValueError("Claude client not initialized")

        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            'answer': response.content[0].text,
            'tokens': response.usage.input_tokens + response.usage.output_tokens,
            'model': model,
            'provider': 'claude'
        }

    def query(
        self,
        question: str,
        provider: str = "openai",
        model: Optional[str] = None,
        n_results: int = 3,
        template: str = "detailed",
        temperature: float = 0.3,
        metadata_filter: Optional[Dict] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Complete RAG query with specified provider

        Args:
            question: User's question
            provider: LLM provider ('openai' or 'claude')
            model: Specific model (or use default)
            n_results: Number of chunks to retrieve
            template: Prompt template to use
            temperature: LLM temperature
            metadata_filter: Optional metadata filter
            verbose: Print progress

        Returns:
            dict: Complete response with answer and metadata
        """
        start_time = time.time()

        if verbose:
            print(f"\n{'='*70}")
            print(f"RAG QUERY: {question}")
            print(f"Provider: {provider}")
            print('='*70)

        # Validate provider
        if provider not in self.available_providers:
            raise ValueError(f"Provider '{provider}' not available. Available: {self.available_providers}")

        # Step 1: Retrieve
        if verbose:
            print("\n[Step 1/3] Retrieving context...")

        retrieval_results = self.retrieve(question, n_results, metadata_filter)

        if verbose:
            print(f"   ✅ Retrieved {len(retrieval_results['documents'])} chunks")

        # Step 2: Create prompt
        if verbose:
            print("\n[Step 2/3] Creating RAG prompt...")

        template_func = getattr(RAGPromptTemplates, f"{template}_template")
        prompt = template_func(question, retrieval_results['documents'])

        # Step 3: Generate
        if verbose:
            print(f"\n[Step 3/3] Generating with {provider}...")

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

        # Compile response
        response = {
            'success': True,
            'question': question,
            'answer': generation_result['answer'],
            'provider': provider,
            'model': generation_result['model'],
            'retrieved_chunks': retrieval_results['documents'],
            'chunk_distances': retrieval_results['distances'],
            'chunk_metadatas': retrieval_results['metadatas'],
            'tokens_used': generation_result['tokens'],
            'avg_distance': np.mean(retrieval_results['distances']),
            'time_seconds': elapsed_time
        }

        if verbose:
            print(f"\n{'='*70}")
            print(f"ANSWER:")
            print('='*70)
            print(generation_result['answer'])
            print(f"\n{'='*70}")
            print("METADATA:")
            print('='*70)
            print(f"   Provider: {response['provider']}")
            print(f"   Model: {response['model']}")
            print(f"   Tokens: {response['tokens_used']}")
            print(f"   Time: {response['time_seconds']:.2f}s")
            print(f"   Avg distance: {response['avg_distance']:.4f}")
            print('='*70)

        return response


class LLMComparator:
    """
    Compare different LLM providers

    Features:
    - Side-by-side comparison
    - Performance metrics
    - Cost estimation
    """

    # Approximate pricing (as of 2024, per 1M tokens)
    PRICING = {
        'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
        'gpt-4o': {'input': 2.50, 'output': 10.00},
        'claude-3-5-haiku-20241022': {'input': 0.80, 'output': 4.00},
        'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00}
    }

    def __init__(self, unified_rag: UnifiedRAGPipeline):
        """
        Initialize LLM comparator

        Args:
            unified_rag: Configured UnifiedRAGPipeline
        """
        self.unified_rag = unified_rag

    def compare_providers(
        self,
        query: str,
        providers: List[str] = None,
        n_results: int = 3
    ) -> Dict:
        """
        Compare multiple providers on same query

        Args:
            query: User's question
            providers: List of providers to compare
            n_results: Number of chunks to retrieve

        Returns:
            dict: Comparison results
        """
        if providers is None:
            providers = self.unified_rag.available_providers

        print(f"\n{'#'*70}")
        print(f"LLM PROVIDER COMPARISON")
        print(f"{'#'*70}")
        print(f"Query: {query}\n")

        results = {}

        for provider in providers:
            if provider not in self.unified_rag.available_providers:
                print(f"\n⚠️  Skipping {provider} (not available)")
                continue

            print(f"\n{'='*70}")
            print(f"PROVIDER: {provider.upper()}")
            print('='*70)

            try:
                result = self.unified_rag.query(
                    query,
                    provider=provider,
                    n_results=n_results,
                    verbose=False
                )

                results[provider] = result

                print(f"\nAnswer:\n{result['answer']}")
                print(f"\n📊 Tokens: {result['tokens_used']}")
                print(f"⏱️  Time: {result['time_seconds']:.2f}s")

            except Exception as e:
                print(f"❌ Error with {provider}: {e}")

        # Analysis
        if len(results) > 1:
            print(f"\n{'#'*70}")
            print("COMPARATIVE ANALYSIS")
            print(f"{'#'*70}")

            # Token comparison
            print("\n📊 Token Usage:")
            for provider, result in results.items():
                print(f"   {provider}: {result['tokens_used']} tokens")

            # Speed comparison
            print("\n⏱️  Response Time:")
            for provider, result in results.items():
                print(f"   {provider}: {result['time_seconds']:.2f}s")

            # Cost estimation
            print("\n💰 Estimated Cost (per query):")
            for provider, result in results.items():
                model = result['model']
                if model in self.PRICING:
                    # Rough estimate (assume 50/50 input/output split)
                    tokens = result['tokens_used']
                    cost = (tokens / 1_000_000) * (
                        (self.PRICING[model]['input'] + self.PRICING[model]['output']) / 2
                    )
                    print(f"   {provider}: ${cost:.6f}")

        return results


def main():
    """Demonstrate multi-LLM support"""
    print("="*70)
    print("EXERCISE 4: MULTI-LLM SUPPORT")
    print("="*70)

    # Check for API keys
    has_openai = bool(os.getenv('OPENAI_API_KEY'))
    has_claude = bool(os.getenv('ANTHROPIC_API_KEY'))

    print("\n🔑 API Key Status:")
    print(f"   OpenAI: {'✅' if has_openai else '❌'}")
    print(f"   Claude: {'✅' if has_claude else '❌'}")

    if not (has_openai or has_claude):
        print("\n❌ Error: No API keys found")
        print("   Please configure at least one provider in .env file")
        return

    # Setup
    print("\n" + "="*70)
    print("SETUP")
    print("="*70)

    client, collection, embedding_model = setup_sample_knowledge_base()

    # Task 4A: Initialize unified pipeline
    print("\n" + "="*70)
    print("TASK 4A: UNIFIED RAG PIPELINE")
    print("="*70)

    unified_rag = UnifiedRAGPipeline(collection, embedding_model)

    # Task 4B: Test with OpenAI
    if has_openai:
        print("\n" + "="*70)
        print("TASK 4B: OPENAI RAG")
        print("="*70)

        result_openai = unified_rag.query(
            "What is machine learning?",
            provider="openai",
            template="detailed",
            n_results=3
        )

    # Task 4C: Test with Claude
    if has_claude:
        print("\n" + "="*70)
        print("TASK 4C: CLAUDE RAG")
        print("="*70)

        result_claude = unified_rag.query(
            "What is machine learning?",
            provider="claude",
            template="detailed",
            n_results=3
        )

    # Task 4D: Compare providers
    if len(unified_rag.available_providers) > 1:
        print("\n" + "="*70)
        print("TASK 4D: PROVIDER COMPARISON")
        print("="*70)

        comparator = LLMComparator(unified_rag)

        test_queries = [
            "Explain deep learning",
            "What is natural language processing?"
        ]

        for query in test_queries:
            comparator.compare_providers(query, n_results=3)
            print("\n" + "-"*70 + "\n")

    # Summary
    print("\n" + "="*70)
    print("✅ EXERCISE 4 COMPLETE!")
    print("="*70)

    print("\n💡 KEY TAKEAWAYS:")
    print("   1. Unified interface simplifies multi-provider support")
    print("   2. Different providers have different strengths")
    print("   3. Token usage varies by provider")
    print("   4. Response times depend on model size")
    print("   5. Cost considerations important for production")

    print("\n🎯 PROVIDER SELECTION GUIDE:")
    print("   OpenAI (GPT-4o-mini):")
    print("     ✓ Fast, cost-effective")
    print("     ✓ Good for high-volume applications")
    print("     ✓ Strong general knowledge")
    print("\n   Claude (Haiku):")
    print("     ✓ Strong reasoning capabilities")
    print("     ✓ Good instruction following")
    print("     ✓ Longer context windows")

    print("\n⚡ PRODUCTION TIPS:")
    print("   • Use faster models for simple queries")
    print("   • Implement fallback providers")
    print("   • Monitor costs and latency")
    print("   • Cache common responses")
    print("   • Load balance across providers")


if __name__ == "__main__":
    main()
