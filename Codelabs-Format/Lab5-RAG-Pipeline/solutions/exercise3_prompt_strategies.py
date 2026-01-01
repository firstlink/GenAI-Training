"""
Lab 5 - Exercise 3: Advanced Prompt Strategies
Solution for experimenting with different RAG prompt formats

Learning Objectives:
- Create multiple prompt templates
- Compare prompt effectiveness
- Format context with metadata
- Optimize RAG prompts for different use cases
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict
from exercise1_basic_rag import setup_sample_knowledge_base

# Load environment variables
load_dotenv()


class RAGPromptTemplates:
    """
    Collection of different RAG prompt strategies

    Templates:
    - Basic: Simple, straightforward
    - Detailed: Explicit instructions
    - Structured: Formatted output
    - Conversational: Friendly tone
    """

    @staticmethod
    def basic_template(query: str, context_chunks: List[str]) -> str:
        """
        Simple, straightforward prompt

        Args:
            query: User's question
            context_chunks: Retrieved text chunks

        Returns:
            str: Basic RAG prompt
        """
        context = "\n\n".join([f"[{i+1}] {chunk}"
                               for i, chunk in enumerate(context_chunks)])

        return f"""Answer the question using the context below.

Context:
{context}

Question: {query}

Answer:"""

    @staticmethod
    def detailed_template(query: str, context_chunks: List[str]) -> str:
        """
        Detailed with explicit instructions

        Args:
            query: User's question
            context_chunks: Retrieved text chunks

        Returns:
            str: Detailed RAG prompt
        """
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
    def structured_template(query: str, context_chunks: List[str]) -> str:
        """
        Structured output format

        Args:
            query: User's question
            context_chunks: Retrieved text chunks

        Returns:
            str: Structured RAG prompt
        """
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
    def conversational_template(query: str, context_chunks: List[str]) -> str:
        """
        Friendly, conversational tone

        Args:
            query: User's question
            context_chunks: Retrieved text chunks

        Returns:
            str: Conversational RAG prompt
        """
        context = "\n\n".join([f"[{i+1}] {chunk}"
                               for i, chunk in enumerate(context_chunks)])

        return f"""You're a friendly AI assistant helping users understand information from documents.

Here's what I found in our documents:
{context}

The user asked: "{query}"

Based on what I found, here's what I can tell you:"""

    @staticmethod
    def chain_of_thought_template(query: str, context_chunks: List[str]) -> str:
        """
        Chain-of-thought reasoning prompt

        Args:
            query: User's question
            context_chunks: Retrieved text chunks

        Returns:
            str: Chain-of-thought RAG prompt
        """
        context = "\n\n".join([f"[{i+1}] {chunk}"
                               for i, chunk in enumerate(context_chunks)])

        return f"""Answer the question by reasoning step-by-step using the provided context.

CONTEXT:
{context}

QUESTION:
{query}

Let's solve this step by step:
1. First, identify relevant information from the context
2. Then, connect the relevant pieces
3. Finally, formulate a clear answer with citations

REASONING:"""

    @staticmethod
    def expert_template(query: str, context_chunks: List[str], domain: str = "AI") -> str:
        """
        Expert persona with domain knowledge

        Args:
            query: User's question
            context_chunks: Retrieved text chunks
            domain: Expert domain

        Returns:
            str: Expert RAG prompt
        """
        context = "\n\n".join([f"[{i+1}] {chunk}"
                               for i, chunk in enumerate(context_chunks)])

        return f"""You are an expert {domain} consultant. Answer the question based on the provided context with authority and precision.

REFERENCE MATERIALS:
{context}

CLIENT QUESTION:
{query}

EXPERT RESPONSE:
[Provide a clear, authoritative answer citing specific sections]"""


class PromptComparator:
    """
    Compare different prompt templates

    Features:
    - Test multiple templates
    - Analyze response differences
    - Identify best template for use case
    """

    def __init__(self, collection, embedding_model, openai_api_key: str):
        """
        Initialize prompt comparator

        Args:
            collection: ChromaDB collection
            embedding_model: SentenceTransformer model
            openai_api_key: OpenAI API key
        """
        self.collection = collection
        self.embedding_model = embedding_model
        self.openai_client = OpenAI(api_key=openai_api_key)

    def retrieve_context(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve context chunks for query"""
        query_embedding = self.embedding_model.encode([query])

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=["documents"]
        )

        return results['documents'][0]

    def test_template(
        self,
        template_name: str,
        template_func,
        query: str,
        context_chunks: List[str],
        model: str = "gpt-4o-mini"
    ) -> Dict:
        """
        Test a single template

        Args:
            template_name: Name of template
            template_func: Template function
            query: User's question
            context_chunks: Retrieved chunks
            model: OpenAI model

        Returns:
            dict: Response and metadata
        """
        # Create prompt
        prompt = template_func(query, context_chunks)

        # Generate response
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )

        answer = response.choices[0].message.content

        return {
            'template': template_name,
            'answer': answer,
            'tokens': response.usage.total_tokens,
            'prompt_length': len(prompt)
        }

    def compare_templates(
        self,
        query: str,
        templates: Dict = None,
        n_results: int = 3
    ) -> Dict:
        """
        Compare multiple prompt templates

        Args:
            query: User's question
            templates: Dict of template_name -> template_func
            n_results: Number of chunks to retrieve

        Returns:
            dict: Comparison results
        """
        if templates is None:
            templates = {
                "Basic": RAGPromptTemplates.basic_template,
                "Detailed": RAGPromptTemplates.detailed_template,
                "Structured": RAGPromptTemplates.structured_template,
                "Conversational": RAGPromptTemplates.conversational_template,
                "Chain-of-Thought": RAGPromptTemplates.chain_of_thought_template,
                "Expert": RAGPromptTemplates.expert_template
            }

        print(f"\n{'='*70}")
        print(f"PROMPT TEMPLATE COMPARISON")
        print(f"Query: {query}")
        print('='*70)

        # Retrieve context once
        context_chunks = self.retrieve_context(query, n_results)
        print(f"\n✅ Retrieved {len(context_chunks)} context chunks")

        # Test each template
        results = {}

        for name, template_func in templates.items():
            print(f"\n{'-'*70}")
            print(f"Template: {name}")
            print('-'*70)

            result = self.test_template(name, template_func, query, context_chunks)
            results[name] = result

            print(f"\n{result['answer']}")
            print(f"\n📊 Tokens: {result['tokens']} | Prompt length: {result['prompt_length']} chars")

        # Analysis
        print(f"\n{'='*70}")
        print("ANALYSIS")
        print('='*70)

        avg_tokens = sum(r['tokens'] for r in results.values()) / len(results)
        max_tokens = max(r['tokens'] for r in results.values())
        min_tokens = min(r['tokens'] for r in results.values())

        print(f"\n📊 Token Usage:")
        print(f"   Average: {avg_tokens:.0f} tokens")
        print(f"   Range: {min_tokens} - {max_tokens} tokens")

        # Check for citations
        print(f"\n📖 Citation Analysis:")
        for name, result in results.items():
            has_citations = any(f'[{i}]' in result['answer'] for i in range(1, 6))
            print(f"   {name}: {'✅ Has citations' if has_citations else '❌ No citations'}")

        return results


def format_context_with_metadata(chunks: List[str], metadatas: List[Dict] = None) -> str:
    """
    Format context with rich metadata

    Args:
        chunks: List of text chunks
        metadatas: Optional list of metadata dicts

    Returns:
        str: Formatted context with metadata
    """
    formatted = ""

    for i, chunk in enumerate(chunks):
        formatted += f"\n{'='*60}\n"
        formatted += f"CONTEXT SECTION [{i+1}]\n"

        if metadatas and i < len(metadatas):
            meta = metadatas[i]
            if 'topic' in meta:
                formatted += f"Topic: {meta['topic']}\n"
            if 'difficulty' in meta:
                formatted += f"Difficulty: {meta['difficulty']}\n"
            if 'category' in meta:
                formatted += f"Category: {meta['category']}\n"

        formatted += f"{'='*60}\n"
        formatted += chunk + "\n"

    return formatted


def create_custom_template(
    query: str,
    context_chunks: List[str],
    style: str = "professional",
    include_metadata: bool = True,
    metadatas: List[Dict] = None
) -> str:
    """
    Create customized RAG prompt

    Args:
        query: User's question
        context_chunks: Retrieved chunks
        style: Response style (professional, casual, academic)
        include_metadata: Include metadata in context
        metadatas: Metadata for chunks

    Returns:
        str: Customized prompt
    """
    # Format context
    if include_metadata and metadatas:
        context = format_context_with_metadata(context_chunks, metadatas)
    else:
        context = "\n\n".join([f"[{i+1}] {chunk}"
                               for i, chunk in enumerate(context_chunks)])

    # Style-specific instructions
    style_instructions = {
        "professional": "Provide a clear, professional response suitable for business communication.",
        "casual": "Explain in a friendly, easy-to-understand way as if talking to a friend.",
        "academic": "Provide a scholarly, well-structured response with precise terminology."
    }

    instruction = style_instructions.get(style, style_instructions["professional"])

    return f"""Answer the question based on the provided context.

CONTEXT:
{context}

QUESTION:
{query}

STYLE: {instruction}

ANSWER:"""


def main():
    """Demonstrate advanced prompt strategies"""
    print("="*70)
    print("EXERCISE 3: ADVANCED PROMPT STRATEGIES")
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

    # Task 3A: Template comparison
    print("\n" + "="*70)
    print("TASK 3A: COMPARE PROMPT TEMPLATES")
    print("="*70)

    comparator = PromptComparator(collection, embedding_model, openai_api_key)

    test_query = "What is deep learning and how does it work?"
    results = comparator.compare_templates(test_query)

    # Task 3B: Context formatting with metadata
    print("\n" + "="*70)
    print("TASK 3B: CONTEXT FORMATTING WITH METADATA")
    print("="*70)

    query = "Explain machine learning"
    query_embedding = embedding_model.encode([query])

    results_with_meta = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3,
        include=["documents", "metadatas"]
    )

    rich_context = format_context_with_metadata(
        results_with_meta['documents'][0],
        results_with_meta['metadatas'][0]
    )

    print("\n📚 Rich Context Format:")
    print(rich_context[:500] + "...")

    # Task 3C: Custom templates
    print("\n" + "="*70)
    print("TASK 3C: CUSTOM TEMPLATE STYLES")
    print("="*70)

    styles = ["professional", "casual", "academic"]
    openai_client = OpenAI(api_key=openai_api_key)

    for style in styles:
        print(f"\n{'-'*70}")
        print(f"Style: {style.upper()}")
        print('-'*70)

        custom_prompt = create_custom_template(
            query="What is neural networks?",
            context_chunks=results_with_meta['documents'][0],
            style=style,
            include_metadata=True,
            metadatas=results_with_meta['metadatas'][0]
        )

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": custom_prompt}],
            temperature=0.3,
            max_tokens=200
        )

        print(f"\n{response.choices[0].message.content}")

    # Summary
    print("\n" + "="*70)
    print("✅ EXERCISE 3 COMPLETE!")
    print("="*70)

    print("\n💡 KEY TAKEAWAYS:")
    print("   1. Prompt structure significantly affects response quality")
    print("   2. Citations improve with explicit instructions")
    print("   3. Structured templates produce consistent outputs")
    print("   4. Conversational style increases engagement")
    print("   5. Metadata enriches context understanding")

    print("\n🎯 TEMPLATE SELECTION GUIDE:")
    print("   • Basic: Quick prototypes, simple queries")
    print("   • Detailed: Production use, citation needed")
    print("   • Structured: APIs, consistent format required")
    print("   • Conversational: Customer-facing applications")
    print("   • Chain-of-Thought: Complex reasoning tasks")
    print("   • Expert: Domain-specific expertise needed")

    print("\n⚡ OPTIMIZATION TIPS:")
    print("   ✓ Test multiple templates for your use case")
    print("   ✓ Include metadata when it adds value")
    print("   ✓ Use specific instructions for citations")
    print("   ✓ Match style to audience")
    print("   ✓ Keep prompts concise to save tokens")


if __name__ == "__main__":
    main()
