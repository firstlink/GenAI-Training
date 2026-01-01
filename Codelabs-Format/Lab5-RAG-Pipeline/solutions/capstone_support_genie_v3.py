"""
Lab 5 - Capstone: SupportGenie v3.0 with Complete RAG
Production-ready RAG-powered customer support system

Features:
- Complete RAG pipeline (Retrieve → Augment → Generate)
- Multi-LLM support (OpenAI, Claude)
- Conversation history management
- Source citations
- Performance metrics
- Error handling
- Interactive chat mode
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
import chromadb
import time
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

# Load environment variables
load_dotenv()


class SupportGenieV3:
    """
    Production RAG-powered support system
    Version 3.0 - Complete with Retrieval-Augmented Generation

    Capabilities:
    - Semantic search with ChromaDB
    - RAG-powered responses with source citations
    - Multi-LLM support (OpenAI, Claude)
    - Conversation history tracking
    - Performance monitoring
    - Error handling and fallbacks
    """

    SYSTEM_PROMPT = """You are SupportGenie, an AI customer support assistant.

CAPABILITIES:
- Answer questions using provided knowledge base context
- Cite sources for all information
- Admit when information is not available in the knowledge base

RESPONSE GUIDELINES:
1. Use ONLY the provided context to answer
2. Cite context sections used: [1], [2], [3]
3. If context is insufficient, clearly state what information is missing
4. Be professional, helpful, and concise
5. Structure answers clearly with proper formatting

TONE: Professional, helpful, and empathetic"""

    def __init__(
        self,
        collection,
        embedding_model,
        llm_provider: str = "openai",
        model: Optional[str] = None
    ):
        """
        Initialize SupportGenie v3.0

        Args:
            collection: ChromaDB collection
            embedding_model: SentenceTransformer model
            llm_provider: 'openai' or 'claude'
            model: Specific model name (optional)
        """
        self.collection = collection
        self.embedding_model = embedding_model
        self.llm_provider = llm_provider

        # Conversation tracking
        self.conversation_history = []
        self.session_start = time.time()

        # Performance metrics
        self.metrics = {
            'total_queries': 0,
            'total_tokens': 0,
            'total_time': 0.0,
            'successful_queries': 0,
            'failed_queries': 0
        }

        # Initialize LLM client
        if llm_provider == "openai":
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

            self.llm_client = OpenAI(api_key=api_key)
            self.model = model or "gpt-4o-mini"

        elif llm_provider == "claude":
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")

            self.llm_client = Anthropic(api_key=api_key)
            self.model = model or "claude-3-5-haiku-20241022"

        else:
            raise ValueError(f"Unknown provider: {llm_provider}")

        print(f"✅ SupportGenie v3.0 initialized")
        print(f"   LLM Provider: {llm_provider}")
        print(f"   Model: {self.model}")
        print(f"   Knowledge Base: {collection.count()} documents")

    def retrieve_context(
        self,
        query: str,
        n_results: int = 3,
        metadata_filter: Optional[Dict] = None
    ) -> Dict:
        """
        Retrieve relevant context from knowledge base

        Args:
            query: User's question
            n_results: Number of chunks to retrieve
            metadata_filter: Optional metadata filter

        Returns:
            dict: Retrieved documents, distances, and metadata
        """
        try:
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
                'success': True,
                'documents': results['documents'][0],
                'distances': results['distances'][0],
                'metadatas': results['metadatas'][0] if results['metadatas'] else []
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'documents': [],
                'distances': [],
                'metadatas': []
            }

    def create_rag_prompt(self, query: str, context_data: Dict) -> str:
        """
        Create RAG prompt with retrieved context

        Args:
            query: User's question
            context_data: Retrieved context information

        Returns:
            str: Complete RAG prompt
        """
        # Format context with metadata
        context_text = ""
        for i, (doc, meta) in enumerate(zip(
            context_data['documents'],
            context_data['metadatas']
        ), 1):
            context_text += f"\n[{i}] "

            # Add source metadata if available
            if meta and 'source' in meta:
                context_text += f"(Source: {meta['source']}) "
            if meta and 'topic' in meta:
                context_text += f"(Topic: {meta['topic']}) "

            context_text += doc + "\n"

        # Build complete prompt
        prompt = f"""{self.SYSTEM_PROMPT}

KNOWLEDGE BASE CONTEXT:
{context_text}

CUSTOMER QUESTION:
{query}

YOUR RESPONSE:"""

        return prompt

    def generate_response(self, prompt: str) -> Dict:
        """
        Generate response using configured LLM

        Args:
            prompt: Complete RAG prompt

        Returns:
            dict: Generated answer and metadata
        """
        try:
            if self.llm_provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )

                return {
                    'success': True,
                    'answer': response.choices[0].message.content,
                    'tokens': response.usage.total_tokens
                }

            elif self.llm_provider == "claude":
                response = self.llm_client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )

                return {
                    'success': True,
                    'answer': response.content[0].text,
                    'tokens': response.usage.input_tokens + response.usage.output_tokens
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'answer': None,
                'tokens': 0
            }

    def chat(
        self,
        user_message: str,
        n_results: int = 3,
        metadata_filter: Optional[Dict] = None
    ) -> Dict:
        """
        Main chat interface with RAG

        Args:
            user_message: Customer's question
            n_results: Number of context chunks to retrieve
            metadata_filter: Optional metadata filter

        Returns:
            dict: Response with answer and metadata
        """
        start_time = time.time()
        self.metrics['total_queries'] += 1

        try:
            # Step 1: Retrieve context
            context_data = self.retrieve_context(user_message, n_results, metadata_filter)

            if not context_data['success']:
                raise Exception(f"Retrieval failed: {context_data.get('error')}")

            # Step 2: Create RAG prompt
            prompt = self.create_rag_prompt(user_message, context_data)

            # Step 3: Generate response
            generation_result = self.generate_response(prompt)

            if not generation_result['success']:
                raise Exception(f"Generation failed: {generation_result.get('error')}")

            # Calculate metrics
            elapsed_time = time.time() - start_time
            avg_relevance = np.mean(context_data['distances']) if context_data['distances'] else 0.0

            # Update metrics
            self.metrics['successful_queries'] += 1
            self.metrics['total_tokens'] += generation_result['tokens']
            self.metrics['total_time'] += elapsed_time

            # Store in conversation history
            interaction = {
                'timestamp': datetime.now().isoformat(),
                'user_message': user_message,
                'assistant_response': generation_result['answer'],
                'context_used': context_data['documents'],
                'relevance_scores': context_data['distances'],
                'tokens_used': generation_result['tokens'],
                'time_seconds': elapsed_time
            }
            self.conversation_history.append(interaction)

            # Check for citations
            has_citations = any(f'[{i}]' in generation_result['answer'] for i in range(1, 6))

            return {
                'success': True,
                'answer': generation_result['answer'],
                'metadata': {
                    'provider': self.llm_provider,
                    'model': self.model,
                    'retrieved_chunks': len(context_data['documents']),
                    'avg_relevance': avg_relevance,
                    'tokens_used': generation_result['tokens'],
                    'latency_ms': elapsed_time * 1000,
                    'has_citations': has_citations
                },
                'sources': context_data['metadatas'],
                'context': context_data['documents']
            }

        except Exception as e:
            self.metrics['failed_queries'] += 1
            elapsed_time = time.time() - start_time

            return {
                'success': False,
                'error': str(e),
                'answer': "I'm sorry, I encountered an error processing your request. Please try again or rephrase your question.",
                'metadata': {
                    'latency_ms': elapsed_time * 1000,
                    'error_type': type(e).__name__
                }
            }

    def display_response(self, response: Dict):
        """
        Display response in user-friendly format

        Args:
            response: Response dictionary from chat()
        """
        print(f"\n{'='*70}")

        if response['success']:
            print("🤖 SupportGenie:")
            print('-'*70)
            print(response['answer'])

            # Metadata
            print(f"\n{'─'*70}")
            print("📊 Response Metadata:")
            meta = response['metadata']
            print(f"   Provider: {meta.get('provider', 'N/A')}")
            print(f"   Model: {meta.get('model', 'N/A')}")
            print(f"   Chunks retrieved: {meta.get('retrieved_chunks', 0)}")
            print(f"   Avg relevance: {meta.get('avg_relevance', 0):.4f}")
            print(f"   Tokens used: {meta.get('tokens_used', 0)}")
            print(f"   Latency: {meta.get('latency_ms', 0):.0f}ms")
            print(f"   Citations: {'✅ Yes' if meta.get('has_citations') else '❌ No'}")

            # Sources
            if response.get('sources'):
                print(f"\n📚 Knowledge Base Sources:")
                unique_sources = set()
                for source in response['sources']:
                    if source and 'topic' in source:
                        unique_sources.add(source['topic'])

                for topic in unique_sources:
                    print(f"   • Topic: {topic}")

        else:
            print("❌ Error:")
            print('-'*70)
            print(response['answer'])
            if 'error' in response:
                print(f"\n⚠️  Technical details: {response['error']}")

        print('='*70)

    def interactive_mode(self):
        """
        Interactive chat mode for testing

        Commands:
        - Regular text: Ask a question
        - :history - Show conversation history
        - :stats - Show usage statistics
        - :quit - Exit
        """
        print("\n" + "="*70)
        print("🤖 SUPPORTGENIE V3.0 - RAG-Powered Support")
        print("="*70)
        print("\n📖 Commands:")
        print("   Type your question to chat")
        print("   :history - Show conversation history")
        print("   :stats - Show usage statistics")
        print("   :clear - Clear conversation history")
        print("   :quit - Exit")
        print("\n" + "-"*70)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in [':quit', ':exit', ':q']:
                    print("\n👋 Thank you for using SupportGenie v3.0!")
                    break

                elif user_input.lower() == ':history':
                    self.show_history()
                    continue

                elif user_input.lower() == ':stats':
                    self.show_stats()
                    continue

                elif user_input.lower() == ':clear':
                    self.conversation_history = []
                    print("\n✅ Conversation history cleared")
                    continue

                # Process query
                response = self.chat(user_input)
                self.display_response(response)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break

            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")

    def show_history(self):
        """Display conversation history"""
        print(f"\n{'='*70}")
        print(f"📜 CONVERSATION HISTORY ({len(self.conversation_history)} interactions)")
        print('='*70)

        if not self.conversation_history:
            print("\nNo conversation history yet.")
            return

        for i, interaction in enumerate(self.conversation_history, 1):
            print(f"\n[{i}] {interaction['timestamp']}")
            print(f"👤 User: {interaction['user_message']}")
            print(f"🤖 Bot: {interaction['assistant_response'][:100]}...")
            print(f"📊 Tokens: {interaction['tokens_used']} | Time: {interaction['time_seconds']:.2f}s")

    def show_stats(self):
        """Display usage statistics"""
        print(f"\n{'='*70}")
        print("📊 USAGE STATISTICS")
        print('='*70)

        if self.metrics['total_queries'] == 0:
            print("\nNo interactions yet.")
            return

        # Session duration
        session_duration = time.time() - self.session_start

        print(f"\n🕐 Session Duration: {session_duration:.0f}s")
        print(f"\n📈 Query Metrics:")
        print(f"   Total queries: {self.metrics['total_queries']}")
        print(f"   Successful: {self.metrics['successful_queries']}")
        print(f"   Failed: {self.metrics['failed_queries']}")
        print(f"   Success rate: {(self.metrics['successful_queries']/self.metrics['total_queries']*100):.1f}%")

        print(f"\n⚡ Performance:")
        print(f"   Total tokens: {self.metrics['total_tokens']:,}")
        print(f"   Avg tokens/query: {self.metrics['total_tokens']/max(self.metrics['successful_queries'], 1):.0f}")
        print(f"   Avg latency: {(self.metrics['total_time']/max(self.metrics['successful_queries'], 1)*1000):.0f}ms")

        print(f"\n🤖 Configuration:")
        print(f"   Provider: {self.llm_provider}")
        print(f"   Model: {self.model}")
        print(f"   Knowledge base size: {self.collection.count()} documents")

        # Conversation insights
        if self.conversation_history:
            avg_chunks = np.mean([
                len(interaction['context_used'])
                for interaction in self.conversation_history
            ])

            avg_relevance = np.mean([
                np.mean(interaction['relevance_scores'])
                for interaction in self.conversation_history
                if interaction['relevance_scores']
            ])

            print(f"\n📚 Retrieval Quality:")
            print(f"   Avg chunks retrieved: {avg_chunks:.1f}")
            print(f"   Avg relevance score: {avg_relevance:.4f}")

        print('='*70)

    def export_conversation(self, filename: str = "conversation_history.txt"):
        """
        Export conversation history to file

        Args:
            filename: Output filename
        """
        if not self.conversation_history:
            print("No conversation history to export.")
            return

        with open(filename, 'w') as f:
            f.write("SupportGenie v3.0 - Conversation History\n")
            f.write("="*70 + "\n\n")

            for i, interaction in enumerate(self.conversation_history, 1):
                f.write(f"[{i}] {interaction['timestamp']}\n")
                f.write(f"User: {interaction['user_message']}\n")
                f.write(f"Assistant: {interaction['assistant_response']}\n")
                f.write(f"Tokens: {interaction['tokens_used']} | ")
                f.write(f"Time: {interaction['time_seconds']:.2f}s\n")
                f.write("-"*70 + "\n\n")

        print(f"\n✅ Conversation exported to {filename}")


def setup_support_knowledge_base():
    """Create sample support knowledge base for SupportGenie"""
    print("\n📚 Setting up support knowledge base...")

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./supportgenie_db")

    # Create collection
    try:
        client.delete_collection("support_docs")
    except:
        pass

    collection = client.create_collection("support_docs")

    # Sample support documents
    documents = [
        "Our product offers a 30-day money-back guarantee. If you're not satisfied with your purchase, you can request a full refund within 30 days of purchase. No questions asked.",

        "To reset your password, click on the 'Forgot Password' link on the login page. Enter your email address, and we'll send you a password reset link. The link expires after 24 hours for security reasons.",

        "We offer three pricing tiers: Basic ($9.99/month), Professional ($29.99/month), and Enterprise ($99.99/month). All plans include core features, with Professional adding advanced analytics and Enterprise providing priority support and custom integrations.",

        "Shipping typically takes 3-5 business days for domestic orders and 7-14 business days for international orders. You'll receive a tracking number via email once your order ships.",

        "Our customer support team is available Monday through Friday, 9 AM to 5 PM EST. For urgent issues outside business hours, you can submit a ticket through our help center, and we'll respond within 24 hours.",

        "To cancel your subscription, go to Account Settings > Billing > Cancel Subscription. Your access will continue until the end of your current billing period. We don't offer pro-rated refunds for partial months.",

        "You can upgrade or downgrade your plan at any time. Changes take effect immediately, and we'll pro-rate the cost difference for the current billing period. Downgrades will restrict features at the next billing cycle.",

        "We accept major credit cards (Visa, Mastercard, American Express), PayPal, and wire transfers for Enterprise customers. All payments are processed securely through our PCI-compliant payment gateway.",

        "Technical support includes email support for all tiers, live chat for Professional and Enterprise, and dedicated account manager for Enterprise customers. Average response time is under 4 hours for Professional tier.",

        "Data security is our top priority. We use AES-256 encryption for data at rest and TLS 1.3 for data in transit. We're SOC 2 Type II certified and fully GDPR compliant. Regular third-party security audits are conducted quarterly."
    ]

    metadatas = [
        {"topic": "refund", "category": "billing", "priority": "high"},
        {"topic": "password", "category": "account", "priority": "medium"},
        {"topic": "pricing", "category": "billing", "priority": "high"},
        {"topic": "shipping", "category": "orders", "priority": "medium"},
        {"topic": "support", "category": "general", "priority": "high"},
        {"topic": "cancellation", "category": "billing", "priority": "high"},
        {"topic": "plan_changes", "category": "billing", "priority": "medium"},
        {"topic": "payment", "category": "billing", "priority": "high"},
        {"topic": "support_channels", "category": "general", "priority": "medium"},
        {"topic": "security", "category": "technical", "priority": "high"}
    ]

    # Generate embeddings and add to collection
    embeddings = embedding_model.encode(documents)

    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=[f"support_doc_{i}" for i in range(len(documents))]
    )

    print(f"   ✅ Created knowledge base with {len(documents)} support documents")

    return client, collection, embedding_model


def main():
    """Demonstrate SupportGenie v3.0"""
    print("="*70)
    print("CAPSTONE: SUPPORTGENIE V3.0 WITH RAG")
    print("="*70)

    # Check for API key
    has_openai = bool(os.getenv('OPENAI_API_KEY'))
    has_claude = bool(os.getenv('ANTHROPIC_API_KEY'))

    print("\n🔑 API Key Status:")
    print(f"   OpenAI: {'✅' if has_openai else '❌'}")
    print(f"   Claude: {'✅' if has_claude else '❌'}")

    if not (has_openai or has_claude):
        print("\n❌ Error: No API keys found")
        print("   Please configure at least one provider in .env file")
        return

    # Choose provider
    provider = "openai" if has_openai else "claude"

    # Setup
    print("\n" + "="*70)
    print("SETUP")
    print("="*70)

    client, collection, embedding_model = setup_support_knowledge_base()

    # Initialize SupportGenie
    print("\n" + "="*70)
    print("INITIALIZE SUPPORTGENIE V3.0")
    print("="*70)

    genie = SupportGenieV3(
        collection=collection,
        embedding_model=embedding_model,
        llm_provider=provider
    )

    # Test queries
    print("\n" + "="*70)
    print("TESTING SUPPORTGENIE V3.0")
    print("="*70)

    test_queries = [
        "How do I reset my password?",
        "What's your refund policy?",
        "What pricing plans do you offer?",
        "How can I contact support?",
        "Is my data secure?"
    ]

    for query in test_queries:
        print(f"\n👤 User: {query}")
        response = genie.chat(query)
        genie.display_response(response)
        print("\n" + "-"*70)

    # Show statistics
    genie.show_stats()

    # Interactive mode prompt
    print("\n" + "="*70)
    print("💡 TIP: Run genie.interactive_mode() for interactive chat!")
    print("="*70)

    # Summary
    print("\n" + "="*70)
    print("✅ CAPSTONE COMPLETE!")
    print("="*70)

    print("\n🎯 WHAT WE BUILT:")
    print("   ✅ Complete RAG pipeline (Retrieve → Augment → Generate)")
    print("   ✅ Multi-LLM support (OpenAI, Claude)")
    print("   ✅ Source citations in responses")
    print("   ✅ Conversation history tracking")
    print("   ✅ Performance metrics and monitoring")
    print("   ✅ Error handling and fallbacks")
    print("   ✅ Interactive chat mode")

    print("\n🚀 PRODUCTION FEATURES:")
    print("   → Knowledge base with 10 support documents")
    print("   → Semantic search for context retrieval")
    print("   → Citation-backed responses")
    print("   → Real-time performance tracking")
    print("   → Conversation export capability")

    print("\n📊 NEXT STEPS:")
    print("   1. Expand knowledge base with more documents")
    print("   2. Implement user feedback collection")
    print("   3. Add conversation summarization")
    print("   4. Deploy as web service API")
    print("   5. Integrate with ticketing system")

    # Uncomment to run interactive mode
    # genie.interactive_mode()


if __name__ == "__main__":
    main()
