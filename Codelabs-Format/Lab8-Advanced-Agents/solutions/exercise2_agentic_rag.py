"""
Lab 8 - Exercise 2: Agentic RAG System
Solution for building RAG system where agent decides when to retrieve

Learning Objectives:
- Implement agentic RAG with dynamic retrieval decisions
- Build agent that decides when retrieval is needed
- Implement multi-hop reasoning
- Track retrieval decisions
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# ==================== KNOWLEDGE BASE SETUP ====================

class KnowledgeBase:
    """Vector database for Agentic RAG"""

    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./agentic_rag_db")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.collection = self.chroma_client.get_or_create_collection(
            name="agentic_rag_docs"
        )

        # Populate if empty
        if self.collection.count() == 0:
            self._populate()

    def _populate(self):
        """Populate with sample documents"""
        documents = [
            {
                "text": "Machine learning is a subset of AI that enables systems to learn from data without explicit programming. Key types include supervised, unsupervised, and reinforcement learning.",
                "metadata": {"topic": "machine_learning", "category": "basics"}
            },
            {
                "text": "Deep learning uses neural networks with multiple layers to automatically learn hierarchical representations. It excels at tasks like image recognition and natural language processing.",
                "metadata": {"topic": "deep_learning", "category": "advanced"}
            },
            {
                "text": "Supervised learning trains models on labeled data, learning to map inputs to outputs. Common applications include classification and regression tasks.",
                "metadata": {"topic": "supervised_learning", "category": "techniques"}
            },
            {
                "text": "Unsupervised learning discovers patterns in unlabeled data through techniques like clustering and dimensionality reduction.",
                "metadata": {"topic": "unsupervised_learning", "category": "techniques"}
            },
            {
                "text": "Natural language processing enables computers to understand and generate human language. Applications include translation, sentiment analysis, and chatbots.",
                "metadata": {"topic": "nlp", "category": "applications"}
            }
        ]

        for i, doc in enumerate(documents):
            embedding = self.embedding_model.encode(doc["text"])
            self.collection.add(
                documents=[doc["text"]],
                embeddings=[embedding.tolist()],
                ids=[f"doc_{i}"],
                metadatas=[doc["metadata"]]
            )

        print(f"✓ Knowledge base populated with {len(documents)} documents")


# ==================== AGENTIC RAG ====================

class AgenticRAG:
    """RAG system where agent decides when to retrieve"""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.retrieval_log = []

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_documents",
                    "description": "Retrieve relevant documents from the knowledge base. Use ONLY when you need specific information you don't know.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for documents"
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Number of documents to retrieve (default: 2)",
                                "default": 2
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        self.functions = {
            "retrieve_documents": self.retrieve_documents
        }

    def retrieve_documents(self, query: str, n_results: int = 2) -> Dict:
        """Retrieve documents from knowledge base"""
        print(f"\n📚 Retrieving: '{query}'")

        query_embedding = self.embedding_model.encode(query)

        results = self.kb.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        documents = []
        for i in range(len(results['documents'][0])):
            doc = {
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "relevance": 1 / (1 + results['distances'][0][i])
            }
            documents.append(doc)
            print(f"   [{i+1}] {doc['metadata'].get('topic')} (relevance: {doc['relevance']:.3f})")

        self.retrieval_log.append({
            "query": query,
            "num_docs": len(documents)
        })

        return {
            "success": True,
            "query": query,
            "documents": documents,
            "count": len(documents)
        }

    def answer_question(self, question: str) -> Dict:
        """Answer question using agentic RAG"""
        print(f"\n{'='*70}")
        print(f"AGENTIC RAG")
        print('='*70)
        print(f"Question: {question}\n")

        system_prompt = """You are an intelligent assistant with access to a knowledge base about machine learning and AI.

CRITICAL INSTRUCTIONS:
- If you already know the answer from general knowledge, answer directly WITHOUT retrieving
- ONLY use retrieve_documents when you need specific information from the knowledge base
- You can retrieve multiple times if needed
- Think step-by-step about whether retrieval is necessary

Examples:
- "What is 2+2?" → NO retrieval needed, answer directly
- "What is machine learning?" → Could answer from general knowledge OR retrieve for specifics
- "According to the knowledge base, what is deep learning?" → MUST retrieve"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        iteration = 0
        max_iterations = 5

        while iteration < max_iterations:
            iteration += 1
            print(f"--- Iteration {iteration} ---")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            # Check if done
            if not response_message.tool_calls:
                answer = response_message.content

                print(f"\n{'='*70}")
                print("✅ ANSWER:")
                print('='*70)
                print(answer)

                return {
                    "question": question,
                    "answer": answer,
                    "retrievals": len(self.retrieval_log),
                    "retrieval_log": self.retrieval_log
                }

            # Show agent reasoning
            if response_message.content:
                print(f"\n💭 Agent: {response_message.content}")

            messages.append(response_message)

            # Execute retrievals
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                result = self.functions[function_name](**arguments)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(result)
                })

        return {"question": question, "answer": "Max iterations reached"}


# ==================== TEST ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("AGENTIC RAG DEMONSTRATION")
    print("="*70)

    kb = KnowledgeBase()
    agentic_rag = AgenticRAG(kb)

    # Test 1: General knowledge (should NOT retrieve)
    print("\n" + "#"*70)
    print("TEST 1: General Knowledge (No Retrieval Expected)")
    print("#"*70)
    result = agentic_rag.answer_question("What is 25 * 4?")
    print(f"\nRetrievals: {result['retrievals']} (expected: 0)")

    # Test 2: KB-specific question (SHOULD retrieve)
    print("\n" + "#"*70)
    print("TEST 2: Knowledge Base Query (Retrieval Expected)")
    print("#"*70)
    agentic_rag.retrieval_log = []  # Reset log
    result = agentic_rag.answer_question("According to the knowledge base, what is supervised learning?")
    print(f"\nRetrievals: {result['retrievals']} (expected: 1+)")

    # Test 3: Multi-hop question (MULTIPLE retrievals)
    print("\n" + "#"*70)
    print("TEST 3: Multi-Hop Query (Multiple Retrievals Expected)")
    print("#"*70)
    agentic_rag.retrieval_log = []
    result = agentic_rag.answer_question("Compare machine learning and deep learning based on the knowledge base")
    print(f"\nRetrievals: {result['retrievals']} (expected: 2+)")

    print("\n✅ Exercise 2 Complete!")
