"""
Lab 3 - Capstone: Complete Document Processing System
Production-ready document processor with all Lab 3 features

Features:
- Multi-format document loading (TXT, PDF, MD)
- Multiple chunking strategies
- Embedding generation
- Vector database storage
- Semantic search
- Statistics and monitoring
"""

import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
import re


class DocumentProcessor:
    """
    Complete document processing system

    Capabilities:
    - Load documents from multiple formats
    - Chunk with configurable strategies
    - Generate embeddings
    - Store in vector database
    - Semantic search
    """

    def __init__(
        self,
        embedding_model='all-MiniLM-L6-v2',
        persist_directory="./document_db",
        chunking_strategy='recursive',
        chunk_size=500
    ):
        """
        Initialize document processor

        Args:
            embedding_model: Sentence-transformer model name
            persist_directory: ChromaDB storage path
            chunking_strategy: Chunking method
            chunk_size: Maximum chunk size
        """
        # Initialize components
        self.embedding_model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Chunking settings
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size

        # Statistics
        self.stats = {
            'documents_loaded': 0,
            'chunks_created': 0,
            'embeddings_generated': 0
        }

    def load_document(self, file_path: str) -> Optional[str]:
        """Load document from file"""
        path = Path(file_path)

        if not path.exists():
            print(f"❌ File not found: {file_path}")
            return None

        try:
            if path.suffix.lower() in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

            elif path.suffix.lower() == '.pdf':
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                content = "\n\n".join([page.extract_text() for page in reader.pages])

            else:
                print(f"❌ Unsupported format: {path.suffix}")
                return None

            self.stats['documents_loaded'] += 1
            return content

        except Exception as e:
            print(f"❌ Error loading document: {e}")
            return None

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using configured strategy

        Args:
            text: Input text

        Returns:
            list: Text chunks
        """
        if self.chunking_strategy == 'fixed_size':
            chunks = self._fixed_size_chunking(text)
        elif self.chunking_strategy == 'sentence':
            chunks = self._sentence_chunking(text)
        elif self.chunking_strategy == 'paragraph':
            chunks = self._paragraph_chunking(text)
        elif self.chunking_strategy == 'recursive':
            chunks = self._recursive_chunking(text)
        else:
            chunks = [text]

        self.stats['chunks_created'] += len(chunks)
        return chunks

    def _fixed_size_chunking(self, text: str) -> List[str]:
        """Fixed-size chunking with overlap"""
        chunks = []
        overlap = int(self.chunk_size * 0.2)  # 20% overlap
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap

        return chunks

    def _sentence_chunking(self, text: str) -> List[str]:
        """Sentence-based chunking"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _paragraph_chunking(self, text: str) -> List[str]:
        """Paragraph-based chunking"""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]

    def _recursive_chunking(self, text: str) -> List[str]:
        """Recursive chunking with hierarchy"""
        separators = ['\n\n', '\n', '. ', ' ']

        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        for separator in separators:
            if separator in text:
                splits = text.split(separator)
                chunks = []
                current = ""

                for split in splits:
                    if len(current) + len(split) + len(separator) <= self.chunk_size:
                        current += separator + split if current else split
                    else:
                        if current.strip():
                            chunks.append(current)
                        current = split

                if current.strip():
                    chunks.append(current)

                return chunks

        return [text] if text.strip() else []

    def process_document(
        self,
        file_path: str,
        collection_name: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Complete document processing pipeline

        Args:
            file_path: Path to document
            collection_name: ChromaDB collection name
            metadata: Optional document metadata

        Returns:
            dict: Processing results
        """
        print(f"\n📄 Processing: {file_path}")

        # 1. Load document
        content = self.load_document(file_path)
        if not content:
            return {"success": False, "error": "Failed to load document"}

        print(f"   ✅ Loaded ({len(content)} chars)")

        # 2. Chunk text
        chunks = self.chunk_text(content)
        print(f"   ✅ Created {len(chunks)} chunks")

        # 3. Generate embeddings
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
        self.stats['embeddings_generated'] += len(chunks)
        print(f"   ✅ Generated embeddings")

        # 4. Store in vector database
        collection = self.client.get_or_create_collection(collection_name)

        # Prepare metadatas
        if metadata is None:
            metadata = {}

        metadatas = [
            {**metadata, "chunk_id": i, "source": file_path}
            for i in range(len(chunks))
        ]

        ids = [f"{Path(file_path).stem}_chunk_{i}" for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )

        print(f"   ✅ Stored in database")

        return {
            "success": True,
            "file": file_path,
            "chunks": len(chunks),
            "embeddings": len(embeddings),
            "collection": collection_name
        }

    def search(
        self,
        query: str,
        collection_name: str,
        n_results: int = 5
    ) -> Dict:
        """
        Search for similar chunks

        Args:
            query: Search query
            collection_name: Collection to search
            n_results: Number of results

        Returns:
            dict: Search results
        """
        try:
            collection = self.client.get_collection(collection_name)
        except:
            return {"success": False, "error": f"Collection '{collection_name}' not found"}

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)

        # Search
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

        return {
            "success": True,
            "query": query,
            "results": [
                {
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
        }

    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return self.stats.copy()


# ============================================================================
# DEMO
# ============================================================================

def main():
    """Demonstrate complete document processing system"""
    print("=" * 70)
    print("CAPSTONE: COMPLETE DOCUMENT PROCESSING SYSTEM")
    print("=" * 70)

    # Initialize processor
    processor = DocumentProcessor(
        embedding_model='all-MiniLM-L6-v2',
        persist_directory="./capstone_db",
        chunking_strategy='recursive',
        chunk_size=300
    )

    print("\n✅ Document Processor initialized")
    print(f"   Embedding model: all-MiniLM-L6-v2")
    print(f"   Chunking: recursive (max 300 chars)")

    # Process sample document
    print("\n" + "=" * 70)
    print("STEP 1: PROCESS DOCUMENT")
    print("=" * 70)

    result = processor.process_document(
        file_path='sample_document.txt',
        collection_name='ai_knowledge',
        metadata={"category": "AI", "source_type": "article"}
    )

    if result['success']:
        print(f"\n✅ Processing complete!")
        print(f"   Chunks: {result['chunks']}")
        print(f"   Collection: {result['collection']}")

    # Search the database
    print("\n" + "=" * 70)
    print("STEP 2: SEMANTIC SEARCH")
    print("=" * 70)

    queries = [
        "neural networks and deep learning",
        "natural language understanding",
        "ethics and AI safety"
    ]

    for query in queries:
        print(f"\n📝 Query: '{query}'")

        search_result = processor.search(
            query=query,
            collection_name='ai_knowledge',
            n_results=2
        )

        if search_result['success']:
            print(f"\n🏆 Top {len(search_result['results'])} Results:")
            for i, res in enumerate(search_result['results'], 1):
                print(f"\n{i}. Distance: {res['distance']:.4f}")
                print(f"   Category: {res['metadata'].get('category', 'N/A')}")
                print(f"   Text: {res['text'][:100]}...")

    # Show statistics
    print("\n" + "=" * 70)
    print("STEP 3: STATISTICS")
    print("=" * 70)

    stats = processor.get_stats()
    print("\n📊 Processing Statistics:")
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")

    print("\n" + "=" * 70)
    print("✅ CAPSTONE COMPLETE!")
    print("=" * 70)

    print("\n🎯 WHAT WE BUILT:")
    print("   ✅ Document loading (TXT, PDF, MD)")
    print("   ✅ Intelligent chunking (4 strategies)")
    print("   ✅ Embedding generation")
    print("   ✅ Vector database storage")
    print("   ✅ Semantic search")
    print("   ✅ Statistics tracking")

    print("\n🔮 READY FOR:")
    print("   → Lab 4: Semantic Search & Retrieval")
    print("   → Lab 5: Complete RAG Pipeline")
    print("   → Production deployment")


if __name__ == "__main__":
    main()
