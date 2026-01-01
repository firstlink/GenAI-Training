"""
Lab 3 - Exercise 3: Generate Embeddings
Solution for generating text embeddings

Learning Objectives:
- Understand what embeddings are
- Generate embeddings with sentence-transformers
- Compare different embedding models
- Visualize embedding properties
"""

from sentence_transformers import SentenceTransformer
import numpy as np


def generate_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """
    Generate embeddings for a list of texts

    Args:
        texts: List of text strings
        model_name: Name of the sentence-transformer model

    Returns:
        numpy array: Embeddings matrix
    """
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True)

    return embeddings, model


class EmbeddingGenerator:
    """
    Embedding generator with caching and batch processing

    Features:
    - Multiple model support
    - Batch processing
    - Embedding statistics
    """

    POPULAR_MODELS = {
        'all-MiniLM-L6-v2': {'dims': 384, 'speed': 'fast', 'quality': 'good'},
        'all-mpnet-base-v2': {'dims': 768, 'speed': 'medium', 'quality': 'best'},
        'paraphrase-MiniLM-L6-v2': {'dims': 384, 'speed': 'fast', 'quality': 'good'}
    }

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize with embedding model"""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_cache = {}

    def encode(self, texts, batch_size=32, show_progress=False):
        """
        Encode texts to embeddings

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            numpy array: Embeddings
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )

        return embeddings

    def get_embedding_stats(self, embeddings):
        """Get statistics about embeddings"""
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        return {
            'num_embeddings': embeddings.shape[0],
            'dimensions': embeddings.shape[1],
            'mean': float(np.mean(embeddings)),
            'std': float(np.std(embeddings)),
            'min': float(np.min(embeddings)),
            'max': float(np.max(embeddings))
        }


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run all demonstrations"""
    print("=" * 70)
    print("EXERCISE 3: GENERATE EMBEDDINGS")
    print("=" * 70)

    # Task 3A: Generate Simple Embeddings
    print("\n🔢 TASK 3A: GENERATE SIMPLE EMBEDDINGS")
    print("=" * 70)

    texts = [
        "Artificial intelligence is transforming technology",
        "Machine learning enables computers to learn from data",
        "The weather is sunny today"
    ]

    embeddings, model = generate_embeddings(texts)

    print(f"\n✅ Generated embeddings!")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dimensions: {embeddings.shape[1]}")
    print(f"\nFirst embedding (first 10 values):")
    print(f"   {embeddings[0][:10]}")

    # Task 3B: Embedding Generator Class
    print("\n\n🏭 TASK 3B: EMBEDDING GENERATOR CLASS")
    print("=" * 70)

    generator = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')

    # Encode texts
    test_texts = [
        "Deep learning uses neural networks",
        "Natural language processing handles text",
        "Computer vision analyzes images"
    ]

    embeddings = generator.encode(test_texts, show_progress=True)

    print(f"\n✅ Embeddings generated!")

    # Get statistics
    stats = generator.get_embedding_stats(embeddings)
    print(f"\n📊 Embedding Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key.title()}: {value:.4f}")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value}")

    # Task 3C: Compare Models
    print("\n\n⚖️  TASK 3C: COMPARE EMBEDDING MODELS")
    print("=" * 70)

    test_text = "Machine learning is a subset of artificial intelligence"

    models_to_test = [
        'all-MiniLM-L6-v2',      # Fast, 384 dims
        'all-mpnet-base-v2'       # Best quality, 768 dims
    ]

    print(f"\n📝 Test text: '{test_text}'\n")

    for model_name in models_to_test:
        print(f"Model: {model_name}")

        gen = EmbeddingGenerator(model_name=model_name)
        emb = gen.encode(test_text)
        stats = gen.get_embedding_stats(emb)

        print(f"   Dimensions: {stats['dimensions']}")
        print(f"   Mean: {stats['mean']:.4f}")
        print(f"   Std: {stats['std']:.4f}")
        print()

    # Task 3D: Batch Processing
    print("\n📦 TASK 3D: BATCH PROCESSING")
    print("=" * 70)

    # Load chunks from previous exercise
    from exercise2_text_chunking import fixed_size_chunking

    with open('sample_document.txt', 'r') as f:
        doc_text = f.read()

    chunks = fixed_size_chunking(doc_text, chunk_size=200, overlap=50)

    print(f"\n📄 Processing {len(chunks)} chunks...")

    generator = EmbeddingGenerator()
    chunk_embeddings = generator.encode(chunks, batch_size=8, show_progress=True)

    print(f"\n✅ Generated embeddings for all chunks!")
    print(f"   Shape: {chunk_embeddings.shape}")
    print(f"   Total vectors: {chunk_embeddings.shape[0]}")
    print(f"   Vector dimensions: {chunk_embeddings.shape[1]}")

    # Key takeaways
    print("\n\n" + "=" * 70)
    print("✅ EXERCISE 3 COMPLETE!")
    print("=" * 70)

    print("\n💡 KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. Embeddings convert text to numerical vectors")
    print("2. Similar texts have similar embeddings")
    print("3. Embedding dimension affects quality and speed")
    print("4. all-MiniLM-L6-v2 is fast and good quality")
    print("5. all-mpnet-base-v2 is best quality but slower")
    print("6. Batch processing is efficient for many texts")
    print("7. Embeddings enable semantic search")

    print("\n📖 POPULAR MODELS:")
    print("=" * 70)
    for model, info in EmbeddingGenerator.POPULAR_MODELS.items():
        print(f"{model}:")
        print(f"   Dimensions: {info['dims']}")
        print(f"   Speed: {info['speed']}")
        print(f"   Quality: {info['quality']}")
        print()


if __name__ == "__main__":
    main()
