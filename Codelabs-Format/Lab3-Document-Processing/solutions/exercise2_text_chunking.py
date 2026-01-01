"""
Lab 3 - Exercise 2: Text Chunking Strategies
Solution for splitting documents into chunks

Learning Objectives:
- Understand why chunking is necessary
- Implement fixed-size chunking
- Implement sentence-based chunking
- Implement paragraph-based chunking
- Implement recursive chunking
- Compare different strategies
"""

import re


def fixed_size_chunking(text, chunk_size=200, overlap=50):
    """
    Split text into fixed-size chunks with overlap

    Args:
        text: Input text
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters

    Returns:
        list: List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Don't include empty chunks
        if chunk.strip():
            chunks.append(chunk)

        # Move start position (accounting for overlap)
        start = end - overlap

    return chunks


def sentence_chunking(text, sentences_per_chunk=3):
    """
    Split text by sentences

    Args:
        text: Input text
        sentences_per_chunk: Number of sentences per chunk

    Returns:
        list: List of sentence-based chunks
    """
    # Simple sentence splitting (handles . ! ?)
    sentences = re.split(r'[.!?]+', text)

    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    # Group into chunks
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunk = '. '.join(chunk_sentences)
        if chunk:
            chunks.append(chunk + '.')

    return chunks


def paragraph_chunking(text):
    """
    Split text by paragraphs

    Args:
        text: Input text

    Returns:
        list: List of paragraphs
    """
    # Split on double newlines (paragraph breaks)
    paragraphs = text.split('\n\n')

    # Clean up paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


def recursive_chunking(text, max_chunk_size=500, separators=None):
    """
    Recursively split text using a hierarchy of separators

    Args:
        text: Input text
        max_chunk_size: Maximum size of chunks
        separators: List of separators in priority order

    Returns:
        list: List of recursively split chunks
    """
    if separators is None:
        separators = ['\n\n', '\n', '. ', ' ']

    # Base case: text is small enough
    if len(text) <= max_chunk_size:
        return [text] if text.strip() else []

    # Try each separator
    for separator in separators:
        if separator in text:
            # Split by this separator
            splits = text.split(separator)

            chunks = []
            current_chunk = ""

            for split in splits:
                # If adding this split keeps us under max size
                if len(current_chunk) + len(split) + len(separator) <= max_chunk_size:
                    if current_chunk:
                        current_chunk += separator + split
                    else:
                        current_chunk = split
                else:
                    # Save current chunk if not empty
                    if current_chunk.strip():
                        chunks.append(current_chunk)

                    # Start new chunk with this split
                    current_chunk = split

            # Don't forget the last chunk
            if current_chunk.strip():
                chunks.append(current_chunk)

            return chunks

    # If no separator found, just return the text
    return [text] if text.strip() else []


class TextChunker:
    """
    Text chunking with multiple strategies

    Strategies:
    - fixed_size: Fixed character length with overlap
    - sentence: Group by sentences
    - paragraph: Split on paragraph breaks
    - recursive: Hierarchical splitting
    """

    STRATEGIES = ['fixed_size', 'sentence', 'paragraph', 'recursive']

    def __init__(self, strategy='fixed_size', **kwargs):
        """
        Initialize chunker with strategy

        Args:
            strategy: Chunking strategy to use
            **kwargs: Strategy-specific parameters
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Strategy must be one of {self.STRATEGIES}")

        self.strategy = strategy
        self.params = kwargs

    def chunk(self, text):
        """
        Chunk text using selected strategy

        Args:
            text: Input text

        Returns:
            list: Text chunks
        """
        if self.strategy == 'fixed_size':
            return fixed_size_chunking(
                text,
                chunk_size=self.params.get('chunk_size', 200),
                overlap=self.params.get('overlap', 50)
            )

        elif self.strategy == 'sentence':
            return sentence_chunking(
                text,
                sentences_per_chunk=self.params.get('sentences_per_chunk', 3)
            )

        elif self.strategy == 'paragraph':
            return paragraph_chunking(text)

        elif self.strategy == 'recursive':
            return recursive_chunking(
                text,
                max_chunk_size=self.params.get('max_chunk_size', 500),
                separators=self.params.get('separators', None)
            )

    def analyze_chunks(self, chunks):
        """Analyze chunk statistics"""
        if not chunks:
            return {}

        lengths = [len(chunk) for chunk in chunks]

        return {
            'num_chunks': len(chunks),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'total_chars': sum(lengths)
        }


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run all demonstrations"""
    print("=" * 70)
    print("EXERCISE 2: TEXT CHUNKING STRATEGIES")
    print("=" * 70)

    # Load sample document
    with open('sample_document.txt', 'r') as f:
        sample_text = f.read()

    print(f"\n📄 Sample Document:")
    print(f"   Length: {len(sample_text):,} characters")
    print(f"   Words: {len(sample_text.split()):,}")

    # Task 2A: Fixed-Size Chunking
    print("\n\n📏 TASK 2A: FIXED-SIZE CHUNKING")
    print("=" * 70)

    fixed_chunks = fixed_size_chunking(sample_text, chunk_size=200, overlap=50)

    print(f"\n✅ Created {len(fixed_chunks)} chunks")
    print(f"\nFirst 2 chunks:")
    for i, chunk in enumerate(fixed_chunks[:2], 1):
        print(f"\n  Chunk {i} ({len(chunk)} chars):")
        print(f"  {chunk[:100]}...")

    # Task 2B: Sentence Chunking
    print("\n\n📝 TASK 2B: SENTENCE-BASED CHUNKING")
    print("=" * 70)

    sentence_chunks = sentence_chunking(sample_text, sentences_per_chunk=2)

    print(f"\n✅ Created {len(sentence_chunks)} chunks")
    print(f"\nFirst 2 chunks:")
    for i, chunk in enumerate(sentence_chunks[:2], 1):
        print(f"\n  Chunk {i}:")
        print(f"  {chunk}")

    # Task 2C: Paragraph Chunking
    print("\n\n📄 TASK 2C: PARAGRAPH-BASED CHUNKING")
    print("=" * 70)

    para_chunks = paragraph_chunking(sample_text)

    print(f"\n✅ Created {len(para_chunks)} chunks (paragraphs)")
    print(f"\nFirst paragraph:")
    print(f"  {para_chunks[0]}")

    # Task 2D: Recursive Chunking
    print("\n\n🔄 TASK 2D: RECURSIVE CHUNKING")
    print("=" * 70)

    recursive_chunks = recursive_chunking(sample_text, max_chunk_size=300)

    print(f"\n✅ Created {len(recursive_chunks)} chunks")
    print(f"\nFirst 2 chunks:")
    for i, chunk in enumerate(recursive_chunks[:2], 1):
        print(f"\n  Chunk {i} ({len(chunk)} chars):")
        print(f"  {chunk[:150]}...")

    # Task 2E: Strategy Comparison
    print("\n\n📊 TASK 2E: STRATEGY COMPARISON")
    print("=" * 70)

    strategies = [
        ('Fixed Size (200 chars)', fixed_chunks),
        ('Sentence (2 per chunk)', sentence_chunks),
        ('Paragraph', para_chunks),
        ('Recursive (300 max)', recursive_chunks)
    ]

    print("\nComparison:")
    print(f"{'Strategy':<30} {'Chunks':<10} {'Avg Length':<15} {'Min':<8} {'Max':<8}")
    print("-" * 75)

    for name, chunks in strategies:
        if chunks:
            lengths = [len(c) for c in chunks]
            avg = sum(lengths) / len(lengths)
            print(f"{name:<30} {len(chunks):<10} {avg:<15.1f} {min(lengths):<8} {max(lengths):<8}")

    # Task 2F: TextChunker Class
    print("\n\n🔧 TASK 2F: TEXTCHUNKER CLASS")
    print("=" * 70)

    # Test different strategies
    chunker_fixed = TextChunker(strategy='fixed_size', chunk_size=250, overlap=50)
    chunks = chunker_fixed.chunk(sample_text)
    stats = chunker_fixed.analyze_chunks(chunks)

    print(f"\n✅ TextChunker with 'fixed_size' strategy:")
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value:.1f}" if isinstance(value, float) else f"   {key.replace('_', ' ').title()}: {value}")

    # Key takeaways
    print("\n\n" + "=" * 70)
    print("✅ EXERCISE 2 COMPLETE!")
    print("=" * 70)

    print("\n💡 KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. Chunking is essential for LLM context limits")
    print("2. Different strategies suit different use cases")
    print("3. Fixed-size is simple but may break mid-sentence")
    print("4. Sentence-based preserves semantic units")
    print("5. Paragraph-based good for structured documents")
    print("6. Recursive chunking is most flexible")
    print("7. Overlap helps preserve context across chunks")

    print("\n📖 WHEN TO USE EACH STRATEGY:")
    print("=" * 70)
    print("Fixed Size → Simple documents, consistent chunk sizes")
    print("Sentence → Q&A, chatbots, semantic search")
    print("Paragraph → Articles, blog posts, documentation")
    print("Recursive → Complex documents, variable structure")

    print("\n⚙️  CHUNKING PARAMETERS:")
    print("=" * 70)
    print("• Chunk size: Balance between context and precision")
    print("• Overlap: 10-20% of chunk size recommended")
    print("• Smaller chunks → Better precision, less context")
    print("• Larger chunks → More context, less precision")


if __name__ == "__main__":
    main()
