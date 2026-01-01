"""
Lab 3 - Exercise 1: Document Loading
Solution for loading documents from different file formats

Learning Objectives:
- Load text files with proper encoding
- Extract text from PDF files
- Handle file reading errors
- Build a document loader class
"""

import os
from pathlib import Path


def load_text_file(file_path):
    """
    Load a text file and return its content

    Args:
        file_path: Path to the text file

    Returns:
        str: File content or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found")
        return None
    except UnicodeDecodeError:
        print(f"❌ Error: Unable to decode {file_path}. Try different encoding.")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None


def load_pdf_file(file_path):
    """
    Load a PDF file and extract text content

    Args:
        file_path: Path to the PDF file

    Returns:
        str: Extracted text content or None if error
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        text_content = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            text_content.append(text)

        return "\n\n".join(text_content)

    except ImportError:
        print("❌ Error: pypdf not installed. Run: pip install pypdf")
        return None
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return None


class DocumentLoader:
    """
    Document loader that supports multiple file formats

    Supported formats:
    - .txt (plain text)
    - .pdf (PDF documents)
    - .md (Markdown)
    """

    SUPPORTED_FORMATS = {
        '.txt': 'text',
        '.md': 'text',
        '.pdf': 'pdf'
    }

    def __init__(self):
        self.loaded_documents = []

    def load(self, file_path):
        """
        Load a document from the given file path

        Args:
            file_path: Path to the document

        Returns:
            dict: Document info with content and metadata
        """
        path = Path(file_path)

        if not path.exists():
            print(f"❌ File not found: {file_path}")
            return None

        # Get file extension
        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_FORMATS:
            print(f"❌ Unsupported file format: {extension}")
            print(f"   Supported: {list(self.SUPPORTED_FORMATS.keys())}")
            return None

        # Load based on format
        format_type = self.SUPPORTED_FORMATS[extension]

        if format_type == 'text':
            content = load_text_file(file_path)
        elif format_type == 'pdf':
            content = load_pdf_file(file_path)
        else:
            print(f"❌ Unknown format type: {format_type}")
            return None

        if content is None:
            return None

        # Create document object
        document = {
            'file_path': str(path),
            'file_name': path.name,
            'file_type': extension,
            'content': content,
            'length': len(content),
            'word_count': len(content.split())
        }

        self.loaded_documents.append(document)

        return document

    def get_stats(self):
        """Get statistics about loaded documents"""
        if not self.loaded_documents:
            return "No documents loaded"

        total_chars = sum(doc['length'] for doc in self.loaded_documents)
        total_words = sum(doc['word_count'] for doc in self.loaded_documents)

        return {
            'total_documents': len(self.loaded_documents),
            'total_characters': total_chars,
            'total_words': total_words,
            'file_types': list(set(doc['file_type'] for doc in self.loaded_documents))
        }


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run all demonstrations"""
    print("=" * 70)
    print("EXERCISE 1: DOCUMENT LOADING")
    print("=" * 70)

    # Task 1A: Load Text File
    print("\n📄 TASK 1A: LOAD TEXT FILE")
    print("=" * 70)

    document_path = 'sample_document.txt'
    content = load_text_file(document_path)

    if content:
        print(f"✅ Loaded: {document_path}")
        print(f"📊 Length: {len(content)} characters")
        print(f"📊 Words: {len(content.split())} words")
        print(f"\n📝 First 200 characters:")
        print(content[:200] + "...")
    else:
        print("❌ Failed to load document")

    # Task 1B: Document Loader Class
    print("\n\n📚 TASK 1B: DOCUMENT LOADER CLASS")
    print("=" * 70)

    loader = DocumentLoader()

    # Load the sample document
    doc = loader.load('sample_document.txt')

    if doc:
        print(f"\n✅ Document loaded successfully!")
        print(f"\n📋 Document Info:")
        print(f"   File name: {doc['file_name']}")
        print(f"   File type: {doc['file_type']}")
        print(f"   Characters: {doc['length']:,}")
        print(f"   Words: {doc['word_count']:,}")

    # Get loader statistics
    print(f"\n📊 Loader Statistics:")
    stats = loader.get_stats()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")

    # Task 1C: Error Handling
    print("\n\n⚠️  TASK 1C: ERROR HANDLING")
    print("=" * 70)

    print("\nTesting with non-existent file:")
    result = loader.load('nonexistent.txt')
    print(f"Result: {result}")

    print("\nTesting with unsupported format:")
    # Create a dummy file for testing
    with open('test.xlsx', 'w') as f:
        f.write("dummy")
    result = loader.load('test.xlsx')
    print(f"Result: {result}")
    os.remove('test.xlsx')  # Cleanup

    # Key takeaways
    print("\n\n" + "=" * 70)
    print("✅ EXERCISE 1 COMPLETE!")
    print("=" * 70)

    print("\n💡 KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. Always handle file reading errors")
    print("2. Use proper encoding (UTF-8) for text files")
    print("3. Different file formats need different loaders")
    print("4. Store metadata along with content")
    print("5. Build reusable loader classes")
    print("6. Track statistics for monitoring")

    print("\n📖 SUPPORTED FILE FORMATS:")
    print("=" * 70)
    print("✅ .txt - Plain text files")
    print("✅ .md - Markdown files")
    print("✅ .pdf - PDF documents (requires pypdf)")

    print("\n🎯 NEXT STEPS:")
    print("=" * 70)
    print("→ Exercise 2: Text Chunking Strategies")
    print("→ Exercise 3: Generate Embeddings")
    print("→ Exercise 4: Semantic Similarity")
    print("→ Exercise 5: Vector Database (ChromaDB)")


if __name__ == "__main__":
    main()
