#!/usr/bin/env python3

import sys
import os
sys.path.append('/app/src')

from src.services.document_loader import PDFDocumentLoader
from src.services.text_processor import TextProcessor

def test_chunking():
    """Test the chunking process to understand why we have so few chunks."""
    
    print("=== Chunking Analysis Test ===")
    
    # Initialize services
    loader = PDFDocumentLoader()
    processor = TextProcessor()
    
    # Extract text from PDF
    print("Extracting text from PDF...")
    pages = loader.extract_text_from_pdf('/app/data/Bangla_book.pdf')
    
    # Get document stats
    doc_stats = loader.get_document_stats(pages)
    print(f"\n=== DOCUMENT STATISTICS ===")
    print(f"Total pages: {doc_stats['total_pages']}")
    print(f"Total characters: {doc_stats['total_chars']:,}")
    print(f"Average characters per page: {doc_stats['avg_chars_per_page']:,}")
    
    # Analyze pages with low content
    low_content_pages = [p for p in pages if p['char_count'] < 200]
    print(f"\nPages with <200 characters: {len(low_content_pages)}")
    for page in low_content_pages[:5]:
        print(f"  Page {page['page_number']}: {page['char_count']} chars")
    
    # Check pages with no content
    empty_pages = [p for p in pages if p['char_count'] == 0]
    print(f"Pages with no content: {len(empty_pages)}")
    
    # Process chunks
    print(f"\n=== CHUNKING ANALYSIS ===")
    chunks = processor.chunk_documents(pages)
    chunk_stats = processor.get_chunking_stats(chunks)
    
    print(f"Total chunks created: {chunk_stats['total_chunks']}")
    print(f"Average chunk size: {chunk_stats['avg_chunk_size']} characters")
    print(f"Average words per chunk: {chunk_stats['avg_words_per_chunk']}")
    
    # Show sample chunks
    print(f"\n=== SAMPLE CHUNKS ===")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- CHUNK {i+1} ---")
        print(f"Page: {chunk['page_number']}")
        print(f"Characters: {chunk['char_count']}")
        print(f"Words: {chunk['word_count']}")
        print("Text preview:")
        print("-" * 40)
        print(chunk['text'][:300])
        print("-" * 40)
    
    # Analyze chunk distribution
    print(f"\n=== CHUNK DISTRIBUTION ===")
    chunks_by_page = {}
    for chunk in chunks:
        page = chunk['page_number']
        if page not in chunks_by_page:
            chunks_by_page[page] = 0
        chunks_by_page[page] += 1
    
    pages_with_chunks = len(chunks_by_page)
    print(f"Pages that produced chunks: {pages_with_chunks}")
    print(f"Pages without chunks: {doc_stats['total_pages'] - pages_with_chunks}")
    
    # Show pages with most chunks
    top_pages = sorted(chunks_by_page.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 pages by chunk count:")
    for page, count in top_pages:
        print(f"  Page {page}: {count} chunks")
    
    return chunks, chunk_stats

if __name__ == "__main__":
    test_chunking() 