#!/usr/bin/env python3

import sys
import os
sys.path.append('/app/src')

from src.services.document_loader import PDFDocumentLoader

def test_ocr_quality():
    """Test OCR quality and text extraction from the PDF."""
    
    print("=== OCR Quality Test ===")
    
    # Initialize the document loader
    loader = PDFDocumentLoader()
    
    # Extract text from the PDF
    print("Extracting text from PDF...")
    pages = loader.extract_text_from_pdf('/app/data/Bangla_book.pdf')
    
    # Get document statistics
    stats = loader.get_document_stats(pages)
    
    print(f"\n=== DOCUMENT STATISTICS ===")
    print(f"Total pages processed: {stats['total_pages']}")
    print(f"Total characters: {stats['total_chars']:,}")
    print(f"Average characters per page: {stats['avg_chars_per_page']:,}")
    print(f"Source file: {stats['source_file']}")
    
    # Show sample text from first few pages
    print(f"\n=== SAMPLE OCR TEXT (First 3 pages) ===")
    for i, page in enumerate(pages[:3]):
        print(f"\n--- PAGE {page['page_number']} ---")
        print(f"Characters: {page['char_count']}")
        print("Text preview:")
        print("-" * 50)
        print(page['text'][:1000])
        print("-" * 50)
    
    # Check for potential OCR issues
    print(f"\n=== OCR QUALITY ANALYSIS ===")
    
    # Count pages with very low character count (potential OCR failures)
    low_char_pages = [p for p in pages if p['char_count'] < 100]
    print(f"Pages with <100 characters: {len(low_char_pages)}")
    
    # Count pages with no Bengali characters
    bengali_chars = set('অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎংঃ')
    no_bengali_pages = []
    for page in pages:
        bengali_count = sum(1 for char in page['text'] if char in bengali_chars)
        if bengali_count < 10:  # Less than 10 Bengali characters
            no_bengali_pages.append(page['page_number'])
    
    print(f"Pages with very few Bengali characters: {len(no_bengali_pages)}")
    if no_bengali_pages:
        print(f"Page numbers: {no_bengali_pages[:10]}...")  # Show first 10
    
    # Check for common OCR artifacts
    total_text = ' '.join([p['text'] for p in pages])
    artifacts = {
        '|': total_text.count('|'),
        'l': total_text.count('l'),
        'I': total_text.count('I'),
        '1': total_text.count('1'),
        '0': total_text.count('0'),
        'O': total_text.count('O'),
    }
    
    print(f"\n=== POTENTIAL OCR ARTIFACTS ===")
    for char, count in artifacts.items():
        if count > 100:
            print(f"'{char}' appears {count} times (potential OCR artifact)")
    
    return pages, stats

if __name__ == "__main__":
    test_ocr_quality() 