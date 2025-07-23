#!/usr/bin/env python3

import sys
import os
sys.path.append('/app/src')

from src.services.document_loader import PDFDocumentLoader
import fitz  # PyMuPDF

def test_ocr_simple():
    """Test OCR quality on first few pages only."""
    
    print("=== Simple OCR Quality Test ===")
    
    # Open PDF and get basic info
    pdf_path = '/app/data/Bangla_book.pdf'
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    
    print(f"PDF has {total_pages} pages")
    
    # Test first 5 pages
    test_pages = min(5, total_pages)
    print(f"Testing first {test_pages} pages...")
    
    loader = PDFDocumentLoader()
    
    # Extract text from first few pages only
    pages = []
    for page_num in range(test_pages):
        page = pdf_document[page_num]
        
        # Try direct text extraction first
        text = page.get_text()
        
        # If no text, try OCR
        if not text.strip():
            print(f"Page {page_num + 1}: No text found, using OCR...")
            text = loader._extract_text_with_ocr(page)
        else:
            print(f"Page {page_num + 1}: Text extracted directly")
        
        # Clean the text
        text = loader._clean_extracted_text(text)
        
        page_data = {
            "page_number": page_num + 1,
            "text": text,
            "char_count": len(text),
            "source_file": "Bangla_book.pdf"
        }
        pages.append(page_data)
    
    pdf_document.close()
    
    # Show results
    print(f"\n=== OCR RESULTS (First {test_pages} pages) ===")
    total_chars = 0
    for page in pages:
        print(f"\n--- PAGE {page['page_number']} ---")
        print(f"Characters: {page['char_count']}")
        total_chars += page['char_count']
        
        # Show first 500 characters
        preview = page['text'][:500]
        print("Text preview:")
        print("-" * 40)
        print(preview)
        print("-" * 40)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total characters in {test_pages} pages: {total_chars:,}")
    print(f"Average characters per page: {total_chars // test_pages:,}")
    
    # Check for Bengali characters
    bengali_chars = set('অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎংঃ')
    total_bengali = 0
    for page in pages:
        bengali_count = sum(1 for char in page['text'] if char in bengali_chars)
        total_bengali += bengali_count
        print(f"Page {page['page_number']}: {bengali_count} Bengali characters")
    
    print(f"Total Bengali characters: {total_bengali}")
    print(f"Bengali character percentage: {(total_bengali/total_chars)*100:.1f}%")
    
    return pages

if __name__ == "__main__":
    test_ocr_simple() 