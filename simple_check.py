#!/usr/bin/env python3

import sys
import os
sys.path.append('/app/src')

from src.services.document_loader import PDFDocumentLoader
import fitz  # PyMuPDF

def simple_check():
    """Very simple check of a few specific pages."""
    
    print("=== Simple Page Check ===")
    
    # Open PDF
    pdf_path = '/app/data/Bangla_book.pdf'
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    
    print(f"PDF has {total_pages} pages")
    
    # Check specific pages
    test_pages = [1, 10, 50, 100, 150, 200, 250, 300, 350]
    
    loader = PDFDocumentLoader()
    
    total_chars = 0
    pages_with_content = 0
    
    for page_num in test_pages:
        if page_num > total_pages:
            continue
            
        page = pdf_document[page_num]
        
        # Try direct text extraction first
        text = page.get_text()
        
        # If no text, try OCR
        if not text.strip():
            text = loader._extract_text_with_ocr(page)
        
        # Clean the text
        text = loader._clean_extracted_text(text)
        char_count = len(text)
        
        if char_count > 0:
            pages_with_content += 1
            total_chars += char_count
        
        print(f"Page {page_num}: {char_count} chars")
    
    pdf_document.close()
    
    print(f"\n=== SUMMARY ===")
    print(f"Pages with content: {pages_with_content}/{len(test_pages)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average chars per content page: {total_chars / pages_with_content if pages_with_content > 0 else 0:,.0f}")
    
    # Estimate total content
    content_ratio = pages_with_content / len(test_pages)
    estimated_content_pages = total_pages * content_ratio
    estimated_total_chars = (total_chars / len(test_pages)) * total_pages
    
    print(f"\n=== ESTIMATES ===")
    print(f"Estimated pages with content: {estimated_content_pages:.0f}")
    print(f"Estimated total characters: {estimated_total_chars:,.0f}")
    print(f"Estimated chunks (765 chars each): {estimated_total_chars / 765:.0f}")

if __name__ == "__main__":
    simple_check() 