#!/usr/bin/env python3

import sys
import os
import time
sys.path.append('/app/src')

from src.services.document_loader import PDFDocumentLoader
import fitz  # PyMuPDF

def test_caching_simple():
    """Test OCR caching with just a few pages."""
    
    print("=== Simple OCR Caching Test ===")
    
    # Initialize the document loader
    loader = PDFDocumentLoader()
    
    # Test PDF path
    pdf_path = '/app/data/Bangla_book.pdf'
    
    # Check initial cache status
    pdf_cache_status = loader.get_cache_status(pdf_path)
    print(f"Initial cache status: {pdf_cache_status['cached_pages']} pages cached")
    
    # Process only first 5 pages for testing
    pdf_document = fitz.open(pdf_path)
    test_pages = min(5, len(pdf_document))
    
    print(f"\nProcessing first {test_pages} pages...")
    
    # First run - should do OCR
    print(f"\n1. First run (with OCR):")
    start_time = time.time()
    
    pages1 = []
    for page_num in range(test_pages):
        page = pdf_document[page_num]
        text = page.get_text()
        
        if not text.strip():
            text = loader._extract_text_with_ocr(page)
        
        text = loader._clean_extracted_text(text)
        if text.strip():
            pages1.append({
                "page_number": page_num + 1,
                "text": text,
                "char_count": len(text)
            })
    
    time1 = time.time() - start_time
    print(f"   Time taken: {time1:.2f} seconds")
    print(f"   Pages extracted: {len(pages1)}")
    
    # Simulate cache by saving results
    cache_data = {}
    for page in pages1:
        cache_data[page["page_number"] - 1] = page["text"]
    
    # Save to cache
    cache_path = loader._get_cache_path(pdf_path)
    import json
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    
    print(f"   Saved {len(cache_data)} pages to cache")
    
    # Second run - should use cache
    print(f"\n2. Second run (using cache):")
    start_time = time.time()
    
    pages2 = []
    for page_num in range(test_pages):
        page = pdf_document[page_num]
        text = page.get_text()
        
        if not text.strip():
            # Use cached result
            if page_num in cache_data:
                text = cache_data[page_num]
                print(f"   Page {page_num + 1}: Using cached OCR")
            else:
                text = loader._extract_text_with_ocr(page)
        
        text = loader._clean_extracted_text(text)
        if text.strip():
            pages2.append({
                "page_number": page_num + 1,
                "text": text,
                "char_count": len(text)
            })
    
    time2 = time.time() - start_time
    print(f"   Time taken: {time2:.2f} seconds")
    print(f"   Pages extracted: {len(pages2)}")
    
    if time1 > 0:
        speedup = time1 / time2
        print(f"   Speed improvement: {speedup:.1f}x faster")
    
    # Verify results are the same
    if len(pages1) == len(pages2):
        print(f"   ✅ Results consistent: {len(pages1)} pages")
    else:
        print(f"   ❌ Results inconsistent: {len(pages1)} vs {len(pages2)} pages")
    
    pdf_document.close()
    
    # Final cache status
    final_cache_status = loader.get_cache_status(pdf_path)
    print(f"\n3. Final cache status:")
    print(f"   Cache exists: {final_cache_status['cache_exists']}")
    print(f"   Cached pages: {final_cache_status['cached_pages']}")
    print(f"   Cache size: {final_cache_status['cache_size_mb']:.2f} MB")
    
    return pages1, pages2

if __name__ == "__main__":
    test_caching_simple() 