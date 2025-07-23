#!/usr/bin/env python3

import sys
import os
import time
sys.path.append('/app/src')

from src.services.document_loader import PDFDocumentLoader

def test_caching():
    """Test the OCR caching functionality."""
    
    print("=== OCR Caching Test ===")
    
    # Initialize the document loader
    loader = PDFDocumentLoader()
    
    # Check initial cache status
    print("\n1. Initial cache status:")
    cache_status = loader.get_cache_status()
    print(f"   Cache files: {cache_status['total_cache_files']}")
    print(f"   Cache size: {cache_status['total_cache_size_mb']:.2f} MB")
    
    # Test PDF path
    pdf_path = '/app/data/Bangla_book.pdf'
    
    # Check specific PDF cache status
    pdf_cache_status = loader.get_cache_status(pdf_path)
    print(f"\n2. PDF cache status:")
    print(f"   Cache exists: {pdf_cache_status['cache_exists']}")
    print(f"   Cached pages: {pdf_cache_status['cached_pages']}")
    
    # First run - should do OCR
    print(f"\n3. First run (with OCR):")
    start_time = time.time()
    pages1 = loader.extract_text_from_pdf(pdf_path, use_cache=True)
    time1 = time.time() - start_time
    
    print(f"   Time taken: {time1:.2f} seconds")
    print(f"   Pages extracted: {len(pages1)}")
    
    # Check cache status after first run
    pdf_cache_status = loader.get_cache_status(pdf_path)
    print(f"   Cached pages after first run: {pdf_cache_status['cached_pages']}")
    
    # Second run - should use cache
    print(f"\n4. Second run (using cache):")
    start_time = time.time()
    pages2 = loader.extract_text_from_pdf(pdf_path, use_cache=True)
    time2 = time.time() - start_time
    
    print(f"   Time taken: {time2:.2f} seconds")
    print(f"   Pages extracted: {len(pages2)}")
    print(f"   Speed improvement: {time1/time2:.1f}x faster")
    
    # Verify results are the same
    if len(pages1) == len(pages2):
        print(f"   ✅ Results consistent: {len(pages1)} pages")
    else:
        print(f"   ❌ Results inconsistent: {len(pages1)} vs {len(pages2)} pages")
    
    # Test without cache
    print(f"\n5. Third run (without cache):")
    start_time = time.time()
    pages3 = loader.extract_text_from_pdf(pdf_path, use_cache=False)
    time3 = time.time() - start_time
    
    print(f"   Time taken: {time3:.2f} seconds")
    print(f"   Pages extracted: {len(pages3)}")
    
    # Final cache status
    final_cache_status = loader.get_cache_status(pdf_path)
    print(f"\n6. Final cache status:")
    print(f"   Cache exists: {final_cache_status['cache_exists']}")
    print(f"   Cached pages: {final_cache_status['cached_pages']}")
    print(f"   Cache size: {final_cache_status['cache_size_mb']:.2f} MB")
    
    return pages1, pages2, pages3

if __name__ == "__main__":
    test_caching() 