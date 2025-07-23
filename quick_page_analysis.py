#!/usr/bin/env python3

import sys
import os
sys.path.append('/app/src')

from src.services.document_loader import PDFDocumentLoader
import fitz  # PyMuPDF

def quick_page_analysis():
    """Quick analysis of page content distribution."""
    
    print("=== Quick Page Content Analysis ===")
    
    # Open PDF
    pdf_path = '/app/data/Bangla_book.pdf'
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    
    print(f"PDF has {total_pages} pages")
    
    # Sample pages at intervals to get a quick overview
    sample_pages = list(range(1, total_pages + 1, 20))  # Every 20th page
    sample_pages.extend([50, 100, 150, 200, 250, 300, 350])  # Add some specific pages
    
    loader = PDFDocumentLoader()
    
    content_distribution = {
        'no_content': 0,      # 0 chars
        'very_low': 0,        # 1-100 chars
        'low': 0,             # 101-500 chars
        'medium': 0,          # 501-1000 chars
        'high': 0,            # 1001-2000 chars
        'very_high': 0        # 2000+ chars
    }
    
    total_chars = 0
    pages_with_content = 0
    
    for page_num in sample_pages:
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
        
        # Categorize by content amount
        if char_count == 0:
            content_distribution['no_content'] += 1
        elif char_count <= 100:
            content_distribution['very_low'] += 1
        elif char_count <= 500:
            content_distribution['low'] += 1
        elif char_count <= 1000:
            content_distribution['medium'] += 1
        elif char_count <= 2000:
            content_distribution['high'] += 1
        else:
            content_distribution['very_high'] += 1
        
        if char_count > 0:
            pages_with_content += 1
            total_chars += char_count
        
        print(f"Page {page_num}: {char_count} chars")
    
    pdf_document.close()
    
    print(f"\n=== CONTENT DISTRIBUTION ===")
    for category, count in content_distribution.items():
        print(f"{category}: {count} pages")
    
    print(f"\n=== ESTIMATED TOTALS ===")
    # Estimate total content based on sample
    sample_ratio = len(sample_pages) / total_pages
    estimated_content_pages = pages_with_content / sample_ratio
    estimated_total_chars = total_chars / sample_ratio
    
    print(f"Pages with content: ~{estimated_content_pages:.0f}")
    print(f"Total characters: ~{estimated_total_chars:,.0f}")
    print(f"Average chars per content page: {total_chars / pages_with_content if pages_with_content > 0 else 0:,.0f}")
    
    # Estimate chunks
    # If average chunk is 765 chars (from your stats), how many chunks should we have?
    estimated_chunks = estimated_total_chars / 765
    print(f"Estimated chunks: ~{estimated_chunks:.0f}")
    print(f"Actual chunks: 903")
    print(f"Missing chunks: ~{estimated_chunks - 903:.0f}")

if __name__ == "__main__":
    quick_page_analysis() 