#!/usr/bin/env python3

import sys
import os
sys.path.append('/app/src')

from src.services.document_loader import PDFDocumentLoader
import fitz  # PyMuPDF

def test_middle_pages():
    """Test OCR quality on a few pages from the middle of the book."""
    
    print("=== Middle Pages OCR Test ===")
    
    # Open PDF
    pdf_path = '/app/data/Bangla_book.pdf'
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    
    print(f"PDF has {total_pages} pages")
    
    # Test pages from middle (around page 180-190)
    test_pages = [180, 185, 190, 195, 200]
    print(f"Testing pages: {test_pages}")
    
    loader = PDFDocumentLoader()
    
    total_chars = 0
    total_bengali = 0
    
    for page_num in test_pages:
        if page_num >= total_pages:
            continue
            
        page = pdf_document[page_num]
        
        # Try direct text extraction first
        text = page.get_text()
        
        # If no text, try OCR
        if not text.strip():
            print(f"Page {page_num}: Using OCR...")
            text = loader._extract_text_with_ocr(page)
        else:
            print(f"Page {page_num}: Direct extraction")
        
        # Clean the text
        text = loader._clean_extracted_text(text)
        
        # Count Bengali characters
        bengali_chars = set('অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎংঃ')
        bengali_count = sum(1 for char in text if char in bengali_chars)
        
        print(f"\n--- PAGE {page_num} ---")
        print(f"Characters: {len(text)}")
        print(f"Bengali characters: {bengali_count}")
        print("Text preview (first 400 chars):")
        print("-" * 50)
        print(text[:400])
        print("-" * 50)
        
        total_chars += len(text)
        total_bengali += bengali_count
    
    pdf_document.close()
    
    print(f"\n=== SUMMARY ===")
    print(f"Total characters in {len(test_pages)} pages: {total_chars:,}")
    print(f"Average characters per page: {total_chars // len(test_pages):,}")
    print(f"Total Bengali characters: {total_bengali}")
    print(f"Bengali character percentage: {(total_bengali/total_chars)*100:.1f}%")
    
    # Check for OCR artifacts
    print(f"\n=== OCR QUALITY CHECK ===")
    if total_chars > 0:
        artifacts = {
            '|': total_chars > 1000,  # Too many vertical bars
            'l': total_chars > 1000,  # Too many lowercase L
            'I': total_chars > 1000,  # Too many capital I
            '1': total_chars > 1000,  # Too many ones
        }
        
        for artifact, threshold in artifacts.items():
            if threshold:
                print(f"⚠️  High occurrence of '{artifact}' - potential OCR artifact")

if __name__ == "__main__":
    test_middle_pages() 