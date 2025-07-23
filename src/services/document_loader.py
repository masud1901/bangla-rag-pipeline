import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import List, Dict, Any
import pytesseract
from PIL import Image
import io

logger = logging.getLogger(__name__)


class PDFDocumentLoader:
    """Service for extracting text from PDF documents using PyMuPDF."""
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from a PDF file with page-level metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing page text and metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {pdf_path.suffix}")
        
        extracted_pages = []
        
        try:
            # Open the PDF document
            pdf_document = fitz.open(pdf_path)
            
            logger.info(f"Processing PDF: {pdf_path.name} with {len(pdf_document)} pages")
            
            # Process pages in batches for efficiency
            max_pages = len(pdf_document)
            # For testing, process only first 10 pages
            # max_pages = min(10, max_pages)
            for page_num in range(max_pages):
                page = pdf_document[page_num]
                
                # Extract text from the page
                text = page.get_text()
                
                # If no text found, try OCR for image-based PDFs
                if not text.strip():
                    logger.info(f"Page {page_num + 1} has no text, attempting OCR...")
                    text = self._extract_text_with_ocr(page)
                
                # Basic cleaning: remove excessive whitespace
                text = self._clean_extracted_text(text)
                
                if text.strip():  # Only include pages with content
                    page_data = {
                        "page_number": page_num + 1,  # 1-indexed
                        "text": text,
                        "char_count": len(text),
                        "source_file": str(pdf_path.name)
                    }
                    extracted_pages.append(page_data)
                    
            pdf_document.close()
            
            logger.info(f"Successfully extracted text from {len(extracted_pages)} pages")
            return extracted_pages
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Basic text cleaning for extracted PDF content.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace and normalize line breaks
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip whitespace and skip empty lines
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        # Join lines with single spaces, preserving paragraph breaks
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Further normalize whitespace
        import re
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # Preserve paragraph breaks
        
        return cleaned_text.strip()
    
    def _extract_text_with_ocr(self, page) -> str:
        """
        Extract text from a page using OCR (Optical Character Recognition).
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Extracted text from the page
        """
        try:
            # Convert page to image with lower resolution for faster processing
            pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))  # 1.0x zoom for faster OCR
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            
            # Extract text using OCR with Bengali language and optimized settings
            text = pytesseract.image_to_string(
                img, 
                lang='ben+eng', 
                config='--psm 6 --oem 3',
                timeout=30  # 30 second timeout
            )
            
            return text
            
        except Exception as e:
            logger.warning(f"OCR failed for page: {e}")
            return ""
    
    def get_document_stats(self, extracted_pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the extracted document.
        
        Args:
            extracted_pages: List of extracted page data
            
        Returns:
            Document statistics
        """
        if not extracted_pages:
            return {"total_pages": 0, "total_chars": 0, "avg_chars_per_page": 0}
        
        total_chars = sum(page["char_count"] for page in extracted_pages)
        
        return {
            "total_pages": len(extracted_pages),
            "total_chars": total_chars,
            "avg_chars_per_page": total_chars // len(extracted_pages),
            "source_file": extracted_pages[0]["source_file"]
        } 