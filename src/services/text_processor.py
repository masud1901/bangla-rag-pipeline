import re
import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TextProcessor:
    """Service for text cleaning and semantic chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text processor with chunking parameters.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize the text splitter with Bengali-aware separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                "।",     # Bengali sentence end
                ".",     # English sentence end
                "?",     # Question mark
                "!",     # Exclamation mark
                ";",     # Semicolon
                ":",     # Colon
                ",",     # Comma
                " ",     # Space
                ""       # Character level fallback
            ],
            length_function=len,
        )
    
    def clean_text(self, text: str) -> str:
        """
        Advanced text cleaning for Bengali and English content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'(?i)page\s*\d+', '', text)  # Remove page numbers
        text = re.sub(r'(?i)chapter\s*\d+', '', text)  # Remove chapter numbers
        
        # Remove isolated numbers (likely page numbers or artifacts)
        text = re.sub(r'\b\d{1,3}\b', '', text)
        
        # Fix common OCR issues with Bengali text
        text = self._fix_bengali_ocr_issues(text)
        
        # Remove excessive punctuation
        text = re.sub(r'[।.]{2,}', '।', text)  # Multiple Bengali periods
        text = re.sub(r'[.]{2,}', '.', text)   # Multiple English periods
        
        # Normalize quotation marks
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s*([।.!?;:,])\s*', r'\1 ', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def _fix_bengali_ocr_issues(self, text: str) -> str:
        """
        Fix common OCR issues specific to Bengali text.
        
        Args:
            text: Text with potential OCR issues
            
        Returns:
            Text with OCR issues fixed
        """
        # Common Bengali OCR fixes (this would be expanded based on actual issues found)
        ocr_fixes = {
            'ও': 'ও',  # Common OCR confusion
            'ব': 'ব',  # Another common issue
            # Add more as needed when testing with actual Bengali PDFs
        }
        
        for wrong, correct in ocr_fixes.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def chunk_documents(self, extracted_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split extracted pages into semantic chunks.
        
        Args:
            extracted_pages: List of page data from document loader
            
        Returns:
            List of text chunks with metadata
        """
        all_chunks = []
        chunk_id = 0
        
        for page_data in extracted_pages:
            page_text = page_data["text"]
            page_number = page_data["page_number"]
            source_file = page_data["source_file"]
            
            # Clean the text before chunking
            cleaned_text = self.clean_text(page_text)
            
            if not cleaned_text:
                logger.warning(f"No content after cleaning for page {page_number}")
                continue
            
            # Split the cleaned text into chunks
            text_chunks = self.text_splitter.split_text(cleaned_text)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk_data = {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "page_number": page_number,
                    "chunk_index": i,
                    "source_file": source_file,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split())
                }
                all_chunks.append(chunk_data)
                chunk_id += 1
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(extracted_pages)} pages")
        return all_chunks
    
    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the chunking process.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Chunking statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "avg_word_count": 0,
                "size_distribution": {}
            }
        
        chunk_sizes = [chunk["char_count"] for chunk in chunks]
        word_counts = [chunk["word_count"] for chunk in chunks]
        
        # Calculate size distribution
        size_ranges = {
            "small (0-500)": sum(1 for size in chunk_sizes if size <= 500),
            "medium (501-1000)": sum(1 for size in chunk_sizes if 501 <= size <= 1000),
            "large (1001+)": sum(1 for size in chunk_sizes if size > 1000)
        }
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) // len(chunks),
            "avg_word_count": sum(word_counts) // len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "size_distribution": size_ranges
        } 