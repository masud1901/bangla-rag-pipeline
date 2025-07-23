import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import List, Dict, Any
import pytesseract
from PIL import Image
import io
import json
import hashlib
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


class PDFDocumentLoader:
    """Service for extracting text from PDF documents using PyMuPDF with OCR caching and threading."""
    
    def __init__(self, max_workers: int = None):
        self.supported_extensions = ['.pdf']
        self.cache_dir = Path("/app/cache/ocr")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Threading configuration
        self.max_workers = max_workers or min(threading.active_count() + 4, 8)
        logger.info(f"Initialized PDFDocumentLoader with {self.max_workers} threads")

    def _extract_text_parallel(self, pdf_document, ocr_cache: Dict[int, str], use_cache: bool, skip_failed_ocr: bool) -> List[Dict[str, Any]]:
        """Extract text using parallel processing with threading."""
        
        page_args_list = [
            (page_num, pdf_document.name, use_cache, ocr_cache, skip_failed_ocr) 
            for page_num in range(len(pdf_document))
        ]

        extracted_pages = []
        new_ocr_results = {}
        ocr_failures = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_page = {
                executor.submit(self._process_page_thread_safe, args): args[0]
                for args in page_args_list
            }
            
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    page_num_result, text, is_cached = future.result()
                    
                    if text.strip():
                        page_data = {
                            "page_number": page_num_result + 1,
                            "text": text,
                            "char_count": len(text),
                            "source_file": pdf_document.name,
                        }
                        extracted_pages.append(page_data)
                        
                        if not is_cached:
                            new_ocr_results[page_num_result] = text
                    else:
                        ocr_failures += 1
                except Exception as e:
                    logger.error(f"Page {page_num + 1} processing failed in future: {e}")
                    ocr_failures += 1

        extracted_pages.sort(key=lambda p: p['page_number'])

        if new_ocr_results and use_cache:
            all_ocr_results = {**ocr_cache, **new_ocr_results}
            self._save_ocr_cache(str(pdf_document.name), all_ocr_results)
        
        logger.info(f"Successfully extracted text from {len(extracted_pages)} pages")
        if ocr_failures > 0:
            logger.warning(f"OCR failed or was skipped on {ocr_failures} pages")
        if new_ocr_results:
            logger.info(f"Added {len(new_ocr_results)} new OCR results to cache")
        
        return extracted_pages

    @staticmethod
    def _process_page_thread_safe(args) -> tuple:
        """
        Worker function to process page data safely in a thread.
        This is a static method to avoid issues with 'self'.
        """
        page_num, pdf_name, use_cache, ocr_cache, skip_failed = args
        
        try:
            # Open the PDF within the thread
            with fitz.open(pdf_name) as pdf_document:
                page = pdf_document.load_page(page_num)
                text = ""
                
                try:
                    text = page.get_text()
                    
                    if not text.strip():
                        if use_cache and page_num in ocr_cache:
                            return page_num, ocr_cache[page_num], True
                        
                        # Perform OCR
                        try:
                            pix = page.get_pixmap(matrix=fitz.Matrix(0.8, 0.8))
                            img_data = pix.tobytes("png")
                            img = Image.open(io.BytesIO(img_data))
                            ocr_text = pytesseract.image_to_string(
                                img, lang='ben+eng', config='--psm 6 --oem 3', timeout=60
                            )
                            text = ocr_text
                        except Exception as ocr_error:
                            if skip_failed:
                                logger.warning(f"OCR for page {page_num + 1} failed: {ocr_error}. Skipping.")
                                return page_num, "", False
                            raise ocr_error

                    # Basic cleaning
                    cleaned_text = re.sub(r'\s+', ' ', text).strip()
                    return page_num, cleaned_text, False

                except Exception as e:
                    if skip_failed:
                        logger.error(f"Error processing page {page_num + 1}: {e}. Skipping.")
                        return page_num, "", False
                    raise e
        except Exception as e:
            logger.error(f"Failed to open PDF in thread for page {page_num + 1}: {e}")
            return page_num, "", False

    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """
        Basic text cleaning. Made static to be used by thread-safe worker.
        """
        # Replace multiple whitespace characters with a single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_text_sequential(self, pdf_document, ocr_cache: Dict[int, str], use_cache: bool, skip_failed_ocr: bool) -> List[Dict[str, Any]]:
        """Extract text using sequential processing (original method)."""
        
        extracted_pages = []
        ocr_failures = 0
        new_ocr_results = {}
        
        max_pages = len(pdf_document)
        for page_num in range(max_pages):
            page = pdf_document[page_num]
            text = page.get_text()
            
            if not text.strip():
                if use_cache and page_num in ocr_cache:
                    text = ocr_cache[page_num]
                    logger.info(f"Page {page_num + 1}: Using cached OCR result")
                else:
                    logger.info(f"Page {page_num + 1} has no text, attempting OCR...")
                    try:
                        ocr_text = self._extract_text_with_ocr(page)
                        if ocr_text.strip():
                            text = ocr_text
                            new_ocr_results[page_num] = text
                        else:
                            ocr_failures += 1
                            if skip_failed_ocr:
                                logger.warning(f"OCR failed for page {page_num + 1}, skipping.")
                                continue
                            else:
                                logger.warning(f"OCR failed for page {page_num + 1}")
                    except Exception as e:
                        ocr_failures += 1
                        if skip_failed_ocr:
                            logger.warning(f"OCR error on page {page_num + 1}: {e}, skipping.")
                            continue
                        else:
                            logger.error(f"OCR error on page {page_num + 1}: {e}")
                            raise # Or continue, depending on desired behavior
            
            cleaned_text = self._clean_extracted_text(text)
            
            if cleaned_text.strip():
                page_data = {
                    "page_number": page_num + 1,
                    "text": cleaned_text,
                    "char_count": len(cleaned_text),
                    "source_file": str(pdf_document.name)
                }
                extracted_pages.append(page_data)
                
        if new_ocr_results and use_cache:
            all_ocr_results = {**ocr_cache, **new_ocr_results}
            self._save_ocr_cache(str(pdf_document.name), all_ocr_results)
        
        logger.info(f"Successfully extracted text from {len(extracted_pages)} pages")
        if ocr_failures > 0:
            logger.warning(f"OCR failed or was skipped on {ocr_failures} pages")
        if new_ocr_results:
            logger.info(f"Added {len(new_ocr_results)} new OCR results to cache")
        
        return extracted_pages
        
    def _get_cache_path(self, pdf_path: str) -> Path:
        """Generate cache file path based on PDF file hash."""
        pdf_hash = hashlib.md5(open(pdf_path, 'rb').read()).hexdigest()
        return self.cache_dir / f"{pdf_hash}_ocr_cache.json"

    def _load_ocr_cache(self, pdf_path: str) -> Dict[int, str]:
        """Load OCR cache if it exists."""
        cache_path = self._get_cache_path(str(pdf_path))
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    # Make sure keys are integers
                    return {int(k): v for k, v in json.load(f).items()}
            except Exception as e:
                logger.warning(f"Failed to load OCR cache: {e}. It will be recreated.")
        return {}

    def _save_ocr_cache(self, pdf_path: str, ocr_results: Dict[int, str]):
        """Save OCR results to cache."""
        cache_path = self._get_cache_path(str(pdf_path))
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(ocr_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved OCR cache with {len(ocr_results)} pages")
        except Exception as e:
            logger.warning(f"Failed to save OCR cache: {e}")

    def extract_text_from_pdf(self, pdf_path: str, use_cache: bool = True, use_parallel: bool = True, skip_failed_ocr: bool = True) -> List[Dict[str, Any]]:
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {pdf_path.suffix}")
        
        ocr_cache = self._load_ocr_cache(str(pdf_path)) if use_cache else {}
        
        try:
            pdf_document = fitz.open(pdf_path)
            
            logger.info(f"Processing PDF: {pdf_path.name} with {len(pdf_document)} pages")
            logger.info(f"Using {'parallel' if use_parallel else 'sequential'} processing with {self.max_workers} threads")
            logger.info(f"Cache contains {len(ocr_cache)} pages")
            
            start_time = time.time()
            
            if use_parallel and len(pdf_document) > 1:
                extracted_pages = self._extract_text_parallel(pdf_document, ocr_cache, use_cache, skip_failed_ocr)
            else:
                extracted_pages = self._extract_text_sequential(pdf_document, ocr_cache, use_cache, skip_failed_ocr)
            
            processing_time = time.time() - start_time
            logger.info(f"Text extraction completed in {processing_time:.2f} seconds")
            
            pdf_document.close()
            return extracted_pages
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise

    def _extract_text_with_ocr(self, page) -> str:
        """
        Extract text from a page using OCR (for sequential processing).
        """
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(0.8, 0.8))
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(
                img, 
                lang='ben+eng', 
                config='--psm 6 --oem 3',
                timeout=60
            )
            return text
        except Exception as e:
            logger.warning(f"OCR failed for page: {e}")
            return ""

    def get_document_stats(self, extracted_pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about the extracted document."""
        if not extracted_pages:
            return {
                "total_pages": 0,
                "total_characters": 0,
                "avg_chars_per_page": 0,
                "source_file": "N/A"
            }
        
        total_chars = sum(p.get("char_count", 0) for p in extracted_pages)
        total_pages = len(extracted_pages)
        
        return {
            "total_pages": total_pages,
            "total_characters": total_chars,
            "avg_chars_per_page": total_chars / total_pages if total_pages > 0 else 0,
            "source_file": extracted_pages[0]["source_file"]
        }

    def clear_ocr_cache(self, pdf_path: str = None):
        """Clear OCR cache for a specific PDF or all caches."""
        if pdf_path:
            cache_path = self._get_cache_path(pdf_path)
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cleared OCR cache for {pdf_path}")
        else:
            for cache_file in self.cache_dir.glob("*_ocr_cache.json"):
                cache_file.unlink()
            logger.info("Cleared all OCR caches")
    
    def get_cache_status(self, pdf_path: str = None) -> Dict[str, Any]:
        if pdf_path:
            cache_path = self._get_cache_path(pdf_path)
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                return {
                    "pdf_path": pdf_path,
                    "cached_pages": len(cache_data),
                    "cache_size_mb": cache_path.stat().st_size / (1024 * 1024),
                    "cache_exists": True
                }
            else:
                return {"pdf_path": pdf_path, "cached_pages": 0, "cache_size_mb": 0, "cache_exists": False}
        else:
            cache_files = list(self.cache_dir.glob("*_ocr_cache.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            return {
                "total_cache_files": len(cache_files),
                "total_cache_size_mb": total_size / (1024 * 1024),
                "cache_directory": str(self.cache_dir)
            } 