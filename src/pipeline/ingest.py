#!/usr/bin/env python3
"""
Data Ingestion Pipeline for Multilingual RAG System

This script processes the HSC26 Bangla 1st paper PDF, extracts text,
chunks it semantically, generates embeddings, and stores in Pinecone.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

from src.services.document_loader import PDFDocumentLoader
from src.services.text_processor import TextProcessor
from src.services.embedding_service import CohereEmbeddingService
from src.services.vector_store_service import ChromaVectorStore
from src.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Main pipeline for processing and indexing documents."""
    
    def __init__(self):
        """Initialize all services."""
        logger.info("Initializing Data Ingestion Pipeline")
        
        self.document_loader = PDFDocumentLoader()
        self.text_processor = TextProcessor()
        self.embedding_service = CohereEmbeddingService()
        self.vector_store = ChromaVectorStore()
        
        logger.info("All services initialized successfully")
    
    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF document through the complete pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Processing results and statistics
        """
        results = {
            "pdf_path": pdf_path,
            "steps_completed": [],
            "errors": []
        }
        
        try:
            # Step 1: Extract text from PDF
            logger.info("Step 1: Extracting text from PDF")
            extracted_pages = self.document_loader.extract_text_from_pdf(pdf_path)
            doc_stats = self.document_loader.get_document_stats(extracted_pages)
            
            results["document_stats"] = doc_stats
            results["steps_completed"].append("text_extraction")
            logger.info(f"Extracted text from {doc_stats['total_pages']} pages")
            
            # Step 2: Clean and chunk text
            logger.info("Step 2: Cleaning and chunking text")
            chunks = self.text_processor.chunk_documents(extracted_pages)
            chunk_stats = self.text_processor.get_chunking_stats(chunks)
            
            results["chunk_stats"] = chunk_stats
            results["steps_completed"].append("text_chunking")
            logger.info(f"Created {chunk_stats['total_chunks']} chunks")
            
            # Step 3: Generate embeddings
            logger.info("Step 3: Generating embeddings")
            chunks_with_embeddings = self.embedding_service.embed_chunks(chunks)
            embedding_info = self.embedding_service.get_embedding_info()
            
            results["embedding_info"] = embedding_info
            results["steps_completed"].append("embedding_generation")
            logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
            
            # Step 4: Store in vector database
            logger.info("Step 4: Storing vectors in ChromaDB")
            upsert_results = self.vector_store.upsert_chunks(chunks_with_embeddings)
            index_stats = self.vector_store.get_index_stats()
            
            results["upsert_results"] = upsert_results
            results["index_stats"] = index_stats
            results["steps_completed"].append("vector_storage")
            logger.info(f"Stored {upsert_results['upserted_count']} vectors")
            
            results["status"] = "success"
            logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            results["status"] = "failed"
            raise
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of the processing results."""
        print("\n" + "="*60)
        print("DATA INGESTION PIPELINE SUMMARY")
        print("="*60)
        
        print(f"Status: {results['status'].upper()}")
        print(f"PDF Processed: {results['pdf_path']}")
        print(f"Steps Completed: {', '.join(results['steps_completed'])}")
        
        if "document_stats" in results:
            stats = results["document_stats"]
            print(f"\nDocument Statistics:")
            print(f"  - Total Pages: {stats['total_pages']}")
            print(f"  - Total Characters: {stats['total_chars']:,}")
            print(f"  - Avg Characters/Page: {stats['avg_chars_per_page']:,}")
        
        if "chunk_stats" in results:
            stats = results["chunk_stats"]
            print(f"\nChunking Statistics:")
            print(f"  - Total Chunks: {stats['total_chunks']}")
            print(f"  - Avg Chunk Size: {stats['avg_chunk_size']} characters")
            print(f"  - Avg Words/Chunk: {stats['avg_word_count']}")
        
        if "embedding_info" in results:
            info = results["embedding_info"]
            print(f"\nEmbedding Information:")
            print(f"  - Model: {info['model_name']}")
            print(f"  - Dimension: {info['dimension']}")
            print(f"  - Provider: {info['provider']}")
        
        if "upsert_results" in results:
            upsert = results["upsert_results"]
            print(f"\nVector Storage:")
            print(f"  - Vectors Stored: {upsert['upserted_count']}")
            print(f"  - Collection: {upsert.get('collection_name', upsert.get('index_name', 'N/A'))}")
        
        if results["errors"]:
            print(f"\nErrors:")
            for error in results["errors"]:
                print(f"  - {error}")
        
        print("="*60)


def main():
    """Main entry point for the ingestion script."""
    parser = argparse.ArgumentParser(description="Process PDF for RAG system")
    parser.add_argument(
        "pdf_path", 
        help="Path to the PDF file to process"
    )
    parser.add_argument(
        "--clear-index", 
        action="store_true",
        help="Clear existing vectors before processing"
    )
    
    args = parser.parse_args()
    
    # Validate PDF path
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline = DataIngestionPipeline()
        
        # Clear index if requested
        if args.clear_index:
            logger.info("Clearing existing vectors from index")
            pipeline.vector_store.delete_all_vectors()
        
        # Process the document
        results = pipeline.process_document(str(pdf_path))
        
        # Print summary
        pipeline.print_summary(results)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 