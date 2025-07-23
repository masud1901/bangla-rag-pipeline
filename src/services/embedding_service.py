import cohere
import logging
import time
from typing import List, Dict, Any
from src.core.config import settings

logger = logging.getLogger(__name__)


class CohereEmbeddingService:
    """Service for generating multilingual embeddings using Cohere API."""
    
    def __init__(self):
        """Initialize the Cohere client with API key from settings."""
        try:
            self.client = cohere.Client(settings.cohere_api_key)
            self.model_name = "embed-multilingual-v3.0"
            # Throttling configuration for Cohere trial limits
            self.batch_size = 50  # Process 50 chunks at a time
            self.delay_seconds = 60  # Wait 60 seconds between batches
            self.max_tokens_per_batch = 80000  # Conservative limit for 100k/min
            logger.info(f"Initialized Cohere client with model: {self.model_name}")
            logger.info(f"Throttling config: batch_size={self.batch_size}, delay={self.delay_seconds}s")
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")
            raise
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Rough estimate of tokens in text (4 chars per token for multilingual).
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def _embed_batch_with_throttling(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with throttling.
        
        Args:
            texts: List of text strings to embed
            input_type: Type of input - "search_document" for indexing, "search_query" for queries
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Estimate tokens for this batch
            total_chars = sum(len(text) for text in texts)
            estimated_tokens = total_chars // 4
            
            logger.info(f"Processing batch of {len(texts)} texts (~{estimated_tokens} tokens)")
            
            response = self.client.embed(
                texts=texts,
                model=self.model_name,
                input_type=input_type
            )
            
            embeddings = response.embeddings
            logger.info(f"Successfully generated {len(embeddings)} embeddings for this batch")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings for batch: {e}")
            raise
    
    def embed_texts(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        """
        Generate embeddings for a list of texts with throttling.
        
        Args:
            texts: List of text strings to embed
            input_type: Type of input - "search_document" for indexing, "search_query" for queries
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts using {self.model_name} with throttling")
        
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_texts)} texts)")
            
            try:
                batch_embeddings = self._embed_batch_with_throttling(batch_texts, input_type)
                all_embeddings.extend(batch_embeddings)
                
                # Add delay between batches (except for the last batch)
                if batch_num < total_batches - 1:
                    logger.info(f"Waiting {self.delay_seconds} seconds before next batch...")
                    time.sleep(self.delay_seconds)
                    
            except Exception as e:
                logger.error(f"Failed to process batch {batch_num + 1}: {e}")
                raise
        
        logger.info(f"Completed all batches. Total embeddings generated: {len(all_embeddings)}")
        return all_embeddings
    
    def embed_single_text(self, text: str, input_type: str = "search_document") -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            input_type: Type of input - "search_document" for indexing, "search_query" for queries
            
        Returns:
            Single embedding vector
        """
        embeddings = self.embed_texts([text], input_type)
        return embeddings[0] if embeddings else []
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks and add them to chunk metadata.
        
        Args:
            chunks: List of text chunks from text processor
            
        Returns:
            List of chunks with added embedding vectors
        """
        if not chunks:
            return []
        
        # Extract just the text content for embedding
        texts = [chunk["text"] for chunk in chunks]
        
        try:
            # Generate embeddings for all texts with throttling
            embeddings = self.embed_texts(texts, input_type="search_document")
            
            # Add embeddings back to chunk metadata
            chunks_with_embeddings = []
            for i, chunk in enumerate(chunks):
                enhanced_chunk = chunk.copy()
                enhanced_chunk["embedding"] = embeddings[i]
                enhanced_chunk["embedding_model"] = self.model_name
                chunks_with_embeddings.append(enhanced_chunk)
            
            logger.info(f"Added embeddings to {len(chunks_with_embeddings)} chunks")
            return chunks_with_embeddings
            
        except Exception as e:
            logger.error(f"Error embedding chunks: {e}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        return self.embed_single_text(query, input_type="search_query")
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "dimension": settings.embedding_dimension,
            "provider": "Cohere",
            "supports_multilingual": True,
            "languages": ["English", "Bengali", "100+ others"],
            "throttling": {
                "batch_size": self.batch_size,
                "delay_seconds": self.delay_seconds,
                "max_tokens_per_batch": self.max_tokens_per_batch
            }
        } 