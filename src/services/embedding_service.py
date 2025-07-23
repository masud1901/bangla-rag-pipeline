import cohere
import openai
import logging
import time
from typing import List, Dict, Any
from src.core.config import settings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)


class CohereEmbeddingService:
    """Service for generating multilingual embeddings using Cohere API with parallel processing."""
    
    def __init__(self, max_workers: int = None):
        """Initialize the Cohere client with API key from settings."""
        try:
            self.client = cohere.Client(settings.cohere_api_key)
            self.model_name = "embed-multilingual-v3.0"
            self.dimension = settings.embedding_dimension
            # Throttling configuration for Cohere trial limits
            self.batch_size = 50  # Process 50 chunks at a time
            self.delay_seconds = 60  # Wait 60 seconds between batches
            self.max_tokens_per_batch = 80000  # Conservative limit for 100k/min
            
            # Parallel processing configuration
            self.max_workers = max_workers or min(mp.cpu_count(), 2)  # Limit to 2 workers for API calls
            self._lock = threading.Lock()  # Thread lock for API rate limiting
            
            logger.info(f"Initialized Cohere client with model: {self.model_name}")
            logger.info(f"Throttling config: batch_size={self.batch_size}, delay={self.delay_seconds}s")
            logger.info(f"Parallel processing: {self.max_workers} workers")
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
    
    def _embed_batch_parallel(self, batch_data: tuple) -> tuple:
        """
        Embed a batch of texts in parallel (worker function).
        
        Args:
            batch_data: Tuple of (batch_id, texts, input_type)
            
        Returns:
            Tuple of (batch_id, embeddings)
        """
        batch_id, texts, input_type = batch_data
        
        with self._lock:  # Ensure thread-safe API calls
            try:
                embeddings = self._embed_batch_with_throttling(texts, input_type)
                return (batch_id, embeddings)
            except Exception as e:
                logger.error(f"Error in parallel batch {batch_id}: {e}")
                return (batch_id, [])
    
    def embed_texts(self, texts: List[str], input_type: str = "search_document", use_parallel: bool = True) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with throttling and optional parallel processing.
        
        Args:
            texts: List of text strings to embed
            input_type: Type of input - "search_document" for indexing, "search_query" for queries
            use_parallel: Whether to use parallel processing
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts using {self.model_name}")
        
        if use_parallel and len(texts) > self.batch_size * 2:
            # Use parallel processing for large datasets
            return self._embed_texts_parallel(texts, input_type)
        else:
            # Use sequential processing for smaller datasets
            return self._embed_texts_sequential(texts, input_type)
    
    def _embed_texts_parallel(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Generate embeddings using parallel processing."""
        
        # Split texts into batches
        batches = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            batches.append((batch_num, batch_texts, input_type))
        
        all_embeddings = []
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._embed_batch_parallel, batch_data): batch_data[0]
                for batch_data in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_id, embeddings = future.result()
                    all_embeddings.extend(embeddings)
                    logger.info(f"Completed batch {batch_id + 1}/{total_batches}")
                    
                    # Add delay between batches (except for the last batch)
                    if batch_id < total_batches - 1:
                        logger.info(f"Waiting {self.delay_seconds} seconds before next batch...")
                        time.sleep(self.delay_seconds)
                        
                except Exception as e:
                    logger.error(f"Failed to process batch {batch_id + 1}: {e}")
                    raise
        
        logger.info(f"Completed all batches. Total embeddings generated: {len(all_embeddings)}")
        return all_embeddings
    
    def _embed_texts_sequential(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Generate embeddings using sequential processing (original method)."""
        
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
        Generates and assigns embeddings to a list of chunks.
        Filters out chunks for which embedding generation fails.
        """
        if not chunks:
            return []

        texts = [chunk['text'] for chunk in chunks]
        
        try:
            # Note: Using sequential embedding to respect trial API rate limits.
            # Parallel processing can be re-enabled for paid tiers.
            embeddings = self.embed_texts(texts, use_parallel=False)
            
            # Check for length mismatch due to potential failures
            if len(embeddings) != len(chunks):
                logger.warning(f"Embedding count ({len(embeddings)}) does not match chunk count ({len(chunks)}). Some chunks may have failed.")
                # Fallback to returning chunks without embeddings if there's a mismatch
                return []

            # Assign embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk['embedding'] = embeddings[i]
            
            return chunks

        except Exception as e:
            logger.error(f"An error occurred during chunk embedding: {e}")
            return [] # Return empty list on failure

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query string."""
        return self.embed_texts([query], input_type="search_query")[0]
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "provider": "Cohere",
            "max_tokens_per_batch": self.max_tokens_per_batch
        }


class OpenAIEmbeddingService:
    """Service for generating multilingual embeddings using OpenAI API."""
    
    def __init__(self):
        """Initialize the OpenAI client with API key from settings."""
        try:
            self.client = openai.OpenAI(
                api_key=settings.openai_api_key,
                timeout=30.0
            )
            self.model_name = "text-embedding-3-small"
            self.dimension = settings.openai_embedding_dimension
            self.batch_size = 100  # OpenAI allows larger batches
            logger.info(f"Initialized OpenAI client with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def embed_texts(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            input_type: Type of input (not used by OpenAI but kept for compatibility)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts using {self.model_name}")
        
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    dimensions=self.dimension
                )
                all_embeddings.extend([res.embedding for res in response.data])
            except Exception as e:
                logger.error(f"Error generating OpenAI embeddings for a batch: {e}")
                # Add None placeholders for the failed batch
                all_embeddings.extend([None] * len(batch))

        # Filter out any None values from failed batches before returning
        successful_embeddings = [emb for emb in all_embeddings if emb is not None]
        if len(successful_embeddings) != len(texts):
            logger.warning(f"OpenAI embedding failed for {len(texts) - len(successful_embeddings)} texts.")

        return successful_embeddings

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates and assigns embeddings to a list of chunks for OpenAI.
        """
        if not chunks:
            return []

        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        if len(embeddings) != len(chunks):
            logger.error("Mismatch between number of chunks and generated OpenAI embeddings. Skipping assignment.")
            return []

        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
            
        return chunks
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        return self.embed_texts([query], input_type="search_query")[0]
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "provider": "OpenAI",
            "supports_multilingual": True,
            "languages": ["English", "Bengali", "100+ others"]
        }


class HybridEmbeddingService:
    """Service that combines both Cohere and OpenAI embeddings for better retrieval."""
    
    def __init__(self):
        """Initialize both embedding services."""
        self.cohere_service = CohereEmbeddingService()
        self.openai_service = OpenAIEmbeddingService()
        logger.info("Initialized hybrid embedding service with Cohere + OpenAI")
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate embeddings using both services.
        
        Returns:
            Dictionary with 'cohere' and 'openai' keys containing respective chunks
        """
        logger.info("Generating dual embeddings for chunks")
        
        # Create deep copies of chunks for each service
        cohere_chunks = [chunk.copy() for chunk in chunks]
        openai_chunks = [chunk.copy() for chunk in chunks]
        
        # Generate embeddings for each service
        cohere_chunks = self.cohere_service.embed_chunks(cohere_chunks)
        openai_chunks = self.openai_service.embed_chunks(openai_chunks)
        
        # Add embedding model info to each chunk
        for chunk in cohere_chunks:
            chunk['embedding_model'] = 'cohere'
        for chunk in openai_chunks:
            chunk['embedding_model'] = 'openai'
        
        return {
            "cohere": cohere_chunks,
            "openai": openai_chunks
        }
    
    def embed_query(self, query: str) -> Dict[str, List[float]]:
        """
        Generate query embeddings using both services.
        
        Returns:
            Dictionary with 'cohere' and 'openai' embeddings
        """
        return {
            "cohere": self.cohere_service.embed_query(query),
            "openai": self.openai_service.embed_query(query)
        }
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about both embedding models."""
        return {
            "cohere": self.cohere_service.get_embedding_info(),
            "openai": self.openai_service.get_embedding_info()
        }


def get_embedding_service():
    """Factory function to get the appropriate embedding service based on settings."""
    provider = settings.embedding_provider.lower()
    
    if provider == "openai":
        return OpenAIEmbeddingService()
    elif provider == "both":
        return HybridEmbeddingService()
    else:  # default to cohere
        return CohereEmbeddingService() 