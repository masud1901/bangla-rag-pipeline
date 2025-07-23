import cohere
import logging
from typing import List, Dict, Any
from src.core.config import settings

logger = logging.getLogger(__name__)


class CohereRerankService:
    """Service for re-ranking retrieved chunks using Cohere's rerank API."""
    
    def __init__(self):
        """Initialize the Cohere client for reranking."""
        try:
            self.client = cohere.Client(settings.cohere_api_key)
            self.model_name = "rerank-multilingual-v3.0"
            logger.info(f"Initialized Cohere rerank service with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Cohere rerank client: {e}")
            raise
    
    def rerank_chunks(self, query: str, chunks: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Re-rank chunks using Cohere's rerank API for better relevance.
        
        Args:
            query: The search query
            chunks: List of retrieved chunks to rerank
            top_k: Number of top chunks to return after reranking
            
        Returns:
            List of reranked chunks with updated relevance scores
        """
        if not chunks:
            return []
        
        if top_k is None:
            top_k = settings.rerank_top_k
        
        # Prepare documents for reranking
        documents = [chunk.get('text', '') for chunk in chunks]
        
        try:
            logger.info(f"Reranking {len(documents)} chunks with query: '{query[:50]}...'")
            
            # Call Cohere rerank API
            response = self.client.rerank(
                model=self.model_name,
                query=query,
                documents=documents,
                top_k=min(top_k, len(documents)),
                return_documents=False  # We already have the documents
            )
            
            # Create reranked chunks with new scores
            reranked_chunks = []
            for result in response.results:
                original_chunk = chunks[result.index].copy()
                # Update the relevance score with rerank score
                original_chunk['score'] = result.relevance_score
                original_chunk['rerank_score'] = result.relevance_score
                original_chunk['original_index'] = result.index
                reranked_chunks.append(original_chunk)
            
            logger.info(f"Reranked {len(reranked_chunks)} chunks. Top score: {reranked_chunks[0]['rerank_score']:.3f}")
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback to original chunks if reranking fails
            logger.warning("Falling back to original chunk ordering")
            return chunks[:top_k]
    
    def get_rerank_info(self) -> Dict[str, Any]:
        """Get information about the rerank model."""
        return {
            "model_name": self.model_name,
            "provider": "Cohere",
            "supports_multilingual": True,
            "max_documents": 1000,
            "languages": ["English", "Bengali", "100+ others"]
        } 