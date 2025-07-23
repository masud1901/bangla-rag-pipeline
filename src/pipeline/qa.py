import time
import logging
from typing import Dict, Any, List

from src.core.config import settings
from src.core.models import QueryRequest, QueryResponse, SourceChunk
from src.services.embedding_service import get_embedding_service
from src.services.vector_store_service import ChromaVectorStore
from src.services.llm_service import OpenAILLMService
from src.services.rerank_service import CohereRerankService

logger = logging.getLogger(__name__)

class QAPipeline:
    """
    Orchestrates the entire Question-Answering pipeline with hybrid retrieval and reranking.
    """
    def __init__(self, embedding_service=None, vector_store=None, llm_service: OpenAILLMService = None):
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or ChromaVectorStore()
        self.llm_service = llm_service
        
        # Initialize reranking service if enabled
        self.rerank_service = None
        if settings.enable_reranking:
            try:
                self.rerank_service = CohereRerankService()
                logger.info("Reranking service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize reranking service: {e}")
        
        logger.info(f"QA Pipeline initialized with provider: {settings.embedding_provider}")

    def answer_query(self, request: QueryRequest) -> QueryResponse:
        """
        Processes a user query and returns a comprehensive answer using hybrid retrieval and reranking.
        
        Args:
            request: The user's query request object.

        Returns:
            A QueryResponse object with the answer and source details.
        """
        start_time = time.time()
        logger.info(f"Received query: '{request.query}'")

        # 1. Embed the user's query using appropriate service
        provider = settings.embedding_provider.lower()
        
        if provider == "both":
            # Hybrid embedding approach
            dual_embeddings = self.embedding_service.embed_query(request.query)
            logger.info(f"Generated dual query embeddings")
            
            # 2. Retrieve from both collections and merge
            retrieved_chunks = self.vector_store.search_dual_embeddings(
                dual_query_embeddings=dual_embeddings,
                top_k=request.top_k or settings.top_k_retrieval
            )
        else:
            # Single embedding approach
            query_embedding = self.embedding_service.embed_query(request.query)
            logger.info(f"Generated query embedding with {len(query_embedding)} dimensions")
            
            # 2. Retrieve from appropriate collection
            collection_name = "openai" if provider == "openai" else "cohere"
            retrieved_chunks = self.vector_store.search_similar_chunks(
                query_embedding=query_embedding,
                collection_name=collection_name,
                top_k=request.top_k or settings.top_k_retrieval
            )
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks from vector store")

        # 3. Apply reranking if enabled and chunks available
        if self.rerank_service and retrieved_chunks:
            try:
                reranked_chunks = self.rerank_service.rerank_chunks(
                    query=request.query,
                    chunks=retrieved_chunks,
                    top_k=settings.rerank_top_k
                )
                logger.info(f"Reranked to {len(reranked_chunks)} high-quality chunks")
                final_chunks = reranked_chunks
            except Exception as e:
                logger.warning(f"Reranking failed, using original chunks: {e}")
                final_chunks = retrieved_chunks[:settings.rerank_top_k]
        else:
            # Apply relevance filtering without reranking
            filtered_chunks = []
            for chunk in retrieved_chunks:
                score = chunk.get('score', 0.0)
                if score > settings.relevance_threshold:
                    filtered_chunks.append(chunk)
            
            if not filtered_chunks:
                logger.warning(f"No high-quality chunks found (all scores < {settings.relevance_threshold}). Using top chunk anyway.")
                filtered_chunks = retrieved_chunks[:1] if retrieved_chunks else []
            
            final_chunks = filtered_chunks[:settings.rerank_top_k]
            logger.info(f"Filtered to {len(final_chunks)} chunks above threshold {settings.relevance_threshold}")

        if final_chunks:
            logger.info(f"Top chunk score: {final_chunks[0].get('score', 0.0):.3f}")

        # 4. Generate a grounded answer using the LLM (or return context if LLM disabled)
        if self.llm_service:
            answer_text, usage_info = self.llm_service.generate_answer(
                query=request.query,
                context_chunks=final_chunks
            )
        else:
            # Return the retrieved context as the answer if LLM is disabled
            context_texts = [chunk.get('text', '') for chunk in final_chunks[:3]]  # Top 3 chunks
            answer_text = f"[LLM Service Disabled] Based on the retrieved context:\n\n" + "\n\n---\n\n".join(context_texts)
            usage_info = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        
        # 5. Format the response
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000

        source_chunks: List[SourceChunk] = []
        if request.include_sources:
            for chunk in final_chunks:
                source_chunks.append(SourceChunk(
                    chunk_id=str(chunk.get('chunk_id', 'N/A')),  # Convert to string for Pydantic validation
                    text=chunk.get('text', ''),
                    page_number=chunk.get('page_number', 0),
                    source_file=chunk.get('source_file', 'N/A'),
                    relevance_score=chunk.get('score', 0.0),
                    char_count=chunk.get('char_count', 0),
                    word_count=chunk.get('word_count', 0),
                ))

        # Get model info based on embedding provider
        if provider == "both":
            model_info = {
                "embedding_model": "Cohere + OpenAI Hybrid",
                "llm": self.llm_service.get_model_info()['model_name'] if self.llm_service else "Disabled",
                "rerank_model": self.rerank_service.get_rerank_info()['model_name'] if self.rerank_service else "Disabled"
            }
        else:
            embedding_info = self.embedding_service.get_embedding_info()
            model_info = {
                "embedding_model": embedding_info['model_name'],
                "llm": self.llm_service.get_model_info()['model_name'] if self.llm_service else "Disabled",
                "rerank_model": self.rerank_service.get_rerank_info()['model_name'] if self.rerank_service else "Disabled"
            }

        response = QueryResponse(
            question=request.query,
            answer=answer_text,
            source_chunks=source_chunks,
            processing_time_ms=processing_time_ms,
            model_info=model_info,
            retrieval_stats={
                "retrieved_chunks_count": len(retrieved_chunks),
                "final_chunks_count": len(final_chunks),
                "top_k": request.top_k or settings.top_k_retrieval,
                "rerank_enabled": settings.enable_reranking,
                "relevance_threshold": settings.relevance_threshold,
                "token_usage": usage_info,
            }
        )
        
        logger.info(f"Query processed in {processing_time_ms:.2f} ms. Final chunks: {len(final_chunks)}, Total tokens: {usage_info.get('total_tokens', 0)}")
        return response 