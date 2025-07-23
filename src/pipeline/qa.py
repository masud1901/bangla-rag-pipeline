import time
import logging
from typing import Dict, Any, List

from src.core.config import settings
from src.core.models import QueryRequest, QueryResponse, SourceChunk
from src.services.embedding_service import CohereEmbeddingService
from src.services.vector_store_service import PineconeVectorStore
from src.services.llm_service import OpenAILLMService

logger = logging.getLogger(__name__)

class QAPipeline:
    """
    Orchestrates the entire Question-Answering pipeline.
    """
    def __init__(self, embedding_service: CohereEmbeddingService, vector_store: PineconeVectorStore, llm_service: OpenAILLMService = None):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service

    def answer_query(self, request: QueryRequest) -> QueryResponse:
        """
        Processes a user query and returns a comprehensive answer.
        
        Args:
            request: The user's query request object.

        Returns:
            A QueryResponse object with the answer and source details.
        """
        start_time = time.time()
        logger.info(f"Received query: '{request.query}'")

        # 1. Embed the user's query
        query_embedding = self.embedding_service.embed_query(request.query)
        logger.info(f"Generated query embedding with {len(query_embedding)} dimensions")

        # 2. Retrieve relevant chunks from the vector store
        retrieved_chunks = self.vector_store.search_similar_chunks(
            query_embedding=query_embedding, 
            top_k=request.top_k or settings.top_k_retrieval
        )
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks from vector store")

        # 3. Filter chunks by relevance score (only use high-quality matches)
        filtered_chunks = []
        for chunk in retrieved_chunks:
            score = chunk.get('score', 0.0)
            if score > 0.6:  # Only use chunks with relevance score > 0.6
                filtered_chunks.append(chunk)
        
        if not filtered_chunks:
            logger.warning(f"No high-quality chunks found (all scores < 0.6). Using top chunk anyway.")
            filtered_chunks = retrieved_chunks[:1]  # Use at least the top result
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks, filtered to {len(filtered_chunks)} high-quality chunks")
        logger.info(f"Top chunk score: {filtered_chunks[0].get('score', 0.0) if filtered_chunks else 'N/A'}")

        # 4. Generate a grounded answer using the LLM (or return context if LLM disabled)
        if self.llm_service:
            answer_text, usage_info = self.llm_service.generate_answer(
                query=request.query,
                context_chunks=filtered_chunks
            )
        else:
            # Return the retrieved context as the answer if LLM is disabled
            context_texts = [chunk.get('text', '') for chunk in filtered_chunks[:3]]  # Top 3 chunks
            answer_text = f"[LLM Service Disabled] Based on the retrieved context:\n\n" + "\n\n---\n\n".join(context_texts)
            usage_info = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        
        # 5. Format the response
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000

        source_chunks: List[SourceChunk] = []
        if request.include_sources:
            for chunk in filtered_chunks:
                source_chunks.append(SourceChunk(
                    chunk_id=str(chunk.get('chunk_id', 'N/A')),  # Convert to string for Pydantic validation
                    text=chunk.get('text', ''),
                    page_number=chunk.get('page_number', 0),
                    source_file=chunk.get('source_file', 'N/A'),
                    relevance_score=chunk.get('score', 0.0),
                    char_count=chunk.get('char_count', 0),
                    word_count=chunk.get('word_count', 0),
                ))

        model_info = {
            "embedding_model": self.embedding_service.get_embedding_info()['model_name'],
            "llm": self.llm_service.get_model_info()['model_name'] if self.llm_service else "Disabled",
        }

        response = QueryResponse(
            question=request.query,
            answer=answer_text,
            source_chunks=source_chunks,
            processing_time_ms=processing_time_ms,
            model_info=model_info,
            retrieval_stats={
                "retrieved_chunks_count": len(retrieved_chunks),
                "top_k": request.top_k or settings.top_k_retrieval,
                "token_usage": usage_info,
            }
        )
        
        logger.info(f"Query processed in {processing_time_ms:.2f} ms. Total tokens: {usage_info.get('total_tokens', 0)}")
        return response 