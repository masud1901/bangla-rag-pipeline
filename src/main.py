import logging
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Disable ChromaDB telemetry globally
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
os.environ["CHROMA_SERVER_TELEMETRY_ENABLED"] = "False"

from src.core.config import settings
from src.core.models import QueryRequest, QueryResponse, HealthResponse, ErrorResponse
from src.services.embedding_service import get_embedding_service
from src.services.vector_store_service import ChromaVectorStore
from src.services.llm_service import OpenAILLMService
from src.services.rerank_service import CohereRerankService
from src.pipeline.qa import QAPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Application state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Initializes all necessary services and the QA pipeline.
    """
    logger.info("Application startup: Initializing services...")
    try:
        app_state["embedding_service"] = get_embedding_service()
        app_state["vector_store"] = ChromaVectorStore()
        app_state["llm_service"] = OpenAILLMService()
        
        # Initialize rerank service if enabled
        if settings.enable_reranking:
            try:
                app_state["rerank_service"] = CohereRerankService()
                logger.info("Rerank service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize rerank service: {e}")
                app_state["rerank_service"] = None
        else:
            app_state["rerank_service"] = None

        app_state["qa_pipeline"] = QAPipeline(
            embedding_service=app_state["embedding_service"],
            vector_store=app_state["vector_store"],
            llm_service=app_state["llm_service"]
        )
        logger.info(f"All services initialized successfully with provider: {settings.embedding_provider}")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)
        # You might want to handle this more gracefully, e.g., by preventing the app from starting
        
    yield
    
    logger.info("Application shutdown.")
    app_state.clear()


app = FastAPI(
    title=settings.app_name,
    description="A RAG system for Bengali and English queries with hybrid embeddings and reranking.",
    version="1.0.0",
    lifespan=lifespan,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get the QA pipeline
def get_qa_pipeline():
    return app_state.get("qa_pipeline")

@app.get("/", tags=["General"])
async def root():
    return {"message": f"{settings.app_name} is running"}

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Performs a health check of the service and its dependencies.
    """
    services_status = {
        "embedding_service": "ok" if app_state.get("embedding_service") else "error",
        "vector_store": "ok" if app_state.get("vector_store") else "error",
        "llm_service": "ok" if app_state.get("llm_service") else "error",
        "rerank_service": "ok" if app_state.get("rerank_service") else ("disabled" if not settings.enable_reranking else "error")
    }

    is_healthy = all(status in ["ok", "disabled"] for status in services_status.values())

    # Get vector store stats
    vector_store = app_state.get("vector_store")
    index_stats = vector_store.get_index_stats() if vector_store else {}

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        service=settings.app_name,
        config={
            "embedding_provider": settings.embedding_provider,
            "embedding_dimension": settings.embedding_dimension,
            "openai_embedding_dimension": settings.openai_embedding_dimension,
            "top_k_retrieval": settings.top_k_retrieval,
            "relevance_threshold": settings.relevance_threshold,
            "rerank_enabled": settings.enable_reranking,
            "rerank_top_k": settings.rerank_top_k,
            "llm_model": app_state.get("llm_service").model_name if app_state.get("llm_service") else "N/A",
            "index_stats": index_stats
        },
        services_status=services_status
    )

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_endpoint(request: QueryRequest, qa_pipeline: QAPipeline = Depends(get_qa_pipeline)):
    """
    The main endpoint to ask a question to the RAG system.
    """
    if not qa_pipeline:
        raise HTTPException(
            status_code=503,
            detail={"error": "Service Unavailable", "message": "The QA pipeline is not initialized."}
        )
        
    try:
        response = qa_pipeline.answer_query(request)
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error for query '{request.query}': {e}")
        raise HTTPException(
            status_code=503,
            detail={"error": "External API Error", "message": "Failed to connect to external services. Please try again."}
        )
    except ValueError as e:
        logger.error(f"Validation error for query '{request.query}': {e}")
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid Request", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Unexpected error processing query '{request.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal Server Error", "message": "An unexpected error occurred. Please try again later."}
        )

@app.post("/debug/retrieve", tags=["Debug"])
async def debug_retrieve_endpoint(request: QueryRequest):
    """
    Debug endpoint to inspect retrieval and reranking scores without generating an answer.
    """
    try:
        embedding_service = app_state.get("embedding_service")
        vector_store = app_state.get("vector_store")
        rerank_service = app_state.get("rerank_service")
        
        if not embedding_service or not vector_store:
            raise HTTPException(status_code=503, detail="Services not available")
        
        # Get embeddings
        provider = settings.embedding_provider.lower()
        if provider == "both":
            dual_embeddings = embedding_service.embed_query(request.query)
            retrieved_chunks = vector_store.search_dual_embeddings(
                dual_query_embeddings=dual_embeddings,
                top_k=request.top_k or settings.top_k_retrieval
            )
        else:
            query_embedding = embedding_service.embed_query(request.query)
            collection_name = "openai" if provider == "openai" else "cohere"
            retrieved_chunks = vector_store.search_similar_chunks(
                query_embedding=query_embedding,
                collection_name=collection_name,
                top_k=request.top_k or settings.top_k_retrieval
            )
        
        # Apply reranking if available
        reranked_chunks = []
        if rerank_service and retrieved_chunks:
            try:
                reranked_chunks = rerank_service.rerank_chunks(
                    query=request.query,
                    chunks=retrieved_chunks,
                    top_k=settings.rerank_top_k
                )
            except Exception as e:
                logger.warning(f"Reranking failed in debug endpoint: {e}")
        
        # Format debug response
        debug_info = {
            "query": request.query,
            "embedding_provider": provider,
            "retrieved_chunks_count": len(retrieved_chunks),
            "reranked_chunks_count": len(reranked_chunks),
            "settings": {
                "top_k_retrieval": settings.top_k_retrieval,
                "relevance_threshold": settings.relevance_threshold,
                "rerank_enabled": settings.enable_reranking,
                "rerank_top_k": settings.rerank_top_k
            },
            "retrieved_chunks": [
                {
                    "chunk_id": chunk.get('chunk_id'),
                    "page_number": chunk.get('page_number'),
                    "relevance_score": chunk.get('score', 0.0),
                    "text_preview": chunk.get('text', '')[:200] + "..." if len(chunk.get('text', '')) > 200 else chunk.get('text', ''),
                    "retrieval_source": chunk.get('retrieval_source', 'unknown')
                }
                for chunk in retrieved_chunks[:10]  # Show top 10
            ],
            "reranked_chunks": [
                {
                    "chunk_id": chunk.get('chunk_id'),
                    "page_number": chunk.get('page_number'),
                    "original_score": chunk.get('score', 0.0),
                    "rerank_score": chunk.get('rerank_score', 0.0),
                    "text_preview": chunk.get('text', '')[:200] + "..." if len(chunk.get('text', '')) > 200 else chunk.get('text', ''),
                    "original_index": chunk.get('original_index')
                }
                for chunk in reranked_chunks
            ]
        }
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Error in debug retrieve endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal Server Error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 