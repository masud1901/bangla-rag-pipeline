import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.core.config import settings
from src.core.models import QueryRequest, QueryResponse, HealthResponse, ErrorResponse
from src.services.embedding_service import CohereEmbeddingService
from src.services.vector_store_service import PineconeVectorStore
from src.services.llm_service import OpenAILLMService
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
        app_state["embedding_service"] = CohereEmbeddingService()
        app_state["vector_store"] = PineconeVectorStore()
        app_state["llm_service"] = OpenAILLMService()

        app_state["qa_pipeline"] = QAPipeline(
            embedding_service=app_state["embedding_service"],
            vector_store=app_state["vector_store"],
            llm_service=app_state["llm_service"]
        )
        logger.info("All services initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)
        # You might want to handle this more gracefully, e.g., by preventing the app from starting
        
    yield
    
    logger.info("Application shutdown.")
    app_state.clear()


app = FastAPI(
    title=settings.app_name,
    description="A RAG system for Bengali and English queries.",
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
    }

    is_healthy = all(status == "ok" for status in services_status.values())

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        service=settings.app_name,
        config={
            "pinecone_index": settings.pinecone_index_name,
            "embedding_dimension": settings.embedding_dimension,
            "top_k_retrieval": settings.top_k_retrieval,
            "llm_model": app_state.get("llm_service").model_name if app_state.get("llm_service") else "N/A"
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
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal Server Error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 