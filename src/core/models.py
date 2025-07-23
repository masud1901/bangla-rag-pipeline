from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for the /query endpoint."""
    
    query: str = Field(
        ..., 
        description="The question to ask in English or Bengali",
        min_length=1,
        max_length=1000,
        example="অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
    )
    
    top_k: Optional[int] = Field(
        default=None,
        description="Number of relevant chunks to retrieve (uses default if not specified)",
        ge=1,
        le=20
    )
    
    include_sources: Optional[bool] = Field(
        default=True,
        description="Whether to include source information in the response"
    )

    # NEW: Optional language hint from the client ("en", "bn", or "auto").
    # The backend can ignore this if not needed, but accepting the field prevents
    # validation errors when the GUI includes it in the payload.
    language: Optional[str] = Field(
        default=None,
        description="Optional language hint ('en', 'bn', or 'auto')."
    )


class SourceChunk(BaseModel):
    """Model for a source text chunk."""
    
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="The text content of the chunk")
    page_number: int = Field(..., description="Page number in the source document")
    source_file: str = Field(..., description="Name of the source file")
    relevance_score: float = Field(..., description="Similarity score (0-1)")
    char_count: int = Field(..., description="Number of characters in the chunk")
    word_count: int = Field(..., description="Number of words in the chunk")


class QueryResponse(BaseModel):
    """Response model for the /query endpoint."""
    
    model_config = {"protected_namespaces": ()}
    
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The generated answer")
    
    # Source information
    source_chunks: List[SourceChunk] = Field(
        default=[], 
        description="Source chunks used to generate the answer"
    )
    
    # Metadata
    processing_time_ms: float = Field(..., description="Time taken to process the query in milliseconds")
    model_info: Dict[str, str] = Field(..., description="Information about the models used")
    
    # Optional fields based on request
    retrieval_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Statistics about the retrieval process"
    )


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""
    
    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")
    timestamp: datetime = Field(default_factory=datetime.now, description="Current timestamp")
    
    config: Dict[str, Any] = Field(..., description="Service configuration")
    
    # Service status
    services_status: Optional[Dict[str, str]] = Field(
        default=None,
        description="Status of external services"
    )


class ErrorResponse(BaseModel):
    """Response model for error cases."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class IndexStatsResponse(BaseModel):
    """Response model for index statistics."""
    
    index_name: str = Field(..., description="Name of the vector index")
    total_vectors: int = Field(..., description="Total number of vectors in the index")
    dimension: int = Field(..., description="Dimension of the vectors")
    index_fullness: float = Field(..., description="How full the index is (0-1)")
    
    embedding_model: str = Field(..., description="Model used for embeddings")
    last_updated: Optional[datetime] = Field(
        default=None,
        description="When the index was last updated"
    ) 