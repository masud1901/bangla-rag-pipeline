from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    cohere_api_key: str
    pinecone_api_key: str
    openai_api_key: str
    
    # Pinecone Configuration
    pinecone_index_name: str = "multilingual-rag"
    pinecone_environment: str = "us-east-1-aws"
    
    # Application Configuration
    app_name: str = "Multilingual RAG System"
    debug: bool = False
    
    # Embedding Configuration
    embedding_provider: str = "cohere"  # Options: "cohere", "openai", "both"
    embedding_dimension: int = 1024  # Cohere embed-multilingual-v3.0 dimension
    openai_embedding_dimension: int = 1536  # OpenAI text-embedding-3-small dimension
    
    # Vector Database Configuration
    top_k_retrieval: int = 12  # Number of chunks to retrieve (increased for better coverage)
    relevance_threshold: float = 0.5  # Minimum relevance score (lowered from 0.6)
    
    # Reranking Configuration
    enable_reranking: bool = True  # Enable Cohere rerank for better accuracy
    rerank_top_k: int = 6  # Final number of chunks after reranking
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings() 