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
    
    # Vector Database Configuration
    embedding_dimension: int = 1024  # Cohere embed-multilingual-v3.0 dimension
    top_k_retrieval: int = 8  # Number of chunks to retrieve (increased from 5)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings() 