import chromadb
import logging
import time
from typing import List, Dict, Any, Optional
from src.core.config import settings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Service for managing vector storage and retrieval using ChromaDB."""
    
    def __init__(self):
        """Initialize ChromaDB client and connect to collection."""
        try:
            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(path="/app/chroma_db")
            self.collection_name = settings.pinecone_index_name  # Reuse the same config
            self.dimension = settings.embedding_dimension
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Connected to ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def upsert_chunks(self, chunks_with_embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Upload chunks with embeddings to ChromaDB.
        
        Args:
            chunks_with_embeddings: List of chunks with embedding vectors
            
        Returns:
            Upsert operation results
        """
        if not chunks_with_embeddings:
            return {"upserted_count": 0}
        
        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for chunk in chunks_with_embeddings:
                chunk_id = f"chunk_{chunk['chunk_id']}"
                ids.append(chunk_id)
                embeddings.append(chunk["embedding"])
                documents.append(chunk["text"])
                metadatas.append({
                    "chunk_id": chunk["chunk_id"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "source_file": chunk["source_file"],
                    "char_count": chunk["char_count"],
                    "word_count": chunk["word_count"],
                    "embedding_model": chunk.get("embedding_model", "unknown")
                })
            
            # Upsert to ChromaDB
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully upserted {len(chunks_with_embeddings)} chunks to ChromaDB")
            
            return {
                "upserted_count": len(chunks_with_embeddings),
                "collection_name": self.collection_name,
                "total_chunks": len(chunks_with_embeddings)
            }
            
        except Exception as e:
            logger.error(f"Error upserting chunks to ChromaDB: {e}")
            raise
    
    def search_similar_chunks(self, query_embedding: List[float], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using query embedding.
        
        Args:
            query_embedding: Query vector to search with
            top_k: Number of top results to return
            
        Returns:
            List of similar chunks with scores
        """
        if top_k is None:
            top_k = settings.top_k_retrieval
        
        try:
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            similar_chunks = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}

                    # Flatten metadata for easier downstream consumption
                    chunk_data = {
                        **metadata,
                        "text": doc,
                        "score": 1 - results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                    }
                    similar_chunks.append(chunk_data)
            
            logger.info(f"Found {len(similar_chunks)} similar chunks")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error searching similar chunks in ChromaDB: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "total_vectors": count,
                "collection_name": self.collection_name,
                "dimension": self.dimension
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def delete_all_vectors(self) -> Dict[str, Any]:
        """
        Delete all vectors from the collection.
        
        Returns:
            Deletion operation results
        """
        try:
            # ChromaDB doesn't have a simple "delete all" method
            # We'll recreate the collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Cleared all vectors from collection: {self.collection_name}")
            return {"deleted_count": "all", "collection_name": self.collection_name}
            
        except Exception as e:
            logger.error(f"Error deleting all vectors: {e}")
            raise


# Alias for backward compatibility
PineconeVectorStore = ChromaVectorStore 