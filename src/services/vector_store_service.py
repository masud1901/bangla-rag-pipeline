import chromadb
import logging
import time
from typing import List, Dict, Any, Optional
from src.core.config import settings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Service for managing vector storage and retrieval using ChromaDB with dual embedding support."""
    
    def __init__(self):
        """Initialize ChromaDB client and connect to collections."""
        try:
            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(path="/app/chroma_db")
            self.base_collection_name = settings.pinecone_index_name
            
            # Initialize collections for different embedding providers
            self.collections = {}
            self._init_collections()
            
            logger.info(f"Connected to ChromaDB collections: {list(self.collections.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _init_collections(self):
        """Initialize collections based on embedding provider settings."""
        provider = settings.embedding_provider.lower()
        
        if provider == "cohere" or provider == "both":
            self.collections["cohere"] = self.client.get_or_create_collection(
                name=f"{self.base_collection_name}-cohere",
                metadata={"hnsw:space": "cosine", "embedding_provider": "cohere"}
            )
        
        if provider == "openai" or provider == "both":
            self.collections["openai"] = self.client.get_or_create_collection(
                name=f"{self.base_collection_name}-openai",
                metadata={"hnsw:space": "cosine", "embedding_provider": "openai"}
            )
        
        # For backward compatibility, also support the original collection
        if provider == "cohere" and "cohere" not in self.collections:
            self.collections["cohere"] = self.client.get_or_create_collection(
                name=self.base_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def upsert_chunks(self, chunks_with_embeddings, collection_name: str = "cohere") -> Dict[str, Any]:
        """
        Upload chunks with embeddings to specified ChromaDB collection.
        
        Args:
            chunks_with_embeddings: List of chunks with embedding vectors
            collection_name: Which collection to use ("cohere" or "openai")
            
        Returns:
            Upsert operation results
        """
        if not chunks_with_embeddings:
            return {"upserted_count": 0}
        
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not initialized")
        
        collection = self.collections[collection_name]
        
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
                    "embedding_model": chunk.get("embedding_model", "unknown"),
                    "collection_type": collection_name
                })
            
            # Upsert to ChromaDB
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully upserted {len(chunks_with_embeddings)} chunks to {collection_name} collection")
            
            return {
                "upserted_count": len(chunks_with_embeddings),
                "collection_name": collection_name,
                "total_chunks": len(chunks_with_embeddings)
            }
            
        except Exception as e:
            logger.error(f"Error upserting chunks to {collection_name} collection: {e}")
            raise
    
    def upsert_dual_embeddings(self, dual_chunks: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Upload chunks with dual embeddings to both collections.
        
        Args:
            dual_chunks: Dictionary with 'cohere' and 'openai' chunks
            
        Returns:
            Combined upsert operation results
        """
        results = {}
        
        if "cohere" in dual_chunks and "cohere" in self.collections:
            results["cohere"] = self.upsert_chunks(dual_chunks["cohere"], "cohere")
        
        if "openai" in dual_chunks and "openai" in self.collections:
            results["openai"] = self.upsert_chunks(dual_chunks["openai"], "openai")
        
        return results
    
    def search_similar_chunks(self, query_embedding, collection_name: str = "cohere", top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using query embedding in specified collection.
        
        Args:
            query_embedding: Query vector to search with
            collection_name: Which collection to search ("cohere" or "openai")
            top_k: Number of top results to return
            
        Returns:
            List of similar chunks with scores
        """
        if top_k is None:
            top_k = settings.top_k_retrieval
        
        if collection_name not in self.collections:
            logger.warning(f"Collection '{collection_name}' not found, falling back to available collections")
            collection_name = list(self.collections.keys())[0]
        
        collection = self.collections[collection_name]
        
        try:
            # Search in ChromaDB
            results = collection.query(
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
            
            logger.info(f"Found {len(similar_chunks)} similar chunks in {collection_name} collection")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error searching similar chunks in {collection_name} collection: {e}")
            raise
    
    def search_dual_embeddings(self, dual_query_embeddings: Dict[str, List[float]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search using both embedding types and merge results.
        
        Args:
            dual_query_embeddings: Dictionary with 'cohere' and 'openai' embeddings
            top_k: Number of results per collection
            
        Returns:
            Merged and deduplicated list of chunks
        """
        if top_k is None:
            top_k = settings.top_k_retrieval
        
        all_chunks = []
        chunk_seen = set()
        
        # Search in each available collection
        for collection_name, query_embedding in dual_query_embeddings.items():
            if collection_name in self.collections:
                chunks = self.search_similar_chunks(query_embedding, collection_name, top_k)
                
                # Add collection source and deduplicate by chunk_id
                for chunk in chunks:
                    chunk_id = chunk.get('chunk_id')
                    if chunk_id not in chunk_seen:
                        chunk['retrieval_source'] = collection_name
                        all_chunks.append(chunk)
                        chunk_seen.add(chunk_id)
        
        # Sort by relevance score (highest first)
        all_chunks.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Return top results
        final_chunks = all_chunks[:top_k]
        logger.info(f"Merged dual embedding search: {len(final_chunks)} unique chunks from {len(chunk_seen)} total")
        
        return final_chunks
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all vector collections.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = {}
            total_vectors = 0
            
            for name, collection in self.collections.items():
                count = collection.count()
                stats[name] = {
                    "total_vectors": count,
                    "collection_name": f"{self.base_collection_name}-{name}",
                }
                total_vectors += count
            
            stats["summary"] = {
                "total_vectors": total_vectors,
                "collections": list(self.collections.keys()),
                "embedding_provider": settings.embedding_provider
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def delete_all_vectors(self, collection_name: str = None) -> Dict[str, Any]:
        """
        Delete all vectors from specified collection or all collections.
        
        Args:
            collection_name: Specific collection to clear, or None for all
            
        Returns:
            Deletion operation results
        """
        try:
            results = {}
            
            collections_to_clear = [collection_name] if collection_name else list(self.collections.keys())
            
            for name in collections_to_clear:
                if name in self.collections:
                    # Delete and recreate the collection
                    full_name = f"{self.base_collection_name}-{name}" if name != "cohere" else self.base_collection_name
                    self.client.delete_collection(full_name)
                    
                    # Recreate the collection
                    if name == "cohere":
                        self.collections[name] = self.client.create_collection(
                            name=full_name,
                            metadata={"hnsw:space": "cosine"}
                        )
                    else:
                        self.collections[name] = self.client.create_collection(
                            name=full_name,
                            metadata={"hnsw:space": "cosine", "embedding_provider": name}
                        )
                    
                    results[name] = {"deleted_count": "all", "collection_name": full_name}
                    logger.info(f"Cleared all vectors from {name} collection")
            
            return results
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise


# Alias for backward compatibility
PineconeVectorStore = ChromaVectorStore 