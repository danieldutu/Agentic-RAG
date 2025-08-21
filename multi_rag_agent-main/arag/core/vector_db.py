"""
Vector database integration for the ARAG system.

This module handles interactions with the vector database for document retrieval and embedding.
"""

import logging
import requests
from typing import List, Dict, Any, Optional
import time
import json
import os
import weaviate
import numpy as np
from sentence_transformers import SentenceTransformer

from arag.config import VECTOR_DB_BASE_URL, VECTOR_DB_TYPE

logger = logging.getLogger(__name__)

class VectorDBClient:
    """
    Client for interacting with the vector database.
    
    This client provides methods to query the vector database for similar documents
    and to embed text for vector search.
    """
    
    def __init__(self, base_url: str = VECTOR_DB_BASE_URL):
        """
        Initialize the vector database client.
        
        Args:
            base_url: Base URL of the vector database API
        """
        self.base_url = base_url
        self.db_type = os.environ.get("VECTOR_DB_TYPE", VECTOR_DB_TYPE)
        self.class_name = os.environ.get("WEAVIATE_CLASS_NAME", "ARAG2")
        self.client = None
        
        # Initialize the embeddings model - using a model with 1024 dimensions to match Weaviate
        try:
            # Use BAAI/bge-large-en-v1.5 which has 1024 dimensions
            self.embedding_model = SentenceTransformer('BAAI/bge-large-en')
            logger.info("Initialized sentence-transformers model")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            self.embedding_model = None
        
        logger.info(f"Vector DB client initialized with base URL: {base_url}")
        
        # Initialize client based on DB type
        if self.db_type.lower() == "weaviate":
            try:
                # Use the v4 client API
                self.client = weaviate.connect_to_local(port=8765)
                logger.info(f"Connected to Weaviate at {base_url}")
            except Exception as e:
                logger.error(f"Failed to initialize Weaviate client: {str(e)}")
                self.client = None

    def query(
        self, 
        query_text: str, 
        limit: int = 10, 
        min_score: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Query the vector database for documents similar to the query text.
        
        Args:
            query_text: Query text to search for
            limit: Maximum number of results to return
            min_score: Minimum similarity score to include in results
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            List of documents with content, metadata, and similarity scores
        """
        if not query_text.strip():
            logger.warning("Empty query text provided to vector_db.query()")
            return []
            
        logger.info(f"Querying vector DB for: {query_text[:100]}...")
        
        # Handle different DB types
        if self.db_type.lower() == "weaviate":
            return self._query_weaviate(query_text, limit, min_score, max_retries, retry_delay)
        else:
            logger.error(f"Unsupported vector DB type: {self.db_type}")
            return []
            
    def _query_weaviate(
        self, 
        query_text: str, 
        limit: int = 10, 
        min_score: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Query Weaviate for documents similar to the query text.
        
        Args:
            query_text: Query text to search for
            limit: Maximum number of results to return
            min_score: Minimum similarity score to include in results
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            List of documents with content, metadata, and similarity scores
        """
        if not self.client:
            logger.error("Weaviate client not initialized")
            return []
            
        # Generate embedding for query text
        try:
            if self.embedding_model:
                vector = self.embedding_model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True).tolist()
            else:
                logger.error("Embedding model not available")
                return self._fallback_text_search(query_text, limit)
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return self._fallback_text_search(query_text, limit)
            
        # Try vector search first
        # If we get a dimension mismatch error, fall back to BM25 text search
        try:
            # Attempt 1: Use the collection API to perform a vector search 
            collection = self.client.collections.get(self.class_name)
            
            # Execute vector query
            query_result = collection.query.near_vector(
                vector=vector,
                limit=limit
            )
            
            # Process results
            documents = []
            for item in query_result.objects:
                documents.append({
                    "content": item.properties.get("text", ""),
                    "metadata": {
                        "source": item.properties.get("pdf_file", ""),
                        "page": item.properties.get("section_number", 0),
                        "chunk_id": item.properties.get("chunk_index", 0),
                        "additional": item.properties
                    },
                    "score": item.metadata.certainty if hasattr(item.metadata, "certainty") else min_score
                })
            
            logger.info(f"Found {len(documents)} documents with vector search")
            return documents
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"Vector search error: {error_str}")
            
            # Check for vector dimension mismatch
            if "vector lengths don't match" in error_str:
                logger.warning("Vector dimension mismatch detected, falling back to BM25 text search")
                return self._fallback_text_search(query_text, limit)
                
            # If max retries exceeded or other error, still try text search as fallback
            logger.warning("Vector search failed, falling back to BM25 text search")
            return self._fallback_text_search(query_text, limit)

    def _fallback_text_search(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Basic text search when vector search is not available.
        """
        try:
            collection = self.client.collections.get(self.class_name)
            query_result = collection.query.bm25(
                query=query_text,
                limit=limit
            )
            
            # Process results
            documents = []
            for item in query_result.objects:
                documents.append({
                    "content": item.properties.get("text", ""),
                    "metadata": {
                        "source": item.properties.get("pdf_file", ""),
                        "page": item.properties.get("section_number", 0),
                        "chunk_id": item.properties.get("chunk_index", 0),
                        "additional": item.properties
                    },
                    "score": 0.7  # Default score for text search
                })
            
            logger.info(f"Found {len(documents)} documents with text search")
            return documents
            
        except Exception as e:
            logger.error(f"Error in fallback text search: {str(e)}")
            return []

    def embed(
        self, 
        text: str,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[List[float]]:
        """
        Embed text using the vector database's embedding model.
        
        Args:
            text: Text to embed
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            Vector embedding for the text, or None on failure
        """
        if not text.strip():
            logger.warning("Empty text provided to vector_db.embed()")
            return None
            
        # No direct embedding endpoint in Weaviate Client API
        # This is handled by Weaviate's text2vec modules internally
        logger.warning("Direct embedding not supported for Weaviate")
        return None

    def batch_query(
        self, 
        queries: List[str], 
        limit_per_query: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple queries and combine the results.
        
        Args:
            queries: List of query texts
            limit_per_query: Maximum number of results per query
            min_score: Minimum similarity score to include in results
            
        Returns:
            Combined list of documents with content, metadata, and similarity scores
        """
        if not queries:
            logger.warning("Empty queries list provided to vector_db.batch_query()")
            return []
            
        logger.info(f"Batch querying vector DB with {len(queries)} queries")
        
        all_results = []
        seen_contents = set()
        
        for query in queries:
            results = self.query(query, limit=limit_per_query, min_score=min_score)
            
            # Deduplicate based on content
            for result in results:
                content = result.get("content", "")
                if content and content not in seen_contents:
                    seen_contents.add(content)
                    all_results.append(result)
        
        # Sort by score descending
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        logger.info(f"Batch query returned {len(all_results)} unique documents")
        return all_results

    def health_check(self) -> bool:
        """Check if the vector database is healthy and responsive."""
        logger.info(f"Performing health check for vector DB type: {self.db_type}")
        
        if self.db_type.lower() == "weaviate":
            try:
                if not self.client:
                    return False
                    
                # Check if Weaviate is responding using v4 API
                return self.client.is_ready()
            except Exception as e:
                logger.error(f"Weaviate health check failed: {str(e)}")
                return False
        else:
            logger.error(f"Unsupported vector DB type for health check: {self.db_type}")
            return False