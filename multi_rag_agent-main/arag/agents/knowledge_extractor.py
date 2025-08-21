"""
Knowledge Extractor Agent for the ARAG system.

This agent extracts structured, factual information from retrieved document chunks
using a comprehensive extraction methodology.
"""

import logging
from typing import List, Dict, Any, Optional

from arag.core.ai_client import AIClient

logger = logging.getLogger(__name__)

class KnowledgeExtractorAgent:
    """
    Agent for extracting structured knowledge from document chunks.
    
    This agent follows specific knowledge extraction principles:
    1. Problem identification first
    2. Query-focused extraction
    3. Prioritization of important facts
    4. Maximum comprehensiveness
    5. Complete context preservation
    6. Full table extraction when relevant
    7. Proper image reference handling
    8. Knowledge items should be self-contained
    9. Source attribution (page numbers, section references)
    10. Token limitation
    """
    
    def __init__(self, ai_client: Optional[AIClient] = None):
        """
        Initialize the Knowledge Extractor Agent.
        
        Args:
            ai_client: AI client to use, or None to create a new one
        """
        self.ai_client = ai_client or AIClient()
        logger.info("Knowledge Extractor Agent initialized")
    
    def extract_knowledge(
        self, 
        document_chunks: List[Dict[str, Any]],
        user_query: str
    ) -> List[Dict[str, Any]]:
        """
        Extract structured knowledge from document chunks.
        
        Args:
            document_chunks: List of document chunks with their content and metadata
            user_query: Original user query for context
            
        Returns:
            List of extracted knowledge items
        """
        logger.info(f"Extracting knowledge from {len(document_chunks)} document chunks")
        
        # Use the AI client to extract knowledge
        knowledge_items = self.ai_client.extract_knowledge(
            document_chunks,
            user_query
        )
        
        logger.info(f"Extracted {len(knowledge_items)} knowledge items")
        return knowledge_items