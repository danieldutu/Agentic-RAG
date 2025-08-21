"""
Missing Info Agent for the ARAG system.

This agent identifies references to external information within retrieved documents
that may be needed for a complete answer.
"""

import logging
from typing import List, Dict, Any, Optional

from arag.core.ai_client import AIClient

logger = logging.getLogger(__name__)

class MissingInfoAgent:
    """
    Agent for identifying references to external information.
    
    This agent detects when documentation refers to other manuals, procedures,
    or specifications not contained in the current knowledge.
    """
    
    def __init__(self, ai_client: Optional[AIClient] = None):
        """
        Initialize the Missing Info Agent.
        
        Args:
            ai_client: AI client to use, or None to create a new one
        """
        self.ai_client = ai_client or AIClient()
        logger.info("Missing Info Agent initialized")
    
    def identify_missing_info(
        self, 
        knowledge_items: List[Dict[str, Any]],
        user_query: str
    ) -> List[Dict[str, Any]]:
        """
        Identify missing information references in the knowledge items.
        
        Args:
            knowledge_items: List of knowledge items to analyze
            user_query: Original user query for context
            
        Returns:
            List of missing information references
        """
        logger.info(f"Identifying missing information in {len(knowledge_items)} knowledge items")
        
        # Use the AI client to identify missing information
        missing_info = self.ai_client.identify_missing_info(
            knowledge_items,
            user_query
        )
        
        logger.info(f"Identified {len(missing_info)} missing information references")
        return missing_info