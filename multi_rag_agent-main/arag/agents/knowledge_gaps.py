"""
Knowledge Gaps Agent for the ARAG system.

This agent analyzes the collected knowledge to identify what information is still
missing to provide a complete answer to the user's query.
"""

import logging
from typing import List, Dict, Any, Optional

from arag.core.ai_client import AIClient

logger = logging.getLogger(__name__)

class KnowledgeGapsAgent:
    """
    Agent for identifying knowledge gaps.
    
    This agent produces both a list of knowledge gaps and a reflection on
    the current state of knowledge.
    """
    
    def __init__(self, ai_client: Optional[AIClient] = None):
        """
        Initialize the Knowledge Gaps Agent.
        
        Args:
            ai_client: AI client to use, or None to create a new one
        """
        self.ai_client = ai_client or AIClient()
        logger.info("Knowledge Gaps Agent initialized")
    
    def identify_knowledge_gaps(
        self, 
        knowledge_items: List[Dict[str, Any]],
        user_query: str
    ) -> Dict[str, Any]:
        """
        Identify gaps in the current knowledge.
        
        Args:
            knowledge_items: List of knowledge items to analyze
            user_query: Original user query for context
            
        Returns:
            Dictionary with knowledge gaps and reflection
        """
        logger.info(f"Identifying knowledge gaps in {len(knowledge_items)} knowledge items")
        
        # Use the AI client to identify knowledge gaps
        knowledge_gaps = self.ai_client.identify_knowledge_gaps(
            knowledge_items,
            user_query
        )
        
        logger.info(f"Identified {len(knowledge_gaps.get('knowledge_gaps', []))} knowledge gaps with completeness score {knowledge_gaps.get('completeness_score', 0)}")
        return knowledge_gaps