"""
Query Rewriter Agent for the ARAG system.

This agent transforms user questions into multiple optimized search queries 
to maximize retrieval effectiveness.
"""

import logging
from typing import List, Optional

from arag.core.ai_client import AIClient

logger = logging.getLogger(__name__)

class QueryRewriterAgent:
    """
    Agent for rewriting user queries into optimized search queries.
    
    This agent analyzes technical terminology, identifies key concepts,
    and generates variations of the query to ensure comprehensive coverage.
    """
    
    def __init__(self, ai_client: Optional[AIClient] = None):
        """
        Initialize the Query Rewriter Agent.
        
        Args:
            ai_client: AI client to use, or None to create a new one
        """
        self.ai_client = ai_client or AIClient()
        logger.info("Query Rewriter Agent initialized")
    
    def rewrite_query(
        self, 
        original_query: str,
        max_variations: int = 3
    ) -> List[str]:
        """
        Rewrite the original query into multiple optimized search queries.
        
        Args:
            original_query: Original user query
            max_variations: Maximum number of query variations to generate
            
        Returns:
            List of rewritten query strings
        """
        logger.info(f"Rewriting query: {original_query}")
        
        # Use the AI client to generate rewritten queries
        rewritten_queries = self.ai_client.generate_query_rewrites(
            original_query,
            max_variations=max_variations
        )
        
        logger.info(f"Generated {len(rewritten_queries)} query variations")
        return rewritten_queries