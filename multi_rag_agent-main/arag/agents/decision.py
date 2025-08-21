"""
Decision Agent for the ARAG system.

This agent determines if sufficient information has been gathered to answer the query
effectively or if additional retrieval iterations are needed.
"""

import logging
from typing import List, Dict, Any, Optional

from arag.core.ai_client import AIClient
from arag.config import MAX_RETRIEVAL_ITERATIONS

logger = logging.getLogger(__name__)

class DecisionAgent:
    """
    Agent for making decisions about the retrieval process.
    
    This agent acts as a gatekeeper for moving to the answer generation phase.
    """
    
    def __init__(self, ai_client: Optional[AIClient] = None):
        """
        Initialize the Decision Agent.
        
        Args:
            ai_client: AI client to use, or None to create a new one
        """
        self.ai_client = ai_client or AIClient()
        logger.info("Decision Agent initialized")
    
    def make_decision(
        self, 
        knowledge_items: List[Dict[str, Any]],
        knowledge_gaps: Dict[str, Any],
        iteration_count: int,
        max_iterations: int = MAX_RETRIEVAL_ITERATIONS
    ) -> Dict[str, Any]:
        """
        Determine if sufficient information has been gathered.
        
        Args:
            knowledge_items: List of knowledge items collected so far
            knowledge_gaps: Knowledge gaps analysis
            iteration_count: Current iteration count
            max_iterations: Maximum number of iterations allowed
            
        Returns:
            Decision dictionary with action and reasoning
        """
        logger.info(f"Making decision at iteration {iteration_count}/{max_iterations}")
        
        # Use the AI client to make a decision
        decision = self.ai_client.make_decision(
            knowledge_items,
            knowledge_gaps,
            iteration_count,
            max_iterations
        )
        
        logger.info(f"Decision: {decision.get('action', 'unknown')} with confidence {decision.get('confidence', 0)}")
        return decision