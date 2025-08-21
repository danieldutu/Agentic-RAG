"""
Improver Agent for the ARAG system.

This agent enhances answers based on evaluation feedback while maintaining
strict grounding in the provided knowledge.
"""

import logging
from typing import List, Dict, Any, Optional

from arag.core.ai_client import AIClient

logger = logging.getLogger(__name__)

class ImproverAgent:
    """
    Agent for improving generated answers.
    
    This agent refines formatting, structure, and clarity without
    adding unsupported information.
    """
    
    def __init__(self, ai_client: Optional[AIClient] = None):
        """
        Initialize the Improver Agent.
        
        Args:
            ai_client: AI client to use, or None to create a new one
        """
        self.ai_client = ai_client or AIClient()
        logger.info("Improver Agent initialized")
    
    def improve_answer(
        self, 
        original_answer: Dict[str, Any],
        evaluation: Dict[str, Any],
        knowledge_items: List[Dict[str, Any]],
        user_query: str
    ) -> Dict[str, Any]:
        """
        Improve the answer based on evaluation feedback.
        
        Args:
            original_answer: Original answer to improve
            evaluation: Evaluation with scores and feedback
            knowledge_items: Knowledge items to use for the improvement
            user_query: Original user query
            
        Returns:
            Dictionary with the improved answer and metadata
        """
        logger.info(f"Improving answer based on evaluation with {len(evaluation.get('feedback', []))} feedback items")
        
        # Use the AI client to improve the answer
        improved_answer = self.ai_client.improve_answer(
            original_answer,
            evaluation,
            knowledge_items,
            user_query
        )
        
        logger.info(f"Improved answer with {len(improved_answer.get('improvements', []))} improvements")
        return improved_answer