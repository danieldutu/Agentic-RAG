"""
Evaluator Agent for the ARAG system.

This agent assesses answer quality against predefined criteria to ensure technical accuracy,
completeness, and usefulness.
"""

import logging
from typing import List, Dict, Any, Optional

from arag.core.ai_client import AIClient

logger = logging.getLogger(__name__)

class EvaluatorAgent:
    """
    Agent for evaluating generated answers.
    
    This agent provides feedback on areas of improvement based on
    predefined quality criteria.
    """
    
    def __init__(self, ai_client: Optional[AIClient] = None):
        """
        Initialize the Evaluator Agent.
        
        Args:
            ai_client: AI client to use, or None to create a new one
        """
        self.ai_client = ai_client or AIClient()
        logger.info("Evaluator Agent initialized")
    
    def evaluate_answer(
        self, 
        answer: Dict[str, Any],
        knowledge_items: List[Dict[str, Any]],
        user_query: str
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of the generated answer.
        
        Args:
            answer: Generated answer to evaluate
            knowledge_items: Knowledge items used to generate the answer
            user_query: Original user query
            
        Returns:
            Dictionary with evaluation scores and feedback
        """
        logger.info(f"Evaluating answer of length {len(answer.get('text', ''))}")
        
        # Use the AI client to evaluate the answer
        evaluation = self.ai_client.evaluate_answer(
            answer,
            knowledge_items,
            user_query
        )
        
        logger.info(f"Evaluation complete with overall score: {evaluation.get('scores', {}).get('overall', 0)}")
        return evaluation