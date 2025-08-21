"""
Process Agent for the ARAG system.

This agent provides a first-person narrative about the system's internal processes,
making the RAG pipeline transparent to users.
"""

import logging
from typing import Dict, Any, Optional

from arag.core.ai_client import AIClient

logger = logging.getLogger(__name__)

class ProcessAgent:
    """
    Agent for providing process narratives.
    
    This agent explains what the system is doing at each step and why,
    using concise, action-centered language.
    """
    
    def __init__(self, ai_client: Optional[AIClient] = None):
        """
        Initialize the Process Agent.
        
        Args:
            ai_client: AI client to use, or None to create a new one
        """
        self.ai_client = ai_client or AIClient()
        logger.info("Process Agent initialized")
    
    def generate_narrative(
        self, 
        stage: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate a narrative about the system's internal processes.
        
        Args:
            stage: Current stage in the pipeline
            context: Context information for the narrative
            
        Returns:
            Process narrative as a string
        """
        logger.info(f"Generating process narrative for stage: {stage}")
        
        # Use the AI client to generate a process narrative
        narrative = self.ai_client.generate_process_narrative(
            stage,
            context
        )
        
        logger.info(f"Generated process narrative for stage: {stage}")
        return narrative