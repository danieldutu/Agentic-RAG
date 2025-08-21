"""
Answer Agent for the ARAG system.

This agent creates comprehensive, structured answers based on the gathered knowledge.
"""

import logging
from typing import List, Dict, Any, Optional

from arag.core.ai_client import AIClient

logger = logging.getLogger(__name__)

class AnswerAgent:
    """
    Agent responsible for generating comprehensive answers based on collected knowledge.
    """
    
    def __init__(self, ai_client):
        """
        Initialize the Answer Agent.
        
        Args:
            ai_client: AI client for answer generation
        """
        self.ai_client = ai_client
        self.logger = logging.getLogger(__name__)
        self.logger.info("Answer Agent initialized")
        
    def generate_answer(self, knowledge_items, user_query):
        """
        Generate a comprehensive answer based on collected knowledge items.
        
        Args:
            knowledge_items: List of knowledge items containing relevant information
            user_query: Original user query
            
        Returns:
            Generated answer with citations and sections
        """
        self.logger.info(f"Generating answer based on {len(knowledge_items)} knowledge items")
        
        # Filter out low-relevance knowledge items if we have many items
        if len(knowledge_items) > 10:
            # Sort by relevance (descending) and take the top items
            filtered_items = sorted(
                knowledge_items, 
                key=lambda x: x.get("relevance", 0), 
                reverse=True
            )[:15]  # Take top 15 to ensure sufficient coverage
            self.logger.info(f"Filtered to {len(filtered_items)} most relevant knowledge items")
        else:
            filtered_items = knowledge_items
            
        # Sort the filtered items by relevance (descending)
        sorted_items = sorted(
            filtered_items, 
            key=lambda x: x.get("relevance", 0), 
            reverse=True
        )
        
        # Extract information content
        knowledge_content = []
        for item in sorted_items:
            content = item.get("content", "").strip()
            if content:  # Only include items with actual content
                knowledge_content.append({
                    "content": content,
                    "source": item.get("source", "unknown"),
                    "section": item.get("section", "unknown"),
                    "page": item.get("page", 0),
                    "is_fallback": item.get("is_fallback", False)
                })
        
        # If we have fallback items, prioritize what seems like more complete content
        # Sort fallback items by content length (longer content is likely more complete)
        for i, item in enumerate(knowledge_content):
            if item.get("is_fallback", False):
                item["sort_priority"] = len(item.get("content", ""))
            else:
                item["sort_priority"] = 100000  # Non-fallback items get top priority
        
        # Re-sort with priority for non-fallback items, then by content length for fallbacks
        knowledge_content = sorted(knowledge_content, key=lambda x: x.get("sort_priority", 0), reverse=True)
        
        # Generate the answer using the AI client
        answer = self.ai_client.generate_answer(knowledge_content, user_query)
        
        self.logger.info(f"Generated answer with {len(answer.get('citations', []))} citations")
        return answer