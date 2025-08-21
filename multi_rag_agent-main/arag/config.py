"""
Configuration settings for the ARAG system.
"""

import os
from typing import Dict, Any

# API Configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Vector Database Configuration
VECTOR_DB_BASE_URL = os.environ.get("VECTOR_DB_BASE_URL", "http://localhost:8081")
VECTOR_DB_TYPE = os.environ.get("VECTOR_DB_TYPE", "weaviate")
WEAVIATE_CLASS_NAME = os.environ.get("WEAVIATE_CLASS_NAME", "Document")
VECTOR_DB_ENDPOINTS = {
    "query": "/api/query",
    "embed": "/api/embed"
}

# Agent Configuration
MAX_TOKENS_PER_KNOWLEDGE_ITEM = 2048

# Pipeline Configuration
MAX_RETRIEVAL_ITERATIONS = 3
MIN_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to proceed to answer generation

# Model Configuration
MODELS = {
    "query_rewriter": "gemini-1.5-pro",
    "knowledge_extractor": "gemini-1.5-pro",
    "missing_info": "gemini-1.5-pro",
    "knowledge_gaps": "gemini-1.5-pro",
    "decision": "gemini-1.5-pro",
    "answer": "gemini-1.5-pro",
    "evaluator": "gemini-1.5-pro",
    "improver": "gemini-1.5-pro",
    "process": "gemini-1.5-flash"
}

# Logging Configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Memory Management
MEMORY_PERSISTENCE = True  # Enable persistent memory across retrieval iterations

def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific agent.
    
    Args:
        agent_name: Name of the agent
        
    Returns:
        Dictionary with agent-specific configuration
    """
    base_config = {
        "model": MODELS.get(agent_name, MODELS["knowledge_extractor"]),
        "max_tokens_per_knowledge_item": MAX_TOKENS_PER_KNOWLEDGE_ITEM,
    }
    
    # Agent-specific configurations can be added here
    agent_specific_config = {
        "query_rewriter": {
            "max_query_variations": 3,
        },
        "knowledge_extractor": {
            "extraction_principles": [
                "Problem identification first",
                "Query-focused extraction",
                "Prioritization of important facts",
                "Maximum comprehensiveness",
                "Complete context preservation",
                "Full table extraction when relevant",
                "Proper image reference handling",
                "Self-contained knowledge items",
                "Source attribution",
                "Token limitation"
            ]
        },
        "decision": {
            "min_confidence_threshold": MIN_CONFIDENCE_THRESHOLD,
        },
        "answer": {
            "include_citations": True,
            "prioritize_safety": True,
        }
    }
    
    # Merge base config with agent-specific config
    if agent_name in agent_specific_config:
        base_config.update(agent_specific_config[agent_name])
    
    return base_config