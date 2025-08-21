"""
Main entry point for the ARAG system.
"""

import os
import argparse
import logging
import json
import uuid
import time
from typing import Dict, Any, Optional

from arag.utils.env import load_env

from arag.core.orchestrator import Orchestrator
from arag.core.vector_db import VectorDBClient
from arag.core.ai_client import AIClient
from arag.utils.logging import setup_logging, get_session_logger
from arag.config import MAX_RETRIEVAL_ITERATIONS

def process_query(
    query: str,
    session_id: Optional[str] = None,
    max_iterations: int = MAX_RETRIEVAL_ITERATIONS,
    output_dir: str = "output"
) -> Dict[str, Any]:
    """
    Process a user query and return the results.
    
    Args:
        query: User query
        session_id: Optional session ID, or None to generate a new one
        max_iterations: Maximum number of retrieval iterations
        output_dir: Directory for output files
        
    Returns:
        Dictionary with the answer and metadata
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize session ID if not provided
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    # Get logger with session context
    logger = get_session_logger(session_id)
    logger.info(f"Processing query: {query}")
    
    try:
        # Initialize clients
        vector_db = VectorDBClient()
        ai_client = AIClient()
        
        # Check if vector DB is reachable
        if not vector_db.health_check():
            logger.error("Vector database is not reachable")
            return {
                "status": "error",
                "message": "Vector database is not reachable",
                "query": query,
                "session_id": session_id
            }
        
        # Initialize orchestrator
        orchestrator = Orchestrator(vector_db, ai_client)
        
        # Process query
        start_time = time.time()
        answer, memory = orchestrator.process_query(query, session_id, max_iterations)
        processing_time = time.time() - start_time
        
        # Save memory to file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        memory_filepath = os.path.join(output_dir, f"memory-{session_id}-{timestamp}.json")
        memory.save_to_file(memory_filepath)
        
        # Get process narratives
        process_narratives = memory.process_narratives
        
        # Build result
        result = {
            "status": "success",
            "query": query,
            "session_id": session_id,
            "answer": answer.get("text", ""),
            "citations": answer.get("citations", []),
            "process_narratives": [narrative.get("text") for narrative in process_narratives],
            "processing_time": processing_time,
            "memory_file": memory_filepath
        }
        
        logger.info(f"Query processed in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.exception(f"Error processing query: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "query": query,
            "session_id": session_id
        }

def main():
    """Main entry point for command-line usage."""
    # Load environment variables
    load_env()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="ARAG: Advanced multi-agent RAG system")
    parser.add_argument("query", help="User query to process")
    parser.add_argument(
        "--session-id", 
        help="Session ID for persistence (optional)",
        default=None
    )
    parser.add_argument(
        "--max-iterations", 
        help="Maximum number of retrieval iterations",
        type=int,
        default=MAX_RETRIEVAL_ITERATIONS
    )
    parser.add_argument(
        "--output-dir", 
        help="Directory for output files",
        default="output"
    )
    parser.add_argument(
        "--log-level", 
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(log_level=args.log_level)
    
    # Process query
    result = process_query(
        args.query,
        session_id=args.session_id,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir
    )
    
    # Print result
    if result["status"] == "success":
        print("\n" + "=" * 80)
        print("ARAG RESULT")
        print("=" * 80)
        print(f"Query: {result['query']}")
        print(f"Processing Time: {result['processing_time']:.2f}s")
        print(f"Session ID: {result['session_id']}")
        print("=" * 80)
        print("\nAnswer:")
        print(result["answer"])
        print("\nProcess Narratives:")
        for i, narrative in enumerate(result["process_narratives"]):
            print(f"{i+1}. {narrative}")
        print("\n" + "=" * 80)
        print(f"Full results saved to: {result['memory_file']}")
    else:
        print("\n" + "=" * 80)
        print("ARAG ERROR")
        print("=" * 80)
        print(f"Query: {result['query']}")
        print(f"Error: {result['message']}")
        print("=" * 80)

if __name__ == "__main__":
    main()