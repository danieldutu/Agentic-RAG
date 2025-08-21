#!/usr/bin/env python
"""
Run script for the ARAG system without vector database.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import argparse

# Load environment variables first
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path)

# Add the parent directory to Python path so 'arag' can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Patch the VectorDBClient class to bypass health check
from arag.core.vector_db import VectorDBClient
original_health_check = VectorDBClient.health_check

def patched_health_check(self):
    """Patched health check that always returns True."""
    return True

VectorDBClient.health_check = patched_health_check

def main():
    """Main entry point for running without vector DB."""
    parser = argparse.ArgumentParser(description="ARAG: Advanced multi-agent RAG system (No Vector DB)")
    parser.add_argument("query", help="User query to process")
    parser.add_argument(
        "--log-level", 
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO"
    )
    
    args = parser.parse_args()
    
    # Import arag.main after patching
    from arag.main import process_query, setup_logging
    
    # Set up logging
    setup_logging(log_level=args.log_level)
    
    # Process query with mocked vector DB
    result = process_query(args.query)
    
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