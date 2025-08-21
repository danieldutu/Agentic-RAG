#!/usr/bin/env python
"""
Simple example demonstrating the use of the ARAG system.
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to Python path to allow importing arag
sys.path.insert(0, str(Path(__file__).parent.parent))

from arag.utils.env import load_env
from arag.utils.logging import setup_logging
from arag.core.orchestrator import Orchestrator

def main():
    """Run a simple ARAG query and display the results."""
    # Load environment variables
    load_env()
    
    # Set up logging
    setup_logging(log_level="INFO")
    
    # Create the orchestrator
    orchestrator = Orchestrator()
    
    # Define a sample query
    query = "How do I adjust the park brake on a CAT 320E excavator?"
    
    print(f"\nProcessing query: {query}\n")
    print("This may take a few minutes...\n")
    
    # Process the query
    answer, memory = orchestrator.process_query(query)
    
    # Display the answer
    print("\n" + "=" * 80)
    print("ARAG RESULT")
    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    print("\nAnswer:")
    print(answer.get("text", "No answer generated"))
    
    # Display process narratives
    print("\nProcess Narratives:")
    for i, narrative in enumerate(memory.process_narratives):
        print(f"{i+1}. {narrative.get('text', '')}")
    
    # Save the full memory state to a file
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    memory_file = os.path.join(output_dir, f"memory-{memory.session_id}.json")
    memory.save_to_file(memory_file)
    
    print("\n" + "=" * 80)
    print(f"Full results saved to: {memory_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()