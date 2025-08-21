"""
Tests for the memory management module.
"""

import unittest
import os
import sys
import tempfile
import json
from pathlib import Path

# Add the parent directory to Python path to allow importing arag
sys.path.insert(0, str(Path(__file__).parent.parent))

from arag.core.memory import Memory

class TestMemory(unittest.TestCase):
    """Tests for the memory management module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.session_id = "test-session-123"
        self.user_query = "How do I adjust the park brake?"
        self.memory = Memory(self.session_id, self.user_query)
    
    def test_initialization(self):
        """Test memory initialization."""
        self.assertEqual(self.memory.session_id, self.session_id)
        self.assertEqual(self.memory.user_query, self.user_query)
        self.assertEqual(self.memory.iteration_count, 0)
        self.assertEqual(len(self.memory.optimized_queries), 0)
        self.assertEqual(len(self.memory.retrieved_documents), 0)
        self.assertEqual(len(self.memory.extracted_knowledge), 0)
    
    def test_add_optimized_queries(self):
        """Test adding optimized queries."""
        queries = ["query1", "query2", "query3"]
        self.memory.add_optimized_queries(queries)
        self.assertEqual(len(self.memory.optimized_queries), 3)
        self.assertEqual(self.memory.optimized_queries, queries)
    
    def test_add_retrieved_documents(self):
        """Test adding retrieved documents."""
        documents = [
            {"id": "doc1", "content": "content1"},
            {"id": "doc2", "content": "content2"}
        ]
        self.memory.add_retrieved_documents(documents)
        self.assertEqual(len(self.memory.retrieved_documents), 2)
        self.assertEqual(self.memory.retrieved_documents, documents)
    
    def test_add_extracted_knowledge(self):
        """Test adding extracted knowledge."""
        knowledge_items = [
            {"source": "doc1", "content": "knowledge1"},
            {"source": "doc2", "content": "knowledge2"}
        ]
        self.memory.add_extracted_knowledge(knowledge_items)
        self.assertEqual(len(self.memory.extracted_knowledge), 2)
        self.assertEqual(self.memory.extracted_knowledge, knowledge_items)
        
        # Test duplicate prevention
        self.memory.add_extracted_knowledge([{"source": "doc1", "content": "new knowledge"}])
        self.assertEqual(len(self.memory.extracted_knowledge), 3)  # Should add non-duplicate
        
        self.memory.add_extracted_knowledge([{"source": "doc1", "content": "knowledge1"}])
        self.assertEqual(len(self.memory.extracted_knowledge), 3)  # Should not add duplicate
    
    def test_increment_iteration(self):
        """Test incrementing the iteration count."""
        self.assertEqual(self.memory.iteration_count, 0)
        self.memory.increment_iteration()
        self.assertEqual(self.memory.iteration_count, 1)
        self.memory.increment_iteration()
        self.assertEqual(self.memory.iteration_count, 2)
    
    def test_get_current_state(self):
        """Test getting the current state."""
        state = self.memory.get_current_state()
        self.assertEqual(state["session_id"], self.session_id)
        self.assertEqual(state["user_query"], self.user_query)
        self.assertEqual(state["iteration_count"], 0)
    
    def test_save_and_load_memory(self):
        """Test saving and loading memory to/from a file."""
        # Add some data to memory
        self.memory.add_optimized_queries(["query1", "query2"])
        self.memory.add_retrieved_documents([{"id": "doc1", "content": "content1"}])
        self.memory.add_extracted_knowledge([{"source": "doc1", "content": "knowledge1"}])
        self.memory.increment_iteration()
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            filepath = temp_file.name
        
        self.memory.save_to_file(filepath)
        
        # Load into a new memory instance
        loaded_memory = Memory.load_from_file(filepath)
        
        # Verify contents
        self.assertEqual(loaded_memory.session_id, self.session_id)
        self.assertEqual(loaded_memory.user_query, self.user_query)
        self.assertEqual(loaded_memory.iteration_count, 1)
        self.assertEqual(len(loaded_memory.optimized_queries), 2)
        self.assertEqual(len(loaded_memory.retrieved_documents), 1)
        self.assertEqual(len(loaded_memory.extracted_knowledge), 1)
        
        # Clean up
        os.remove(filepath)

if __name__ == "__main__":
    unittest.main()