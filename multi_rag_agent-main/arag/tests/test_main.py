"""
Tests for the main module.
"""

import unittest
import os
import sys
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the parent directory to Python path to allow importing arag
sys.path.insert(0, str(Path(__file__).parent.parent))

import main

class TestMain(unittest.TestCase):
    """Tests for the main module."""
    
    @patch('main.Orchestrator')
    def test_process_query(self, mock_orchestrator_class):
        """Test processing a query."""
        # Set up mocks
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator
        
        mock_memory = MagicMock()
        mock_memory.process_narratives = [{"text": "Test narrative"}]
        
        mock_answer = {
            "text": "Test answer",
            "citations": [{"text": "Test citation", "source": "Test source"}]
        }
        
        mock_orchestrator.process_query.return_value = (mock_answer, mock_memory)
        
        # Call the function
        query = "How do I adjust the park brake?"
        result = main.process_query(query, session_id="test-session")
        
        # Verify the function called the orchestrator correctly
        mock_orchestrator_class.assert_called_once()
        mock_orchestrator.process_query.assert_called_once_with(
            query, "test-session", main.MAX_RETRIEVAL_ITERATIONS
        )
        
        # Verify the result has the expected structure
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["query"], query)
        self.assertEqual(result["session_id"], "test-session")
        self.assertEqual(result["answer"], "Test answer")
        self.assertIn("process_narratives", result)
        self.assertEqual(result["process_narratives"], ["Test narrative"])
    
    @patch('main.VectorDBClient')
    def test_process_query_error_vector_db(self, mock_vector_db_class):
        """Test processing a query when vector DB is not reachable."""
        # Set up mocks
        mock_vector_db = MagicMock()
        mock_vector_db_class.return_value = mock_vector_db
        mock_vector_db.health_check.return_value = False
        
        # Call the function
        query = "How do I adjust the park brake?"
        result = main.process_query(query, session_id="test-session")
        
        # Verify the result has the expected structure for an error
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Vector database is not reachable")
        self.assertEqual(result["query"], query)
        self.assertEqual(result["session_id"], "test-session")
    
    @patch('main.Orchestrator')
    def test_process_query_exception(self, mock_orchestrator_class):
        """Test processing a query when an exception occurs."""
        # Set up mocks
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator
        mock_orchestrator.process_query.side_effect = Exception("Test exception")
        
        # Call the function
        query = "How do I adjust the park brake?"
        result = main.process_query(query, session_id="test-session")
        
        # Verify the result has the expected structure for an error
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Test exception")
        self.assertEqual(result["query"], query)
        self.assertEqual(result["session_id"], "test-session")

if __name__ == "__main__":
    unittest.main()