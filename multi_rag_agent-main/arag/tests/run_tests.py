#!/usr/bin/env python
"""
Run all tests for the ARAG system.
"""

import unittest
import sys
import os
from pathlib import Path

# Add the parent directory to Python path to allow importing arag
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_tests():
    """Run all tests and return the result."""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=os.path.dirname(__file__))
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)