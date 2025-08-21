#!/usr/bin/env python
"""
Run script for the ARAG system.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path)

# Add the parent directory to Python path so 'arag' can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import and run the main function
from arag.main import main

if __name__ == "__main__":
    main() 