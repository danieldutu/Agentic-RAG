#!/usr/bin/env python
"""
Test script for Google Generative AI.
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path)

# Get the API key
api_key = os.getenv("GOOGLE_API_KEY")
print(f"API key exists: {bool(api_key)}")

try:
    import google.generativeai as genai
    print("Successfully imported google.generativeai")
    print(f"Version: {genai.__version__}")
    
    # Configure the Google AI client
    genai.configure(api_key=api_key)
    
    # List available models
    models = genai.list_models()
    print("\nAvailable models:")
    for model in models:
        print(f"- {model.name}")
    
    print("\nTest successful!")
    
except Exception as e:
    print(f"Error: {e}") 