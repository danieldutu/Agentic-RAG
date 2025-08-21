"""
Environment variable utilities for the ARAG system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv as _load_dotenv

def load_env(env_file: str = None) -> None:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Path to .env file, or None to use default
    """
    if env_file is None:
        # Try to find .env file in the project root
        root_dir = Path(__file__).parent.parent.parent
        env_file = root_dir / ".env"
    
    # Load environment variables
    _load_dotenv(env_file)
    
    # Verify required environment variables
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}. "
            f"Please check your .env file at {env_file}"
        ) 