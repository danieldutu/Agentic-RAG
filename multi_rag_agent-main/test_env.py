import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Print all environment variables
print("Environment variables:")
print(f"GOOGLE_API_KEY: {os.getenv('GOOGLE_API_KEY', 'Not found')}")
print(f"VECTOR_DB_BASE_URL: {os.getenv('VECTOR_DB_BASE_URL', 'Not found')}")
print(f"LOG_LEVEL: {os.getenv('LOG_LEVEL', 'Not found')}") 