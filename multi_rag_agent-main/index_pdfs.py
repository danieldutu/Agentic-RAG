#!/usr/bin/env python
"""
PDF Indexing Script for ARAG system.

This script indexes PDF documents into Weaviate for retrieval by the ARAG system.
"""

import os
import sys
import argparse
import weaviate
import uuid
import gc
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from tqdm import tqdm
import logging
import time
import numpy as np
import random

try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence_transformers not installed. Using random vectors instead.")
    print("Install with: pip install sentence-transformers")
    
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/pdf_indexing_{time.strftime('%Y%m%d-%H%M%S')}.log")
    ]
)
logger = logging.getLogger("pdf_indexer")

# Global model for embedding to avoid recreating it
model = None
if HAVE_SENTENCE_TRANSFORMERS:
    try:
        # Use BAAI/bge-large-en-v1.5 which has 768 dimensions, matching nomic-embed-text
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        print("Loaded sentence-transformers model successfully")
    except Exception as e:
        print(f"Error loading sentence-transformers model: {e}")
        HAVE_SENTENCE_TRANSFORMERS = False

def generate_embedding(text):
    """
    Generate embedding vector for text.
    
    Args:
        text (str): The text to embed
        
    Returns:
        list: Vector embedding (or random vector if model not available)
    """
    global model
    if HAVE_SENTENCE_TRANSFORMERS and model:
        try:
            # Generate embedding using sentence-transformers
            embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            
    # Fallback to random vector (for testing only)
    print("Using random vector (for testing only)")
    return [random.uniform(-1, 1) for _ in range(768)]  # 768 dimensions for BAAI/bge-large-en-v1.5

def extract_text_from_pdf(pdf_path, start_page=None, end_page=None):
    """
    Extract text from a PDF file, one page at a time.
    
    Args:
        pdf_path (str): Path to the PDF file
        start_page (int, optional): First page to process (1-indexed)
        end_page (int, optional): Last page to process (1-indexed)
        
    Returns:
        generator: Generator that yields (page_num, text) tuples
    """
    print(f"\n[EXTRACTION] Starting text extraction from {pdf_path}")
    logger.info(f"Extracting text from {pdf_path}")
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    # Adjust page range if provided
    if start_page is None:
        start_page = 0  # 0-indexed
    else:
        start_page = max(0, start_page - 1)  # Convert from 1-indexed to 0-indexed
        
    if end_page is None:
        end_page = total_pages - 1  # 0-indexed
    else:
        end_page = min(total_pages - 1, end_page - 1)  # Convert from 1-indexed to 0-indexed
    
    print(f"[EXTRACTION] Processing pages {start_page+1} to {end_page+1} of {total_pages} total pages")
    logger.info(f"Processing pages {start_page+1} to {end_page+1} of {total_pages}")
    
    for i in range(start_page, end_page + 1):
        try:
            page = reader.pages[i]
            text = page.extract_text()
            if text.strip():  # Skip empty pages
                yield i, text
            
            # Free memory - explicit cleanup
            page = None
            
            if i % 5 == 0 and i > 0:
                print(f"[EXTRACTION] Processed page {i+1}/{end_page+1} ({(i-start_page+1)*100/(end_page-start_page+1):.1f}%)")
                logger.info(f"Processed {i+1} of {end_page+1} pages from {pdf_path} ({(i-start_page+1)*100/(end_page-start_page+1):.1f}% complete)")
                gc.collect()  # Force garbage collection
                
        except Exception as e:
            print(f"[ERROR] Error processing page {i+1}: {str(e)}")
            logger.error(f"Error processing page {i+1}: {str(e)}")
    
    print(f"[EXTRACTION] Completed: Extracted text from {end_page-start_page+1} pages")
    logger.info(f"Extracted text from {end_page-start_page+1} pages from {pdf_path}")

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Generator function to split text into chunks with overlap.
    Memory-efficient implementation that processes text incrementally.
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum size of each chunk
        overlap (int): Overlap size between chunks
        
    Yields:
        str: Text chunks
    """
    # Handle small text case - no need to chunk
    if len(text) <= chunk_size:
        yield text
        return
    
    # Track position in text
    start = 0
    text_len = len(text)
    
    # Process text in segments to avoid loading everything at once
    while start < text_len:
        # Calculate end position with bounds checking
        end = min(start + chunk_size, text_len)
        
        # Try to find a natural breaking point if not at the end
        if end < text_len:
            # Look for paragraph breaks, periods, or spaces to create natural chunks
            # Search only a small window near the end position
            search_start = max(start, end - 50)
            text_segment = text[search_start:end]
            
            # Try different break characters in order of preference
            for break_char in ['\n\n', '\n', '. ', ': ', ', ', ' ']:
                pos = text_segment.rfind(break_char)
                if pos != -1:
                    # Adjust the end position to this break point
                    end = search_start + pos + len(break_char)
                    break
        
        # Yield the current chunk
        yield text[start:end]
        
        # Update start position for next chunk with overlap
        start = max(0, end - overlap)
        
        # Optional: release memory
        if start > 0 and start % 10000 == 0:
            gc.collect()

def setup_weaviate_client():
    """
    Set up and configure the Weaviate client.
    
    Returns:
        weaviate.Client: Configured Weaviate client
    """
    url = os.getenv("VECTOR_DB_BASE_URL", "http://localhost:8081")
    print(f"[WEAVIATE] Connecting to Weaviate at {url}")
    logger.info(f"Connecting to Weaviate at {url}")
    
    # Create Weaviate client with v3 API for compatibility with older Weaviate versions
    try:
        # Use v3 client which is more compatible with older Weaviate versions
        client = weaviate.Client(url=url)
        print(f"[WEAVIATE] Connected successfully")
        
        # Check if schema exists, create if it doesn't
        class_name = os.getenv("WEAVIATE_CLASS_NAME", "Document")
        
        # Check if the schema already exists
        try:
            schema = client.schema.get()
            existing_classes = [c["class"] for c in schema["classes"]] if "classes" in schema else []
            
            if class_name not in existing_classes:
                print(f"[WEAVIATE] Creating schema for class {class_name}")
                logger.info(f"Creating schema for class {class_name}")
                
                # Create schema
                class_obj = {
                    "class": class_name,
                    "description": "Document chunks from PDFs",
                    "vectorizer": "none",
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "Text content of the document chunk"
                        },
                        {
                            "name": "source",
                            "dataType": ["string"],
                            "description": "Source PDF file name"
                        },
                        {
                            "name": "page",
                            "dataType": ["int"],
                            "description": "Page number in the source document"
                        },
                        {
                            "name": "chunk_id",
                            "dataType": ["int"],
                            "description": "Chunk ID within the page"
                        },
                        {
                            "name": "metadata",
                            "dataType": ["text"],
                            "description": "Additional metadata about the document"
                        }
                    ]
                }
                
                client.schema.create_class(class_obj)
                print(f"[WEAVIATE] Schema created successfully")
                logger.info(f"Created schema for class {class_name}")
            else:
                print(f"[WEAVIATE] Schema for class {class_name} already exists")
                logger.info(f"Schema for class {class_name} already exists")
        except Exception as e:
            print(f"[ERROR] Error checking/creating schema: {str(e)}")
            logger.error(f"Error checking/creating schema: {str(e)}")
            raise
            
        return client
    except Exception as e:
        print(f"[ERROR] Failed to connect to Weaviate: {str(e)}")
        logger.error(f"Failed to connect to Weaviate: {str(e)}")
        raise

def index_pdf(client, pdf_path, chunk_size=1000, overlap=200, batch_size=20, start_page=None, end_page=None):
    """
    Index a PDF document into Weaviate.
    
    Args:
        client (weaviate.Client): Weaviate client
        pdf_path (str): Path to the PDF file
        chunk_size (int): Maximum size of each text chunk
        overlap (int): Overlap size between chunks
        batch_size (int): Number of objects to batch insert
        start_page (int, optional): First page to process (1-indexed)
        end_page (int, optional): Last page to process (1-indexed)
        
    Returns:
        int: Number of chunks indexed
    """
    pdf_name = os.path.basename(pdf_path)
    print(f"\n[INDEXING] Starting indexing of {pdf_name}")
    print(f"[INDEXING] Configuration: chunk_size={chunk_size}, overlap={overlap}, batch_size={batch_size}")
    logger.info(f"Indexing {pdf_name}")
    
    # Get schema class name
    class_name = os.getenv("WEAVIATE_CLASS_NAME", "Document")
    
    # Track total chunk count
    total_chunk_count = 0
    
    # Pre-calculate total number of pages for progress reporting
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    if start_page is None:
        start_page = 1
    if end_page is None:
        end_page = total_pages
        
    actual_pages = min(total_pages, end_page) - start_page + 1
    print(f"[INDEXING] Will process {actual_pages} pages out of {total_pages} total pages")
    logger.info(f"Will process {actual_pages} pages out of {total_pages} total pages")
    
    # Create a progress bar for total progress
    with tqdm(total=actual_pages, desc=f"Indexing {pdf_name}", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} pages [{elapsed}<{remaining}, {rate_fmt}]") as progress_bar:
        # Process pages sequentially
        for page_num, page_text in extract_text_from_pdf(pdf_path, start_page, end_page):
            # Process the page using a memory-efficient approach
            print(f"[CHUNKING] Processing page {page_num+1}: Splitting into chunks (size={chunk_size}, overlap={overlap})")
            
            # Use a streaming approach instead of loading all chunks into memory
            # Avoid name conflict by using the global chunk_text function
            chunk_generator = globals()['chunk_text'](page_text, chunk_size, overlap)
            
            # Process chunks in small batches to reduce memory usage
            batch_count = 0
            chunk_count = 0
            current_batch = []
            
            print(f"[MEMORY] Starting batch processing for page {page_num+1}")
            
            # Process chunks one at a time from the generator
            for chunk_content in chunk_generator:
                current_batch.append(chunk_content)
                
                # When we reach the batch size, process the batch
                if len(current_batch) >= batch_size:
                    batch_count += 1
                    print(f"[INDEXING] Processing batch {batch_count} with {len(current_batch)} chunks")
                    
                    # Process this batch
                    with client.batch as batch:
                        batch.batch_size = len(current_batch)
                        
                        for i, chunk_content in enumerate(current_batch):
                            # Generate unique ID
                            obj_id = str(uuid.uuid4())
                            
                            # Generate embedding vector for the chunk
                            vector = generate_embedding(chunk_content)
                            
                            # Create object properties
                            properties = {
                                "content": chunk_content,
                                "source": pdf_name,
                                "page": page_num + 1,  # 1-indexed pages for human readability
                                "chunk_id": chunk_count + i,
                                "metadata": f"Page {page_num + 1} of {pdf_name}, chunk {chunk_count + i}"
                            }
                            
                            # Add object to batch with vector
                            batch.add_data_object(properties, class_name, obj_id, vector=vector)
                    
                    # Update chunk count
                    chunk_count += len(current_batch)
                    
                    # Clear the batch
                    current_batch = []
                    
                    # Force cleanup after each batch
                    gc.collect()
                    
                    print(f"[INDEXING] Completed batch {batch_count}: {chunk_count} chunks processed so far for page {page_num+1}")
            
            # Process any remaining chunks in the last partial batch
            if current_batch:
                batch_count += 1
                print(f"[INDEXING] Processing final batch {batch_count} with {len(current_batch)} chunks")
                
                with client.batch as batch:
                    batch.batch_size = len(current_batch)
                    
                    for i, chunk_content in enumerate(current_batch):
                        # Generate unique ID
                        obj_id = str(uuid.uuid4())
                        
                        # Generate embedding vector for the chunk
                        vector = generate_embedding(chunk_content)
                        
                        # Create object properties
                        properties = {
                            "content": chunk_content,
                            "source": pdf_name,
                            "page": page_num + 1,
                            "chunk_id": chunk_count + i,
                            "metadata": f"Page {page_num + 1} of {pdf_name}, chunk {chunk_count + i}"
                        }
                        
                        # Add object to batch with vector
                        batch.add_data_object(properties, class_name, obj_id, vector=vector)
                
                # Update chunk count
                chunk_count += len(current_batch)
                print(f"[INDEXING] Completed final batch: {chunk_count} total chunks for page {page_num+1}")
            
            # Update total count
            total_chunk_count += chunk_count
            
            # Update progress
            progress_bar.update(1)
            progress_info = {
                "page": f"{page_num+1}/{actual_pages+start_page-1}",
                "chunks": total_chunk_count,
                "progress": f"{(page_num-start_page+2)*100/actual_pages:.1f}%"
            }
            progress_bar.set_postfix(progress_info)
            
            # Print stats
            print(f"[STATS] Page {page_num+1} completed: {chunk_count} chunks indexed, {total_chunk_count} total chunks so far")
            print(f"[PROGRESS] {(page_num-start_page+2)*100/actual_pages:.1f}% of pages processed")
            
            # Force garbage collection between pages
            gc.collect()
                
    print(f"\n[COMPLETE] Indexing completed: {total_chunk_count} total chunks from {pdf_name}")
    logger.info(f"Indexed {total_chunk_count} total chunks from {pdf_name}")
    return total_chunk_count

def index_directory(client, directory, chunk_size=1000, overlap=200, batch_size=20):
    """
    Index all PDF files in a directory.
    
    Args:
        client (weaviate.Client): Weaviate client
        directory (str): Directory containing PDF files
        chunk_size (int): Maximum size of each text chunk
        overlap (int): Overlap size between chunks
        batch_size (int): Number of objects to batch insert
        
    Returns:
        int: Total number of chunks indexed
    """
    print(f"\n[DIRECTORY] Indexing PDFs in {directory}")
    logger.info(f"Indexing PDFs in {directory}")
    
    # Find all PDF files
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
                
    print(f"[DIRECTORY] Found {len(pdf_files)} PDF files to index")
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Index each PDF
    total_chunks = 0
    for pdf_path in tqdm(pdf_files, desc="Indexing PDFs"):
        chunks = index_pdf(client, pdf_path, chunk_size, overlap, batch_size)
        total_chunks += chunks
        # Force garbage collection between files
        gc.collect()
        
    print(f"\n[COMPLETE] All PDFs indexed: {total_chunks} total chunks from {len(pdf_files)} files")
    logger.info(f"Indexed {total_chunks} total chunks from {len(pdf_files)} PDF files")
    return total_chunks

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Index PDF documents into Weaviate")
    parser.add_argument("--path", help="Path to PDF file or directory", required=True)
    parser.add_argument("--chunk-size", type=int, default=750, help="Maximum chunk size in characters")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap size between chunks")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for indexing")
    parser.add_argument("--start-page", type=int, help="First page to process (1-indexed)")
    parser.add_argument("--end-page", type=int, help="Last page to process (1-indexed)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print(" "*30 + "PDF INDEXING TOOL")
    print("="*80)
    print(f"Target: {args.path}")
    print(f"Configuration: chunk_size={args.chunk_size}, overlap={args.overlap}, batch_size={args.batch_size}")
    if args.start_page or args.end_page:
        print(f"Page range: {args.start_page or 1} to {args.end_page or 'end'}")
    print("="*80 + "\n")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    try:
        start_time = time.time()
        # Set up Weaviate client
        client = setup_weaviate_client()
        
        # Index PDFs
        if os.path.isfile(args.path) and args.path.lower().endswith(".pdf"):
            # Index single PDF
            index_pdf(
                client, args.path, args.chunk_size, args.overlap, args.batch_size, 
                args.start_page, args.end_page
            )
        elif os.path.isdir(args.path):
            # Index directory of PDFs
            index_directory(
                client, args.path, args.chunk_size, args.overlap, args.batch_size
            )
        else:
            print(f"[ERROR] Invalid path: {args.path}")
            logger.error(f"Invalid path: {args.path}")
            return 1
            
        end_time = time.time()
        duration = end_time - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n" + "="*80)
        print(f"INDEXING COMPLETED SUCCESSFULLY")
        print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print("="*80 + "\n")
        
        logger.info(f"Indexing completed successfully in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Error indexing PDFs: {str(e)}")
        logger.exception(f"Error indexing PDFs: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 