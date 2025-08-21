# Multi-RAG Agent

A scalable Retrieval-Augmented Generation (RAG) system that can index and query multiple documents with memory-efficient processing.

## Features

- 📄 Index multiple PDFs with memory-efficient processing
- 🔎 Vector database integration (Weaviate) for semantic search
- 🧠 Advanced document chunking with configurable sizes and overlap
- 🤖 Google's Generative AI (Gemini) integration
- 🚀 Memory-efficient processing designed for large documents
- 🛠️ Configurable page range processing for partial document indexing
- 📊 Detailed logging and progress tracking

## Requirements

- Python 3.10+
- Docker and Docker Compose (for Weaviate vector database)
- Conda (recommended for environment management)

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n arag_env python=3.10
conda activate arag_env

# Install dependencies
pip install -r requirements.txt

# Install the local package in development mode
pip install -e .
```

### 2. Start Weaviate (Vector Database)

```bash
# Start Weaviate using Docker Compose
docker-compose up -d weaviate
```

### 3. Configure API Keys

Create or edit a `.env` file in the root directory with the following:

```
# Google AI API Key
GOOGLE_API_KEY=your_google_api_key_here

# Vector Database Configuration
VECTOR_DB_BASE_URL=http://localhost:8081
VECTOR_DB_TYPE=weaviate
WEAVIATE_CLASS_NAME=Document

# Logging Configuration
LOG_LEVEL=INFO
```

You can obtain a Google AI API key from the [Google AI Studio](https://makersuite.google.com/app/apikey).

### 4. Indexing Documents

#### Memory-Efficient Indexing (Recommended)

For large documents, index with memory-efficient settings and specific page ranges:

```bash
# For PowerShell
conda activate arag_env; python index_pdfs.py --path "data/your_document.pdf" --chunk-size 150 --overlap 25 --batch-size 2 --start-page 1 --end-page 20
```

#### Index All Documents in a Directory

```bash
# For PowerShell
conda activate arag_env; python index_pdfs.py --path "data/" --chunk-size 150 --overlap 25 --batch-size 2
```

### 5. Query the System

```bash
# For PowerShell
conda activate arag_env; python run.py "Your question about the indexed documents?"
```

## Troubleshooting

### Memory Issues During Indexing

If you encounter Out of Memory (OOM) errors:

1. Reduce `chunk-size` (e.g., 150 chars)
2. Reduce `batch-size` (e.g., 2)
3. Process smaller page ranges with `--start-page` and `--end-page`
4. Increase system swap space

### Docker Issues

If Weaviate doesn't start properly:

```bash
# Check Weaviate container logs
docker logs weaviate

# Restart Weaviate
docker-compose restart weaviate
```

## Project Structure

```
.
├── arag/               # Core RAG system modules
│   ├── agents/         # Agent implementations
│   ├── core/           # Core functionality
│   │   ├── ai_client.py      # AI models integration
│   │   ├── memory.py         # Memory management
│   │   ├── orchestrator.py   # Query processing orchestration
│   │   └── vector_db.py      # Vector database client
│   └── utils/          # Utilities
├── data/               # Store your PDF documents here
├── logs/               # Log files
├── output/             # Query output and memory files
├── docker-compose.yaml # Docker configuration
├── Dockerfile          # For containerization
├── index_pdfs.py       # PDF indexing script
├── run.py              # Main query script
└── run_indexing.bat    # Windows batch script for indexing
```

## Advanced Configuration

### Chunking Settings

- `chunk-size`: Size of text chunks (smaller = more memory-efficient)
- `overlap`: Overlap between chunks to maintain context
- `batch-size`: Number of chunks to process at once (smaller = more memory-efficient)

### Page Range Processing

Process specific pages to:
1. Test indexing before committing to full document
2. Resume interrupted indexing
3. Update specific sections of documents

```bash
# Process pages 5-15 only
python index_pdfs.py --path "data/document.pdf" --start-page 5 --end-page 15
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 