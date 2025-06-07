# FastAPI RAG Project

A Retrieval-Augmented Generation (RAG) system built with FastAPI and LangChain, designed to provide AI-powered question answering over PDF documents.

## Features

- PDF document processing and text extraction
- Document chunking and embedding generation
- Vector database storage using Chroma
- Hugging Face model integration for text generation
- FastAPI endpoints for easy interaction
- Support for both online and offline model operation

## Project Structure

```
fastapi_rag_proj/
├── data_sources/           # Directory for PDF documents
│   └── biology/           # Biology-specific documents
├── src/
│   ├── endpoints/         # API endpoints
│   ├── models/           # LLM model configurations
│   ├── rag/              # RAG implementation
│   └── app.py            # FastAPI application
├── Dockerfile            # Docker configuration
├── .dockerignore        # Docker ignore rules
└── requirements.txt     # Project dependencies
```

## Setup

### Local Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
HUGGINGFACE_TOKEN=your_token_here
```

### Docker Setup

1. Build the Docker image:
```bash
docker build -t fastapi-rag .
```

2. Run the container:
```bash
docker run -d \
    --name fastapi-rag \
    -p 8000:8000 \
    -v $(pwd)/data_sources:/app/data_sources \
    -e HUGGINGFACE_TOKEN=your_token_here \
    fastapi-rag
```

3. To stop the container:
```bash
docker stop fastapi-rag
```

## Usage

1. Start the FastAPI server:
```bash
# Local
uvicorn src.app:app --reload

# Docker (if using Docker setup)
# The server will start automatically when the container runs
```

2. Access the API documentation at `http://localhost:8000/docs`

3. Available endpoints:
   - `/bio-ai/invoke`: Ask questions about biology documents
   - `/test/load_model`: Test model loading
   - `/test/load_hf_model`: Test Hugging Face model loading

## RAG Implementation

The project implements RAG in several steps:

1. **Document Loading**
   - PDF documents are loaded and processed
   - Text is extracted and cleaned
   - Documents are split into manageable chunks

2. **Vector Storage**
   - Text chunks are converted to embeddings
   - Stored in a vector database for efficient retrieval

3. **Question Answering**
   - Questions are processed and matched with relevant document chunks
   - LLM generates answers based on retrieved context

## Models

The system supports multiple models:
- Hugging Face models (Mistral, Mixtral, etc.)
- Google's Gemini model
- Custom model configurations

## Contributing

Feel free to submit issues and enhancement requests!

