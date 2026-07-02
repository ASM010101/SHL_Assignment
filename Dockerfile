# Use official Python lightweight base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    HF_HOME=/app/.cache/huggingface

# Set working directory
WORKDIR /app

# Install system dependencies needed for compiling certain python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install CPU-only PyTorch first to avoid massive CUDA wheels
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ app/
COPY data/ data/
COPY scripts/ scripts/

# Pre-run catalog enrichment during build to speed up container startup
RUN python scripts/enrich_catalog.py

# Pre-download the SentenceTransformer embedding model into the cached directory
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose port
EXPOSE 8000

# Start FastAPI application using uvicorn
CMD ["sh", "-c", "uvicorn app.main:app --host $HOST --port $PORT"]
