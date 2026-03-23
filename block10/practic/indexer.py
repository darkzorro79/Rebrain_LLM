#!/usr/bin/env python3
"""Indexer script for Kubernetes CHANGELOG v1.35.

This script:
1. Downloads the CHANGELOG-1.35.md file from the official Kubernetes repo.
2. Splits the markdown into logical chunks using ``MarkdownTextSplitter``
   from ``langchain_text_splitters``.
3. Generates a 3072‑dimensional embedding for each chunk using the
   gemini-embedding-001 model via OpenAI-compatible API.
4. Persists the chunk text, optional metadata and the embedding into the
   ``knowledge_base`` table that was created in the PostgreSQL container.

Prerequisites
-------------
* Python 3.9+ (the Docker image uses Python 3.11, but any recent version works).
* The following Python packages must be installed:

  ```
  pip install psycopg2-binary google-generativeai langchain-text-splitters
  ```

* OpenAI-compatible API key for accessing gemini-embedding-001 model.
  The key should be provided via the ``OPENAI_API_KEY`` environment variable.
* PostgreSQL with pgvector extension running and accessible via DB_DSN
  environment variable.

Running the script
-------------------
```bash
export OPENAI_API_KEY=YOUR_API_KEY
export OPENAI_BASE_URL=https://llm.rebrainme.study
export DB_DSN="postgresql://user:password@localhost:5432/rag_db"
python3 indexer.py
```
"""

import os
import sys
import json
import logging
from typing import List, Dict
from urllib.parse import urlparse

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

import requests
from langchain_text_splitters import MarkdownTextSplitter
import psycopg2
from typing import Any

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("google-generativeai package is required. Install with 'pip install google-generativeai'")

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai package is required. Install with 'pip install openai'")

# Configuration for Google Generative AI (optional, for reference)
# We use gemini-embedding-001 through OpenAI-compatible API, not directly
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "not-used-but-imported")

# Configure Google Generative AI (for compatibility, even if not used for embeddings)
if GOOGLE_API_KEY and GOOGLE_API_KEY != "not-used-but-imported":
    genai.configure(api_key=GOOGLE_API_KEY)

# Configuration for OpenAI-compatible API (for 3072-dimensional embeddings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://llm.rebrainme.study")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client for embeddings
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# PostgreSQL connection settings using DB_DSN
DB_DSN = os.getenv("DB_DSN")
if not DB_DSN:
    # Fallback to individual environment variables for backward compatibility
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_db")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "secretpassword")
    DB_DSN = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# URL of the changelog file
CHANGELOG_URL = (
    "https://raw.githubusercontent.com/kubernetes/kubernetes/master/"
    "CHANGELOG/CHANGELOG-1.35.md"
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def download_changelog(url: str) -> str:
    """Download the markdown file and return its text.

    Raises:
        RuntimeError: If the request fails.
    """
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download changelog: {resp.status_code}")
    return resp.text


def split_markdown(text: str) -> List[Dict[str, str]]:
    """Split markdown into chunks.

    Returns a list of dictionaries with ``content`` and optional ``metadata``.
    The ``MarkdownTextSplitter`` respects headings, code blocks and lists.
    """
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    # The splitter returns a list of strings. We'll attach simple metadata.
    chunks = splitter.split_text(text)
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "content": chunk,
            "metadata": json.dumps({"chunk_index": i, "source": "CHANGELOG-1.35.md"}),
        })
    return result


def init_clients():
    """Initialize API clients."""
    pass  # Already configured at module level

def embed_chunk(chunk_text: str) -> List[float]:
    """Generate a 3072‑dimensional embedding for a text chunk using gemini-embedding-001.

    Uses OpenAI-compatible API with gemini-embedding-001 model to generate 
    3072-dimensional embeddings as required by the task specification.
    """
    try:
        # Use gemini-embedding-001 through OpenAI-compatible API
        # The endpoint provides this model with 3072-dimensional output
        response = openai_client.embeddings.create(
            model="gemini-embedding-001",  # Use the exact model name from task
            input=chunk_text
        )
        embedding = response.data[0].embedding
        
        if not isinstance(embedding, list):
            raise RuntimeError("Unexpected embedding format returned by API")
        
        if len(embedding) != 3072:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected 3072, got {len(embedding)}"
            )
        
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        raise RuntimeError(f"Failed to generate embedding: {e}")


def get_connection():
    """Create a psycopg2 connection using DB_DSN."""
    conn = psycopg2.connect(DB_DSN)
    return conn


def upsert_chunk(conn, content: str, metadata: str, embedding: List[float]):
    """Insert a chunk into ``knowledge_base``.

    The table uses the ``vector`` type, which ``psycopg2`` can handle by passing
    the list directly – the ``pgvector`` extension registers a binary adapter.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO knowledge_base (content, metadata, embedding)
            VALUES (%s, %s::jsonb, %s)
            """,
            (content, metadata, embedding),
        )
    conn.commit()

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Downloading changelog …")
    markdown = download_changelog(CHANGELOG_URL)

    logging.info("Splitting markdown into chunks …")
    chunks = split_markdown(markdown)
    logging.info(f"Created {len(chunks)} chunks")

    logging.info("Initialising embedding clients …")
    init_clients()

    logging.info("Connecting to PostgreSQL …")
    conn = get_connection()

    try:
        for idx, item in enumerate(chunks):
            content = item["content"]
            metadata = item["metadata"]
            logging.info(f"Embedding chunk {idx + 1}/{len(chunks)} …")
            embedding = embed_chunk(content)
            upsert_chunk(conn, content, metadata, embedding)
        logging.info("All chunks indexed successfully.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
