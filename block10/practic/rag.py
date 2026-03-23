#!/usr/bin/env python3
"""RAG query script.

Usage:
    python3 rag.py "Your question here"

The script:
1. Loads configuration (Google Generative AI for embeddings, OpenAI client for LLM).
2. Generates an embedding for the user question using gemini-embedding-001 (3072 dimensions).
3. Queries the PostgreSQL `knowledge_base` table for the 3 most similar chunks
   using the `<=>` (vector distance) operator.
4. Sends the concatenated context together with the original question to the
   LLM (model ``gemini-2.5-flash`` via OpenAI-compatible API) and prints the answer.
"""

import os
import sys
import json
import logging
from typing import List

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

import psycopg2

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

# Configuration for OpenAI-compatible LLM (for text generation)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://llm.rebrainme.study")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client for text generation
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

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_connection():
    """Create a psycopg2 connection using DB_DSN."""
    return psycopg2.connect(DB_DSN)


def embed_question(question: str) -> List[float]:
    """Generate an embedding for the user question using gemini-embedding-001.

    Uses the same embedding model as the indexer for consistency,
    ensuring 3072-dimensional vectors for proper similarity search.
    """
    try:
        # Use the same embedding model as indexer for consistency
        response = openai_client.embeddings.create(
            model="gemini-embedding-001",  # Same model as indexer
            input=question
        )
        embedding = response.data[0].embedding
        
        if not isinstance(embedding, list):
            raise RuntimeError("Unexpected embedding format from API")
        
        if len(embedding) != 3072:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected 3072, got {len(embedding)}"
            )
                
        return embedding
    except Exception as e:
        logging.error(f"Error generating question embedding: {e}")
        raise RuntimeError(f"Failed to generate question embedding: {e}")


def retrieve_top_chunks(conn, query_vec: List[float], top_k: int = 3) -> List[dict]:
    """Return the `top_k` most similar chunks from `knowledge_base`.

    The SQL uses the `<=>` operator provided by the pgvector extension to compute
    Euclidean distance (you can switch to `<->` for inner product if desired).
    """
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT content, metadata, embedding
            FROM knowledge_base
            ORDER BY embedding <=> %s
            LIMIT %s;
            """,
            (query_vec, top_k),
        )
        rows = cur.fetchall()
    # Convert to list of dicts for easier handling
    return [
        {"content": row[0], "metadata": row[1], "embedding": row[2]}
        for row in rows
    ]


def generate_answer(context: str, question: str) -> str:
    """Call the LLM (gemini-2.5-flash) via OpenAI-compatible API.

    The prompt follows the required format:
        "Используя предоставленные данные из CHANGELOG: {context}, ответь на вопрос: {question}"
    """
    prompt = (
        f"Используя предоставленные данные из CHANGELOG: {context}, "
        f"ответь на вопрос: {question}"
    )
    
    try:
        # Use OpenAI-compatible API to call gemini-2.5-flash
        response = openai_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        raise RuntimeError(f"Failed to generate answer: {e}")

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: rag.py \"your question\"\n")
        sys.exit(1)

    question = sys.argv[1]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Embedding user question …")
    q_vec = embed_question(question)

    logging.info("Connecting to PostgreSQL …")
    conn = get_connection()
    try:
        logging.info("Retrieving top 3 relevant chunks …")
        chunks = retrieve_top_chunks(conn, q_vec, top_k=3)
        if not chunks:
            logging.warning("No chunks found in the database.")
            sys.stdout.write("No relevant information found.\n")
            return
        # Concatenate the content of the retrieved chunks as context
        context = "\n\n---\n\n".join(chunk["content"] for chunk in chunks)
        logging.info("Generating answer via LLM …")
        answer = generate_answer(context, question)
        sys.stdout.write(answer + "\n")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
