CREATE EXTENSION IF NOT EXISTS vector;

-- Create knowledge_base table for semantic search
CREATE TABLE IF NOT EXISTS knowledge_base (
    id serial PRIMARY KEY,
    content text,
    metadata jsonb,
    embedding vector(3072)
);
