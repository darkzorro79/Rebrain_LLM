-- Инициализация PostgreSQL с pgvector для RAG системы техподдержки
CREATE EXTENSION IF NOT EXISTS vector;

-- Создание таблицы knowledge_base (только если не существует)
CREATE TABLE IF NOT EXISTS knowledge_base (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(3072)
);

-- Создание индекса для быстрого поиска по векторам (только если не существует)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE tablename = 'knowledge_base' 
        AND indexname = 'knowledge_base_embedding_idx'
    ) THEN
        CREATE INDEX knowledge_base_embedding_idx 
        ON knowledge_base USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    END IF;
END $$;
